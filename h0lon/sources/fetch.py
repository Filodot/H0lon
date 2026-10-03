"""Download of web pages given by link (`h0lon add <тема> https://…`).

Only HTML is accepted (`text/html`, `application/xhtml+xml`); the page is decoded with the
charset from the HTTP header, a BOM or `<meta charset>` (UTF-8, then cp1251 as fallbacks;
see `detect.decode_html_detailed`) and stored as UTF-8 with the `<meta charset>` declarations
rewritten, so every later reader (trafilatura, a browser, lxml) sees the same text. Cookies
are kept for the duration of one download (sites that set a cookie and redirect to
themselves); a compressed body must end with its end-of-stream marker, otherwise the page
is reported as incomplete.
"""

from __future__ import annotations

import contextlib
import http.cookiejar
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

from h0lon import __version__
from h0lon.sources.detect import decode_html_detailed, html_title

USER_AGENT = f"H0lon/{__version__} (+https://github.com/Filodot/H0lon; study notes builder)"
TIMEOUT_S = 30.0  # per socket operation
TOTAL_TIMEOUT_S = 120.0  # whole download
MAX_BYTES = 20 * 1024 * 1024
HTML_TYPES = ("text/html", "application/xhtml+xml")
_CHUNK = 64 * 1024


class FetchError(Exception):
    """The page could not be downloaded; the message is a readable Russian sentence."""


@dataclass
class FetchedPage:
    url: str  # as given
    final_url: str  # after redirects
    content_type: str
    encoding: str  # charset the page was decoded with
    text: str  # decoded HTML
    title: str | None  # <title> / og:title
    replaced: int = 0  # undecodable bytes replaced with U+FFFD while decoding
    guessed: bool = False  # the site declared no usable charset: the encoding was guessed

    def utf8_bytes(self) -> bytes:
        """The page as UTF-8 with its charset declarations pointing to UTF-8."""
        return to_utf8_html(self.text)


def iri_to_uri(url: str) -> str:
    """Percent-encode non-ASCII parts of a URL (Cyrillic paths, IDN hosts) for urllib."""
    parts = urlsplit(url.strip())
    host = parts.hostname or ""
    try:
        host_ascii = host.encode("idna").decode("ascii") if host else ""
    except UnicodeError:
        host_ascii = host
    netloc = f"[{host_ascii}]" if ":" in host_ascii else host_ascii  # IPv6 literal
    if parts.port:
        netloc += f":{parts.port}"
    if parts.username:
        auth = quote(parts.username, safe="")
        if parts.password:
            auth += ":" + quote(parts.password, safe="")
        netloc = f"{auth}@{netloc}"
    path = quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="=&%:@!$'()*+,;/?-._~")
    return urlunsplit((parts.scheme.lower(), netloc, path or "/", query, ""))


_META_CHARSET_TAG_RE = re.compile(r"<meta\s+charset\s*=\s*[\"']?[^\"'\s/>]*[\"']?", re.IGNORECASE)
_META_HTTP_EQUIV_RE = re.compile(
    r"(<meta[^>]+content\s*=\s*[\"'][^\"']*charset\s*=\s*)([^\"'\s;]+)", re.IGNORECASE
)
_XML_DECL_RE = re.compile(r"^(<\?xml[^>]*encoding\s*=\s*[\"'])([^\"']+)", re.IGNORECASE)


def to_utf8_html(text: str) -> bytes:
    """Encode HTML as UTF-8 and make its charset declarations say so."""
    head, rest = text[:8192], text[8192:]
    head = _XML_DECL_RE.sub(r"\1utf-8", head, count=1)
    head = _META_CHARSET_TAG_RE.sub('<meta charset="utf-8"', head)
    head = _META_HTTP_EQUIV_RE.sub(r"\1utf-8", head)
    return (head + rest).encode("utf-8")


def _describe_url_error(exc: BaseException, timeout: float) -> str:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, TimeoutError):
        return f"сервер не ответил за {timeout:g} с"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return f"не удалось проверить сертификат сайта ({reason.verify_message})"
    if isinstance(reason, ssl.SSLError):
        return f"ошибка защищённого соединения ({reason})"
    if isinstance(reason, socket.gaierror):
        return "адрес сайта не найден (нет сети или опечатка в ссылке)"
    if isinstance(reason, ConnectionRefusedError):
        return "сервер отклонил соединение"
    if isinstance(reason, ConnectionError):
        return f"соединение прервано ({reason})"
    return str(reason) or exc.__class__.__name__


def _http_error_text(code: int, reason: str) -> str:
    known = {
        401: "нужна авторизация",
        403: "доступ запрещён",
        404: "страница не найдена",
        410: "страница удалена",
        429: "сайт ограничил частоту запросов, попробуйте позже",
    }
    if code in known:
        return f"{known[code]} (HTTP {code})"
    if 300 <= code < 400:
        # urllib gives up on a redirect loop (or a redirect without a target) with a long
        # English message; a cookie or a login wall is the usual reason.
        return (
            f"не удалось пройти перенаправления сайта (HTTP {code}): сайт зациклил "
            "перенаправление или требует входа — откройте страницу в браузере, сохраните её "
            "(Ctrl+S) и добавьте как файл"
        )
    if code >= 500:
        return f"ошибка на стороне сайта (HTTP {code} {reason})".rstrip()
    return f"сервер ответил HTTP {code} {reason}".rstrip()


class _Decompressor:
    def __init__(self, encoding: str) -> None:
        enc = encoding.lower().strip()
        if enc in ("", "identity"):
            self._obj = None
        elif enc in ("gzip", "x-gzip"):
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif enc == "deflate":
            self._obj = zlib.decompressobj()
        else:
            raise FetchError(f"сервер прислал страницу в неподдерживаемом сжатии «{encoding}»")
        self._fed = False

    def feed(self, data: bytes) -> bytes:
        if self._obj is None:
            return data
        self._fed = self._fed or bool(data)
        try:
            return self._obj.decompress(data)
        except zlib.error as exc:
            raise FetchError(f"страница пришла повреждённой ({exc})") from exc

    def flush(self) -> bytes:
        if self._obj is None:
            return b""
        try:
            tail = self._obj.flush()
        except zlib.error as exc:
            raise FetchError(f"страница пришла повреждённой ({exc})") from exc
        # A cut compressed stream decodes without errors up to the cut: only the missing
        # end-of-stream marker tells that the page is incomplete.
        if self._fed and not self._obj.eof:
            raise FetchError("страница пришла не полностью (сжатые данные оборваны)")
        return tail


def _looks_like_html(data: bytes) -> bool:
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith((b"<!doctype html", b"<html")) or b"<html" in head[:512]


def fetch_html(
    url: str,
    *,
    timeout: float | None = None,
    total_timeout: float | None = None,
    max_bytes: int | None = None,
) -> FetchedPage:
    """Download one HTML page. Raises FetchError with a readable reason.

    Limits default to the module constants (read at call time).
    """
    timeout = TIMEOUT_S if timeout is None else timeout
    total_timeout = TOTAL_TIMEOUT_S if total_timeout is None else total_timeout
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    try:
        target = iri_to_uri(url)
    except ValueError as exc:  # a bad port or IPv6 literal; ingest checks links earlier
        raise FetchError(f"ссылка записана с ошибкой ({exc})") from exc
    request = urllib.request.Request(
        target,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Language": "ru,en;q=0.8",
            "Accept-Encoding": "gzip",
        },
    )
    started = time.monotonic()
    limit_mb = max_bytes / (1024 * 1024)
    try:
        # A fresh opener per call: proxy settings are read now, not cached from the first call.
        # Cookies live for this download only: some sites set one and redirect to themselves.
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        with opener.open(request, timeout=timeout) as resp:
            final_url = resp.geturl()
            if urlsplit(final_url).scheme.lower() not in ("http", "https"):
                raise FetchError("ссылка перенаправила на неподдерживаемый адрес")
            raw_type = resp.headers.get("Content-Type")
            content_type = resp.headers.get_content_type() if raw_type else ""
            charset = resp.headers.get_content_charset() if raw_type else None
            if content_type and content_type not in HTML_TYPES:
                raise FetchError(
                    f"по ссылке не веб-страница, а «{content_type}» — "
                    "скачайте файл и добавьте его как файл"
                )
            length = resp.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise FetchError(f"страница больше {limit_mb:g} МБ")
            decoder = _Decompressor(resp.headers.get("Content-Encoding", ""))
            chunks: list[bytes] = []
            size = 0
            while True:
                if time.monotonic() - started > total_timeout:
                    raise FetchError(f"скачивание заняло больше {total_timeout:g} с")
                piece = resp.read(_CHUNK)
                if not piece:
                    break
                data = decoder.feed(piece)
                size += len(data)
                if size > max_bytes:
                    raise FetchError(f"страница больше {limit_mb:g} МБ")
                chunks.append(data)
            tail = decoder.flush()
            size += len(tail)
            if size > max_bytes:
                raise FetchError(f"страница больше {limit_mb:g} МБ")
            chunks.append(tail)
    except FetchError:
        raise
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            exc.close()
        raise FetchError(_http_error_text(exc.code, exc.reason or "")) from exc
    except urllib.error.URLError as exc:
        raise FetchError(_describe_url_error(exc, timeout)) from exc
    except TimeoutError as exc:
        raise FetchError(f"сервер не ответил за {timeout:g} с") from exc
    except (ConnectionError, OSError, ValueError) as exc:
        raise FetchError(_describe_url_error(exc, timeout)) from exc
    except Exception as exc:  # http.client.IncompleteRead, BadStatusLine …
        raise FetchError(f"не удалось скачать страницу ({exc.__class__.__name__}: {exc})") from exc

    body = b"".join(chunks)
    if not body.strip():
        raise FetchError("сервер прислал пустую страницу")
    if not content_type:
        if not _looks_like_html(body):
            raise FetchError("сервер не указал тип содержимого, и это не похоже на HTML")
        content_type = "text/html"
    decoded = decode_html_detailed(body, charset)
    return FetchedPage(
        url=url,
        final_url=final_url,
        content_type=content_type,
        encoding=decoded.encoding,
        text=decoded.text,
        title=html_title(decoded.text),
        replaced=decoded.replaced,
        guessed=decoded.guessed,
    )
