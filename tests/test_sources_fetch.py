"""Web page download against a local http.server (no external network)."""

from __future__ import annotations

import gzip
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from h0lon.config import Settings
from h0lon.sources import fetch
from h0lon.sources.fetch import FetchError, fetch_html, iri_to_uri, to_utf8_html
from h0lon.sources.ingest import add_sources, list_sources
from h0lon.workspace import create_topic

PAGE_UTF8 = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>Случайная величина — Википедия</title></head>"
    "<body><p>Случайная величина — функция на пространстве исходов.</p></body></html>"
)
PAGE_1251 = (
    "<html><head><meta http-equiv='Content-Type' content='text/html; charset=windows-1251'>"
    "<title>Мера Лебега</title></head><body><p>Мера — счётно-аддитивная функция.</p></body>"
    "</html>"
)


@dataclass
class Route:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    chunks: int = 0  # >0: send the body in that many pieces without Content-Length


ROUTES: dict[str, Route] = {
    "/page": Route(body=PAGE_UTF8.encode(), headers={"Content-Type": "text/html; charset=utf-8"}),
    "/cp1251": Route(
        body=PAGE_1251.encode("cp1251"),
        headers={"Content-Type": "text/html; charset=windows-1251"},
    ),
    "/meta-only": Route(body=PAGE_1251.encode("cp1251"), headers={"Content-Type": "text/html"}),
    "/lying-header": Route(
        body=PAGE_1251.encode("cp1251"), headers={"Content-Type": "text/html; charset=utf-8"}
    ),
    "/no-type": Route(body=PAGE_UTF8.encode()),
    "/no-type-binary": Route(body=b"\x00\x01binary"),
    "/xhtml": Route(
        body=b"<?xml version='1.0' encoding='windows-1251'?><html><head><title>X</title>"
        b"</head></html>",
        headers={"Content-Type": "application/xhtml+xml"},
    ),
    "/gzip": Route(
        body=gzip.compress(PAGE_UTF8.encode()),
        headers={"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"},
    ),
    "/gzip-cut": Route(
        body=gzip.compress(PAGE_UTF8.encode() * 20)[:150],
        headers={"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"},
    ),
    "/stray-byte": Route(
        body=PAGE_UTF8.encode().replace(b"<p>", b"<p>\xff"),
        headers={"Content-Type": "text/html; charset=utf-8"},
    ),
    "/notitle": Route(body=b"<html><body>x</body></html>", headers={"Content-Type": "text/html"}),
    "/pdf": Route(body=b"%PDF-1.7 ...", headers={"Content-Type": "application/pdf"}),
    "/empty": Route(body=b"", headers={"Content-Type": "text/html"}),
    "/missing": Route(status=404, body=b"no", headers={"Content-Type": "text/html"}),
    "/broken": Route(status=500, body=b"err", headers={"Content-Type": "text/html"}),
    "/redirect": Route(status=302, headers={"Location": "/page"}),
    "/slow": Route(body=PAGE_UTF8.encode(), headers={"Content-Type": "text/html"}, delay=2.0),
    "/big": Route(body=b"<html>" + b"x" * 5000, headers={"Content-Type": "text/html"}),
    "/big-chunked": Route(
        body=b"<html>" + b"x" * 5000, headers={"Content-Type": "text/html"}, chunks=5
    ),
    "/wiki/%D0%A1%D0%BB%D1%83%D1%87%D0%B0%D0%B9": Route(
        body=PAGE_UTF8.encode(), headers={"Content-Type": "text/html; charset=utf-8"}
    ),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen_headers: ClassVar[dict[str, str]] = {}

    def _redirect(self, location: str, cookie: str | None = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        Handler.seen_headers = dict(self.headers)
        if self.path == "/loop":
            self._redirect("/loop")
            return
        if self.path == "/cookie" and "sid=1" not in (self.headers.get("Cookie") or ""):
            self._redirect("/cookie", cookie="sid=1; Path=/")  # sets a cookie, comes back
            return
        route = ROUTES.get("/page" if self.path == "/cookie" else self.path)
        if route is None:
            self.send_error(404)
            return
        if route.delay:
            time.sleep(route.delay)
        self.send_response(route.status)
        for k, v in route.headers.items():
            self.send_header(k, v)
        if route.chunks:
            self.send_header("Connection", "close")
            self.end_headers()
            step = max(1, len(route.body) // route.chunks)
            for i in range(0, len(route.body), step):
                self.wfile.write(route.body[i : i + step])
                self.wfile.flush()
            return
        self.send_header("Content-Length", str(len(route.body)))
        self.end_headers()
        self.wfile.write(route.body)

    def log_message(self, format: str, *args: object) -> None:  # silence
        pass


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests to 127.0.0.1 must not go through a system proxy."""
    monkeypatch.setattr(urllib.request, "getproxies", dict)


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def topic(make_settings: Callable[..., Settings]) -> tuple[Settings, Path]:
    settings = make_settings(general={"git_per_topic": False})
    return settings, create_topic(settings, title="Мера", course="Матан")


# ---------------------------------------------------------------- fetch_html


def test_fetch_utf8_page(server: str) -> None:
    page = fetch_html(server + "/page")
    assert page.title == "Случайная величина — Википедия"
    assert page.encoding == "utf-8"
    assert page.content_type == "text/html"
    assert "функция на пространстве исходов" in page.text
    assert Handler.seen_headers["User-Agent"].startswith("H0lon/")


@pytest.mark.parametrize("path", ["/cp1251", "/meta-only", "/lying-header"])
def test_fetch_cp1251_page(server: str, path: str) -> None:
    page = fetch_html(server + path)
    assert page.title == "Мера Лебега"
    assert page.encoding == "cp1251"
    assert "счётно-аддитивная" in page.text
    stored = page.utf8_bytes().decode("utf-8")  # strict: valid UTF-8
    assert "charset=utf-8" in stored and "windows-1251" not in stored


def test_fetch_without_content_type_sniffs_html(server: str) -> None:
    assert fetch_html(server + "/no-type").title == "Случайная величина — Википедия"
    with pytest.raises(FetchError, match="не похоже на HTML"):
        fetch_html(server + "/no-type-binary")


def test_fetch_xhtml_and_gzip(server: str) -> None:
    page = fetch_html(server + "/xhtml")
    assert page.content_type == "application/xhtml+xml" and page.title == "X"
    assert page.utf8_bytes().startswith(b"<?xml version='1.0' encoding='utf-8'?>")
    assert fetch_html(server + "/gzip").title == "Случайная величина — Википедия"


def test_fetch_follows_redirects(server: str) -> None:
    page = fetch_html(server + "/redirect")
    assert page.final_url == server + "/page"
    assert page.url == server + "/redirect"


def test_fetch_cyrillic_path_is_encoded(server: str) -> None:
    assert fetch_html(server + "/wiki/Случай").title == "Случайная величина — Википедия"


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("/pdf", "не веб-страница, а «application/pdf» — скачайте файл"),
        ("/missing", "страница не найдена \\(HTTP 404\\)"),
        ("/broken", "ошибка на стороне сайта \\(HTTP 500"),
        ("/empty", "пустую страницу"),
    ],
)
def test_fetch_errors(server: str, path: str, match: str) -> None:
    with pytest.raises(FetchError, match=match):
        fetch_html(server + path)


def test_fetch_size_limit(server: str) -> None:
    with pytest.raises(FetchError, match="больше"):
        fetch_html(server + "/big", max_bytes=1000)
    with pytest.raises(FetchError, match="больше"):
        fetch_html(server + "/big-chunked", max_bytes=1000)
    assert fetch_html(server + "/big", max_bytes=10_000).title is None


def test_fetch_timeout(server: str) -> None:
    started = time.monotonic()
    with pytest.raises(FetchError, match=r"не ответил за 0\.3 с"):
        fetch_html(server + "/slow", timeout=0.3)
    assert time.monotonic() - started < 1.9


def test_fetch_connection_refused() -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(FetchError):
        fetch_html(f"http://127.0.0.1:{port}/", timeout=2)


def test_iri_to_uri_and_charset_rewrite() -> None:
    assert (
        iri_to_uri("https://ru.wikipedia.org/wiki/Случайная_величина#x")
        == "https://ru.wikipedia.org/wiki/%D0%A1%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D0%B0%D1%8F_"
        "%D0%B2%D0%B5%D0%BB%D0%B8%D1%87%D0%B8%D0%BD%D0%B0"
    )
    assert iri_to_uri("http://пример.рф/a b?q=мера") == (
        "http://xn--e1afmkfd.xn--p1ai/a%20b?q=%D0%BC%D0%B5%D1%80%D0%B0"
    )
    assert iri_to_uri("https://x.org/%D0%90") == "https://x.org/%D0%90"  # no double escaping
    html = '<html><head><meta charset="windows-1251"><title>Т</title>'
    assert (
        to_utf8_html(html).decode("utf-8") == '<html><head><meta charset="utf-8"><title>Т</title>'
    )


# ---------------------------------------------------------------- add_sources with links


def test_add_web_page(topic, server: str) -> None:
    settings, topic_dir = topic
    url = server + "/cp1251"
    report = add_sources(settings, topic_dir, [url])
    assert report.warnings == []
    (record,) = report.added
    assert (record.id, record.kind, record.url) == ("W1", "web", url)
    assert record.title == "Мера Лебега"
    assert record.file == "sources/W1_mera-lebega.html"
    assert record.quality == {"encoding": "cp1251"}
    stored = (topic_dir / record.file).read_bytes()
    assert "счётно-аддитивная" in stored.decode("utf-8")
    import hashlib

    assert record.sha256 == hashlib.sha256(stored).hexdigest() and record.size == len(stored)
    assert record.original_name is None

    again = add_sources(settings, topic_dir, [url + "/"])  # same link, trailing slash
    assert again.added == [] and "уже добавлен как W1" in again.warnings[0]


def test_add_web_page_title_fallback_and_redirect(topic, server: str) -> None:
    settings, topic_dir = topic
    (record,) = add_sources(settings, topic_dir, [server + "/notitle"]).added
    assert record.title == "127.0.0.1/notitle"
    assert record.file == "sources/W1_127-0-0-1-notitle.html"
    (record,) = add_sources(settings, topic_dir, [server + "/redirect"]).added
    assert record.model_extra["final_url"] == server + "/page"
    assert record.title == "Случайная величина — Википедия"


def test_failed_download_is_a_warning(topic, server: str, monkeypatch) -> None:
    settings, topic_dir = topic
    monkeypatch.setattr(fetch, "MAX_BYTES", 1000)
    report = add_sources(
        settings,
        topic_dir,
        [server + "/missing", server + "/pdf", server + "/big", server + "/page"],
    )
    assert [r.id for r in report.added] == ["W1"]
    assert len(report.warnings) == 3
    assert "HTTP 404" in report.warnings[0] and "Источник не добавлен" in report.warnings[0]
    assert "application/pdf" in report.warnings[1]
    assert "больше" in report.warnings[2]
    assert [r.id for r in list_sources(topic_dir)] == ["W1"]
    assert sorted(p.name for p in (topic_dir / "sources").iterdir()) == [
        "W1_sluchaynaya-velichina-vikipediya.html"
    ]


def test_same_content_from_another_url_is_a_duplicate(topic, server: str) -> None:
    settings, topic_dir = topic
    add_sources(settings, topic_dir, [server + "/page"])
    report = add_sources(settings, topic_dir, [server + "/gzip"])  # same HTML after decoding
    assert report.added == [] and "уже добавлен как W1" in report.warnings[0]


# ---------------------------------------------------------------- review fixes


def test_fetch_cut_gzip_stream_is_incomplete(server: str) -> None:
    with pytest.raises(FetchError, match="страница пришла не полностью"):
        fetch_html(server + "/gzip-cut")


def test_fetch_keeps_cookies_between_redirects(server: str) -> None:
    page = fetch_html(server + "/cookie")
    assert page.title == "Случайная величина — Википедия"


def test_fetch_redirect_loop_is_explained_in_russian(server: str) -> None:
    with pytest.raises(FetchError) as info:
        fetch_html(server + "/loop")
    message = str(info.value)
    assert "зациклил перенаправление" in message and "HTTP 302" in message
    assert "infinite loop" not in message and "\n" not in message


def test_stray_byte_keeps_the_declared_charset(topic, server: str) -> None:
    page = fetch_html(server + "/stray-byte")
    assert (page.encoding, page.replaced) == ("utf-8", 1)
    assert "функция на пространстве исходов" in page.text
    settings, topic_dir = topic
    (record,) = add_sources(settings, topic_dir, [server + "/stray-byte"]).added
    assert record.quality["encoding"] == "utf-8" and record.quality["replaced_bytes"] == 1
    assert "заменой повреждённых байтов" in record.quality["notes"][0]
    stored = (topic_dir / record.file).read_text(encoding="utf-8")
    assert "Случайная величина — функция" in stored


class PortHandler(BaseHTTPRequestHandler):
    """A different page on every port (the same bytes would be a duplicate by sha256)."""

    def do_GET(self) -> None:
        port = self.server.server_address[1]
        body = f"<html><head><title>port {port}</title></head></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_same_path_on_another_port_is_another_source(topic) -> None:
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), PortHandler) for _ in range(2)]
    for httpd in servers:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    urls = [f"http://127.0.0.1:{httpd.server_address[1]}/page" for httpd in servers]
    settings, topic_dir = topic
    try:
        report = add_sources(settings, topic_dir, urls)
    finally:
        for httpd in servers:
            httpd.shutdown()
            httpd.server_close()
    assert report.warnings == []
    assert [r.url for r in report.added] == urls


def test_iri_to_uri_keeps_ipv6_brackets() -> None:
    assert iri_to_uri("http://[::1]:8000/x?a=b") == "http://[::1]:8000/x?a=b"
    with pytest.raises(FetchError, match="ссылка записана с ошибкой"):
        fetch_html("http://example.com:abc/")
