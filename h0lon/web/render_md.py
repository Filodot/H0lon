"""Source Doc (`extracted/<ID>/source.md`) → HTML fragment for the review screen.

Pandoc does the conversion (`--mathml`: formulas become MathML, no JavaScript or fonts from the
network). The Source Doc is data taken from user materials, so the page treats it as untrusted:

- the HTML sanitiser of the render pipeline (`filters/sanitize.lua`) drops raw HTML and image
  sources outside the document;
- here, links with active schemes (`javascript:`, `data:` …) are neutralised, external links
  open in a new tab without a referrer;
- the page itself is served with a strict Content-Security-Policy (see `app.py`).

On top of the Pandoc output the fragment gets: block markers (`<!-- P1.b007 definition -->` →
`<div class="blk">`), clickable location headings (`## [[S1:s12]] Слайд 12` carry `data-loc` and
`data-page`, which the page uses to link a heading to the page image), image URLs resolved
against `files_base`, tables wrapped for horizontal scrolling. Results are cached by file
mtime and size.
"""

from __future__ import annotations

import html
import re
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, unquote

from h0lon import tools
from h0lon.render.pandoc import build_argv, escape_anchor_syntax, run_pandoc
from h0lon.render.templates import FILTERS_DIR

CACHE_SIZE = 32
PANDOC_TIMEOUT_S = 120

_FRONT_MATTER_RE = re.compile(
    r"\A﻿?---[ \t]*\r?\n.*?\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_BLOCK_COMMENT_RE = re.compile(r"<!--[ \t]*([A-Z][A-Za-z0-9]*\.b\d+)[ \t]+([\w-]+)[ \t]*-->")
_ANY_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"<h([1-6])((?:\s[^>]*)?)>\[\[([A-Z][A-Za-z0-9]*:[^\[\]\s]+)\]\][ \t]*")
_PAGE_ANCHOR_RE = re.compile(r"^[A-Z][A-Za-z0-9]*:[ps](\d+)$")
_IMG_SRC_RE = re.compile(r'(<img\b[^>]*?\ssrc=")([^"]*)(")')
_LINK_RE = re.compile(r"<a\b([^>]*)>")
_HREF_RE = re.compile(r'\shref="([^"]*)"')
_UNSAFE_SCHEMES = ("javascript", "data", "vbscript", "file")
_EXTERNAL_RE = re.compile(r"^(?:https?:)?//", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x20\x7f]+")

_CACHE: OrderedDict[tuple[str, str], tuple[tuple[int, int, str], str]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


class SourceDocError(Exception):
    """The Source Doc could not be converted (no Pandoc, Pandoc failed, unreadable file)."""


def source_doc_html(path: Path, *, files_base: str = "", pandoc: Path | None = None) -> str:
    """HTML fragment of the Source Doc at `path`.

    `files_base` is the URL prefix (with a trailing slash) of the directory that holds the
    document: relative image paths are resolved against it. `pandoc` is the executable
    (default: `tools.find_pandoc()`). Raises `SourceDocError` with a Russian message,
    `FileNotFoundError` if the file does not exist.
    """
    path = Path(path)
    stat = path.stat()  # FileNotFoundError is the caller's «not extracted yet»
    exe = pandoc or tools.find_pandoc("")
    if exe is None:
        raise SourceDocError(
            "Не найден Pandoc: он нужен для показа Source Doc. "
            "Выполните h0lon doctor — там написано, как его установить."
        )
    key = (str(path.resolve()), files_base)
    stamp = (stat.st_mtime_ns, stat.st_size, str(exe))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            _CACHE.move_to_end(key)
            return hit[1]
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise SourceDocError(f"Не удалось прочитать {path.name}: {exc}") from exc
    fragment = _convert(_FRONT_MATTER_RE.sub("", text, count=1), exe, files_base)
    with _CACHE_LOCK:
        _CACHE[key] = (stamp, fragment)
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return fragment


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------- conversion


def _convert(markdown: str, pandoc: Path, files_base: str) -> str:
    argv = build_argv(
        pandoc,
        src=None,
        to="html5",
        output=None,
        standalone=False,
        filters=[FILTERS_DIR / "sanitize.lua"],
        extra_args=["--mathml", "--wrap=none", "--no-highlight"],
    )
    result = run_pandoc(argv, input_text=escape_anchor_syntax(markdown), timeout=PANDOC_TIMEOUT_S)
    if not result.ok:
        raise SourceDocError(result.error or "Pandoc не смог преобразовать Source Doc.")
    return postprocess(result.stdout, files_base)


def postprocess(fragment: str, files_base: str = "") -> str:
    """Block markers, location headings, image URLs, links and tables of Pandoc's HTML."""
    out = _BLOCK_COMMENT_RE.sub(_block_marker, fragment)
    out = _ANY_COMMENT_RE.sub("", out)
    out = _HEADING_RE.sub(_location_heading, out)
    out = _IMG_SRC_RE.sub(
        lambda m: m.group(1) + _image_url(m.group(2), files_base) + m.group(3), out
    )
    out = _LINK_RE.sub(_safe_link, out)
    out = out.replace("<table", '<div class="table-wrap"><table').replace(
        "</table>", "</table></div>"
    )
    return out


def _block_marker(match: re.Match[str]) -> str:
    block_id, kind = match.group(1), match.group(2)
    return f'<div class="blk" id="{block_id}" data-type="{kind}">{block_id} · {kind}</div>'


def _location_heading(match: re.Match[str]) -> str:
    level, attrs, anchor = match.group(1), match.group(2), match.group(3)
    page = _PAGE_ANCHOR_RE.match(anchor)
    data = f' data-loc="{html.escape(anchor, quote=True)}"'
    if page:
        data += f' data-page="{int(page.group(1))}"'
    cls = ' class="loc"'
    if " class=" in attrs:  # Pandoc heading with its own class: keep both
        attrs = attrs.replace(' class="', ' class="loc ', 1)
        cls = ""
    return (
        f"<h{level}{attrs}{cls}{data}>"
        f'<span class="loc-tag">{html.escape(anchor, quote=False)}</span> '
    )


def _image_url(src: str, files_base: str) -> str:
    """Relative image paths are served from the document's directory under /files/."""
    raw = html.unescape(src).strip()
    if not raw or _EXTERNAL_RE.match(raw) or raw.startswith(("/", "data:")) or ":" in raw[:12]:
        return src
    return html.escape(files_base + quote(unquote(raw), safe="/"), quote=True)


def _safe_link(match: re.Match[str]) -> str:
    attrs = match.group(1)
    href_match = _HREF_RE.search(attrs)
    if href_match is None:
        return match.group(0)
    href = html.unescape(href_match.group(1))
    probe = _CONTROL_RE.sub("", href).lower()
    scheme = probe.split(":", 1)[0] if ":" in probe else ""
    if scheme in _UNSAFE_SCHEMES:
        attrs = attrs.replace(href_match.group(0), ' href="#"', 1)
    elif scheme in ("http", "https") or probe.startswith("//"):
        if "target=" not in attrs:
            attrs += ' target="_blank"'
        if "rel=" not in attrs:
            attrs += ' rel="noopener noreferrer"'
    return f"<a{attrs}>"
