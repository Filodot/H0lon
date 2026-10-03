"""Page-level PDF analysis shared by the PDF and slide extractors.

Everything here is deterministic (PyMuPDF + pymupdf4llm): a profile of every page, its
class (`text`, `math`, `scan`, `graphic`), running headers/footers, letter-spaced lines, the
PNG render for the vision agent and the Markdown of text pages.

Classification thresholds were measured on real course materials (copies only, never in
the repository):

* LaTeX articles (XCharter + CMMI/CMSY/CMEX, EURM/TeX-math, 600–2400 characters per page):
  every page with formulas has ≥ 6 characters in math fonts, usually 30–370. pymupdf4llm
  drops display formulas of such pages completely (an empty line is left) and flattens
  indices (`v_k` → `vk`), so ``MATH_FONT_MIN_CHARS = 3`` sends any page with more than a
  stray symbol to the agent; a single italic letter (1–2 characters) stays text.
* Beamer slides (454×255 pt, SFSS/SFSX text, CM math): itemize bullets are `∙` from CMSY,
  so line-initial bullet glyphs are never counted as math; text-only slides then have 0
  math characters, formula slides 7–139.
* Chromium print of a KaTeX page (Type3 fonts without names, 200–1700 characters): font
  names are useless, but math glyphs are not — formula pages carry 13–108 of them
  (∪ ∑ ⟹ μ ε ⊂ …), text pages 0. ``MATH_GLYPH_MIN = 8`` separates them; a lone `→` or
  `≤` in prose does not reach it. The text layer of such a page reads `A = +∞ ⋃ n=1 An`
  for a display union — unusable.
* Phone/tablet scans (iOS, Samsung): 0 characters and one image covering the page, or
  thousands of vector strokes (handwriting exported as paths). ``SCAN_MAX_CHARS = 20``
  (ARCHITECTURE): fewer characters than that is no text layer.
* Decorations are not graphics: theorem boxes (rounded rectangles), rules, table grids and
  logos (Chromium notes carry a ~0.3 % logo and ~290 curves on every page) are filtered
  out before the figure share is measured. A document page is `graphic` from 35 % of
  pictures with < 400 characters (the illustrated cover of the notes, 28 %, stays text and
  keeps its picture in figures/) or from 60 % whatever the text; on slides a picture or a
  diagram of 10 % is already the content (Beamer pictures cover 11–25 %).
* A full-page picture (≥ 85 % of the page) under visible text is a background only when
  it repeats on another page (template backgrounds: the Beamer title and final slides
  share one 1920×1080 JPEG) or carries ≥ 200 visible characters
  (``BACKGROUND_MIN_CHARS``). Otherwise it is the content: a phone scan with a
  «Отсканировано с помощью CamScanner» stamp (30–40 characters) is a `scan`, a slide that
  is a screenshot with a caption is `graphic`.
* Small pictures (0.05–1.5 % of the page, outside the header/footer bands, not repeated on
  ≥ 3 pages like icons and logos) are not kept as figures. Three or more of them — formulas
  pasted as pictures (MathType, screenshots) — make the page `graphic`; one or two are
  reported in quality notes. The real notes carry none: their icons (0.04 %) repeat on
  4 pages, the dotted leaders and rules are 1–2 pt high.

Text layers are repaired where it is unambiguous: letter-spaced headings of Chromium prints
(«М А Т Е М А Т И К А»), lines that lost their word spaces (XeLaTeX with system fonts:
«Вовторойработе…»), decomposed letters («и» + U+0306 → «й», NFC), words hyphenated at a
line end (glued when the document confirms it, see `join_hyphenated`).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import itertools
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf

PageKind = Literal["text", "math", "scan", "graphic"]

# ---------------------------------------------------------------- thresholds (see docstring)

SCAN_MAX_CHARS = 20  # fewer non-space characters: no usable text layer
MATH_FONT_MIN_CHARS = 3  # characters in math fonts that make a page "math"
MATH_GLYPH_MIN = 8  # math glyphs (any font) that make a page "math"
GARBLED_MIN_SHARE = 0.3  # share of U+FFFD / private use / control characters
BACKGROUND_MIN_COVER = 0.85  # a "full-page" picture covers at least this much of the page
BACKGROUND_MIN_CHARS = 200  # visible characters over a unique full-page picture: background
SCAN_MIN_IMAGE = 0.5  # image share of a text-less page that makes it a scan
SCAN_MIN_STROKES = 300  # vector path items of a text-less page (handwriting as paths)
EMPTY_MAX_VISUAL = 0.02  # a text-less page with less visual content is empty
# Documents: a page is "graphic" when pictures/figures cover this much and there is little
# text, or when they dominate the page whatever the text.
DOC_GRAPHIC_MIN = 0.35
DOC_GRAPHIC_MAX_CHARS = 400
DOC_GRAPHIC_DOMINANT = 0.6
# Slides: a picture or a diagram is usually the content of the slide.
SLIDE_GRAPHIC_MIN = 0.10
# Ignored when measuring the figure share: tiny pictures (icons, leader dots, logos).
MIN_IMAGE_SHARE = 0.005
MIN_DRAWING_SHARE = 0.004
# Small pictures that are not kept as figures but may be content (formulas as pictures).
SMALL_PICTURE_MIN_SHARE = 0.0005  # a 20×12 pt picture on A4
SMALL_PICTURE_MIN_SIDE = 3.0  # pt: thinner pictures are rules and dotted leaders
SMALL_PICTURES_VISION = 3  # this many on a text page send it to the agent
REPEATED_PICTURE_PAGES = 3  # a picture on this many pages is an icon or a logo

# Running headers/footers: lines in the top/bottom band repeated on ≥ 60 % of the pages
# of a document of at least 3 pages.
BAND_SHARE = 0.08
RUNNING_MIN_SHARE = 0.6
RUNNING_MIN_PAGES = 3

# Vision renders: long side ≤ 2000 px, ~200 dpi; small pages (slides) get at least 1400 px.
RENDER_DPI = 200
RENDER_MAX_SIDE = 2000
RENDER_MIN_SIDE = 1400
RENDER_MAX_DPI = 300
FIGURE_DPI = 150
FIGURE_MIN_SHARE = 0.015  # pictures of text pages smaller than this are dropped (icons)
FIGURE_MIN_SIDE_PX = 24  # strips (rules, borders) are not figures
FIGURE_MIN_BYTES_PER_PX = 0.01  # PNG this compressible is a flat area, not a picture

_GRID = 48  # coverage grid (cells per side) for union areas of pictures and figures

# Font names of math fonts (after the subset prefix). `Symbol` only as a whole name:
# SegoeUISymbol and friends carry dingbats, not formulas.
MATH_FONT_RE = re.compile(
    r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EURM|EUSM|EUFM|EUEX|TeX-?math|TeXGyre\w*Math|"
    r"Cambria-?Math|STIX|LatinModern-?Math|LMMath|XITS|Asana|KaTeX|MathJax|rsfs|"
    r"stmary|esint|wasy|bbold|dsrom|MTSY|MTEX|MT-?Extra|Euclid|^Symbol(?:MT)?$",
    re.IGNORECASE,
)
OCR_FONT_RE = re.compile(r"GlyphLess|Tesseract|OCR-?Invisible|Invisible", re.IGNORECASE)
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

# Glyphs that open list items (bullets of Beamer, Word, Chromium) — never math at line start.
BULLETS = frozenset("∙•◦▪▫■□►▶▸‣⁃●○◆◇◈❖★☆✓✔✗✦✴❅☀–—-*·⋅")
# Large operators: a strong sign of display formulas.
BIG_OPERATORS = frozenset("∑∏∐∫∬∭∮∯∰⋀⋁⋂⋃⨀⨁⨂⨄⨆⨅")
_EXTRA_MATH = frozenset("±×÷¬∂∇√∞")
# Letterlike symbols of ordinary prose («задача №3», ™, ℃) are not formulas.
_PROSE_LETTERLIKE = frozenset("№™℃℉℅℗")


def is_math_glyph(ch: str) -> bool:
    """True for operators, relations, arrows, math alphanumerics, ℝ-style letters, Greek."""
    o = ord(ch)
    if ch in _PROSE_LETTERLIKE:
        return False
    return (
        0x2200 <= o <= 0x22FF  # mathematical operators
        or 0x2A00 <= o <= 0x2AFF  # supplemental operators
        or 0x27C0 <= o <= 0x27EF  # misc mathematical symbols A
        or 0x2980 <= o <= 0x29FF  # misc mathematical symbols B
        or 0x2190 <= o <= 0x21FF  # arrows
        or 0x27F0 <= o <= 0x27FF  # supplemental arrows A (⟹)
        or 0x2900 <= o <= 0x297F  # supplemental arrows B
        or 0x1D400 <= o <= 0x1D7FF  # mathematical alphanumerics (𝑋, 𝜌)
        or 0x2100 <= o <= 0x214F  # letterlike (ℝ, ℓ)
        or 0x0391 <= o <= 0x03C9  # Greek
        or o in (0x03D1, 0x03D5, 0x03F1, 0x03F5)  # ϑ ϕ ϱ ϵ
        or ch in _EXTRA_MATH
    )


def _is_garbled(ch: str) -> bool:
    o = ord(ch)
    return (
        ch == "�"
        or 0xE000 <= o <= 0xF8FF
        or o >= 0xF0000
        or (unicodedata.category(ch) == "Cc" and ch not in "\t\n\r")
    )


def clean_font(name: str) -> str:
    return _SUBSET_PREFIX.sub("", name or "")


# ---------------------------------------------------------------- profiles


@dataclass
class LineInfo:
    """One text line of a page (PyMuPDF `rawdict` line)."""

    index: int  # position on the page (extraction order)
    text: str  # as extracted, spaces normalized
    bbox: tuple[float, float, float, float]
    size: float  # largest span size
    font: str  # font of the largest span
    bold: bool
    fixed: str | None = None  # glued version of a letter-spaced line, if unambiguous
    block: int = 0  # index of the text block
    fix_kind: str | None = None  # "despaced" (М А Т → МАТ) or "respaced" (lost spaces)
    # (text, style) pieces of the line: style bits STYLE_BOLD | STYLE_ITALIC | STYLE_STRIKE.
    runs: list[tuple[str, int]] = field(default_factory=list)
    tracking: float | None = None  # letter gap (em) of an unambiguous letter-spaced line
    # A single letter-spaced word («Л Е К Т О Р»): (glued, gap in em), confirmed or not by
    # the letter-spaced lines of the document (`resolve_spacing`).
    spaced_candidate: tuple[str, float] | None = None

    @property
    def shown(self) -> str:
        """Text to use: the fixed version if any, composed (NFC: «й», not «и» + breve)."""
        return unicodedata.normalize("NFC", self.fixed if self.fixed is not None else self.text)

    def markdown(self) -> str:
        """The line as inline Markdown: escaped text, bold/italic/strikeout kept."""
        if self.fixed is not None or not self.runs:
            return escape_inline(self.shown)
        merged: list[list[Any]] = []
        for text, style in self.runs:
            if merged and merged[-1][1] == style:
                merged[-1][0] += text
            else:
                merged.append([text, style])
        out = []
        for text, style in merged:
            piece = escape_inline(text)
            core = piece.strip()
            if core and style:
                lead = piece[: len(piece) - len(piece.lstrip())]
                trail = piece[len(piece.rstrip()) :]
                if style & STYLE_STRIKE:
                    core = f"~~{core}~~"
                if style & STYLE_BOLD and style & STYLE_ITALIC:
                    core = f"***{core}***"
                elif style & STYLE_BOLD:
                    core = f"**{core}**"
                elif style & STYLE_ITALIC:
                    core = f"*{core}*"
                piece = lead + core + trail
            out.append(piece)
        return unicodedata.normalize("NFC", _normalize_spaces("".join(out)))


@dataclass
class PageProfile:
    number: int  # 1-based
    width: float
    height: float
    chars: int = 0  # non-space characters of the text layer
    visible_chars: int = 0  # of them drawn visibly (not OCR-invisible)
    lines: list[LineInfo] = field(default_factory=list)
    math_font_chars: int = 0
    math_glyphs: int = 0
    big_operators: int = 0
    garbled: int = 0
    fonts: Counter[str] = field(default_factory=Counter)
    image_frac: float = 0.0  # pictures (without full-page ones) — share of the page
    image_frac_all: float = 0.0  # all pictures including full-page ones
    full_image: float = 0.0  # share of the largest full-page picture (0: none)
    full_image_digest: str | None = None
    full_image_repeated: bool = False  # the same full-page picture is on another page
    background_image: bool = False  # the full-page picture is a background (decoration)
    small_pictures: int = 0  # small pictures that may be content (not kept as figures)
    small_digests: list[str] = field(default_factory=list)
    pictures: int = 0  # figure-sized pictures (not full-page, not logos repeated on pages)
    picture_digests: list[str] = field(default_factory=list)
    figure_frac: float = 0.0  # vector figures
    visual_frac: float = 0.0  # union of pictures (without full-page ones) and figures
    vector_items: int = 0  # path items of the page (lines + curves)
    invisible_text: bool = False  # the text layer is mostly invisible (OCR over a scan)
    ocr_font: bool = False
    scan_dpi: float | None = None
    kind: PageKind = "text"
    reason: str = ""  # vision reason ("math", "scan", "graphic", "low-text"); "empty"
    # Words of the whole document (lower case) and whether its producer hyphenates by
    # syllables (TeX): both decide how a line-end hyphen is joined (`join_hyphenated`).
    vocabulary: frozenset[str] = frozenset()
    hyphenating: bool = True

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def garbled_share(self) -> float:
        return self.garbled / self.chars if self.chars else 0.0

    @property
    def needs_vision(self) -> bool:
        return self.kind != "text"

    def math_fonts(self) -> list[str]:
        return sorted(f for f in self.fonts if MATH_FONT_RE.search(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "kind": self.kind,
            "reason": self.reason,
            "chars": self.chars,
            "math_font_chars": self.math_font_chars,
            "math_glyphs": self.math_glyphs,
            "big_operators": self.big_operators,
            "image_frac": round(self.image_frac, 3),
            "figure_frac": round(self.figure_frac, 3),
            "visual_frac": round(self.visual_frac, 3),
            "full_image": round(self.full_image, 3),
            "background_image": self.background_image,
            "small_pictures": self.small_pictures,
            "invisible_text": self.invisible_text,
            "scan_dpi": self.scan_dpi,
        }


class _Coverage:
    """Union area of rectangles on a coarse grid (pictures overlap and repeat)."""

    def __init__(self, rect: pymupdf.Rect) -> None:
        self.rect = rect
        self.cells: set[tuple[int, int]] = set()

    def add(self, r: pymupdf.Rect) -> None:
        r = pymupdf.Rect(r) & self.rect
        if r.is_empty or r.width <= 0 or r.height <= 0:
            return
        w, h = self.rect.width / _GRID, self.rect.height / _GRID
        x0 = int((r.x0 - self.rect.x0) / w)
        x1 = math.ceil((r.x1 - self.rect.x0) / w)
        y0 = int((r.y0 - self.rect.y0) / h)
        y1 = math.ceil((r.y1 - self.rect.y0) / h)
        for i in range(max(0, x0), min(_GRID, x1)):
            for j in range(max(0, y0), min(_GRID, y1)):
                self.cells.add((i, j))

    def merge(self, other: _Coverage) -> _Coverage:
        out = _Coverage(self.rect)
        out.cells = self.cells | other.cells
        return out

    @property
    def share(self) -> float:
        return len(self.cells) / (_GRID * _GRID)


def _is_frame_drawing(d: dict[str, Any]) -> bool:
    """Boxes, rounded boxes, pills, rules and table grids — decoration, not a figure.

    No slanted lines, and every curve lies at a corner of the path's rectangle (Chromium
    draws one rounded corner with up to 8 Béziers, so the curve count alone says nothing).
    """
    r = pymupdf.Rect(d.get("rect") or (0, 0, 0, 0))
    radius = max(16.0, 0.25 * min(r.width, r.height))
    corners = (r.tl, r.tr, r.bl, r.br)
    for item in d.get("items") or ():
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
            if abs(p1.x - p2.x) > 0.5 and abs(p1.y - p2.y) > 0.5:
                return False
        elif op == "c":
            for pt in item[1:5]:
                if min(abs(pt - c) for c in corners) > radius:
                    return False
    return True


def _figure_coverage(page: pymupdf.Page, band: float) -> tuple[_Coverage, int]:
    cov = _Coverage(page.rect)
    items_total = 0
    page_area = page.rect.width * page.rect.height
    try:
        drawings = page.get_drawings()
    except Exception:  # broken content streams: no figure information
        return cov, 0
    top, bottom = page.rect.y0 + band, page.rect.y1 - band
    for d in drawings:
        items = d.get("items") or ()
        items_total += sum(1 for it in items if it[0] in ("l", "c"))
        r = pymupdf.Rect(d["rect"]) & page.rect
        if r.is_empty or r.width < 2 or r.height < 2:
            continue  # rules, underlines, fraction bars
        if r.width * r.height < MIN_DRAWING_SHARE * page_area:
            continue  # glyph-sized paths, bullets, logos
        if r.y1 <= top or r.y0 >= bottom:
            continue  # decorations in the header/footer bands
        if _is_frame_drawing(d):
            continue
        cov.add(r)
    return cov, items_total


STYLE_BOLD, STYLE_ITALIC, STYLE_STRIKE = 1, 2, 4


def _span_style(span: dict[str, Any], font: str) -> int:
    flags = span.get("flags", 0)
    style = 0
    if flags & 16 or re.search(r"bold|black|heavy|semibold|-sx|sfsx|cmbx", font, re.I):
        style |= STYLE_BOLD
    if flags & 2 or re.search(r"italic|oblique|-it\b|sfti|cmti", font, re.I):
        style |= STYLE_ITALIC
    if span.get("char_flags", 0) & 1:
        style |= STYLE_STRIKE
    return style


def _line_bold(span: dict[str, Any]) -> bool:
    font = span.get("font") or ""
    return bool(span.get("flags", 0) & 16) or bool(re.search(r"bold|black|heavy", font, re.I))


def profile_page(page: pymupdf.Page, number: int | None = None) -> PageProfile:
    """Measure one page (no classification; see `classify_page`)."""
    rect = page.rect
    prof = PageProfile(number=number or page.number + 1, width=rect.width, height=rect.height)
    band = rect.height * BAND_SHARE
    try:
        # Image blocks come clipped to their visible area (Chromium draws dotted leaders
        # as page-sized images clipped to a 1 pt strip).
        raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT)
    except Exception:
        raw = {"blocks": []}
    invisible = 0
    line_index = 0
    image_blocks: list[dict[str, Any]] = []
    for b_index, block in enumerate(raw.get("blocks") or ()):
        if block.get("type") == 1:
            image_blocks.append(block)
            continue
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or ():
            spans = line.get("spans") or ()
            chars: list[dict[str, Any]] = []
            first_seen = False
            line_math = False
            best_size, best_font, bold = 0.0, "", False
            runs: list[tuple[str, int]] = []
            for span in spans:
                font = clean_font(span.get("font", ""))
                style = _span_style(span, font)
                is_math_font = bool(MATH_FONT_RE.search(font))
                line_math = line_math or is_math_font
                hidden = (span.get("char_flags", 16) & (16 | 32)) == 0 or span.get(
                    "alpha", 255
                ) == 0
                if OCR_FONT_RE.search(font):
                    prof.ocr_font = True
                size = float(span.get("size") or 0.0)
                for ch in span.get("chars") or ():
                    c = ch.get("c", "")
                    chars.append(ch)
                    if not c or c.isspace():
                        continue
                    if not first_seen:
                        first_seen = True
                        if c in BULLETS:
                            continue
                    prof.chars += 1
                    prof.fonts[font] += 1
                    if hidden:
                        invisible += 1
                    else:
                        prof.visible_chars += 1
                    if _is_garbled(c):
                        prof.garbled += 1
                    if is_math_font:
                        prof.math_font_chars += 1
                    if is_math_glyph(c):
                        prof.math_glyphs += 1
                    if c in BIG_OPERATORS:
                        prof.big_operators += 1
                span_text = "".join(ch.get("c", "") for ch in span.get("chars") or ())
                if span_text:
                    runs.append((span_text, 0 if is_math_font else style))
                if size > best_size and span_text.strip():
                    best_size, best_font, bold = size, font, _line_bold(span)
            text = _normalize_spaces("".join(ch.get("c", "") for ch in chars))
            if not text:
                continue
            info = LineInfo(
                index=line_index,
                text=text,
                bbox=tuple(line.get("bbox") or (0, 0, 0, 0)),  # type: ignore[arg-type]
                size=round(best_size, 2),
                font=best_font,
                bold=bold,
                block=b_index,
                runs=runs,
            )
            if not line_math:
                spaced = analyze_spacing(chars, best_size)
                if spaced is not None and not spaced.single_word:
                    info.fixed, info.fix_kind, info.tracking = spaced.text, "despaced", spaced.gap
                else:
                    if spaced is not None:  # one spaced word: confirmed by the document
                        info.spaced_candidate = (spaced.text, spaced.gap)
                    info.fixed = respace_chars(chars, best_size)
                    info.fix_kind = "respaced" if info.fixed is not None else None
            prof.lines.append(info)
            line_index += 1
    prof.invisible_text = prof.chars >= SCAN_MAX_CHARS and invisible > 0.8 * prof.chars

    # Pictures: union coverage. Full-page pictures are kept apart: whether one is a
    # background or the content is decided with the whole document (`classify_page`).
    page_area = rect.width * rect.height
    cov_all = _Coverage(rect)
    cov_pics = _Coverage(rect)
    best_dpi: tuple[float, float] | None = None  # (area, dpi) of the largest picture
    for info in image_blocks:
        r = pymupdf.Rect(info.get("bbox") or (0, 0, 0, 0)) & rect
        if r.is_empty:
            continue
        share = r.width * r.height / page_area if page_area else 0.0
        in_band = r.y1 <= rect.y0 + band or r.y0 >= rect.y1 - band
        if (
            SMALL_PICTURE_MIN_SHARE <= share < FIGURE_MIN_SHARE
            and min(r.width, r.height) >= SMALL_PICTURE_MIN_SIDE
            and not in_band
        ):
            prof.small_digests.append(_image_digest(info))
        if share < MIN_IMAGE_SHARE:
            continue
        cov_all.add(r)
        px_w = info.get("width") or 0
        if r.width > 0 and px_w:
            dpi = px_w / (r.width / 72.0)
            if best_dpi is None or share > best_dpi[0]:
                best_dpi = (share, dpi)
        if share >= BACKGROUND_MIN_COVER:
            if share > prof.full_image:
                prof.full_image, prof.full_image_digest = share, _image_digest(info)
            continue
        cov_pics.add(r)
        if share >= FIGURE_MIN_SHARE and not in_band:
            prof.picture_digests.append(_image_digest(info))
    prof.image_frac_all = cov_all.share
    prof.image_frac = cov_pics.share
    prof.small_pictures = len(prof.small_digests)
    prof.pictures = len(prof.picture_digests)
    prof.background_image = is_background(prof)
    if best_dpi is not None:
        prof.scan_dpi = round(best_dpi[1])

    fig_cov, items = _figure_coverage(page, band)
    prof.vector_items = items
    prof.figure_frac = fig_cov.share
    prof.visual_frac = cov_pics.merge(fig_cov).share
    # Hyphen joins: the page's own words (`profile_document` widens it to the document).
    prof.vocabulary = document_vocabulary([prof])
    prof.hyphenating = document_hyphenates(page.parent, [prof])
    return prof


_TEX_PRODUCER = re.compile(r"TeX|dvipdfm|dvips", re.IGNORECASE)
# Fonts that only TeX documents carry (Computer Modern, cm-super, Euler, AMS, Latin Modern).
_TEX_FONT = re.compile(
    r"^(?:CM[A-Z]{1,5}\d+|CMU[A-Z]|EU[A-Z]{2}\d+|MS[AB]M\d+|SF[A-Z]{2,5}\d+|"
    r"LM(?:Roman|Sans|Mono|Math)|LatinModern|TeX-|XCharter)"
)


def document_hyphenates(doc: pymupdf.Document | None, profiles: Sequence[PageProfile]) -> bool:
    """Does the document hyphenate words by syllables at line ends (as TeX does)?

    The producer says so (pdfTeX, XeTeX, xdvipdfmx …) unless the file was re-saved by
    another tool; then TeX fonts do (CMR, CMMI, SFSS, XCharter …); otherwise the document
    itself: more line-end breaks whose glued word occurs elsewhere than breaks of
    hyphenated compounds that occur elsewhere.
    """
    meta = (doc.metadata if doc is not None else None) or {}
    if _TEX_PRODUCER.search(f"{meta.get('producer')} {meta.get('creator')}"):
        return True
    if any(_TEX_FONT.match(font) for p in profiles for font in p.fonts):
        return True
    vocabulary = document_vocabulary(profiles)
    glued = compound = 0
    for p in profiles:
        for left, right in itertools.pairwise(p.lines):
            m = _HYPHEN_END.search(left.shown)
            w = _FIRST_WORD.match(right.shown)
            if left.block != right.block or not m or not w or not right.shown[:1].islower():
                continue
            fragment, word = m.group(1).lower(), w.group(0).lower()
            if f"{fragment}{word}" in vocabulary:
                glued += 1
            elif f"{fragment}-{word}" in vocabulary:
                compound += 1
    return glued > compound


def _image_digest(block: dict[str, Any]) -> str:
    data = block.get("image") or b""
    if not isinstance(data, bytes | bytearray):
        data = repr(data).encode()
    size = f"{block.get('width')}x{block.get('height')}".encode()
    return hashlib.blake2b(bytes(data) + size, digest_size=12).hexdigest()


def is_background(p: PageProfile) -> bool:
    """A full-page picture under visible text that is decoration, not content.

    It is a background when the same picture is on another page (a template) or when the
    page carries ≥ ``BACKGROUND_MIN_CHARS`` visible characters over it; a unique full-page
    picture under a stamp or a caption («Отсканировано с помощью CamScanner») is content.
    """
    if not p.full_image or p.visible_chars < SCAN_MAX_CHARS:
        return False
    return p.full_image_repeated or p.visible_chars >= BACKGROUND_MIN_CHARS


def classify_page(p: PageProfile, *, slides: bool = False) -> tuple[PageKind, str]:
    """(kind, reason) of a profiled page; reason "" for text, "empty" for a blank page."""
    full_picture = p.full_image > 0 and not is_background(p)
    visual = max(p.visual_frac, p.full_image) if full_picture else p.visual_frac
    if p.chars < SCAN_MAX_CHARS:
        if p.image_frac_all >= SCAN_MIN_IMAGE or p.vector_items >= SCAN_MIN_STROKES:
            return "scan", "scan"
        if visual >= EMPTY_MAX_VISUAL:
            return "graphic", "graphic"
        if p.small_pictures >= SMALL_PICTURES_VISION or (slides and p.pictures):
            return "graphic", "graphic"
        # A short text («Глава 3», «Литература») is text; only a page without any is empty.
        return ("text", "empty") if p.chars == 0 else ("text", "")
    if p.invisible_text or (p.ocr_font and p.image_frac_all >= BACKGROUND_MIN_COVER):
        return "scan", "scan"  # OCR layer over a scanned image: the image is the source
    if p.garbled_share >= GARBLED_MIN_SHARE:
        return "scan", "low-text"  # broken ToUnicode: the text layer is unusable
    if full_picture:
        # A scan with a visible stamp or watermark; on slides a screenshot with a caption.
        return ("graphic", "graphic") if slides else ("scan", "scan")
    if p.math_font_chars >= MATH_FONT_MIN_CHARS or p.math_glyphs >= MATH_GLYPH_MIN:
        return "math", "math"
    if slides:
        # Text slides keep no pictures: any picture that is not a logo is the content.
        if visual >= SLIDE_GRAPHIC_MIN or p.pictures:
            return "graphic", "graphic"
    elif (visual >= DOC_GRAPHIC_MIN and p.chars < DOC_GRAPHIC_MAX_CHARS) or (
        visual >= DOC_GRAPHIC_DOMINANT
    ):
        return "graphic", "graphic"
    if p.small_pictures >= SMALL_PICTURES_VISION:
        return "graphic", "graphic"  # formulas pasted as pictures
    return "text", ""


def mark_repeated_pictures(profiles: Sequence[PageProfile]) -> None:
    """Document-level picture facts: repeated full-page pictures, logos and icons.

    A full-page picture on two pages is a template background; a small or figure-sized
    picture on ``REPEATED_PICTURE_PAGES`` pages is an icon or a logo, not content.
    """
    full = Counter(p.full_image_digest for p in profiles if p.full_image_digest)
    pages_of: Counter[str] = Counter()
    for p in profiles:
        pages_of.update(set(p.small_digests) | set(p.picture_digests))
    for p in profiles:
        p.full_image_repeated = bool(p.full_image_digest) and full[p.full_image_digest] >= 2
        p.small_pictures = sum(1 for d in p.small_digests if pages_of[d] < REPEATED_PICTURE_PAGES)
        p.pictures = sum(1 for d in p.picture_digests if pages_of[d] < REPEATED_PICTURE_PAGES)
        p.background_image = is_background(p)


_WORD_RE = re.compile(r"\w+(?:-\w+)*")


def document_vocabulary(profiles: Sequence[PageProfile]) -> frozenset[str]:
    """Lower-case words (hyphenated compounds included) of all lines of the document."""
    words: set[str] = set()
    for p in profiles:
        for line in p.lines:
            words.update(w.lower() for w in _WORD_RE.findall(line.shown))
    return frozenset(words)


def profile_document(doc: pymupdf.Document, *, slides: bool = False) -> list[PageProfile]:
    """Profile and classify every page of `doc` (with the document-level passes)."""
    profiles = [profile_page(page) for page in doc]
    mark_repeated_pictures(profiles)
    resolve_spacing(profiles)
    vocabulary = document_vocabulary(profiles)
    hyphenating = document_hyphenates(doc, profiles)
    for prof in profiles:
        prof.vocabulary, prof.hyphenating = vocabulary, hyphenating
        prof.kind, prof.reason = classify_page(prof, slides=slides)
    return profiles


# ---------------------------------------------------------------- letter-spaced text

_WS = re.compile(r"[ \t  -​  　]+")


def _normalize_spaces(text: str) -> str:
    return _WS.sub(" ", text).strip()


DESPACE_TIGHT_EM = 0.15  # narrower "spaces" than this are tracking in any font
DESPACE_TRACKING_TOL = 0.03  # em: a single spaced word matches the document's tracking


@dataclass
class Spacing:
    text: str  # glued line
    single_word: bool  # one uniform group of gaps: a word, or letters with real spaces
    gap: float  # median letter gap inside words, in em


def despace_chars(
    chars: Sequence[dict[str, Any]], size: float, *, trackings: Sequence[float] = ()
) -> str | None:
    """Glue a letter-spaced line («М А Т Е М А Т И К А» → «МАТЕМАТИКА») when unambiguous.

    Two words or more («А В Т О Р   К О Н С П Е К Т А»: a wider gap between words) are
    unambiguous. A single spaced word is not: «Л Е К Т О Р» of a Chromium print and the
    answer options «А Б В Г Д» look the same (Chromium tracking is 0.19–0.26 em, a real
    space 0.22–0.28 em). It is glued only when its gaps are narrower than any space
    (< 0.15 em) or match the tracking of an unambiguous spaced line of the same document
    (`trackings`, ±0.03 em).
    """
    spaced = analyze_spacing(chars, size)
    if spaced is None:
        return None
    if not spaced.single_word or _confirmed(spaced.gap, trackings):
        return spaced.text
    return None


def _confirmed(gap: float, trackings: Iterable[float]) -> bool:
    return gap < DESPACE_TIGHT_EM or any(abs(gap - t) <= DESPACE_TRACKING_TOL for t in trackings)


def resolve_spacing(profiles: Sequence[PageProfile]) -> None:
    """Glue single spaced words confirmed by the tracking of the document (`despace_chars`)."""
    trackings = [line.tracking for p in profiles for line in p.lines if line.tracking is not None]
    for p in profiles:
        for line in p.lines:
            if line.spaced_candidate is None or line.fixed is not None:
                continue
            glued, gap = line.spaced_candidate
            if _confirmed(gap, trackings):
                line.fixed, line.fix_kind = glued, "despaced"


def analyze_spacing(chars: Sequence[dict[str, Any]], size: float) -> Spacing | None:
    """Letter-spaced line → glued text, whether it is a single word, the letter gap.

    The line qualifies when ≥ 4 tokens and ≥ 80 % of them are single characters. Gaps
    between neighbouring glyphs then form two clearly separated groups (tracking inside a
    word, tracking + a space between words) or one uniform group (a single word). Anything
    in between is left as is (None).
    """
    text = _normalize_spaces("".join(ch.get("c", "") for ch in chars))
    tokens = text.split(" ")
    if len(tokens) < 4:
        return None
    singles = sum(1 for t in tokens if len(t) == 1)
    letters = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    # Letters, not formula symbols: «0 < ε < 1», «L = I +», «i > j и j < n» stay as they are.
    if singles < 4 or singles < 0.8 * len(tokens) or letters < max(4, 0.7 * len(tokens)):
        return None
    if any((is_math_glyph(c) and c not in "·") or c in "=<>+*/^_|" for c in text):
        return None
    glyphs = [ch for ch in chars if ch.get("c") and not ch["c"].isspace()]
    if len(glyphs) < 4:
        return None
    gaps = [glyphs[i + 1]["bbox"][0] - glyphs[i]["bbox"][2] for i in range(len(glyphs) - 1)]
    if any(g < -0.5 * max(size, 1.0) for g in gaps):
        return None  # glyphs go backwards: several visual lines merged, unsafe
    ordered = sorted(gaps)
    unit = max(size, 1.0)
    split_at: float | None = None
    best = 0.0
    for a, b in itertools.pairwise(ordered):
        jump = b - a
        if jump >= 0.15 * unit and (a <= 0 or b / max(a, 1e-6) >= 1.6) and jump > best:
            best, split_at = jump, (a + b) / 2
    if split_at is None:
        if ordered[-1] - ordered[0] > 0.2 * unit:
            return None  # spread without a clear break — ambiguous
        words = ["".join(ch["c"] for ch in glyphs)]
        inner = gaps
    else:
        words, cur = [], [glyphs[0]["c"]]
        for g, ch in zip(gaps, glyphs[1:], strict=False):
            if g > split_at:
                words.append("".join(cur))
                cur = []
            cur.append(ch["c"])
        words.append("".join(cur))
        inner = [g for g in gaps if g <= split_at]
    glued = " ".join(words)
    if glued.replace(" ", "") != text.replace(" ", "") or glued == text:
        return None
    # A real spaced line yields mostly multi-letter words, at least one of ≥ 3 letters.
    if sum(1 for w in words if len(w) > 1) < max(1, len(words) // 3):
        return None
    if not any(sum(c.isalpha() for c in w) >= 3 for w in words):
        return None
    gap = float(statistics.median(inner)) / unit if inner else 0.0
    return Spacing(text=glued, single_word=split_at is None, gap=round(gap, 4))


def spaced_pattern(text: str) -> re.Pattern[str]:
    """Regex for the glyphs of `text` with any (or no) whitespace between them."""
    glyphs = [re.escape(c) for c in text if not c.isspace()]
    return re.compile(r"[ \t]*".join(glyphs))


RESPACE_MIN_TOKEN = 18  # a "word" this long in Russian or English prose is glued words
RESPACE_GAP_EM = 0.12  # interword glue of justified text is ≥ ~0.2 em, kerning ≪ 0.1 em


def respace_chars(chars: Sequence[dict[str, Any]], size: float) -> str | None:
    """Restore lost word spaces («Речьпойдётосистемах» → «Речь пойдёт о системах»).

    XeLaTeX with system fonts writes interword glue as positioning only, and MuPDF does not
    always infer the spaces (justified lines lose them, ragged ones keep them). A line with
    tokens of ≥ 18 characters gets a space wherever two glyphs are more than 0.12 em apart.
    """
    text = "".join(ch.get("c", "") for ch in chars)
    tokens = text.split()
    if not tokens or max(len(t) for t in tokens) < RESPACE_MIN_TOKEN:
        return None
    threshold = max(RESPACE_GAP_EM * max(size, 1.0), 0.8)
    out: list[str] = []
    prev: dict[str, Any] | None = None
    added = 0
    for ch in chars:
        c = ch.get("c", "")
        if not c:
            continue
        if c.isspace():
            out.append(" ")
            prev = None
            continue
        gap = ch["bbox"][0] - prev["bbox"][2] if prev is not None else 0.0
        if gap > threshold and not unicodedata.combining(c):
            out.append(" ")
            added += 1
        out.append(c)
        if not unicodedata.combining(c):
            prev = ch
    if added < 2:
        return None
    return unicodedata.normalize("NFC", _normalize_spaces("".join(out)))


# ---------------------------------------------------------------- running headers / footers

# A bare number is a page number up to 3 digits («2026» on a title page is a year); framed
# («— 1234 —») up to 4.
_PAGE_NUMBER_RES = [
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$"),
    re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$"),
    re.compile(r"^\d{1,4}\s*(?:из|of)\s*\d{1,4}$", re.I),
    re.compile(
        r"^(?:стр\.?|страница|с\.|page|p\.|pg\.|слайд|slide)\s*\d{1,4}"
        r"(?:\s*(?:из|of|/)\s*\d{1,4})?$",
        re.I,
    ),
    re.compile(r"^(?=[ivxlc])c{0,3}(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$", re.I),  # roman
]


def is_page_number(text: str) -> bool:
    t = _normalize_spaces(text).strip()
    return bool(t) and len(t) <= 24 and any(r.match(t) for r in _PAGE_NUMBER_RES)


_PAGE_WORDS = re.compile(
    r"(?<!\w)(?:стр|страница|page|pg|слайд|slide)(?!\w)|(?<!\w)[сp]\.\s*\d|\d\s*(?:из|of|/)\s*\d",
    re.IGNORECASE,
)
_EDGE_NUMBER = re.compile(r"^\d{1,4}(?=\s*[|·•—–-]\s)|(?<=[|·•—–-])(\s*)\d{1,4}$")


def running_key(text: str) -> str:
    """Key of a band line for repetition: page numbers in it are masked, other digits kept.

    «СТР. 3 ИЗ 25», «Лекция 3 · стр. 5», «Глава 2 — 15» share a key with the other pages;
    «Задача 1»…«Задача 6» (headings that happen to sit in the band) do not.
    """
    t = _normalize_spaces(text).lower()
    if _PAGE_WORDS.search(t):
        t = re.sub(r"\d+", "#", t)
    else:
        t = _EDGE_NUMBER.sub(lambda m: (m.group(1) or "") + "#", t)
    return t.strip(" .·|-–—")


@dataclass
class RunningLines:
    drop: dict[int, list[LineInfo]]  # page number → lines to drop
    texts: list[str]  # distinct repeated texts (for quality notes)


def find_running_lines(profiles: Sequence[PageProfile]) -> RunningLines:
    """Headers/footers: band lines repeated on ≥ 60 % of pages (2 of 3), page numbers.

    Bands are the top and bottom 8 % of the page. Page numbers inside a line are ignored
    when comparing (`running_key`), other digits are not: «Задача 1»…«Задача 6» in the
    band are six different lines. Repetition is also counted separately on odd and even
    pages (books alternate their headers) when there are ≥ 6 pages; lines that look like
    page numbers («4/31», «СТР. 3 ИЗ 25», «— 12 —») are dropped in the bands even in short
    documents.
    """
    n = len(profiles)
    occurrences: dict[tuple[str, str], set[int]] = defaultdict(set)
    band_lines: list[tuple[PageProfile, LineInfo, str, str]] = []
    for p in profiles:
        band = p.height * BAND_SHARE
        for line in p.lines:
            _x0, y0, _x1, y1 = line.bbox
            if y1 <= band:
                where = "top"
            elif y0 >= p.height - band:
                where = "bottom"
            else:
                continue
            key = running_key(line.shown)
            if not key:
                continue
            band_lines.append((p, line, where, key))
            occurrences[(where, key)].add(p.number)

    def repeated(pages: set[int]) -> bool:
        if n < RUNNING_MIN_PAGES:
            return False
        # 2 of 3 pages is enough: a short excerpt has its title page without the header.
        if len(pages) >= max(2, math.ceil(RUNNING_MIN_SHARE * n)):
            return True
        if n >= 6:
            odd = [p.number for p in profiles if p.number % 2]
            even = [p.number for p in profiles if not p.number % 2]
            for group in (odd, even):
                hits = len(pages & set(group))
                if group and hits >= max(3, math.ceil(RUNNING_MIN_SHARE * len(group))):
                    return True
        return False

    drop: dict[int, list[LineInfo]] = defaultdict(list)
    texts: list[str] = []
    for p, line, where, key in band_lines:
        if is_page_number(line.shown) or repeated(occurrences[(where, key)]):
            drop[p.number].append(line)
            if not is_page_number(line.shown) and line.shown not in texts:
                texts.append(line.shown)
    return RunningLines(drop=dict(drop), texts=texts)


# ---------------------------------------------------------------- text of a page


SOFT_HYPHEN = "\u00ad"
_HYPHEN_END = re.compile(r"(\w+)([-\u00ad\u2010])$")
_FIRST_WORD = re.compile(r"\w+")
# Second parts of Russian compounds that are never syllables of a hyphenated word.
_PARTICLES = frozenset({"то", "либо", "нибудь", "ка", "таки", "де"})


def join_hyphenated(
    left: str, right: str, *, vocabulary: frozenset[str] = frozenset(), hyphenating: bool = True
) -> str | None:
    """Join a line ending with a hyphen and the next line, or None (not a hyphenated word).

    «ин-» + «тересовать» → «интересовать», «санкт-» + «петербургский» →
    «санкт-петербургский». A soft hyphen is always a syllable break; otherwise the
    document decides: the hyphenated compound or the glued word occurring elsewhere in it,
    a particle («кто-» + «то»). Without evidence, documents of TeX (which hyphenates by
    syllables: 110 breaks in a 31-page lab report) are glued and others keep the hyphen
    (Chromium, Word and PowerPoint break lines at hyphens of compounds).
    """
    m = _HYPHEN_END.search(left)
    if not m or not right[:1].islower():
        return None
    w = _FIRST_WORD.match(right)
    if w is None:
        return None
    fragment, mark, word = m.group(1).lower(), m.group(2), w.group(0).lower()
    glue = left[:-1] + right
    keep = left + right
    if mark == SOFT_HYPHEN:
        return glue
    if word in _PARTICLES or f"{fragment}-{word}" in vocabulary:
        return keep
    if f"{fragment}{word}" in vocabulary:
        return glue
    return glue if hyphenating else keep


def page_text(profile: PageProfile, drop: Iterable[LineInfo] = (), *, reflow: bool = False) -> str:
    """Plain text of the page (letter-spacing fixed, dropped lines removed), block by block.

    Words hyphenated at a line end are joined (`join_hyphenated`). With `reflow` the lines
    of one block become one paragraph; otherwise line breaks are kept (agent hints).
    """
    skip = {line.index for line in drop}
    blocks: list[list[str]] = []
    current_block: int | None = None
    for line in profile.lines:
        if line.index in skip:
            continue
        if current_block is None or line.block != current_block:
            blocks.append([])
        current_block = line.block
        lines = blocks[-1]
        text = line.shown
        joined = (
            join_hyphenated(
                lines[-1],
                text,
                vocabulary=profile.vocabulary,
                hyphenating=profile.hyphenating,
            )
            if lines
            else None
        )
        if joined is not None:
            lines[-1] = joined
        else:
            lines.append(text)
    joiner = " " if reflow else "\n"
    text = "\n\n".join(joiner.join(b) for b in blocks if b)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def cyrillic_ratio(texts: Iterable[str]) -> float | None:
    """Share of Cyrillic letters among letters outside `$…$` formulas (None: no letters)."""
    cyr = total = 0
    for text in texts:
        stripped = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", text, flags=re.S)
        for ch in stripped:
            if ch.isalpha():
                total += 1
                if "Ѐ" <= ch <= "ӿ":
                    cyr += 1
    return round(cyr / total, 3) if total else None


def median_or_zero(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


# ---------------------------------------------------------------- rendering


def render_zoom(width: float, height: float) -> float:
    long_side = max(width, height, 1.0)
    zoom = RENDER_DPI / 72.0
    if long_side * zoom < RENDER_MIN_SIDE:
        zoom = min(RENDER_MIN_SIDE / long_side, RENDER_MAX_DPI / 72.0)
    if long_side * zoom > RENDER_MAX_SIDE:
        zoom = RENDER_MAX_SIDE / long_side
    return zoom


def render_page_png(page: pymupdf.Page, target: Path) -> Path:
    """PNG of the page for the vision agent: ~200 dpi, long side within 1400–2000 px."""
    zoom = render_zoom(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    pix.save(str(tmp), output="png")
    tmp.replace(target)
    return target


def image_to_png(src: Path | bytes, target: Path, *, max_side: int = RENDER_MAX_SIDE) -> Path:
    """Convert a picture (PNG/JPEG/GIF/BMP/TIFF …) to an RGB PNG with long side ≤ max_side."""
    pix = pymupdf.Pixmap(src) if isinstance(src, bytes) else pymupdf.Pixmap(str(src))
    if pix.alpha or (pix.colorspace and pix.colorspace.n not in (1, 3)):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix, 0)
    long_side = max(pix.width, pix.height)
    if long_side > max_side:
        # Pixmap.shrink only halves; re-render through a one-page PDF for exact scaling.
        doc = pymupdf.open()
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        zoom = max_side / long_side
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        doc.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(target), output="png")
    return target


# ---------------------------------------------------------------- Markdown of text pages


def redact_lines(page: pymupdf.Page, lines: Iterable[LineInfo]) -> int:
    """Remove the text of `lines` from `page` (a working copy): images and art stay."""
    n = 0
    for line in lines:
        x0, y0, x1, y1 = line.bbox
        h = y1 - y0
        r = pymupdf.Rect(x0 + 0.2, y0 + 0.3 * h, x1 - 0.2, y1 - 0.3 * h)
        if r.is_empty:
            continue
        page.add_redact_annot(r)
        n += 1
    if n:
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    return n


_P4L_KW: dict[str, Any] | None = None


def _p4l_kwargs() -> dict[str, Any]:
    """Keyword arguments for pymupdf4llm.to_markdown valid in the active mode."""
    global _P4L_KW
    if _P4L_KW is None:
        import pymupdf4llm

        kw: dict[str, Any] = {"show_progress": False, "force_text": True}
        if getattr(pymupdf4llm, "_use_layout", False):
            kw.update(use_ocr=False, header=True, footer=True)
        _P4L_KW = kw
    return dict(_P4L_KW)


def pymupdf4llm_page(doc: pymupdf.Document, index: int, *, images: bool) -> str:
    """Raw pymupdf4llm Markdown of one page (0-based index).

    Pictures are embedded as `data:` URIs: pymupdf4llm mangles the paths it writes images
    to (spaces, brackets and dashes become `_`/`-`), so any temporary directory under a
    user name with a space would fail.
    """
    import pymupdf4llm

    kw = _p4l_kwargs()
    if images:
        kw.update(embed_images=True, image_format="png", dpi=FIGURE_DPI)
    return pymupdf4llm.to_markdown(doc, pages=[index], **kw)


_IMG_LINK = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_DATA_URI = re.compile(r"^data:image/[\w.+-]+;base64,(.*)$", re.S)
_SUP = re.compile(r"<sup>(.*?)</sup>", re.S)
_SUB = re.compile(r"<sub>(.*?)</sub>", re.S)
_TAGS = re.compile(r"</?(?:mark|u|span|font|small|big|ins|del)\b[^>]*>", re.I)
_PICTURE_TEXT = re.compile(
    r"<!--\s*Start of picture text\s*-->(.*?)<!--\s*End of picture text\s*-->", re.S
)
_OMITTED = re.compile(r"^\s*\*\*==>.*?<==\*\*\s*$", re.M)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t#]*$")
_TEX_COMMAND = re.compile(r"\\(?=[A-Za-z])")


_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")


def sub_outside_code(pattern: re.Pattern[str], repl: str, line: str) -> str:
    """`pattern.sub(repl, …)` on the parts of a Markdown line outside `code spans`."""
    out: list[str] = []
    pos = 0
    for m in _CODE_SPAN.finditer(line):
        out += [pattern.sub(repl, line[pos : m.start()]), m.group(0)]
        pos = m.end()
    out.append(pattern.sub(repl, line[pos:]))
    return "".join(out)


def _script(m: re.Match[str], mark: str) -> str:
    inner = m.group(1).strip()
    if not inner:
        return ""
    inner = inner.replace(mark, "\\" + mark).replace(" ", "\\ ")
    return f"{mark}{inner}{mark}"


def demote_heading(line: str, shift: int = 2) -> str:
    """`# Title` → `### Title` (levels # and ## belong to the Source Doc structure)."""
    m = _HEADING.match(line)
    if not m:
        return line
    level = min(6, len(m.group(1)) + shift)
    text = m.group(2).strip()
    while text.startswith("**") and text.endswith("**") and len(text) > 4:
        text = text[2:-2].strip()
    return "#" * level + " " + text


def cleanup_markdown(
    md: str, *, fixes: Sequence[tuple[str, str]] = (), heading_shift: int = 2
) -> str:
    """Post-process pymupdf4llm output into the Pandoc Markdown of a Source Doc page."""
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = _PICTURE_TEXT.sub(
        lambda m: (
            "\n\n"
            + "\\\n".join(s.strip() for s in re.split(r"<br\s*/?>", m.group(1)) if s.strip())
            + "\n\n"
        ),
        text,
    )
    text = _OMITTED.sub("", text)
    text = re.sub(r"<br\s*/?>", "\\\n", text)
    text = _SUP.sub(lambda m: _script(m, "^"), text)
    text = _SUB.sub(lambda m: _script(m, "~"), text)
    text = _TAGS.sub("", text)
    for original, fixed in fixes:
        text = spaced_pattern(original).sub(lambda _m, f=fixed: f, text)
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        stripped = line.strip()
        if fence is not None:
            out.append(line.rstrip())
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            out.append(line.rstrip())
            continue
        line = line.rstrip()
        if heading_shift and _HEADING.match(line):
            line = demote_heading(line, heading_shift)
        elif line.lstrip().startswith(":::"):
            line = "\\" + line.lstrip()
        # Raw TeX that a page shows as text (a failed KaTeX render) must stay text; inside
        # `code` a backslash is already literal.
        if "\\" in line:
            line = sub_outside_code(_TEX_COMMAND, r"\\\\", line)
        out.append(line)
    text = unicodedata.normalize("NFC", "\n".join(out))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class PageMarkdown:
    markdown: str
    figures: list[Path] = field(default_factory=list)


def text_page_markdown(
    work: pymupdf.Document,
    profile: PageProfile,
    *,
    figures_dir: Path | None,
    figure_prefix: str,
    fixes: Sequence[tuple[str, str]] = (),
) -> PageMarkdown:
    """Markdown of a text page through pymupdf4llm (headers already redacted from `work`).

    Pictures larger than `FIGURE_MIN_SHARE` of the page are saved as
    `figures/<prefix>_<k>.png` and linked relatively; smaller ones (icons) are dropped
    (pages with several small pictures are not text pages: see `classify_page`).
    """
    index = profile.number - 1
    figures: list[Path] = []
    md = pymupdf4llm_page(work, index, images=figures_dir is not None)
    page_px = (profile.width / 72 * FIGURE_DPI) * (profile.height / 72 * FIGURE_DPI)
    counter = 0

    def relink(m: re.Match[str]) -> str:
        nonlocal counter
        data = _DATA_URI.match(m.group(2))
        if figures_dir is None or data is None:
            return ""  # not a picture embedded for this page
        try:
            blob = base64.b64decode(data.group(1), validate=False)
            pix = pymupdf.Pixmap(blob)
            pixels = max(1, pix.width * pix.height)
            share = pixels / page_px if page_px else 1.0
            thin = min(pix.width, pix.height) < FIGURE_MIN_SIDE_PX
            # A few hundred bytes for a large area: rules, gaps, a flat backdrop.
            blank = len(blob) / pixels < FIGURE_MIN_BYTES_PER_PX
        except Exception:
            return ""
        if share < FIGURE_MIN_SHARE or thin or blank:
            return ""
        counter += 1
        figures_dir.mkdir(parents=True, exist_ok=True)
        target = figures_dir / f"{figure_prefix}_{counter}.png"
        if pix.alpha or (pix.colorspace and pix.colorspace.n not in (1, 3)):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix, 0)
        pix.save(str(target), output="png")
        figures.append(target)
        return f"![]({figures_dir.name}/{target.name})"

    md = _IMG_LINK.sub(relink, md)
    return PageMarkdown(markdown=cleanup_markdown(md, fixes=fixes), figures=figures)


def line_fixes(profile: PageProfile, drop: Iterable[LineInfo] = ()) -> list[tuple[str, str]]:
    skip = {line.index for line in drop}
    return [
        (line.text, line.fixed)
        for line in profile.lines
        if line.fixed is not None and line.index not in skip
    ]


# ---------------------------------------------------------------- Markdown helpers

_INLINE_SPECIAL = re.compile(r"([\\`*_\[\]<>$~^|])")


def escape_inline(text: str) -> str:
    """Escape Markdown inline syntax (no line-start handling)."""
    return _INLINE_SPECIAL.sub(r"\\\1", text)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#$~^|])")
# Line starts Pandoc would read as structure. A backslash may only precede punctuation
# (`\1)` stays a literal backslash), so for «1) …» and «2. …» the dot/paren is escaped.
_LINE_START = re.compile(r"^(\s*)(?:(?P<num>\d+)(?P<np>[.)])|(?P<mark>[-+]|:{3,}|=+|>))(?=\s|$)")


def _escape_line_start(m: re.Match[str]) -> str:
    if m.group("num"):
        return m.group(1) + m.group("num") + "\\" + m.group("np")
    return m.group(1) + "\\" + m.group("mark")


def escape_markdown(text: str) -> str:
    """Literal text for Pandoc Markdown (text layer in `uncertain` blocks, slide titles)."""
    out = []
    for line in text.split("\n"):
        line = _MD_SPECIAL.sub(r"\\\1", line)
        line = _LINE_START.sub(_escape_line_start, line)
        out.append(line)
    return "\n".join(out)


def uncertain_block(title: str, text: str, *, intro: str | None = None) -> str:
    """`::: {.uncertain title="…"}` with the escaped text layer (or a note that there is none)."""
    safe_title = title.replace("\\", "").replace('"', "'")
    body: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if paragraphs:
        if intro:
            body.append(intro)
        body += [escape_markdown(p).replace("\n", "\\\n") for p in paragraphs]
    else:
        body.append("Текстового слоя нет.")
    return f'::: {{.uncertain title="{safe_title}"}}\n' + "\n\n".join(body) + "\n:::"


def uncertain_note(title: str, message: str) -> str:
    """`::: {.uncertain}` with a short message of ours (already valid Markdown)."""
    safe_title = title.replace("\\", "").replace('"', "'")
    return f'::: {{.uncertain title="{safe_title}"}}\n{message}\n:::'


EMPTY_PAGE = "<!-- пустая страница -->"


def open_pdf(path: Path) -> pymupdf.Document:
    doc = pymupdf.open(str(path))
    if doc.needs_pass:
        doc.close()
        raise ValueError("PDF защищён паролем — снимите защиту и добавьте файл заново")
    return doc


@contextlib.contextmanager
def opened(path: Path):
    doc = open_pdf(path)
    try:
        yield doc
    finally:
        doc.close()
