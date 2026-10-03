"""Formal checks of a rendered PDF with PyMuPDF."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
_WS_RE = re.compile(r"\s+")


@dataclass
class PdfCheckReport:
    ok: bool
    path: str
    pages: int = 0
    text_chars: int = 0
    title_meta: str | None = None  # /Title from the PDF metadata
    title_found: bool | None = None  # expected title present in the extracted text
    bookmarks: int = 0
    fonts: list[str] = field(default_factory=list)
    not_embedded: list[str] = field(default_factory=list)
    type3: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    text = text.replace("­", "").replace(" ", " ")
    text = re.sub(r"-\s*\n\s*", "", text)  # hyphenated line breaks
    return _WS_RE.sub(" ", text).strip().casefold()


def _font_name(basefont: str) -> str:
    return _SUBSET_RE.sub("", basefont or "") or "(без имени)"


def check_pdf(
    path: Path,
    *,
    expected_title: str | Sequence[str] | None = None,
    expect_bookmarks: bool = False,
) -> PdfCheckReport:
    """Pages > 0, extractable text with the title, embedded fonts, bookmarks, /Title.

    `expected_title` is the title or its pieces (pandoc.title_fragments: the text between
    formulas and quotes), each of which must occur in the extracted text.
    """
    import pymupdf

    report = PdfCheckReport(ok=False, path=str(path))
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # PyMuPDF raises several unrelated exception types
        report.errors.append(f"PDF не открывается: {exc}")
        return report

    with doc:
        if doc.needs_pass:
            report.errors.append("PDF зашифрован")
            return report
        report.pages = doc.page_count
        if report.pages == 0:
            report.errors.append("В PDF нет страниц")
            return report

        chunks: list[str] = []
        fonts: dict[str, tuple[str, str]] = {}  # name -> (type, ext)
        for page in doc:
            chunks.append(page.get_text("text"))
            for font in page.get_fonts(full=True):
                # (xref, ext, type, basefont, name, encoding, referencer)
                ext, ftype, basefont = font[1], font[2], font[3]
                fonts.setdefault(_font_name(basefont), (ftype, ext))
        text = "".join(chunks)
        report.text_chars = len(text.strip())
        report.fonts = sorted(fonts)
        report.type3 = sorted(n for n, (t, _) in fonts.items() if t == "Type3")
        report.not_embedded = sorted(
            n for n, (t, ext) in fonts.items() if t != "Type3" and ext in ("n/a", "")
        )
        report.bookmarks = len(doc.get_toc(simple=True))
        report.title_meta = (doc.metadata or {}).get("title") or None

    if report.text_chars == 0:
        report.errors.append("Из PDF не извлекается текст (нет текстового слоя)")
    if report.not_embedded:
        report.errors.append("Шрифты не встроены в PDF: " + ", ".join(report.not_embedded))
    if report.type3:
        report.warnings.append(
            "Растровые шрифты Type 3 (текст может выглядеть размыто): " + ", ".join(report.type3)
        )
    fragments = [expected_title] if isinstance(expected_title, str) else expected_title or []
    fragments = [f for f in fragments if f.strip()]
    if fragments:
        normalised = _norm(text)
        report.title_found = all(_norm(f) in normalised for f in fragments)
        if not report.title_found:
            shown = " … ".join(fragments)
            report.warnings.append(f"Заголовок «{shown}» не найден в тексте PDF")
        if not report.title_meta:
            report.warnings.append("В метаданных PDF нет заголовка (Title)")
    if expect_bookmarks and report.bookmarks == 0:
        report.warnings.append("В PDF нет закладок, хотя в документе есть заголовки")

    report.ok = not report.errors
    return report
