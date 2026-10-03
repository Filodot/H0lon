from __future__ import annotations

from pathlib import Path

import pytest
from test_sources_kit import (
    A4,
    WIDE,
    break_xref,
    make_docx,
    make_pdf,
    make_png,
    make_pptx,
    truncate_file,
)

from h0lon.sources import detect
from h0lon.sources.detect import PdfProfile, classify_pdf
from h0lon.sources.ingest import IngestError, detect_kind


def _profile(**kw) -> PdfProfile:
    base = dict(
        pages=10,
        sampled=10,
        chars_per_page=1500.0,
        text_layer=1.0,
        aspect=0.707,
        aspect_median=0.707,
        image_cover=0.0,
    )
    base.update(kw)
    return PdfProfile(**base)


# ---------------------------------------------------------------- classification rules


@pytest.mark.parametrize(
    ("profile", "kind"),
    [
        # The five real PDFs from the module docstring, as profiles.
        (
            _profile(chars_per_page=218, text_layer=0.97, aspect=1.778, aspect_median=1.778),
            "slides",
        ),
        (_profile(chars_per_page=1582), "pdf-text"),
        (_profile(chars_per_page=696), "pdf-text"),
        (_profile(chars_per_page=0, text_layer=0, aspect=0.773, aspect_median=0.773), "pdf-scan"),
        (_profile(chars_per_page=0, text_layer=0, image_cover=1.0), "pdf-scan"),
        # Boundaries.
        (_profile(chars_per_page=39, text_layer=0.1), "pdf-scan"),
        (_profile(chars_per_page=40, text_layer=0.6), "pdf-text"),
        (_profile(chars_per_page=1200, aspect_median=1.33), "slides"),
        (_profile(chars_per_page=1201, aspect_median=1.33), "pdf-text"),
        (_profile(chars_per_page=300, aspect_median=1.29), "pdf-text"),
        # Landscape without text: a scan (book spread, whiteboard photo), not slides.
        (_profile(chars_per_page=0, text_layer=0, aspect_median=1.5), "pdf-scan"),
        # Picture slides with titles on some of them are still slides.
        (_profile(chars_per_page=12, text_layer=0.3, aspect_median=1.78), "slides"),
    ],
)
def test_classify_pdf_rules(profile: PdfProfile, kind: str) -> None:
    assert classify_pdf(profile) == kind


@pytest.mark.parametrize(
    ("creator", "producer"),
    [
        ("LaTeX with Beamer class", "pdfTeX-1.40.29"),
        ("Microsoft® PowerPoint® для Microsoft 365", "Microsoft® PowerPoint® для Microsoft 365"),
        ("Keynote", "macOS Version 15.0 Quartz PDFContext"),
        ("Impress", "LibreOffice 24.2"),
    ],
)
def test_presentation_software_means_slides(creator: str, producer: str) -> None:
    # Even portrait and text-heavy: the metadata wins.
    assert classify_pdf(_profile(creator=creator, producer=producer)) == "slides"


# ---------------------------------------------------------------- real (generated) PDFs


def test_profile_and_kind_of_generated_pdfs(tmp_path: Path) -> None:
    text = make_pdf(tmp_path / "text.pdf", chars=1500)
    wide = make_pdf(tmp_path / "wide.pdf", size=WIDE, chars=250)
    scan = make_pdf(tmp_path / "scan.pdf", chars=0, image=True)
    beamer = make_pdf(
        tmp_path / "beamer.pdf",
        size=(400, 300),
        chars=200,
        metadata={"creator": "LaTeX with Beamer class", "producer": "pdfTeX"},
    )
    assert detect_kind(str(text)) == "pdf-text"
    assert detect_kind(str(wide)) == "slides"
    assert detect_kind(str(scan)) == "pdf-scan"
    assert detect_kind(str(beamer)) == "slides"

    p = detect.profile_pdf(text)
    assert p.pages == 3 and p.sampled == 3
    assert p.text_layer == 1.0 and p.chars_per_page >= 1400
    assert p.aspect == pytest.approx(A4[0] / A4[1], abs=1e-3)
    s = detect.profile_pdf(scan)
    assert s.chars_per_page == 0 and s.text_layer == 0 and s.image_cover > 0.9


def test_pdf_signals_units_quality_and_notes(tmp_path: Path) -> None:
    scan = detect.profile_pdf(make_pdf(tmp_path / "scan.pdf", chars=0, image=True))
    units, quality = detect.pdf_signals(scan, "pdf-scan")
    assert units == {"pages": 3}
    assert set(quality) >= {"text_layer", "chars_per_page", "aspect", "image_cover"}
    assert any("текстового слоя" in n for n in quality["notes"])

    wide = detect.profile_pdf(make_pdf(tmp_path / "w.pdf", size=WIDE, chars=200, pages=5))
    units, quality = detect.pdf_signals(wide, "slides")
    assert units == {"pages": 5, "slides": 5}
    assert quality["aspect"] == pytest.approx(16 / 9, abs=1e-3)
    assert "notes" not in quality

    ocr = _profile(image_cover=1.0)
    assert any("OCR" in n for n in detect.pdf_signals(ocr, "pdf-text")[1]["notes"])
    mixed = _profile(text_layer=0.5)
    assert any("только на 5 из 10" in n for n in detect.pdf_signals(mixed, "pdf-text")[1]["notes"])


def test_large_pdf_is_sampled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "PROFILE_MAX_PAGES", 4)
    p = detect.profile_pdf(make_pdf(tmp_path / "big.pdf", pages=10, chars=100))
    assert p.pages == 10 and p.sampled == 4
    notes = detect.pdf_signals(p, "pdf-text")[1]["notes"]
    assert any("4 из 10" in n for n in notes)


def test_broken_and_encrypted_pdf(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    with pytest.raises(detect.InspectError):
        detect.profile_pdf(broken)

    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    locked = tmp_path / "locked.pdf"
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    doc.close()
    with pytest.raises(detect.InspectError, match="паролем"):
        detect.profile_pdf(locked)
    with pytest.raises(IngestError, match="паролем"):
        detect_kind(str(locked))


# ---------------------------------------------------------------- other formats


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("notes.md", "md"),
        ("notes.markdown", "md"),
        ("notes.TXT", "md"),
        ("paper.tex", "tex"),
        ("page.html", "web"),
        ("page.htm", "web"),
        ("photo.JPG", "handwritten"),
        ("photo.jpeg", "handwritten"),
        ("photo.heic", "handwritten"),
        ("lecture.mp4", "video"),
        ("lecture.MKV", "video"),
        ("lecture.webm", "video"),
        ("lecture.mov", "video"),
        ("talk.mp3", "audio"),
        ("talk.m4a", "audio"),
        ("talk.wav", "audio"),
        ("talk.ogg", "audio"),
        ("talk.flac", "audio"),
    ],
)
def test_kind_by_extension(tmp_path: Path, name: str, kind: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    assert detect_kind(str(path)) == kind


def test_office_files(tmp_path: Path) -> None:
    assert detect_kind(str(make_pptx(tmp_path / "deck.pptx"))) == "slides"
    assert detect_kind(str(make_docx(tmp_path / "doc.docx"))) == "docx"
    units, quality = detect.pptx_signals(tmp_path / "deck.pptx")
    assert units == {"slides": 4}
    assert quality["aspect"] == pytest.approx(16 / 9, abs=1e-2)
    assert quality["text_layer"] == 1.0 and quality["slides_with_notes"] == 4


@pytest.mark.parametrize(
    ("make", "kind"),
    [
        (lambda p: make_pdf(p, chars=1500), "pdf-text"),
        (lambda p: make_pptx(p), "slides"),
        (lambda p: make_docx(p), "docx"),
        (lambda p: make_png(p), "handwritten"),
        (lambda p: p.write_text("<!DOCTYPE html><html><head></head></html>"), "web"),
        (lambda p: p.write_text("\\documentclass{article}\n\\begin{document}x"), "tex"),
        (lambda p: p.write_text("# Лекция\n\nТекст.", encoding="utf-8"), "md"),
        (lambda p: p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20), "video"),
        (lambda p: p.write_bytes(b"ID3\x03\x00" + b"\x00" * 20), "audio"),
    ],
)
def test_kind_by_content_without_extension(tmp_path: Path, make, kind: str) -> None:
    # «Л3. Теория меры» has no real extension: Path.suffix would say «. Теория меры».
    path = tmp_path / "Л3. Теория меры"
    make(path)
    assert detect.real_suffix(path) == ""
    assert detect_kind(str(path)) == kind


def test_unknown_and_unsupported(tmp_path: Path) -> None:
    for name, hint in (("old.doc", "docx"), ("book.djvu", "PDF"), ("deck.ppt", "pptx")):
        path = tmp_path / name
        path.write_bytes(b"\xd0\xcf\x11\xe0 binary")
        with pytest.raises(IngestError, match=hint):
            detect_kind(str(path))
    data = tmp_path / "table.csv"
    data.write_text("a,b\n1,2\n")
    with pytest.raises(IngestError, match="--kind"):
        detect_kind(str(data))
    blob = tmp_path / "blob"
    blob.write_bytes(b"\x00\x01\x02\x03" * 10)
    with pytest.raises(IngestError, match="--kind"):
        detect_kind(str(blob))


def test_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="не найден"):
        detect_kind(str(tmp_path / "нет такого.pdf"))
    with pytest.raises(IngestError, match="каталог"):
        detect_kind(str(tmp_path))


# ---------------------------------------------------------------- URLs


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "video"),
        ("https://m.youtube.com/watch?v=x", "video"),
        ("https://youtu.be/dQw4w9WgXcQ", "video"),
        ("HTTPS://YOUTU.BE/x", "video"),
        ("https://vk.com/video-12345_678", "video"),
        ("https://vk.com/clip-1_2", "video"),
        ("https://vk.com/wall-1_2?z=video-1_2%2Fpl_wall", "video"),
        ("https://vkvideo.ru/video-1_2", "video"),
        ("https://rutube.ru/video/0123456789abcdef/", "video"),
        ("https://example.org/lecture.mp4", "video"),
        ("https://example.org/talk.mp3?x=1", "audio"),
        ("https://vk.com/id1", "web"),
        ("https://ru.wikipedia.org/wiki/Случайная_величина", "web"),
        ("http://example.org/notes.pdf", "web"),  # downloaded only if it is HTML
        ("https://notyoutube.com/watch", "web"),
    ],
)
def test_classify_url(url: str, kind: str) -> None:
    assert detect_kind(url) == kind


def test_url_display() -> None:
    assert (
        detect.url_display("https://ru.wikipedia.org/wiki/%D0%A1%D0%BB%D1%83%D1%87%D0%B0%D0%B9")
        == "ru.wikipedia.org/wiki/Случай"
    )
    assert detect.url_display("https://www.example.org/") == "example.org"


# ---------------------------------------------------------------- HTML helpers


def test_decode_html_charsets() -> None:
    text = "<html><head><title>Случайная величина</title></head></html>"
    assert detect.decode_html(text.encode("utf-8")) == (text, "utf-8")
    cp = text.encode("cp1251")
    assert detect.decode_html(cp, "windows-1251") == (text, "cp1251")
    assert detect.decode_html(cp)[0] == text  # UTF-8 fails, cp1251 fallback
    meta = '<html><head><meta charset="koi8-r"><title>Мера</title></head></html>'
    assert detect.decode_html(meta.encode("koi8_r")) == (meta, "koi8-r")
    # A lying header: declared UTF-8, bytes are cp1251 with a meta that tells the truth.
    meta1251 = '<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">Мера'
    assert detect.decode_html(meta1251.encode("cp1251"), "utf-8")[0] == meta1251
    assert detect.decode_html(b"\xef\xbb\xbf<p>\xd0\x96</p>") == ("<p>Ж</p>", "utf-8")


def test_html_title() -> None:
    assert detect.html_title("<title>\n  A &amp; B \n</title>") == "A & B"
    og = '<head><meta property="og:title" content="OG заголовок"></head><body>'
    assert detect.html_title(og) == "OG заголовок"
    assert detect.html_title("<html><body><h1>x</h1></body></html>") is None


# ---------------------------------------------------------------- review fixes


def test_decode_html_keeps_the_declared_charset_on_a_stray_byte() -> None:
    page = (
        "<html><head><meta charset='utf-8'><title>Случайная величина</title></head>"
        "<body>Случайная величина — функция на пространстве исходов.</body></html>"
    ).encode()
    damaged = page.replace(b"<body>", b"<body>\xff")
    for declared in ("utf-8", None):  # HTTP header + meta, meta only
        decoded = detect.decode_html_detailed(damaged, declared)
        assert (decoded.encoding, decoded.replaced, decoded.guessed) == ("utf-8", 1, False)
        assert "функция на пространстве исходов" in decoded.text
    # Nothing declared at all: still UTF-8, not cp1251 mojibake.
    bare = damaged.replace(b"<meta charset='utf-8'>", b"")
    decoded = detect.decode_html_detailed(bare)
    assert (decoded.encoding, decoded.replaced, decoded.guessed) == ("utf-8", 1, True)
    assert "Случайная величина" in decoded.text
    # A page with little non-ASCII text and one bad byte.
    english = b"<p>Hello \xe2\x80\x94 world \xff caf\xc3\xa9</p>"
    assert detect.decode_html(english) == ("<p>Hello — world \ufffd café</p>", "utf-8")
    # A declaration that breaks almost every character is still a lie.
    lying = "<p>Мера — счётно-аддитивная функция</p>".encode("cp1251")
    decoded = detect.decode_html_detailed(lying, "utf-8")
    assert (decoded.encoding, decoded.text) == ("cp1251", lying.decode("cp1251"))
    # U+FFFD that is really in the page is not counted as damage.
    assert detect.decode_html_detailed("a \ufffd b".encode()).replaced == 0


def test_wide_text_codec() -> None:
    text = "Определение 1. Случайная величина — функция.\n"
    assert detect.wide_text_codec(text.encode("utf-16")) == "utf-16"
    assert detect.wide_text_codec(("﻿" + text).encode("utf-16-be")) == "utf-16"  # BE BOM
    assert detect.wide_text_codec(text.encode("utf-32")) == "utf-32"
    assert detect.wide_text_codec(text.encode("utf-16-le")) == "utf-16-le"
    assert detect.wide_text_codec(text.encode("utf-16-be")) == "utf-16-be"
    assert detect.wide_text_codec(text.encode("utf-8")) is None
    assert detect.wide_text_codec(text.encode("cp1251")) is None
    assert detect.wide_text_codec(b"\x00\x01\x02\x03" * 10) == ""  # binary
    assert detect.wide_text_codec(b"PK\x03\x04\x14\x00\x00\x00") == ""


def test_sniff_utf16_text_without_extension(tmp_path: Path) -> None:
    path = tmp_path / "Л3. Теория меры"
    path.write_bytes("# Лекция 3\r\n\r\nМера.\r\n".encode("utf-16"))
    assert detect_kind(str(path)) == "md"


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("https://ru.wikipedia.org/wiki/Мера", None),
        ("http://[::1]:8000/x", None),
        ("http://example.com:abc/", "неверный номер порта"),
        ("http://example.com:99999/", "неверный номер порта"),
        ("http://[abc", "неверно записан IPv6-адрес"),
        ("https://", "в ссылке нет адреса сайта"),
        ("http://exa mple.com/", "в адресе сайта есть пробел"),
    ],
)
def test_url_problem(url: str, problem: str | None) -> None:
    assert detect.url_problem(url) == problem


def test_profile_of_damaged_pdfs(tmp_path: Path) -> None:
    full = make_pdf(tmp_path / "full.pdf", pages=20, chars=1500, seed="t")
    intact = detect.profile_pdf(full)
    assert (intact.repaired, intact.truncated, intact.damage_note) == (False, False, None)

    cut = detect.profile_pdf(truncate_file(full, tmp_path / "cut.pdf", 0.5))
    assert cut.repaired and cut.truncated and cut.pages == 20
    assert cut.text_layer < 0.8  # the tail of the file is lost
    _, quality = detect.pdf_signals(cut, "pdf-text")
    assert quality["repaired"] is True and quality["truncated"] is True
    assert quality["notes"][0] == cut.damage_note and "недокачан" in cut.damage_note
    assert any(n.startswith("Текст есть только на") for n in quality["notes"])
    assert not any("агент" in n for n in quality["notes"])

    broken = detect.profile_pdf(break_xref(make_pdf(tmp_path / "x.pdf"), tmp_path / "b.pdf"))
    assert broken.repaired and not broken.truncated
    assert broken.damage_note is not None and "повреждён" in broken.damage_note
    info = detect.inspect_file(tmp_path / "b.pdf", "pdf")
    assert info.warnings == [broken.damage_note]
    assert detect.inspect_file(full, "pdf").warnings == []


def test_truncated_pdf_without_pages_says_so(tmp_path: Path) -> None:
    # With object streams (pdfTeX, xdvipdfmx) the page tree sits near the end of the file:
    # a cut download has no pages at all.
    import pymupdf

    src = make_pdf(tmp_path / "src.pdf", pages=5, seed="z")
    with pymupdf.open(src) as doc:
        doc.save(tmp_path / "objstm.pdf", use_objstms=1)
    cut = truncate_file(tmp_path / "objstm.pdf", tmp_path / "cut.pdf", 0.5)
    with pytest.raises(detect.InspectError, match="обрезан \\(похоже, недокачан\\)"):
        detect.profile_pdf(cut)
