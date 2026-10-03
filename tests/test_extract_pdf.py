"""PDF extractor (h0lon/extract/pdf.py) and page analysis (h0lon/extract/pages.py).

Fixtures are synthetic: `tests/fixtures/m1/pdf/latex_math.pdf` is compiled by XeLaTeX from
the .tex next to it (real CM math fonts; rebuilt by a needs_xelatex test), the rest is drawn
with PyMuPDF inside the tests. No agents: vision goes through a fake `run_task`.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from h0lon.agents import Usage
from h0lon.agents.runner import RunResult
from h0lon.extract import pages as pg
from h0lon.extract import pdf as pdfx
from h0lon.extract import vision
from h0lon.extract.model import ExtractContext
from h0lon.sources.models import SourceRecord

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "m1" / "pdf"
LATEX_PDF = FIXTURES / "latex_math.pdf"
HAIR = " "
PROSE = (
    "Этот абзац придуман для проверки извлечения и не взят ни из какого курса. Речь пойдёт о "
    "линейных системах: сколько операций стоит их решение, насколько точен полученный ответ и "
    "как он меняется, если исходные числа слегка искажены. Почти всё держится на разложении "
    "матрицы в произведение более простых множителей. Разложение достаточно построить один "
    "раз, а потом применять его к разным правым частям. "
)


# ---------------------------------------------------------------- synthetic PDFs


def _noise(w: int, h: int) -> pymupdf.Pixmap:
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, os.urandom(w * h * 3), False)


def _html(page: pymupdf.Page, rect: tuple[float, ...], text: str, size: float = 11) -> None:
    page.insert_htmlbox(pymupdf.Rect(*rect), f'<p style="font-size:{size}pt">{text}</p>')


def text_page(doc: pymupdf.Document, text: str = PROSE * 2) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    _html(page, (56, 90, 540, 760), text)
    return page


def scan_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=_noise(120, 170))
    return page


def vector_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    shape = page.new_shape()
    for i in range(40):
        shape.draw_line((80 + i * 10, 150), (480 - i * 5, 650))
    shape.draw_bezier((80, 650), (200, 100), (350, 900), (500, 200))
    shape.finish(color=(0, 0, 1), width=0.8)
    shape.commit()
    _html(page, (80, 680, 520, 720), "Рисунок 1. График функции распределения")
    return page


def katex_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    _html(
        page,
        (56, 90, 540, 400),
        "Пусть μ — мера на алгебре. Если A = ⋃ Aₙ, то μA = lim μAₙ ⟹ для любого ε > 0 "
        "найдётся n, ∀ x ∈ ℝ выполнено неравенство.",
    )
    return page


def background_page(
    doc: pymupdf.Document,
    picture: pymupdf.Pixmap | None = None,
    text: str = "Непараметрические и метрические методы. Курс лекций.",
) -> pymupdf.Page:
    page = doc.new_page(width=454, height=255)
    page.insert_image(page.rect, pixmap=picture or _noise(90, 50))
    _html(page, (60, 80, 400, 200), text, 16)
    return page


def stamped_scan_page(doc: pymupdf.Document) -> pymupdf.Page:
    """A phone scan: a unique full-page picture and a visible stamp of the scanner app."""
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=_noise(300, 420))
    _html(page, (180, 805, 560, 830), "Отсканировано с помощью CamScanner", 9)
    return page


def small_pictures_page(doc: pymupdf.Document, n: int) -> pymupdf.Page:
    """Prose and `n` formula-sized pictures (160×40 pt, 1.3 % of the page each)."""
    page = doc.new_page(width=595, height=842)
    _html(page, (56, 60, 540, 200), PROSE)
    for i in range(n):
        page.insert_image(pymupdf.Rect(150, 220 + i * 70, 310, 260 + i * 70), pixmap=_noise(64, 16))
    _html(page, (56, 680, 540, 780), PROSE[:250])
    return page


def ocr_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=_noise(120, 170))
    page.insert_text((72, 100), "Scanned page with an invisible OCR layer", render_mode=3)
    return page


def picture_page(doc: pymupdf.Document) -> pymupdf.Page:
    """Plenty of text and a picture of ~25 % of the page: stays a text page."""
    page = doc.new_page(width=595, height=842)
    _html(page, (56, 60, 540, 330), PROSE)
    page.insert_image(pymupdf.Rect(100, 360, 495, 700), pixmap=_noise(200, 170))
    _html(page, (56, 720, 540, 800), PROSE[:200])
    return page


def _save(doc: pymupdf.Document, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _profiles(path: Path, *, slides: bool = False) -> list[pg.PageProfile]:
    with pg.opened(path) as doc:
        return pg.profile_document(doc, slides=slides)


def _ctx(
    tmp_path: Path, settings: Any, pdf: Path, *, kind: str = "pdf-text", **kw: Any
) -> ExtractContext:
    topic = tmp_path / "topic"
    (topic / "sources").mkdir(parents=True, exist_ok=True)
    (topic / "runs").mkdir(exist_ok=True)
    shutil.copyfile(pdf, topic / "sources" / "P1_src.pdf")
    rec = SourceRecord(
        id="P1",
        kind=kind,
        title="src",
        file="sources/P1_src.pdf",
        added=datetime.now(UTC).isoformat(),
    )
    out = topic / "extracted" / "P1"
    out.mkdir(parents=True, exist_ok=True)
    return ExtractContext(topic_dir=topic, source=rec, settings=settings, out_dir=out, **kw)


class FakeAgent:
    def __init__(self, write=None, ok: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.write = write or (lambda n: f"Распознано: $x_{n}$")
        self.ok = ok

    def __call__(
        self, bundle, *, settings, tier="strong", backend=None, fallback=True, on_event=None
    ):
        names = [f.path for f in bundle.contract.files]
        self.calls.append(
            {
                "files": names,
                "task": bundle.task_path.read_text(encoding="utf-8"),
                "hints": {
                    p.name: p.read_text(encoding="utf-8")
                    for p in bundle.inputs_dir.iterdir()
                    if p.suffix == ".txt"
                },
            }
        )
        for name in names:
            text = self.write(int(name[1:5]))
            if text is not None:
                (bundle.out_dir / name).write_text(text, encoding="utf-8")
        return RunResult(
            ok=self.ok,
            bundle=bundle,
            backend_used="claude",
            attempts=[],
            usage_total=Usage(),
            final_text="",
            problems=[] if self.ok else ["агент не справился"],
        )


# ---------------------------------------------------------------- classification


def test_synthetic_pages_are_classified(tmp_path: Path) -> None:
    doc = pymupdf.open()
    text_page(doc)
    scan_page(doc)
    vector_page(doc)
    katex_page(doc)
    ocr_page(doc)
    picture_page(doc)
    doc.new_page()  # blank
    profiles = _profiles(_save(doc, tmp_path / "mix.pdf"))
    got = [(p.kind, p.reason) for p in profiles]
    assert got == [
        ("text", ""),
        ("scan", "scan"),
        ("graphic", "graphic"),
        ("math", "math"),
        ("scan", "scan"),
        ("text", ""),
        ("text", "empty"),
    ]
    assert profiles[1].scan_dpi and profiles[1].scan_dpi < 30  # 120 px over 8.3 in
    assert profiles[3].math_glyphs >= pg.MATH_GLYPH_MIN and profiles[3].math_font_chars == 0
    assert profiles[4].invisible_text


def test_slide_background_image_is_not_a_graphic(tmp_path: Path) -> None:
    """A template background repeats (the Beamer title and final slides share one JPEG)."""
    doc = pymupdf.open()
    picture = _noise(90, 50)
    background_page(doc, picture)
    background_page(doc, picture, "Спасибо за внимание! Вопросы по курсу лекций.")
    profiles = _profiles(_save(doc, tmp_path / "bg.pdf"), slides=True)
    for p in profiles:
        assert p.full_image_repeated and p.background_image
        assert p.image_frac == 0.0 and p.image_frac_all > 0.9
        assert (p.kind, p.reason) == ("text", "")


def test_unique_full_slide_picture_with_caption_is_graphic(tmp_path: Path) -> None:
    """A screenshot slide with a caption: the picture is the content (was lost as text)."""
    doc = pymupdf.open()
    background_page(doc, text="Пример работы алгоритма kNN на данных Iris")
    (p,) = _profiles(_save(doc, tmp_path / "shot.pdf"), slides=True)
    assert p.full_image > 0.9 and not p.background_image
    assert (p.kind, p.reason) == ("graphic", "graphic")


def test_scan_with_visible_stamp_is_a_scan(tmp_path: Path, settings: Any, monkeypatch) -> None:
    doc = pymupdf.open()
    stamped_scan_page(doc)
    (p,) = _profiles(_save(doc, tmp_path / "stamp.pdf"))
    assert p.chars >= pg.SCAN_MAX_CHARS and not p.background_image
    assert (p.kind, p.reason) == ("scan", "scan")
    agent = FakeAgent(write=lambda n: "Текст скана страницы.")
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings, tmp_path / "stamp.pdf", kind="pdf-scan")
    out = pdfx.PdfExtractor().extract(ctx)
    assert "Страницы — сканы или фотографии" in agent.calls[0]["task"]
    assert "Текст скана страницы." in out.body_md.read_text(encoding="utf-8")
    assert out.quality["pages_scan"] == 1 and out.quality["pages_text"] == 0


def test_full_page_picture_under_much_text_is_a_background(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=_noise(60, 85))
    _html(page, (56, 90, 540, 760), PROSE)
    (p,) = _profiles(_save(doc, tmp_path / "paper.pdf"))
    assert p.visible_chars >= pg.BACKGROUND_MIN_CHARS and p.background_image
    assert (p.kind, p.reason) == ("text", "")


def test_short_text_pages_are_not_empty(tmp_path: Path, settings: Any) -> None:
    """«Глава 2. Графы» on a separator page is text, not «пустая страница»."""
    doc = pymupdf.open()
    for title in ("Глава 2. Графы", "Литература"):
        page = doc.new_page(width=595, height=842)
        _html(page, (56, 300, 540, 400), title, 28)
    doc.new_page(width=595, height=842)  # blank
    text_page(doc)
    pdf = _save(doc, tmp_path / "short.pdf")
    profiles = _profiles(pdf)
    assert [(p.kind, p.reason) for p in profiles] == [
        ("text", ""),
        ("text", ""),
        ("text", "empty"),
        ("text", ""),
    ]
    ctx = _ctx(tmp_path, settings, pdf, use_vision=False)
    out = pdfx.PdfExtractor().extract(ctx)
    pages = out.body_md.read_text(encoding="utf-8").split("## [[P1:")[1:]
    assert "Глава 2. Графы" in pages[0] and "Литература" in pages[1]
    assert vision.EMPTY_PAGE in pages[2] and vision.EMPTY_PAGE not in pages[0] + pages[1]
    assert out.quality["text_layer"] == 0.75  # only the blank page has no text


def test_small_pictures_go_to_the_agent_or_into_notes(tmp_path: Path, settings: Any) -> None:
    """Formulas pasted as pictures are not dropped silently."""
    doc = pymupdf.open()
    small_pictures_page(doc, 6)
    small_pictures_page(doc, 1)
    pdf = _save(doc, tmp_path / "small.pdf")
    many, one = _profiles(pdf)
    assert many.small_pictures == 6 and (many.kind, many.reason) == ("graphic", "graphic")
    assert one.small_pictures == 1 and one.kind == "text"
    out = pdfx.PdfExtractor().extract(_ctx(tmp_path, settings, pdf, use_vision=False))
    assert out.quality["pages_graphic"] == 1
    assert any("Мелкие картинки" in n and "стр. 2" in n for n in out.quality["notes"])


def test_repeated_small_pictures_are_icons(tmp_path: Path) -> None:
    doc = pymupdf.open()
    icon = _noise(40, 40)
    for _ in range(3):
        page = text_page(doc)
        for i in range(3):
            page.insert_image(pymupdf.Rect(60, 300 + i * 30, 80, 320 + i * 30), pixmap=icon)
    profiles = _profiles(_save(doc, tmp_path / "icons.pdf"))
    assert [(p.small_pictures, p.kind) for p in profiles] == [(0, "text")] * 3


def test_latex_fixture_math_fonts() -> None:
    profiles = _profiles(LATEX_PDF)
    assert [p.kind for p in profiles] == ["text", "math", "text"]
    assert any(f.startswith(("CMMI", "CMSY", "CMEX")) for f in profiles[1].math_fonts())
    assert profiles[1].math_font_chars >= 30
    # a single italic letter on page 3 does not make it a formula page
    assert 0 < profiles[2].math_font_chars < pg.MATH_FONT_MIN_CHARS


@pytest.mark.needs_xelatex
def test_latex_fixture_rebuilds_with_same_classes(tmp_path: Path) -> None:
    from h0lon import tools
    from h0lon.render.latex import compile_latex

    xelatex = tools.find_xelatex()
    if xelatex is None:
        pytest.skip("XeLaTeX не найден")
    tex = tmp_path / "latex_math.tex"
    shutil.copyfile(FIXTURES / "latex_math.tex", tex)
    res = compile_latex(tex, xelatex=xelatex, passes=1)
    assert res.ok, res.errors
    assert [p.kind for p in _profiles(res.pdf)] == ["text", "math", "text"]


def _profile(**kw: Any) -> pg.PageProfile:
    base = {"number": 1, "width": 595.0, "height": 842.0, "chars": 800, "visible_chars": 800}
    return pg.PageProfile(**{**base, **kw})


@pytest.mark.parametrize(
    ("kw", "slides", "expected"),
    [
        ({"chars": 5, "vector_items": 5000}, False, ("scan", "scan")),  # handwriting as paths
        ({"chars": 0}, False, ("text", "empty")),
        ({"chars": 12, "visible_chars": 12}, False, ("text", "")),  # «Глава 2. Графы»
        ({"chars": 12, "small_pictures": 3}, False, ("graphic", "graphic")),
        ({"chars": 12, "pictures": 1}, True, ("graphic", "graphic")),
        # a unique full-page picture under a short visible text: a scan / a screenshot
        (
            {"chars": 31, "visible_chars": 31, "full_image": 1.0, "image_frac_all": 1.0},
            False,
            ("scan", "scan"),
        ),
        (
            {"chars": 31, "visible_chars": 31, "full_image": 1.0, "image_frac_all": 1.0},
            True,
            ("graphic", "graphic"),
        ),
        (
            {"chars": 31, "visible_chars": 31, "full_image": 1.0, "full_image_repeated": True},
            True,
            ("text", ""),
        ),
        ({"chars": 250, "visible_chars": 250, "full_image": 1.0}, False, ("text", "")),
        ({"small_pictures": 2}, False, ("text", "")),
        ({"small_pictures": 3}, False, ("graphic", "graphic")),
        ({"pictures": 1}, False, ("text", "")),  # documents keep pictures as figures
        ({"pictures": 1, "chars": 150}, True, ("graphic", "graphic")),
        ({"garbled": 400}, False, ("scan", "low-text")),
        ({"math_font_chars": 2}, False, ("text", "")),
        ({"math_font_chars": 3}, False, ("math", "math")),
        ({"math_glyphs": 7}, False, ("text", "")),
        ({"math_glyphs": 8}, False, ("math", "math")),
        ({"visual_frac": 0.3, "chars": 200}, False, ("text", "")),  # a cover picture
        ({"visual_frac": 0.4, "chars": 200}, False, ("graphic", "graphic")),
        ({"visual_frac": 0.7, "chars": 2000}, False, ("graphic", "graphic")),
        ({"visual_frac": 0.12, "chars": 150}, True, ("graphic", "graphic")),
        ({"visual_frac": 0.08, "chars": 150}, True, ("text", "")),
        ({"ocr_font": True, "image_frac_all": 1.0}, False, ("scan", "scan")),
    ],
)
def test_classify_thresholds(kw: dict[str, Any], slides: bool, expected: tuple[str, str]) -> None:
    assert pg.classify_page(_profile(**kw), slides=slides) == expected


def test_math_font_names() -> None:
    for name in ("CMMI12", "CMSY10", "CMEX10", "MSBM10", "EURM10", "TeX-mathx10", "CambriaMath"):
        assert pg.MATH_FONT_RE.search(name), name
    for name in ("SegoeUISymbol", "XCharter-Roman", "SFSS1095", "CMR12", "Helvetica"):
        assert not pg.MATH_FONT_RE.search(name), name


# ---------------------------------------------------------------- running lines, spacing


def _paged(doc: pymupdf.Document, n: int, *, header: str, footer: str) -> None:
    for i in range(1, n + 1):
        page = doc.new_page(width=595, height=842)
        _html(page, (40, 18, 555, 40), header, 8)
        _html(page, (56, 100, 540, 400), f"Раздел {i}. " + PROSE)
        _html(page, (56, 420, 540, 450), "Повтор в теле страницы")
        _html(page, (400, 805, 555, 825), footer.format(i=i, n=n), 7)


def test_running_lines_are_found(tmp_path: Path) -> None:
    doc = pymupdf.open()
    _paged(doc, 5, header="МАТАНАЛИЗ · II КУРС · I СЕМЕСТР", footer="СТР. {i} ИЗ {n}")
    profiles = _profiles(_save(doc, tmp_path / "run.pdf"))
    running = pg.find_running_lines(profiles)
    assert running.texts == ["МАТАНАЛИЗ · II КУРС · I СЕМЕСТР"]
    for p in profiles:
        dropped = [line.shown for line in running.drop[p.number]]
        assert dropped == ["МАТАНАЛИЗ · II КУРС · I СЕМЕСТР", f"СТР. {p.number} ИЗ 5"]
        assert "Повтор в теле страницы" in pg.page_text(p, running.drop[p.number])


def test_short_document_drops_only_page_numbers(tmp_path: Path) -> None:
    doc = pymupdf.open()
    _paged(doc, 2, header="Курс лекций", footer="{i}/{n}")
    profiles = _profiles(_save(doc, tmp_path / "two.pdf"))
    running = pg.find_running_lines(profiles)
    assert running.texts == []
    assert [line.shown for line in running.drop[1]] == ["1/2"]


def test_alternating_book_headers(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for i in range(1, 9):
        page = doc.new_page(width=595, height=842)
        _html(page, (40, 18, 555, 40), "Глава 2. Меры" if i % 2 else "Учебник анализа", 8)
        _html(page, (56, 100, 540, 400), PROSE)
    running = pg.find_running_lines(_profiles(_save(doc, tmp_path / "book.pdf")))
    assert sorted(running.texts) == ["Глава 2. Меры", "Учебник анализа"]


@pytest.mark.parametrize(
    "text",
    ["4/31", "СТР. 3 ИЗ 25", "— 12 —", "— 1234 —", "Страница 7", "page 2 of 9", "xiv", "15"],
)
def test_page_number_patterns(text: str) -> None:
    assert pg.is_page_number(text)


@pytest.mark.parametrize("text", ["Глава 3", "2026 год", "2026", "Теорема 1.2", "x = 4/3 + y"])
def test_not_page_numbers(text: str) -> None:
    assert not pg.is_page_number(text)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("СТР. 3 ИЗ 25", "СТР. 4 ИЗ 25", True),
        ("Лекция 3 · стр. 5", "Лекция 3 · стр. 6", True),
        ("Глава 2 — 15", "Глава 2 — 16", True),
        ("12 | Теория меры", "14 | Теория меры", True),
        ("Задача 1", "Задача 2", False),
        ("Пример 3", "Пример 4", False),
        ("ИТМО · 2026", "ИТМО · 2026", True),
    ],
)
def test_running_key_masks_only_page_numbers(a: str, b: str, same: bool) -> None:
    assert (pg.running_key(a) == pg.running_key(b)) is same


def test_headings_in_the_top_band_are_not_running_lines(tmp_path: Path, settings: Any) -> None:
    """«Задача 1»…«Задача 6» at the top of each page are content, not a header."""
    doc = pymupdf.open()
    for i in range(1, 7):
        page = doc.new_page(width=595, height=842)
        _html(page, (56, 14, 540, 50), f"Задача {i}", 14)
        _html(page, (56, 120, 540, 600), PROSE + f" Уникальный текст {i}.")
    pdf = _save(doc, tmp_path / "tasks.pdf")
    running = pg.find_running_lines(_profiles(pdf))
    assert running.texts == [] and not running.drop
    body = pdfx.PdfExtractor().extract(_ctx(tmp_path, settings, pdf, use_vision=False))
    text = body.body_md.read_text(encoding="utf-8")
    assert all(f"Задача {i}" in text for i in range(1, 7))


def test_year_on_a_title_page_is_kept(tmp_path: Path, settings: Any) -> None:
    doc = pymupdf.open()
    page = text_page(doc, PROSE)
    page.insert_text((250, 795), "Saint Petersburg", fontsize=12)
    page.insert_text((280, 815), "2026", fontsize=12)
    text_page(doc, PROSE)
    pdf = _save(doc, tmp_path / "title.pdf")
    body = pdfx.PdfExtractor().extract(_ctx(tmp_path, settings, pdf, use_vision=False))
    assert "2026" in body.body_md.read_text(encoding="utf-8").split("[[P1:p2]]")[0]


def test_three_page_excerpt_header_without_title_page(tmp_path: Path) -> None:
    """A 3-page excerpt of the notes: the title page has no header, pages 2–3 do."""
    doc = pymupdf.open()
    for i in range(3):
        page = text_page(doc, PROSE)
        if i:
            _html(page, (40, 18, 555, 40), "Конспект · Теория меры", 8)
            _html(page, (40, 50, 555, 66), f"Задача {i}", 8)  # also in the band, distinct
    running = pg.find_running_lines(_profiles(_save(doc, tmp_path / "three.pdf")))
    assert running.texts == ["Конспект · Теория меры"]
    assert [line.shown for line in running.drop[2]] == ["Конспект · Теория меры"]


def _chars(spec: list[tuple[str, float]], width: float = 6.0) -> list[dict[str, Any]]:
    """Glyph dicts: each (char, gap before it) — spaces get their own small boxes."""
    out, x = [], 0.0
    for ch, gap in spec:
        x += gap
        out.append({"c": ch, "bbox": (x, 0.0, x + width, 10.0)})
        x += width
    return out


def _spaced(word_gaps: list[str], tracking: float, word_gap: float) -> list[dict[str, Any]]:
    spec: list[tuple[str, float]] = []
    for w, word in enumerate(word_gaps):
        for i, ch in enumerate(word):
            gap = 0.0 if not spec else (word_gap if i == 0 and w else tracking)
            if spec:
                spec.append((" ", 0.0))
                gap = max(0.0, gap - 6.0)
            spec.append((ch, gap))
    return _chars(spec)


def test_despace_two_words() -> None:
    chars = _spaced(["АВТОР", "КОНСПЕКТА"], tracking=2.5, word_gap=14.0)
    assert pg.despace_chars(chars, 10.0) == "АВТОР КОНСПЕКТА"


def test_despace_single_word_needs_confirmation() -> None:
    """«Л Е К Т О Р» and «А Б В Г Д» look the same: the document's tracking decides."""
    chars = _spaced(["ЛЕКТОР"], 2.5, 0)  # glyph gaps of 0.6 em at 10 pt
    spaced = pg.analyze_spacing(chars, 10.0)
    assert spaced is not None and spaced.single_word and spaced.gap == pytest.approx(0.6)
    assert pg.despace_chars(chars, 10.0) is None
    assert pg.despace_chars(chars, 10.0, trackings=[0.62]) == "ЛЕКТОР"
    assert pg.despace_chars(chars, 10.0, trackings=[0.4]) is None
    tight: list[dict[str, Any]] = []
    x = 0.0
    for i, ch in enumerate("ЛЕКТОР"):  # glyphs 6 pt wide, «spaces» of 1 pt between them
        if i:
            tight.append({"c": " ", "bbox": (x, 0.0, x + 1.0, 10.0)})
            x += 1.0
        tight.append({"c": ch, "bbox": (x, 0.0, x + 6.0, 10.0)})
        x += 6.0
    assert pg.despace_chars(tight, 10.0) == "ЛЕКТОР"  # 0.1 em is narrower than any space


def test_resolve_spacing_uses_two_word_lines_of_the_document() -> None:
    two = pg.analyze_spacing(_spaced(["АВТОР", "КОНСПЕКТА"], 2.5, 14.0), 10.0)
    one = pg.analyze_spacing(_spaced(["ЛЕКТОР"], 2.5, 0), 10.0)
    assert two is not None and not two.single_word and one is not None
    line_two = pg.LineInfo(0, "А В Т О Р", (0, 0, 1, 1), 10, "f", False, fixed=two.text)
    line_two.fix_kind, line_two.tracking = "despaced", two.gap
    line_one = pg.LineInfo(0, "Л Е К Т О Р", (0, 0, 1, 1), 10, "f", False)
    line_one.spaced_candidate = (one.text, one.gap)
    letters = pg.LineInfo(1, "А Б В Г Д", (0, 0, 1, 1), 10, "f", False)
    letters.spaced_candidate = ("АБВГД", 0.28)
    p1 = pg.PageProfile(number=1, width=595, height=842, lines=[line_two])
    p2 = pg.PageProfile(number=2, width=595, height=842, lines=[line_one, letters])
    pg.resolve_spacing([p1, p2])
    assert line_one.shown == "ЛЕКТОР" and line_one.fix_kind == "despaced"
    assert letters.shown == "А Б В Г Д" and letters.fix_kind is None


def test_single_letters_with_real_spaces_are_kept(tmp_path: Path, settings: Any) -> None:
    doc = pymupdf.open()
    page = text_page(doc, PROSE)
    for i, line in enumerate(["А Б В Г Д", "a b c d e f", "x y z t u v"]):
        _html(page, (60, 480 + 30 * i, 400, 505 + 30 * i), line, 12)
    pdf = _save(doc, tmp_path / "abc.pdf")
    (p,) = _profiles(pdf)
    candidates = [line for line in p.lines if line.spaced_candidate]
    assert len(candidates) == 3 and all(line.fixed is None for line in candidates)
    out = pdfx.PdfExtractor().extract(_ctx(tmp_path, settings, pdf, use_vision=False))
    body = out.body_md.read_text(encoding="utf-8")
    assert "А Б В Г Д" in body and "a b c d e f" in body and "x y z t u v" in body
    assert "АБВГД" not in body and "abcdef" not in body and "xyztuv" not in body
    assert not any("разреженный" in n for n in out.quality["notes"])


@pytest.mark.parametrize(
    "tokens",
    [
        ["0", "<", "ε", "<", "1"],
        ["L", "=", "I", "+"],
        ["i", ">", "j", "и", "j", "<", "n"],
        ["a", "b"],
    ],
)
def test_despace_leaves_formulas_alone(tokens: list[str]) -> None:
    spec: list[tuple[str, float]] = []
    for t in tokens:
        if spec:
            spec.append((" ", 0.0))
        spec.append((t, 2.0))
    assert pg.despace_chars(_chars(spec), 10.0) is None


def test_respace_restores_lost_word_spaces() -> None:
    spec: list[tuple[str, float]] = []
    for w, word in enumerate(["Речь", "пойдёт", "о", "линейных", "системах"]):
        for i, ch in enumerate(word):
            spec.append((ch, 3.0 if (i == 0 and w) else 0.0))
    chars = _chars(spec)
    assert pg.respace_chars(chars, 12.0) == "Речь пойдёт о линейных системах"
    tight = _chars([(c, 0.0) for c in "Длинноесловобезпробеловвообще"])
    assert pg.respace_chars(tight, 12.0) is None  # no gaps: nothing to restore
    short = _chars([("а", 0.0), ("б", 3.0), ("в", 3.0)])
    assert pg.respace_chars(short, 12.0) is None  # short tokens: not a glued line


def test_decomposed_letters_are_composed() -> None:
    line = pg.LineInfo(index=0, text="займёмся й", bbox=(0, 0, 1, 1), size=1, font="f", bold=False)
    assert line.shown == "займёмся й"
    assert pg.cleanup_markdown("Мой текст") == "Мой текст"


def test_despace_ambiguous_gaps_are_left() -> None:
    spec = [("М", 0.0)]
    for ch, gap in zip("АТЕМАТИКА", [2, 3.5, 5, 6.5, 2, 4, 5.5, 3, 7], strict=True):
        spec += [(" ", 0.0), (ch, gap)]
    assert pg.despace_chars(_chars(spec), 10.0) is None


# ---------------------------------------------------------------- rendering and Markdown


@pytest.mark.parametrize(
    ("size", "long_side"),
    [((595, 842), 2000), ((454, 255), 1400), ((612, 792), 2000), ((200, 150), 833)],
)
def test_render_sizes(tmp_path: Path, size: tuple[int, int], long_side: int) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    png = pg.render_page_png(page, tmp_path / "p.png")
    pix = pymupdf.Pixmap(str(png))
    assert abs(max(pix.width, pix.height) - long_side) <= 2
    assert max(pix.width, pix.height) <= pg.RENDER_MAX_SIDE


def test_cleanup_markdown() -> None:
    BS = "\\"
    raw = (
        "# **Заголовок**\n\n## Раздел\n\n"
        "Текст<sup>2</sup> и H<sub>2</sub>O, <mark>важно</mark>.\n\n"
        "<!-- Start of picture text -->\nподпись<br>ось x<br>\n<!-- End of picture text -->\n\n"
        "**==> picture [10 x 10] intentionally omitted <==**\n\n"
        f"сноска {BS}mu A_1 и М А Т Е М А Т И К А\n\n```\n# код {BS}n\n```\n"
    )
    out = pg.cleanup_markdown(raw, fixes=[("М А Т Е М А Т И К А", "МАТЕМАТИКА")])
    lines = out.splitlines()
    assert lines[0] == "### Заголовок" and "#### Раздел" in lines
    assert "Текст^2^ и H~2~O, важно." in out
    assert "подпись" + BS + "\nось x" in out
    assert "intentionally omitted" not in out
    assert f"сноска {BS}{BS}mu A_1 и МАТЕМАТИКА" in out
    assert f"# код {BS}n" in out  # code is untouched


def test_cleanup_markdown_keeps_backslashes_in_code_spans() -> None:
    BS = "\\"
    raw = f"Пакет `{BS}usepackage{{amsmath}}`, путь ``C:{BS}new{BS}table.tex`` и {BS}mu вне кода"
    out = pg.cleanup_markdown(raw)
    assert out == (
        f"Пакет `{BS}usepackage{{amsmath}}`, путь ``C:{BS}new{BS}table.tex`` и {BS}{BS}mu вне кода"
    )


def test_escape_and_uncertain_block() -> None:
    assert pg.escape_markdown("1) мера\n2. шаг\n- дефис\n::: блок") == (
        "1\\) мера\n2\\. шаг\n\\- дефис\n\\::: блок"
    )
    assert pg.escape_inline("a*b [c] $d$") == "a\\*b \\[c\\] \\$d\\$"
    block = pg.uncertain_block('T "x"', "строка один\nстрока два\n\nвторой абзац", intro="Вступ:")
    assert block.startswith("::: {.uncertain title=\"T 'x'\"}\nВступ:\n\n")
    assert "строка один\\\nстрока два" in block and block.endswith(":::")
    assert "Текстового слоя нет." in pg.uncertain_block("T", "  ", intro="Вступ:")


@pytest.mark.parametrize(
    ("producer", "expected"),
    [
        ("XeTeX 0.999996", "Nas budut interesovat rezultaty"),  # TeX hyphenates by syllables
        ("Skia/PDF m126", "Nas budut in-teresovat rezultaty"),  # Chromium breaks at hyphens
    ],
)
def test_page_text_dehyphenates_by_producer(tmp_path: Path, producer: str, expected: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Nas budut in-")
    page.insert_text((72, 114), "teresovat rezultaty")
    doc.set_metadata({"producer": producer})
    (p,) = _profiles(_save(doc, tmp_path / "hy.pdf"))
    assert p.hyphenating == ("TeX" in producer)
    assert pg.page_text(p, reflow=True) == expected
    assert pg.page_text(p) == expected  # the two lines of one word become one line


def test_tex_fonts_mean_syllable_hyphenation(tmp_path: Path) -> None:
    """A TeX PDF re-saved by another tool loses its producer; its CM fonts remain."""
    doc = pymupdf.open(str(LATEX_PDF))
    doc.set_metadata({"producer": "pdf-lib", "creator": "pdf-lib"})
    resaved = _save(doc, tmp_path / "resaved.pdf")
    with pg.opened(resaved) as d:
        assert "TeX" not in (d.metadata.get("producer") or "") + (d.metadata.get("creator") or "")
    assert all(p.hyphenating for p in _profiles(resaved))


def test_hyphenation_evidence_of_the_document(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, text in enumerate(
        ["Nas budut in-", "teresovat rezultaty i fak-", "ta. Nas budut interesovat oni."]
    ):
        page.insert_text((72, 100 + 14 * i), text)
    doc.set_metadata({"producer": "Microsoft Word"})
    (p,) = _profiles(_save(doc, tmp_path / "evidence.pdf"))
    assert p.hyphenating  # «interesovat» confirms a syllable break, no compound is confirmed
    assert pg.page_text(p, reflow=True).startswith("Nas budut interesovat rezultaty i fakta.")


def test_join_hyphenated_uses_the_document() -> None:
    vocab = frozenset({"санкт-петербургский", "интересовать"})
    join = pg.join_hyphenated
    assert join("санкт-", "петербургский вуз", vocabulary=vocab) == "санкт-петербургский вуз"
    assert join("на ин-", "тересовать", vocabulary=vocab, hyphenating=False) == "на интересовать"
    assert join("кто-", "то пришёл") == "кто-то пришёл"  # a particle, even in TeX
    assert join("физико-", "математический", hyphenating=False) == "физико-математический"
    assert join("постро-", "им", hyphenating=True) == "построим"
    assert join("ин" + pg.SOFT_HYPHEN, "тересовать", hyphenating=False) == "интересовать"
    assert join("Глава 2 -", "раздел") is None  # no word before the hyphen
    assert join("слово-", "Заглавная") is None


def test_hyphenated_compounds_in_the_text_layer(tmp_path: Path) -> None:
    """A narrow column breaks «санкт-|петербургский»: the hyphen of the compound stays."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _html(
        page,
        (56, 90, 200, 400),
        "В физико-математический лицей поступают ученики; научно-технический прогресс и "
        "санкт-петербургский вуз требуют знаний. " * 3,
    )
    (p,) = _profiles(_save(doc, tmp_path / "compound.pdf"))
    text = pg.page_text(p, reflow=True)
    assert "санктпетербургский" not in text and "физикоматематический" not in text
    assert text.count("санкт-петербургский") == 3


# ---------------------------------------------------------------- extractor


def test_plan(tmp_path: Path, settings: Any) -> None:
    ctx = _ctx(tmp_path, settings, LATEX_PDF)
    plan = pdfx.PdfExtractor().plan(ctx)
    assert (plan.pages_total, plan.pages_vision, plan.agent_runs) == (3, 1, 1)
    assert "формулы: 1" in plan.notes[0]
    ctx.use_vision = False
    off = pdfx.PdfExtractor().plan(ctx)
    assert (off.pages_vision, off.agent_runs) == (0, 0)
    assert any("Агент отключён" in n for n in off.notes)


def test_extract_without_vision(tmp_path: Path, settings: Any) -> None:
    ctx = _ctx(tmp_path, settings, LATEX_PDF, use_vision=False)
    out = pdfx.PdfExtractor().extract(ctx)
    body = out.body_md.read_text(encoding="utf-8")
    assert out.body_md == ctx.out_dir / "body.md"
    assert body.startswith("## [[P1:p1]] Страница 1\n\n")
    assert "## [[P1:p2]] Страница 2" in body and "## [[P1:p3]] Страница 3" in body
    assert "Численные методы" not in body  # running header removed
    assert "### Введение" in body and "придуман для проверки извлечения" in body
    page2 = body.split("## [[P1:p2]]")[1].split("## [[P1:p3]]")[0]
    assert '::: {.uncertain title="не распознано: агент отключён"}' in page2
    assert "Нормы и пределы" in page2
    assert not list(ctx.out_dir.glob("pages/*.png"))  # renders are only for the agent
    q = out.quality
    assert (q["pages_text"], q["pages_math"], q["pages_vision"], q["pages_failed"]) == (2, 1, 0, 0)
    assert q["text_layer"] == 1.0 and q["chars_per_page"] > 200
    assert q["cyrillic_ratio"] > 0.8 and q["scan_dpi"] is None
    assert any("Агент отключён" in n for n in q["notes"])
    assert any("Удалены колонтитулы" in n for n in q["notes"])
    assert out.agent_runs == 0 and out.pages_vision == 0 and out.pages_total == 3
    assert out.warnings and "отключено" in out.warnings[0]


def test_extract_with_vision(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings, LATEX_PDF)
    out = pdfx.PdfExtractor().extract(ctx)
    body = out.body_md.read_text(encoding="utf-8")
    assert "## [[P1:p2]] Страница 2\n\nРаспознано: $x_2$" in body
    assert out.agent_runs == 1 and out.pages_vision == 1 and out.quality["pages_vision"] == 1
    (call,) = agent.calls
    assert call["files"] == ["p0002.md"]
    assert "Страницы документа" in call["task"]
    hint = call["hints"]["p0002.txt"]
    assert "Нормы и пределы" in hint and "Численные методы" not in hint
    # second run: the page comes from the cache
    out2 = pdfx.PdfExtractor().extract(ctx)
    assert out2.agent_runs == 0 and len(agent.calls) == 1
    assert pdfx.PdfExtractor().plan(ctx).notes[-1] == "Из кэша распознавания: 1 стр."


def test_extract_failed_page_gets_text_layer(tmp_path: Path, settings: Any, monkeypatch) -> None:
    monkeypatch.setattr(vision, "run_task", FakeAgent(write=lambda n: None, ok=False))
    ctx = _ctx(tmp_path, settings, LATEX_PDF)
    out = pdfx.PdfExtractor().extract(ctx)
    body = out.body_md.read_text(encoding="utf-8")
    assert f'::: {{.uncertain title="{pdfx.FAILED_TITLE}"}}' in body
    assert out.quality["pages_failed"] == 1 and out.quality["pages_vision"] == 0
    assert any("не распознаны агентом" in w for w in out.warnings)


def test_scan_source_uses_scan_flavor(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    doc = pymupdf.open()
    scan_page(doc)
    scan_page(doc)
    pdf = _save(doc, tmp_path / "scan.pdf")
    ctx = _ctx(tmp_path, settings, pdf, kind="pdf-scan")
    out = pdfx.PdfExtractor().extract(ctx)
    assert "Страницы — сканы или фотографии" in agent.calls[0]["task"]
    assert out.quality["pages_scan"] == 2 and out.quality["text_layer"] == 0.0
    assert out.quality["scan_dpi"] and out.quality["scan_dpi"] < 150
    assert any("низкого разрешения" in n for n in out.quality["notes"])


def test_text_pages_keep_pictures_and_glue_spacing(tmp_path: Path, settings: Any) -> None:
    doc = pymupdf.open()
    page = picture_page(doc)
    spaced = f"{HAIR}".join("МАТЕМАТИКА") + f" {HAIR}" + f"{HAIR}".join("ЛЕКЦИЯ")
    _html(page, (56, 30, 540, 50), spaced, 9)
    pdf = _save(doc, tmp_path / "pic.pdf")
    ctx = _ctx(tmp_path, settings, pdf, use_vision=False)
    out = pdfx.PdfExtractor().extract(ctx)
    body = out.body_md.read_text(encoding="utf-8")
    assert "![](figures/P1_p0001_1.png)" in body
    assert (ctx.out_dir / "figures" / "P1_p0001_1.png").is_file()
    assert "МАТЕМАТИКА ЛЕКЦИЯ" in body and "М А Т" not in body
    assert any("разреженный текст" in n for n in out.quality["notes"])


def test_figures_survive_a_temp_dir_with_spaces(tmp_path: Path, settings: Any, monkeypatch) -> None:
    """pymupdf4llm mangles image paths («Иван Петров» → «Иван_Петров»): nothing is written
    through it any more, pictures come embedded and are saved by the extractor."""
    import tempfile

    odd = tmp_path / "Иван Петров" / "Temp (dir) – x"
    odd.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(odd))
    doc = pymupdf.open()
    picture_page(doc)
    ctx = _ctx(tmp_path, settings, _save(doc, tmp_path / "pic.pdf"), use_vision=False)
    out = pdfx.PdfExtractor().extract(ctx)
    assert "![](figures/P1_p0001_1.png)" in out.body_md.read_text(encoding="utf-8")
    assert (ctx.out_dir / "figures" / "P1_p0001_1.png").is_file()
    assert not out.warnings and not list(odd.iterdir())


def test_repaired_pdf_is_reported(tmp_path: Path, settings: Any) -> None:
    doc = pymupdf.open()
    for _ in range(3):
        text_page(doc)
    good = _save(doc, tmp_path / "good.pdf")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    ctx = _ctx(tmp_path, settings, broken, use_vision=False)
    out = pdfx.PdfExtractor().extract(ctx)
    assert pdfx.REPAIRED_NOTE in out.quality["notes"]
    assert any(pdfx.REPAIRED_NOTE in w for w in out.warnings)


def test_running_lines_removed_from_text_pages(tmp_path: Path, settings: Any) -> None:
    doc = pymupdf.open()
    _paged(doc, 3, header="Курс · Тест", footer="СТР. {i} ИЗ {n}")
    ctx = _ctx(tmp_path, settings, _save(doc, tmp_path / "hf.pdf"), use_vision=False)
    body = pdfx.PdfExtractor().extract(ctx).body_md.read_text(encoding="utf-8")
    assert "Курс · Тест" not in body and "СТР." not in body
    assert body.count("Повтор в теле страницы") == 3


def test_missing_file_is_an_extract_error(tmp_path: Path, settings: Any) -> None:
    from h0lon.extract.registry import ExtractError

    ctx = _ctx(tmp_path, settings, LATEX_PDF)
    ctx.source.file = "sources/nope.pdf"
    with pytest.raises(ExtractError):
        pdfx.PdfExtractor().extract(ctx)
