"""render_document end to end, pdfcheck, front-matter helpers.

Full renders are marked needs_xelatex / needs_browser and skipped when the tool is missing.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from rich.console import Console

from h0lon import tools
from h0lon.config import Settings
from h0lon.render import check_pdf, render_document
from h0lon.render.pandoc import has_headings, plain_title, read_front_matter, title_fragments
from h0lon.render.pipeline import _prepare_metadata

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_master.md"
SAMPLE_TITLE = "Случайные величины и функции распределения"


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    """sample_master.md with its image, copied to a directory with spaces and Cyrillic."""
    d = tmp_path / "Курс теорвер" / "лекция 3"
    (d / "img").mkdir(parents=True)
    shutil.copyfile(SAMPLE, d / "sample_master.md")
    shutil.copyfile(FIXTURES / "img" / "sample.png", d / "img" / "sample.png")
    return d / "sample_master.md"


def _xelatex_or_skip() -> None:
    if tools.find_xelatex() is None:
        pytest.skip("XeLaTeX not found")
    if tools.find_pandoc() is None:
        pytest.skip("pandoc not found")


def _browser_or_skip() -> None:
    if tools.find_browser("auto") is None:
        pytest.skip("Edge/Chrome/Chromium not found")
    if tools.find_pandoc() is None:
        pytest.skip("pandoc not found")


def _pdf_text(pdf: Path) -> str:
    with pymupdf.open(str(pdf)) as doc:
        return "".join(page.get_text() for page in doc)


# ---------------------------------------------------------------- helpers


def test_read_front_matter() -> None:
    meta = read_front_matter(SAMPLE.read_text(encoding="utf-8"))
    assert meta["title"] == SAMPLE_TITLE
    assert meta["course"] == "Теория вероятностей"
    assert [s["id"] for s in meta["sources"]] == ["H1", "S1", "V1"]
    assert read_front_matter("# Нет шапки\n") == {}
    assert read_front_matter("---\n: [broken\n---\n") == {}
    assert read_front_matter("---\n- list\n---\n") == {}


def test_has_headings_ignores_code() -> None:
    assert has_headings("Текст\n\n## Раздел\n")
    assert not has_headings("Текст\n\n```python\n# comment\n```\n")
    assert not has_headings("#хэштег без пробела\n")


def test_plain_title() -> None:
    assert plain_title("О $\\sigma$-алгебре [[H1:p1]] *кратко*") == "О -алгебре кратко"
    assert plain_title(None) == ""


# ---------------------------------------------------------------- pdfcheck (synthetic PDFs)


def _make_pdf(path: Path, *, embed: bool, toc: bool, title: str | None) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    if embed:
        page.insert_font(fontname="F0", fontbuffer=pymupdf.Font("helv").buffer)
        page.insert_text((72, 72), "Sample Title", fontname="F0", fontsize=14)
    else:
        page.insert_text((72, 72), "Sample Title", fontname="helv", fontsize=14)
    if toc:
        doc.set_toc([[1, "Chapter", 1]])
    if title:
        doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()


def test_pdfcheck_good_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "good.pdf"
    _make_pdf(pdf, embed=True, toc=True, title="Sample Title")
    rep = check_pdf(pdf, expected_title="sample  title", expect_bookmarks=True)
    assert rep.ok, rep.errors
    assert rep.pages == 1
    assert rep.title_found is True
    assert rep.title_meta == "Sample Title"
    assert rep.bookmarks == 1
    assert rep.not_embedded == []
    assert rep.warnings == []
    assert json.dumps(rep.to_dict())


def test_pdfcheck_problems(tmp_path: Path) -> None:
    pdf = tmp_path / "bad.pdf"
    _make_pdf(pdf, embed=False, toc=False, title=None)
    rep = check_pdf(pdf, expected_title="Другой заголовок", expect_bookmarks=True)
    assert not rep.ok
    assert rep.not_embedded  # base-14 Helvetica is not embedded
    assert any("не встроены" in e for e in rep.errors)
    assert rep.title_found is False
    assert any("Заголовок" in w for w in rep.warnings)
    assert any("закладок" in w for w in rep.warnings)
    assert any("Title" in w for w in rep.warnings)


def test_pdfcheck_unreadable_file(tmp_path: Path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a pdf")
    rep = check_pdf(pdf)
    assert not rep.ok
    assert rep.errors


# ---------------------------------------------------------------- pipeline without TeX


def test_missing_source(settings: Settings, tmp_path: Path) -> None:
    rep = render_document(tmp_path / "nope.md", settings=settings)
    assert not rep.ok
    assert rep.errors and "не найден" in rep.errors[0]


def test_unknown_template(settings: Settings, sample: Path) -> None:
    rep = render_document(sample, settings=settings, template="no-such")
    assert not rep.ok
    assert "no-such" in rep.errors[0]
    assert not (sample.parent / "sample_master.build").exists()


def test_no_xelatex_without_fallback(
    make_settings: Callable[..., Settings], sample: Path, tmp_path: Path
) -> None:
    settings = make_settings(
        render={"xelatex": str(tmp_path / "missing" / "xelatex.exe"), "fallback_html": False}
    )
    out = tmp_path / "out"
    rep = render_document(sample, settings=settings, out_dir=out)
    assert not rep.ok
    assert rep.engine_used is None
    assert any("XeLaTeX не найден" in e for e in rep.errors)
    assert not (out / "sample_master.pdf").exists()
    assert not (out / "sample_master.build").exists()  # build dir removed by default
    assert rep.build_dir is None
    data = rep.to_dict()
    assert data["ok"] is False and data["pdf"] is None
    json.dumps(data)


def test_report_print_does_not_crash(
    make_settings: Callable[..., Settings], sample: Path, tmp_path: Path
) -> None:
    settings = make_settings(render={"xelatex": str(tmp_path / "x.exe"), "fallback_html": False})
    rep = render_document(sample, settings=settings, out_dir=tmp_path / "o", keep_build=True)
    console = Console(record=True, width=100)
    rep.print(console)
    text = console.export_text()
    assert "Сборка PDF не удалась" in text
    assert "XeLaTeX не найден" in text
    assert rep.build_dir is not None and rep.build_dir.is_dir()


# ---------------------------------------------------------------- XeLaTeX


@pytest.mark.needs_xelatex
def test_render_sample_xelatex(settings: Settings, sample: Path, tmp_path: Path) -> None:
    _xelatex_or_skip()
    out = tmp_path / "out"
    rep = render_document(sample, settings=settings, out_dir=out, engine="xelatex", keep_build=True)
    assert rep.ok, rep.errors
    assert rep.engine_used == "xelatex"
    assert rep.fallback_used is False
    assert rep.pdf == out / "sample_master.pdf"
    assert rep.pdf.is_file()
    assert 2 <= rep.passes <= settings.render.passes
    assert not any("Шрифт не содержит" in w for w in rep.warnings), rep.warnings

    c = rep.checks
    assert c is not None and c.ok, c.errors if c else None
    assert c.pages >= 4
    assert c.title_found is True
    assert c.title_meta == SAMPLE_TITLE
    assert c.not_embedded == []
    assert c.type3 == []
    assert c.bookmarks >= 8

    assert rep.build_dir is not None and rep.tex is not None and rep.tex.is_file()
    log = (rep.build_dir / f"{rep.tex.stem}.log").read_text(encoding="utf-8", errors="replace")
    assert "Missing character" not in log

    with pymupdf.open(str(rep.pdf)) as doc:
        toc = doc.get_toc()
        titles = [t[1] for t in toc]
        assert any("Функция распределения" in t for t in titles)
        assert any("Расхождения между источниками" in t for t in titles)
        assert len(doc[3].get_images()) + len(doc[2].get_images()) >= 1  # the figure
    text = _pdf_text(rep.pdf)
    for needle in (
        "Определение 1.1.",
        "Теорема 1.1.",
        "Доказательство",
        "Решение",
        "H1:p1",
        "Приложение А.",
        "Содержание",
        "Оценка математического ожидания",
    ):
        assert needle in text, needle
    assert "[[" not in text
    assert "<!--" not in text


@pytest.mark.needs_xelatex
def test_render_clean_xelatex_removes_build(
    settings: Settings, sample: Path, tmp_path: Path
) -> None:
    _xelatex_or_skip()
    rep = render_document(
        sample,
        settings=settings,
        clean=True,
        engine="xelatex",
        metadata={"subtitle": "Чистая версия"},
    )
    assert rep.ok, rep.errors
    assert rep.pdf == sample.with_suffix(".pdf")
    assert rep.build_dir is None and rep.tex is None
    assert not (sample.parent / "sample_master.build").exists()
    text = _pdf_text(rep.pdf)
    assert "H1:p1" not in text
    assert "V1:12:34" not in text
    assert "Чистая версия" in text


# ---------------------------------------------------------------- HTML → browser


@pytest.mark.needs_browser
def test_render_sample_html(settings: Settings, sample: Path, tmp_path: Path) -> None:
    _browser_or_skip()
    rep = render_document(sample, settings=settings, out_dir=tmp_path / "html", engine="html")
    assert rep.ok, rep.errors
    assert rep.engine_used == "html"
    assert rep.passes == 0
    c = rep.checks
    assert c is not None and c.ok, c.errors if c else None
    assert c.pages >= 3
    assert c.title_found is True
    assert c.not_embedded == []
    text = _pdf_text(rep.pdf)  # type: ignore[arg-type]
    for needle in ("Определение 1.1.", "Приложение А.", "H1:p1", "Рис. 1."):
        assert needle in text, needle
    assert "[[" not in text


# ---------------------------------------------------------------- review regressions


def test_has_headings_ignores_front_matter() -> None:
    assert not has_headings("---\ntitle: X\n# комментарий YAML\n---\n\nТекст.\n")
    assert has_headings("---\ntitle: X\n# комментарий YAML\n---\n\n# Раздел\n")


def test_title_fragments() -> None:
    assert title_fragments("О $\\sigma$-алгебрах") == ["О", "-алгебрах"]
    assert title_fragments('Слово "в кавычках" -- и [[H1:p1]] *всё*') == [
        "Слово",
        "в кавычках",
        "и всё",
    ]
    assert title_fragments("[Ссылка](http://x) и текст...") == ["Ссылка и текст"]
    assert title_fragments("$x$") == []
    assert title_fragments(None) == []


def test_pdfcheck_title_fragments(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, embed=True, toc=True, title="Sample Title")
    assert check_pdf(pdf, expected_title=["Sample", "title"]).title_found is True
    rep = check_pdf(pdf, expected_title=["Sample", "Other"])
    assert rep.title_found is False
    assert any("Sample … Other" in w for w in rep.warnings)
    assert check_pdf(pdf, expected_title=[]).title_found is None


def test_prepare_metadata_leaves_clean_to_front_matter(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    meta, effective = _prepare_metadata(src, {"h0lon-clean": True, "title": "T"}, None, False)
    assert "h0lon-clean" not in meta  # a -M value would override the front matter
    assert effective["h0lon-clean"] is True
    meta, _ = _prepare_metadata(src, {}, {"h0lon-clean": "true"}, False)
    assert meta["h0lon-clean"] == "true"
    meta, effective = _prepare_metadata(src, {"h0lon-clean": False}, None, True)
    assert meta["h0lon-clean"] is True and effective["h0lon-clean"] is True


def _link_pages(pdf: Path) -> list[tuple[int, int]]:
    """(page of the link, target page) for every internal link."""
    out = []
    with pymupdf.open(str(pdf)) as doc:
        for page in doc:
            for link in page.get_links():
                if link.get("kind") in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED):
                    out.append((page.number, link["page"]))
    return out


def _edge_master(n: int) -> str:
    parts = [
        "---",
        "title: 'О $\\sigma$-алгебрах и \"кавычках\"'",
        "# служебный комментарий YAML",
        "course: Теория меры",
        "h0lon-clean: true",
        "toc: false",
        "---",
        "",
        "# Раздел с якорем [[H1:p1]] {#разд}",
        "",
        "[[H1:p2]]: определение со страницы два.",
        "",
        '::: {.definition #опр:0 title="Понятие"}',
        "<!-- src: H1.b002 -->",
        "Текст определения.",
        ":::",
        "",
    ]
    filler = "слово " * 60
    for i, (kind, _) in enumerate(_EDGE_KINDS[:n], 1):
        parts += [
            filler * (i % 4 + 1),
            "",
            f"::: {{.{kind} #опр:{i}}}",
            f"Номер {i}.",
            ":::",
            "",
        ]
    parts += ["См. [раздел](#разд), [понятие](#опр:0)"]
    parts += [f", [{i}](#опр:{i})" for i in range(1, n + 1)]
    parts += ["."]
    return "\n".join(parts) + "\n"


# Mostly definitions (the case that broke), plus theorems and lemmas (their own macros).
_EDGE_KINDS = [
    ("theorem", "Теорема")
    if i % 6 == 2
    else ("lemma", "Лемма")
    if i % 6 == 4
    else ("definition", "Определение")
    for i in range(1, 37)
]


def _edge_labels() -> list[str]:
    counters = {"definition": 1}  # опр:0 is «Определение 1.1.»
    out = []
    for kind, word in _EDGE_KINDS:
        counters[kind] = counters.get(kind, 0) + 1
        out.append(f"{word} 1.{counters[kind]}.")
    return out


@pytest.mark.needs_xelatex
def test_render_edge_cases_xelatex(settings: Settings, tmp_path: Path) -> None:
    """Front-matter clean mode, Cyrillic ids, comment in a block, anchors after \\Needspace."""
    _xelatex_or_skip()
    src = tmp_path / "edge.md"
    src.write_text(_edge_master(len(_EDGE_KINDS)), encoding="utf-8")
    rep = render_document(src, settings=settings, engine="xelatex")
    assert rep.ok, rep.errors
    assert not any("Неразрешённые ссылки" in w for w in rep.warnings), rep.warnings
    assert not any("Заголовок" in w or "закладок" in w for w in rep.warnings), rep.warnings
    assert rep.checks is not None and rep.checks.title_found is True
    text = _pdf_text(rep.pdf)  # type: ignore[arg-type]
    assert "H1:p" not in text  # h0lon-clean: true from the front matter
    assert "определение со страницы два." in text  # not eaten as a link definition
    assert "Понятие. Текст определения." in text  # label and text on one line
    links = _link_pages(rep.pdf)  # type: ignore[arg-type]
    labels = _edge_labels()
    assert len(links) == len(labels) + 2
    with pymupdf.open(str(rep.pdf)) as doc:
        for label, (_, target) in zip(labels, links[2:], strict=True):
            pages = [p.number for p in doc if p.search_for(label)]
            assert len(pages) == 1, label
            assert target == pages[0], (label, target, pages)


@pytest.mark.needs_browser
def test_render_html_isolated_from_foreign_files(settings: Settings, tmp_path: Path) -> None:
    """Raw HTML and foreign images never reach the PDF; outline without anchors; date."""
    _browser_or_skip()
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("SECRET-LOCAL-FILE-CONTENT-42", encoding="utf-8")
    src = tmp_path / "doc" / "inj.md"
    src.parent.mkdir()
    src.write_text(
        "---\ntitle: Ряды и интегралы\ncourse: Матанализ\n---\n\n"
        "# Раздел с якорем [[H1:p2]]\n\nТекст перед.\n\n"
        f'<iframe src="{secret.as_uri()}" width="600" height="100"></iframe>\n\n'
        '<img src="https://example.invalid/track.png">\n\n'
        f"![Чужой]({secret.as_posix()})\n\n"
        + "Текст после. " * 400
        + "\n\n# Второй раздел\n\nКонец.\n",
        encoding="utf-8",
    )
    rep = render_document(src, settings=settings, engine="html")
    assert rep.ok, rep.errors
    text = _pdf_text(rep.pdf)  # type: ignore[arg-type]
    assert "SECRET" not in text
    assert not any("fetch" in w for w in rep.warnings), rep.warnings
    assert any("сырого HTML" in w for w in rep.warnings)
    assert text.count("Ряды и интегралы") >= 2  # title page + running heads
    assert "�" not in text
    assert re.search(r"\d{1,2} [а-я]+ \d{4} г\.", text)  # build date on the title page
    with pymupdf.open(str(rep.pdf)) as doc:
        titles = [t[1] for t in doc.get_toc()]
    assert any(t.startswith("1. Раздел с якорем") for t in titles), titles
    assert not any("H1:p2" in t or "⁠" in t for t in titles), titles
