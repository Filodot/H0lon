"""Built-in template a4-compact: discovery, packages, and real renders (XeLaTeX, HTML fallback).

The template is a dense reference sheet: A4, 9 pt, 12 mm margins, two columns (multicol), no title
page and no contents. Real renders are skipped when XeLaTeX / a browser is missing.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pymupdf
import pytest
import yaml

from h0lon import tools
from h0lon.config import Settings
from h0lon.render import get_template, list_templates, render_document
from h0lon.render.templates import BUILTIN_TEMPLATES_DIR

FIXTURES = Path(__file__).parent / "fixtures"
A4 = (595, 842)

WIDE = " + ".join(f"x_{{{i}}}^{{2}}" for i in range(1, 40))
PARAGRAPH = (
    "Расстояние между объектами задаётся метрикой, которая симметрична и удовлетворяет "
    "неравенству треугольника. "
) * 4

DOC = f"""---
title: "Компактный лист"
subtitle: "Подзаголовок листа"
course: "Курс"
date: "2026-10-04"
---

# Первый раздел {{#sec:a}}

{PARAGRAPH}

| Имя | Значение |
|-----|----------|
| альфа | 1 |
| бета | 2 |

После таблицы без подписи текст продолжается в колонках. {PARAGRAPH}

| x | y |
|---|---|
| 1 | 2 |

: Подпись таблицы {{#tbl:x}}

{PARAGRAPH}

Формула, которая не помещается в колонку:

$$ {WIDE} $$

::: {{.definition #def:m title="Метрика"}}
Функция $\\rho$ называется метрикой.[[S1:s3]]
:::

::: {{.theorem title="Тест"}}
Утверждение с формулой $x^2 \\ge 0$.
:::

::: {{.proof}}
Доказательство: очевидно. {PARAGRAPH}
:::

![Подпись рисунка](img/sample.png)

См. [метрику](#def:m). {PARAGRAPH}

```
код
```

# Приложение {{.appendix #app:a}}

Текст приложения.
"""


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    d = tmp_path / "Лист"
    (d / "img").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "img" / "sample.png", d / "img" / "sample.png")
    (d / "sheet.md").write_text(DOC, encoding="utf-8")
    return d / "sheet.md"


def _text(pdf: Path) -> str:
    with pymupdf.open(str(pdf)) as pdf_doc:
        return "".join(page.get_text() for page in pdf_doc)


# ---------------------------------------------------------------- discovery


def test_builtin_a4_compact(settings: Settings) -> None:
    info = get_template("a4-compact", settings)
    assert info.dir == BUILTIN_TEMPLATES_DIR / "a4-compact"
    assert info.template_tex.is_file() and info.title != "a4-compact"
    assert info.html_css is not None and info.html_css.name == "style.css"
    assert info.template_html is not None and info.template_html.is_file()
    assert "multicol.sty" in info.latex_packages and "mdframed.sty" in info.latex_packages
    assert "a4-compact" in [t.name for t in list_templates(settings)]
    assert "a4-notes" in [t.name for t in list_templates(settings)]


def test_latex_packages_match_template_usepackage(settings: Settings) -> None:
    info = get_template("a4-compact", settings)
    tex = info.template_tex.read_text(encoding="utf-8")
    used: set[str] = set()
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        used.update(f"{name.strip()}.sty" for name in m.group(1).split(","))
    if_exists = set(re.findall(r"\\IfFileExists\{([^}]+\.sty)\}", tex))
    assert if_exists == {"upquote.sty", "xurl.sty", "footnotehyper.sty"}
    optional = if_exists | {"footnote.sty", "multirow.sty", "subcaption.sty"}
    assert set(info.latex_packages) == used - optional
    meta = yaml.safe_load((info.dir / "meta.yaml").read_text(encoding="utf-8"))
    assert set(meta["optional_latex_packages"]) == if_exists | {"footnote.sty"}
    # no title page and no contents; the 9 pt size does not need extsizes
    assert (
        "titlepage" not in tex
        and "tableofcontents" not in tex
        and "extarticle" not in tex.replace("extarticle (extsizes)", "")
    )
    assert "listings.sty" not in used and "soul.sty" not in used


def test_html_template_has_no_title_page_or_contents(settings: Settings) -> None:
    info = get_template("a4-compact", settings)
    html = info.template_html.read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "h0-titlepage" not in html and "h0-toc" not in html and "$table-of-contents$" not in html
    css = info.html_css.read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "column-count: 2" in css and "font-size: 9pt" in css and "margin: 11mm 12mm" in css


# ---------------------------------------------------------------- XeLaTeX


@pytest.mark.needs_xelatex
def test_render_compact_xelatex(settings: Settings, doc: Path, tmp_path: Path) -> None:
    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")
    out = tmp_path / "out"
    rep = render_document(
        doc, settings=settings, out_dir=out, template="a4-compact", engine="xelatex"
    )
    assert rep.ok, rep.errors
    assert rep.engine_used == "xelatex" and rep.fallback_used is False
    # nothing is lost: no unresolved references (floats are dropped by multicol), no overfull
    # lines, no missing glyphs
    assert rep.warnings == [], rep.warnings
    checks = rep.checks
    assert checks is not None and checks.ok and checks.title_found is True
    assert checks.not_embedded == [] and checks.bookmarks >= 2

    with pymupdf.open(str(rep.pdf)) as pdf:
        assert pdf.page_count <= 2
        first = pdf[0]
        assert (round(first.rect.width), round(first.rect.height)) == A4
        words = first.get_text("words")
        # the heading of the sheet is the first thing on the page, there is no title page
        top = min(words, key=lambda w: w[1])
        assert top[1] < 50 and "Компактный" in first.get_text()[:60]
        # two columns: text starts at the left margin and at the middle of the page
        starts = {round(w[0]) for w in words if w[1] > 120}
        assert any(x < 45 for x in starts) and any(300 < x < 330 for x in starts), sorted(starts)[
            :20
        ]
        # margins of about 12 mm (34 pt) on every side
        left = min(w[0] for w in words)
        right = max(w[2] for w in words)
        assert 30 < left < 40 and first.rect.width - right > 30, (left, right)
    text = _text(rep.pdf)  # type: ignore[arg-type]
    for needle in (
        "Определение 1.1.",
        "Теорема 1.1.",
        "Доказательство",
        "Таблица 1.",
        "Рис. 1.",
        "Приложение А.",
        "Подпись таблицы",
        "Курс",
        "2026-10-04",
    ):
        assert needle in text, needle
    assert "Содержание" not in text and "Подзаголовок листа" not in text
    assert "[[" not in text and "S1:s3" in text  # anchors are shown as small superscripts


@pytest.mark.needs_xelatex
def test_a4_compact_keeps_the_text_of_a_table_in_a_group_and_a_caption(
    settings: Settings, tmp_path: Path
) -> None:
    """Pandoc puts a table without a caption into { ... }; the columns must be left from there."""
    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")
    src = tmp_path / "tables.md"
    parts = ["# Таблицы\n"]
    for n in range(3):  # tables one after another and at the very start/end
        parts.append(f"| a{n} | b{n} |\n|---|---|\n| {n} | {n + 1} |\n")
        parts.append(f"| c{n} | d{n} |\n|---|---|\n| {n} | {n + 2} |\n\n: Подпись {n}\n")
        parts.append("Текст между таблицами. " * 5 + "\n")
    src.write_text("\n".join(parts), encoding="utf-8")
    rep = render_document(src, settings=settings, template="a4-compact", engine="xelatex")
    assert rep.ok, rep.errors
    text = _text(rep.pdf)  # type: ignore[arg-type]
    for n in range(3):
        assert f"a{n}" in text and f"d{n}" in text and f"Подпись {n}" in text
    flat = re.sub(r"\s+", " ", re.sub(r"-\s+", "", text))  # undo hyphenation and line breaks
    assert flat.count("Текст между таблицами.") == 15  # 3 times 5: none of the text is lost


# ---------------------------------------------------------------- HTML fallback


@pytest.mark.needs_browser
def test_render_compact_html(settings: Settings, doc: Path, tmp_path: Path) -> None:
    if tools.find_browser("auto") is None or tools.find_pandoc() is None:
        pytest.skip("browser or pandoc not found")
    rep = render_document(
        doc, settings=settings, out_dir=tmp_path / "html", template="a4-compact", engine="html"
    )
    assert rep.ok, rep.errors
    assert rep.engine_used == "html"
    checks = rep.checks
    assert checks is not None and checks.ok and checks.title_found is True
    with pymupdf.open(str(rep.pdf)) as pdf:
        assert pdf.page_count <= 3
        assert (round(pdf[0].rect.width), round(pdf[0].rect.height)) == (595, 842)
    text = _text(rep.pdf)  # type: ignore[arg-type]
    for needle in ("Определение 1.1.", "Теорема 1.1.", "Приложение А.", "Подпись таблицы"):
        assert needle in text, needle
    assert "Содержание" not in text and "[[" not in text
