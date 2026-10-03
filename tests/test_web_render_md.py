"""Source Doc → HTML: MathML, block markers, location headings, safety, cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from h0lon import tools
from h0lon.web import render_md
from h0lon.web.render_md import SourceDocError, postprocess, source_doc_html

pytestmark = pytest.mark.skipif(tools.find_pandoc("") is None, reason="Pandoc не найден")

SOURCE = """---
id: S1
kind: slides
title: Лекция
---

## [[S1:s1]] Слайд 1. Метрики

<!-- S1.b001 paragraph -->

Расстояние между $x$ и $y$ задаётся формулой.

<!-- S1.b002 formula -->

$$\\rho(a,b)=\\left(\\sum_i |a_i-b_i|^p\\right)^{1/p}$$

## [[P1:p3]] Страница 3

<!-- P1.b003 definition -->

::: {.definition title="Метрика"}
Функция $\\rho$ называется метрикой.
:::

## [[W1:§2]] Раздел

![схема](figures/fig 1.png)

| a | b |
|---|---|
| 1 | 2 |
"""


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    render_md.clear_cache()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "extracted" / "S1" / "source.md"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE, encoding="utf-8")
    return path


def test_formulas_become_mathml_without_scripts(source: Path) -> None:
    html = source_doc_html(source, files_base="/files/k/t/extracted/S1/")
    assert html.count("<math") == 4  # x, y, the display formula and rho in the definition
    assert 'display="block"' in html
    assert "application/x-tex" in html  # the TeX source stays in the annotation
    assert "<script" not in html and "katex" not in html.lower()


def test_front_matter_is_not_rendered(source: Path) -> None:
    html = source_doc_html(source)
    assert "kind: slides" not in html and "id: S1" not in html


def test_location_headings_carry_anchor_and_page(source: Path) -> None:
    html = source_doc_html(source)
    assert 'data-loc="S1:s1" data-page="1"' in html
    assert 'data-loc="P1:p3" data-page="3"' in html
    # an anchor that is not a page or slide (a section of a web page) has no page number
    assert 'data-loc="W1:§2"' in html
    assert 'data-loc="W1:§2" data-page' not in html
    assert '<span class="loc-tag">S1:s1</span> Слайд 1. Метрики' in html
    assert "[[S1:s1]]" not in html


def test_block_comments_become_markers(source: Path) -> None:
    html = source_doc_html(source)
    assert '<div class="blk" id="S1.b001" data-type="paragraph">S1.b001 · paragraph</div>' in html
    assert 'id="P1.b003" data-type="definition"' in html
    assert "<!--" not in html


def test_semantic_div_keeps_class_and_title(source: Path) -> None:
    html = source_doc_html(source)
    assert '<div class="definition" title="Метрика">' in html


def test_images_resolve_against_files_base(source: Path) -> None:
    html = source_doc_html(source, files_base="/files/kurs/tema/extracted/S1/")
    assert 'src="/files/kurs/tema/extracted/S1/figures/fig%201.png"' in html


def test_tables_are_wrapped_for_scrolling(source: Path) -> None:
    html = source_doc_html(source)
    assert '<div class="table-wrap"><table>' in html
    assert html.count("</table></div>") == 1


def test_untrusted_content_is_neutralised(tmp_path: Path) -> None:
    hostile = tmp_path / "source.md"
    hostile.write_text(
        """## [[W1:§1]] Страница

<script>alert(1)</script>

<iframe src="http://evil.example/"></iframe>

<img src="x" onerror="alert(2)">

[щёлк](javascript:alert(3)) и [ещё](  JaVaScRiPt:alert(4)) и [данные](data:text/html,<b>x</b>)

[сайт](https://example.org/page) и [локально](#section)

![внешняя](http://evil.example/pixel.png)

![абсолютная](C:/Windows/win.ini)
""",
        encoding="utf-8",
    )
    html = source_doc_html(hostile, files_base="/files/k/t/extracted/W1/")
    lowered = html.lower()
    assert "<script" not in lowered
    assert "<iframe" not in lowered
    assert "onerror" not in lowered
    assert "javascript:" not in lowered.replace(" ", "")
    assert 'href="data:' not in lowered
    assert 'href="#"' in html  # the neutralised links
    # an external link opens elsewhere without a referrer; an in-page link is left alone
    assert 'href="https://example.org/page" target="_blank" rel="noopener noreferrer"' in html
    assert 'href="#section"' in html and 'href="#section" target' not in html
    # images from the network or outside the document are replaced by their description
    assert "evil.example/pixel.png" not in html
    assert "win.ini" not in html
    assert "h0-image-blocked" in html


def test_result_is_cached_by_mtime(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = render_md.run_pandoc

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(render_md, "run_pandoc", counting)
    first = source_doc_html(source, files_base="/a/")
    assert source_doc_html(source, files_base="/a/") == first
    assert len(calls) == 1  # served from the cache
    source_doc_html(source, files_base="/b/")  # another URL base: its own entry
    assert len(calls) == 2
    source.write_text(SOURCE + "\nДобавлено.\n", encoding="utf-8")
    changed = source_doc_html(source, files_base="/a/")
    assert "Добавлено." in changed and len(calls) == 3


def test_missing_file_and_missing_pandoc(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError):
        source_doc_html(tmp_path / "nope.md")
    monkeypatch.setattr(render_md.tools, "find_pandoc", lambda override="": None)
    with pytest.raises(SourceDocError, match="Не найден Pandoc"):
        source_doc_html(source)
    with pytest.raises(SourceDocError, match="Не удалось запустить Pandoc"):
        source_doc_html(source, pandoc=tmp_path / "no-such-pandoc.exe")


def test_pandoc_failure_is_a_readable_error(source: Path, tmp_path: Path) -> None:
    broken = tmp_path / "pandoc-fail.cmd" if tools.IS_WINDOWS else tmp_path / "pandoc-fail.sh"
    if tools.IS_WINDOWS:
        broken.write_text("@echo off\r\necho boom 1>&2\r\nexit /b 3\r\n", encoding="utf-8")
    else:
        broken.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
        broken.chmod(0o755)
    with pytest.raises(SourceDocError, match="Pandoc"):
        source_doc_html(source, pandoc=broken)


def test_postprocess_is_pure_text_work() -> None:
    out = postprocess(
        '<h2 id="x">[[S1:s12]] Слайд 12</h2>\n<!-- S1.b009 list -->\n<p><a href="http://a.b/">a</a></p>',
        "/files/",
    )
    assert 'data-page="12"' in out
    assert 'class="blk" id="S1.b009"' in out
    assert 'rel="noopener noreferrer"' in out
