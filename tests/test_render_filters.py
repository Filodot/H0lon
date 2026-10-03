"""Markdown → LaTeX/HTML through Pandoc with the H0lon Lua filters (no TeX needed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from h0lon import tools
from h0lon.render.pandoc import convert_text
from h0lon.render.templates import BUILTIN_TEMPLATES_DIR

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATE_DIR = BUILTIN_TEMPLATES_DIR / "a4-notes"


@pytest.fixture(scope="module")
def pandoc() -> Path:
    path = tools.find_pandoc()
    if path is None:
        pytest.skip("pandoc not found")
    return path


def _flat(text: str) -> str:
    """Pandoc wraps long lines; compare with single spaces instead."""
    return re.sub(r"\s+", " ", text)


def to_latex(pandoc: Path, md: str, **kwargs: object) -> str:
    res = convert_text(md, pandoc=pandoc, to="latex", **kwargs)  # type: ignore[arg-type]
    assert res.ok, res.error
    return _flat(res.stdout)


def to_html(pandoc: Path, md: str, **kwargs: object) -> str:
    kwargs.setdefault("extra_args", ["--mathml"])
    res = convert_text(md, pandoc=pandoc, to="html5", **kwargs)  # type: ignore[arg-type]
    assert res.ok, res.error
    return _flat(res.stdout)


# ---------------------------------------------------------------- semantic blocks (LaTeX)


def test_definition_title_and_id(pandoc: Path) -> None:
    md = '::: {.definition #def:rv title="Случайная величина"}\nОтображение $X$.\n:::\n'
    out = to_latex(pandoc, md)
    assert r"\begin{hzdefinition}[{Случайная величина}]\label{def:rv}Отображение" in out
    assert r"\end{hzdefinition}" in out
    assert "phantomsection" not in out  # the div wrapper itself is gone


@pytest.mark.parametrize(
    ("cls", "env"),
    [
        ("definition", "hzdefinition"),
        ("theorem", "hztheorem"),
        ("lemma", "hzlemma"),
        ("proposition", "hzproposition"),
        ("corollary", "hzcorollary"),
        ("proof", "hzproof"),
        ("example", "hzexample"),
        ("problem", "hzproblem"),
        ("solution", "hzsolution"),
        ("remark", "hzremark"),
        ("editorial", "hzeditorial"),
        ("conflict", "hzconflict"),
        ("uncertain", "hzuncertain"),
        ("author-question", "hzauthorquestion"),
    ],
)
def test_every_block_class_has_an_environment(pandoc: Path, cls: str, env: str) -> None:
    out = to_latex(pandoc, f"::: {cls}\nТекст блока.\n:::\n")
    assert rf"\begin{{{env}}}Текст блока." in out
    assert rf"\end{{{env}}}" in out


def test_environments_exist_in_template() -> None:
    tex = (TEMPLATE_DIR / "template.tex").read_text(encoding="utf-8")
    lua = (Path(BUILTIN_TEMPLATES_DIR).parent / "filters" / "blocks.lua").read_text(
        encoding="utf-8"
    )
    envs = set(re.findall(r"env = '(hz[a-z]+)'", lua))
    assert len(envs) == 14
    for env in envs:
        assert rf"\newenvironment{{{env}}}" in tex, env


def test_block_starting_with_list_keeps_label_line(pandoc: Path) -> None:
    md = "::: {.theorem #thm:x}\n1. первое;\n2. второе.\n:::\n"
    out = to_latex(pandoc, md)
    begin = out.index(r"\begin{hztheorem}\label{thm:x}")
    assert out.index(r"\begin{enumerate}") > begin
    assert out.index(r"\end{hztheorem}") > out.index(r"\end{enumerate}")


def test_title_with_math_and_trailing_period(pandoc: Path) -> None:
    out = to_latex(pandoc, '::: {.lemma title="О $\\sigma$-алгебре."}\nТекст.\n:::\n')
    assert r"\begin{hzlemma}[{О \(\sigma\)-алгебре}]Текст." in out


def test_nested_proof_inside_unknown_div(pandoc: Path) -> None:
    md = "::: wrapper\n::: proof\nДоказательство.\n:::\n:::\n"
    out = to_latex(pandoc, md)
    assert r"\begin{hzproof}Доказательство." in out


def test_appendix_header_switches_once(pandoc: Path) -> None:
    md = "# Тело\n\nТекст.\n\n# Расхождения {.appendix}\n\nА.\n\n# Журнал {.appendix}\n\nБ.\n"
    out = to_latex(pandoc, md)
    assert out.count(r"\hzappendix") == 1
    assert out.index(r"\hzappendix") < out.index(r"\section{Расхождения}")
    assert out.index(r"\section{Тело}") < out.index(r"\hzappendix")


def test_russian_quotes(pandoc: Path) -> None:
    out = to_latex(pandoc, 'Слово "величина" историческое.\n')
    assert "«величина»" in out


# ---------------------------------------------------------------- source anchors


def test_anchor_glued_to_word_and_punctuation(pandoc: Path) -> None:
    out = to_latex(pandoc, "вероятностная мера [[H1:p1]]. Дальше\n")
    assert r"мера\hzsrc{H1:p1}. Дальше" in out


def test_consecutive_anchors_merge(pandoc: Path) -> None:
    out = to_latex(pandoc, "все $t$.[[H1:p1]][[S1:s12]] и ещё [[S1:s13]] [[V1:15:02]] конец\n")
    assert r"\hzsrc{H1:p1, S1:s12}" in out
    assert r"ещё\hzsrc{S1:s13, V1:15:02} конец" in out


def test_several_anchors_inside_one_str(pandoc: Path) -> None:
    out = to_latex(pandoc, "a[[H1:p1]]b[[P2:p45-47]]c\n")
    assert r"a\hzsrc{H1:p1}b\hzsrc{P2:p45-47}c" in out


def test_timecode_anchor(pandoc: Path) -> None:
    out = to_latex(pandoc, "в записи [[V1:12:34]] сказано\n")
    assert r"записи\hzsrc{V1:12:34} сказано" in out


def test_anchor_opening_a_line_is_inline(pandoc: Path) -> None:
    out = to_latex(
        pandoc, "## [[H1:p1]] Страница 1\n\n| Якорь | Текст |\n|---|---|\n| [[H1:p2]] | x |\n"
    )
    assert r"\hzsrcinline{H1:p1} Страница 1}" in out
    assert r"\hzsrcinline{H1:p2} & x" in out


def test_clean_mode_removes_anchors_with_space(pandoc: Path) -> None:
    md = (
        "---\nh0lon-clean: true\n---\n\n"
        "## [[H1:p1]] Страница 1\n\n"
        "вероятностная мера [[H1:p1]]. Слово[[S1:s2]] и [[S1:s13]] [[V1:15:02]] конец.\n"
    )
    out = to_latex(pandoc, md)
    assert "hzsrc" not in out
    assert "[[" not in out
    assert "мера. Слово и конец." in out
    assert r"\subsection{Страница 1}" in out


def test_clean_mode_from_metadata_argument(pandoc: Path) -> None:
    out = to_latex(pandoc, "мера [[H1:p1]].\n", metadata={"h0lon-clean": True})
    assert "мера." in out
    assert "H1:p1" not in out


def test_text_without_anchors_is_untouched(pandoc: Path) -> None:
    md = "Ссылка [текст](#sec:x) и [квадратные скобки] и [[не якорь]].\n"
    out = to_latex(pandoc, md)
    assert "hzsrc" not in out
    assert "не якорь" in out.replace("{[}", "[").replace("{]}", "]")


def test_unreadable_is_gray(pandoc: Path) -> None:
    out = to_latex(pandoc, "Подпись: «[неразборчиво] скачок».\n")
    assert r"\hzunreadable{[неразборчиво]}" in out


# ---------------------------------------------------------------- comments and images


def test_html_comments_are_removed(pandoc: Path) -> None:
    md = (
        "<!-- H1.b001 definition -->\n"
        "::: definition\nТекст.\n:::\n\n"
        "<!-- src: H1.b007 -->\n\n"
        "Абзац <!-- inline note --> продолжение и $x$. <!-- tail -->Дальше.\n"
    )
    latex = to_latex(pandoc, md)
    html = to_html(pandoc, md)
    for out in (latex, html):
        assert "H1.b001" not in out
        assert "H1.b007" not in out
        assert "inline note" not in out
        assert "<!--" not in out
    assert "Абзац продолжение" in latex
    assert r"\(x\). Дальше." in latex


def test_relative_image_paths_become_absolute_forward_slashes(pandoc: Path) -> None:
    md = (
        "![Рисунок](img/sample.png){#fig:a}\n\n"
        "![Абсолютный](C:/data/x.png)\n\n"
        "![Сеть](https://example.org/x.png)\n"
    )
    out = to_latex(pandoc, md, resource_dir=FIXTURES)
    expected = (FIXTURES / "img" / "sample.png").resolve().as_posix()
    assert "{" + expected + "}" in out
    assert "\\" not in expected
    assert "{C:/data/x.png}" in out
    assert "https://example.org/x.png" in out


def test_image_paths_untouched_without_resource_dir(pandoc: Path) -> None:
    out = to_latex(pandoc, "![Рисунок](img/sample.png)\n")
    assert "{img/sample.png}" in out


# ---------------------------------------------------------------- HTML output


def test_html_blocks_numbering_and_anchors(pandoc: Path) -> None:
    md = (
        "# Первый\n\n"
        '::: {.definition #def:a title="Понятие"}\nТекст [[H1:p1]].\n:::\n\n'
        "::: definition\nВторое.\n:::\n\n"
        "## Подраздел\n\n::: proof\nДоказательство.\n:::\n\n"
        "# Второй\n\n::: theorem\nФормулировка.\n:::\n\n"
        "# Расхождения {.appendix}\n\n::: conflict\nРазное.\n:::\n\n"
        "# Карта {.unnumbered}\n\nТаблица.\n"
    )
    out = to_html(pandoc, md)
    assert 'class="definition h0-block"' in out
    assert 'id="def:a"' in out
    assert "Определение 1.1." in out
    assert "Определение 1.2." in out
    assert "Теорема 2.1." in out
    assert '<span class="header-section-number">1.1.</span>' in out
    assert "Приложение А." in out
    assert 'class="h0-head"' in out  # proof label on its own line
    assert '<sup class="h0src">⁠H1:p1</sup>' in out  # word joiner inside the mark
    assert "Расхождение источников." in out
    assert ">Карта<" in out  # unnumbered: no number span


def test_html_raw_align_becomes_mathml(pandoc: Path) -> None:
    md = "Текст\n\n\\begin{align*}\na &= b \\\\\nc &= d\n\\end{align*}\n\nДальше.\n"
    out = to_html(pandoc, md)
    assert '<math display="block"' in out
    assert 'class="math display"' not in out  # not left as raw TeX text
    body = re.sub(r"<annotation.*?</annotation>", "", out)
    assert "begin{align" not in body


def test_html_captions_numbered(pandoc: Path) -> None:
    md = "![Рисунок](img/sample.png)\n\n: Подпись\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = to_html(pandoc, md)
    assert "Рис. 1." in out
    assert "Таблица 1." in out


# ---------------------------------------------------------------- full template


def test_sample_master_standalone_latex(pandoc: Path) -> None:
    md = (FIXTURES / "sample_master.md").read_text(encoding="utf-8")
    out = to_latex(
        pandoc,
        md,
        template=TEMPLATE_DIR / "template.tex",
        standalone=True,
        toc=True,
        number_sections=True,
        resource_dir=FIXTURES,
        variables={"mainfont": "Times New Roman", "mathfont": "Cambria Math"},
    )
    assert out.startswith("% Options for packages loaded elsewhere")
    assert r"\documentclass[11pt,twoside,a4paper]{article}" in out
    assert r"\setmainfont[]{Times New Roman}" in out
    assert r"\setmathfont[]{Cambria Math}" in out
    assert r"\setdefaultlanguage{russian}" in out
    assert r"\begin{titlepage}" in out
    assert r"\item[H1] Лекция 3" in out
    assert "(рукопись, 2 стр.)" in out
    assert "(видео, 85 мин)" in out
    assert r"\newcommand{\HzCourse}{Теория вероятностей}" in out
    assert r"\tableofcontents" in out
    assert r"\setcounter{secnumdepth}{3}" in out
    assert "pdftitle={Случайные величины и функции распределения}" in out
    assert r"\begin{Highlighting}" in out  # fancyvrb, not listings
    assert "lstlisting" not in out
    assert "{" + (FIXTURES / "img" / "sample.png").resolve().as_posix() + "}" in out
    assert r"\hzappendix" in out
    assert "<!--" not in out


def test_sample_master_standalone_html(pandoc: Path) -> None:
    md = (FIXTURES / "sample_master.md").read_text(encoding="utf-8")
    out = to_html(
        pandoc,
        md,
        template=TEMPLATE_DIR / "template.html",
        standalone=True,
        toc=True,
        resource_dir=FIXTURES,
    )
    assert 'class="h0-titlepage"' in out
    assert "Content-Security-Policy" in out
    assert "<dt>H1</dt>" in out
    assert '@top-left { content: "Теория вероятностей"; }' in out
    assert 'id="TOC"' in out
    assert "<!--" not in out.split("</head>", 1)[1]


# ---------------------------------------------------------------- review regressions

FILTERS_DIR = Path(BUILTIN_TEMPLATES_DIR).parent / "filters"


def test_lua_filters_use_no_locale_dependent_classes() -> None:
    # Pandoc's Lua on Windows runs with LC_CTYPE=Russian_Russia.1251: %s also matches 0xA0,
    # a byte of UTF-8 «Р». Only explicit ASCII sets may be used on text.
    for path in sorted(FILTERS_DIR.glob("*.lua")):
        code = path.read_text(encoding="utf-8")
        code = re.sub(r"--\[(=*)\[.*?\]\1\]", "", code, flags=re.DOTALL)  # block comments
        for no, line in enumerate(code.splitlines(), 1):
            line = line.split("--", 1)[0]
            if "string.format" in line:
                continue  # %s / %d there are format specifiers
            bad = re.findall(r"(?<!%)%[aAcCdDgGlLpPsSuUwWxX]", line)
            assert not bad, f"{path.name}:{no}: {line.strip()}"


def test_capital_er_survives_running_head_and_code_wrap(pandoc: Path) -> None:
    md = "---\ntitle: Ромб и хорда\n---\n\nТекст.\n"
    res = convert_text(
        md, pandoc=pandoc, to="html5", standalone=True, template=TEMPLATE_DIR / "template.html"
    )
    assert res.ok, res.error
    assert '@top-right { content: "Ромб и хорда"; }' in res.stdout
    assert "\ufffd" not in res.stdout

    code = "# ВЕКТОР " + "б" * 100
    out = convert_text(f"```\n{code}\n```\n", pandoc=pandoc, to="latex")
    assert out.ok, out.error
    assert "# ВЕКТОР\n" in out.stdout
    assert "\ufffd" not in out.stdout


def test_cyrillic_block_id_matches_link_label(pandoc: Path) -> None:
    md = (
        "# Раздел {#разд}\n\n"
        '::: {.definition #опр:св title="Случайная величина"}\nТекст.\n:::\n\n'
        "::: {.theorem #теор-1}\nФормулировка.\n:::\n\n"
        "См. [определение](#опр:св), [теорему](#теор-1) и [раздел](#разд).\n"
    )
    out = to_latex(pandoc, md)
    targets = re.findall(r"\\hyperref\[([^\]]+)\]", out)
    assert len(targets) == 3
    for target in targets:
        assert rf"\label{{{target}}}" in out, target
    assert r"\label{ux43eux43fux440:ux441ux432}" in out
    assert r"\label{опр:св}" not in out


def test_comment_opening_a_block_keeps_label_run_in(pandoc: Path) -> None:
    md = (
        '::: {.definition #def:a title="Понятие"}\n<!-- src: H1.b002 -->\nТекст определения.\n'
        ":::\n\n"
        "::: example\n<!-- src: H1.b003 --> Пример в строку.\n:::\n\n"
        "::: proof\n<!-- src: H1.b004 -->\nТекст доказательства.\n:::\n"
    )
    latex = to_latex(pandoc, md)
    assert r"\begin{hzdefinition}[{Понятие}]\label{def:a}Текст определения." in latex
    assert r"\begin{hzexample}Пример в строку." in latex
    assert r"\begin{hzproof}Текст доказательства." in latex
    html = to_html(pandoc, md)
    assert '<span class="h0-block-title">Понятие.</span> Текст определения.</p>' in html
    assert "</span> </p>" not in html  # no paragraph that holds only the label
    assert "H1.b00" not in latex + html


def test_anchor_link_syntax_collisions(pandoc: Path) -> None:
    md = (
        "Абзац один.\n\n"
        "[[H1:p1]]: определение со страницы один.\n\n"
        "Текст [[H1:p2]](примечание) дальше, [[S1:s3]]{.x} и [[V1:1:02]][[V1:1:05]].\n\n"
        "- [[H1:p4]]: пункт списка.\n\n"
        "Код `[[H1:p5]](x)` не трогаем.\n\n"
        "```\n[[H1:p6]]: в блоке кода\n```\n"
    )
    out = to_latex(pandoc, md)
    assert r"\hzsrcinline{H1:p1}: определение со страницы один." in out
    assert r"Текст\hzsrc{H1:p2}(примечание) дальше" in out
    assert r"\href" not in out
    assert r"\hzsrc{S1:s3}\{.x\}" in out
    assert r"\hzsrc{V1:1:02, V1:1:05}" in out
    assert r"\hzsrcinline{H1:p4}: пункт списка." in out
    assert r"\texttt{{[}{[}H1:p5{]}{]}(x)}" in out
    assert "[[H1:p6]]: в блоке кода" in out


def test_escape_anchor_syntax_keeps_front_matter_and_code() -> None:
    from h0lon.render.pandoc import escape_anchor_syntax

    md = (
        '---\ntitle: "Тема [[H1:p1]]: обзор"\n---\n\n'
        "[[H1:p2]]: текст и `[[H1:p3]](x)`\n\n~~~\n[[H1:p4]](y)\n~~~\n\nобычный [[H1:p5]].\n"
    )
    out = escape_anchor_syntax(md)
    assert '"Тема [[H1:p1]]: обзор"' in out
    assert r"\[\[H1:p2\]\]: текст и `[[H1:p3]](x)`" in out
    assert "\n[[H1:p4]](y)\n" in out
    assert "обычный [[H1:p5]]." in out
    assert escape_anchor_syntax("Без якорей.\n") == "Без якорей.\n"


# ---------------------------------------------------------------- HTML sanitising


def test_html_drops_raw_html_and_foreign_resources(pandoc: Path, tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    md = (
        "Текст перед.\n\n"
        '<iframe src="file:///C:/secret.txt" width="600"></iframe>\n\n'
        '<img src="https://example.invalid/track.png">\n\n'
        'Строка<br>перенос, H<sub>2</sub>O и <span style="x">спан</span>.\n\n'
        '![Свой](img/ok.png){width=40% style="background:url(file:///C:/s.png)"}\n\n'
        "![Выше](../secret.png)\n\n![Абсолютный](C:/Windows/win.ini)\n\n"
        "![Сеть](https://example.invalid/x.png)\n\n![Файл](file:///C:/x.png)\n\n"
        "![Скрыто](img/..%2F..%2Fsecret.png)\n\n"
        '[текст]{src="file:///C:/secret.txt" data-src="https://example.invalid/"}\n\n'
        "Текст после.\n"
    )
    res = convert_text(md, pandoc=pandoc, to="html5", resource_dir=tmp_path)
    assert res.ok, res.error
    out = res.stdout
    for needle in ("iframe", "example.invalid", "secret", "win.ini", "file:", 'style="x"'):
        assert needle not in out, needle
    assert "<br />" in out or "<br>" in out
    assert "<sub>2</sub>" in out
    assert "спан" in out
    assert (tmp_path / "img" / "ok.png").as_posix() in out
    assert 'style="width:40.0%"' in out
    assert "background" not in out
    for alt in ("Выше", "Абсолютный", "Сеть", "Файл", "Скрыто"):
        assert f'<span class="h0-image-blocked">{alt}</span>' in out, alt
    warnings = " ".join(res.warnings)
    assert "сырого HTML" in warnings
    assert "../secret.png" in warnings
    assert "Scripting warning" not in warnings


def test_html_allows_images_inside_source_dir(pandoc: Path, tmp_path: Path) -> None:
    inside = (tmp_path / "img" / "a.png").as_posix()
    md = (
        f"![Абс]({inside})\n\n![Отн](./img/../img/a.png)\n\n![Данные](data:image/png;base64,AAAA)\n"
    )
    out = to_html(pandoc, md, resource_dir=tmp_path)
    assert "h0-image-blocked" not in out
    assert out.count("<img") == 3


def test_latex_remote_image_becomes_a_link(pandoc: Path) -> None:
    # XeLaTeX cannot load remote pictures: paths.lua turns them into a link with the caption
    out = to_latex(pandoc, "![Сеть](https://example.org/x.png){width=50%}\n")
    assert r"\includegraphics" not in out
    assert r"\href{https://example.org/x.png}{Рисунок: Сеть}" in out


def test_html_heading_anchor_hidden_from_outline_and_default_date(pandoc: Path) -> None:
    md = (
        "---\ntitle: Тема\n---\n\n# Раздел [[H1:p2]]\n\n"
        "## [[H1:p3]] Страница 3\n\nТекст [[H1:p4]].\n"
    )
    res = convert_text(
        md,
        pandoc=pandoc,
        to="html5",
        standalone=True,
        toc=True,
        template=TEMPLATE_DIR / "template.html",
        extra_args=["--mathml"],
    )
    assert res.ok, res.error
    out = _flat(res.stdout)
    assert '<sup class="h0src" aria-hidden="true">\u2060H1:p2</sup>' in out
    assert '<span class="h0src h0src-inline" aria-hidden="true">H1:p3</span>' in out
    assert '<sup class="h0src">\u2060H1:p4</sup>' in out  # body text: not hidden
    assert "\u2060<sup" not in out  # the word joiner lives inside the mark
    assert re.search(r'<p class="h0-date">\d{1,2} [а-я]+ \d{4} г\.</p>', out)
    dated = convert_text(
        "---\ntitle: Т\ndate: 1 сентября 2026\n---\n\nТекст.\n",
        pandoc=pandoc,
        to="html5",
        standalone=True,
        template=TEMPLATE_DIR / "template.html",
    )
    assert '<p class="h0-date">1 сентября 2026</p>' in dated.stdout
