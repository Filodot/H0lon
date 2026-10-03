"""DOCX / Markdown / LaTeX / web extraction (h0lon/extract/docs.py)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from h0lon import procutil, tools
from h0lon.config import Settings
from h0lon.extract import blocks as bl
from h0lon.extract.docs import DocsExtractor, normalize_tex_math, web_markdown
from h0lon.extract.model import ExtractContext
from h0lon.extract.registry import ExtractError
from h0lon.sources.models import SourceRecord

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOCS = FIXTURES / "m1" / "docs"
IMAGE = FIXTURES / "img" / "sample.png"
PAGE_URL = "https://example.org/teorver/rv.html"


@pytest.fixture(scope="module")
def pandoc() -> Path:
    path = tools.find_pandoc()
    if path is None:
        pytest.skip("Pandoc не найден")
    return path


def make_ctx(
    tmp_path: Path,
    settings: Settings,
    src: Path,
    kind: str,
    *,
    sid: str = "D1",
    url: str | None = None,
    extra: dict[str, Path] | None = None,
) -> ExtractContext:
    topic = tmp_path / "topic"
    (topic / "sources").mkdir(parents=True, exist_ok=True)
    name = f"{sid}_{src.stem}{src.suffix}"
    shutil.copy(src, topic / "sources" / name)
    for rel, path in (extra or {}).items():
        target = topic / "sources" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    record = SourceRecord(
        id=sid,
        kind=kind,  # type: ignore[arg-type]
        title=src.stem,
        file=f"sources/{name}",
        url=url,
        added="2026-10-03T00:00:00Z",
    )
    return ExtractContext(
        topic_dir=topic, source=record, settings=settings, out_dir=topic / "extracted" / sid
    )


def body_of(ctx: ExtractContext) -> str:
    return (ctx.out_dir / "body.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- DOCX


DOCX_MD = r"""---
title: Случайные величины
subtitle: Лекция 3
---

# Определения

Пусть $(\Omega,\mathcal F,P)$ — вероятностное пространство.

$$F_X(t) = P(X \le t)$$

## Свойства

1. монотонность;
2. непрерывность справа.

| Свойство | Формула |
|---|---|
| Нормировка | $F(+\infty)=1$ |

![График функции распределения](sample.png)

# Примеры

Пример текста.
"""


@pytest.fixture
def docx_file(tmp_path: Path, pandoc: Path) -> Path:
    work = tmp_path / "make-docx"
    work.mkdir()
    shutil.copy(IMAGE, work / "sample.png")
    (work / "lecture.md").write_text(DOCX_MD, encoding="utf-8")
    res = procutil.run([str(pandoc), "lecture.md", "-o", "lecture.docx"], cwd=work, timeout=120)
    assert res.ok, res.stderr
    return work / "lecture.docx"


def test_docx(tmp_path: Path, settings: Settings, pandoc: Path, docx_file: Path) -> None:
    ctx = make_ctx(tmp_path, settings, docx_file, "docx")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert out.body_md == ctx.out_dir / "body.md"
    assert body.startswith("# Случайные величины {.source-title}\n\n*Лекция 3*\n")
    assert "# Определения\n" in body and "{#" not in body  # no bookmark ids on headings
    assert "$$F_{X}(t) = P(X \\leq t)$$" in body
    assert "| Нормировка |" in body
    images = [line for line in body.splitlines() if line.startswith("![")]
    assert len(images) == 1
    rel = images[0].split("](", 1)[1].split(")", 1)[0]
    assert rel.startswith("figures/") and "\\" not in rel
    assert (ctx.out_dir / rel).is_file()
    assert out.quality["figures"] == 1 and out.quality["tables"] == 1
    assert out.quality["formulas"] >= 3

    doc = bl.build_source_doc(body, "D1", pandoc=pandoc)
    assert doc.anchor_mode == "section"
    heads = [(b.title, b.anchor) for b in doc.blocks if b.type == "heading"]
    assert heads == [
        ("Случайные величины", "D1:§0"),
        ("Определения", "D1:§1"),
        ("Свойства", "D1:§1"),
        ("Примеры", "D1:§2"),
    ]


def test_docx_reextract_replaces_figures(
    tmp_path: Path, settings: Settings, docx_file: Path
) -> None:
    ctx = make_ctx(tmp_path, settings, docx_file, "docx")
    stale = ctx.out_dir / "figures" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    DocsExtractor().extract(ctx)
    assert not stale.exists()


# ---------------------------------------------------------------- Markdown


def test_markdown_with_local_image(tmp_path: Path, settings: Settings) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.md", "md", extra={"img/venn.png": IMAGE})
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert body.startswith("# Конспект по теории вероятностей {.source-title}\n")
    assert '::: {.definition title="Вероятность"}' in body
    assert "$$P(A\\cup B) = P(A) + P(B) - P(A\\cap B)$$" in body
    assert "![Диаграмма Венна](figures/venn.png)" in body
    assert (ctx.out_dir / "figures" / "venn.png").read_bytes() == IMAGE.read_bytes()
    assert out.warnings == []


def test_markdown_missing_image_is_a_warning(tmp_path: Path, settings: Settings) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.md", "md")
    out = DocsExtractor().extract(ctx)
    assert "![Диаграмма Венна](img/venn.png)" in body_of(ctx)
    assert len(out.warnings) == 1 and "img/venn.png" in out.warnings[0]
    assert out.quality["notes"] == out.warnings


def test_markdown_outside_paths_are_not_followed(tmp_path: Path, settings: Settings) -> None:
    secret = tmp_path / "secret.png"
    secret.write_bytes(IMAGE.read_bytes())
    src = tmp_path / "notes.md"
    src.write_text("Текст.\n\n![x](../../secret.png)\n", encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    assert "](../../secret.png)" in body_of(ctx)
    assert not (ctx.out_dir / "figures").exists()
    assert out.warnings


def test_markdown_backslash_math(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "chat.md"
    src.write_text("Формула \\(a^2+b^2\\) и выключная:\n\n\\[c^2\\]\n", encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "md")
    DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert "$a^2+b^2$" in body and "$$c^2$$" in body


def test_markdown_display_backslash_math_only(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "display.md"
    src.write_text(
        "Текст.\n\n\\[\nD X = \\sum_{i=1}^{n} (x_i - \\bar{x})^2 p_i\n\\]\n\n"
        "и ещё \\[ a_1 + b_2 = c_3 \\]\n\n```\nкод \\[не формула\\]\n```\n",
        encoding="utf-8",
    )
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert "$$\nD X = \\sum_{i=1}^{n} (x_i - \\bar{x})^2 p_i\n$$" in body
    assert "$$ a_1 + b_2 = c_3 $$" in body
    assert "{=tex}" not in body and out.quality["formulas"] == 2


def test_markdown_escaped_brackets_are_not_math(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "refs.md"
    src.write_text("См. \\[1\\] и \\[Иванов, 2001\\].\n", encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    assert "См. \\[1\\] и \\[Иванов, 2001\\]." in body_of(ctx)
    assert out.quality["formulas"] == 0


def test_markdown_yaml_like_block_inside_text_is_kept(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "rules.md"
    src.write_text(
        "Первый абзац.\n\n---\nОпределение: случайная величина — измеримая функция.\n"
        "Пример: бросок кубика.\n---\n\nАбзац после блока.\n\n---\nключ: значение\n...\n\n"
        "Последний абзац.\n",
        encoding="utf-8",
    )
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    for fragment in (
        "Определение: случайная величина — измеримая функция.",
        "Пример: бросок кубика.",
        "Абзац после блока.",
        "ключ: значение",
        "Последний абзац.",
    ):
        assert fragment in body, fragment
    assert out.warnings == []


def test_markdown_broken_front_matter(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "lecture.md"
    src.write_text(
        '---\ntitle: Лекция 3: случайные величины\ntags: [теорвер, "лекция]\n---\n\n'
        "# Введение\n\nСодержательный текст.\n",
        encoding="utf-8",
    )
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert body.startswith("# Лекция 3: случайные величины {.source-title}\n\n# Введение\n")
    assert "Содержательный текст." in body and "tags" not in body
    assert len(out.warnings) == 1 and "не разобрана" in out.warnings[0]
    assert "строка 2 файла" in out.warnings[0] and "название взято" in out.warnings[0]


def test_markdown_front_matter_fields_and_plain_rules(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "front.md"
    src.write_text(
        "---\ntitle: Случайные *величины*\nsubtitle: Лекция 3\nabstract: Кратко о $X$.\n"
        "tags: [теорвер]\n---\n\nТекст.\n",
        encoding="utf-8",
    )
    ctx = make_ctx(tmp_path, settings, src, "md")
    DocsExtractor().extract(ctx)
    assert body_of(ctx) == (
        "# Случайные *величины* {.source-title}\n\n*Лекция 3*\n\nКратко о $X$.\n\nТекст.\n"
    )
    # Text between two rules at the top is not metadata.
    src2 = tmp_path / "rules.md"
    src2.write_text("---\nПросто строка текста\n---\n\nАбзац.\n", encoding="utf-8")
    ctx2 = make_ctx(tmp_path, settings, src2, "md", sid="D2")
    DocsExtractor().extract(ctx2)
    assert "Просто строка текста" in body_of(ctx2)


def test_txt_in_cp1251(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "old.txt"
    src.write_bytes("Старый конспект в кодировке Windows.\n".encode("cp1251"))
    ctx = make_ctx(tmp_path, settings, src, "md")
    out = DocsExtractor().extract(ctx)
    assert "Старый конспект" in body_of(ctx)
    assert any("Windows-1251" in n for n in out.quality["notes"])


def test_empty_document_raises(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "empty.md"
    src.write_text("\n<!-- пусто -->\n", encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "md")
    with pytest.raises(ExtractError, match="не найдено текста"):
        DocsExtractor().extract(ctx)


def test_missing_file_raises(tmp_path: Path, settings: Settings) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.md", "md")
    (ctx.topic_dir / ctx.source.file).unlink()  # type: ignore[operator]
    with pytest.raises(ExtractError, match="Файл источника не найден"):
        DocsExtractor().extract(ctx)


# ---------------------------------------------------------------- LaTeX


def test_latex(tmp_path: Path, settings: Settings, pandoc: Path) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.tex", "tex", extra={"img/sample.png": IMAGE})
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert body.startswith(
        "# Случайные величины {.source-title}\n\nУчебный пример для проверки извлечения.\n"
    )
    # Theorem environments become blocks with titles; Pandoc's labels are gone.
    assert '::: {#def:rv .definition title="Случайная величина"}' in body
    assert '::: {.theorem title="Неравенство Чебышёва"}' in body
    assert "**Определение" not in body and "**Теорема" not in body
    assert "::: proof\nСледует из неравенства Маркова.\n:::" in body
    assert "Proof" not in body and "◻" not in body
    assert "::: remark\nЗамечание без номера.\n:::" in body
    # \newcommand is expanded; numbered environments become plain display math.
    assert "$X:\\Omega\\to\\mathbb{R}$" in body
    assert "\\mathsf{P}\\left(|X-a|\\ge\\varepsilon\\right)" in body
    assert "$$F(x) = \\mathsf{P}\\left(X\\le x\\right),$$" in body
    assert "\\begin{aligned}" in body and "\\begin{align}" not in body
    assert "\\label" not in body and "\\nonumber" not in body
    assert "::: center" not in body and "Текст по центру." in body
    assert "figures/sample.png" in body and (ctx.out_dir / "figures" / "sample.png").is_file()
    assert out.warnings == []

    doc = bl.build_source_doc(body, "D1", pandoc=pandoc)
    types = [b.type for b in doc.blocks]
    for expected in ("definition", "theorem", "proof", "remark", "formula", "list", "table"):
        assert expected in types
    formula = next(b for b in doc.blocks if b.type == "formula")
    assert formula.text == "$$F(x) = \\mathsf{P}\\left(X\\le x\\right),$$"
    after = doc.blocks[doc.blocks.index(formula) + 1]
    assert after.type == "paragraph" and after.text.startswith("где $x\\in\\mathbb{R}$")


UNKNOWN_MACROS_TEX = r"""\documentclass{article}
\begin{document}
\section{Основы}\label{sec:a}
Обычный текст. \term{Случайная величина} --- это функция. \textbf{Жирный} и \emph{курсив}.
\vspace{1cm}Ссылка на раздел \ref{sec:a}, рамка \fbox{в рамке}.

\important{Это важное замечание про дисперсию $DX$.}

\defn{Мат. ожидание}{среднее значение случайной величины}

\note[на полях]{Заметка с необязательным аргументом.} Символ \R и формула $\myop{x}$, но
\myop{текст аргумента}.

\resizebox{\textwidth}{!}{%
\begin{tabular}{cc}
$x$ & $p$ \\
0 & 0.5 \\
\end{tabular}}

\begin{boxedtext}
Внутри окружения: \term{вложенный термин}.
\end{boxedtext}
\end{document}
"""


def test_latex_unknown_commands_keep_their_text(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "unk.tex"
    src.write_text(UNKNOWN_MACROS_TEX, encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "tex")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    for fragment in (
        "Случайная величина — это функция. **Жирный** и *курсив*.",
        "Это важное замечание про дисперсию $DX$.",
        "Мат. ожидание — среднее значение случайной величины",
        "Заметка с необязательным аргументом.",
        "в рамке",
        "вложенный термин",
        "$\\myop{x}$",  # formulas are never touched
    ):
        assert fragment in body, fragment
    assert "1cm" not in body  # layout arguments are not text
    assert "[1](#sec:a)" in body  # \ref is handled by Pandoc, not unwrapped
    assert "| $x$ | $p$ |" in body  # the table inside \resizebox survives
    assert "на полях" not in body
    warnings = "\n".join(out.warnings)
    assert "\\defn, \\fbox, \\important, \\note, \\resizebox, \\term" in warnings
    assert "без аргументов пропущены: \\R" in warnings
    assert "\\myop не определены в файле и встречаются в формулах" in warnings
    assert all(w in out.quality["notes"] for w in out.warnings)


NO_NEWTHEOREM_TEX = r"""\documentclass{article}
\newtheorem{thm}{Теор.}
\begin{document}
\begin{theorem}\textbf{Если} $X$ ограничена, то $\mathsf{E}X$ существует.
\end{theorem}

\begin{definition}\textbf{Дисперсия} --- это мера разброса.
\end{definition}

\begin{definition}\emph{Случайной величиной} называется измеримая функция.
\end{definition}

\begin{theorem}[Чебышёв] \emph{Всякая} оценка верна.
\end{theorem}

\begin{thm}[Марков] Текст теоремы Маркова.
\end{thm}

\begin{proof}
\emph{Индукция} по $n$.
\end{proof}

\begin{proof}[Доказательство леммы 2]
Очевидно.
\end{proof}
\end{document}
"""


def test_latex_theorems_declared_elsewhere_keep_their_words(
    tmp_path: Path, settings: Settings, pandoc: Path
) -> None:
    src = tmp_path / "nodefs.tex"
    src.write_text(NO_NEWTHEOREM_TEX, encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "tex")
    DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert "::: theorem\n**Если** $X$ ограничена" in body
    assert "::: definition\n**Дисперсия** — это мера разброса." in body
    assert "::: definition\n*Случайной величиной* называется измеримая функция." in body
    assert '::: {.theorem title="Чебышёв"}\n*Всякая* оценка верна.' in body
    assert '::: {.theorem title="Марков"}\nТекст теоремы Маркова.' in body  # «Теор. 1» stripped
    assert "::: proof\n*Индукция* по $n$.\n:::" in body
    assert '::: {.proof title="Доказательство леммы 2"}\nОчевидно.' in body
    assert "Теорема" not in body and "Определение" not in body and "Теор." not in body

    doc = bl.build_source_doc(body, "D1", pandoc=pandoc)
    first = doc.blocks[0]
    assert first.type == "theorem" and first.text.startswith("Если $X$ ограничена")


FIGURES_TEX = r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
Текст.

\begin{figure}[htbp]
\centering
\includegraphics[width=6cm]{img/sample.png}
\caption{Подпись рисунка}\label{fig:a}
\end{figure}

\begin{figure}[H]
\includegraphics{img/sample.png}
\end{figure}
\end{document}
"""


def test_latex_figure_with_placement(tmp_path: Path, settings: Settings, pandoc: Path) -> None:
    src = tmp_path / "fig.tex"
    src.write_text(FIGURES_TEX, encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "tex", extra={"img/sample.png": IMAGE})
    DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert '![Подпись рисунка](figures/sample.png){#fig:a width="6cm"}' in body
    assert "\n![](figures/sample.png)\n" in body
    assert "<figure" not in body and "<img" not in body and "latex-placement" not in body
    doc = bl.build_source_doc(body, "D1", pandoc=pandoc)
    assert [b.type for b in doc.blocks] == ["paragraph", "figure", "figure"]


def test_latex_included_files_are_not_read(tmp_path: Path, settings: Settings) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("СЕКРЕТНАЯ-СТРОКА-42\n", encoding="utf-8")
    path = secret.as_posix()
    src = tmp_path / "inc.tex"
    src.write_text(
        "\\documentclass{article}\n\\usepackage{verbatim,listings}\n\\begin{document}\n"
        f"Начало.\n\n\\input{{{path}}}\n\n\\verbatiminput{{{path}}}\n\n"
        f"\\lstinputlisting{{{path}}}\n\nКонец.\n\\end{{document}}\n",
        encoding="utf-8",
    )
    ctx = make_ctx(tmp_path, settings, src, "tex")
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    assert "СЕКРЕТНАЯ" not in body and "secret" not in body
    assert body.endswith("Начало.\n\nКонец.\n")
    assert any(
        "Включаемые файлы не прочитаны (\\input, \\lstinputlisting, \\verbatiminput)" in w
        for w in out.warnings
    )


def test_normalize_tex_math() -> None:
    assert normalize_tex_math("\\begin{equation*}x=1\\label{a}\\end{equation*}", display=True) == (
        "x=1"
    )
    assert normalize_tex_math("\\begin{gather}a\\\\b\\end{gather}", display=True) == (
        "\\begin{gathered}\na\\\\b\n\\end{gathered}"
    )
    assert normalize_tex_math("\\begin{alignat}{2}a&=b\\notag\\end{alignat}", display=True) == (
        "\\begin{alignedat}{2}\na&=b\n\\end{alignedat}"
    )
    assert normalize_tex_math("x\\label{q}", display=False) == "x"


# ---------------------------------------------------------------- web


def test_web_page(tmp_path: Path, settings: Settings, pandoc: Path) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.html", "web", sid="W1", url=PAGE_URL)
    out = DocsExtractor().extract(ctx)
    body = body_of(ctx)
    # The page's own <h1> is the title (the site name is not repeated).
    assert body.startswith("# Случайные величины {.source-title}\n")
    assert body.count("Случайные величины") == 1
    assert "Учебный сайт" not in body and "Главная" not in body  # navigation, footer
    assert "править" not in body and "[1]" not in body  # edit links, citation markers
    # Formulas: MediaWiki (data-mw and annotation), dd display, KaTeX, MathJax, \( \) / \[ \].
    for fragment in (
        "буквой $\\xi$.",
        "$(\\Omega ,{\\mathcal {F}},P)$",
        "$$F(x)=\\mathsf{P}(\\xi\\le x)$$.",
        "$\\mathsf{E}\\xi$",
        "$D\\xi = \\mathsf{E}\\xi^2 - (\\mathsf{E}\\xi)^2$",
        "$a+b$",
        "$$c^2 = a^2 + b^2$$",
    ):
        assert fragment in body, fragment
    assert "| Значение | Вероятность |" in body
    assert "](https://example.org/images/cdf.png)" in body
    assert out.quality["formulas"] == 7
    assert any("картинки" in n for n in out.quality["notes"])

    doc = bl.build_source_doc(body, "W1", pandoc=pandoc)
    assert doc.headings[0].is_title
    heads = [(b.title, b.anchor) for b in doc.blocks if b.type == "heading"]
    assert heads == [
        ("Случайные величины", "W1:§0"),
        ("Определение", "W1:§1"),
        ("Пример", "W1:§2"),
    ]
    formulas = [b.text for b in doc.blocks if b.type == "formula"]
    assert formulas == ["$$F(x)=\\mathsf{P}(\\xi\\le x)$$."]


def test_web_title_from_head_when_no_heading() -> None:
    html = (
        "<html><head><title>Заметка о дисперсии | Блог</title>"
        '<meta property="og:site_name" content="Блог"></head><body><article>'
        "<p>Дисперсия показывает разброс значений случайной величины вокруг её среднего "
        "значения и равна среднему квадрату отклонения.</p>"
        "<p>Чем больше дисперсия, тем сильнее значения отклоняются от математического "
        "ожидания, и тем менее точен прогноз.</p></article></body></html>"
    ).encode()
    page = web_markdown(html, url="https://example.org/a")
    assert page.title == "Заметка о дисперсии"
    assert "Дисперсия показывает разброс" in page.markdown
    assert page.formulas == 0 and page.lost_formulas == 0


WEB_MATH_PAGE = """<html><head><title>Формулы</title></head><body><article>
<p>Дисперсия <math><msup><mi>x</mi><mn>2</mn></msup></math> здесь. Длинный текст абзаца про
дисперсию, чтобы trafilatura не выбросила его как слишком короткий фрагмент страницы.</p>
<p>Формула <mjx-container class="MathJax" jax="SVG"><svg></svg><mjx-assistive-mml><math>
<mi>y</mi></math></mjx-assistive-mml></mjx-container> здесь. Ещё длинный текст про
математическое ожидание, чтобы абзац был достаточно длинным для извлечения.</p>
<p>Картинка <img class="tex" alt="x^2+y^2" src="f.png"> здесь, а обычная картинка
<img alt="схема опыта" src="scheme.png"> остаётся картинкой. Ещё немного текста.</p>
<p>Пустая формула <math></math> и SVG без источника <mjx-container class="MathJax"><svg>
</svg></mjx-container> теряются. Текст абзаца, чтобы он остался в извлечении.</p>
</article></body></html>
"""


def test_web_formulas_without_tex_source(pandoc: Path) -> None:
    page = web_markdown(WEB_MATH_PAGE.encode(), url="https://example.org/f", pandoc=pandoc)
    md = page.markdown
    assert "Дисперсия $x^{2}$ здесь." in md  # MathML → TeX by Pandoc
    assert "Формула $y$ здесь." in md  # MathJax 3 container with assistive MathML
    assert "Картинка $x^2+y^2$ здесь" in md  # formula image: TeX in alt
    assert "![схема опыта](https://example.org/scheme.png)" in md
    assert page.formulas == 3 and page.lost_formulas == 2


def test_web_lost_formulas_are_reported(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "page.html"
    src.write_text(WEB_MATH_PAGE, encoding="utf-8")
    ctx = make_ctx(tmp_path, settings, src, "web", sid="W1", url="https://example.org/f")
    out = DocsExtractor().extract(ctx)
    assert out.quality["formulas"] == 3
    assert any("Формул без TeX-источника на странице: 2" in w for w in out.warnings)
    assert out.warnings[0] in out.quality["notes"]


def test_web_fragment_without_html_tag() -> None:
    fragment = "<div><p>Фрагмент страницы про дисперсию случайной величины и её свойства.</p></div>"
    page = web_markdown(fragment.encode())
    assert "Фрагмент страницы про дисперсию" in page.markdown
    with pytest.raises(ExtractError, match="Не удалось разобрать HTML"):
        web_markdown(b"   ")


def test_web_without_main_text_raises(tmp_path: Path, settings: Settings) -> None:
    src = tmp_path / "blank.html"
    src.write_text("<html><head><title>Пусто</title></head><body></body></html>", "utf-8")
    ctx = make_ctx(tmp_path, settings, src, "web", sid="W1", url="https://example.org/blank")
    with pytest.raises(ExtractError, match="основной текст"):
        DocsExtractor().extract(ctx)


def test_web_record_without_file(tmp_path: Path, settings: Settings) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.html", "web", sid="W1", url=PAGE_URL)
    ctx.source.file = None
    with pytest.raises(ExtractError, match="не скачана"):
        DocsExtractor().extract(ctx)


def test_plan_is_cheap(tmp_path: Path, settings: Settings) -> None:
    ctx = make_ctx(tmp_path, settings, DOCS / "sample.tex", "tex")
    plan = DocsExtractor().plan(ctx)
    assert plan.source_id == "D1" and plan.agent_runs == 0 and plan.pages_vision == 0
    assert "LaTeX" in plan.notes[0]
    assert not ctx.out_dir.exists()
