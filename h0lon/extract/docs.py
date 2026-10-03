"""DOCX, Markdown, LaTeX and saved web pages → body.md (docs/ARCHITECTURE.md, «Экстракторы»).

Every kind becomes a Pandoc AST first (docx/latex/markdown readers; web pages through
trafilatura → Markdown), gets the same post-processing (document title as the first
heading, figures next to source.md) and is written as Pandoc Markdown by `blocks`.
No agents are involved. Markdown and LaTeX are read with `--sandbox`: a source never makes
Pandoc read other local files (`\\input`, `\\lstinputlisting` …).
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urljoin, urlparse

import yaml

from h0lon import tools
from h0lon.extract import blocks as bl
from h0lon.extract.model import ExtractContext, ExtractOutput, ExtractPlan
from h0lon.extract.registry import ExtractError

VERSION = "1.1"
FIGURES_DIR = "figures"
BODY_FILE = "body.md"
LATEX_READER = "latex-auto_identifiers"
SANDBOX = ("--sandbox",)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".eps", ".webp", ".bmp")
_SHOWN_NAMES = 10

# ---------------------------------------------------------------- LaTeX vocabulary

# \newtheorem names seen in course materials → block classes of the master.
_ENV_ALIASES: dict[str, str] = {
    "theorem": "theorem",
    "thm": "theorem",
    "teo": "theorem",
    "teorema": "theorem",
    "lemma": "lemma",
    "lem": "lemma",
    "proposition": "proposition",
    "prop": "proposition",
    "utv": "proposition",
    "statement": "proposition",
    "corollary": "corollary",
    "cor": "corollary",
    "sled": "corollary",
    "definition": "definition",
    "defn": "definition",
    "def": "definition",
    "dfn": "definition",
    "opr": "definition",
    "remark": "remark",
    "rem": "remark",
    "zam": "remark",
    "note": "remark",
    "example": "example",
    "ex": "example",
    "exmp": "example",
    "exa": "example",
    "prim": "example",
    "problem": "problem",
    "prob": "problem",
    "exercise": "problem",
    "zad": "problem",
    "task": "problem",
    "solution": "solution",
    "sol": "solution",
    "proof": "proof",
}
# The label Pandoc prints for a theorem environment («Теорема 1», «Опр. 2») without its
# number and final dot → block type (keys are casefolded, see `_label_key`).
_LABEL_TYPES: dict[str, str] = {
    "определение": "definition",
    "теорема": "theorem",
    "лемма": "lemma",
    "утверждение": "proposition",
    "предложение": "proposition",
    "следствие": "corollary",
    "замечание": "remark",
    "пример": "example",
    "задача": "problem",
    "упражнение": "problem",
    "решение": "solution",
    "доказательство": "proof",
    "опр": "definition",
    "теор": "theorem",
    "лем": "lemma",
    "утв": "proposition",
    "предл": "proposition",
    "сл": "corollary",
    "след": "corollary",
    "зам": "remark",
    "прим": "example",
    "упр": "problem",
    "зад": "problem",
    "реш": "solution",
    "док": "proof",
    "definition": "definition",
    "theorem": "theorem",
    "lemma": "lemma",
    "proposition": "proposition",
    "corollary": "corollary",
    "remark": "remark",
    "example": "example",
    "problem": "problem",
    "exercise": "problem",
    "solution": "solution",
    "proof": "proof",
    "def": "definition",
    "defn": "definition",
    "thm": "theorem",
    "lem": "lemma",
    "prop": "proposition",
    "cor": "corollary",
    "rem": "remark",
    "ex": "example",
    "exmp": "example",
    "prob": "problem",
    "sol": "solution",
    "pf": "proof",
}
# Theorem environments used without `\newtheorem` in the file (it lives in an external
# preamble or class) are declared before reading, so Pandoc prints a label and keeps the
# optional title instead of dropping it.
_TYPE_LABEL_RU: dict[str, str] = {
    "definition": "Определение",
    "theorem": "Теорема",
    "lemma": "Лемма",
    "proposition": "Утверждение",
    "corollary": "Следствие",
    "remark": "Замечание",
    "example": "Пример",
    "problem": "Задача",
    "solution": "Решение",
}
_THEOREM_STYLE: dict[str, str] = {
    "theorem": "plain",
    "lemma": "plain",
    "proposition": "plain",
    "corollary": "plain",
    "definition": "definition",
    "example": "definition",
    "problem": "definition",
    "solution": "definition",
    "remark": "remark",
}
_ITALIC_BODY = {"theorem", "lemma", "proposition", "corollary"}
_LAYOUT_ENVS = {"center", "flushleft", "flushright", "minipage", "raggedright"}
_LABEL_NUMBER_RE = re.compile(r"\s+(?:[\w.]*\d[\w.]*|[IVXLCM]+)$")
_QED = ("◻", "□", "∎")  # what Pandoc appends to a proof
_BEGIN_RE = re.compile(r"\\begin\s*\{([^{}]+)\}")
_ENV_DEFINED_RE = re.compile(r"\\(?:newtheorem\*?|(?:re)?newenvironment\*?)\s*\{([^{}]+)\}")
_NEWTHEOREM_LABEL_RE = re.compile(r"\\newtheorem\*?\s*\{[^{}]+\}\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}")


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# ---------------------------------------------------------------- LaTeX: unknown commands

# Commands Pandoc does not know are dropped together with their arguments (macros of an
# external preamble, a package or a class: ingest copies the .tex file alone). They are
# found by a `+raw_tex` read; those whose arguments are text are declared as macros that
# keep the text of the arguments, the rest is reported.
_RAW_FORMATS = ("latex", "tex")
_PROBE = "h0lonprobe"
_RAW_ENV_RE = re.compile(r"\A\s*\\begin\s*\{([^{}]+)\}(.*)\\end\s*\{\1\}\s*\Z", re.DOTALL)
_RAW_CMD_RE = re.compile(r"\\([A-Za-z]+)(\*?)")
_CMD_NAME_RE = re.compile(r"\\([A-Za-z]+)")
_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_TEX_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_INCLUDE_RE = re.compile(
    r"\\(input|include|subfile|import|subimport|includefrom|subincludefrom|verbatiminput"
    r"|lstinputlisting|inputminted|VerbatimInput)(?![A-Za-z])"
)
# Environments whose body is not text (pictures, code, comments): not scanned for commands.
_OPAQUE_ENVS = frozenset(
    {
        "tikzpicture",
        "tikzcd",
        "pgfpicture",
        "circuitikz",
        "pspicture",
        "picture",
        "axis",
        "forest",
        "asy",
        "comment",
        "filecontents",
        "lstlisting",
        "minted",
        "verbatim",
        "Verbatim",
        "dmath",
        "dgroup",
        "IEEEeqnarray",
        "empheq",
    }
)
# Layout, style, counters, labels: arguments are not content, dropping them loses nothing.
_IGNORED_COMMANDS = _words(
    """
    vspace hspace vskip hskip kern setlength addtolength settowidth settoheight newlength
    setcounter addtocounter stepcounter refstepcounter newcounter numberwithin
    noindent indent centering raggedright raggedleft newpage clearpage cleardoublepage
    pagebreak nopagebreak linebreak nolinebreak break nobreak allowbreak newline
    bigskip medskip smallskip vfill hfill hrulefill dotfill par relax protect null strut
    small footnotesize scriptsize tiny normalsize large Large LARGE huge Huge
    normalfont bfseries mdseries itshape upshape slshape scshape rmfamily sffamily ttfamily
    selectfont fontsize linespread onehalfspacing doublespacing singlespacing setstretch
    color pagecolor nopagecolor definecolor colorlet
    pagestyle thispagestyle pagenumbering enlargethispage raggedbottom flushbottom
    maketitle tableofcontents listoffigures listoftables appendix FloatBarrier
    label index glossary pageref nameref cpageref vpageref phantomsection
    addcontentsline addtocontents pdfbookmark hypersetup
    geometry newgeometry restoregeometry captionsetup graphicspath
    selectlanguage setmainlanguage setotherlanguage setdefaultlanguage
    setmainfont setsansfont setmonofont newfontfamily setromanfont
    usetikzlibrary tikzset pgfplotsset theoremstyle newtheorem
    bibliographystyle bibliography addbibresource printbibliography nocite
    frenchspacing nonfrenchspacing sloppy fussy allowdisplaybreaks
    phantom hphantom vphantom makeatletter makeatother columnbreak needspace Needspace
    setbeamertemplate setbeamercolor setbeamerfont usetheme usecolortheme usefonttheme
    useinnertheme useoutertheme beamertemplatenavigationsymbolsempty titlepage
    toprule midrule bottomrule hline cline cmidrule
    input include includeonly excludeonly subfile import subimport includefrom subincludefrom
    verbatiminput lstinputlisting inputminted VerbatimInput
    """
)
# Boxes whose last argument is the content and the others are sizes or colours.
_KEEP_LAST_ARG = _words(
    """
    fbox framebox parbox raisebox resizebox scalebox rotatebox reflectbox colorbox fcolorbox
    adjustbox captionof shortstack ovalbox Ovalbox doublebox shadowbox tcbox hypertarget
    uline uuline uwave sout xout hl ul st so caps highlight todo
    """
)
# Math-only commands met outside math (inside an unknown math environment): never declared.
_MATH_COMMANDS = _words(
    """
    frac dfrac tfrac cfrac sqrt binom dbinom tbinom mathbb mathcal mathrm mathbf mathsf
    mathit mathfrak mathscr mathtt boldsymbol bm operatorname overline underline hat bar
    tilde vec dot ddot widehat widetilde overrightarrow overleftarrow overbrace underbrace
    stackrel overset underset xrightarrow xleftarrow pmod bmod left right sum prod int
    lim text textrm
    """
)
# Text symbols Pandoc drops without `raw_tex`.
_SYMBOLS: dict[str, str] = {
    "textendash": "–",
    "textemdash": "—",
    "og": "«",
    "fg": "»",
    "guillemotleft": "«",
    "guillemotright": "»",
    "textquotedblleft": "“",
    "textquotedblright": "”",
    "textellipsis": "…",
    "No": "№",
    "textnumero": "№",
    "textdegree": "°",
    "textpm": "±",
    "texttimes": "×",
    "quad": " ",
    "qquad": " ",
    "enspace": " ",
    "nobreakspace": "\N{NO-BREAK SPACE}",
}

_MATH_ENV_RE = re.compile(
    r"^\s*\\begin\{(equation|align|alignat|gather|multline|eqnarray|flalign)(\*?)\}"
    r"(\{[^}]*\})?(.*?)\\end\{\1\2\}\s*$",
    re.DOTALL,
)
_MATH_ENV_TARGET = {
    "equation": None,
    "align": "aligned",
    "alignat": "alignedat",
    "gather": "gathered",
    "multline": "gathered",
    "eqnarray": "aligned",
    "flalign": "aligned",
}
_MATH_LABEL_RE = re.compile(r"\\label\{[^{}]*\}")
_MATH_NONUMBER_RE = re.compile(r"\\(?:nonumber|notag)(?![A-Za-z])")

# ---------------------------------------------------------------- web vocabulary

_TEX_TEXT_RE = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"h0lonmath(i|d)(\d+)z")
_DISPLAYSTYLE_RE = re.compile(r"^\{\\displaystyle\s*(.*)\}$", re.DOTALL)
_PUNCT_TEXT_RE = re.compile(r"^[\s.,;:]*$")
_TITLE_SEPARATORS = (" — ", " – ", " - ", " | ", " · ", " :: ")
# Page furniture that trafilatura keeps (MediaWiki edit links, citation markers, back-links).
_WEB_JUNK_CLASSES = ("mw-editsection", "mw-cite-backlink", "mw-jump-link", "noprint")
# Formula images: TeX in `alt` (old MediaWiki, blogs with codecogs/mimetex/WordPress LaTeX).
_FORMULA_IMG_CLASSES = frozenset(
    {
        "tex",
        "latex",
        "math",
        "formula",
        "equation",
        "mwe-math-fallback-image-inline",
        "mwe-math-fallback-image-display",
    }
)
_FORMULA_IMG_SOURCES = ("codecogs.com", "/math/render/", "latex.php", "mimetex", "mathtex.cgi")
_QUEUED_ATTR = "data-h0lon-mathml"
_MATHML_MARK = "h0lonmathml"


def _has_class(cls: str) -> str:
    return f'contains(concat(" ", normalize-space(@class), " "), " {cls} ")'


# ---------------------------------------------------------------- extractor


class DocsExtractor:
    """docx / md / tex / web → body.md (deterministic, Pandoc and trafilatura)."""

    kinds: tuple[str, ...] = ("docx", "md", "tex", "web")
    version: str = VERSION

    def plan(self, ctx: ExtractContext) -> ExtractPlan:
        kind = ctx.source.kind
        how = {
            "docx": "Pandoc: DOCX → Markdown, картинки в figures/",
            "md": "Pandoc: Markdown → Markdown",
            "tex": "Pandoc: LaTeX → Markdown (макросы раскрываются, теоремы → блоки, "
            "неизвестные команды → текст их аргументов)",
            "web": "trafilatura: основной текст страницы → Markdown, формулы → LaTeX",
        }.get(kind, "Pandoc")
        notes = [how]
        if not ctx.source.file:
            notes.append("файл источника не указан — извлечение завершится ошибкой")
        return ExtractPlan(source_id=ctx.source.id, notes=notes)

    def extract(self, ctx: ExtractContext) -> ExtractOutput:
        kind = ctx.source.kind
        if kind not in self.kinds:
            raise ExtractError(f"DocsExtractor не обрабатывает вид «{kind}»")
        src = source_path(ctx)
        pandoc = tools.find_pandoc(ctx.settings.render.pandoc)
        if pandoc is None:
            raise ExtractError(
                "Pandoc не найден: установите пакет pypandoc-binary (uv sync) или укажите "
                "render.pandoc в h0lon.toml"
            )
        out_dir = ctx.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = out_dir / FIGURES_DIR
        if figures.exists():
            shutil.rmtree(figures)  # generated by the previous extraction of this source
        warnings: list[str] = []
        notes: list[str] = []

        title: list[Any] | None = None
        try:
            if kind == "docx":
                ctx.emit(f"{ctx.source.id}: Pandoc читает DOCX")
                doc = bl.read_ast(
                    pandoc,
                    path=src.resolve(),
                    reader="docx",
                    extra_args=[f"--extract-media={FIGURES_DIR}"],
                    cwd=out_dir,
                )
                _fix_docx(doc)
                _normalize_local_paths(doc)
            elif kind == "md":
                ctx.emit(f"{ctx.source.id}: Pandoc читает Markdown")
                text = _read_text(src, notes)
                doc = read_markdown_source(pandoc, text, cwd=src.parent, warnings=warnings)
                _localize_images(doc, src.parent, figures, warnings)
            elif kind == "tex":
                ctx.emit(f"{ctx.source.id}: Pandoc читает LaTeX")
                text = _read_text(src, notes)
                doc = read_latex_source(
                    pandoc, text, cwd=src.parent, notes=notes, warnings=warnings
                )
                _localize_images(doc, src.parent, figures, warnings)
            else:
                ctx.emit(f"{ctx.source.id}: trafilatura извлекает основной текст страницы")
                page = web_markdown(src.read_bytes(), url=ctx.source.url, pandoc=pandoc)
                doc = bl.read_ast(pandoc, text=page.markdown, reader=bl.READER_FORMAT)
                _absolutize_images(doc, ctx.source.url)
                if page.title:
                    title = _web_title(doc, page.title)
                if page.formulas:
                    notes.append(f"формул со страницы перенесено в LaTeX: {page.formulas}")
                if page.lost_formulas:
                    warnings.append(
                        f"Формул без TeX-источника на странице: {page.lost_formulas} — они "
                        "потеряны или перенесены искажённым текстом, сверьте со страницей"
                    )
                if any(n.get("t") == "Image" for n in bl.walk_elements(doc["blocks"])):
                    notes.append("картинки страницы не скачаны — ссылки ведут на сайт")
        except bl.BlocksError as exc:
            raise ExtractError(str(exc)) from exc

        if title is None:
            title = _meta_inlines(doc.get("meta", {}).get("title"))
        _prepend_front(doc, title)

        stats = _stats(doc)
        if not stats["chars"] and not stats["figures"]:
            raise ExtractError(_empty_message(kind))
        try:
            body = bl.write_markdown(doc, pandoc, cwd=out_dir)
        except bl.BlocksError as exc:
            raise ExtractError(str(exc)) from exc
        body_path = out_dir / BODY_FILE
        bl.atomic_write_text(body_path, body)

        for w in warnings:
            notes.append(w)
        quality: dict[str, Any] = dict(stats)
        if notes:
            quality["notes"] = notes
        return ExtractOutput(body_md=body_path, quality=quality, warnings=warnings)


def source_path(ctx: ExtractContext) -> Path:
    rec = ctx.source
    if not rec.file:
        if rec.url:
            raise ExtractError(
                f"Страница {rec.url} не скачана: в topic.yaml нет файла источника "
                "(добавьте ссылку заново через h0lon add)"
            )
        raise ExtractError("В topic.yaml у источника нет файла")
    path = Path(rec.file)
    if not path.is_absolute():
        path = ctx.topic_dir / path
    if not path.is_file():
        raise ExtractError(f"Файл источника не найден: {path}")
    return path


def _empty_message(kind: str) -> str:
    if kind == "web":
        return (
            "На странице не найден основной текст (trafilatura вернула пустой результат): "
            "возможно, страница строится скриптами или требует входа. Сохраните её из "
            "браузера как HTML и добавьте файлом"
        )
    return "В документе не найдено текста: проверьте, что файл не пустой и не повреждён"


def _read_text(path: Path, notes: list[str]) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        notes.append("файл не в UTF-8 — прочитан как Windows-1251, проверьте буквы")
        return data.decode("cp1251", errors="replace")


def _shown(names: list[str], *, prefix: str = "") -> str:
    shown = ", ".join(prefix + n for n in names[:_SHOWN_NAMES])
    if len(names) > _SHOWN_NAMES:
        shown += f" и ещё {len(names) - _SHOWN_NAMES}"
    return shown


# ---------------------------------------------------------------- Markdown

_FENCED_CODE_RE = re.compile(r"^(`{3,}|~{3,}).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_BACKSLASH_DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_TEXLIKE_RE = re.compile(r"\\[A-Za-z]|[=^_]|\n")
_TITLE_LINE_RE = re.compile(r"^title\s*:\s*(.+?)\s*$", re.MULTILINE)
_FRONT_KEYS = ("title", "subtitle", "abstract")


def uses_backslash_math(text: str) -> bool:
    """True if formulas are written as `\\( … \\)` / `\\[ … \\]` (chat exports, notes).

    `\\[1\\]` is also how Pandoc escapes literal brackets, so a `\\[ … \\]` pair counts only
    when its content looks like TeX. Code is not looked at.
    """
    text = _INLINE_CODE_RE.sub("", _FENCED_CODE_RE.sub("", text))
    if "\\(" in text and "\\)" in text:
        return True
    return any(_TEXLIKE_RE.search(m.group(1)) for m in _BACKSLASH_DISPLAY_RE.finditer(text))


def read_markdown_source(
    pandoc: Path, text: str, *, cwd: Path, warnings: list[str]
) -> dict[str, Any]:
    """AST of a Markdown source; front matter is split off here, not by Pandoc.

    Pandoc reads any `---` … `---` fragment that parses as YAML as metadata and drops it
    from the text, and stops on broken YAML. Only a front matter at the top is metadata;
    broken YAML there becomes a warning (the title is still taken from its `title:` line).
    """
    fm = bl.parse_front_matter(text)
    reader = bl.READER_FORMAT
    if uses_backslash_math(fm.body):
        reader += "+tex_math_single_backslash"
    values: dict[str, str] = {}
    if fm.error:
        m = _TITLE_LINE_RE.search(fm.raw)
        title = m.group(1).strip().strip("\"'").strip() if m else ""
        if title:
            values["title"] = title
        warnings.append(
            f"YAML-шапка файла (front matter) не разобрана: {fm.error} — метаданные пропущены"
            + (", название взято из строки title" if title else "")
        )
    else:
        for key in _FRONT_KEYS:
            value = fm.data.get(key)
            if isinstance(value, str | int | float) and str(value).strip():
                values[key] = str(value)
    doc = bl.read_ast(pandoc, text=fm.body, reader=reader, extra_args=SANDBOX, cwd=cwd)
    doc["meta"] = _front_meta(pandoc, values, reader) if values else {}
    return doc


def _front_meta(pandoc: Path, values: dict[str, str], reader: str) -> dict[str, Any]:
    """Pandoc metadata (Markdown inlines/blocks) of title, subtitle and abstract strings."""
    yaml_text = yaml.safe_dump(values, allow_unicode=True, sort_keys=False, width=1000)
    doc = bl.read_ast(
        pandoc,
        text=f"---\n{yaml_text}---\n",
        reader=reader + "+yaml_metadata_block",
        extra_args=SANDBOX,
    )
    meta = doc.get("meta")
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------- AST helpers


def _str_inlines(text: str) -> list[Any]:
    out: list[Any] = []
    for i, word in enumerate(text.split()):
        if i:
            out.append(bl.el("Space"))
        out.append(bl.el("Str", word))
    return out


def _meta_inlines(value: Any) -> list[Any] | None:
    if not isinstance(value, dict):
        return None
    tag, c = value.get("t"), value.get("c")
    if tag == "MetaInlines":
        return c or None
    if tag == "MetaString":
        return _str_inlines(c) or None
    if tag == "MetaBlocks":
        inlines: list[Any] = []
        for b in c:
            if b.get("t") in ("Para", "Plain"):
                inlines += ([bl.el("Space")] if inlines else []) + b["c"]
        return inlines or None
    return None


def _meta_blocks(value: Any) -> list[Any]:
    if not isinstance(value, dict):
        return []
    if value.get("t") == "MetaBlocks":
        return list(value.get("c") or [])
    inlines = _meta_inlines(value)
    return [bl.el("Para", inlines)] if inlines else []


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _prepend_front(doc: dict[str, Any], title: list[Any] | None) -> None:
    """Document title as the first heading (class source-title), subtitle and abstract."""
    meta = doc.get("meta") or {}
    blocks: list[Any] = doc["blocks"]
    front: list[Any] = []
    if title:
        first = next((b for b in blocks if bl.classify(b) is not None), None)
        if (
            first is not None
            and first.get("t") == "Header"
            and _norm(bl.inline_text(first["c"][2])) == _norm(bl.inline_text(title))
        ):
            classes = first["c"][1][1]
            if bl.TITLE_CLASS not in classes:
                classes.append(bl.TITLE_CLASS)
        else:
            front.append(bl.el("Header", [1, ["", [bl.TITLE_CLASS], []], title]))
    subtitle = _meta_inlines(meta.get("subtitle"))
    if subtitle:
        front.append(bl.el("Para", [bl.el("Emph", subtitle)]))
    front += _meta_blocks(meta.get("abstract"))
    doc["blocks"] = front + blocks


def _stats(doc: dict[str, Any]) -> dict[str, int]:
    figures = tables = formulas = 0
    for node in bl.walk_elements(doc["blocks"]):
        tag = node.get("t")
        if tag == "Image":
            figures += 1
        elif tag == "Table":
            tables += 1
        elif tag == "Math":
            formulas += 1
    return {
        "chars": len(bl.blocks_text(doc["blocks"]).strip()),
        "figures": figures,
        "tables": tables,
        "formulas": formulas,
    }


def _image_targets(doc: dict[str, Any]) -> list[list[Any]]:
    """The [url, title] pairs of all images (mutable)."""
    return [n["c"][2] for n in bl.walk_elements(doc["blocks"]) if n.get("t") == "Image"]


def _is_remote(target: str) -> bool:
    scheme = urlparse(target).scheme.lower()
    return scheme in ("http", "https", "ftp", "data") or target.startswith("//")


def _normalize_local_paths(doc: dict[str, Any]) -> None:
    for target in _image_targets(doc):
        if not _is_remote(target[0]):
            target[0] = target[0].replace("\\", "/")


def _localize_images(
    doc: dict[str, Any], base_dir: Path, figures: Path, warnings: list[str]
) -> None:
    """Copy images that lie next to the source file into figures/ and point at the copies.

    Only paths inside the source's directory are followed; ingest copies the document
    alone, so relative images of the original folder are usually missing.
    """
    missing: list[str] = []
    copied: dict[Path, str] = {}
    base = base_dir.resolve()
    for target in _image_targets(doc):
        url = target[0]
        if not url or _is_remote(url):
            continue
        found = _find_local(base, unquote(url))
        if found is None:
            if url not in missing:
                missing.append(url)
            continue
        if found not in copied:
            figures.mkdir(parents=True, exist_ok=True)
            name = found.name
            stem, suffix, i = found.stem, found.suffix, 1
            while (figures / name).exists():
                i += 1
                name = f"{stem}-{i}{suffix}"
            shutil.copy2(found, figures / name)
            copied[found] = f"{FIGURES_DIR}/{name}"
        target[0] = copied[found]
    if missing:
        shown = ", ".join(missing[:5]) + (f" и ещё {len(missing) - 5}" if len(missing) > 5 else "")
        warnings.append(
            f"Картинки не найдены рядом с файлом источника ({len(missing)}): {shown} — "
            "добавьте их в папку темы вручную или вставьте в документ"
        )


def _find_local(base: Path, rel: str) -> Path | None:
    rel_path = Path(rel)
    if rel_path.is_absolute() or rel_path.drive:
        return None
    candidate = (base / rel_path).resolve()
    if not candidate.is_relative_to(base):
        return None
    options = [candidate] if candidate.suffix else []
    options += [candidate.with_name(candidate.name + ext) for ext in _IMAGE_EXTS]
    return next((p for p in options if p.is_file()), None)


def _absolutize_images(doc: dict[str, Any], page_url: str | None) -> None:
    if not page_url:
        return
    for target in _image_targets(doc):
        if target[0] and not target[0].startswith("data:"):
            target[0] = urljoin(page_url, target[0])


# ---------------------------------------------------------------- DOCX


def _fix_docx(doc: dict[str, Any]) -> None:
    """Drop heading bookmarks nobody links to and Word's internal links (`#_Toc…`)."""

    def unwrap(inlines: list[Any]) -> list[Any]:
        out: list[Any] = []
        for x in inlines:
            if x.get("t") == "Link" and x["c"][2][0].startswith("#_"):
                out += x["c"][1]
            else:
                out.append(x)
        return out

    doc["blocks"] = bl.map_inline_lists(doc["blocks"], unwrap)
    targets = {
        n["c"][2][0][1:]
        for n in bl.walk_elements(doc["blocks"])
        if n.get("t") == "Link" and n["c"][2][0].startswith("#")
    }
    for node in bl.walk_elements(doc["blocks"]):
        if node.get("t") == "Header" and node["c"][1][0] not in targets:
            node["c"][1][0] = ""


# ---------------------------------------------------------------- LaTeX


def read_latex_source(
    pandoc: Path, text: str, *, cwd: Path, notes: list[str], warnings: list[str]
) -> dict[str, Any]:
    """AST of a LaTeX source, prepared for the master (see the module docstring).

    Before the final read: theorem environments used without `\\newtheorem` are declared,
    and commands Pandoc does not know (an external preamble) are declared as macros that
    keep the text of their arguments. Included files are never read (`--sandbox`).
    """
    plain = _TEX_COMMENT_RE.sub("", text)
    includes = sorted({m.group(1) for m in _INCLUDE_RE.finditer(plain)})
    if includes:
        warnings.append(
            "Включаемые файлы не прочитаны (" + _shown(includes, prefix="\\") + "): "
            "из соображений безопасности читается только сам .tex — вставьте их содержимое "
            "в файл или добавьте отдельными источниками"
        )
    env_prelude, labels = _theorem_prelude(plain)
    labels |= {_label_key(m) for m in _NEWTHEOREM_LABEL_RE.findall(plain)}
    scan = _scan_latex(pandoc, env_prelude + text, cwd=cwd)
    macro_prelude = _command_prelude(pandoc, scan, cwd=cwd, notes=notes, warnings=warnings)
    doc = bl.read_ast(
        pandoc,
        text=env_prelude + macro_prelude + text,
        reader=LATEX_READER,
        extra_args=SANDBOX,
        cwd=cwd,
    )
    _fix_latex(doc, labels)
    return doc


def _theorem_prelude(text: str) -> tuple[str, set[str]]:
    """(`\\newtheorem` lines for known environments the file uses but does not declare,
    label keys of those declarations)."""
    defined = set(_ENV_DEFINED_RE.findall(text))
    by_style: dict[str, list[str]] = {}
    labels: set[str] = set()
    for name in dict.fromkeys(n.strip() for n in _BEGIN_RE.findall(text)):
        typ = _ENV_ALIASES.get(name.rstrip("*").lower())
        if typ is None or typ == "proof" or name in defined:
            continue  # proof is built into Pandoc
        label = _TYPE_LABEL_RU[typ]
        by_style.setdefault(_THEOREM_STYLE[typ], []).append(f"\\newtheorem{{{name}}}{{{label}}}")
        labels.add(_label_key(label))
    if not by_style:
        return "", labels
    lines: list[str] = []
    for style in ("plain", "definition", "remark"):
        if style in by_style:
            lines += [f"\\theoremstyle{{{style}}}", *by_style[style]]
    lines.append("\\theoremstyle{plain}")  # the file's own declarations keep the default
    return "\n".join(lines) + "\n", labels


@dataclass
class _CommandUse:
    arg_counts: Counter[int] = field(default_factory=Counter)
    optional: int = 0  # most optional arguments seen in one call
    star: bool = False


@dataclass
class _LatexScan:
    uses: dict[str, _CommandUse] = field(default_factory=dict)
    math: set[str] = field(default_factory=set)  # command names used inside formulas


def _group_end(text: str, start: int, close: str) -> int | None:
    """Index of the bracket closing the group opened at `start` (`{` or `[`)."""
    depth = 0
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "%":
            nl = text.find("\n", i)
            i = len(text) if nl < 0 else nl + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if close == "}" and depth == 0:
                return i
            depth -= 1
        elif ch == close and depth == 0:
            return i
        i += 1
    return None


def parse_raw_command(raw: str) -> tuple[str, bool, int, int] | None:
    """(name, star, optional arguments, brace arguments) of `\\name*[…]{…}{…}`."""
    m = _RAW_CMD_RE.match(raw.lstrip())
    if not m:
        return None
    text = raw.lstrip()
    i, n = m.end(), len(text)
    optional = args = 0
    while True:
        j = i
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j < n and text[j] == "[" and not args:
            end = _group_end(text, j, "]")
            if end is None:
                break
            optional, i = optional + 1, end + 1
        elif j < n and text[j] == "{":
            end = _group_end(text, j, "}")
            if end is None:
                break
            args, i = args + 1, end + 1
        else:
            break
    return m.group(1), bool(m.group(2)), optional, args


def _scan_latex(pandoc: Path, text: str, *, cwd: Path) -> _LatexScan:
    """Commands Pandoc keeps only as raw LaTeX (unknown to it) and command names of formulas.

    Bodies of environments Pandoc does not know come back as one raw block; they are
    scanned once more (after the file's preamble, so its macros stay known).
    """
    scan = _LatexScan()
    m = _DOCUMENT_RE.search(text)
    preamble = text[: m.start()] if m else ""
    batch = text
    for depth in range(2):
        doc = bl.read_ast(
            pandoc, text=batch, reader=LATEX_READER + "+raw_tex", extra_args=SANDBOX, cwd=cwd
        )
        bodies: list[str] = []
        for node in bl.walk_elements(doc):
            tag = node.get("t")
            if tag == "Math":
                scan.math.update(_CMD_NAME_RE.findall(node["c"][1]))
            elif tag in ("RawInline", "RawBlock") and node["c"][0] in _RAW_FORMATS:
                raw = node["c"][1]
                env = _RAW_ENV_RE.match(raw)
                if env is not None:
                    name = env.group(1).strip().rstrip("*")
                    if name not in _OPAQUE_ENVS and "\\" in env.group(2):
                        bodies.append(env.group(2))
                    continue
                parsed = parse_raw_command(raw)
                if parsed is None:
                    continue
                name, star, optional, args = parsed
                if name in ("begin", "end"):
                    continue
                use = scan.uses.setdefault(name, _CommandUse())
                use.arg_counts[args] += 1
                use.optional = max(use.optional, optional)
                use.star = use.star or star
        if not bodies or depth:
            break
        batch = f"{preamble}\\begin{{document}}\n" + "\n\n".join(bodies) + "\n\\end{document}\n"
    return scan


def _probe_kept(pandoc: Path, calls: dict[str, tuple[int, bool]], *, cwd: Path) -> set[str]:
    """Commands whose arguments Pandoc keeps without `raw_tex` (`\\ref`, `\\mbox` …)."""
    if not calls:
        return set()
    parts = []
    for i, (name, (args, optional)) in enumerate(calls.items()):
        call = f"\\{name}" + (f"[{_PROBE}{i}x0]" if optional else "")
        call += "".join(f"{{{_PROBE}{i}x{k + 1}}}" for k in range(args))
        parts.append(call)
    text = "\\begin{document}\n" + "\n\n".join(parts) + "\n\\end{document}\n"
    try:
        doc = bl.read_ast(pandoc, text=text, reader=LATEX_READER, extra_args=SANDBOX, cwd=cwd)
    except bl.BlocksError:
        return set(calls)  # cannot tell: leave them to Pandoc
    dumped = json.dumps(doc, ensure_ascii=False)
    return {name for i, name in enumerate(calls) if re.search(rf"{_PROBE}{i}x\d", dumped)}


def _common_count(counts: Counter[int]) -> int:
    best = max(counts.values())
    return max(n for n, c in counts.items() if c == best)


def _command_prelude(
    pandoc: Path, scan: _LatexScan, *, cwd: Path, notes: list[str], warnings: list[str]
) -> str:
    """`\\newcommand` lines for unknown commands; what could not be kept is reported."""
    defs: list[str] = []
    calls: dict[str, tuple[int, bool]] = {}
    bare: list[str] = []
    in_math: list[str] = []
    odd: list[str] = []
    for name in sorted(scan.uses):
        use = scan.uses[name]
        if name in _IGNORED_COMMANDS or name in _MATH_COMMANDS:
            continue
        if name in _SYMBOLS:
            defs.append(f"\\newcommand{{\\{name}}}{{{_SYMBOLS[name]}}}")
            continue
        args = _common_count(use.arg_counts)
        if args == 0:
            bare.append(name)
        elif name in scan.math:
            in_math.append(name)
        elif use.star or use.optional > 1:
            odd.append(name)
        else:
            calls[name] = (args, use.optional == 1)
    kept = _probe_kept(pandoc, calls, cwd=cwd)
    unwrapped: list[str] = []
    for name, (args, optional) in calls.items():
        if name in kept:
            continue
        shift = 1 if optional else 0
        if name in _KEEP_LAST_ARG:
            body = f"#{args + shift}"
        else:
            body = " --- ".join(f"#{k + shift}" for k in range(1, args + 1))
        head = f"[{args + 1}][]" if optional else f"[{args}]"
        defs.append(f"\\newcommand{{\\{name}}}{head}{{{body}}}")
        unwrapped.append(name)
    if unwrapped:
        message = (
            "Команды LaTeX, не определённые в файле (видимо, из внешней преамбулы или пакета): "
            + _shown(unwrapped, prefix="\\")
            + " — перенесён только текст их аргументов, оформление потеряно"
        )
        warnings.append(message)
    if bare:
        warnings.append(
            "Неизвестные команды без аргументов пропущены: "
            + _shown(bare, prefix="\\")
            + " — проверьте, не стоял ли за ними текст"
        )
    if in_math:
        warnings.append(
            "Команды " + _shown(in_math, prefix="\\") + " не определены в файле и встречаются "
            "в формулах: в тексте они пропущены вместе с аргументами — определите их в файле"
        )
    if odd:
        warnings.append(
            "Неизвестные команды со звёздочкой или несколькими необязательными аргументами "
            "пропущены вместе с текстом: " + _shown(odd, prefix="\\")
        )
    return "\n".join(defs) + "\n" if defs else ""


def _fix_latex(doc: dict[str, Any], labels: set[str] | frozenset[str] = frozenset()) -> None:
    doc["blocks"] = _fix_latex_blocks(doc["blocks"], labels)
    for node in bl.walk_elements(doc["blocks"]):
        tag = node.get("t")
        if tag == "Math":
            node["c"][1] = normalize_tex_math(
                node["c"][1], display=node["c"][0]["t"] == "DisplayMath"
            )
        elif tag == "Figure":
            # `\begin{figure}[htbp]` gives latex-placement, which the Markdown writer cannot
            # put on an implicit figure: it would fall back to raw HTML.
            attr = node["c"][0]
            attr[2] = [kv for kv in attr[2] if not kv[0].startswith("latex-")]


def normalize_tex_math(tex: str, *, display: bool) -> str:
    """Formula as the master expects it: no numbering environments, labels or \\nonumber."""
    tex = _MATH_NONUMBER_RE.sub("", _MATH_LABEL_RE.sub("", tex))
    if display:
        m = _MATH_ENV_RE.match(tex)
        if m:
            env, _star, arg, inner = m.groups()
            target = _MATH_ENV_TARGET[env]
            inner = inner.strip()
            if target is None:
                tex = inner
            else:
                tex = f"\\begin{{{target}}}{arg or ''}\n{inner}\n\\end{{{target}}}"
    return tex.strip()


def _fix_latex_blocks(blocks: list[Any], labels: set[str] | frozenset[str]) -> list[Any]:
    out: list[Any] = []
    for b in blocks:
        tag = b.get("t")
        if tag == "Div":
            attr, inner = b["c"]
            classes = attr[1]
            if any(c in _LAYOUT_ENVS for c in classes) and not attr[0]:
                out += _fix_latex_blocks(inner, labels)
                continue
            b["c"][1] = _fix_latex_blocks(inner, labels)
            _fix_theorem_div(b, labels)
            out.append(b)
        elif tag == "Para":
            out += _split_display_math(b["c"])
        elif tag == "BlockQuote":
            b["c"] = _fix_latex_blocks(b["c"], labels)
            out.append(b)
        elif tag == "CodeBlock" and not b["c"][1].strip():
            continue  # \lstinputlisting of a file that is not read (sandbox)
        else:
            out.append(b)
    return out


def _split_display_math(inlines: list[Any]) -> list[Any]:
    """A paragraph with display formulas inside → text / formula / text paragraphs."""
    if not any(x.get("t") == "Math" and x["c"][0]["t"] == "DisplayMath" for x in inlines):
        return [bl.el("Para", inlines)]
    paras: list[list[Any]] = []
    current: list[Any] = []
    i = 0
    while i < len(inlines):
        x = inlines[i]
        if x.get("t") == "Math" and x["c"][0]["t"] == "DisplayMath":
            paras.append(current)
            formula = [x]
            # Punctuation that closes the formula (", где", ".") stays with it.
            if i + 1 < len(inlines) and inlines[i + 1].get("t") == "Str":
                m = re.match(r"^[.,;:]+", inlines[i + 1]["c"])
                if m:
                    formula.append(bl.el("Str", m.group(0)))
                    rest = inlines[i + 1]["c"][m.end() :]
                    inlines = [
                        *inlines[: i + 1],
                        *([bl.el("Str", rest)] if rest else []),
                        *inlines[i + 2 :],
                    ]
            paras.append(formula)
            current = []
        else:
            current.append(x)
        i += 1
    paras.append(current)
    out = []
    for p in paras:
        while p and p[0].get("t") in ("Space", "SoftBreak", "LineBreak"):
            p = p[1:]
        while p and p[-1].get("t") in ("Space", "SoftBreak", "LineBreak"):
            p = p[:-1]
        if p:
            out.append(bl.el("Para", p))
    return out


def _env_type(classes: list[str]) -> str | None:
    for c in classes:
        key = c.rstrip("*").lower()
        if key in _ENV_ALIASES:
            return _ENV_ALIASES[key]
    return None


def _fix_theorem_div(div: dict[str, Any], labels: set[str] | frozenset[str] = frozenset()) -> None:
    """Pandoc's theorem Div (`**Теорема 1** (Название). *текст*`) → `.theorem title=…`."""
    attr, inner = div["c"]
    classes: list[str] = attr[1]
    typ = _env_type(classes)
    title: str | None = None
    if inner and inner[0].get("t") == "Para":
        stripped = _strip_label(inner[0]["c"], typ=typ, labels=labels)
        if stripped is not None:
            label_type, title, rest = stripped
            typ = typ or label_type
            if rest:
                inner[0] = bl.el("Para", rest)
            else:
                inner.pop(0)
    if typ is None:
        return
    attr[1] = [typ] + [c for c in classes if _env_type([c]) is None and c != typ]
    if title and not any(k == "title" for k, _v in attr[2]):
        attr[2].append(["title", title])
    if typ == "proof" and inner and inner[-1].get("t") == "Para":
        last = inner[-1]["c"]
        while last and (
            last[-1].get("t") in ("Space", "SoftBreak")
            or (last[-1].get("t") == "Str" and last[-1]["c"].strip(" \N{NO-BREAK SPACE}") in _QED)
        ):
            last.pop()
        if not last:
            inner.pop()
    if typ in _ITALIC_BODY:  # theorem style "plain" sets the whole body in italics
        for j, b in enumerate(inner):
            if b.get("t") == "Para" and len(b["c"]) == 1 and b["c"][0].get("t") == "Emph":
                inner[j] = bl.el("Para", b["c"][0]["c"])


def _label_key(label: str) -> str:
    """«Теорема 1.2.» → «теорема», «Опр. 3» → «опр»: a label without number and final dot."""
    text = re.sub(r"\s+", " ", label).strip().rstrip(".").strip()
    text = _LABEL_NUMBER_RE.sub("", text)
    return text.rstrip(".").strip().casefold()


def _strip_label(
    inlines: list[Any], *, typ: str | None, labels: set[str] | frozenset[str] = frozenset()
) -> tuple[str | None, str | None, list[Any]] | None:
    """(type from the label, title, remaining inlines) if the paragraph opens with a label.

    Pandoc prints theorem environments as `**Теорема 1** (Название). …` and proofs as
    `*Proof.* …` or `*Доказательство леммы 2.* …`. Only a bold/italic run that is such a
    label counts: one declared in the file (`labels`), a word of `_LABEL_TYPES` (of the
    environment's own type when that is known) or, in a proof, an italic run ending with a
    dot. Anything else is the author's text and stays.
    """
    if not inlines or inlines[0].get("t") not in ("Strong", "Emph"):
        return None
    label = bl.inline_text(inlines[0]["c"]).strip()
    key = _label_key(label)
    label_type = _LABEL_TYPES.get(key)
    proof_title = (
        typ == "proof" and inlines[0].get("t") == "Emph" and label.endswith(".") and bool(key)
    )
    if not (key in labels or (label_type is not None and typ in (None, label_type)) or proof_title):
        return None
    rest = inlines[1:]
    i = 0
    while i < len(rest) and rest[i].get("t") in ("Space", "SoftBreak"):
        i += 1
    title: str | None = None
    if i < len(rest) and rest[i].get("t") == "Str" and rest[i]["c"].startswith("("):
        depth, j = 0, i
        while j < len(rest):
            if rest[j].get("t") == "Str":
                depth += rest[j]["c"].count("(") - rest[j]["c"].count(")")
                if depth <= 0:
                    break
            j += 1
        if j < len(rest):
            text = bl.inline_text(rest[i : j + 1]).strip()
            mt = re.match(r"^\((.*)\)\.?$", text, re.DOTALL)
            if mt:
                title = mt.group(1).strip() or None
                i = j + 1
    if title is None and proof_title and label_type is None and key not in labels:
        title = label.rstrip(".").strip() or None  # \begin{proof}[Доказательство леммы 2]
    if i < len(rest) and rest[i].get("t") == "Str" and rest[i]["c"] == ".":
        i += 1
    while i < len(rest) and rest[i].get("t") in ("Space", "SoftBreak"):
        i += 1
    return label_type, title, rest[i:]


# ---------------------------------------------------------------- web


class WebPage(NamedTuple):
    markdown: str  # main text of the page
    title: str | None  # page title without the site name
    formulas: int  # formulas carried over as LaTeX
    lost_formulas: int = 0  # formulas without a TeX source that could not be converted


def web_markdown(html: bytes, *, url: str | None = None, pandoc: Path | None = None) -> WebPage:
    """Main text of a saved HTML page as Markdown, its title and formula counts.

    Formulas (MediaWiki, KaTeX, MathML, MathJax, formula images with TeX in `alt`,
    `\\(…\\)`/`\\[…\\]`/`$$…$$` in text) are replaced by placeholders before trafilatura and
    restored as `$…$` / `$$…$$`; MathML without a TeX annotation is converted by Pandoc.
    A fragment without `<html>` is parsed as a page body.
    """
    import trafilatura
    from trafilatura.utils import load_html

    tree = load_html(html)
    if tree is None and html.strip():
        html = b"<html><body>" + html + b"</body></html>"
        tree = load_html(html)
    if tree is None:
        raise ExtractError("Не удалось разобрать HTML страницы: файл пустой или не HTML")
    meta = trafilatura.extract_metadata(load_html(html), default_url=url)
    formulas: list[tuple[str, bool]] = []
    lost = _replace_math(tree, formulas, pandoc or tools.find_pandoc())
    for cls in _WEB_JUNK_CLASSES:
        for node in list(tree.xpath(f"//*[{_has_class(cls)}]")):
            _replace_with_text(node, "")
    for node in list(tree.xpath(f"//sup[{_has_class('reference')}]")):
        _replace_with_text(node, "")
    markdown = trafilatura.extract(
        tree,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_images=True,
        include_links=False,
        include_comments=False,
        include_formatting=True,
    )
    if not markdown or not markdown.strip():
        raise ExtractError(_empty_message("web"))

    def restore(m: re.Match[str]) -> str:
        n = int(m.group(2))
        if n >= len(formulas):
            return m.group(0)
        tex, display = formulas[n]
        return f"$${tex}$$" if display else f"${tex}$"

    markdown = _PLACEHOLDER_RE.sub(restore, markdown)
    return WebPage(markdown, _page_title(meta), len(formulas), lost)


def _web_title(doc: dict[str, Any], page_title: str) -> list[Any]:
    """The page's own first heading when `<title>` is that heading plus the site name."""
    first = next((b for b in doc["blocks"] if bl.classify(b) is not None), None)
    if first is not None and first.get("t") == "Header":
        heading = _norm(bl.inline_text(first["c"][2]))
        title = _norm(page_title)
        if heading and (
            title == heading
            or any(title.startswith(heading + sep.casefold()) for sep in _TITLE_SEPARATORS)
        ):
            return first["c"][2]
    return _str_inlines(page_title)


def _page_title(meta: Any) -> str | None:
    title = (getattr(meta, "title", None) or "").strip()
    if not title:
        return None
    site = (getattr(meta, "sitename", None) or "").strip()
    for sep in _TITLE_SEPARATORS:
        if site and title.endswith(sep + site):
            return title[: -len(sep + site)].strip() or title
    return title


def _replace_with_text(node: Any, text: str) -> None:
    """Remove an lxml element, leaving `text` (and its tail) in its place."""
    parent = node.getparent()
    if parent is None:
        return
    tail = text + (node.tail or "")
    prev = node.getprevious()
    if prev is not None:
        prev.tail = (prev.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(node)


def _clean_tex(tex: str) -> str:
    tex = tex.strip()
    m = _DISPLAYSTYLE_RE.match(tex)
    if m:
        tex = m.group(1).strip()
    return re.sub(r"\s+", " ", tex) if "\n" in tex else tex


def _tex_of(node: Any) -> str | None:
    data = node.get("data-mw")
    if data:
        try:
            src = json.loads(data)["body"]["extsrc"]
            if isinstance(src, str) and src.strip():
                return src
        except (ValueError, KeyError, TypeError):
            pass
    for ann in node.iter("annotation"):
        if (ann.get("encoding") or "").lower() in ("application/x-tex", "tex"):
            return ann.text or ""
    for math in node.iter("math"):
        if math.get("alttext"):
            return math.get("alttext")
    return None


def _is_display(node: Any) -> bool:
    classes = f" {node.get('class') or ''} "
    if any(
        f" {c} " in classes
        for c in (
            "mwe-math-element-block",
            "katex-display",
            "MathJax_Display",
            "mwe-math-fallback-image-display",
        )
    ):
        return True
    if node.tag == "math" and node.get("display") == "block":
        return True
    if node.tag == "mjx-container" and node.get("display") == "true":
        return True
    return any(m.get("display") == "block" for m in node.iter("math"))


def _formula_image_tex(node: Any) -> str | None:
    """TeX of a formula image (`<img class="tex" alt="x^2">`, codecogs, MediaWiki fallback)."""
    alt = (node.get("alt") or "").strip()
    if not alt:
        return None
    classes = set((node.get("class") or "").split())
    src = (node.get("src") or "").lower()
    if not (classes & _FORMULA_IMG_CLASSES or any(s in src for s in _FORMULA_IMG_SOURCES)):
        return None
    m = re.fullmatch(r"\$\$(.+)\$\$|\$(.+)\$|\\\((.+)\\\)|\\\[(.+)\\\]", alt, re.DOTALL)
    if m:
        alt = next(g for g in m.groups() if g is not None)
    return alt.strip() or None


def _replace_math(tree: Any, formulas: list[tuple[str, bool]], pandoc: Path | None) -> int:
    """Formulas → placeholders; returns how many formulas had no TeX and were lost."""
    lost = 0
    pending: list[tuple[Any, Any, bool]] = []  # (element to replace, its <math>, display)

    def put(node: Any, tex: str, display: bool) -> None:
        formulas.append((_clean_tex(tex), display))
        kind = "d" if display else "i"
        _replace_with_text(node, f"h0lonmath{kind}{len(formulas) - 1}z")

    def queue(node: Any, math: Any, display: bool) -> None:
        math.set(_QUEUED_ATTR, "1")
        pending.append((node, math, display))

    # MediaWiki, KaTeX and MathJax 3 wrappers first (they hold a <math> and a fallback).
    for xp in (
        f"//*[{_has_class('mwe-math-element')}]",
        f"//*[{_has_class('katex-display')}]",
        f"//*[{_has_class('katex')}]",
        "//mjx-container",
    ):
        for node in list(tree.xpath(xp)):
            if not _attached(node, tree):
                continue
            display = _is_display(node) or _alone_in_dd(node)
            tex = _tex_of(node)
            if tex is not None:
                put(node, tex, display)
                continue
            math = next(iter(node.iter("math")), None)
            if math is not None:
                queue(node, math, display)
            elif node.tag == "mjx-container":
                lost += 1  # an SVG/CHTML rendering only: trafilatura drops it
                _replace_with_text(node, "")
            else:
                lost += 1  # KaTeX HTML without annotation stays as garbled text
    for node in list(tree.xpath("//math")):
        if not _attached(node, tree) or node.get(_QUEUED_ATTR):
            continue
        display = _is_display(node) or _alone_in_dd(node)
        tex = _tex_of(node)
        if tex is not None:
            put(node, tex, display)
        else:
            queue(node, node, display)
    for node in list(tree.xpath("//img[@alt]")):
        tex = _formula_image_tex(node)
        if tex is not None and _attached(node, tree):
            put(node, tex, _is_display(node) or _alone_in_dd(node))
    for node in list(tree.xpath("//script[starts-with(@type, 'math/tex')]")):
        put(node, node.text or "", "mode=display" in (node.get("type") or ""))
    for node in list(tree.xpath(f"//*[{_has_class('MathJax_Preview')}]")):
        _replace_with_text(node, "")
    converted = _mathml_to_tex([math for _node, math, _display in pending], pandoc)
    for (node, _math, display), tex in zip(pending, converted, strict=True):
        if tex:
            put(node, tex, display)
        else:
            lost += 1
            _replace_with_text(node, "")
    _replace_text_delimiters(tree, formulas)
    _promote_display_dd(tree)
    return lost


def _mathml_to_tex(maths: list[Any], pandoc: Path | None) -> list[str | None]:
    """TeX of MathML elements without a TeX annotation (Pandoc's HTML reader, texmath)."""
    if not maths:
        return []
    if pandoc is None:
        return [None] * len(maths)
    from lxml import etree

    parts = []
    for i, math in enumerate(maths):
        if _QUEUED_ATTR in math.attrib:
            del math.attrib[_QUEUED_ATTR]
        xml = etree.tostring(math, encoding="unicode", with_tail=False)
        parts.append(f"<p>{_MATHML_MARK}{i}: {xml}</p>")
    try:
        doc = bl.read_ast(
            pandoc,
            text="<html><body>" + "".join(parts) + "</body></html>",
            reader="html",
            extra_args=SANDBOX,
        )
    except bl.BlocksError:
        return [None] * len(maths)
    out: list[str | None] = [None] * len(maths)
    for block in doc["blocks"]:
        if block.get("t") not in ("Para", "Plain") or not block["c"]:
            continue
        first = block["c"][0]
        if first.get("t") != "Str":
            continue
        m = re.fullmatch(rf"{_MATHML_MARK}(\d+):", first["c"])
        if m is None or int(m.group(1)) >= len(maths):
            continue
        tex = next((n["c"][1] for n in bl.walk_elements(block["c"]) if n.get("t") == "Math"), None)
        out[int(m.group(1))] = tex.strip() if tex and tex.strip() else None
    return out


def _attached(node: Any, root: Any) -> bool:
    """False for an element inside a subtree that was already removed."""
    top = node
    for ancestor in node.iterancestors():
        top = ancestor
    return top is root


def _alone_in_dd(node: Any) -> bool:
    """MediaWiki writes display formulas as `<dl><dd>formula</dd></dl>`."""
    parent = node.getparent()
    if parent is None or parent.tag != "dd":
        return False
    rest = (parent.text or "") + "".join(
        ("".join(c.itertext()) if c is not node else "") + (c.tail or "") for c in parent
    )
    return bool(_PUNCT_TEXT_RE.match(rest))


def _replace_text_delimiters(tree: Any, formulas: list[tuple[str, bool]]) -> None:
    skip = {"script", "style", "code", "pre", "textarea"}

    def sub(text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            inline, display, dollars = m.groups()
            tex, is_display = (inline, False) if inline is not None else (display or dollars, True)
            formulas.append((_clean_tex(tex), is_display))
            return f"h0lonmath{'d' if is_display else 'i'}{len(formulas) - 1}z"

        return _TEX_TEXT_RE.sub(repl, text)

    for node in tree.iter():
        if not isinstance(node.tag, str):
            continue
        if node.tag not in skip and node.text and ("\\" in node.text or "$$" in node.text):
            node.text = sub(node.text)
        parent = node.getparent()
        if (
            node.tail
            and parent is not None
            and parent.tag not in skip
            and ("\\" in node.tail or "$$" in node.tail)
        ):
            node.tail = sub(node.tail)


def _promote_display_dd(tree: Any) -> None:
    """`<dd>` with only a display formula → its own `<p>` (not a list item)."""
    for dl in list(tree.xpath("//dl")):
        children = list(dl)
        if not children:
            continue

        def is_formula(c: Any) -> bool:
            text = "".join(c.itertext())
            return (
                c.tag == "dd"
                and _PLACEHOLDER_RE.search(text) is not None
                and bool(_PUNCT_TEXT_RE.match(_PLACEHOLDER_RE.sub("", text)))
                and "h0lonmathd" in text
            )

        if not any(is_formula(c) for c in children):
            continue
        group: Any = None
        for c in children:
            if is_formula(c):
                p = dl.makeelement("p", {})
                p.text = "".join(c.itertext()).strip()
                dl.addprevious(p)
                group = None
            else:
                if group is None:
                    group = dl.makeelement("dl", {})
                    dl.addprevious(group)
                group.append(c)
        tail = dl.tail
        parent = dl.getparent()
        if parent is not None:
            prev = dl.getprevious()
            if tail and prev is not None:
                prev.tail = (prev.tail or "") + tail
            parent.remove(dl)
