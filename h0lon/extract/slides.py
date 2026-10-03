"""Slides extractor: PDF slides (page = slide) and PPTX (python-pptx; LibreOffice if found).

PDF: every page is a slide. Its title is the line of the largest font size in the top third
of the slide (with the lines continuing it), the place heading is
`## [[S1:s12]] Слайд 12. <title>`; text slides go through pymupdf4llm with the title and
the running lines redacted, `math`/`graphic` slides through `vision.transcribe_pages`
(flavor `slides`, batches of 10).

PPTX: python-pptx gives the title, texts of shapes with list levels, tables (pipe tables),
pictures (`figures/`) and speaker notes («Заметки докладчика: …»). A slide without body text
but with a picture is transcribed from that picture. When LibreOffice (`soffice`) is found,
the deck is converted to PDF and slides that are graphic, have formulas (OMML is invisible
to python-pptx) or SmartArt are transcribed from their rendered page instead.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from h0lon import procutil, tools
from h0lon.extract import pages as pg
from h0lon.extract import pdf as pdfx
from h0lon.extract import vision
from h0lon.extract.model import ExtractContext, ExtractOutput, ExtractPlan, PageImage
from h0lon.extract.registry import ExtractError

VERSION = f"1.2+mupdf{pymupdf.VersionBind}"
BATCH_SIZE = vision.SLIDES_BATCH
PICTURES_ONLY = "графика слайдов описана по встроенным картинкам"
NO_SOFFICE_NOTE = f"LibreOffice не найден: {PICTURES_ONLY}"
SLIDE_FAILED_TITLE = "Слайд не распознан агентом"
SOFFICE_TIMEOUT_S = 300
TITLE_ZONE = 1 / 3  # the title is searched in the top third of the slide
TITLE_MAX_LINES_ALONE = 2  # a slide of more same-style lines and nothing else is a paragraph


# ---------------------------------------------------------------- helpers

escape_inline = pg.escape_inline


def _norm_title(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def plain_math_letters(text: str) -> str:
    """Map Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF: 𝑘, 𝐱, 𝔻 …) to plain letters.

    Beamer and LaTeX put math-italic glyphs into the text layer of titles ("Метод 𝑘-ближайших
    соседей"); in a heading they read as odd symbols and break search. Only this block is
    normalised (NFKC), so superscripts and other compatibility characters stay intact.
    """
    return "".join(
        unicodedata.normalize("NFKC", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF else ch for ch in text
    )


def slide_heading(sid: str, number: int, title: str | None) -> str:
    head = f"## [[{sid}:s{number}]] Слайд {number}"
    if title and title.strip():
        clean = re.sub(r"\s+", " ", plain_math_letters(title)).strip()
        head += ". " + escape_inline(clean).replace("#", "\\#")
    return head


def strip_title(md: str, title: str | None) -> str:
    """Drop a leading heading or line that repeats the slide title (agents do that)."""
    if not title:
        return md
    key = _norm_title(title)
    if not key:
        return md
    lines = md.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        first = re.sub(r"^#{1,6}\s*|^\*\*|\*\*$", "", lines[i].strip()).strip()
        if _norm_title(first) == key:
            rest = "\n".join(lines[i + 1 :]).strip()
            return rest or vision.EMPTY_PAGE
    return md


def slide_title(
    profile: pg.PageProfile, drop: Iterable[pg.LineInfo] = ()
) -> tuple[str | None, list[pg.LineInfo]]:
    """Title of a PDF slide: the largest line in the top third that stands out from the body.

    Candidates go by size, then from the top. A candidate is the title when its style
    (font, boldness, size) differs from that of most of the remaining text: Beamer sets
    SFSX 14.35 over SFSS of the same or even a larger size (17.22 on a slide of the real
    lecture), so a body line is never taken for the title. Lines right below it with the
    same size and font continue the title (two-line titles; the second line may leave the
    top third). Page numbers, one-character lines, lines in math fonts and list items
    («• Пункт», «1. Шаг» in the body size) are never titles; a slide with no line that
    stands out has no title.
    """
    skip = {line.index for line in drop}
    usable = [
        line
        for line in profile.lines
        if line.index not in skip
        and len(line.shown.strip()) >= 2
        and not pg.is_page_number(line.shown)
    ]
    if not usable:
        return None, []
    body_size = _body_style(usable)[2]

    def list_item(line: pg.LineInfo) -> bool:
        # «2. Метрические методы» in a large font is a numbered title, not an item.
        return _is_list_line(line) and line.size <= body_size + 0.5 and len(usable) > 1

    cands = sorted(
        (
            line
            for line in usable
            if line.bbox[1] < profile.height * TITLE_ZONE
            and not list_item(line)
            and not pg.MATH_FONT_RE.search(line.font)
        ),
        key=lambda line: (-round(line.size * 2) / 2, line.bbox[1], line.bbox[0]),
    )
    ordered = sorted(usable, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    for top in cands:
        chosen = _title_lines(top, ordered, list_item)
        taken = {line.index for line in chosen}
        rest = [line for line in usable if line.index not in taken]
        if rest:
            stands_out = _body_style(rest) != _style(top)
        else:  # the whole slide in one style: a title of one or two lines, or a paragraph
            stands_out = len(chosen) <= TITLE_MAX_LINES_ALONE
        if stands_out:
            title = " ".join(line.shown.strip() for line in chosen).strip()
            return (title or None), chosen
    return None, []  # every candidate looks like the body: no title


def _title_lines(
    top: pg.LineInfo, ordered: Sequence[pg.LineInfo], list_item: Any
) -> list[pg.LineInfo]:
    """`top` and the lines right below it that continue it (same size and font)."""
    chosen = [top]
    for line in ordered:
        last = chosen[-1]
        if line is top or line.bbox[1] <= last.bbox[1]:
            continue
        gap = line.bbox[1] - last.bbox[3]
        same = abs(line.size - top.size) <= 0.5 and line.font == top.font
        overlaps = line.bbox[0] < last.bbox[2] and line.bbox[2] > last.bbox[0]
        close = -0.5 * top.size <= gap <= 0.6 * top.size
        if same and overlaps and close and not list_item(line):
            chosen.append(line)
        elif line.bbox[1] > last.bbox[3] + top.size:
            break
    return chosen


def _is_list_line(line: pg.LineInfo) -> bool:
    return bool(_BULLET_RE.match(line.shown) or _NUMBER_RE.match(line.shown))


def _style(line: pg.LineInfo) -> tuple[str, bool, float]:
    return line.font, line.bold, round(line.size * 2) / 2


def _body_style(lines: Iterable[pg.LineInfo]) -> tuple[str, bool, float]:
    """Font, boldness and size (to 0.5 pt) of most characters of `lines`."""
    weights: dict[tuple[str, bool, float], int] = {}
    for line in lines:
        key = _style(line)
        weights[key] = weights.get(key, 0) + len(line.shown)
    return max(weights.items(), key=lambda kv: kv[1])[0]


_BULLET_RE = re.compile(r"^\s*([∙•◦▪▫■□►▶▸‣⁃●○◆◇–—*·-])\s*")
_NUMBER_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+")


def _strip_bullet(md: str) -> str:
    """Remove a leading bullet glyph from the Markdown of a line (it may be emphasized)."""
    m = re.match(r"^(\s*(?:\*{1,3}|~~)?)\\?([∙•◦▪▫■□►▶▸‣⁃●○◆◇–—*·-])\s*", md)
    if not m:
        return md
    return (m.group(1) + md[m.end() :]).strip()


def slide_text_markdown(profile: pg.PageProfile, drop: Iterable[pg.LineInfo] = ()) -> str:
    """Markdown of a text slide from its lines: bullets → lists (levels by indent).

    pymupdf4llm merges separate outline items of Beamer slides into one paragraph, so text
    slides are rendered from PyMuPDF lines directly: a line starting with a bullet glyph
    opens a list item, its wrapped continuation lines are joined to it, other lines of one
    text block form a paragraph.
    """
    skip = {line.index for line in drop}
    lines = [line for line in profile.lines if line.index not in skip]
    marks = sorted(
        {
            round(ln.bbox[0])
            for ln in lines
            if _BULLET_RE.match(ln.shown) or _NUMBER_RE.match(ln.shown)
        }
    )
    levels: list[int] = []
    for x in marks:
        if not levels or x - levels[-1] > 4:
            levels.append(x)

    def level_of(x: float) -> int:
        best = 0
        for i, lx in enumerate(levels):
            if x >= lx - 4:
                best = i
        return min(best, 8)

    blocks: list[str] = []
    items: list[list[Any]] = []  # [indent level, marker, text, x0 of the bullet, block]
    para: list[str] = []
    para_block: int | None = None

    def join(left: str, right: str) -> str:
        joined = pg.join_hyphenated(
            left, right, vocabulary=profile.vocabulary, hyphenating=profile.hyphenating
        )
        return joined if joined is not None else f"{left} {right}"

    def flush_para() -> None:
        nonlocal para_block
        if para:
            text = para[0]
            for piece in para[1:]:
                text = join(text, piece)
            blocks.append(_protect_start(text))
            para.clear()
        para_block = None

    def flush_items() -> None:
        if items:
            blocks.append(
                "\n".join("    " * lvl + f"{mark} {text}" for lvl, mark, text, _x, _b in items)
            )
            items.clear()

    for line in lines:
        md = line.markdown().strip()
        if not md:
            continue
        x0 = line.bbox[0]
        bullet = _BULLET_RE.match(line.shown)
        number = _NUMBER_RE.match(line.shown) if not bullet else None
        if bullet or number:
            flush_para()
            mark = "-" if bullet else "1."
            text = _strip_bullet(md) if bullet else re.sub(r"^\s*\d{1,3}\\?[.)]\s+", "", md)
            items.append([level_of(x0), mark, text, x0, line.block])
            continue
        if items and para_block is None and x0 >= items[-1][3] + 2 and line.block == items[-1][4]:
            items[-1][2] = join(items[-1][2], md)  # wrapped continuation of the item
            continue
        flush_items()
        if para and para_block != line.block:
            flush_para()
        para.append(md)
        para_block = line.block
    flush_para()
    flush_items()
    return "\n\n".join(blocks).strip()


def has_table(page: pymupdf.Page) -> bool:
    """A ruled table of at least 2×2 cells on the page (PyMuPDF table finder)."""
    try:
        tables = page.find_tables().tables
    except Exception:
        return False
    return any(t.row_count >= 2 and t.col_count >= 2 for t in tables)


# ---------------------------------------------------------------- LibreOffice


def find_soffice() -> list[str] | None:
    found = tools.find_soffice()
    return [str(found)] if found else None


def convert_to_pdf(argv: Sequence[str], src: Path, work_dir: Path) -> tuple[Path | None, str]:
    """PPTX → PDF with LibreOffice in its own profile directory; (pdf, error)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    profile = (work_dir / "lo-profile").resolve()
    res = procutil.run(
        [
            *argv,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(work_dir),
            str(src),
        ],
        cwd=work_dir,
        timeout=SOFFICE_TIMEOUT_S,
    )
    pdf = work_dir / f"{src.stem}.pdf"
    if pdf.is_file() and pdf.stat().st_size > 0:
        return pdf, ""
    detail = res.error or ("таймаут" if res.timed_out else (res.stderr or res.stdout).strip())
    return None, detail[:300] or f"код {res.exit_code}"


# ---------------------------------------------------------------- PPTX reading


@dataclass
class PptxSlide:
    number: int
    title: str | None = None
    blocks: list[str] = field(default_factory=list)  # Markdown blocks of the body
    plain: list[str] = field(default_factory=list)  # plain body text (hints, statistics)
    notes: str | None = None
    pictures: list[Path] = field(default_factory=list)
    picture_png: Path | None = None  # largest picture as PNG (for vision)
    has_math: bool = False  # OMML equations: python-pptx does not see their text
    has_diagram: bool = False  # SmartArt / OLE: no text through python-pptx
    hidden: bool = False

    @property
    def has_text(self) -> bool:
        return any(t.strip() for t in self.plain)

    @property
    def chars(self) -> int:
        return sum(len("".join(t.split())) for t in self.plain)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _emph(text: str, bold: bool, italic: bool) -> str:
    stripped = text.strip()
    if not stripped or not (bold or italic):
        return text
    mark = "***" if bold and italic else "**" if bold else "*"
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    return f"{lead}{mark}{stripped}{mark}{trail}"


def _run_flags(r: Any) -> tuple[bool, bool]:
    rpr = None
    for child in r:
        if _local(child.tag) == "rPr":
            rpr = child
            break
    if rpr is None:
        return False, False
    return rpr.get("b") in ("1", "true"), rpr.get("i") in ("1", "true")


def _paragraph(p: Any) -> tuple[str, str, bool]:
    """(Markdown, plain text, has OMML) of a python-pptx paragraph."""
    pieces: list[tuple[str, bool, bool]] = []  # (text or "\n", bold, italic)
    has_math = False
    for child in p._p:
        tag = _local(child.tag)
        if tag == "r":
            text = "".join(t.text or "" for t in child.iter() if _local(t.tag) == "t")
            bold, italic = _run_flags(child)
            pieces.append((text, bold, italic))
        elif tag == "br":
            pieces.append(("\n", False, False))
        elif tag == "fld":
            if (child.get("type") or "").startswith("slidenum"):
                continue
            text = "".join(t.text or "" for t in child.iter() if _local(t.tag) == "t")
            pieces.append((text, False, False))
        if any(_local(x.tag) in ("oMath", "oMathPara") for x in child.iter()):
            has_math = True
    md_parts: list[str] = []
    plain_parts: list[str] = []
    i = 0
    while i < len(pieces):
        text, bold, italic = pieces[i]
        if text == "\n":
            md_parts.append("\\\n")
            plain_parts.append("\n")
            i += 1
            continue
        j = i + 1
        while j < len(pieces) and pieces[j][0] != "\n" and pieces[j][1:] == (bold, italic):
            text += pieces[j][0]
            j += 1
        md_parts.append(_emph(escape_inline(text), bold, italic))
        plain_parts.append(text)
        i = j
    md = re.sub(r"[ \t]+", " ", "".join(md_parts)).strip()
    plain = re.sub(r"[ \t]+", " ", "".join(plain_parts)).strip()
    return md, plain, has_math


def _bullet_kind(p: Any, default_bullets: bool) -> str | None:
    ppr = p._p.pPr
    if ppr is not None:
        for child in ppr:
            tag = _local(child.tag)
            if tag == "buNone":
                return None
            if tag == "buAutoNum":
                return "number"
            if tag in ("buChar", "buBlip"):
                return "bullet"
    return "bullet" if default_bullets else None


def _protect_start(md: str) -> str:
    """A plain paragraph must not start like a list item or a heading."""
    m = re.match(r"^(\d+)([.)])(\s)", md)
    if m:
        return m.group(1) + "\\" + m.group(2) + md[m.end(2) :]
    if re.match(r"^([-+#>]|:::)", md):
        return "\\" + md
    return md


def _text_frame(tf: Any, *, default_bullets: bool) -> tuple[list[str], list[str], bool]:
    blocks: list[str] = []
    plain: list[str] = []
    items: list[str] = []
    has_math = False

    def flush() -> None:
        if items:
            blocks.append("\n".join(items))
            items.clear()

    for p in tf.paragraphs:
        md, text, math = _paragraph(p)
        has_math = has_math or math
        if not md:
            flush()
            continue
        plain.append(text)
        kind = _bullet_kind(p, default_bullets)
        if kind is None:
            flush()
            blocks.append(_protect_start(md))
            continue
        marker = "-" if kind == "bullet" else "1."
        items.append("    " * max(0, min(p.level, 8)) + f"{marker} {md.replace(chr(10), ' ')}")
    flush()
    return blocks, plain, has_math


def _table(table: Any) -> tuple[str, list[str]]:
    rows: list[list[str]] = []
    plain: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            parts = [_paragraph(p) for p in cell.text_frame.paragraphs]
            md = " ".join(m for m, _, _ in parts if m).replace("\\\n", " ")
            cells.append(md or " ")  # `|` is already escaped by escape_inline
            plain.extend(t for _, t, _ in parts if t)
        rows.append(cells)
    if not rows:
        return "", plain
    width = max(len(r) for r in rows)
    rows = [r + [" "] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines), plain


def _chart(chart: Any) -> tuple[str, list[str]]:
    try:
        title = chart.chart_title.text_frame.text.strip() if chart.has_title else ""
    except Exception:
        title = ""
    try:
        ctype = str(chart.chart_type).split(".")[-1].split(" ")[0]
    except Exception:
        ctype = ""
    lines = [
        "::: {.figure-description}",
        "Диаграмма"
        + (f" «{escape_inline(title)}»" if title else "")
        + (f" (тип {escape_inline(ctype)})" if ctype else "")
        + ".",
    ]
    plain = [title] if title else []
    try:
        plot = chart.plots[0]
        cats = [str(c) for c in plot.categories]
        series = [(s.name or "", list(s.values)) for s in plot.series]
        if cats and series:
            head = ["Категория", *[escape_inline(n) or " " for n, _ in series]]
            lines += ["", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
            for i, cat in enumerate(cats):
                vals = [("" if i >= len(v) or v[i] is None else f"{v[i]:g}") for _, v in series]
                lines.append("| " + " | ".join([escape_inline(cat), *vals]) + " |")
            plain += cats
    except Exception:
        lines.append("Данные диаграммы не прочитаны.")
    lines.append(":::")
    return "\n".join(lines), plain


_SKIP_PLACEHOLDERS = {"SLIDE_NUMBER", "FOOTER", "DATE", "HEADER"}
_BODY_PLACEHOLDERS = {"BODY", "OBJECT", "VERTICAL_BODY", "VERTICAL_OBJECT"}


def _placeholder_type(shape: Any) -> str | None:
    if not getattr(shape, "is_placeholder", False):
        return None
    try:
        return str(shape.placeholder_format.type).split(".")[-1].split(" ")[0]
    except Exception:
        return None


def _iter_shapes(shapes: Iterable[Any]) -> Iterable[Any]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    ordered = sorted(shapes, key=lambda s: ((s.top or 0), (s.left or 0)))
    for shape in ordered:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def read_pptx(path: Path, *, figures_dir: Path, prefix: str) -> list[PptxSlide]:
    """Slides of a PPTX as Markdown blocks; pictures are written to `figures_dir`."""
    from pptx import Presentation

    prs = Presentation(str(path))
    slides: list[PptxSlide] = []
    for number, slide in enumerate(prs.slides, 1):
        s = PptxSlide(number=number, hidden=slide._element.get("show") in ("0", "false"))
        title_shape = slide.shapes.title
        if title_shape is not None and title_shape.has_text_frame:
            s.title = re.sub(r"\s+", " ", title_shape.text_frame.text).strip() or None
        title_id = title_shape.shape_id if title_shape is not None else None
        biggest = 0
        pic_no = 0
        for shape in _iter_shapes(slide.shapes):
            if title_id is not None and shape.shape_id == title_id:
                continue
            ph = _placeholder_type(shape)
            if ph in _SKIP_PLACEHOLDERS:
                continue
            xml = shape._element.xml if hasattr(shape, "_element") else ""
            if "drawingml/2006/diagram" in xml or "oleObj" in xml:
                s.has_diagram = True
            if getattr(shape, "has_table", False) and shape.has_table:
                md, plain = _table(shape.table)
                if md:
                    s.blocks.append(md)
                    s.plain += plain
                continue
            if getattr(shape, "has_chart", False) and shape.has_chart:
                md, plain = _chart(shape.chart)
                s.blocks.append(md)
                s.plain += plain
                continue
            image = None
            try:
                image = shape.image  # pictures and picture placeholders
            except Exception:
                image = None
            if image is not None:
                pic_no += 1
                figures_dir.mkdir(parents=True, exist_ok=True)
                ext = (image.ext or "bin").lower()
                target = figures_dir / f"{prefix}s{number:04d}_{pic_no}.{ext}"
                target.write_bytes(image.blob)
                s.pictures.append(target)
                alt = ""
                try:
                    alt = (shape._element.nvPicPr.cNvPr.get("descr") or "").strip()
                except Exception:
                    alt = ""
                if re.fullmatch(r"[\w .()-]+\.(?:png|jpe?g|gif|bmp|tiff?|emf|wmf|svg)", alt, re.I):
                    alt = ""  # PowerPoint and python-pptx put the file name there
                s.blocks.append(f"![{escape_inline(alt)}]({figures_dir.name}/{target.name})")
                area = int(shape.width or 0) * int(shape.height or 0)
                if area >= biggest:
                    biggest = area
                    s.picture_png = target
                continue
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                blocks, plain, math = _text_frame(
                    shape.text_frame, default_bullets=ph in _BODY_PLACEHOLDERS
                )
                s.blocks += blocks
                s.plain += plain
                s.has_math = s.has_math or math
        if slide.has_notes_slide:
            try:
                notes = slide.notes_slide.notes_text_frame.text
            except Exception:
                notes = ""
            notes = (notes or "").strip()
            if notes:
                s.notes = notes
        slides.append(s)
    return slides


def notes_block(notes: str | None) -> str | None:
    if not notes:
        return None
    lines = [escape_inline(line.strip()) for line in notes.splitlines() if line.strip()]
    return "Заметки докладчика: " + "\\\n".join(lines) if lines else None


# ---------------------------------------------------------------- extractor


class SlidesExtractor:
    kinds: tuple[str, ...] = ("slides",)
    version: str = VERSION
    prompts: tuple[str, ...] = (vision.PROMPT_REF,)
    tier: str = "light"
    batch_size: int = BATCH_SIZE

    # ---- plan

    def plan(self, ctx: ExtractContext) -> ExtractPlan:
        src = pdfx.source_path(ctx)
        if src.suffix.lower() == ".pptx":
            return self._plan_pptx(ctx, src)
        return self._plan_pdf(ctx, src)

    def _plan_pdf(self, ctx: ExtractContext, src: Path) -> ExtractPlan:
        with pg.opened(src) as doc:
            analysis = pdfx.analyze(doc, slides=True)
        plan = ExtractPlan(source_id=ctx.source.id, pages_total=len(analysis.profiles))
        plan.notes.append(
            f"Слайдов: {plan.pages_total} (текст: {analysis.count('text')}, формулы: "
            f"{analysis.count('math')}, графика: "
            f"{analysis.count('graphic') + analysis.count('scan')})"
        )
        self._plan_vision(ctx, analysis, plan)
        return plan

    def _plan_vision(self, ctx: ExtractContext, analysis: pdfx.Analysis, plan: ExtractPlan) -> None:
        if not analysis.vision:
            return
        if not ctx.use_vision:
            plan.notes.append(
                f"Агент отключён: {len(analysis.vision)} слайдов пойдут текстовым слоем"
            )
            return
        titles = self._title_drops(analysis)
        pending, cached, runs = pdfx.planned_runs(
            ctx,
            analysis,
            flavor_for=lambda p: "slides",
            batch_size={"slides": self.batch_size},
            pages_dir=ctx.out_dir / "pages",
            extra_drop=titles,
        )
        plan.pages_vision += pending
        plan.agent_runs += runs
        if pending:
            plan.notes.append(
                f"Распознавание агентом: {pending} слайдов, прогонов: {runs} "
                f"(батчи по {self.batch_size})"
            )
        if cached:
            plan.notes.append(f"Из кэша распознавания: {cached} слайдов")

    def _plan_pptx(self, ctx: ExtractContext, src: Path) -> ExtractPlan:
        tmp = Path(tempfile.mkdtemp(prefix="h0lon-pptx-plan-"))  # a dry run writes nothing
        try:
            slides = read_pptx(src, figures_dir=tmp, prefix="")
        except Exception as exc:
            raise ExtractError(f"{ctx.source.id}: не удалось прочитать PPTX: {exc}") from exc
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        plan = ExtractPlan(source_id=ctx.source.id, pages_total=len(slides))
        picture_only = [s.number for s in slides if not s.has_text and s.pictures]
        plan.notes.append(
            f"Слайдов: {len(slides)}, с заметками: {sum(1 for s in slides if s.notes)}, "
            f"только картинка: {len(picture_only)}"
        )
        if find_soffice() is None:
            plan.notes.append(NO_SOFFICE_NOTE)
        else:
            plan.notes.append(
                "LibreOffice найден: графические слайды определятся после конвертации в PDF"
            )
        if ctx.use_vision and picture_only:
            plan.pages_vision = len(picture_only)
            plan.agent_runs = vision.batches_needed(len(picture_only), self.batch_size)
        return plan

    # ---- extract

    def extract(self, ctx: ExtractContext) -> ExtractOutput:
        src = pdfx.source_path(ctx)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".pptx":
            return self._extract_pptx(ctx, src)
        return self._extract_pdf(ctx, src)

    @staticmethod
    def _title_drops(analysis: pdfx.Analysis) -> dict[int, list[pg.LineInfo]]:
        drops: dict[int, list[pg.LineInfo]] = {}
        for p in analysis.profiles:
            _, lines = slide_title(p, analysis.drop(p.number))
            drops[p.number] = lines
        return drops

    def _extract_pdf(self, ctx: ExtractContext, src: Path) -> ExtractOutput:
        t0 = time.monotonic()
        sid = ctx.source.id
        warnings: list[str] = []
        try:
            doc = pg.open_pdf(src)
        except Exception as exc:
            raise ExtractError(f"{sid}: не удалось открыть PDF: {exc}") from exc
        try:
            analysis = pdfx.analyze(doc, slides=True)
            if analysis.repaired:
                warnings.append(pdfx.repaired_warning(sid))
            titles: dict[int, str | None] = {}
            title_drops: dict[int, list[pg.LineInfo]] = {}
            for p in analysis.profiles:
                titles[p.number], title_drops[p.number] = slide_title(p, analysis.drop(p.number))
            ctx.emit(
                f"{sid}: {len(analysis.profiles)} слайдов — текст {analysis.count('text')}, "
                f"формулы {analysis.count('math')}, графика "
                f"{analysis.count('graphic') + analysis.count('scan')}"
            )
            # Text slides with a ruled table go through pymupdf4llm (pipe tables), the rest
            # through the line renderer.
            with_tables = [p for p in analysis.text if has_table(doc[p.number - 1])]
            contents = {
                p.number: slide_text_markdown(p, analysis.drop(p.number) + title_drops[p.number])
                or vision.EMPTY_PAGE
                for p in analysis.text
                if p.number not in {t.number for t in with_tables}
            }
            contents.update(
                pdfx.text_pages_markdown(
                    src,
                    analysis,
                    with_tables,
                    figures_dir=ctx.out_dir / "figures",
                    figure_prefix=f"{sid}_",
                    extra_drop=title_drops,
                    warnings=warnings,
                )
            )
            # Slide renders are only for the agent: none without it.
            images = (
                pdfx.render_vision_pages(
                    doc,
                    analysis,
                    analysis.vision,
                    pages_dir=ctx.out_dir / "pages",
                    extra_drop=title_drops,
                )
                if ctx.use_vision
                else {}
            )
        finally:
            doc.close()

        outcome = pdfx.VisionOutcome()
        if images:
            outcome = pdfx.run_vision(
                ctx, {"slides": list(images.values())}, batch_size={"slides": self.batch_size}
            )
            warnings += outcome.warnings
        fallback = pdfx.fallback_texts(analysis, analysis.vision, extra_drop=title_drops)
        ratio_texts = dict(contents)
        for n in (p.number for p in analysis.vision):
            if n in outcome.pages:
                contents[n] = ratio_texts[n] = strip_title(outcome.pages[n], titles.get(n))
            else:
                block = pdfx.fallback_block(fallback[n], agent_off=not ctx.use_vision)
                contents[n] = block.replace(pdfx.FAILED_TITLE, SLIDE_FAILED_TITLE, 1)
                ratio_texts[n] = fallback[n]
        failed = sorted(outcome.failed) if ctx.use_vision else []
        if failed:
            warnings.append(
                f"{sid}: не распознаны агентом слайды {pdfx.pages_list(failed)} — вставлен "
                "текстовый слой в блоке uncertain"
            )
        if analysis.vision and not ctx.use_vision:
            warnings.append(
                f"{sid}: распознавание агентом отключено — {len(analysis.vision)} слайдов "
                "перенесены текстовым слоем без проверки"
            )
        body: list[str] = []
        for p in analysis.profiles:
            body.append(slide_heading(sid, p.number, titles.get(p.number)))
            body.append(contents.get(p.number) or vision.EMPTY_PAGE)
            ratio_texts.setdefault(p.number, "")
            if titles.get(p.number):
                ratio_texts[p.number] = f"{titles[p.number]}\n{ratio_texts[p.number]}"
        body_md = pdfx.write_body(ctx.out_dir / "body.md", "\n\n".join(body))
        quality = pdfx.base_quality(
            analysis, contents=ratio_texts, vision_done=len(outcome.pages), failed=failed
        )
        quality["slides_titled"] = sum(1 for t in titles.values() if t)
        quality["notes"] = pdfx.common_notes(
            analysis, quality, use_vision=ctx.use_vision, failed=failed, unit="слайды"
        )
        ctx.emit(f"{sid}: body.md готов за {time.monotonic() - t0:.1f} с")
        return ExtractOutput(
            body_md=body_md,
            pages_total=len(analysis.profiles),
            pages_vision=len(outcome.pages),
            agent_runs=outcome.agent_runs,
            quality=quality,
            warnings=warnings,
        )

    def _extract_pptx(self, ctx: ExtractContext, src: Path) -> ExtractOutput:
        t0 = time.monotonic()
        sid = ctx.source.id
        warnings: list[str] = []
        notes: list[str] = []
        figures_dir = ctx.out_dir / "figures"
        pages_dir = ctx.out_dir / "pages"
        try:
            slides = read_pptx(src, figures_dir=figures_dir, prefix=f"{sid}_")
        except Exception as exc:
            raise ExtractError(f"{sid}: не удалось прочитать PPTX: {exc}") from exc
        ctx.emit(f"{sid}: {len(slides)} слайдов (PPTX)")

        # Slides that need the agent (`need`) and, with the agent on, the image it gets.
        images: dict[int, PageImage] = {}
        need: set[int] = set()
        rendered: set[int] = set()
        analysis: pdfx.Analysis | None = None
        soffice = find_soffice()
        if soffice is None:
            notes.append(NO_SOFFICE_NOTE)
        else:
            pdf, error = convert_to_pdf(soffice, src, ctx.out_dir / ".soffice")
            if pdf is None:
                notes.append(
                    f"LibreOffice не сконвертировал презентацию ({error}): {PICTURES_ONLY}"
                )
            else:
                target = ctx.out_dir / "slides.pdf"
                shutil.move(str(pdf), str(target))
                with pg.opened(target) as doc:
                    visible = [s for s in slides if not s.hidden]
                    if doc.page_count == len(slides):
                        mapping = {s.number: i for i, s in enumerate(slides)}
                    elif doc.page_count == len(visible):
                        mapping = {s.number: i for i, s in enumerate(visible)}
                    else:
                        mapping = {}
                        notes.append(
                            f"PDF из LibreOffice: {doc.page_count} стр. на {len(slides)} слайдов — "
                            "рендер слайдов не использован"
                        )
                    if mapping:
                        analysis = pdfx.analyze(doc, slides=True)
                        if analysis.repaired:
                            warnings.append(pdfx.repaired_warning(sid))
                        for s in slides:
                            idx = mapping.get(s.number)
                            if idx is None:
                                continue
                            prof = analysis.profiles[idx]
                            if not (
                                prof.needs_vision
                                or s.has_math
                                or s.has_diagram
                                or (not s.has_text and bool(s.pictures))
                            ):
                                continue
                            need.add(s.number)
                            rendered.add(s.number)
                            if not ctx.use_vision:
                                continue  # renders are only for the agent
                            png = pg.render_page_png(doc[idx], pages_dir / f"p{s.number:04d}.png")
                            images[s.number] = PageImage(
                                number=s.number,
                                image=png,
                                text_hint="\n".join([s.title or "", *s.plain]).strip(),
                                reason=prof.reason or ("math" if s.has_math else "graphic"),
                            )
                shutil.rmtree(ctx.out_dir / ".soffice", ignore_errors=True)
        for s in slides:
            if s.number in need or s.has_text or not s.picture_png:
                continue
            need.add(s.number)
            if not ctx.use_vision:
                continue
            try:
                png = pg.image_to_png(s.picture_png, pages_dir / f"p{s.number:04d}.png")
            except Exception as exc:
                warnings.append(
                    f"{sid}: картинку слайда {s.number} не удалось подготовить для агента ({exc})"
                )
                continue
            images[s.number] = PageImage(
                number=s.number, image=png, text_hint=s.title or "", reason="graphic"
            )
        math_unrendered = [s.number for s in slides if s.has_math and s.number not in rendered]
        if math_unrendered:
            notes.append(
                f"Формулы (OMML) на слайдах {pdfx.pages_list(math_unrendered)} не извлекаются "
                "без LibreOffice — проверьте слайды вручную"
            )
        diagrams = [s.number for s in slides if s.has_diagram and s.number not in rendered]
        if diagrams:
            notes.append(
                f"SmartArt или внедрённые объекты на слайдах {pdfx.pages_list(diagrams)} "
                "не извлекаются без LibreOffice — проверьте слайды вручную"
            )

        outcome = pdfx.VisionOutcome()
        if images:
            outcome = pdfx.run_vision(
                ctx, {"slides": list(images.values())}, batch_size={"slides": self.batch_size}
            )
            warnings += outcome.warnings
        failed = sorted(outcome.failed) if ctx.use_vision else []

        body: list[str] = []
        texts: list[str] = []
        for s in slides:
            body.append(slide_heading(sid, s.number, s.title))
            parts: list[str] = []
            if s.number in outcome.pages:
                result = strip_title(outcome.pages[s.number], s.title)
                if s.number in rendered:
                    # The whole slide was transcribed; its pictures stay available as files.
                    parts += [result, *(b for b in s.blocks if b.startswith("!["))]
                else:  # transcribed from the picture: keep the picture itself too
                    parts += [*s.blocks, result]
                texts.append(result)
            else:
                parts += s.blocks
                if s.number in need:
                    title = pdfx.AGENT_OFF_TITLE if not ctx.use_vision else SLIDE_FAILED_TITLE
                    parts.append(
                        pg.uncertain_note(title, "Графика и формулы слайда не перенесены.")
                    )
            note = notes_block(s.notes)
            if note:
                parts.append(note)
            texts += [s.title or "", *s.plain, s.notes or ""]
            body.append("\n\n".join(p for p in parts if p.strip()) or vision.EMPTY_PAGE)
        body_md = pdfx.write_body(ctx.out_dir / "body.md", "\n\n".join(body))

        if failed:
            warnings.append(f"{sid}: не распознаны агентом слайды {pdfx.pages_list(failed)}")
        if need and not ctx.use_vision:
            warnings.append(
                f"{sid}: распознавание агентом отключено — {len(need)} слайдов без описания графики"
            )
        chars = [s.chars for s in slides]
        quality: dict[str, Any] = {
            "text_layer": round(sum(1 for s in slides if s.has_text) / len(slides), 3)
            if slides
            else 0.0,
            "chars_per_page": round(pg.median_or_zero(chars)),
            "pages_text": sum(1 for s in slides if s.number not in need),
            "pages_math": sum(1 for s in slides if s.has_math),
            "pages_scan": 0,
            "pages_graphic": sum(1 for s in slides if s.number in need and not s.has_math),
            "pages_vision": len(outcome.pages),
            "pages_failed": len(failed),
            "cyrillic_ratio": pg.cyrillic_ratio(texts),
            "scan_dpi": None,
            "slides_with_notes": sum(1 for s in slides if s.notes),
            "pictures": sum(len(s.pictures) for s in slides),
            "renderer": "libreoffice" if rendered or analysis is not None else "python-pptx",
        }
        if failed:
            notes.append(
                f"Не распознаны агентом: слайды {pdfx.pages_list(failed)} — проверьте вручную"
            )
        if need and not ctx.use_vision:
            notes.append(
                f"Агент отключён: графика слайдов {pdfx.pages_list(sorted(need))} не описана"
            )
        hidden = [s.number for s in slides if s.hidden]
        if hidden:
            notes.append(f"Скрытые слайды включены: {pdfx.pages_list(hidden)}")
        quality["notes"] = notes
        ctx.emit(f"{sid}: body.md готов за {time.monotonic() - t0:.1f} с")
        return ExtractOutput(
            body_md=body_md,
            pages_total=len(slides),
            pages_vision=len(outcome.pages),
            agent_runs=outcome.agent_runs,
            quality=quality,
            warnings=warnings,
        )
