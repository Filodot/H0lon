"""body.md → blocks and source.md (docs/ARCHITECTURE.md, «Конвейер», steps 2–4).

Everything goes through the Pandoc AST (`pandoc -t json`): top-level blocks are classified,
numbered `<ID>.b001…` and given the anchor of their place in the source; the body of
source.md is written back by Pandoc (`-t markdown --wrap=none`) with an HTML comment
`<!-- P1.b007 definition -->` before every block. Old block comments are dropped before
numbering, so building source.md from its own body gives the same text (idempotent).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from h0lon import procutil
from h0lon.extract.model import DIV_BLOCK_TYPES, Block, BlockType
from h0lon.render.pandoc import MARKDOWN_FORMAT, escape_anchor_syntax

# Readers and writer use the master's extensions (PRD 5.4). Identifiers are never invented:
# anchors are the references in a Source Doc, and auto ids would change whenever an anchor
# is rewritten. Metadata blocks are not read from a body: Pandoc takes any `---` … `---`
# fragment that parses as YAML for metadata, anywhere in the text, and drops it; front
# matter is split off by `parse_front_matter` instead. The writer keeps Unicode punctuation
# as is (no `---` for «—») and prefers pipe tables; tables with block content fall back to
# grid tables. Footnote definitions are written right after their block, so the Markdown of
# a block carries its notes.
READER_FORMAT = MARKDOWN_FORMAT + "-auto_identifiers-yaml_metadata_block-pandoc_title_block"
WRITER_FORMAT = MARKDOWN_FORMAT + "-smart-simple_tables-multiline_tables-auto_identifiers"
WRITER_ARGS = ("--wrap=none", "--reference-location=block")
PANDOC_TIMEOUT_S = 300

# `[[P1:p3]]`, `[[S1:s12]]`, `[[V1:12:34]]`, `[[W1:§2]]`.
ANCHOR_PATTERN = r"\[\[([A-Z][A-Za-z0-9]*):([^\[\]\s]+)\]\]"
_ANCHOR_RE = re.compile(ANCHOR_PATTERN)
_LOCATION_RE = re.compile(r"^\s*" + ANCHOR_PATTERN)
# `<!-- P1.b007 definition -->` written by this module (any source id, any type word).
BLOCK_COMMENT_RE = re.compile(r"^\s*<!--\s*[A-Z][A-Za-z0-9]*\.b\d+(?:\s+[\w-]+)?\s*-->\s*$")
_FRONT_MATTER_RE = re.compile(
    r"\A\N{ZERO WIDTH NO-BREAK SPACE}?---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$", re.DOTALL)
_PUNCT_ONLY_RE = re.compile(r"^[.,;:!?…]+$")
_MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", re.DOTALL)
_SPACES_RE = re.compile(r"[ \t\N{NO-BREAK SPACE}]+")

# Heading class for a document title inserted by an extractor (docx/tex title, page title):
# it does not count as a section when anchors `§k` are numbered.
TITLE_CLASS = "source-title"
FIGURE_DESCRIPTION = "figure-description"

INLINE_TAGS = frozenset(
    {
        "Str",
        "Emph",
        "Underline",
        "Strong",
        "Strikeout",
        "Superscript",
        "Subscript",
        "SmallCaps",
        "Quoted",
        "Cite",
        "Code",
        "Space",
        "SoftBreak",
        "LineBreak",
        "Math",
        "RawInline",
        "Link",
        "Image",
        "Note",
        "Span",
    }
)
_WRAPPER_INLINES = frozenset(
    {"Emph", "Underline", "Strong", "Strikeout", "Superscript", "Subscript", "SmallCaps"}
)
_SPACE_INLINES = frozenset({"Space", "SoftBreak", "LineBreak"})
# What may follow a bare anchor without Pandoc reading it as a link, span or definition.
_UNSAFE_AFTER_ANCHOR = frozenset(":([{")
_UNSAFE_NEXT_INLINES = frozenset({"Link", "Span", "Cite", "RawInline"})


class BlocksError(Exception):
    """Pandoc could not read or write the document (message in Russian)."""


@dataclass
class HeadingInfo:
    """A heading of the body: location headings (`## [[P1:p3]] Страница 3`) included."""

    level: int
    text: str  # heading text without the location anchor
    anchor: str  # "P1:p3" for a location heading, else the anchor of its block
    block_id: str | None = None  # None for location headings (they are not blocks)
    location: bool = False
    is_title: bool = False  # the document title (class source-title or a leading top heading)


@dataclass
class SourceDoc:
    source_id: str
    blocks: list[Block]
    headings: list[HeadingInfo]
    body: str  # Markdown body of source.md with block comments, ends with a newline
    anchor_mode: str = "location"  # "location" (headings [[X:loc]]) or "section" (§k)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- Pandoc calls


def _run_pandoc(
    pandoc: Path, args: Sequence[str], *, input_text: str | None, cwd: Path | None
) -> str:
    res = procutil.run(
        [str(pandoc), *args], input_text=input_text, cwd=cwd, timeout=PANDOC_TIMEOUT_S
    )
    if res.error:
        raise BlocksError(f"Не удалось запустить Pandoc: {res.error}")
    if res.timed_out:
        raise BlocksError(f"Pandoc не уложился в {PANDOC_TIMEOUT_S} с")
    if res.exit_code != 0:
        detail = (res.stderr or res.stdout).strip() or f"код выхода {res.exit_code}"
        raise BlocksError(f"Pandoc завершился с ошибкой: {detail[:2000]}")
    return res.stdout


def read_ast(
    pandoc: Path,
    *,
    text: str | None = None,
    path: Path | None = None,
    reader: str = READER_FORMAT,
    extra_args: Sequence[str] = (),
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Pandoc JSON AST of a text (stdin) or a file."""
    args = ["--from", reader, "--to", "json", *extra_args]
    if path is not None:
        args.append(str(path))
    out = _run_pandoc(pandoc, args, input_text=text if path is None else None, cwd=cwd)
    try:
        doc = json.loads(out)
    except json.JSONDecodeError as exc:
        raise BlocksError(f"Pandoc вернул некорректный JSON: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("blocks"), list):
        raise BlocksError("Pandoc вернул JSON без списка блоков")
    return doc


def read_markdown_ast(markdown: str, pandoc: Path) -> dict[str, Any]:
    """AST of Pandoc Markdown; anchors that look like links are escaped first."""
    return read_ast(pandoc, text=escape_anchor_syntax(markdown))


def write_markdown(doc: dict[str, Any], pandoc: Path, *, cwd: Path | None = None) -> str:
    """Pandoc Markdown of an AST after `normalize_ast`, ending with exactly one newline."""
    normalize_ast(doc)
    out = _run_pandoc(
        pandoc,
        ["--from", "json", "--to", WRITER_FORMAT, *WRITER_ARGS],
        input_text=json.dumps(doc, ensure_ascii=False),
        cwd=cwd,
    )
    out = out.replace("\r\n", "\n").strip("\n")
    return out + "\n" if out else ""


# ---------------------------------------------------------------- AST helpers


def el(tag: str, content: Any = None) -> dict[str, Any]:
    return {"t": tag} if content is None else {"t": tag, "c": content}


def raw_comment(text: str) -> dict[str, Any]:
    return el("RawBlock", ["html", text])


def _is_inline_list(node: list[Any]) -> bool:
    return bool(node) and all(isinstance(x, dict) and x.get("t") in INLINE_TAGS for x in node)


def map_inline_lists(node: Any, fn: Callable[[list[Any]], list[Any]]) -> Any:
    """Apply `fn` to every inline list of the tree (innermost lists first)."""
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = map_inline_lists(value, fn)
        return node
    if isinstance(node, list):
        mapped = [map_inline_lists(x, fn) for x in node]
        return fn(mapped) if _is_inline_list(mapped) else mapped
    return node


def walk_elements(node: Any) -> Iterable[dict[str, Any]]:
    """Every AST element (dict with "t"), depth first."""
    if isinstance(node, dict):
        if "t" in node:
            yield node
        for value in node.values():
            yield from walk_elements(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_elements(value)


def _merge_strs(inlines: list[Any]) -> list[Any]:
    out: list[Any] = []
    for x in inlines:
        if x.get("t") == "Str" and out and out[-1].get("t") == "Str":
            out[-1] = el("Str", out[-1]["c"] + x["c"])
        else:
            out.append(x)
    return out


def _safe_after(rest: str, nxt: dict[str, Any] | None) -> bool:
    if rest:
        return rest[0] not in _UNSAFE_AFTER_ANCHOR
    if nxt is None:
        return True
    tag = nxt.get("t")
    if tag == "Str":
        return not nxt["c"] or nxt["c"][0] not in _UNSAFE_AFTER_ANCHOR
    return tag not in _UNSAFE_NEXT_INLINES


def _rawify_anchors(inlines: list[Any]) -> list[Any]:
    """Anchors `[[X:loc]]` become raw Markdown, so the writer does not escape them.

    An anchor followed by `:`, `(`, `[`, `{`, a link or a span stays plain text (escaped
    by the writer): written bare, Pandoc would read it back as a link or a definition.
    """
    if not any(x.get("t") == "Str" and "[[" in x["c"] for x in inlines):
        return inlines
    merged = _merge_strs(inlines)
    out: list[Any] = []
    for i, x in enumerate(merged):
        if x.get("t") != "Str" or "[[" not in x["c"]:
            out.append(x)
            continue
        text, pos = x["c"], 0
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        for m in _ANCHOR_RE.finditer(text):
            if not _safe_after(text[m.end() :], nxt):
                continue
            if m.start() > pos:
                out.append(el("Str", text[pos : m.start()]))
            out.append(el("RawInline", ["markdown", m.group(0)]))
            pos = m.end()
        if pos < len(text):
            out.append(el("Str", text[pos:]))
    return out


def _unrawify_anchors(inlines: list[Any]) -> list[Any]:
    """Inverse of `_rawify_anchors` (a source.md read back has plain Str anchors)."""
    changed = False
    out: list[Any] = []
    for x in inlines:
        if (
            x.get("t") == "RawInline"
            and x["c"][0] == "markdown"
            and _ANCHOR_RE.fullmatch(x["c"][1])
        ):
            out.append(el("Str", x["c"][1]))
            changed = True
        else:
            out.append(x)
    return _merge_strs(out) if changed else inlines


def normalize_ast(doc: dict[str, Any]) -> dict[str, Any]:
    """Layout-only normalization before writing: content is not changed.

    Top-level Plain → Para (a Plain would glue to the next block), table column widths →
    default (pipe tables instead of grid tables for simple cells), anchors → raw Markdown.
    """
    blocks = doc["blocks"]
    for i, b in enumerate(blocks):
        if b.get("t") == "Plain":
            blocks[i] = el("Para", b["c"])
    for node in walk_elements(blocks):
        if node.get("t") == "Table":
            node["c"][2] = [[spec[0], el("ColWidthDefault")] for spec in node["c"][2]]
    doc["blocks"] = map_inline_lists(map_inline_lists(blocks, _unrawify_anchors), _rawify_anchors)
    return doc


# ---------------------------------------------------------------- plain text


def inline_text(inlines: Sequence[Any]) -> str:
    """Plain text of inlines; formulas stay LaTeX (`$…$`, `$$…$$`)."""
    parts: list[str] = []
    for x in inlines:
        tag, c = x.get("t"), x.get("c")
        if tag == "Str":
            parts.append(c)
        elif tag in _SPACE_INLINES:
            parts.append(" ")
        elif tag in _WRAPPER_INLINES:
            parts.append(inline_text(c))
        elif tag == "Quoted":
            quote = ("“", "”") if c[0].get("t") == "DoubleQuote" else ("‘", "’")
            parts.append(quote[0] + inline_text(c[1]) + quote[1])
        elif tag == "Cite":
            parts.append(inline_text(c[1]))
        elif tag == "Code":
            parts.append(c[1])
        elif tag == "Math":
            delim = "$$" if c[0].get("t") == "DisplayMath" else "$"
            parts.append(f"{delim}{c[1].strip()}{delim}")
        elif tag == "RawInline":
            fmt, text = c
            if fmt in ("tex", "latex", "markdown") or (
                fmt == "html" and not text.lstrip().startswith("<!--")
            ):
                parts.append(text if fmt != "html" else re.sub(r"<[^>]+>", "", text))
        elif tag in ("Link", "Span", "Image"):
            parts.append(inline_text(c[1]))
        elif tag == "Note":
            note = blocks_text(c)
            if note:
                parts.append(f" ({note})")
    return "".join(parts)


def _clean_line(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def _table_rows(c: list[Any]) -> list[list[Any]]:
    _attr, _caption, _specs, head, bodies, foot = c
    rows = list(head[1])
    for body in bodies:
        rows += body[2] + body[3]
    rows += foot[1]
    return rows


def block_text(b: dict[str, Any]) -> str:
    """Plain text of one block (lines joined with newlines)."""
    tag, c = b.get("t"), b.get("c")
    if tag in ("Para", "Plain"):
        return _clean_line(inline_text(c))
    if tag == "Header":
        return _clean_line(inline_text(c[2]))
    if tag == "LineBlock":
        return "\n".join(_clean_line(inline_text(line)) for line in c)
    if tag == "CodeBlock":
        return c[1]
    if tag == "RawBlock":
        fmt, text = c
        if fmt == "html" and _HTML_COMMENT_RE.match(text):
            return ""
        return text.strip()
    if tag in ("BlockQuote",):
        return blocks_text(c)
    if tag == "BulletList":
        return "\n".join(blocks_text(item) for item in c)
    if tag == "OrderedList":
        return "\n".join(blocks_text(item) for item in c[1])
    if tag == "DefinitionList":
        lines = []
        for term, defs in c:
            lines.append(_clean_line(inline_text(term)))
            lines += [blocks_text(d) for d in defs]
        return "\n".join(x for x in lines if x)
    if tag == "Table":
        lines = []
        caption = blocks_text(c[1][1])
        if caption:
            lines.append(caption)
        for row in _table_rows(c):
            cells = [_clean_line(blocks_text(cell[4]).replace("\n", " ")) for cell in row[1]]
            lines.append(" | ".join(cells))
        return "\n".join(lines)
    if tag == "Figure":
        caption = blocks_text(c[1][1])
        inner = blocks_text(c[2])
        return caption or inner
    if tag == "Div":
        return blocks_text(c[1])
    return ""


def blocks_text(blocks: Sequence[Any]) -> str:
    return "\n".join(t for t in (block_text(b) for b in blocks) if t)


def strip_math(text: str) -> str:
    return _MATH_RE.sub(" ", text)


def cyrillic_ratio(texts: Iterable[str]) -> float | None:
    """Share of Cyrillic letters among all letters outside formulas (None: no letters)."""
    cyr = total = 0
    for text in texts:
        for ch in strip_math(text):
            if ch.isalpha():
                total += 1
                if 0x0400 <= ord(ch) <= 0x04FF:
                    cyr += 1
    return round(cyr / total, 3) if total else None


# ---------------------------------------------------------------- classification


def _classes(attr: list[Any]) -> list[str]:
    return list(attr[1]) if attr and len(attr) > 1 else []


def _attr_value(attr: list[Any], key: str) -> str | None:
    for k, v in attr[2] if attr and len(attr) > 2 else []:
        if k == key:
            return v
    return None


def _is_formula_para(inlines: Sequence[Any]) -> bool:
    has_math = False
    for x in inlines:
        tag = x.get("t")
        if tag == "Math" and x["c"][0].get("t") == "DisplayMath":
            has_math = True
        elif not (tag in _SPACE_INLINES or (tag == "Str" and _PUNCT_ONLY_RE.match(x["c"]))):
            return False
    return has_math


def _is_image_para(inlines: Sequence[Any]) -> bool:
    has_image = False
    for x in inlines:
        tag = x.get("t")
        if tag == "Image":
            has_image = True
        elif (
            tag == "Link"
            and x["c"][1]
            and all(y.get("t") in ("Image", *_SPACE_INLINES) for y in x["c"][1])
        ):
            has_image = has_image or any(y.get("t") == "Image" for y in x["c"][1])
        elif tag not in _SPACE_INLINES:
            return False
    return has_image


def _div_type(attr: list[Any]) -> BlockType | None:
    for cls in _classes(attr):
        if cls in DIV_BLOCK_TYPES:
            return cls  # type: ignore[return-value]
        if cls == FIGURE_DESCRIPTION:
            return "figure"
    return None


def classify(b: dict[str, Any]) -> BlockType | None:
    """Block type of a top-level AST block; None — not a block (comments, rules, empty)."""
    tag, c = b.get("t"), b.get("c")
    if tag in ("Para", "Plain"):
        if not any(x.get("t") not in _SPACE_INLINES for x in c):
            return None
        if _is_formula_para(c):
            return "formula"
        if _is_image_para(c):
            return "figure"
        return "paragraph"
    if tag == "Header":
        return "heading"
    if tag in ("BulletList", "OrderedList", "DefinitionList"):
        return "list"
    if tag == "Table":
        return "table"
    if tag == "Figure":
        return "figure"
    if tag == "CodeBlock":
        return "code"
    if tag == "BlockQuote":
        return "quote"
    if tag == "LineBlock":
        return "paragraph"
    if tag == "RawBlock":
        fmt, text = c
        if fmt == "html" and _HTML_COMMENT_RE.match(text):
            return None
        return "paragraph" if text.strip() else None
    if tag == "Div":
        typ = _div_type(c[0])
        if typ is not None:
            return typ
        inner = [x for x in c[1] if classify(x) is not None]
        if not inner:
            return None
        return classify(inner[0]) if len(inner) == 1 else "paragraph"
    return None  # HorizontalRule, Null


def location_anchor(inlines: Sequence[Any]) -> tuple[str, str] | None:
    """("P1:p3", rest of the heading text) if the heading starts with an anchor."""
    text = _clean_line(inline_text(inlines))
    m = _LOCATION_RE.match(text)
    if not m:
        return None
    return f"{m.group(1)}:{m.group(2)}", text[m.end() :].strip()


def _is_block_comment(b: dict[str, Any]) -> bool:
    return (
        b.get("t") == "RawBlock"
        and b["c"][0] == "html"
        and BLOCK_COMMENT_RE.match(b["c"][1]) is not None
    )


def _section_level(blocks: Sequence[dict[str, Any]]) -> tuple[int | None, int | None]:
    """(top section level, index of a leading title heading) for `§k` anchors.

    The top level is the largest heading level (smallest number). A title is not a section:
    a heading with class `source-title`, or the first block when it is a heading whose level
    is above every other heading's level (a page title over `##` sections).
    """
    headers = [(i, b["c"][0]) for i, b in enumerate(blocks) if b.get("t") == "Header"]
    if not headers:
        return None, None
    title_idx: int | None = None
    for i, _level in headers:
        if TITLE_CLASS in _classes(blocks[i]["c"][1]):
            title_idx = i
            break
    if title_idx is None:
        first_content = next((i for i, b in enumerate(blocks) if classify(b) is not None), None)
        first_i, first_level = headers[0]
        rest = [lvl for i, lvl in headers[1:]]
        if first_content == first_i and rest and first_level < min(rest):
            title_idx = first_i
    levels = [lvl for i, lvl in headers if i != title_idx]
    return (min(levels) if levels else None), title_idx


# ---------------------------------------------------------------- building


def strip_block_comments(doc: dict[str, Any]) -> dict[str, Any]:
    doc["blocks"] = [b for b in doc["blocks"] if not _is_block_comment(b)]
    return doc


@dataclass
class FrontMatter:
    data: dict[str, Any]  # {} when there is no front matter or its YAML is broken
    body: str  # the text after the front matter (the whole text when there is none)
    present: bool = False  # a front matter block was found and removed from `body`
    raw: str = ""  # its YAML text
    error: str | None = None  # why the YAML could not be read (Russian)


def parse_front_matter(text: str) -> FrontMatter:
    """YAML front matter at the very top of a Markdown text.

    A block that parses to a mapping is front matter; broken YAML is front matter too
    (`error` says why, `data` is empty). A block that parses to anything else — a line of
    text between two rules — is not metadata and stays in the body.
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return FrontMatter({}, text)
    raw = m.group(1)
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (строка {mark.line + 2} файла)" if mark is not None else ""
        problem = getattr(exc, "problem", None) or type(exc).__name__
        return FrontMatter({}, text[m.end() :], present=True, raw=raw, error=f"{problem}{where}")
    if isinstance(data, dict):
        return FrontMatter(data, text[m.end() :], present=True, raw=raw)
    return FrontMatter({}, text)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """(YAML front matter, body). Invalid YAML gives {} and the body after the block."""
    fm = parse_front_matter(text)
    return fm.data, fm.body


@dataclass
class _Entry:
    block_id: str
    type: BlockType
    title: str | None
    anchor: str


def build_source_doc(markdown: str, source_id: str, *, pandoc: Path) -> SourceDoc:
    """Blocks and the source.md body of a Pandoc Markdown document (body.md or source.md).

    A YAML front matter at the top is ignored; old block comments are removed first.
    """
    _front, body = split_front_matter(markdown)
    doc = strip_block_comments(read_markdown_ast(body, pandoc))
    normalize_ast(doc)
    blocks_ast: list[dict[str, Any]] = doc["blocks"]

    location_of: dict[int, tuple[str, str]] = {}
    for i, b in enumerate(blocks_ast):
        if b.get("t") == "Header":
            loc = location_anchor(b["c"][2])
            if loc is not None:
                location_of[i] = loc
    mode = "location" if location_of else "section"
    if location_of:
        section_level, title_idx = None, None
        current = location_of[min(location_of)][0]
    else:
        section_level, title_idx = _section_level(blocks_ast)
        current = f"{source_id}:§0"
    section = 0

    # One writer call for the whole document: every block is wrapped in unique start/end
    # markers, the Markdown of a block is what lies between them.
    nonce = secrets.token_hex(6)
    end_marker = f"<!-- h0lon-{nonce}-end -->"
    marked: list[dict[str, Any]] = []
    entries: list[_Entry] = []
    headings: list[HeadingInfo] = []
    for i, b in enumerate(blocks_ast):
        if i in location_of:
            current, rest = location_of[i]
            headings.append(HeadingInfo(level=b["c"][0], text=rest, anchor=current, location=True))
            marked.append(b)
            continue
        is_header = b.get("t") == "Header"
        if is_header and i != title_idx and b["c"][0] == section_level:
            section += 1
            current = f"{source_id}:§{section}"
        typ = classify(b)
        if typ is None:
            marked.append(b)
            continue
        block_id = f"{source_id}.b{len(entries) + 1:03d}"
        title: str | None = None
        if is_header:
            title = _clean_line(inline_text(b["c"][2])) or None
            headings.append(
                HeadingInfo(
                    level=b["c"][0],
                    text=title or "",
                    anchor=current,
                    block_id=block_id,
                    is_title=i == title_idx or TITLE_CLASS in _classes(b["c"][1]),
                )
            )
        elif b.get("t") == "Div":
            title = _attr_value(b["c"][0], "title") or None
        entries.append(_Entry(block_id, typ, title, current))
        marked += [
            raw_comment(f"<!-- h0lon-{nonce} {block_id} {typ} -->"),
            b,
            raw_comment(end_marker),
        ]

    # Plain text before writing: the writer call normalizes the same objects again.
    texts = [block_text(marked[j + 1]) for j, b in enumerate(marked) if _is_start(b, nonce)]
    doc["blocks"] = marked
    written = write_markdown(doc, pandoc)
    body_out, md_by_id = _split_marked(written, nonce=nonce, end_marker=end_marker)

    blocks = [
        Block(
            id=e.block_id,
            source=source_id,
            type=e.type,
            anchor=e.anchor,
            text=text,
            md=md_by_id.get(e.block_id, ""),
            title=e.title,
        )
        for e, text in zip(entries, texts, strict=True)
    ]
    warnings = []
    missing = [e.block_id for e in entries if e.block_id not in md_by_id]
    if missing:  # pragma: no cover - would mean Pandoc dropped a raw block
        warnings.append("Pandoc потерял разметку блоков: " + ", ".join(missing[:10]))
    return SourceDoc(
        source_id=source_id,
        blocks=blocks,
        headings=headings,
        body=body_out,
        anchor_mode=mode,
        warnings=warnings,
    )


def _is_start(b: dict[str, Any], nonce: str) -> bool:
    return b.get("t") == "RawBlock" and b["c"][1].startswith(f"<!-- h0lon-{nonce} ")


def _split_marked(text: str, *, nonce: str, end_marker: str) -> tuple[str, dict[str, str]]:
    """(source.md body with final block comments, Markdown of every block by id)."""
    start_re = re.compile(rf"^<!-- h0lon-{nonce} (\S+) ([\w-]+) -->$")
    body: list[str] = []
    md_by_id: dict[str, str] = {}
    current: str | None = None
    collected: list[str] = []
    skip_blank = False
    for line in text.split("\n"):
        if skip_blank:
            skip_blank = False
            if not line.strip():
                continue
        m = start_re.match(line)
        if m:
            current, collected = m.group(1), []
            body.append(f"<!-- {m.group(1)} {m.group(2)} -->")
            continue
        if line == end_marker:
            if current is not None:
                md_by_id[current] = "\n".join(collected).strip("\n")
            current = None
            skip_blank = True  # the blank line after the marker; the one before it stays
            continue
        if current is not None:
            collected.append(line)
        body.append(line)
    out = "\n".join(body).strip("\n")
    return (out + "\n" if out else ""), md_by_id


def compose_source_md(front_matter: dict[str, Any], body: str) -> str:
    """source.md text: YAML front matter + body."""
    yaml_text = yaml.safe_dump(
        front_matter, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000
    )
    return f"---\n{yaml_text}---\n\n{body}"


# ---------------------------------------------------------------- files


def atomic_write_text(path: Path, text: str) -> None:
    """UTF-8, LF line endings, replaced atomically (readers never see half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: the target may be open in a reader for a moment
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def blocks_jsonl(blocks: Sequence[Block]) -> str:
    return "".join(json.dumps(b.to_dict(), ensure_ascii=False) + "\n" for b in blocks)


def read_blocks_jsonl(path: Path) -> list[Block]:
    out: list[Block] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Block(**json.loads(line)))
    return out
