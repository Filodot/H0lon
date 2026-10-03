"""Pandoc invocation: master Markdown → LaTeX / HTML with the H0lon Lua filters."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from h0lon import procutil
from h0lon.render.templates import FILTERS_DIR

# PRD 5.4: Pandoc Markdown with these extensions (most are on by default; listed explicitly).
# autolink_bare_uris: bare URLs become \url{...}, so xurl can break them at the margin.
MARKDOWN_FORMAT = (
    "markdown+fenced_divs+tex_math_dollars+pipe_tables+footnotes+implicit_figures"
    "+link_attributes+header_attributes+autolink_bare_uris"
)
# sanitize.lua acts only on HTML output and must see the document before the others.
FILTER_NAMES = ("sanitize.lua", "blocks.lua", "srcrefs.lua", "paths.lua")
DEFAULT_TOC_DEPTH = 2
PANDOC_TIMEOUT_S = 180

_FRONT_MATTER_RE = re.compile(
    r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_FENCE_RE = re.compile(r"^(```+|~~~+)")
_ATX_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S")
_ANCHOR_SRC = r"\[\[([A-Z][A-Za-z0-9]*:[^\[\]\s]+)\]\]"
# Before ":" an anchor may form a link reference definition ("[[H1:p1]]: текст" opening a
# paragraph silently disappears); before "(", "[" or "{" it becomes a link or a span. Such
# anchors are escaped (\[\[H1:p1\]\]), so Pandoc reads them as text, as srcrefs.lua expects.
_AMBIGUOUS_ANCHOR_RE = re.compile(_ANCHOR_SRC + r"(?=[:(\[{])")
_ESCAPED_ANCHOR = r"\\[\\[\1\\]\\]"
_CODE_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)")
# Title pieces that reach the PDF verbatim: formulas and characters Pandoc typesets
# differently (quotes become «», -- a dash, ... an ellipsis) split the title.
_TITLE_SPLIT_RE = re.compile(r"\$[^$]*\$|[\"'«»„“”]|-{2,}|\.{3}")
_WORD_RE = re.compile(r"\w")
# pandoc.log.warn() in a filter: "Scripting warning at <file> line 1 column 1: <message>".
_SCRIPT_WARNING_RE = re.compile(r"^Scripting warning at .*? line \d+ column \d+: ")


@dataclass
class PandocResult:
    ok: bool
    argv: list[str]
    output: Path | None = None
    stdout: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def filter_paths() -> list[Path]:
    return [FILTERS_DIR / name for name in FILTER_NAMES]


def read_front_matter(text: str) -> dict[str, Any]:
    """YAML metadata block at the very top of the document ({} if absent or invalid)."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _split_front_matter(text: str) -> tuple[str, str]:
    m = _FRONT_MATTER_RE.match(text)
    return (text[: m.end()], text[m.end() :]) if m else ("", text)


def _body_lines(body: str) -> Iterator[tuple[str, bool]]:
    """(line with its line ending, belongs to a fenced code block) for every line."""
    fence: str | None = None
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        m = _FENCE_RE.match(stripped)
        if m:
            marker = m.group(1)[0] * 3
            if fence is None:
                fence = marker
            elif stripped.startswith(fence):
                fence = None
            yield line, True
            continue
        yield line, fence is not None


def has_headings(text: str) -> bool:
    """True if the Markdown body has an ATX heading outside front matter and code blocks."""
    _, body = _split_front_matter(text)
    return any(not code and _ATX_HEADING_RE.match(line) for line, code in _body_lines(body))


def _escape_outside_code_spans(line: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _CODE_SPAN_RE.finditer(line):
        out.append(_AMBIGUOUS_ANCHOR_RE.sub(_ESCAPED_ANCHOR, line[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_AMBIGUOUS_ANCHOR_RE.sub(_ESCAPED_ANCHOR, line[pos:]))
    return "".join(out)


def escape_anchor_syntax(text: str) -> str:
    """Escape source anchors that Pandoc would read as links or link definitions.

    Front matter, fenced code blocks and inline code spans are left untouched.
    """
    front, body = _split_front_matter(text)
    if "[[" not in body:
        return text
    lines = [
        line if code or "[[" not in line else _escape_outside_code_spans(line)
        for line, code in _body_lines(body)
    ]
    return front + "".join(lines)


def _title_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    text = re.sub(r"\[\[[^\]]*\]\]", "", str(value))
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) -> text


def _strip_markup(text: str) -> str:
    text = re.sub(r"[*_`~\\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def plain_title(value: object) -> str:
    """Front-matter title as plain text: no anchors, markup characters or LaTeX math."""
    return _strip_markup(re.sub(r"\$[^$]*\$", " ", _title_text(value)))


def title_fragments(value: object) -> list[str]:
    """Pieces of the title that must appear verbatim in the text of the PDF."""
    parts = (_strip_markup(p) for p in _TITLE_SPLIT_RE.split(_title_text(value)))
    return [p for p in parts if _WORD_RE.search(p)]


def _meta_arg(key: str, value: object) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return f"--metadata={key}={value}"


def build_argv(
    pandoc: Path,
    *,
    src: Path | None,
    to: str,
    output: Path | None,
    template: Path | None = None,
    standalone: bool = True,
    filters: list[Path] | None = None,
    metadata: dict[str, object] | None = None,
    variables: dict[str, str] | None = None,
    toc: bool = False,
    toc_depth: int | None = None,
    number_sections: bool = False,
    resource_path: list[Path] | None = None,
    css: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    argv: list[str] = [str(pandoc)]
    if src is not None:
        argv.append(str(src))
    argv += ["--from", MARKDOWN_FORMAT, "--to", to]
    if standalone:
        argv.append("--standalone")
    if template is not None:
        argv.append(f"--template={template}")
    for flt in filters if filters is not None else filter_paths():
        argv.append(f"--lua-filter={flt}")
    if toc:
        argv.append("--toc")
        if toc_depth is not None:
            argv.append(f"--toc-depth={toc_depth}")
    if number_sections:
        argv.append("--number-sections")
    if resource_path:
        sep = ";" if procutil.IS_WINDOWS else ":"
        argv.append("--resource-path=" + sep.join(str(p) for p in resource_path))
    for key, value in (metadata or {}).items():
        argv.append(_meta_arg(key, value))
    for key, value in (variables or {}).items():
        argv.append(f"--variable={key}={value}")
    for href in css or []:
        argv.append(f"--css={href}")
    argv += list(extra_args or [])
    if output is not None:
        argv.append(f"--output={output}")
    return argv


def _warnings(stderr: str) -> list[str]:
    out: list[str] = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[WARNING]"):
            text = _SCRIPT_WARNING_RE.sub("", line[len("[WARNING]") :].strip())
            out.append("Pandoc: " + text)
        elif out and line.startswith(("  ", "\t")):
            out[-1] += " " + line.strip()
    return out


def run_pandoc(
    argv: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    output: Path | None = None,
    timeout: float = PANDOC_TIMEOUT_S,
) -> PandocResult:
    res = procutil.run(argv, cwd=cwd, input_text=input_text, timeout=timeout)
    result = PandocResult(ok=False, argv=list(argv), stdout=res.stdout, warnings=[])
    if res.error:
        result.error = f"Не удалось запустить Pandoc: {res.error}"
        return result
    if res.timed_out:
        result.error = f"Pandoc не уложился в {int(timeout)} с"
        return result
    result.warnings = _warnings(res.stderr)
    if res.exit_code != 0:
        detail = res.stderr.strip() or res.stdout.strip() or f"код выхода {res.exit_code}"
        result.error = f"Pandoc завершился с ошибкой: {detail}"
        return result
    if output is not None:
        if not output.is_file():
            result.error = f"Pandoc не создал файл {output}"
            return result
        result.output = output
    result.ok = True
    return result


def convert_text(
    markdown: str,
    *,
    pandoc: Path,
    to: str,
    template: Path | None = None,
    standalone: bool = False,
    metadata: dict[str, object] | None = None,
    variables: dict[str, str] | None = None,
    resource_dir: Path | None = None,
    number_sections: bool = False,
    toc: bool = False,
    extra_args: list[str] | None = None,
) -> PandocResult:
    """Convert a Markdown string (stdin) with the H0lon filters; output in .stdout."""
    meta = dict(metadata or {})
    if resource_dir is not None:
        meta.setdefault("h0lon-resource-dir", resource_dir.as_posix())
    argv = build_argv(
        pandoc,
        src=None,
        to=to,
        output=None,
        template=template,
        standalone=standalone,
        metadata=meta,
        variables=variables,
        toc=toc,
        number_sections=number_sections,
        extra_args=extra_args,
    )
    return run_pandoc(argv, input_text=escape_anchor_syntax(markdown))
