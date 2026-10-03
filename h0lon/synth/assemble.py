"""Assembly of master.md: front matter, introduction, sections, glossary, appendices А–Г.

Contract: docs/ARCHITECTURE.md, «Сборка master.md». The body of the master stays clean: all
corrections, conflicts and editorial additions that the synthesis agents recorded in
`*.notes.json` land in the appendices only. Texts that come from notes are agent output, so
everything is sanitised before it is written: HTML comments are removed (they could fake
`src` references), table cells become single lines with escaped pipes, stray `$` are escaped,
fence lines (`:::`, ```) cannot close or open a block by accident.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from h0lon.synth.common import load_blocks, percent, stage_key
from h0lon.synth.model import MASTER_MD, BuildContext, CoverageReport, Outline, OutlineSection

if TYPE_CHECKING:
    from h0lon.sources.models import SourceRecord

ASSEMBLE_VERSION = "assemble@1"
FINAL_DIR = "final"
GLOBAL_DIR = "global"
SECTIONS_DIR = "sections"
SUBTITLE = "Мастер-конспект темы"
AUTHOR = "H0lon"
EMPTY_CORRECTIONS = "Прямых правок не было."
EMPTY_CONFLICTS = "Расхождений между источниками не обнаружено."
EMPTY_EDITORIAL = "Редакторских дополнений нет."
MAX_UNCOVERED_LISTED = 500
SNIPPET_CHARS = 80

# Human names of source kinds (title page of the template uses its own, this one is for the
# generated introduction and the coverage table).
KIND_LABELS: dict[str, str] = {
    "handwritten": "рукопись",
    "pdf-text": "PDF",
    "pdf-scan": "скан PDF",
    "slides": "слайды",
    "video": "видео",
    "audio": "аудио",
    "web": "веб-страница",
    "docx": "DOCX",
    "md": "Markdown",
    "tex": "LaTeX",
}
VERDICT_LABELS: dict[str, str] = {
    "missing": "пропущен",
    "unknown": "не разобран",
    "duplicate": "дубликат",
    "admin": "служебный",
}
EDITORIAL_TITLES: dict[str, str] = {
    "answer": "Ответ на пометку автора",
    "reconstruction": "Восстановленный пропуск",
    "clarification": "Пояснение",
}
_UNIT_KEYS = ("pages", "slides", "minutes", "frames")
_FIXED_SECTION_TITLES = {"intro": "Введение", "glossary": "Глоссарий терминов и обозначений"}

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_ID_RE = re.compile(r"^[A-Z][A-Za-z0-9]*\.b\d+$")
_ANCHOR_BARE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*:\S+$")
# Code spans and TeX math (Pandoc rules: `$` opens before a non-space and closes after one).
_PROTECTED_RE = re.compile(
    r"(`+)(?!`).+?(?<!`)\1(?!`)"
    r"|(?<!\\)\$\$.+?(?<!\\)\$\$"
    r"|(?<!\\)\$(?![\s$])(?:[^$\\]|\\.)+?(?<![\s\\])\$(?!\d)",
    re.DOTALL,
)
_LINE_START_RE = re.compile(r"^(#{1,6}(?=\s)|>|:{3,}|[-+*](?=\s)|\d+[.)](?=\s)|\|)")
_FENCE_LINE_RE = re.compile(r"^\s*(```+|~~~+)")
_MATH_ANY_RE = re.compile(r"\$\$.+?\$\$|\$[^$]*\$", re.DOTALL)
_PLAIN_SPECIALS_RE = re.compile(r"([\\`*_\[\]<>$|~^#@])")


class AssembleError(Exception):
    """master.md cannot be assembled (message in Russian)."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


# ---------------------------------------------------------------- text safety


def _split_protected(text: str) -> list[tuple[str, bool]]:
    """Cut text into (piece, protected) where protected pieces are code spans and TeX math."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for m in _PROTECTED_RE.finditer(text):
        if m.start() > pos:
            parts.append((text[pos : m.start()], False))
        parts.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], False))
    return parts


def _escape_unprotected(text: str, *, pipes: bool) -> str:
    """Escape what would be misread outside of math and code: stray `$`, `<tag`, table pipes."""
    out: list[str] = []
    for piece, protected in _split_protected(text):
        if protected:
            out.append(piece)
            continue
        piece = re.sub(r"(?<!\\)\$", r"\\$", piece)
        piece = re.sub(r"<(?=[A-Za-z/!])", r"\\<", piece)
        if pipes:
            piece = re.sub(r"(?<!\\)\|", r"\\|", piece)
        out.append(piece)
    return "".join(out)


def _paragraphwise(text: str, fn: Any) -> str:
    """Apply `fn` to every blank-line separated paragraph (math never spans a blank line)."""
    pieces = re.split(r"(\n[ \t]*\n+)", text)
    return "".join(fn(p) if i % 2 == 0 else p for i, p in enumerate(pieces))


def _escape_text(paragraph: str) -> str:
    return _escape_unprotected(paragraph, pipes=False)


def clean_inline(value: Any, *, table: bool = False, line_start: bool = True) -> str:
    """One line of safe Markdown from an agent-written string (list item, cell, title).

    `line_start`: neutralise a leading `#`, `:::`, `- ` … (the text opens a paragraph or an
    item); titles and cells do not need it.
    """
    text = "" if value is None else str(value)
    text = _COMMENT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if line_start and not table:
        text = _LINE_START_RE.sub(lambda m: "\\" + m.group(0), text, count=1)
    return _escape_unprotected(text, pipes=table)


def clean_block(value: Any) -> str:
    """Safe multi-paragraph Markdown from an agent-written string (body of an appendix item)."""
    text = "" if value is None else str(value)
    text = _COMMENT_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    fences = [i for i, ln in enumerate(lines) if _FENCE_LINE_RE.match(ln)]
    odd_fences = len(fences) % 2 == 1
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        is_fence_line = odd_fences and i in fences
        if is_fence_line or re.match(r":{3,}|#{1,6}(?=\s)", stripped):
            lines[i] = "\\" + stripped
    # fenced code blocks keep their content; the rest is escaped paragraph by paragraph
    segments: list[tuple[bool, list[str]]] = [(False, [])]
    in_code = False
    for ln in lines:
        if _FENCE_LINE_RE.match(ln):
            if not in_code:
                segments.append((True, [ln]))
                in_code = True
            else:
                segments[-1][1].append(ln)
                segments.append((False, []))
                in_code = False
        else:
            segments[-1][1].append(ln)
    parts = [
        "\n".join(seg) if code else _paragraphwise("\n".join(seg), _escape_text)
        for code, seg in segments
        if seg
    ]
    return "\n".join(parts).strip("\n")


def plain(value: Any) -> str:
    """Text that must stay literal (file names, titles): Markdown specials are backslashed."""
    text = re.sub(r"\s+", " ", _COMMENT_RE.sub("", "" if value is None else str(value))).strip()
    return _PLAIN_SPECIALS_RE.sub(r"\\\1", text)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Short literal excerpt of a block text: formulas are dropped (raw TeX would leak)."""
    s = _MATH_ANY_RE.sub(" … ", _COMMENT_RE.sub("", text or ""))
    s = re.sub(r"\s+", " ", s.replace("\\", " ")).strip()
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return plain(s)


def attr_value(value: Any) -> str:
    """Value for a fenced-div attribute: `title="…"` (quotes escaped, no trailing backslash)."""
    text = clean_inline(value, line_start=False).replace('"', '\\"')
    return re.sub(r"[\\\s]+$", "", text)


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[str]], widths: Sequence[int]) -> str:
    """Pipe table; dash counts of the separator line set the relative column widths."""
    seps = []
    for w, h in zip(widths, headers, strict=True):
        dashes = "-" * max(3, w)
        seps.append(dashes + ":" if h.startswith(">") else ":" + dashes)
    heads = [h.lstrip(">") for h in headers]
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join(seps) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- notes


def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _anchor(value: Any) -> str:
    return _s(value).removeprefix("[[").removesuffix("]]").strip()


def _norm_correction(rec: Mapping[str, Any], section: str) -> dict[str, Any]:
    return {
        "section": _s(rec.get("section")) or section,
        "anchor": _anchor(rec.get("anchor")),
        "block": _s(rec.get("block")),
        "as_written": _s(rec.get("as_written")),
        "corrected": _s(rec.get("corrected")),
        "reason": _s(rec.get("reason")),
    }


def _norm_conflict(rec: Mapping[str, Any], section: str) -> dict[str, Any]:
    variants = []
    for v in rec.get("variants") or []:
        if isinstance(v, Mapping):
            variants.append({"anchor": _anchor(v.get("anchor")), "text": _s(v.get("text"))})
        elif v:
            variants.append({"anchor": "", "text": _s(v)})
    resolution = rec.get("resolution")
    return {
        "section": _s(rec.get("section")) or section,
        "topic": _s(rec.get("topic")),
        "variants": variants,
        "resolution": _s(resolution) or None,
        "reason": _s(rec.get("reason")),
    }


def _norm_editorial(rec: Mapping[str, Any], section: str) -> dict[str, Any]:
    kind = _s(rec.get("kind"))
    return {
        "section": _s(rec.get("section")) or section,
        "kind": kind if kind in EDITORIAL_TITLES else "clarification",
        "refers_to": _s(rec.get("refers_to")),
        "text": _s(rec.get("text")),
    }


_NORMALIZERS = {
    "corrections": _norm_correction,
    "conflicts": _norm_conflict,
    "editorial": _norm_editorial,
}


def _notes_files(ctx: BuildContext) -> list[tuple[Path, str]]:
    """(file, section id) of S2 notes first, then the global pass notes (section id "")."""
    found: list[tuple[Path, str]] = []
    sections = ctx.synth_dir / SECTIONS_DIR
    if sections.is_dir():
        for p in sorted(sections.glob("*.notes.json")):
            found.append((p, p.name.removesuffix(".notes.json")))
    glob_notes = ctx.synth_dir / GLOBAL_DIR / "global.notes.json"
    if glob_notes.is_file():
        found.append((glob_notes, ""))
    return found


def collect_notes(
    ctx: BuildContext,
    *,
    outline: Outline | None = None,
    warnings: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """corrections / conflicts / editorial of all notes files, every record with `section`.

    Identical records are listed once. With `outline` the records follow the section order.
    An unreadable file is skipped; the reason is appended to `warnings`.
    """
    result: dict[str, list[dict[str, Any]]] = {k: [] for k in _NORMALIZERS}
    seen: set[str] = set()
    for path, section in _notes_files(ctx):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            if warnings is not None:
                warnings.append(f"Заметки {path.name} не прочитаны: {exc}")
            continue
        if not isinstance(data, Mapping):
            if warnings is not None:
                warnings.append(f"Заметки {path.name}: ожидался JSON-объект")
            continue
        for kind, normalize in _NORMALIZERS.items():
            for rec in data.get(kind) or []:
                if not isinstance(rec, Mapping):
                    continue
                item = normalize(rec, section)
                # The journal has no section column, so the same correction made in two
                # sections is one row; conflicts and additions name their section.
                shown = {k: v for k, v in item.items() if k != "section" or kind != "corrections"}
                key = kind + json.dumps(shown, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                result[kind].append(item)
    if outline is not None:
        rank = {s.id: i for i, s in enumerate(outline.sections)}
        for items in result.values():
            items.sort(key=lambda r: rank.get(r["section"], len(rank)))
    return result


# ---------------------------------------------------------------- sources


def source_units(record: SourceRecord) -> dict[str, int | float]:
    """Units for the title page: only known keys, no zeros; slides count slides, not pages."""
    units = {
        k: v
        for k, v in (record.units or {}).items()
        if k in _UNIT_KEYS and isinstance(v, int | float) and v
    }
    if record.kind == "slides" and "slides" in units:
        units.pop("pages", None)
    return units


def _volume_text(units: Mapping[str, Any]) -> str:
    labels = {"pages": "стр.", "slides": "сл.", "minutes": "мин", "frames": "кадр."}
    return ", ".join(f"{units[k]:g} {labels[k]}" for k in _UNIT_KEYS if k in units)


def _source_line(record: SourceRecord) -> str:
    kind = KIND_LABELS.get(record.kind, record.kind)
    volume = _volume_text(source_units(record))
    head = f"{kind}, {volume}" if volume else kind
    return f"**{plain(record.id)}** ({head}) — {plain(record.title)}"


# ---------------------------------------------------------------- front matter, intro


def _front_matter(*, title: str, course: str, today: str, sources: Sequence[SourceRecord]) -> str:
    data: dict[str, Any] = {"title": title, "subtitle": SUBTITLE}
    if course:
        data["course"] = course
    data["date"] = today
    data["author"] = AUTHOR
    data["sources"] = [
        {
            "id": r.id,
            "kind": r.kind,
            "title": re.sub(r"\s+", " ", r.title).strip(),
            "units": source_units(r),
        }
        for r in sources
    ]
    body = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False, width=10_000
    )
    return f"---\n{body}---\n"


def _minimal_intro(title: str, sources: Sequence[SourceRecord]) -> str:
    n = len(sources)
    lines = [
        "# Введение {#sec:intro .unnumbered}",
        "",
        f"Мастер-конспект темы «{plain(title)}» собран из источников ({n}):",
        "",
    ]
    lines += [f"- {_source_line(r)}" for r in sources]
    lines += [
        "",
        "Материал изложен по логике предмета, а не по порядку источников; у каждого "
        "абзаца есть ссылка на место в источнике. Расхождения между источниками, правки "
        "описок и редакторские дополнения вынесены в приложения.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- appendices


class _Links:
    """Section links and block anchors for the appendices."""

    def __init__(self, ctx: BuildContext, outline: Outline) -> None:
        self.by_id: dict[str, OutlineSection] = {s.id: s for s in outline.sections}
        self._ctx = ctx
        self._blocks: dict[str, Any] | None = None

    def blocks(self) -> dict[str, Any]:
        if self._blocks is None:
            try:
                self._blocks = load_blocks(self._ctx.topic_dir)
            except (OSError, ValueError, TypeError):
                self._blocks = {}
        return self._blocks

    def section(self, section_id: str) -> str:
        """`[«Название»](#sec:id)` for a known section, "" otherwise."""
        sec = self.by_id.get(section_id)
        title = sec.title if sec else _FIXED_SECTION_TITLES.get(section_id)
        if not title:
            return ""
        text = clean_inline(title, line_start=False).replace("[", "\\[").replace("]", "\\]")
        return f"[«{text}»](#sec:{section_id})"

    def reference(self, value: str) -> str:
        """Markdown for a block id (code + anchor) or an anchor (`[[P1:p3]]`) or free text."""
        value = value.strip()
        if not value:
            return ""
        if _BLOCK_ID_RE.match(value):
            block = self.blocks().get(value)
            anchor = f" [[{block.anchor}]]" if block is not None and block.anchor else ""
            return f"`{value}`{anchor}"
        bare = _anchor(value)
        if _ANCHOR_BARE_RE.match(bare):
            return f"[[{bare}]]"
        return clean_inline(value)


def _anchor_md(anchor: str) -> str:
    return f"[[{anchor}]]" if anchor and _ANCHOR_BARE_RE.match(anchor) else ""


def _conflicts_appendix(conflicts: Sequence[Mapping[str, Any]], links: _Links) -> str:
    head = "# Расхождения между источниками {#app:conflicts .appendix}"
    if not conflicts:
        return f"{head}\n\n{EMPTY_CONFLICTS}\n"
    out = [head, ""]
    intro = (
        "Здесь собраны места, где источники расходятся. В тексте конспекта принят один "
        "вариант; остальные и причина выбора приведены ниже."
    )
    out += [intro, ""]
    for c in conflicts:
        topic = c["topic"] or "Расхождение источников"
        out.append(f'::: {{.conflict title="{attr_value(topic)}"}}')
        link = links.section(c["section"])
        if link:
            out += [f"Раздел {link}.", ""]
        variants = []
        for v in c["variants"]:
            anchor, text = _anchor_md(v["anchor"]), clean_inline(v["text"])
            variants.append(f"- {anchor}: {text}" if anchor and text else f"- {anchor or text}")
        if variants:
            out += [*variants, ""]
        if c["resolution"]:
            decision = f"В тексте принят вариант: {clean_inline(c['resolution'])}."
        else:
            decision = "В тексте дана нейтральная формулировка: выбрать вариант нельзя."
        if c["reason"]:
            decision += f" Причина: {clean_inline(c['reason'])}"
        out += [decision, ":::", ""]
    return "\n".join(out)


def _corrections_appendix(corrections: Sequence[Mapping[str, Any]], links: _Links) -> str:
    head = "# Журнал правок {#app:corrections}"
    if not corrections:
        return f"{head}\n\n{EMPTY_CORRECTIONS}\n"
    rows = []
    for c in corrections:
        where = _anchor_md(c["anchor"]) or (f"`{c['block']}`" if c["block"] else "—")
        rows.append(
            [
                where,
                clean_inline(c["as_written"], table=True) or "—",
                clean_inline(c["corrected"], table=True) or "—",
                clean_inline(c["reason"], table=True) or "—",
            ]
        )
    table = _md_table(
        ["Якорь", "Как написано", "Как исправлено", "Причина"], rows, [14, 30, 30, 26]
    )
    note = (
        "Исправлены однозначные описки и мелкие несогласованности; содержательные "
        "изменения в журнал не входят."
    )
    return f"{head}\n\n{note}\n\n{table}\n"


def _editorial_appendix(editorial: Sequence[Mapping[str, Any]], links: _Links) -> str:
    head = "# Редакторские дополнения {#app:editorial}"
    if not editorial:
        return f"{head}\n\n{EMPTY_EDITORIAL}\n"
    out = [head, ""]
    out += [
        "Ответы на пометки авторов, восстановленные пропуски и пояснения. В тело конспекта "
        "они не входят.",
        "",
    ]
    for e in editorial:
        title = EDITORIAL_TITLES.get(e["kind"], EDITORIAL_TITLES["clarification"])
        out.append(f'::: {{.editorial title="{attr_value(title)}"}}')
        refs = []
        link = links.section(e["section"])
        if link:
            refs.append(f"Раздел {link}")
        ref = links.reference(e["refers_to"])
        if ref:
            refs.append(f"относится к {ref}")
        if refs:
            out += ["; ".join(refs) + ".", ""]
        text = clean_block(e["text"])
        out += [text or "—", ":::", ""]
    return "\n".join(out)


def _block_sort_key(block_id: str) -> tuple[str, int]:
    m = re.match(r"^([A-Z][A-Za-z0-9]*)\.b(\d+)$", block_id)
    return (m.group(1), int(m.group(2))) if m else (block_id, 0)


def _coverage_appendix(
    coverage: CoverageReport, sources: Sequence[SourceRecord], links: _Links
) -> str:
    head = "# Карта покрытия {#app:coverage}"
    titles = {r.id: r for r in sources}
    order = [r.id for r in sources] + [s for s in coverage.by_source if s not in titles]
    rows = []
    for sid in dict.fromkeys(order):
        stat = coverage.by_source.get(sid)
        if stat is None:
            continue
        total, covered = int(stat.get("total", 0)), int(stat.get("covered", 0))
        pct = percent(covered, total)
        rec = titles.get(sid)
        label = f"{plain(sid)} — {plain(rec.title)}" if rec else plain(sid)
        rows.append([label, str(total), str(covered), pct])
    pct_all = percent(coverage.covered, coverage.total)
    rows.append(["**Всего**", f"**{coverage.total}**", f"**{coverage.covered}**", f"**{pct_all}**"])
    table = _md_table(["Источник", ">Блоков", ">Покрыто", ">%"], rows, [46, 9, 9, 7])
    intro = (
        "Карта показывает, какая доля блоков источников учтена в тексте мастера. "
        "Служебные блоки (даты, объявления, титульные слайды) в подсчёт не входят."
    )
    out = [head, "", intro, "", table, "", "## Непокрытые блоки {#app:uncovered .unnumbered}", ""]
    uncovered = sorted(
        coverage.uncovered,
        key=lambda u: (
            list(VERDICT_LABELS).index(u.get("verdict", "unknown"))
            if u.get("verdict") in VERDICT_LABELS
            else 1,
            _block_sort_key(str(u.get("block", ""))),
        ),
    )
    if not uncovered:
        out.append("Непокрытых блоков нет: на каждый блок источников есть ссылка в тексте мастера.")
        return "\n".join(out) + "\n"
    out.append(
        "Блоки без ссылки в тексте мастера и вердикт проверки полноты "
        "(дубликаты и служебные блоки в покрытие засчитаны):"
    )
    out.append("")
    blocks = links.blocks()
    for u in uncovered[:MAX_UNCOVERED_LISTED]:
        bid = str(u.get("block", "")).strip()
        verdict = VERDICT_LABELS.get(str(u.get("verdict", "unknown")), "не разобран")
        block = blocks.get(bid)
        where = f" [[{block.anchor}]]" if block is not None and block.anchor else ""
        note = clean_inline(u.get("note"))
        excerpt = snippet(block.text) if block is not None else ""
        tail = []
        if note:
            tail.append(note)
        if excerpt:
            tail.append(f"«{excerpt}»")
        line = f"- `{bid}`{where} — {verdict}"
        if tail:
            line += ": " + " ".join(tail)
        out.append(line)
    if len(uncovered) > MAX_UNCOVERED_LISTED:
        out += [
            "",
            f"… и ещё {len(uncovered) - MAX_UNCOVERED_LISTED} блоков (список — в coverage.json).",
        ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- body


def _has_heading(text: str, section_id: str) -> bool:
    pattern = re.compile(rf"^#{{1,6}}[ \t].*\{{[^}}]*#sec:{re.escape(section_id)}[\s}}]", re.M)
    return bool(pattern.search(text))


def _section_text(ctx: BuildContext, section: OutlineSection, has_children: bool) -> str:
    path = ctx.synth_dir / FINAL_DIR / SECTIONS_DIR / f"{section.id}.md"
    heading = f"{'#' * max(1, min(section.level, 6))} {clean_inline(section.title)} " + (
        f"{{#sec:{section.id}}}"
    )
    if path.is_file():
        text = _read(path).strip("\n")
        if text.strip():
            return text if _has_heading(text, section.id) else f"{heading}\n\n{text}"
    if has_children or not section.blocks:
        return heading  # chapter whose text lives in its sub-sections, or an empty leaf
    raise AssembleError(
        f"Нет текста раздела {section.id} «{section.title}»: ожидался файл "
        f"{path.relative_to(ctx.topic_dir).as_posix()}"
    )


def _optional_text(path: Path) -> str:
    if path.is_file():
        text = _read(path).strip("\n")
        if text.strip():
            return text
    return ""


def assemble_key(ctx: BuildContext, sources: Sequence[SourceRecord]) -> str:
    """Cache key of the assemble stage: contents of everything the master is built from."""
    synth = ctx.synth_dir
    files: list[Path] = [
        synth / "outline.json",
        synth / "coverage.json",
        synth / FINAL_DIR / "intro.md",
        synth / FINAL_DIR / "glossary.md",
        synth / GLOBAL_DIR / "global.notes.json",
    ]
    for folder, pattern in (
        (synth / FINAL_DIR / SECTIONS_DIR, "*.md"),
        (synth / SECTIONS_DIR, "*.notes.json"),
    ):
        if folder.is_dir():
            files += sorted(folder.glob(pattern))
    parts: list[Any] = [ASSEMBLE_VERSION]
    for p in files:
        parts += [p.relative_to(ctx.topic_dir).as_posix(), p]
    parts.append(
        [{"id": r.id, "kind": r.kind, "title": r.title, "units": source_units(r)} for r in sources]
    )
    try:
        from h0lon.workspace import load_topic

        meta = load_topic(ctx.topic_dir)
        parts.append({"title": meta.title, "course": meta.course})
    except (OSError, ValueError):
        parts.append(None)
    return stage_key(*parts)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def assemble_master(
    ctx: BuildContext,
    outline: Outline,
    coverage: CoverageReport,
    sources: Sequence[SourceRecord],
    *,
    today: str | None = None,
    warnings: list[str] | None = None,
) -> Path:
    """Write <topic>/master.md and return its path (see the module docstring)."""
    title = re.sub(r"\s+", " ", outline.title or "").strip()
    course = ""
    try:
        from h0lon.workspace import load_topic

        meta = load_topic(ctx.topic_dir)
        title = title or meta.title
        course = meta.course
    except (OSError, ValueError):
        pass
    title = title or ctx.topic_dir.name

    parts: list[str] = [
        _front_matter(
            title=title,
            course=course,
            today=today or date.today().isoformat(),
            sources=sources,
        ).rstrip("\n")
    ]
    intro = _optional_text(ctx.synth_dir / FINAL_DIR / "intro.md")
    parts.append(intro or _minimal_intro(title, sources))

    parents = {s.parent for s in outline.sections if s.parent}
    for section in outline.sections:
        parts.append(_section_text(ctx, section, section.id in parents))

    glossary = _optional_text(ctx.synth_dir / FINAL_DIR / "glossary.md")
    if glossary:
        parts.append(glossary)

    notes = collect_notes(ctx, outline=outline, warnings=warnings)
    links = _Links(ctx, outline)
    parts.append(_conflicts_appendix(notes["conflicts"], links).strip("\n"))
    parts.append(_corrections_appendix(notes["corrections"], links).strip("\n"))
    parts.append(_editorial_appendix(notes["editorial"], links).strip("\n"))
    parts.append(_coverage_appendix(coverage, sources, links).strip("\n"))

    text = "\n\n".join(p.strip("\n") for p in parts if p.strip()) + "\n"
    target = ctx.topic_dir / MASTER_MD
    _write_atomic(target, text)
    return target
