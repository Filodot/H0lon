"""S1 of the synthesis: the unified topic structure (docs/ARCHITECTURE.md, «Синтез (M2)»).

The code prepares the agent inputs (`summaries.md`, `blocks_index.md`, `terms.md`), runs one strong
agent with the prompt `outline@<version>.md`, checks the result and, when it breaks the contract,
repeats the run in a new bundle with a «Обратная связь по предыдущей попытке» section (the previous
files are placed into out/ as a seed, so the agent edits them in place). What the agent still gets
wrong after the retries is repaired by code when that is safe (duplicates, unknown ids, a few
unassigned blocks) and is an error of the stage otherwise.

Files: `<topic>/synthesis/inputs/{summaries,blocks_index,terms}.md`, `outline.json`, `outline.md`,
`outline.meta.json` (the stage cache).
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h0lon.agents import ExpectedFile, OutputContract
from h0lon.extract.blocks import atomic_write_text
from h0lon.extract.model import Block
from h0lon.sources.models import SourceRecord
from h0lon.synth.common import (
    SECTION_ID_RE,
    cached,
    is_admin,
    model_for,
    prompt_body,
    prompt_id,
    read_meta,
    run_agent,
    stage_key,
    write_json_atomic,
    write_meta,
)
from h0lon.synth.model import BuildContext, Outline, OutlineSection, StageResult

STAGE = "outline"
INPUTS_DIR = "inputs"
SUMMARIES_FILE = "summaries.md"
BLOCKS_INDEX_FILE = "blocks_index.md"
TERMS_FILE = "terms.md"
OUTLINE_JSON = "outline.json"
OUTLINE_MD = "outline.md"

INDEX_TEXT_CHARS = 160  # text of a block in blocks_index.md
MAX_RETRIES = 2  # repeated runs after the first one (new bundle, feedback, seed)
# Unassigned blocks left after the retries are added by code only when there are few of them.
AUTOFILL_MIN = 5
AUTOFILL_RATIO = 0.03
UNASSIGNED_WARN_RATIO = 0.15  # warn when more non-admin blocks than this share are «unassigned»
MAX_LISTED = 40  # ids listed in one problem line

_BLOCK_ID = {"type": "string", "pattern": r"^[A-Z][0-9]+\.b[0-9]{3,}$"}
OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "sections"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "sections": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "title", "level", "blocks"],
                "properties": {
                    "id": {"type": "string", "pattern": r"^s[0-9]{2}(-[0-9]{2})?$"},
                    "title": {"type": "string", "minLength": 1},
                    "level": {"type": "integer", "enum": [1, 2]},
                    "summary": {"type": "string"},
                    "blocks": {"type": "array", "items": _BLOCK_ID},
                    "parent": {"type": ["string", "null"]},
                },
            },
        },
        "unassigned": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["block", "reason"],
                "properties": {"block": _BLOCK_ID, "reason": {"type": "string", "minLength": 1}},
            },
        },
        "conflict_hints": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic", "blocks"],
                "properties": {
                    "topic": {"type": "string"},
                    "blocks": {"type": "array", "items": _BLOCK_ID},
                    "note": {"type": "string"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------- small helpers


def inputs_dir(ctx: BuildContext) -> Path:
    return ctx.synth_dir / INPUTS_DIR


def _as_mapping(blocks: Mapping[str, Block] | Iterable[Block]) -> Mapping[str, Block]:
    if isinstance(blocks, Mapping):
        return blocks
    return {b.id: b for b in blocks}


def _write(path: Path, text: str) -> None:
    atomic_write_text(path, text if text.endswith("\n") else text + "\n")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _listed(ids: Sequence[str], limit: int = MAX_LISTED) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" … (и ещё {len(ids) - limit})" if len(ids) > limit else "")


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_H2_RE = re.compile(r"^##\s+(\S.*?)\s*$")
_ATX_RE = re.compile(r"^#{1,5}\s")
_LEADING_COMMENT_RE = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)


def _level2_sections(text: str) -> dict[str, str]:
    """`## Title` sections of a Markdown text as title -> body (fenced code is respected)."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := _H2_RE.match(line)):
            current = m.group(1)
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {title: "\n".join(lines).strip() for title, lines in sections.items()}


def _demote_headings(text: str) -> str:
    """One level deeper (## → ###), so that source headers in summaries.md stay the top level."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and _ATX_RE.match(line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def read_summary(topic_dir: Path, source_id: str) -> str | None:
    """Text of extracted/<ID>/summary.md without the leading service comment, or None."""
    path = topic_dir / "extracted" / source_id / "summary.md"
    if not path.is_file():
        return None
    text = _LEADING_COMMENT_RE.sub("", path.read_text(encoding="utf-8-sig"), count=1).strip()
    return text or None


# ---------------------------------------------------------------- inputs of the agent


@dataclass
class _SourceBlocks:
    id: str
    title: str
    kind: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def header(self) -> str:
        return f"{self.id} — {_one_line(self.title)} ({self.kind})"


def _grouped(blocks: Mapping[str, Block], sources: Sequence[SourceRecord]) -> list[_SourceBlocks]:
    """Sources that have blocks, in topic.yaml order (sources unknown to topic.yaml last)."""
    by_source: dict[str, list[Block]] = {}
    for block in blocks.values():
        by_source.setdefault(block.source, []).append(block)
    groups: list[_SourceBlocks] = []
    for rec in sources:
        if rec.id in by_source:
            groups.append(_SourceBlocks(rec.id, rec.title, rec.kind, by_source.pop(rec.id)))
    groups += [_SourceBlocks(sid, sid, "?", bl) for sid, bl in sorted(by_source.items())]
    return groups


def _headings_toc(blocks: Sequence[Block], limit: int = 300) -> str:
    """Table of contents from heading blocks, when the source has no summary.md."""
    items: list[tuple[int, str, str]] = []
    for block in blocks:
        if block.type != "heading":
            continue
        title = _one_line(block.title or block.text or "")
        if not title:
            continue
        m = re.match(r"(#{1,6})\s", block.md or "")
        items.append((len(m.group(1)) if m else 1, title, block.anchor))
    if not items:
        return "- (заголовков в источнике нет)"
    base = min(level for level, _, _ in items)
    lines = [f"{'  ' * (level - base)}- {title} [[{anchor}]]" for level, title, anchor in items]
    if len(lines) > limit:
        lines = [*lines[:limit], f"- … (и ещё {len(lines) - limit} заголовков)"]
    return "\n".join(lines)


def build_summaries_md(topic_dir: Path, groups: Sequence[_SourceBlocks]) -> str:
    parts = ["# Источники темы: аннотации, оглавления, термины", ""]
    for group in groups:
        parts += [f"## {group.header}", ""]
        summary = read_summary(topic_dir, group.id)
        if summary is not None:
            parts += [_demote_headings(summary), ""]
        else:
            parts += [
                "*Файла summary.md нет: оглавление построено из заголовков блоков.*",
                "",
                "### Оглавление",
                "",
                _headings_toc(group.blocks),
                "",
            ]
    return "\n".join(parts)


def index_line(block: Block) -> str:
    """`P1.b007 [definition] (P1:p3) начало текста` — one line of blocks_index.md."""
    text = _one_line(block.text or block.title or block.md or "") or "(пусто)"
    if len(text) > INDEX_TEXT_CHARS:
        text = text[:INDEX_TEXT_CHARS].rstrip() + "…"
    if is_admin(block):
        text = f"(служебный блок) {text}"
    return f"{block.id} [{block.type}] ({block.anchor}) {text}"


def build_blocks_index_md(groups: Sequence[_SourceBlocks]) -> str:
    return "\n".join(index_line(b) for g in groups for b in g.blocks) + "\n"


def build_terms_md(topic_dir: Path, groups: Sequence[_SourceBlocks]) -> str:
    parts = ["# Термины и обозначения источников", ""]
    found = False
    for group in groups:
        summary = read_summary(topic_dir, group.id)
        if summary is None:
            continue
        body = next(
            (
                text
                for title, text in _level2_sections(summary).items()
                if title.casefold().startswith("термин")
            ),
            "",
        )
        if body:
            found = True
            parts += [f"## {group.header}", "", body, ""]
    if not found:
        parts += ["В источниках не найдено разделов «Термины и обозначения».", ""]
    return "\n".join(parts)


def prepare_inputs(
    ctx: BuildContext,
    blocks: Mapping[str, Block] | Iterable[Block],
    sources: Sequence[SourceRecord],
) -> dict[str, Path]:
    """Write synthesis/inputs/{summaries,blocks_index,terms}.md; returns their paths by name."""
    mapping = _as_mapping(blocks)
    groups = _grouped(mapping, sources)
    root = inputs_dir(ctx)
    paths = {
        "summaries": root / SUMMARIES_FILE,
        "blocks_index": root / BLOCKS_INDEX_FILE,
        "terms": root / TERMS_FILE,
    }
    _write(paths["summaries"], build_summaries_md(ctx.topic_dir, groups))
    _write(paths["blocks_index"], build_blocks_index_md(groups))
    _write(paths["terms"], build_terms_md(ctx.topic_dir, groups))
    return paths


def ensure_terms(ctx: BuildContext, blocks: Mapping[str, Block] | Iterable[Block]) -> Path:
    """inputs/terms.md, built from the summaries when S1 has not written it (e.g. S2 alone)."""
    path = inputs_dir(ctx) / TERMS_FILE
    if path.is_file():
        return path
    try:
        from h0lon.sources.ingest import list_sources

        sources = list_sources(ctx.topic_dir)
    except Exception:  # no topic.yaml: the terms are still taken from extracted/
        sources = []
    _write(path, build_terms_md(ctx.topic_dir, _grouped(_as_mapping(blocks), sources)))
    return path


# ---------------------------------------------------------------- outline files


def parse_outline(data: Mapping[str, Any]) -> Outline:
    """Outline from the (schema-checked) JSON; unknown fields are dropped, gaps get defaults."""
    sections: list[OutlineSection] = []
    for raw in data.get("sections") or []:
        if not isinstance(raw, Mapping):
            continue
        parent = raw.get("parent")
        sections.append(
            OutlineSection(
                id=str(raw.get("id", "")),
                title=_one_line(str(raw.get("title", ""))),
                level=int(raw.get("level") or 0),
                summary=_one_line(str(raw.get("summary") or "")),
                blocks=[str(b) for b in raw.get("blocks") or []],
                parent=str(parent) if parent else None,
            )
        )
    unassigned = [
        {"block": str(u.get("block", "")), "reason": _one_line(str(u.get("reason") or ""))}
        for u in data.get("unassigned") or []
        if isinstance(u, Mapping)
    ]
    hints = [
        {
            "topic": _one_line(str(h.get("topic", ""))),
            "blocks": [str(b) for b in h.get("blocks") or []],
            "note": _one_line(str(h.get("note") or "")),
        }
        for h in data.get("conflict_hints") or []
        if isinstance(h, Mapping)
    ]
    return Outline(
        title=_one_line(str(data.get("title", ""))),
        sections=sections,
        unassigned=unassigned,
        conflict_hints=hints,
    )


def load_outline(ctx: BuildContext) -> Outline:
    """synthesis/outline.json as an Outline. FileNotFoundError / ValueError with Russian text."""
    path = ctx.synth_dir / OUTLINE_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Структура темы не найдена: {path} (стадия outline не выполнена)")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: некорректный JSON ({exc.msg}, строка {exc.lineno})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError(f"{path}: нет списка sections")
    return parse_outline(data)


def render_outline_md(outline: Outline) -> str:
    """Outline for humans and for agents: numbering, ids (for `#sec:<id>` links), block counts."""
    children: dict[str, list[OutlineSection]] = {}
    for s in outline.sections:
        if s.level == 2 and s.parent:
            children.setdefault(s.parent, []).append(s)

    def describe(s: OutlineSection, count: int) -> str:
        text = f"{s.title} (`{s.id}`)"
        if s.summary:
            text += f" — {s.summary.rstrip('.')}"
        return text + f". Блоков: {count}"

    lines = [f"# {outline.title or 'Структура темы'}", ""]
    chapter = sub = 0
    for s in outline.sections:
        if s.level == 1:
            chapter, sub = chapter + 1, 0
            count = len(s.blocks) + sum(len(c.blocks) for c in children.get(s.id, []))
            lines.append(f"- **{chapter}. {describe(s, count)}**")
        else:
            sub += 1
            lines.append(f"  - {chapter}.{sub}. {describe(s, len(s.blocks))}")
    if outline.conflict_hints:
        lines += ["", "## Возможные расхождения источников", ""]
        for hint in outline.conflict_hints:
            note = f" — {hint['note']}" if hint.get("note") else ""
            blocks = hint.get("blocks") or []
            where = f" (блоки: {', '.join(blocks)})" if blocks else ""
            lines.append(f"- {hint.get('topic', '')}{note}{where}")
    if outline.unassigned:
        lines += ["", f"## Не вошли в разделы ({len(outline.unassigned)})", ""]
        for item in outline.unassigned[:MAX_LISTED]:
            lines.append(f"- {item['block']} — {item['reason']}")
        if len(outline.unassigned) > MAX_LISTED:
            lines.append(f"- … и ещё {len(outline.unassigned) - MAX_LISTED}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- checks of the outline


@dataclass(frozen=True)
class Problem:
    kind: str  # structure | missing | unknown | duplicate | both | reason
    text: str  # Russian, shown to the agent
    count: int = 1  # blocks involved (structure problems count as one)


@dataclass
class OutlineCheck:
    problems: list[Problem]
    missing: list[str]  # non-admin blocks that have no place (in the order of `blocks`)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def structural(self) -> list[Problem]:
        return [p for p in self.problems if p.kind == "structure"]

    @property
    def score(self) -> tuple[int, int]:
        """Lower is better: (structure problems, blocks with problems)."""
        return (
            len(self.structural),
            sum(p.count for p in self.problems if p.kind != "structure"),
        )


def missing_blocks(outline: Outline, blocks: Mapping[str, Block]) -> list[str]:
    """Non-admin blocks that are neither in a section nor in `unassigned`."""
    placed = {b for s in outline.sections for b in s.blocks}
    placed |= {u["block"] for u in outline.unassigned}
    return [b.id for b in blocks.values() if not is_admin(b) and b.id not in placed]


def _check_structure(sections: Sequence[OutlineSection]) -> list[Problem]:
    problems: list[Problem] = []

    def bad(text: str) -> None:
        problems.append(Problem("structure", text))

    seen: set[str] = set()
    chapters = {s.id for s in sections if s.level == 1}
    for s in sections:
        if s.id in seen:
            bad(f"Идентификатор раздела `{s.id}` повторяется: id должны быть уникальными.")
        seen.add(s.id)
        if not SECTION_ID_RE.match(s.id):
            bad(f"Идентификатор `{s.id}` не соответствует формату: главы `s01`, разделы `s01-01`.")
        if not s.title:
            bad(f"У раздела `{s.id}` пустое название.")
        if s.level == 1:
            if "-" in s.id:
                bad(f"Глава `{s.id}` (level 1) должна иметь id вида `s01`.")
            if s.parent:
                bad(f"У главы `{s.id}` указан parent `{s.parent}`: у глав parent равен null.")
        elif s.level == 2:
            if not s.parent:
                bad(f"У раздела `{s.id}` (level 2) не указан parent — id главы.")
            elif s.parent not in chapters:
                bad(f"У раздела `{s.id}` parent `{s.parent}` — не существующая глава (level 1).")
            elif not s.id.startswith(s.parent + "-"):
                bad(f"Id раздела `{s.id}` должен начинаться с id главы `{s.parent}-`.")
        else:
            bad(f"У раздела `{s.id}` level {s.level}: допустимы только 1 (глава) и 2 (раздел).")
    parents = {s.parent for s in sections if s.level == 2 and s.parent}
    for s in sections:
        if s.level == 1 and s.id in parents and s.blocks:
            bad(
                f"У главы `{s.id}` есть разделы, но блоки назначены и ей: блоки только у листьев "
                "(у главы с разделами `blocks` пустой)."
            )
    if not any(s.blocks for s in sections):
        bad("В структуре нет ни одного раздела с блоками.")
    return problems


def check_outline(outline: Outline, blocks: Mapping[str, Block]) -> OutlineCheck:
    """Contract checks of docs/ARCHITECTURE.md for S1 (see the module docstring)."""
    problems = _check_structure(outline.sections)

    placed: dict[str, list[str]] = {}
    for s in outline.sections:
        for bid in s.blocks:
            placed.setdefault(bid, []).append(s.id)
    unassigned: dict[str, str] = {}
    for item in outline.unassigned:
        unassigned.setdefault(item["block"], item["reason"])

    unknown = [b for b in dict.fromkeys([*placed, *unassigned]) if b not in blocks]
    if unknown:
        problems.append(
            Problem(
                "unknown",
                f"В структуре указаны несуществующие блоки ({len(unknown)}): {_listed(unknown)}. "
                "Используйте только идентификаторы из `inputs/blocks_index.md`.",
                len(unknown),
            )
        )
    dup = {b: ids for b, ids in placed.items() if len(ids) > 1 and b in blocks}
    if dup:
        shown = [f"{b} ({', '.join(ids)})" for b, ids in list(dup.items())[:MAX_LISTED]]
        more = f" … (и ещё {len(dup) - MAX_LISTED})" if len(dup) > MAX_LISTED else ""
        problems.append(
            Problem(
                "duplicate",
                f"Блоки назначены больше чем одному разделу ({len(dup)}): "
                f"{', '.join(shown)}{more}. Каждый блок должен быть ровно в одном разделе-листе.",
                len(dup),
            )
        )
    both = [b for b in unassigned if b in placed and b in blocks]
    if both:
        problems.append(
            Problem(
                "both",
                f"Блоки одновременно в разделе и в `unassigned` ({len(both)}): {_listed(both)}. "
                "Оставьте каждый блок в одном месте.",
                len(both),
            )
        )
    no_reason = [b for b, reason in unassigned.items() if not reason and b in blocks]
    if no_reason:
        problems.append(
            Problem(
                "reason",
                f"В `unassigned` не указана причина у блоков ({len(no_reason)}): "
                f"{_listed(no_reason)}.",
                len(no_reason),
            )
        )
    missing = missing_blocks(outline, blocks)
    if missing:
        problems.append(
            Problem(
                "missing",
                f"Не назначены блоки ({len(missing)}): {_listed(missing)}. Каждый блок, кроме "
                "служебных (`admin`), должен попасть в `blocks` ровно одного раздела-листа или в "
                "`unassigned` с причиной.",
                len(missing),
            )
        )
    return OutlineCheck(problems, missing)


# ---------------------------------------------------------------- repairs by code


def normalize_outline(outline: Outline) -> list[str]:
    """Order and emptiness fixes that never change the meaning; returns Russian warnings.

    Requires a structurally valid outline (parents exist). Level-2 sections are put right after
    their chapter; sections without blocks that have no subsections are removed.
    """
    warnings: list[str] = []
    sections = outline.sections
    chapters = [s for s in sections if s.level == 1]
    ordered: list[OutlineSection] = []
    for chapter in chapters:
        ordered.append(chapter)
        ordered.extend(s for s in sections if s.level == 2 and s.parent == chapter.id)
    if [s.id for s in ordered] != [s.id for s in sections]:
        warnings.append("Порядок разделов исправлен кодом: подразделы стоят сразу за своей главой.")
    removed: list[str] = []
    while True:
        parents = {s.parent for s in ordered if s.level == 2}
        empty = [s for s in ordered if not s.blocks and s.id not in parents]
        if not empty:
            break
        gone = {s.id for s in empty}
        ordered = [s for s in ordered if s.id not in gone]
        removed += [f"`{s.id}` «{s.title}»" for s in empty]
    if removed:
        warnings.append("Удалены разделы без блоков: " + ", ".join(removed) + ".")
    outline.sections = ordered
    return warnings


def repair_outline(outline: Outline, blocks: Mapping[str, Block]) -> list[str]:
    """Safe repairs of block placement: unknown ids, duplicates, conflicts with `unassigned`."""
    warnings: list[str] = []
    unknown = {b for s in outline.sections for b in s.blocks if b not in blocks}
    unknown |= {u["block"] for u in outline.unassigned if u["block"] not in blocks}
    if unknown:
        for s in outline.sections:
            s.blocks = [b for b in s.blocks if b in blocks]
        outline.unassigned = [u for u in outline.unassigned if u["block"] in blocks]
        warnings.append(
            f"Удалены несуществующие блоки ({len(unknown)}): {_listed(sorted(unknown))}."
        )
    seen: set[str] = set()
    duplicates: list[str] = []
    for s in outline.sections:
        kept: list[str] = []
        for bid in s.blocks:
            if bid in seen:
                duplicates.append(bid)
            else:
                seen.add(bid)
                kept.append(bid)
        s.blocks = kept
    if duplicates:
        warnings.append(
            f"Блоки, назначенные нескольким разделам, оставлены в первом ({len(duplicates)}): "
            f"{_listed(list(dict.fromkeys(duplicates)))}."
        )
    unassigned: list[dict[str, str]] = []
    dropped_both = fixed_reason = 0
    for item in outline.unassigned:
        if item["block"] in seen:
            dropped_both += 1
        elif any(u["block"] == item["block"] for u in unassigned):
            continue
        else:
            if not item["reason"]:
                item = {**item, "reason": "причина не указана"}
                fixed_reason += 1
            unassigned.append(item)
    outline.unassigned = unassigned
    if dropped_both:
        warnings.append(
            f"Блоки из `unassigned`, уже назначенные разделам, убраны ({dropped_both})."
        )
    if fixed_reason:
        warnings.append(f"У блоков в `unassigned` не было причины ({fixed_reason}).")
    return warnings


def autofill_missing(
    outline: Outline, blocks: Mapping[str, Block], missing: Sequence[str]
) -> list[str]:
    """Put every missing block next to its nearest assigned neighbour of the same source."""
    order: dict[str, list[str]] = {}
    for block in blocks.values():
        order.setdefault(block.source, []).append(block.id)
    position = {bid: i for ids in order.values() for i, bid in enumerate(ids)}
    owner = {b: s for s in outline.sections for b in s.blocks}
    fallback = next((s for s in outline.sections if s.blocks), None)
    placed_in: list[tuple[str, str]] = []
    for bid in missing:
        ids, i = order[blocks[bid].source], position[bid]
        prev = next((ids[j] for j in range(i - 1, -1, -1) if ids[j] in owner), None)
        nxt = next((ids[j] for j in range(i + 1, len(ids)) if ids[j] in owner), None)
        if prev is not None and (nxt is None or i - position[prev] <= position[nxt] - i):
            section = owner[prev]
            section.blocks.insert(section.blocks.index(prev) + 1, bid)
        elif nxt is not None:
            section = owner[nxt]
            section.blocks.insert(section.blocks.index(nxt), bid)
        elif fallback is not None:
            section = fallback
            section.blocks.append(bid)
        else:
            continue
        owner[bid] = section
        placed_in.append((bid, section.id))
    if not placed_in:
        return []
    shown = ", ".join(f"{b} → `{s}`" for b, s in placed_in[:MAX_LISTED])
    more = f" … (и ещё {len(placed_in) - MAX_LISTED})" if len(placed_in) > MAX_LISTED else ""
    return [
        f"Агент не назначил блоки ({len(placed_in)}): код добавил их в разделы с ближайшими по "
        f"порядку источника блоками: {shown}{more}."
    ]


def autofill_limit(blocks: Mapping[str, Block]) -> int:
    """How many unassigned blocks count as «few» (code adds them instead of failing)."""
    content = sum(1 for b in blocks.values() if not is_admin(b))
    return max(AUTOFILL_MIN, math.ceil(round(AUTOFILL_RATIO * content, 6)))


# ---------------------------------------------------------------- the stage


def _contract() -> OutputContract:
    return OutputContract(
        files=[
            ExpectedFile(
                path=OUTLINE_JSON,
                kind="json",
                json_schema=OUTLINE_SCHEMA,
                description="Единая структура темы по схеме из задания.",
            ),
            ExpectedFile(
                path=OUTLINE_MD,
                kind="markdown",
                min_chars=20,
                description="Та же структура для человека.",
            ),
        ]
    )


def _task_text(blocks: Mapping[str, Block], source_count: int) -> str:
    admin = sum(1 for b in blocks.values() if is_admin(b))
    return (
        prompt_body("outline").rstrip()
        + "\n\n# Входные файлы\n\n"
        + "- `inputs/summaries.md` — аннотации, оглавления и термины каждого источника "
        + f"(источников: {source_count});\n"
        + "- `inputs/blocks_index.md` — все блоки всех источников, по строке на блок "
        + f"(блоков: {len(blocks)}, из них служебных `admin`: {admin}).\n"
    )


def _feedback_text(problems: Sequence[Problem]) -> str:
    lines = [
        "# Обратная связь по предыдущей попытке",
        "",
        "Структура из предыдущей попытки не прошла автоматическую проверку. Проблемы:",
        "",
        *[f"- {p.text}" for p in problems],
        "",
        "Файлы предыдущей попытки (`outline.json`, `outline.md`) уже лежат в `out/`: исправь их "
        "на месте, не меняя того, что верно. Если структура изменилась, обнови и `outline.md`.",
    ]
    return "\n".join(lines) + "\n"


@dataclass
class _Attempt:
    outline: Outline
    check: OutlineCheck
    md: str
    out_dir: Path
    bundle: Path


def _fail(
    ctx: BuildContext,
    *,
    pid: str,
    model: str,
    t0: float,
    runs: int,
    errors: list[str],
    **details: Any,
) -> StageResult:
    duration = round(time.monotonic() - t0, 3)
    # key=None: a failed run must never look like a cache hit next to old outputs.
    write_meta(
        ctx,
        STAGE,
        {
            "key": None,
            "prompt": pid,
            "model": model,
            "duration_s": duration,
            "agent_runs": runs,
            "ok": False,
            "errors": errors,
        },
    )
    return StageResult(
        stage=STAGE, ok=False, agent_runs=runs, duration_s=duration, errors=errors, details=details
    )


def run_outline(
    ctx: BuildContext,
    blocks: Mapping[str, Block] | Iterable[Block],
    sources: Sequence[SourceRecord],
) -> StageResult:
    """S1: inputs → agent → checks (retries with feedback, repairs) → outline.json/.md."""
    t0 = time.monotonic()
    mapping = _as_mapping(blocks)
    pid, model = prompt_id("outline"), model_for(ctx, "strong", STAGE)
    if not any(not is_admin(b) for b in mapping.values()):
        return _fail(
            ctx,
            pid=pid,
            model=model,
            t0=t0,
            runs=0,
            errors=[
                "Нет блоков для построения структуры: сначала выполните извлечение источников."
            ],
        )
    paths = prepare_inputs(ctx, mapping, sources)
    key = stage_key(paths["summaries"], paths["blocks_index"], pid, model)
    out_json, out_md = ctx.synth_dir / OUTLINE_JSON, ctx.synth_dir / OUTLINE_MD

    if cached(ctx, STAGE, key, [out_json, out_md]):
        meta = read_meta(ctx, STAGE) or {}
        outline = load_outline(ctx)
        return StageResult(
            stage=STAGE,
            ok=True,
            cached=True,
            warnings=list(meta.get("warnings") or []),
            details=_details(outline, attempts=0),
        )

    task_base = _task_text(mapping, len({b.source for b in mapping.values()}))
    contract = _contract()
    inputs = [paths["summaries"], paths["blocks_index"]]
    runs = 0
    best: _Attempt | None = None
    feedback: list[Problem] = []
    seed: dict[str, Path] | None = None
    errors: list[str] = []

    for attempt_no in range(1, MAX_RETRIES + 2):
        task = task_base + ("\n" + _feedback_text(feedback) if feedback else "")
        ctx.emit(
            f"Структура: попытка {attempt_no} из {MAX_RETRIES + 1}"
            + (f" (проблем в прошлой: {len(feedback)})" if feedback else "")
        )
        try:
            result = run_agent(
                ctx,
                stage=STAGE,
                task=task,
                contract=contract,
                inputs=inputs,
                seed=seed,
                tier="strong",
            )
        except (OSError, ValueError) as exc:
            errors.append(f"Не удалось запустить агента: {exc}")
            break
        runs += 1
        if not result.ok:
            errors.append("Агент не вернул допустимую структуру: " + _listed(result.problems, 6))
            break
        out_dir = result.bundle.out_dir
        try:
            data = json.loads((out_dir / OUTLINE_JSON).read_text(encoding="utf-8-sig"))
            outline = parse_outline(data)
        except (OSError, ValueError, AttributeError) as exc:
            errors.append(f"Не удалось прочитать результат агента: {exc}")
            break
        check = check_outline(outline, mapping)
        md_path = out_dir / OUTLINE_MD
        md = md_path.read_text(encoding="utf-8-sig") if md_path.is_file() else ""
        current = _Attempt(outline, check, md, out_dir, result.bundle.root)
        if best is None or check.score <= best.check.score:
            best = current
        if check.ok:
            break
        feedback = check.problems
        seed = {OUTLINE_JSON: out_dir / OUTLINE_JSON}
        if md_path.is_file():
            seed[OUTLINE_MD] = md_path
        ctx.emit("Структура не прошла проверку: " + "; ".join(p.text[:80] for p in feedback[:3]))

    if best is None:
        return _fail(
            ctx, pid=pid, model=model, t0=t0, runs=runs, errors=errors or ["нет результата"]
        )
    return _finish(
        ctx, best, mapping, key=key, pid=pid, model=model, t0=t0, runs=runs, errors=errors
    )


def _details(outline: Outline, *, attempts: int, **extra: Any) -> dict[str, Any]:
    parents = {s.parent for s in outline.sections if s.level == 2}
    leaves = [s for s in outline.sections if s.id not in parents]
    return {
        "sections": len(outline.sections),
        "leaves": len(leaves),
        "blocks_assigned": sum(len(s.blocks) for s in outline.sections),
        "unassigned": len(outline.unassigned),
        "conflict_hints": len(outline.conflict_hints),
        "attempts": attempts,
        **extra,
    }


def _finish(
    ctx: BuildContext,
    best: _Attempt,
    blocks: Mapping[str, Block],
    *,
    key: str,
    pid: str,
    model: str,
    t0: float,
    runs: int,
    errors: list[str],
) -> StageResult:
    """Repair what is safe to repair, save outline.json/.md and the stage meta."""
    outline, check = best.outline, best.check
    warnings: list[str] = []
    changed = False
    if check.structural:
        return _fail(
            ctx,
            pid=pid,
            model=model,
            t0=t0,
            runs=runs,
            errors=[
                "Структура темы не прошла проверку после повторов: "
                + " ".join(p.text for p in check.structural),
                *errors,
            ],
        )
    if errors:  # a retry failed, the best earlier attempt is used
        warnings.append(
            "Повторный прогон агента не удался ("
            + "; ".join(errors)
            + "): использован результат предыдущей попытки."
        )
    norm = normalize_outline(outline)
    warnings += norm
    changed = bool(norm)
    if check.problems:
        warnings += repair_outline(outline, blocks)
        changed = True
        missing = missing_blocks(outline, blocks)
        limit = autofill_limit(blocks)
        if len(missing) > limit:
            return _fail(
                ctx,
                pid=pid,
                model=model,
                t0=t0,
                runs=runs,
                errors=[
                    f"После повторов не назначены блоки ({len(missing)}; допустимо до {limit}): "
                    f"{_listed(missing)}. Запустите стадию ещё раз (--from outline).",
                    *errors,
                ],
            )
        warnings += autofill_missing(outline, blocks, missing)
        # repair/autofill can empty a leaf (duplicates removed): normalise once more.
        warnings += normalize_outline(outline)
    if not any(s.blocks for s in outline.sections):
        return _fail(
            ctx,
            pid=pid,
            model=model,
            t0=t0,
            runs=runs,
            errors=["В структуре нет разделов с блоками."],
        )
    content = sum(1 for b in blocks.values() if not is_admin(b))
    admin_ids = {b.id for b in blocks.values() if is_admin(b)}
    skipped = sum(1 for u in outline.unassigned if u["block"] not in admin_ids)
    if content and skipped / content > UNASSIGNED_WARN_RATIO:
        warnings.append(
            f"В `unassigned` {skipped} из {content} содержательных блоков "
            f"({round(100 * skipped / content)} %): проверьте, не потеряно ли содержание."
        )
    # Later stages (S3–S5) take section ids from outline.md, so it is always rendered by code
    # (with ids); the agent's own outline.md is kept next to it for reference.
    md = render_outline_md(outline)
    if best.md.strip() and not changed:
        _write(ctx.synth_dir / "outline.agent.md", best.md)
    write_json_atomic(ctx.synth_dir / OUTLINE_JSON, outline.to_dict())
    _write(ctx.synth_dir / OUTLINE_MD, md)
    duration = round(time.monotonic() - t0, 3)
    write_meta(
        ctx,
        STAGE,
        {
            "key": key,
            "prompt": pid,
            "model": model,
            "duration_s": duration,
            "agent_runs": runs,
            "ok": True,
            "warnings": warnings,
            "bundle": str(best.bundle),
        },
    )
    return StageResult(
        stage=STAGE,
        ok=True,
        agent_runs=runs,
        duration_s=duration,
        warnings=warnings,
        details=_details(outline, attempts=runs, bundle=str(best.bundle)),
    )
