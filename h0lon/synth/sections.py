"""S2 of the synthesis: the sections of the master written from the blocks of the sources.

Consecutive sections of the outline that have blocks are packed into groups (decision A4 in
docs/DECISIONS-AUTONOMOUS.md: 1–4 sections, about 60 000 characters of blocks per group); the
groups run in parallel, one strong agent per group, with the prompt `sections@<version>.md`. The
code checks every section (a heading with `{#sec:<id>}` of the right level, every block id of the
section in `<!-- src: … -->` comments); sections that miss block ids are rewritten in place (the
previous files are seeded into out/) up to two more times, after that the gaps are only a warning:
S4/S5 fill them.
Chapters and sections without blocks get a heading-only file written by code.

Cache: one record per group in `sections.meta.json` (key = its inputs + prompt + model), so groups
whose inputs did not change are not rewritten. For that the groups of the previous run are kept
while they are valid (a section added in the middle must not shift every later group), and a
group depends only on the terms of the sources it draws on (a new source adds terms to the
glossary, which must not rewrite old sections); records of the older key format are adopted.
Files: `synthesis/inputs/<id>.blocks.md`, `synthesis/sections/<id>.md` and `<id>.notes.json`.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h0lon.agents import ExpectedFile, OutputContract
from h0lon.extract.blocks import atomic_write_text
from h0lon.extract.model import Block
from h0lon.synth.common import (
    cached,
    model_for,
    parse_src_ids,
    prompt_body,
    prompt_id,
    read_meta,
    run_agent,
    sections_dir,
    stage_key,
    text_size,
    topic_relative_md,
    write_json_atomic,
    write_meta,
)
from h0lon.synth.model import BuildContext, Outline, OutlineSection, StageResult
from h0lon.synth.outline import (
    TERMS_PREV_FILE,
    ensure_terms,
    inputs_dir,
    render_outline_md,
    terms_for_sources,
)

STAGE = "sections"
MAX_CHARS = 60_000  # blocks per group (decision A4)
MAX_SECTIONS = 4  # sections per group (decision A4)
MAX_RETRIES = 2  # extra runs of a group for sections with block ids missing from `src`
CONTEXT_CHARS = 1500  # a neighbour block longer than this is shown as plain text, cut
MAX_CONTEXT = 24  # neighbour blocks per section
SHORT_RATIO = 0.3  # warn when the text is shorter than this share of the blocks' text
SHRINK_RATIO = 0.85  # a retry that shrinks the text below this share of the old one is rejected
MAX_LISTED = 12

NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["corrections", "conflicts", "editorial"],
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["anchor", "as_written", "corrected", "reason"],
                "properties": {
                    "anchor": {"type": "string"},
                    "block": {"type": ["string", "null"]},
                    "as_written": {"type": "string"},
                    "corrected": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic", "variants", "reason"],
                "properties": {
                    "topic": {"type": "string"},
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["anchor", "text"],
                            "properties": {
                                "anchor": {"type": "string"},
                                "text": {"type": "string"},
                            },
                        },
                    },
                    "resolution": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "editorial": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "refers_to", "text"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["answer", "reconstruction", "clarification"],
                    },
                    "refers_to": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
    },
}

CONTEXT_BANNER = (
    "<!-- ========== КОНТЕКСТ: соседние блоки тех же источников. Они не входят в этот раздел: "
    "не переноси их в текст и не указывай в src ========== -->"
)


# ---------------------------------------------------------------- helpers


def section_blocks(section: OutlineSection, blocks: Mapping[str, Block]) -> list[str]:
    """Ids of the section's blocks that exist, without repeats, in the order of the outline."""
    return [b for b in dict.fromkeys(section.blocks) if b in blocks]


def _as_mapping(blocks: Mapping[str, Block] | Iterable[Block]) -> Mapping[str, Block]:
    return blocks if isinstance(blocks, Mapping) else {b.id: b for b in blocks}


def _listed(ids: Sequence[str], limit: int = MAX_LISTED) -> str:
    return ", ".join(ids[:limit]) + (f" … (и ещё {len(ids) - limit})" if len(ids) > limit else "")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def section_size(section: OutlineSection, blocks: Mapping[str, Block]) -> int:
    """Characters of Markdown in the section's blocks (what goes into <id>.blocks.md)."""
    return sum(len(blocks[b].md) for b in section_blocks(section, blocks))


def _pack(
    sections: Sequence[OutlineSection],
    blocks: Mapping[str, Block],
    max_chars: int,
    max_sections: int,
) -> list[list[OutlineSection]]:
    """Greedy packing of consecutive sections that have blocks (see `plan_groups`)."""
    groups: list[list[OutlineSection]] = []
    current: list[OutlineSection] = []
    size = 0
    for section in sections:
        n = section_size(section, blocks)
        if current and (len(current) >= max_sections or size + n > max_chars):
            groups.append(current)
            current, size = [], 0
        current.append(section)
        size += n
        if size > max_chars:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    return groups


def _kept_groups(
    leaves: Sequence[OutlineSection],
    blocks: Mapping[str, Block],
    keep: Iterable[Sequence[str]],
    max_chars: int,
    max_sections: int,
) -> dict[int, list[OutlineSection]]:
    """Groups of the previous run that are still valid, by the position of their first section."""
    position = {s.id: i for i, s in enumerate(leaves)}
    kept: dict[int, list[OutlineSection]] = {}
    taken: set[int] = set()
    for ids in keep:
        ids = list(ids)
        if not ids or len(ids) > max_sections or any(sid not in position for sid in ids):
            continue
        first = position[ids[0]]
        places = list(range(first, first + len(ids)))
        if [position[sid] for sid in ids] != places or taken & set(places):
            continue  # a section is gone, moved, or has a new one between its neighbours
        group = list(leaves[first : first + len(ids)])
        if len(group) > 1 and sum(section_size(s, blocks) for s in group) > max_chars:
            continue  # grown over the limit
        kept[first] = group
        taken.update(places)
    return kept


def plan_groups(
    outline: Outline,
    blocks: Mapping[str, Block],
    *,
    max_chars: int = MAX_CHARS,
    max_sections: int = MAX_SECTIONS,
    keep: Iterable[Sequence[str]] = (),
) -> list[list[OutlineSection]]:
    """Consecutive sections with blocks packed into groups (decision A4).

    A group has at most `max_sections` sections and at most `max_chars` characters of blocks;
    a section above the limit makes a group of its own. Sections without blocks are not part of
    any group (the code writes their files).

    `keep`: the section ids of the groups of the previous run. A previous group stays as it was
    while it is valid (all its sections are there, next to each other, in the same order, and the
    group still fits the limits); the other sections are packed as above, run by run between the
    kept groups. Without it a section added in the middle of the topic would shift every group
    after it, and all of them would be written again.
    """
    max_sections = max(1, max_sections)
    leaves = [s for s in outline.sections if section_blocks(s, blocks)]
    kept = _kept_groups(leaves, blocks, keep, max_chars, max_sections)
    groups: list[list[OutlineSection]] = []
    run: list[OutlineSection] = []
    i = 0
    while i < len(leaves):
        group = kept.get(i)
        if group is None:
            run.append(leaves[i])
            i += 1
            continue
        groups += _pack(run, blocks, max_chars, max_sections)
        run = []
        groups.append(group)
        i += len(group)
    groups += _pack(run, blocks, max_chars, max_sections)
    return groups


# ---------------------------------------------------------------- inputs of a section


def _context_blocks(ids: Sequence[str], blocks: Mapping[str, Block]) -> list[tuple[Block, str]]:
    """Neighbours (one before and one after every run of blocks) of the same source."""
    order: dict[str, list[str]] = {}
    index = {bid: i for i, bid in enumerate(blocks)}
    position: dict[str, int] = {}
    for block in blocks.values():
        siblings = order.setdefault(block.source, [])
        position[block.id] = len(siblings)
        siblings.append(block.id)
    inside = set(ids)
    labels: dict[str, list[str]] = {}
    for bid in ids:
        siblings = order[blocks[bid].source]
        i = position[bid]
        if i > 0 and siblings[i - 1] not in inside:
            labels.setdefault(siblings[i - 1], []).append(f"предшествует блоку {bid}")
        if i + 1 < len(siblings) and siblings[i + 1] not in inside:
            labels.setdefault(siblings[i + 1], []).append(f"следует за блоком {bid}")
    chosen = sorted(labels, key=index.__getitem__)[:MAX_CONTEXT]
    return [(blocks[b], "; ".join(labels[b][:2])) for b in chosen]


def prepare_section_inputs(
    ctx: BuildContext, section: OutlineSection, blocks: Mapping[str, Block]
) -> Path:
    """synthesis/inputs/<id>.blocks.md: the blocks of the section in the order of the outline.

    Every block is `<!-- P1.b007 definition [[P1:p3]] -->` + its Markdown. After them come the
    neighbours of the same source as context; they carry no block id on purpose, so that an
    agent cannot count them as covered in this section.
    """
    ids = section_blocks(section, blocks)
    lines = [
        f"<!-- Раздел {section.id} «{section.title}» (уровень {section.level}): блоки источников "
        "в порядке, рекомендованном структурой -->",
        "",
    ]
    for bid in ids:
        block = blocks[bid]
        lines += [
            f"<!-- {block.id} {block.type} [[{block.anchor}]] -->",
            topic_relative_md(block).rstrip(),
            "",
        ]
    context = _context_blocks(ids, blocks)
    if context:
        lines += [CONTEXT_BANNER, ""]
        for block, relation in context:
            md = block.md.rstrip()
            if len(md) > CONTEXT_CHARS:
                md = _one_line(block.text or md)[:CONTEXT_CHARS].rstrip() + "…"
            lines += [
                f"<!-- контекст, не переносить: источник {block.source}, {relation} "
                f"[[{block.anchor}]] -->",
                md,
                "",
            ]
    path = inputs_dir(ctx) / f"{section.id}.blocks.md"
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    return path


# ---------------------------------------------------------------- heading-only sections


def heading_only_text(section: OutlineSection) -> str:
    """File of a chapter without blocks: the heading and, if present, the summary line."""
    text = f"{'#' * section.level} {section.title} {{#sec:{section.id}}}\n"
    summary = section.summary.strip()
    if summary:
        if summary[-1] not in ".!?…:;":
            summary += "."
        text += f"\n{summary}\n"
    return text


# ---------------------------------------------------------------- checks of a written section

_HEADING_RE = re.compile(r"^(#{1,6})(\s+\S.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SEC_LINK_RE = re.compile(r"\]\(#sec:([A-Za-z0-9_-]+)\)")


def _is_comment(line: str) -> bool:
    text = line.strip()
    return text.startswith("<!--") and text.endswith("-->")


def fix_heading(markdown: str, section: OutlineSection) -> tuple[str, str | None]:
    """Make the heading with `{#sec:<id>}` of the right level; returns (text, note or None)."""
    lines = markdown.split("\n")
    first = next((i for i, ln in enumerate(lines) if ln.strip() and not _is_comment(ln)), None)
    anchor = re.compile(r"\{[^}]*#sec:" + re.escape(section.id) + r"(?=[\s}])[^}]*\}\s*$")
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if not (m and anchor.search(line)):
            continue
        notes = []
        if len(m.group(1)) != section.level:
            lines[i] = "#" * section.level + m.group(2)
            notes.append(f"уровень заголовка исправлен кодом ({len(m.group(1))} → {section.level})")
        if i != first:
            notes.append("перед заголовком раздела есть посторонний текст")
        return "\n".join(lines), (f"Раздел `{section.id}`: " + "; ".join(notes) if notes else None)
    heading = f"{'#' * section.level} {section.title} {{#sec:{section.id}}}"
    return (
        heading + "\n\n" + markdown.lstrip("\n"),
        f"Раздел `{section.id}`: заголовок с `{{#sec:{section.id}}}` не найден, добавлен кодом.",
    )


def _missing_src(section: OutlineSection, markdown: str, blocks: Mapping[str, Block]) -> list[str]:
    src = parse_src_ids(markdown)
    return [b for b in section_blocks(section, blocks) if b not in src]


def _quality_warnings(
    section: OutlineSection,
    markdown: str,
    blocks: Mapping[str, Block],
    outline_ids: set[str],
) -> list[str]:
    warnings: list[str] = []
    own = set(section_blocks(section, blocks))
    foreign = sorted(parse_src_ids(markdown) - own)
    if foreign:
        warnings.append(
            f"Раздел `{section.id}`: в `src` указаны блоки, которые разделу не назначены "
            f"({len(foreign)}): {_listed(foreign)}."
        )
    broken = sorted({m for m in _SEC_LINK_RE.findall(markdown) if m not in outline_ids})
    if broken:
        warnings.append(
            f"Раздел `{section.id}`: ссылки на несуществующие разделы: "
            + ", ".join(f"#sec:{b}" for b in broken)
            + "."
        )
    source_size = sum(text_size(blocks[b].md) for b in own)
    if source_size >= 1500 and text_size(markdown) < SHORT_RATIO * source_size:
        warnings.append(
            f"Раздел `{section.id}`: текст заметно короче источников "
            f"({round(100 * text_size(markdown) / source_size)} % от объёма блоков) — "
            "возможна потеря содержания."
        )
    return warnings


# ---------------------------------------------------------------- agent runs


def _contract(sections: Sequence[OutlineSection]) -> OutputContract:
    files: list[ExpectedFile] = []
    for s in sections:
        files.append(
            ExpectedFile(
                path=f"{s.id}.md",
                kind="markdown",
                min_chars=20,
                required_text=[f"{{#sec:{s.id}}}"],
                description=f"Текст раздела `{s.id}` «{s.title}».",
            )
        )
        files.append(
            ExpectedFile(
                path=f"{s.id}.notes.json",
                kind="json",
                json_schema=NOTES_SCHEMA,
                description=f"Правки, расхождения и дополнения раздела `{s.id}`.",
            )
        )
    return OutputContract(files=files)


@dataclass
class _Shared:
    """What every group needs; read-only while the groups run."""

    outline: Outline
    blocks: Mapping[str, Block]
    block_files: dict[str, Path]  # section id -> inputs/<id>.blocks.md
    outline_md: Path  # staged as inputs/outline.md of the bundle
    glossary_md: Path  # staged as inputs/glossary.md of the bundle
    outline_ids: set[str]
    total_groups: int


def _task_text(sections: Sequence[OutlineSection], shared: _Shared, feedback: str = "") -> str:
    titles = {s.id: s.title for s in shared.outline.sections}
    lines = [
        prompt_body("sections").rstrip(),
        "",
        "# Разделы этого задания",
        "",
        "Напиши следующие разделы: для каждого — файлы `out/<id>.md` и `out/<id>.notes.json`.",
        "",
    ]
    for i, s in enumerate(sections, 1):
        kind = "глава" if s.level == 1 else "раздел"
        line = (
            f"{i}. `{s.id}` — {kind} (level {s.level}). Первая строка файла: "
            f"`{'#' * s.level} {s.title} {{#sec:{s.id}}}`."
        )
        if s.parent:
            line += f" Глава: `{s.parent}` «{titles.get(s.parent, '')}»."
        line += f" Блоков: {len(section_blocks(s, shared.blocks))}."
        if s.summary:
            line += f" Содержание по структуре: {s.summary.rstrip('.')}."
        lines.append(line)
    lines += [
        "",
        "Подзаголовки внутри раздела — на уровень глубже его заголовка "
        "(`##` внутри главы, `###` внутри раздела).",
        "",
        "# Входные файлы",
        "",
        "- `inputs/outline.md` — структура всей темы с идентификаторами разделов "
        "(для ссылок `[текст](#sec:<id>)`);",
        "- `inputs/glossary.md` — термины и обозначения из всех источников;",
    ]
    for s in sections:
        lines.append(
            f"- `inputs/{s.id}.blocks.md` — блоки раздела `{s.id}`; соседние блоки в конце файла — "
            "только контекст: не переносить и не указывать в `src`."
        )
    if feedback:
        lines += ["", feedback]
    return "\n".join(lines) + "\n"


def _feedback_text(problems: Mapping[str, list[str]]) -> str:
    lines = [
        "# Обратная связь по предыдущей попытке",
        "",
        "Автоматическая проверка нашла в результате предыдущей попытки проблемы:",
        "",
    ]
    for sid, missing in problems.items():
        lines.append(
            f"- Раздел `{sid}`: в комментариях `<!-- src: … -->` нет блоков ({len(missing)}): "
            f"{_listed(missing, 80)}."
        )
    lines += [
        "",
        "Файлы предыдущей попытки для этих разделов (`<id>.md`, `<id>.notes.json`) уже лежат в "
        "`out/`: допиши их на месте. Добавь в текст недостающее содержание этих блоков "
        "(если блок — полный дубль другого, укажи его id в `<!-- src: … -->` абзаца, который его "
        "покрывает), ничего не удаляя и не сокращая в уже написанном. Файлы остальных разделов "
        "не нужны.",
    ]
    return "\n".join(lines)


@dataclass
class _Draft:
    markdown: str
    notes: Any
    md_path: Path
    notes_path: Path
    missing: list[str]


def _load_draft(
    section: OutlineSection, out_dir: Path, blocks: Mapping[str, Block]
) -> _Draft | None:
    md_path, notes_path = out_dir / f"{section.id}.md", out_dir / f"{section.id}.notes.json"
    try:
        markdown = md_path.read_text(encoding="utf-8-sig")
        notes = json.loads(notes_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return _Draft(markdown, notes, md_path, notes_path, _missing_src(section, markdown, blocks))


@dataclass
class _GroupResult:
    ids: list[str]
    ok: bool
    cached: bool = False
    agent_runs: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing: dict[str, list[str]] = field(default_factory=dict)
    bundles: list[str] = field(default_factory=list)


_meta_lock = threading.Lock()


def _forget_groups(ctx: BuildContext, section_ids: Iterable[str]) -> None:
    """Drop cache records of the groups that own any of these sections and the stage key."""
    ids = set(section_ids)
    with _meta_lock:
        meta = read_meta(ctx, STAGE) or {}
        groups = {
            k: v
            for k, v in (meta.get("groups") or {}).items()
            if not ids & set(v.get("sections") or [])
        }
        write_meta(ctx, STAGE, {**meta, "key": None, "ok": False, "groups": groups})


def _store_group(ctx: BuildContext, key: str, record: dict[str, Any]) -> None:
    with _meta_lock:
        meta = read_meta(ctx, STAGE) or {}
        groups = dict(meta.get("groups") or {})
        groups[key] = record
        write_meta(ctx, STAGE, {**meta, "key": None, "ok": False, "groups": groups})


def _run_group(
    ctx: BuildContext, group: Sequence[OutlineSection], shared: _Shared, key: str, number: int
) -> _GroupResult:
    t0 = time.monotonic()
    ids = [s.id for s in group]
    label = f"группа {number}/{shared.total_groups} ({', '.join(ids)})"
    drafts: dict[str, _Draft] = {}
    warnings: list[str] = []
    bundles: list[str] = []
    pending = list(group)
    feedback = ""
    runs = 0
    for attempt in range(1, MAX_RETRIES + 2):
        ctx.emit(f"Разделы: {label}, попытка {attempt}: " + ", ".join(s.id for s in pending))
        seed: dict[str, Path] = {}
        for s in pending:
            if s.id in drafts:
                seed[f"{s.id}.md"] = drafts[s.id].md_path
                seed[f"{s.id}.notes.json"] = drafts[s.id].notes_path
        inputs = [
            shared.outline_md,
            shared.glossary_md,
            *(shared.block_files[s.id] for s in pending),
        ]
        try:
            result = run_agent(
                ctx,
                stage=STAGE,
                task=_task_text(pending, shared, feedback),
                contract=_contract(pending),
                inputs=inputs,
                seed=seed or None,
                tier="strong",
            )
        except (OSError, ValueError) as exc:
            result = None
            problem = f"не удалось запустить агента: {exc}"
        else:
            runs += 1
            problem = "; ".join(result.problems[:4]) if not result.ok else ""
        if result is None or not result.ok:
            if drafts:
                warnings.append(
                    f"{label}: повторный прогон не удался ({problem}); "
                    "оставлен предыдущий результат."
                )
                break
            return _GroupResult(ids, False, agent_runs=runs, errors=[f"{label}: {problem}"])
        bundles.append(str(result.bundle.root))
        for s in pending:
            new = _load_draft(s, result.bundle.out_dir, shared.blocks)
            old = drafts.get(s.id)
            if new is None:
                if old is None:
                    return _GroupResult(
                        ids, False, agent_runs=runs, errors=[f"{label}: не прочитан файл {s.id}.md"]
                    )
                continue
            if old is not None and (
                len(new.missing) > len(old.missing)
                or text_size(new.markdown) < SHRINK_RATIO * text_size(old.markdown)
            ):
                warnings.append(
                    f"Раздел `{s.id}`: повторный прогон ухудшил результат, "
                    "оставлен прежний вариант."
                )
                continue
            drafts[s.id] = new
        pending = [s for s in group if drafts[s.id].missing]
        if not pending:
            break
        feedback = _feedback_text({s.id: drafts[s.id].missing for s in pending})
        ctx.emit(
            f"Разделы: {label}: в src нет блоков у разделов " + ", ".join(s.id for s in pending)
        )

    # Accepted: write the files (the old cache records of these sections go first).
    _forget_groups(ctx, ids)
    sdir = sections_dir(ctx)
    missing: dict[str, list[str]] = {}
    for s in group:
        draft = drafts[s.id]
        text, note = fix_heading(draft.markdown, s)
        if note:
            warnings.append(note)
        if draft.missing:
            missing[s.id] = draft.missing
            warnings.append(
                f"Раздел `{s.id}`: в комментариях `src` нет блоков ({len(draft.missing)}): "
                f"{_listed(draft.missing)} — их добавят стадии S4/S5."
            )
        warnings += _quality_warnings(s, text, shared.blocks, shared.outline_ids)
        atomic_write_text(sdir / f"{s.id}.md", text if text.endswith("\n") else text + "\n")
        write_json_atomic(sdir / f"{s.id}.notes.json", draft.notes)
    _store_group(
        ctx,
        key,
        {
            "sections": ids,
            "ok": True,
            "warnings": warnings,
            "missing": missing,
            "agent_runs": runs,
            "duration_s": round(time.monotonic() - t0, 3),
            "bundles": bundles,
        },
    )
    return _GroupResult(
        ids, True, agent_runs=runs, warnings=warnings, missing=missing, bundles=bundles
    )


# ---------------------------------------------------------------- the stage

_SECTION_FILE_RE = re.compile(r"^(s\d{2}(?:-\d{2})?)(\.notes\.json|\.md)$")
_BLOCKS_FILE_RE = re.compile(r"^(s\d{2}(?:-\d{2})?)\.blocks\.md$")


def _group_key(
    ctx: BuildContext,
    group: Sequence[OutlineSection],
    outline: Outline,
    paths: Mapping[str, Path],
    glossary: Path,
    blocks: Mapping[str, Block],
) -> str:
    """Cache key of a group: its prompt, model, section specs, block files and the part of the
    terms file for the sources it draws on. The terms of other sources are left out on purpose:
    a new source adds its terms to every run of S1, and that must not rewrite the old sections."""
    sources = {blocks[b].source for s in group for b in section_blocks(s, blocks)}
    text = glossary.read_text(encoding="utf-8") if glossary.is_file() else ""
    return _key_of(ctx, group, outline, paths, terms_for_sources(text, sources))


def _previous_groups(ctx: BuildContext) -> list[list[str]]:
    """Section ids of the groups that sections.meta.json remembers (finished ones only)."""
    groups = (read_meta(ctx, STAGE) or {}).get("groups") or {}
    return [
        [str(s) for s in rec.get("sections") or []]
        for rec in groups.values()
        if isinstance(rec, dict) and rec.get("ok")
    ]


def _legacy_group_key(
    ctx: BuildContext,
    group: Sequence[OutlineSection],
    outline: Outline,
    paths: Mapping[str, Path],
    glossary: Path,
) -> str:
    """The key of versions before the incremental update (the whole terms file in it). A group
    record stored under it is still valid and is adopted, so an upgrade rewrites nothing."""
    return _key_of(ctx, group, outline, paths, glossary)


def _adopt_legacy(
    ctx: BuildContext,
    group: Sequence[OutlineSection],
    outline: Outline,
    paths: Mapping[str, Path],
    glossary: Path,
    prev_terms: Path,
    blocks: Mapping[str, Block],
) -> dict | None:
    """The record of a group stored under a key of the older format, or None.

    Such a record is valid when the group's inputs are unchanged and the glossary it was written
    with is the one now, or the previous one (`terms.prev.md`, kept by S1) with the parts of the
    group's own sources equal to the current ones: the glossary may have grown since by the terms
    of a new source.
    """
    candidates = [glossary]
    if prev_terms.is_file() and glossary.is_file():
        sources = {blocks[b].source for s in group for b in section_blocks(s, blocks)}
        before = terms_for_sources(prev_terms.read_text(encoding="utf-8"), sources)
        if before == terms_for_sources(glossary.read_text(encoding="utf-8"), sources):
            candidates.append(prev_terms)
    for candidate in candidates:
        record = _group_cached(ctx, group, _legacy_group_key(ctx, group, outline, paths, candidate))
        if record is not None:
            return record
    return None


def _key_of(
    ctx: BuildContext,
    group: Sequence[OutlineSection],
    outline: Outline,
    paths: Mapping[str, Path],
    glossary: str | Path,
) -> str:
    titles = {s.id: s.title for s in outline.sections}
    specs = [
        {
            "id": s.id,
            "title": s.title,
            "level": s.level,
            "parent": s.parent,
            "parent_title": titles.get(s.parent or ""),
            "summary": s.summary,
        }
        for s in group
    ]
    return stage_key(
        prompt_id("sections"),
        model_for(ctx, "strong", STAGE),
        specs,
        glossary,
        *(paths[s.id] for s in group),
    )


def _group_cached(ctx: BuildContext, group: Sequence[OutlineSection], key: str) -> dict | None:
    if ctx.force:
        return None
    record = ((read_meta(ctx, STAGE) or {}).get("groups") or {}).get(key)
    if (
        not record
        or not record.get("ok")
        or set(record.get("sections") or []) != {s.id for s in group}
    ):
        return None
    sdir = sections_dir(ctx)
    files = [sdir / f"{s.id}{ext}" for s in group for ext in (".md", ".notes.json")]
    return record if all(p.is_file() for p in files) else None


def _remove_stale(ctx: BuildContext, outline: Outline, writable: set[str]) -> None:
    keep = {f"{s.id}.md" for s in outline.sections} | {f"{i}.notes.json" for i in writable}
    sdir = sections_dir(ctx)
    if sdir.is_dir():
        for path in sdir.iterdir():
            if _SECTION_FILE_RE.match(path.name) and path.name not in keep:
                path.unlink(missing_ok=True)
    root = inputs_dir(ctx)
    if root.is_dir():
        for path in root.iterdir():
            m = _BLOCKS_FILE_RE.match(path.name)
            if m and m.group(1) not in writable:
                path.unlink(missing_ok=True)


def run_sections(
    ctx: BuildContext, outline: Outline, blocks: Mapping[str, Block] | Iterable[Block]
) -> StageResult:
    """S2: write `synthesis/sections/<id>.md` and `<id>.notes.json` for every section."""
    t0 = time.monotonic()
    mapping = _as_mapping(blocks)
    pid, model = prompt_id("sections"), model_for(ctx, "strong", STAGE)
    sdir = sections_dir(ctx)
    sdir.mkdir(parents=True, exist_ok=True)
    glossary = ensure_terms(ctx, mapping)

    # The groups of the previous run stay as they were while they are valid (see plan_groups);
    # --force plans from scratch.
    previous = [] if ctx.force else _previous_groups(ctx)
    groups = plan_groups(
        outline, mapping, max_chars=MAX_CHARS, max_sections=MAX_SECTIONS, keep=previous
    )
    writable = {s.id for g in groups for s in g}
    heading_only = [s for s in outline.sections if s.id not in writable]
    paths = {s.id: prepare_section_inputs(ctx, s, mapping) for g in groups for s in g}
    keys = [_group_key(ctx, g, outline, paths, glossary, mapping) for g in groups]
    stage_k = stage_key(pid, model, keys, [heading_only_text(s) for s in heading_only])
    outputs = [sdir / f"{s.id}.md" for s in outline.sections]
    outputs += [sdir / f"{i}.notes.json" for i in writable]

    warnings: list[str] = []
    unknown = sorted({b for s in outline.sections for b in s.blocks if b not in mapping})
    if unknown:
        warnings.append(
            f"В структуре есть неизвестные блоки ({len(unknown)}): {_listed(unknown)} — пропущены."
        )
    if cached(ctx, STAGE, stage_k, outputs):
        record = read_meta(ctx, STAGE) or {}
        return StageResult(
            stage=STAGE,
            ok=True,
            cached=True,
            warnings=list(record.get("warnings") or warnings),
            details={"groups": len(groups), "groups_cached": len(groups), "sections": len(outputs)},
        )

    for s in heading_only:
        atomic_write_text(sdir / f"{s.id}.md", heading_only_text(s))
    total = len(groups)
    cached_records: dict[int, dict] = {}
    pending: list[int] = []
    prev_terms = inputs_dir(ctx) / TERMS_PREV_FILE
    for i, (group, key) in enumerate(zip(groups, keys, strict=True)):
        record = _group_cached(ctx, group, key)
        if record is None:
            record = _adopt_legacy(ctx, group, outline, paths, glossary, prev_terms, mapping)
            if record is not None:
                _store_group(ctx, key, record)  # the same inputs, an older key format
        if record is not None:
            cached_records[i] = record
        else:
            pending.append(i)
    ctx.emit(
        f"Разделы: групп {total}, из кэша {len(cached_records)}, к написанию {len(pending)} "
        f"(параллельно до {max(1, ctx.settings.agents.parallel_runs)})"
    )

    results: dict[int, _GroupResult] = {}
    with tempfile.TemporaryDirectory(prefix="h0lon-s2-") as tmp:
        # The prompt names its inputs outline.md and glossary.md: stage files under those names.
        outline_md = Path(tmp) / "outline.md"
        glossary_md = Path(tmp) / "glossary.md"
        atomic_write_text(outline_md, render_outline_md(outline))
        atomic_write_text(glossary_md, glossary.read_text(encoding="utf-8"))
        shared = _Shared(
            outline=outline,
            blocks=mapping,
            block_files=paths,
            outline_md=outline_md,
            glossary_md=glossary_md,
            outline_ids={s.id for s in outline.sections},
            total_groups=total,
        )
        if pending:
            workers = max(1, min(ctx.settings.agents.parallel_runs, len(pending)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h0lon-s2") as pool:
                futures = {
                    i: pool.submit(_run_group, ctx, groups[i], shared, keys[i], i + 1)
                    for i in pending
                }
                for i, future in futures.items():
                    try:
                        results[i] = future.result()
                    except Exception as exc:  # a bug in one group must not lose the others
                        ids = [s.id for s in groups[i]]
                        results[i] = _GroupResult(
                            ids, False, errors=[f"Группа {i + 1} ({', '.join(ids)}): {exc!r}"]
                        )

    errors: list[str] = []
    missing: dict[str, list[str]] = {}
    bundles: list[str] = []
    runs = 0
    for i in range(total):
        if i in cached_records:
            record = cached_records[i]
            warnings += record.get("warnings") or []
            missing.update(record.get("missing") or {})
            continue
        res = results[i]
        runs += res.agent_runs
        warnings += res.warnings
        errors += res.errors
        missing.update(res.missing)
        bundles += res.bundles
    ok = not errors
    _remove_stale(ctx, outline, writable)

    # Keep records of the current groups only; the stage key is set only for a complete result.
    with _meta_lock:
        meta = read_meta(ctx, STAGE) or {}
        current = {k: v for k, v in (meta.get("groups") or {}).items() if k in set(keys)}
        duration = round(time.monotonic() - t0, 3)
        write_meta(
            ctx,
            STAGE,
            {
                "key": stage_k if ok else None,
                "prompt": pid,
                "model": model,
                "duration_s": duration,
                "agent_runs": runs,
                "ok": ok,
                "warnings": warnings,
                "errors": errors,
                "groups": current,
            },
        )
    return StageResult(
        stage=STAGE,
        ok=ok,
        cached=not pending,
        agent_runs=runs,
        duration_s=duration,
        warnings=warnings,
        errors=errors,
        details={
            "groups": total,
            "groups_cached": len(cached_records),
            "groups_run": len(pending),
            "sections": len(outline.sections),
            "sections_written_by_agent": len(writable),
            "sections_written_by_code": len(heading_only),
            "missing": missing,
            "bundles": bundles,
        },
    )
