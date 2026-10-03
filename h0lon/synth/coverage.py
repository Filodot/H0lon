"""S4 (coverage) and S5 (supplement) — nothing from the sources may be lost (PRD 8).

`compute_coverage` is deterministic: a block counts as covered when its id occurs in a
`<!-- src: … -->` comment of any `final/sections/*.md` (or of `final/intro.md`,
`final/glossary.md`); blocks of type `admin` are not counted at all.

`run_coverage` adds the two agent stages on top of it. Without uncovered blocks no agent runs.

S4 (tier light): for the uncovered blocks the agent decides `duplicate` (already in the master),
`admin` (organisational) or `missing` (to be added; the verdict names the target section). The
input is `inputs/uncovered.md` (full Markdown of every block with id and anchor),
`inputs/master_index.md` (headings and the first 120 characters of every paragraph with its src)
and `outline.md` (the file of S1). A block without a verdict counts as `missing`, and so does
every block of a run whose result never passed the contract. Very long lists are split into
several runs (about 80 000 characters of blocks each).

S5 (tier strong): `missing` blocks are grouped by their target section (the verdict, else the
section the outline assigned the block to, else the section of the nearest assigned neighbour of
the same source). Sections are packed into runs of about 40 000 characters (the section texts plus
the missing blocks; a bigger section goes alone). Every run gets copies of its sections as a seed
(edited in place), `inputs/missing.md` and `outline.md`. A result is taken over into
`final/sections/<id>.md` only when, for that section, the old src ids are all still there, the
headings `{#sec:…}` are intact, the text did not shrink below 90 % and at least one of the wanted
block ids appeared in src; otherwise the section stays as it was (a rollback) and a warning
is recorded. Wanted ids that did not appear stay `missing` for the next round.

At most two rounds S4 → S5 (docs/ARCHITECTURE.md). Blocks judged `duplicate` / `admin` are not
asked about again. In the report such blocks count as covered, with their verdict. Whatever is still
uncovered after the last round stays `missing` in the report (a map of coverage in the master).

Files under `<topic>/synthesis/`: `inputs/{uncovered,master_index,missing}.md` (rewritten before
every run; `outline.md` is S1's file, a rendered copy goes to `inputs/coverage/` only when it is
absent), `coverage.json` (`CoverageReport.to_dict()` plus `verdicts`), `coverage.meta.json`
(cache: key = final/ texts, blocks, outline, prompts, models).

The stage fails (`ok=False`) only when an agent could not run at all (login, limits, crash): the
report is still written, and a later run continues from the current `final/`. A result that did not
pass the checks is a warning, not a failure.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h0lon.agents import ExpectedFile, OutputContract
from h0lon.agents.runner import RunResult
from h0lon.extract.model import Block
from h0lon.synth.common import (
    cached,
    model_for,
    parse_src_ids,
    prompt_body,
    prompt_id,
    read_meta,
    run_agent,
    src_ids_in_files,
    stage_key,
    text_size,
    topic_relative_md,
    write_json_atomic,
    write_meta,
)
from h0lon.synth.globalpass import (
    FINAL_DIR,
    MIN_SIZE_RATIO,
    anchor_problems,
    fmt_ids,
    heading_anchors,
    outline_text,
    read_text,
    render_outline,
    write_text,
)
from h0lon.synth.model import BuildContext, CoverageReport, Outline, StageResult

STAGE = "coverage"
MAX_ROUNDS = 2  # S4 → S5 rounds (docs/ARCHITECTURE.md)
S4_BATCH_CHARS = 80_000  # block text per S4 run
S5_RUN_CHARS = 40_000  # section text + missing blocks per S5 run
SNIPPET_CHARS = 120  # paragraph start in master_index.md

VERDICTS = ("duplicate", "admin", "missing")
COVERED_VERDICTS = ("duplicate", "admin")  # count as covered in the report

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["block", "verdict"],
                "properties": {
                    "block": {"type": "string"},
                    "verdict": {"enum": list(VERDICTS)},
                    "section": {"type": ["string", "null"]},
                    "note": {"type": "string"},
                },
            },
        }
    },
}


# ---------------------------------------------------------------- deterministic coverage


def _block_list(blocks: Mapping[str, Block] | Iterable[Block]) -> list[Block]:
    return list(blocks.values()) if isinstance(blocks, Mapping) else list(blocks)


def master_src_ids(sections_dir: Path) -> set[str]:
    """Block ids mentioned in src comments of sections_dir/*.md and the sibling intro/glossary."""
    files = sorted(sections_dir.glob("*.md")) if sections_dir.is_dir() else []
    files += [sections_dir.parent / "intro.md", sections_dir.parent / "glossary.md"]
    return src_ids_in_files(files)


def compute_coverage(
    blocks: Mapping[str, Block] | Iterable[Block], sections_dir: Path
) -> CoverageReport:
    """Coverage by src comments (no agents). Every uncovered block has verdict `unknown`."""
    present = master_src_ids(sections_dir)
    total = covered = 0
    by_source: dict[str, dict[str, int]] = {}
    uncovered: list[dict[str, str]] = []
    for block in _block_list(blocks):
        stat = by_source.setdefault(block.source, {"total": 0, "covered": 0})
        if block.type == "admin":
            continue
        total += 1
        stat["total"] += 1
        if block.id in present:
            covered += 1
            stat["covered"] += 1
        else:
            uncovered.append({"block": block.id, "verdict": "unknown", "note": ""})
    return CoverageReport(total=total, covered=covered, by_source=by_source, uncovered=uncovered)


def _with_verdicts(
    report: CoverageReport,
    blocks: Mapping[str, Block],
    verdicts: Mapping[str, Mapping[str, Any]],
    rounds: int,
) -> CoverageReport:
    """Attach the verdicts to the uncovered blocks; duplicate/admin ones count as covered."""
    covered = report.covered
    by_source = {s: dict(stat) for s, stat in report.by_source.items()}
    uncovered: list[dict[str, str]] = []
    for item in report.uncovered:
        verdict = verdicts.get(item["block"])
        kind = str(verdict["verdict"]) if verdict else "unknown"
        uncovered.append(
            {
                "block": item["block"],
                "verdict": kind,
                "note": str(verdict.get("note", "")) if verdict else "",
            }
        )
        if kind in COVERED_VERDICTS:
            covered += 1
            by_source[blocks[item["block"]].source]["covered"] += 1
    return CoverageReport(
        total=report.total,
        covered=covered,
        by_source=by_source,
        uncovered=uncovered,
        rounds=rounds,
    )


# ---------------------------------------------------------------- agent inputs


def render_block(block: Block) -> str:
    """A block for the agent: delimiting comments carry id, type and place in the source."""
    head = f"<!-- {block.id} {block.type} [[{block.anchor}]] -->"
    return f"{head}\n{topic_relative_md(block).strip()}\n<!-- /{block.id} -->\n"


_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_ANCHOR_RE = re.compile(r"\[\[[^\]\n]+\]\]")
_DIV_RE = re.compile(r"^:{3,}\s*\{?\s*\.?([A-Za-z][\w-]*)")
_BLANK_RE = re.compile(r"\n\s*\n")


def master_index(final_dir: Path, ids: Sequence[str]) -> str:
    """Headings of final/ and the first 120 characters of every paragraph with its src."""
    files: list[tuple[str, Path]] = [("intro.md", final_dir / "intro.md")]
    files += [(f"sections/{sid}.md", final_dir / "sections" / f"{sid}.md") for sid in ids]
    files.append(("glossary.md", final_dir / "glossary.md"))
    out = ["# Индекс мастер-конспекта", ""]
    out.append(
        "Для каждого файла: заголовки и начало каждого абзаца (до "
        f"{SNIPPET_CHARS} символов) со списком блоков источников из комментария `src`."
    )
    for name, path in files:
        if not path.is_file():
            continue
        out += ["", f"## Файл {name}", ""]
        items: list[dict[str, Any]] = []
        for chunk in _BLANK_RE.split(read_text(path)):
            chunk = chunk.strip("\n")
            if not chunk.strip():
                continue
            src = sorted(parse_src_ids(chunk))
            body = _COMMENT_RE.sub("", chunk)
            lines = [ln for ln in body.splitlines() if ln.strip()]
            if not lines:  # a standalone src comment belongs to the previous paragraph
                if items:
                    items[-1]["src"] = sorted(set(items[-1]["src"]) | set(src))
                continue
            first = lines[0].lstrip()
            if first.startswith("#"):
                items.append({"heading": first.strip(), "src": []})
                lines = lines[1:]
                if not lines:
                    continue
            label = ""
            text_lines = []
            for ln in lines:
                m = _DIV_RE.match(ln.strip())
                if m:
                    label = label or f"[{m.group(1)}] "
                    continue
                if ln.strip().startswith(":::"):
                    continue
                text_lines.append(ln.strip())
            text = _ANCHOR_RE.sub("", " ".join(text_lines))
            text = re.sub(r"\s+", " ", text).strip()
            if not text and not label:
                continue
            if len(text) > SNIPPET_CHARS:
                text = text[: SNIPPET_CHARS - 1].rstrip() + "…"
            items.append({"text": label + text, "src": src})
        for item in items:
            if "heading" in item:
                out.append(f"- **Заголовок:** `{item['heading']}`")
            else:
                refs = " ".join(item["src"]) if item["src"] else "(без src)"
                out.append(f"- {item['text']} ← {refs}")
    return "\n".join(out) + "\n"


def _batches(blocks: Sequence[Block], limit: int) -> list[list[Block]]:
    batches: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        n = len(block.md) + 80
        if current and size + n > limit:
            batches.append(current)
            current, size = [], 0
        current.append(block)
        size += n
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------- S4


def _is_validation_failure(result: RunResult) -> bool:
    return bool(result.attempts) and result.attempts[-1].error_kind == "validation"


def _parse_verdicts(path: Path, wanted: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Verdicts of out/coverage.json for the wanted block ids (the last entry per block wins)."""
    data = json.loads(read_text(path))
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("verdicts") or []:
        if not isinstance(item, dict) or item.get("block") not in wanted:
            continue
        if item.get("verdict") not in VERDICTS:
            continue
        out[item["block"]] = {
            "block": item["block"],
            "verdict": item["verdict"],
            "section": item.get("section") or None,
            "note": str(item.get("note") or ""),
        }
    return out


@dataclass
class _S4Outcome:
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _run_s4(
    ctx: BuildContext,
    pending: Sequence[Block],
    *,
    outline_file: Path,
    index_md: str,
    round_no: int,
) -> _S4Outcome:
    outcome = _S4Outcome()
    inputs_dir = ctx.synth_dir / "inputs"
    write_text(inputs_dir / "master_index.md", index_md)
    batches = _batches(pending, S4_BATCH_CHARS)
    for number, batch in enumerate(batches, start=1):
        header = (
            "# Непокрытые блоки источников\n\n"
            "Блок начинается комментарием `<!-- id тип [[якорь]] -->` и заканчивается "
            "комментарием `<!-- /id -->`; между ними — его Markdown.\n\n"
        )
        write_text(inputs_dir / "uncovered.md", header + "\n".join(render_block(b) for b in batch))
        ids = [b.id for b in batch]
        task = prompt_body("coverage").rstrip() + "\n"
        if len(batches) > 1:
            task += (
                f"\nЭто часть {number} из {len(batches)}: "
                "вердикты нужны только для блоков из неё.\n"
            )
        suffix = f", часть {number}/{len(batches)}" if len(batches) > 1 else ""
        ctx.emit(f"Покрытие: разбор {len(batch)} непокрытых блоков (круг {round_no}{suffix})")
        result = run_agent(
            ctx,
            stage=STAGE,
            task=task,
            contract=OutputContract(
                files=[
                    ExpectedFile(
                        path="coverage.json",
                        kind="json",
                        json_schema=VERDICT_SCHEMA,
                        description="Вердикт на каждый блок из inputs/uncovered.md.",
                    )
                ]
            ),
            inputs=[
                inputs_dir / "uncovered.md",
                inputs_dir / "master_index.md",
                outline_file,
            ],
            tier="light",
        )
        outcome.runs += 1
        if not result.ok:
            reason = "; ".join(str(p) for p in result.problems[:3]) or "агент не ответил"
            if not _is_validation_failure(result):
                outcome.error = reason
                return outcome
            # The agent ran but gave nothing usable: doubt means `missing` (cheaper than a loss).
            for bid in ids:
                outcome.verdicts[bid] = {
                    "block": bid,
                    "verdict": "missing",
                    "section": None,
                    "note": "вердикт не получен: результат агента не прошёл проверку",
                }
            outcome.warnings.append(
                f"Агент не вернул годных вердиктов для {len(ids)} блоков (считаются пропущенными): "
                + reason
            )
            continue
        got = _parse_verdicts(result.bundle.out_dir / "coverage.json", ids)
        silent = [i for i in ids if i not in got]
        for sid in silent:
            got[sid] = {
                "block": sid,
                "verdict": "missing",
                "section": None,
                "note": "вердикт не получен; блок считается пропущенным",
            }
        if silent:
            outcome.warnings.append(
                f"Агент не вынес вердикт для {len(silent)} блоков (считаются пропущенными): "
                + fmt_ids(silent, limit=10)
            )
        outcome.verdicts.update(got)
    return outcome


# ---------------------------------------------------------------- S5


def _assignment(outline: Outline) -> dict[str, str]:
    return {b: s.id for s in outline.sections for b in s.blocks}


def _resolve_target(
    block: Block,
    verdict_section: str | None,
    *,
    outline: Outline,
    assigned: Mapping[str, str],
    available: Mapping[str, Path],
    source_order: Mapping[str, list[str]],
) -> str | None:
    """Section file that should receive the missing block (None: no section files at all)."""
    if verdict_section in available:
        return verdict_section
    if verdict_section:  # a chapter: its first section that has text
        for s in outline.sections:
            if s.parent == verdict_section and s.id in available:
                return s.id
    own = assigned.get(block.id)
    if own in available:
        return own
    order = source_order.get(block.source, [])
    if block.id in order:
        pos = order.index(block.id)
        for dist in range(1, len(order)):
            for idx in (pos - dist, pos + dist):
                if 0 <= idx < len(order):
                    neighbour = assigned.get(order[idx])
                    if neighbour in available:
                        return neighbour
    return next(reversed(available), None) if available else None


def _pack(
    targets: Mapping[str, list[str]], weight: Mapping[str, int], order: Sequence[str]
) -> list[list[str]]:
    """Section ids (in outline order) packed into runs of about S5_RUN_CHARS."""
    runs: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sid in order:
        if sid not in targets:
            continue
        w = weight[sid]
        if current and size + w > S5_RUN_CHARS:
            runs.append(current)
            current, size = [], 0
        current.append(sid)
        size += w
    if current:
        runs.append(current)
    return runs


def _missing_md(
    group: Sequence[str],
    targets: Mapping[str, list[str]],
    blocks: Mapping[str, Block],
    verdicts: Mapping[str, Mapping[str, Any]],
    titles: Mapping[str, str],
) -> str:
    lines = [
        "# Пропущенные блоки",
        "",
        "Для каждого раздела — блоки источников, которых в нём нет; файл раздела лежит в "
        "`out/sections/<id>.md`. Блок начинается комментарием `<!-- id тип [[якорь]] -->` и "
        "заканчивается комментарием `<!-- /id -->`.",
    ]
    for sid in group:
        lines += ["", f"## Раздел `{sid}` — {titles.get(sid, sid)} (`out/sections/{sid}.md`)"]
        for bid in targets[sid]:
            block = blocks[bid]
            note = str(verdicts.get(bid, {}).get("note") or "").strip()
            lines += ["", f"### Блок {bid}"]
            if note:
                lines += ["", f"Замечание проверки полноты: {note}"]
            lines += ["", render_block(block).rstrip()]
    return "\n".join(lines) + "\n"


@dataclass
class _S5Outcome:
    runs: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)  # wanted ids that now occur in src


def _apply_section(
    out_path: Path, final_path: Path, sid: str, wanted: Sequence[str]
) -> tuple[bool, list[str], list[str]]:
    """Check one edited section; (accepted, problems, wanted ids that appeared in src)."""
    if not out_path.is_file():
        return False, [f"нет файла out/sections/{sid}.md"], []
    old, new = read_text(final_path), read_text(out_path)
    problems = anchor_problems(heading_anchors(old), heading_anchors(new), f"разделе {sid}")
    lost = parse_src_ids(old) - parse_src_ids(new)
    if lost:
        problems.append("потеряны ссылки на блоки: " + fmt_ids(lost, limit=10))
    before, after = text_size(old), text_size(new)
    if before and after < MIN_SIZE_RATIO * before:
        problems.append(f"объём текста упал с {before} до {after} символов")
    added = sorted(set(wanted) & parse_src_ids(new))
    if not added:
        problems.append("ни один из пропущенных блоков не появился в комментариях src")
    return not problems, problems, added


def _run_s5(
    ctx: BuildContext,
    outline: Outline,
    blocks: Mapping[str, Block],
    missing: Sequence[str],
    verdicts: Mapping[str, Mapping[str, Any]],
    *,
    outline_file: Path,
    round_no: int,
) -> _S5Outcome:
    outcome = _S5Outcome()
    final_sections = ctx.synth_dir / FINAL_DIR / "sections"
    order = [s.id for s in outline.sections]
    titles = {s.id: s.title for s in outline.sections}
    available = {
        sid: final_sections / f"{sid}.md"
        for sid in order
        if (final_sections / f"{sid}.md").is_file()
    }
    if not available:
        outcome.warnings.append("В final/sections/ нет разделов — дополнять нечего.")
        return outcome
    assigned = _assignment(outline)
    source_order: dict[str, list[str]] = {}
    for b in blocks.values():
        source_order.setdefault(b.source, []).append(b.id)

    targets: dict[str, list[str]] = {}
    for bid in missing:
        sid = _resolve_target(
            blocks[bid],
            verdicts.get(bid, {}).get("section"),
            outline=outline,
            assigned=assigned,
            available=available,
            source_order=source_order,
        )
        if sid is not None:
            targets.setdefault(sid, []).append(bid)
    weight = {
        sid: len(read_text(available[sid])) + sum(len(blocks[b].md) + 80 for b in ids)
        for sid, ids in targets.items()
    }
    groups = _pack(targets, weight, order)
    inputs_dir = ctx.synth_dir / "inputs"
    for number, group in enumerate(groups, start=1):
        write_text(inputs_dir / "missing.md", _missing_md(group, targets, blocks, verdicts, titles))
        listing = "\n".join(
            f"- `out/sections/{sid}.md` — {titles.get(sid, sid)}; добавить блоки: "
            + ", ".join(targets[sid])
            for sid in group
        )
        task = (
            prompt_body("supplement").rstrip() + "\n\n## Разделы этого прогона\n\n" + listing + "\n"
        )
        n_blocks = sum(len(targets[s]) for s in group)
        ctx.emit(
            f"Дополнение: разделов {len(group)}, блоков {n_blocks} "
            f"(круг {round_no}, прогон {number}/{len(groups)})"
        )
        result = run_agent(
            ctx,
            stage="supplement",
            task=task,
            contract=OutputContract(
                files=[
                    ExpectedFile(
                        path=f"sections/{sid}.md",
                        kind="markdown",
                        required_text=[f"{{#sec:{sid}"],
                        description=f"Раздел «{titles.get(sid, sid)}» — правь на месте.",
                    )
                    for sid in group
                ]
            ),
            inputs=[inputs_dir / "missing.md", outline_file],
            seed={f"sections/{sid}.md": available[sid] for sid in group},
            tier="strong",
        )
        outcome.runs += 1
        if not result.ok:
            reason = "; ".join(str(p) for p in result.problems[:3]) or "агент не ответил"
            if _is_validation_failure(result):
                outcome.warnings.append(
                    f"Дополнение разделов {', '.join(group)} не прошло проверку агента: {reason}"
                )
                continue
            outcome.error = reason
            return outcome
        for sid in group:
            ok, problems, added = _apply_section(
                result.bundle.out_dir / "sections" / f"{sid}.md", available[sid], sid, targets[sid]
            )
            if not ok:
                outcome.warnings.append(
                    f"Дополнение раздела {sid} отклонено, раздел оставлен как был: "
                    + "; ".join(problems)
                )
                continue
            new_text = (result.bundle.out_dir / "sections" / f"{sid}.md").read_bytes()
            available[sid].write_bytes(new_text)
            outcome.applied += added
            left = [b for b in targets[sid] if b not in added]
            if left:
                outcome.warnings.append(
                    f"В раздел {sid} не добавлены блоки: " + fmt_ids(left, limit=10)
                )
    return outcome


# ---------------------------------------------------------------- stage


def _outline_file(ctx: BuildContext, outline: Outline) -> Path:
    """synthesis/outline.md (S1); when it is absent a rendered copy named outline.md."""
    path = ctx.synth_dir / "outline.md"
    if path.is_file():
        return path
    fallback = ctx.synth_dir / "inputs" / "coverage" / "outline.md"
    write_text(fallback, render_outline(outline))
    return fallback


def _key(ctx: BuildContext, blocks: Mapping[str, Block], outline_md: str, final_dir: Path) -> str:
    files = sorted(final_dir.rglob("*.md")) if final_dir.is_dir() else []
    return stage_key(
        prompt_id("coverage"),
        prompt_id("supplement"),
        model_for(ctx, "light", "coverage"),
        model_for(ctx, "strong", "supplement"),
        outline_md,
        [(b.id, b.type, b.md) for b in blocks.values()],
        [(p.relative_to(final_dir).as_posix(), read_text(p)) for p in files],
    )


def _report_from_json(path: Path) -> CoverageReport | None:
    try:
        data = json.loads(read_text(path))
        return CoverageReport(
            total=int(data["total"]),
            covered=int(data["covered"]),
            by_source={s: dict(v) for s, v in data["by_source"].items()},
            uncovered=[dict(u) for u in data["uncovered"]],
            rounds=int(data.get("rounds", 0)),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def run_coverage(
    ctx: BuildContext, outline: Outline, blocks: Mapping[str, Block]
) -> tuple[StageResult, CoverageReport]:
    """S4 → S5 (at most two rounds) over final/; writes final/sections/*, coverage.json."""
    t0 = time.monotonic()
    result = StageResult(stage=STAGE, ok=False)
    final_dir = ctx.synth_dir / FINAL_DIR
    final_sections = final_dir / "sections"
    coverage_path = ctx.synth_dir / "coverage.json"
    block_map = {b.id: b for b in _block_list(blocks)}
    outline_md = outline_text(ctx, outline)
    outline_file = _outline_file(ctx, outline)

    if not final_sections.is_dir():
        report = compute_coverage(block_map, final_sections)
        result.errors.append("Нет final/sections/ — сначала должен выполниться этап global.")
        result.duration_s = round(time.monotonic() - t0, 3)
        return result, report

    key = _key(ctx, block_map, outline_md, final_dir)
    meta = read_meta(ctx, STAGE)
    if meta and meta.get("ok") and cached(ctx, STAGE, key, [coverage_path]):
        stored = _report_from_json(coverage_path)
        if stored is not None:
            result.ok, result.cached = True, True
            result.warnings += [str(w) for w in meta.get("warnings") or []]
            result.details = {"ratio": round(stored.ratio, 4), "rounds": stored.rounds}
            result.duration_s = round(time.monotonic() - t0, 3)
            return result, stored

    verdicts: dict[str, dict[str, Any]] = {}
    ordered = [s.id for s in outline.sections]
    rounds = 0
    error: str | None = None
    while True:
        base = compute_coverage(block_map, final_sections)
        pending = [
            block_map[u["block"]]
            for u in base.uncovered
            if verdicts.get(u["block"], {}).get("verdict") not in COVERED_VERDICTS
        ]
        if not pending or rounds >= MAX_ROUNDS:
            break
        rounds += 1
        index_md = master_index(final_dir, ordered)
        s4 = _run_s4(ctx, pending, outline_file=outline_file, index_md=index_md, round_no=rounds)
        result.agent_runs += s4.runs
        result.warnings += s4.warnings
        if s4.error:
            error = "Разбор непокрытых блоков (S4) не выполнен: " + s4.error
            break
        verdicts.update(s4.verdicts)
        missing = [b.id for b in pending if verdicts[b.id]["verdict"] == "missing"]
        if not missing:
            break
        s5 = _run_s5(
            ctx, outline, block_map, missing, verdicts, outline_file=outline_file, round_no=rounds
        )
        result.agent_runs += s5.runs
        result.warnings += s5.warnings
        if s5.error:
            error = "Дополнение разделов (S5) не выполнено: " + s5.error
            break

    report = _with_verdicts(
        compute_coverage(block_map, final_sections), block_map, verdicts, rounds
    )
    still_missing = [u["block"] for u in report.uncovered if u["verdict"] == "missing"]
    unknown = [u["block"] for u in report.uncovered if u["verdict"] == "unknown"]
    if still_missing:
        result.warnings.append(
            f"После {rounds} кругов в мастере по-прежнему нет {len(still_missing)} блоков: "
            + fmt_ids(still_missing, limit=20)
        )
    if error:
        result.errors.append(error)
        if unknown:
            result.warnings.append(f"Блоков без вердикта: {len(unknown)}")

    payload = report.to_dict()
    payload["verdicts"] = list(verdicts.values())
    write_json_atomic(coverage_path, payload)
    result.ok = error is None
    result.details = {
        "ratio": round(report.ratio, 4),
        "rounds": rounds,
        "uncovered": len(report.uncovered),
        "verdicts": {
            kind: sum(1 for v in verdicts.values() if v["verdict"] == kind) for kind in VERDICTS
        },
    }
    write_meta(
        ctx,
        STAGE,
        {
            "key": _key(ctx, block_map, outline_md, final_dir),  # after S5 edits to final/
            "prompt": f"{prompt_id('coverage')}+{prompt_id('supplement')}",
            "model": " / ".join(
                (model_for(ctx, "light", "coverage"), model_for(ctx, "strong", "supplement"))
            ),
            "duration_s": round(time.monotonic() - t0, 3),
            "agent_runs": result.agent_runs,
            "ok": result.ok,
            "warnings": list(result.warnings),
        },
    )
    result.duration_s = round(time.monotonic() - t0, 3)
    return result, report
