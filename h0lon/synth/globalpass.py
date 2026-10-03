"""S3 — global pass over the sections written by S2 (docs/ARCHITECTURE.md, «Синтез (M2)»).

Input: `synthesis/sections/<id>.md` (+ `<id>.notes.json`) in the order of the outline. One strong
agent run unifies notation and terms, removes repetitions between sections, links them and writes
the introduction and the glossary. The agent edits copies of the sections *in place*: they are
placed into `out/sections/` as a seed (the runner restores them before every attempt).

Files (all under `<topic>/synthesis/`):

- `inputs/global/` — what the agent gets (rewritten before every part): `outline.md`,
  `glossary.md` (a copy of `inputs/terms.md`, or built from the `summary.md` of the sources when
  that file is absent),
  `notes.json` (all S2 notes merged, every record has `section`), `sources.md` (titles of the
  sources for the introduction), and for a split run `overview.md` (headings of the whole master);
- `global/` — accepted result: `global/sections/<id>.md`, `intro.md`, `glossary.md`,
  `global.notes.json`;
- `final/` — the current text read by assembly and extended by S5: `final/sections/<id>.md`,
  `final/intro.md`, `final/glossary.md`. On success it is a copy of `global/`; when the global pass
  fails (decision A5 in docs/DECISIONS-AUTONOMOUS.md) it is a copy of `sections/` without intro and
  glossary (assembly then writes a minimal introduction itself);
- `global.meta.json` — stage cache (key = section texts, notes, outline, terms, prompt, model).

Checks in code after every run (a failure means a retry with feedback in a *new* bundle that again
starts from the S2 texts; at most `MAX_RETRIES` = 2 repeats, then A5):

1. the set of block ids in `<!-- src: … -->` comments did not shrink (intro and glossary count too);
2. every `{#sec:…}` heading of a section is still there at the same level;
3. the total text size (`common.text_size`) is at least 90 % of the original.

A runner failure that is not a validation failure (not logged in, rate limit, crash) is an
infrastructure problem: the stage returns `ok=False` instead of falling back to A5, because the
same would happen on the next stage.

Large themes: when the sections together exceed `MAX_RUN_CHARS` (350 000 characters), the pass is
split by chapters: whole chapters are packed into parts of at most that size (a bigger chapter is a
part of its own). Every part is a separate run with the same prompt, the checks above apply to the
part, and `intro.md` / `glossary.md` are written by the *last* run, which in addition gets
`overview.md` (all headings of the master, from the outline) and the glossary of terms. If any part
fails, the whole stage falls back to A5 (a half-unified master is worse than a consistent one).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h0lon.agents import ExpectedFile, OutputContract
from h0lon.agents.runner import RunResult
from h0lon.synth.common import (
    cached,
    model_for,
    parse_src_ids,
    prompt_body,
    prompt_id,
    read_meta,
    run_agent,
    sections_dir,
    src_ids_in_files,
    stage_key,
    text_size,
    write_json_atomic,
    write_meta,
)
from h0lon.synth.model import BuildContext, Outline, OutlineSection, StageResult

STAGE = "global"
PROMPT = "globalpass"
FINAL_DIR = "final"
GLOBAL_DIR = "global"
MAX_RETRIES = 2  # repeats after the first run (so at most 3 runs per part)
MIN_SIZE_RATIO = 0.9  # total text size after the pass / before
MAX_RUN_CHARS = 350_000  # above this the pass is split by chapters
MAX_LISTED = 40  # ids listed in a feedback message

_NOTE_KEYS = ("corrections", "conflicts", "editorial")
_str = {"type": "string"}
_str_or_null = {"type": ["string", "null"]}

# global.notes.json: same sections of notes as in S2, every record names its section.
GLOBAL_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": list(_NOTE_KEYS),
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["section", "as_written", "corrected"],
                "properties": {
                    "section": _str,
                    "anchor": _str_or_null,
                    "block": _str_or_null,
                    "as_written": _str,
                    "corrected": _str,
                    "reason": _str,
                },
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["section", "topic", "variants"],
                "properties": {
                    "section": _str,
                    "topic": _str,
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["text"],
                            "properties": {"anchor": _str_or_null, "text": _str},
                        },
                    },
                    "resolution": _str_or_null,
                    "reason": _str,
                },
            },
        },
        "editorial": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["section", "kind", "text"],
                "properties": {
                    "section": _str,
                    "kind": {"enum": ["answer", "reconstruction", "clarification"]},
                    "refers_to": _str_or_null,
                    "text": _str,
                },
            },
        },
    },
}


# ---------------------------------------------------------------- shared helpers
# (coverage.py imports these: they are not part of common.py, which is not mine to change)

_HEADING_RE = re.compile(r"^(#{1,6})\s+.*\{[^}\n]*#(sec:[^\s}]+)[^}\n]*\}\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def heading_anchors(markdown: str) -> dict[str, int]:
    """`{#sec:…}` anchors of the headings outside code fences: {'sec:s01': 1, …} (level)."""
    found: dict[str, int] = {}
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            found[m.group(2)] = len(m.group(1))
    return found


def anchor_problems(before: dict[str, int], after: dict[str, int], where: str) -> list[str]:
    """Headings of the original that vanished or changed their level."""
    problems = []
    for anchor, level in before.items():
        if anchor not in after:
            problems.append(f"в {where} пропал заголовок {{#{anchor}}} — его нужно вернуть как был")
        elif after[anchor] != level:
            problems.append(
                f"в {where} у заголовка {{#{anchor}}} изменился уровень "
                f"({'#' * level} → {'#' * after[anchor]})"
            )
    return problems


def fmt_ids(ids: Iterable[str], *, limit: int = MAX_LISTED) -> str:
    items = sorted(ids)
    text = ", ".join(items[:limit])
    return text + (f" … (ещё {len(items) - limit})" if len(items) > limit else "")


def render_outline(outline: Outline) -> str:
    """Fallback `outline.md` for the agent when S1 left no file."""
    lines = [f"# {outline.title}", ""]
    for s in outline.sections:
        pad = "  " * max(0, s.level - 1)
        line = f"{pad}- `{s.id}` {s.title}"
        if s.summary:
            line += f" — {s.summary}"
        if s.blocks:
            line += f" ({len(s.blocks)} блоков)"
        lines.append(line)
    return "\n".join(lines) + "\n"


def outline_text(ctx: BuildContext, outline: Outline) -> str:
    path = ctx.synth_dir / "outline.md"
    return read_text(path) if path.is_file() else render_outline(outline)


def sources_listing(ctx: BuildContext) -> str:
    """`sources.md`: titles of the sources (the introduction names them)."""
    lines: list[str] = []
    try:
        from h0lon.sources.ingest import list_sources

        lines = [f"- {r.id} ({r.kind}) — {r.title}" for r in list_sources(ctx.topic_dir)]
    except Exception:
        lines = []
    if not lines:
        root = ctx.topic_dir / "extracted"
        if root.is_dir():
            lines = [f"- {p.name}" for p in sorted(root.iterdir()) if p.is_dir()]
    body = "\n".join(lines) if lines else "(список источников недоступен)"
    return "# Источники темы\n\n" + body + "\n"


_TERMS_SECTION = re.compile(r"^##\s+Термины и обозначения\s*$", re.MULTILINE)
_NEXT_H2 = re.compile(r"^##\s", re.MULTILINE)


def terms_text(ctx: BuildContext) -> str:
    """`inputs/terms.md`; when S1/S2 did not write it, the terms of the sources' summaries."""
    path = ctx.synth_dir / "inputs" / "terms.md"
    if path.is_file():
        return read_text(path)
    parts: list[str] = []
    root = ctx.topic_dir / "extracted"
    for src in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        summary = src / "summary.md"
        if not summary.is_file():
            continue
        text = read_text(summary)
        m = _TERMS_SECTION.search(text)
        if not m:
            continue
        rest = text[m.end() :]
        nxt = _NEXT_H2.search(rest)
        body = (rest[: nxt.start()] if nxt else rest).strip()
        if body:
            parts.append(f"### Источник {src.name}\n\n{body}")
    return "# Термины и обозначения\n\n" + ("\n\n".join(parts) if parts else "(нет данных)") + "\n"


def chapter_of(sec: OutlineSection) -> str:
    return sec.parent or sec.id.split("-")[0]


# ---------------------------------------------------------------- sections


@dataclass
class _Sec:
    id: str
    title: str
    level: int
    path: Path
    text: str
    chapter: str
    srcs: set[str] = field(default_factory=set)
    anchors: dict[str, int] = field(default_factory=dict)
    size: int = 0  # text_size of the original


def _collect(ctx: BuildContext, outline: Outline) -> tuple[list[_Sec], list[str], list[str]]:
    """Sections of S2 in outline order; (sections, warnings, errors)."""
    sdir = sections_dir(ctx)
    sections: list[_Sec] = []
    warnings: list[str] = []
    missing: list[str] = []
    known = set()
    for s in outline.sections:
        known.add(s.id)
        path = sdir / f"{s.id}.md"
        text = read_text(path) if path.is_file() else ""
        if text.strip():
            sections.append(
                _Sec(
                    id=s.id,
                    title=s.title,
                    level=s.level,
                    path=path,
                    text=text,
                    chapter=chapter_of(s),
                    srcs=parse_src_ids(text),
                    anchors=heading_anchors(text),
                    size=text_size(text),
                )
            )
        elif s.blocks:
            missing.append(s.id)
    if sdir.is_dir():
        extra = sorted(p.stem for p in sdir.glob("*.md") if p.stem not in known)
        if extra:
            warnings.append(
                "Файлы разделов, которых нет в структуре, пропущены: " + ", ".join(extra)
            )
    errors = []
    if missing:
        errors.append("Нет текста разделов (этап S2 не завершён): " + ", ".join(missing))
    elif not sections:
        errors.append("Нет ни одного раздела в synthesis/sections/ (этап S2 не выполнен).")
    return sections, warnings, errors


def collect_notes(ctx: BuildContext, ids: Sequence[str]) -> tuple[dict[str, list], list[str]]:
    """All `sections/<id>.notes.json` merged into one structure; every record has `section`."""
    merged: dict[str, list] = {k: [] for k in _NOTE_KEYS}
    warnings: list[str] = []
    for sid in ids:
        path = sections_dir(ctx) / f"{sid}.notes.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(read_text(path))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Заметки раздела {sid} не прочитаны: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        for key in _NOTE_KEYS:
            for rec in data.get(key) or []:
                if isinstance(rec, dict):
                    own = rec.get("section") or sid
                    merged[key].append(
                        {"section": own, **{k: v for k, v in rec.items() if k != "section"}}
                    )
    return merged, warnings


def _split_parts(sections: Sequence[_Sec]) -> list[list[_Sec]]:
    """Whole chapters packed into runs of at most MAX_RUN_CHARS (one run for a normal theme)."""
    chapters: dict[str, list[_Sec]] = {}
    for s in sections:
        chapters.setdefault(s.chapter, []).append(s)
    parts: list[list[_Sec]] = []
    current: list[_Sec] = []
    size = 0
    for chapter in chapters.values():
        chapter_size = sum(len(s.text) for s in chapter)
        if current and size + chapter_size > MAX_RUN_CHARS:
            parts.append(current)
            current, size = [], 0
        current += chapter
        size += chapter_size
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------- checks


def _check_part(
    out_dir: Path,
    part: Sequence[_Sec],
    *,
    owner: dict[str, str],
    with_front: bool,
) -> tuple[list[str], dict[str, Any]]:
    """Code checks of one run; (problems for the feedback, details)."""
    problems: list[str] = []
    after_texts: dict[str, str] = {}
    for s in part:
        path = out_dir / "sections" / f"{s.id}.md"
        if not path.is_file():
            problems.append(f"нет файла out/sections/{s.id}.md")
            continue
        after_texts[s.id] = read_text(path)
    for s in part:
        if s.id in after_texts:
            problems += anchor_problems(
                s.anchors, heading_anchors(after_texts[s.id]), f"out/sections/{s.id}.md"
            )

    before_srcs = set().union(*(s.srcs for s in part)) if part else set()
    after_srcs = (
        set().union(*(parse_src_ids(t) for t in after_texts.values())) if after_texts else set()
    )
    if with_front:
        after_srcs |= src_ids_in_files([out_dir / "intro.md", out_dir / "glossary.md"])
    lost = before_srcs - after_srcs
    if lost:
        detail = ", ".join(f"{b} (раздел {owner.get(b, '?')})" for b in sorted(lost)[:MAX_LISTED])
        more = f" … (ещё {len(lost) - MAX_LISTED})" if len(lost) > MAX_LISTED else ""
        problems.append(
            f"потеряны ссылки на блоки источников ({len(lost)}): {detail}{more}. Идентификатор "
            "блока должен остаться в комментарии `<!-- src: … -->` того фрагмента, который "
            "теперь покрывает его содержание; при слиянии абзацев комментарии объединяются"
        )

    size_before = sum(s.size for s in part)
    size_after = sum(text_size(t) for t in after_texts.values())
    if size_before and size_after < MIN_SIZE_RATIO * size_before:
        drops = sorted(
            (
                (text_size(after_texts[s.id]) - s.size, s)
                for s in part
                if s.id in after_texts and text_size(after_texts[s.id]) < s.size
            ),
            key=lambda item: item[0],
        )[:6]
        worst = "; ".join(
            f"{s.id}: {s.size} → {s.size + delta} ({delta / s.size:+.0%})" for delta, s in drops
        )
        problems.append(
            f"просел объём текста: было {size_before} символов, стало {size_after} "
            f"({size_after / size_before:.0%}; нужно не меньше {MIN_SIZE_RATIO:.0%}). "
            f"Сильнее всего: {worst}. Не сокращай: перенеси опущенные детали обратно"
        )
    details = {
        "lost": sorted(lost),
        "size_before": size_before,
        "size_after": size_after,
    }
    return problems, details


# ---------------------------------------------------------------- one run


@dataclass
class _Outcome:
    ok: bool = False
    infra_error: str | None = None
    problems: list[str] = field(default_factory=list)
    runs: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, list] = field(default_factory=dict)  # accepted global.notes.json


def _is_validation_failure(result: RunResult) -> bool:
    return bool(result.attempts) and result.attempts[-1].error_kind == "validation"


def _contract(part: Sequence[_Sec], *, last: bool) -> OutputContract:
    files = [
        ExpectedFile(
            path=f"sections/{s.id}.md",
            kind="markdown",
            required_text=[f"{{#sec:{s.id}"],
            description=f"Раздел «{s.title}» — правь на месте.",
        )
        for s in part
    ]
    if last:
        files.append(
            ExpectedFile(
                path="intro.md",
                kind="markdown",
                min_chars=80,
                required_text=["{#sec:intro"],
                description="Введение.",
            )
        )
        files.append(
            ExpectedFile(
                path="glossary.md",
                kind="markdown",
                min_chars=40,
                required_text=["{#sec:glossary"],
                description="Глоссарий терминов и обозначений.",
            )
        )
    files.append(
        ExpectedFile(
            path="global.notes.json",
            kind="json",
            json_schema=GLOBAL_NOTES_SCHEMA,
            description="Правки, расхождения и дополнения этого прогона (пустые списки допустимы).",
        )
    )
    return OutputContract(files=files)


def _task(
    part: Sequence[_Sec],
    *,
    number: int,
    total: int,
    last: bool,
    has_overview: bool,
    feedback: Sequence[str],
) -> str:
    lines = [prompt_body("globalpass").rstrip(), "", "## Файлы этого прогона", ""]
    lines.append("Разделы для правки (копии лежат в `out/sections/`, в порядке структуры):")
    for s in part:
        lines.append(f"- `out/sections/{s.id}.md` — {s.title} ({len(s.text)} символов)")
    lines += [
        "",
        "Входы: `inputs/outline.md` — структура темы; `inputs/glossary.md` — объединённые "
        "термины и обозначения источников (ориентир для единой нотации); `inputs/notes.json` — "
        "заметки этапа написания разделов (у каждой записи поле `section`); "
        "`inputs/sources.md` — названия источников для введения.",
    ]
    if total > 1:
        lines += [
            "",
            f"Это часть {number} из {total}: тема большая, поэтому правка разделена по главам. "
            "В этом прогоне правь только перечисленные разделы; остальные правятся отдельными "
            "прогонами — их файлов здесь нет, не создавай их. Ссылки на них (`#sec:<id>`) "
            "допустимы, идентификаторы разделов — в `inputs/outline.md`.",
        ]
        if last and has_overview:
            lines.append(
                "В этом прогоне ты пишешь `out/intro.md` и `out/glossary.md` для всего документа: "
                "заголовки всех разделов — в `inputs/overview.md`, термины — в "
                "`inputs/glossary.md`; в глоссарий и введение включай материал всех частей."
            )
        else:
            lines.append(
                "Введение и глоссарий в этом прогоне не нужны (их пишет последняя часть): "
                "`out/intro.md` и `out/glossary.md` не создавай."
            )
    if feedback:
        lines += [
            "",
            "## Замечания автоматической проверки к предыдущему прогону",
            "",
            "Предыдущий результат отклонён и не используется: в `out/sections/` снова лежат "
            "исходные тексты разделов. Сделай правку заново и исправь:",
        ]
        lines += [f"- {p}" for p in feedback]
    return "\n".join(lines) + "\n"


def _stage_inputs(
    base: Path,
    *,
    outline_md: str,
    glossary: str,
    notes: dict[str, list],
    sources_md: str,
    overview: str | None,
) -> list[Path]:
    base.mkdir(parents=True, exist_ok=True)
    files = {"outline.md": outline_md, "glossary.md": glossary, "sources.md": sources_md}
    if overview is not None:
        files["overview.md"] = overview
    else:
        (base / "overview.md").unlink(missing_ok=True)  # left by the previous part
    paths = []
    for name, text in files.items():
        write_text(base / name, text)
        paths.append(base / name)
    write_json_atomic(base / "notes.json", notes)
    paths.append(base / "notes.json")
    return paths


def _overview(outline: Outline) -> str:
    lines = ["# Заголовки всего документа", ""]
    for s in outline.sections:
        pad = "  " * max(0, s.level - 1)
        lines.append(
            f"{pad}- `{s.id}` {'#' * s.level} {s.title}" + (f" — {s.summary}" if s.summary else "")
        )
    return "\n".join(lines) + "\n"


def _run_part(
    ctx: BuildContext,
    part: Sequence[_Sec],
    *,
    number: int,
    total: int,
    owner: dict[str, str],
    inputs_base: Path,
    outline_md: str,
    glossary: str,
    notes: dict[str, list],
    sources_md: str,
    overview: str | None,
    global_dir: Path,
) -> _Outcome:
    last = number == total
    ids = {s.id for s in part}
    part_notes = {k: [r for r in v if r.get("section") in ids] for k, v in notes.items()}
    inputs = _stage_inputs(
        inputs_base,
        outline_md=outline_md,
        glossary=glossary,
        notes=part_notes,
        sources_md=sources_md,
        overview=overview if (last and total > 1) else None,
    )
    seed = {f"sections/{s.id}.md": s.path for s in part}
    outcome = _Outcome()
    feedback: list[str] = []
    for attempt in range(1 + MAX_RETRIES):
        label = f"часть {number}/{total}, " if total > 1 else ""
        ctx.emit(f"Глобальная правка: {label}запуск {attempt + 1} из {1 + MAX_RETRIES}")
        result = run_agent(
            ctx,
            stage=STAGE,
            task=_task(
                part,
                number=number,
                total=total,
                last=last,
                has_overview=overview is not None,
                feedback=feedback,
            ),
            contract=_contract(part, last=last),
            inputs=inputs,
            seed=seed,
            tier="strong",
        )
        outcome.runs += 1
        if not result.ok:
            if _is_validation_failure(result):
                feedback = [str(p) for p in result.problems[:10]] or [
                    "результат не прошёл проверку контракта"
                ]
                outcome.problems = feedback
                continue
            outcome.infra_error = (
                "; ".join(str(p) for p in result.problems[:3]) or "агент не ответил"
            )
            return outcome
        out_dir = result.bundle.out_dir
        problems, details = _check_part(out_dir, part, owner=owner, with_front=last)
        outcome.details = details
        if not problems:
            for s in part:
                _copy(out_dir / "sections" / f"{s.id}.md", global_dir / "sections" / f"{s.id}.md")
            if last:
                _copy(out_dir / "intro.md", global_dir / "intro.md")
                _copy(out_dir / "glossary.md", global_dir / "glossary.md")
            outcome.notes = _read_notes(out_dir / "global.notes.json")
            outcome.ok = True
            outcome.problems = []
            return outcome
        feedback = problems
        outcome.problems = problems
    return outcome


def _read_notes(path: Path) -> dict[str, list]:
    data = json.loads(read_text(path))
    return {key: list(data.get(key) or []) for key in _NOTE_KEYS}


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


# ---------------------------------------------------------------- stage


def _final_outputs(ids: Sequence[str], final_dir: Path, *, fallback: bool) -> list[Path]:
    outs = [final_dir / "sections" / f"{i}.md" for i in ids]
    if not fallback:
        outs += [final_dir / "intro.md", final_dir / "glossary.md"]
    return outs


def _publish(
    ctx: BuildContext, ids: Sequence[str], *, fallback: bool, global_dir: Path, final_dir: Path
) -> None:
    """Fill final/ from global/ (success) or from sections/ (A5)."""
    shutil.rmtree(final_dir, ignore_errors=True)
    (final_dir / "sections").mkdir(parents=True, exist_ok=True)
    source = sections_dir(ctx) if fallback else global_dir / "sections"
    for sid in ids:
        _copy(source / f"{sid}.md", final_dir / "sections" / f"{sid}.md")
    if not fallback:
        _copy(global_dir / "intro.md", final_dir / "intro.md")
        _copy(global_dir / "glossary.md", final_dir / "glossary.md")


def run_global(ctx: BuildContext, outline: Outline) -> StageResult:
    """S3: global pass (see the module docstring). Writes global/, final/ and global.meta.json."""
    t0 = time.monotonic()
    result = StageResult(stage=STAGE, ok=False)
    sections, warnings, errors = _collect(ctx, outline)
    result.warnings += warnings
    if errors:
        result.errors = errors
        result.duration_s = round(time.monotonic() - t0, 3)
        return result

    ids = [s.id for s in sections]
    notes, note_warnings = collect_notes(ctx, ids)
    result.warnings += note_warnings
    outline_md = outline_text(ctx, outline)
    glossary = terms_text(ctx)
    sources_md = sources_listing(ctx)
    model = model_for(ctx, "strong", STAGE)
    prompt = prompt_id(PROMPT)
    key = stage_key(
        ids,
        [s.text for s in sections],
        notes,
        outline_md,
        glossary,
        sources_md,
        prompt,
        model,
    )

    final_dir = ctx.synth_dir / FINAL_DIR
    global_dir = ctx.synth_dir / GLOBAL_DIR
    meta = read_meta(ctx, STAGE)
    if meta and meta.get("ok"):
        fallback_prev = bool(meta.get("fallback"))
        if cached(ctx, STAGE, key, _final_outputs(ids, final_dir, fallback=fallback_prev)):
            result.ok, result.cached = True, True
            result.warnings += [str(w) for w in meta.get("warnings") or []]
            result.details = {"fallback": fallback_prev, "sections": len(ids)}
            result.duration_s = round(time.monotonic() - t0, 3)
            return result

    shutil.rmtree(global_dir, ignore_errors=True)
    parts = _split_parts(sections)
    owner: dict[str, str] = {}
    for s in sections:
        for b in s.srcs:
            owner.setdefault(b, s.id)
    overview = _overview(outline) if len(parts) > 1 else None

    failed_problems: list[str] = []
    merged_notes: dict[str, list] = {name: [] for name in _NOTE_KEYS}
    stage_details: dict[str, Any] = {"parts": len(parts), "sections": len(ids), "runs": []}
    infra_error: str | None = None
    for number, part in enumerate(parts, start=1):
        outcome = _run_part(
            ctx,
            part,
            number=number,
            total=len(parts),
            owner=owner,
            inputs_base=ctx.synth_dir / "inputs" / GLOBAL_DIR,
            outline_md=outline_md,
            glossary=glossary,
            notes=notes,
            sources_md=sources_md,
            overview=overview,
            global_dir=global_dir,
        )
        result.agent_runs += outcome.runs
        stage_details["runs"].append(
            {"part": number, "runs": outcome.runs, "ok": outcome.ok, **outcome.details}
        )
        if outcome.infra_error:
            infra_error = outcome.infra_error
            break
        if not outcome.ok:
            failed_problems = outcome.problems
            break
        for name in _NOTE_KEYS:
            merged_notes[name] += outcome.notes.get(name, [])

    if infra_error is not None:
        shutil.rmtree(global_dir, ignore_errors=True)
        result.errors.append("Глобальная правка не выполнена: " + infra_error)
        _write_meta(ctx, key, prompt, model, result, t0, ok=False, fallback=False)
        result.duration_s = round(time.monotonic() - t0, 3)
        return result

    fallback = bool(failed_problems)
    if fallback:
        # Decision A5: a master without global unification is better than a master with losses.
        shutil.rmtree(global_dir, ignore_errors=True)
        result.warnings.append(
            "Глобальная правка не прошла проверку после "
            f"{1 + MAX_RETRIES} запусков; разделы остаются в том виде, в каком их написал S2 "
            "(без единой нотации, введения и глоссария). Причины: " + "; ".join(failed_problems)
        )
    else:
        write_json_atomic(global_dir / "global.notes.json", merged_notes)
    _publish(ctx, ids, fallback=fallback, global_dir=global_dir, final_dir=final_dir)

    stage_details["fallback"] = fallback
    result.details = stage_details
    result.ok = True
    _write_meta(ctx, key, prompt, model, result, t0, ok=True, fallback=fallback)
    result.duration_s = round(time.monotonic() - t0, 3)
    return result


def _write_meta(
    ctx: BuildContext,
    key: str,
    prompt: str,
    model: str,
    result: StageResult,
    t0: float,
    *,
    ok: bool,
    fallback: bool,
) -> None:
    write_meta(
        ctx,
        STAGE,
        {
            "key": key,
            "prompt": prompt,
            "model": model,
            "duration_s": round(time.monotonic() - t0, 3),
            "agent_runs": result.agent_runs,
            "ok": ok,
            "fallback": fallback,
            "warnings": list(result.warnings),
        },
    )
