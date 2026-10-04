"""`h0lon build`: the stages of the synthesis in order, review gate, status, git commit.

Contract: docs/ARCHITECTURE.md, «Синтез мастер-конспекта (M2)». The stage modules S1–S5
(outline, sections, globalpass, coverage) are imported lazily, each stage is cached by its own
module (`synthesis/<stage>.meta.json`). Decisions of this module:

- `from_stage` and `force` make the stages from that one on ignore their caches. `force` alone
  covers the synthesis stages (S1…render); re-extraction is `from_stage="extract"` (it costs
  agent runs for page recognition and summaries, and has its own cache).
- Extraction always uses the extraction agent of the settings: passing `backend` there would
  change the cache key of every extracted source. `backend` is for the synthesis stages.
- `synthesis/build.json` remembers, for every stage, the cache keys of the stages before it as
  they were when it ran; `topic_status` compares them with the current ones to tell «устарело».
- After a successful build the topic repository gets a commit (`runs/` and `build.json` are
  not committed: agent transcripts are large and local, build.json changes on every run).
- Before that commit `synthesis/changelog.md` gets an entry (sources added or changed since the
  previous build, what S1 did, the sections S2 rewrote, coverage). The previous set of sources is
  kept in build.json (`sources`). A build that changed nothing writes no entry.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from h0lon import procutil, tools
from h0lon.extract.model import Block
from h0lon.synth.common import (
    load_blocks,
    percent,
    read_meta,
    stage_key,
    write_json_atomic,
    write_meta,
)
from h0lon.synth.model import (
    MASTER_MD,
    MASTER_PDF,
    STAGES,
    BuildContext,
    BuildResult,
    CoverageReport,
    EventCallback,
    Outline,
    StageResult,
)

if TYPE_CHECKING:
    from rich.console import Console

    from h0lon.config import Settings
    from h0lon.sources.models import SourceRecord

BUILD_FILE = "build.json"
COVERAGE_FILE = "coverage.json"
CHANGELOG_FILE = "changelog.md"
MAX_CHANGELOG_SECTIONS = 12  # section ids listed in one changelog line
FINAL_SECTIONS = ("final", "sections")
STAGE_TITLES: dict[str, str] = {
    "extract": "Извлечение источников",
    "outline": "Структура темы (S1)",
    "sections": "Разделы (S2)",
    "global": "Глобальная правка (S3)",
    "coverage": "Полнота (S4/S5)",
    "assemble": "Сборка master.md",
    "render": "PDF (S6)",
}
MAX_WARNINGS_PRINTED = 6
REVIEW_TITLE = "Review gate"
GIT_TITLE = "Коммит в git темы"
_GIT_ENV_DROP = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_NAMESPACE",
)
_GIT_TIMEOUT_S = 120
# `runs/` (agent transcripts), build.json (bookkeeping of the last run) and render build dirs
# stay local. They are kept out through the topic .gitignore (older topics get the missing
# patterns appended): an explicit `:(exclude)` of an already ignored path makes `git add` fail.
_GIT_PATHSPEC = (".",)
_GIT_LOCAL_PATTERNS = ("*.build/", "*.build-*/")


def _ensure_gitignore(topic: Path) -> None:
    from h0lon.workspace import GITIGNORE_PATTERNS

    path = topic / ".gitignore"
    try:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return
    present = {line.strip() for line in text.splitlines()}
    missing = [p for p in (*GITIGNORE_PATTERNS, *_GIT_LOCAL_PATTERNS) if p not in present]
    if missing:
        sep = "" if not text or text.endswith("\n") else "\n"
        path.write_text(text + sep + "\n".join(missing) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- small helpers


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def topic_ref(settings: Settings, topic_dir: Path) -> str:
    """How to name the topic in a command hint: `<курс>/<тема>` inside the workspaces dir."""
    topic_dir = Path(topic_dir).resolve()
    try:
        ref = topic_dir.relative_to(Path(settings.general.workspaces_dir).resolve()).as_posix()
    except (ValueError, OSError):
        ref = str(topic_dir)
    return f'"{ref}"' if " " in ref else ref


def _stage_fn(module: str, name: str) -> Callable[..., Any]:
    """Function of a stage module, imported at call time (the modules are written separately
    and tests replace them in `sys.modules`)."""
    try:
        mod = importlib.import_module(f"h0lon.synth.{module}")
    except ImportError as exc:
        raise RuntimeError(f"Модуль стадии h0lon.synth.{module} недоступен: {exc}") from exc
    return getattr(mod, name)


def _extracted(records: Sequence[SourceRecord]) -> list[SourceRecord]:
    return [r for r in records if r.status == "extracted"]


def _save_topic_atomic(topic_dir: Path, meta: Any) -> None:
    from h0lon.workspace import TOPIC_FILE, save_topic

    target = topic_dir / TOPIC_FILE
    fd, tmp = tempfile.mkstemp(prefix=f".{TOPIC_FILE}.", suffix=".tmp", dir=topic_dir)
    os.close(fd)
    try:
        save_topic(Path(tmp), meta)
        for attempt in range(20):  # os.replace fails on Windows while another process reads
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------- review gate


def review_state(
    settings: Settings, topic_dir: Path, records: Sequence[SourceRecord]
) -> dict[str, Any]:
    """Review gate of a topic: is it on, was the extraction approved, is the approval current."""
    from h0lon.workspace import load_topic

    meta = load_topic(Path(topic_dir))
    gate = meta.review_gate if meta.review_gate is not None else settings.general.review_gate
    raw = (meta.model_extra or {}).get("review")
    approved_at = raw.get("approved_at") if isinstance(raw, dict) else None
    keys = raw.get("extracted_keys") if isinstance(raw, dict) else None
    current = {r.id: r.extracted_key or "" for r in _extracted(records)}
    changed: list[str] = []
    if isinstance(keys, dict):
        stored = {str(k): str(v or "") for k, v in keys.items()}
        changed = sorted(
            sid for sid in set(stored) | set(current) if stored.get(sid) != current.get(sid)
        )
    elif isinstance(keys, list):  # a hand-written list of keys: compare as a set
        if sorted(str(k) for k in keys) != sorted(current.values()):
            changed = sorted(current)
    approved = bool(approved_at) and isinstance(keys, dict | list)
    return {
        "gate": bool(gate),
        "approved": approved,
        "approved_at": approved_at if approved else None,
        "valid": approved and not changed,
        "changed": changed,
    }


def _review_message(
    settings: Settings, topic_dir: Path, state: dict[str, Any], records: Sequence[SourceRecord]
) -> str:
    ref = topic_ref(settings, topic_dir)
    if state["approved"] and state["changed"]:
        why = "Извлечение изменилось после одобрения (" + ", ".join(state["changed"]) + "). "
    else:
        why = "Извлечение ещё не одобрено. "
    noted = [
        f"{r.id} ({len((r.quality or {}).get('notes') or [])})"
        for r in _extracted(records)
        if (r.quality or {}).get("notes")
    ]
    notes = f"Замечания к качеству извлечения: {', '.join(noted)}. " if noted else ""
    return (
        f"{why}{notes}Проверьте Source Docs в extracted/ и выполните h0lon approve {ref} "
        "(или h0lon build --no-review)."
    )


def approve_topic(settings: Settings, topic_dir: Path) -> dict[str, Any]:
    """Record the approval of the current extraction in topic.yaml (`review:`).

    Raises ValueError when a source is not extracted yet (`h0lon extract` first).
    """
    from h0lon.sources.ingest import list_sources, topic_lock
    from h0lon.workspace import load_topic

    topic_dir = Path(topic_dir).resolve()
    ref = topic_ref(settings, topic_dir)
    with topic_lock(topic_dir):
        records = list_sources(topic_dir)
        if not records:
            raise ValueError(f"В теме нет источников: добавьте их командой h0lon add {ref} <файлы>")
        pending = [r.id for r in records if r.status not in ("extracted", "skipped")]
        if pending:
            raise ValueError(
                f"Не все источники извлечены ({', '.join(pending)}). "
                f"Сначала выполните h0lon extract {ref}."
            )
        extracted = _extracted(records)
        if not extracted:
            raise ValueError(
                "Ни один источник не извлечён (все пропущены): одобрять нечего. "
                f"Добавьте поддерживаемые источники и выполните h0lon extract {ref}."
            )
        approval = {
            "approved_at": _now_iso(),
            "extracted_keys": {r.id: r.extracted_key or "" for r in extracted},
        }
        meta = load_topic(topic_dir)
        meta.review = approval  # type: ignore[attr-defined]  # extra="allow"
        _save_topic_atomic(topic_dir, meta)
    return {"topic": str(topic_dir), "sources": [r.id for r in extracted], **approval}


# ---------------------------------------------------------------- build.json


def _stage_keys(ctx: BuildContext, records: Sequence[SourceRecord]) -> dict[str, str | None]:
    """Current cache key of every stage ("extract": hash of the extracted keys of the sources)."""
    keys: dict[str, str | None] = {
        "extract": stage_key({r.id: r.extracted_key for r in _extracted(records)})
    }
    for stage in STAGES[1:]:
        meta = read_meta(ctx, stage)
        keys[stage] = str(meta["key"]) if meta and meta.get("key") else None
    return keys


def _read_build_file(ctx: BuildContext) -> dict[str, Any]:
    import json

    path = ctx.synth_dir / BUILD_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_stage(
    ctx: BuildContext, stage: str, result: StageResult, records: Sequence[SourceRecord]
) -> None:
    """Remember which keys the stages before `stage` had when it ran (see module docstring)."""
    try:
        data = _read_build_file(ctx)
        stages = data.setdefault("stages", {})
        keys = _stage_keys(ctx, records)
        stages[stage] = {
            "finished_at": _now_iso(),
            "cached": result.cached,
            "key": keys.get(stage),
            "upstream": {s: keys[s] for s in STAGES[: STAGES.index(stage)]},
        }
        write_json_atomic(ctx.synth_dir / BUILD_FILE, data)
    except OSError:
        pass  # bookkeeping only: status degrades to «готово»


# ---------------------------------------------------------------- stages


@dataclass
class _State:
    records: list[SourceRecord] = field(default_factory=list)
    blocks: dict[str, Block] = field(default_factory=dict)
    outline: Outline | None = None
    coverage: CoverageReport | None = None
    sections_rewritten: list[str] | None = None  # ids of the sections S2 wrote (None: unknown)


def _stage_extract(ctx: BuildContext, force: bool, state: _State) -> StageResult:
    from h0lon.extract.pipeline import extract_topic
    from h0lon.sources.ingest import list_sources

    ref = topic_ref(ctx.settings, ctx.topic_dir)
    stage = StageResult(stage="extract", ok=False)
    if not list_sources(ctx.topic_dir):
        stage.errors.append(f"В теме нет источников: добавьте их командой h0lon add {ref} <файлы>")
        return stage
    results = extract_topic(ctx.settings, ctx.topic_dir, force=force, on_event=ctx.on_event)
    for r in results:
        stage.agent_runs += r.agent_runs
        stage.warnings += [
            f"{r.source_id}: {w.removeprefix(r.source_id + ': ')}" for w in r.warnings
        ]
        stage.errors += [f"{r.source_id}: {e.removeprefix(r.source_id + ': ')}" for e in r.errors]
        if not r.ok and not r.errors:
            stage.errors.append(f"{r.source_id}: извлечение не удалось")
    stage.cached = bool(results) and all(
        r.cached or (r.ok and r.source_md is None) for r in results
    )
    state.records = list_sources(ctx.topic_dir)
    extracted = _extracted(state.records)
    stage.details = {
        "sources": len(state.records),
        "extracted": len(extracted),
        "skipped": [r.id for r in state.records if r.status == "skipped"],
    }
    if stage.errors:
        stage.errors.append(f"Исправьте ошибки и повторите h0lon extract {ref}")
        return stage
    if not extracted:
        stage.errors.append(
            "Ни один источник не извлечён: все источники пропущены (видео и аудио пока не "
            f"поддерживаются). Добавьте текстовые источники: h0lon add {ref} <файлы>"
        )
        return stage
    state.blocks = load_blocks(ctx.topic_dir, [r.id for r in extracted])
    if not state.blocks:
        stage.errors.append("В извлечённых источниках нет блоков (пустые blocks.jsonl)")
        return stage
    stage.ok = True
    return stage


def _stage_review(ctx: BuildContext, state: _State) -> StageResult:
    stage = StageResult(stage="review", ok=True)
    info = review_state(ctx.settings, ctx.topic_dir, state.records)
    stage.details = dict(info)
    if not info["gate"]:
        stage.details["note"] = "review gate выключен"
        return stage
    if not ctx.review:
        stage.details["note"] = "пропущен (--no-review)"
        stage.warnings.append("Review gate пропущен (--no-review): извлечение не проверено")
        return stage
    if info["valid"]:
        return stage
    stage.ok = False
    stage.errors.append(_review_message(ctx.settings, ctx.topic_dir, info, state.records))
    return stage


def _stage_outline(ctx: BuildContext, state: _State) -> StageResult:
    run = _stage_fn("outline", "run_outline")
    result = run(ctx, state.blocks, _extracted(state.records))
    if result.ok:
        state.outline = _stage_fn("outline", "load_outline")(ctx)
    return result


def _group_records(ctx: BuildContext) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    """(sections, agent bundles) of every S2 group record in sections.meta.json."""
    meta = read_meta(ctx, "sections") or {}
    return {
        (tuple(rec.get("sections") or ()), tuple(rec.get("bundles") or ()))
        for rec in (meta.get("groups") or {}).values()
        if isinstance(rec, dict)
    }


def _stage_sections(ctx: BuildContext, state: _State) -> StageResult:
    assert state.outline is not None
    before = _group_records(ctx)
    result = _stage_fn("sections", "run_sections")(ctx, state.outline, state.blocks)
    # A group written in this run has agent bundles that no earlier record has; the groups taken
    # from the cache keep their records unchanged.
    order = {s.id: i for i, s in enumerate(state.outline.sections)}
    written = {sid for sections, _ in _group_records(ctx) - before for sid in sections}
    state.sections_rewritten = sorted(written, key=lambda sid: order.get(sid, len(order)))
    return result


def _stage_global(ctx: BuildContext, state: _State) -> StageResult:
    assert state.outline is not None
    return _stage_fn("globalpass", "run_global")(ctx, state.outline)


def _stage_coverage(ctx: BuildContext, state: _State) -> StageResult:
    assert state.outline is not None
    result, report = _stage_fn("coverage", "run_coverage")(ctx, state.outline, state.blocks)
    state.coverage = report
    return result


def _stage_assemble(ctx: BuildContext, state: _State) -> StageResult:
    from h0lon.synth.assemble import assemble_key, assemble_master

    assert state.outline is not None
    sources = _extracted(state.records)
    stage = StageResult(stage="assemble", ok=False)
    if state.coverage is None:  # the coverage stage did not run in this process
        state.coverage = _coverage_from_disk(ctx, state)
    target = ctx.topic_dir / MASTER_MD
    key = assemble_key(ctx, sources)
    meta = read_meta(ctx, "assemble")
    if not ctx.force and meta and meta.get("ok") and meta.get("key") == key and target.is_file():
        stage.ok = stage.cached = True
        stage.details = {"master_md": str(target)}
        return stage
    t0 = time.monotonic()
    path = assemble_master(ctx, state.outline, state.coverage, sources, warnings=stage.warnings)
    stage.ok = True
    stage.duration_s = round(time.monotonic() - t0, 3)
    stage.details = {"master_md": str(path), "chars": len(path.read_text(encoding="utf-8"))}
    write_meta(
        ctx,
        "assemble",
        {
            "key": key,
            "prompt": None,
            "model": None,
            "duration_s": stage.duration_s,
            "agent_runs": 0,
            "ok": True,
        },
    )
    return stage


def _coverage_from_disk(ctx: BuildContext, state: _State) -> CoverageReport:
    """CoverageReport from coverage.json (written by the coverage stage), or recomputed."""
    import json

    path = ctx.synth_dir / COVERAGE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CoverageReport(
            total=int(data["total"]),
            covered=int(data["covered"]),
            by_source={k: dict(v) for k, v in data.get("by_source", {}).items()},
            uncovered=list(data.get("uncovered", [])),
            rounds=int(data.get("rounds", 0)),
        )
    except (OSError, ValueError, KeyError, TypeError):
        compute = _stage_fn("coverage", "compute_coverage")
        return compute(state.blocks, ctx.synth_dir.joinpath(*FINAL_SECTIONS))


def _stage_render(ctx: BuildContext, state: _State) -> StageResult:
    return _stage_fn("render", "render_master")(ctx, ctx.topic_dir / MASTER_MD)


# ---------------------------------------------------------------- git


def _git_commit(ctx: BuildContext, message: str) -> StageResult:
    stage = StageResult(stage="git", ok=True)
    topic = ctx.topic_dir
    if not ctx.settings.general.git_per_topic:
        stage.details = {"committed": False, "note": "general.git_per_topic выключен"}
        return stage
    if not (topic / ".git").exists():
        stage.details = {"committed": False, "note": "тема не является git-репозиторием"}
        return stage
    git = tools.find_simple("git")
    if git is None:
        stage.warnings.append("git не найден: сборка не закоммичена в репозиторий темы.")
        stage.details = {"committed": False}
        return stage
    env = procutil.clean_env(drop_names=_GIT_ENV_DROP)

    def run(*args: str) -> procutil.ProcResult:
        return procutil.run([str(git), *args], cwd=topic, env=env, timeout=_GIT_TIMEOUT_S)

    def first_line(res: procutil.ProcResult) -> str:
        text = res.stderr or res.stdout or res.error or f"код {res.exit_code}"
        return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")

    _ensure_gitignore(topic)
    res = run("add", "-A", "--", *_GIT_PATHSPEC)
    if not res.ok:
        stage.warnings.append(f"git add не удался ({first_line(res)}): сборка не закоммичена.")
        stage.details = {"committed": False}
        return stage
    if run("diff", "--cached", "--quiet").exit_code == 0:
        stage.details = {"committed": False, "note": "изменений нет"}
        return stage
    res = run("commit", "-q", "-m", message)
    if not res.ok:
        text = (res.stderr + res.stdout).lower()
        if "user.email" in text or "identity" in text or "who you are" in text:
            stage.warnings.append(
                "Сборка не закоммичена: не настроена git-идентичность. Выполните в каталоге "
                'темы git config user.name "Имя" и git config user.email "почта", затем '
                f'git commit -m "{message}".'
            )
        else:
            stage.warnings.append(f"git commit не удался ({first_line(res)}).")
        stage.details = {"committed": False}
        return stage
    head = run("rev-parse", "--short", "HEAD")
    stage.details = {"committed": True, "commit": head.stdout.strip(), "message": message}
    return stage


# ---------------------------------------------------------------- changelog


def _sources_snapshot(records: Sequence[SourceRecord]) -> dict[str, dict[str, str]]:
    """The extracted sources as build.json remembers them (to tell what the next build adds)."""
    return {
        r.id: {"extracted_key": r.extracted_key or "", "title": r.title, "kind": r.kind}
        for r in _extracted(records)
    }


def _source_label(sid: str, info: dict[str, str]) -> str:
    from h0lon.sources.ingest import KIND_LABELS

    kind = info.get("kind", "")
    title = " ".join(info.get("title", "").split())
    return f"{sid} «{title}» ({KIND_LABELS.get(kind, kind)})" if title else sid


def _section_list(ids: Sequence[str], titles: dict[str, str]) -> str:
    shown = [f"`{i}` «{titles[i]}»" if titles.get(i) else f"`{i}`" for i in ids]
    text = ", ".join(shown[:MAX_CHANGELOG_SECTIONS])
    if len(shown) > MAX_CHANGELOG_SECTIONS:
        text += f" … (и ещё {len(shown) - MAX_CHANGELOG_SECTIONS})"
    return text


def _changelog_entry(
    state: _State,
    result: BuildResult,
    previous: dict[str, Any] | None,
    had_master: bool,
    when: datetime,
) -> str | None:
    """One entry of changelog.md, or None when this build changed nothing."""
    current = _sources_snapshot(state.records)
    stages = {s.stage: s for s in result.stages}
    added: list[str] = []
    changed: list[str] = []
    removed: list[str] = []
    if previous is None:
        # No record of the earlier build: a first build adds everything; for a topic built before
        # the changelog existed the set of sources before this build is unknown.
        added = [] if had_master else list(current)
    else:
        added = [i for i in current if i not in previous]
        changed = [
            i
            for i in current
            if i in previous
            and (previous[i] or {}).get("extracted_key", "") != current[i]["extracted_key"]
        ]
        removed = [i for i in previous if i not in current]
    synthesis = [stages[s] for s in ("outline", "sections", "global", "coverage") if s in stages]
    if not (added or changed or removed) and all(s.cached for s in synthesis):
        return None

    titles = {s.id: s.title for s in state.outline.sections} if state.outline else {}
    lines = [f"## {when:%Y-%m-%d %H:%M} — сборка мастера", ""]
    parts = []
    if added:
        parts.append("добавлены: " + "; ".join(_source_label(i, current[i]) for i in added))
    if changed:
        parts.append("изменены: " + "; ".join(_source_label(i, current[i]) for i in changed))
    if removed:
        parts.append(
            "удалены: "
            + "; ".join(_source_label(i, (previous or {}).get(i) or {}) for i in removed)
        )
    if parts:
        lines.append("- **Источники:** " + "; ".join(parts) + ".")
    elif previous is None and had_master:
        lines.append(
            "- **Источники:** журнал ведётся с этой сборки, состав источников до неё неизвестен."
        )
    else:
        lines.append("- **Источники:** без изменений.")

    outline_stage = stages.get("outline")
    if outline_stage is not None:
        d = outline_stage.details or {}
        if outline_stage.cached:
            text = "без изменений (из кэша)"
        elif d.get("mode") == "update":
            text = (
                f"обновлена по новым блокам (+{d.get('new_blocks', '?')}): существующие разделы "
                "и назначения блоков сохранены"
            )
            if d.get("new_sections"):
                text += "; новые разделы: " + _section_list(d["new_sections"], titles)
        elif d.get("mode") == "full":
            text = "построена заново"
            if d.get("update_failed"):
                text += " (обновление существующей структуры не удалось)"
        else:
            text = "выполнена"
        lines.append(f"- **Структура (S1):** {text}.")

    sections_stage = stages.get("sections")
    if sections_stage is not None:
        d = sections_stage.details or {}
        if sections_stage.cached:
            text = "без изменений (из кэша)"
        elif state.sections_rewritten:
            text = f"переписаны разделы ({len(state.sections_rewritten)}): " + _section_list(
                state.sections_rewritten, titles
            )
            if d.get("groups") is not None and d.get("groups_cached") is not None:
                text += f"; групп {d['groups']}, из кэша {d['groups_cached']}"
        else:
            text = "выполнены"
        lines.append(f"- **Разделы (S2):** {text}.")

    global_stage = stages.get("global")
    if global_stage is not None:
        lines.append(
            "- **Глобальная правка (S3):** "
            + ("без изменений (из кэша)." if global_stage.cached else "выполнена.")
        )

    if state.coverage is not None:
        cov = state.coverage
        text = f"{percent(cov.covered, cov.total)} ({cov.covered} из {cov.total} блоков)"
        if cov.rounds:
            text += f", кругов дополнения: {cov.rounds}"
        lines.append(f"- **Покрытие:** {text}.")
    runs = sum(s.agent_runs for s in result.stages)
    lines.append(f"- **Прогонов агента:** {runs}.")
    return "\n".join(lines) + "\n"


def _write_changelog(
    ctx: BuildContext,
    state: _State,
    result: BuildResult,
    previous: dict[str, Any] | None,
    had_master: bool,
) -> list[str]:
    """Append the entry of this build to synthesis/changelog.md and remember the sources in
    build.json. Never raises: the changelog must not fail a finished build."""
    from h0lon.extract.blocks import atomic_write_text
    from h0lon.workspace import load_topic

    try:
        entry = _changelog_entry(state, result, previous, had_master, datetime.now())
        if entry is not None:
            path = ctx.synth_dir / CHANGELOG_FILE
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            else:
                text = (
                    "# Журнал изменений мастер-конспекта\n\n"
                    f"Тема: «{load_topic(ctx.topic_dir).title}». Запись добавляется после каждой "
                    "сборки, в которой что-то изменилось.\n"
                )
            atomic_write_text(path, text.rstrip("\n") + "\n\n" + entry)
        data = _read_build_file(ctx)
        data["sources"] = _sources_snapshot(state.records)
        write_json_atomic(ctx.synth_dir / BUILD_FILE, data)
    except Exception as exc:  # bookkeeping only
        return [f"Журнал изменений не записан: {type(exc).__name__}: {exc}"]
    return []


# ---------------------------------------------------------------- build


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1 с"
    return f"{seconds:.1f} с".replace(".", ",") if seconds < 10 else f"{seconds:.0f} с"


def _stage_summary(stage: StageResult) -> str:
    if stage.cached:
        return "из кэша"
    if stage.stage == "review" and not stage.ok:
        return "нужно одобрение"
    parts = ["готово" if stage.ok else "ошибка"]
    if stage.agent_runs:
        parts.append(f"прогонов агента: {stage.agent_runs}")
    if stage.duration_s:
        parts.append(_fmt_duration(stage.duration_s))
    return ", ".join(parts)


def build_topic(
    settings: Settings,
    topic_dir: Path,
    *,
    review: bool = True,
    from_stage: str | None = None,
    force: bool = False,
    backend: str | None = None,
    on_event: EventCallback | None = None,
) -> BuildResult:
    """All stages of the synthesis in order; see the module docstring and ARCHITECTURE."""
    started = time.monotonic()
    topic_dir = Path(topic_dir).resolve()
    if from_stage is not None and from_stage not in STAGES:
        raise ValueError(f"Неизвестная стадия «{from_stage}»: допустимо {', '.join(STAGES)}.")
    if backend not in (None, "claude", "codex"):
        raise ValueError(f"Неизвестный агент «{backend}»: допустимо claude | codex.")
    from_idx = STAGES.index(from_stage) if from_stage else None
    ctx = BuildContext(
        topic_dir=topic_dir,
        settings=settings,
        force=force,
        review=review,
        backend=backend,
        on_event=on_event,
    )
    result = BuildResult(ok=False, topic_dir=topic_dir)
    state = _State()
    previous = _read_build_file(ctx).get("sources")
    previous = previous if isinstance(previous, dict) else None
    had_master = (topic_dir / MASTER_MD).is_file()

    def forced(stage: str) -> bool:
        return from_idx is not None and STAGES.index(stage) >= from_idx

    def finish(stopped_at: str | None, message: str) -> BuildResult:
        result.stopped_at = stopped_at
        result.message = message
        result.coverage = state.coverage
        master = topic_dir / MASTER_MD
        if master.is_file() and any(s.stage == "assemble" and s.ok for s in result.stages):
            result.master_md = master  # the master exists even if the PDF could not be built
        result.duration_s = time.monotonic() - started
        return result

    def execute(name: str, title: str, run: Callable[[], StageResult]) -> StageResult:
        ctx.emit(f"{title}…")
        t0 = time.monotonic()
        try:
            stage = run()
        except Exception as exc:
            stage = StageResult(
                stage=name,
                ok=False,
                errors=[f"Внутренняя ошибка: {type(exc).__name__}: {exc}"],
            )
            try:
                log = ctx.synth_dir / f"{name}.error.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(traceback.format_exc(), encoding="utf-8")
                stage.errors.append(f"Подробности: {log.relative_to(topic_dir).as_posix()}")
            except OSError:
                pass
        if not stage.duration_s:
            stage.duration_s = round(time.monotonic() - t0, 3)
        result.stages.append(stage)
        ctx.emit(f"{title}: {_stage_summary(stage)}")
        for w in stage.warnings:
            ctx.emit(f"Предупреждение ({title}): {w}")
        if stage.ok and name in STAGES:
            _record_stage(ctx, name, stage, state.records)
        return stage

    def stop(stage: StageResult, title: str) -> BuildResult:
        reason = "; ".join(stage.errors[:3]) if stage.errors else "стадия не выполнена"
        return finish(stage.stage, f"Стадия «{title}» остановлена: {reason}")

    # extract
    ctx.force = forced("extract")
    stage = execute(
        "extract", STAGE_TITLES["extract"], lambda: _stage_extract(ctx, ctx.force, state)
    )
    if not stage.ok:
        return stop(stage, STAGE_TITLES["extract"])

    # review gate
    review_stage = execute("review", REVIEW_TITLE, lambda: _stage_review(ctx, state))
    if not review_stage.ok:
        return finish("review", review_stage.errors[0])

    steps: list[tuple[str, Callable[[], StageResult]]] = [
        ("outline", lambda: _stage_outline(ctx, state)),
        ("sections", lambda: _stage_sections(ctx, state)),
        ("global", lambda: _stage_global(ctx, state)),
        ("coverage", lambda: _stage_coverage(ctx, state)),
        ("assemble", lambda: _stage_assemble(ctx, state)),
        ("render", lambda: _stage_render(ctx, state)),
    ]
    for name, step in steps:
        ctx.force = force or forced(name)
        stage = execute(name, STAGE_TITLES[name], step)
        if not stage.ok:
            return stop(stage, STAGE_TITLES[name])

    result.master_md = topic_dir / MASTER_MD
    result.master_pdf = topic_dir / MASTER_PDF
    result.ok = True
    pct = percent(state.coverage.covered, state.coverage.total) if state.coverage else "—"
    commit_message = f"Сборка мастера: {date.today().isoformat()}, покрытие {pct}"

    def commit() -> StageResult:
        notes = _write_changelog(ctx, state, result, previous, had_master)
        stage = _git_commit(ctx, commit_message)
        stage.warnings = [*notes, *stage.warnings]
        return stage

    git_stage = execute("git", GIT_TITLE, commit)
    note = f" Покрытие блоков источников: {pct}."
    if git_stage.warnings:
        note += " Предупреждение git: " + git_stage.warnings[0]
    return finish(None, f"Готово: {MASTER_PDF}.{note}")


# ---------------------------------------------------------------- status


def _stage_state(
    ctx: BuildContext,
    stage: str,
    records: Sequence[SourceRecord],
    build_data: dict[str, Any],
    keys: dict[str, str | None],
) -> dict[str, Any]:
    """{state, duration_s, agent_runs, finished_at} of one stage from its meta and build.json."""
    if stage == "extract":
        relevant = [r for r in records if r.status != "skipped"]
        failed = [r.id for r in relevant if r.status == "failed"]
        if not relevant:
            return {"state": "не выполнялась"}
        if failed:
            return {"state": "ошибка", "sources": failed}
        if all(r.status == "extracted" for r in relevant):
            return {"state": "готово"}
        return {"state": "не выполнялась"}
    meta = read_meta(ctx, stage)
    info: dict[str, Any] = {"state": "не выполнялась"}
    if not meta:
        return info
    info["duration_s"] = meta.get("duration_s")
    info["agent_runs"] = meta.get("agent_runs")
    if not meta.get("ok", True):
        info["state"] = "ошибка"
        return info
    outputs_missing = (stage == "assemble" and not (ctx.topic_dir / MASTER_MD).is_file()) or (
        stage == "render" and not (ctx.topic_dir / MASTER_PDF).is_file()
    )
    snapshot = (build_data.get("stages") or {}).get(stage) or {}
    upstream = snapshot.get("upstream") or {}
    stale = outputs_missing or any(
        keys.get(s) is not None and upstream.get(s) is not None and keys.get(s) != upstream.get(s)
        for s in STAGES[: STAGES.index(stage)]
    )
    if stale:
        info["state"] = "устарело"
    else:
        info["state"] = "из кэша" if snapshot.get("cached") else "готово"
    info["finished_at"] = snapshot.get("finished_at")
    return info


def topic_status(settings: Settings, topic_dir: Path) -> dict[str, Any]:
    """Sources, review gate, stages, coverage and output files of a topic (JSON-friendly)."""
    from h0lon.sources.ingest import list_sources
    from h0lon.workspace import load_topic

    topic_dir = Path(topic_dir).resolve()
    meta = load_topic(topic_dir)
    records = list_sources(topic_dir)
    ctx = BuildContext(topic_dir=topic_dir, settings=settings)
    build_data = _read_build_file(ctx)
    keys = _stage_keys(ctx, records)
    stages = {s: _stage_state(ctx, s, records, build_data, keys) for s in STAGES}

    coverage = None
    cov_path = ctx.synth_dir / COVERAGE_FILE
    if cov_path.is_file():
        import json

        try:
            data = json.loads(cov_path.read_text(encoding="utf-8"))
            coverage = {
                "total": data.get("total"),
                "covered": data.get("covered"),
                "ratio": data.get("ratio"),
                # truly uncovered: no verdict or «missing»; admin/duplicate count as covered
                "uncovered": sum(
                    1
                    for u in data.get("uncovered") or []
                    if (u or {}).get("verdict") in (None, "missing", "unknown")
                ),
                "excluded": sum(
                    1
                    for u in data.get("uncovered") or []
                    if (u or {}).get("verdict") in ("admin", "duplicate")
                ),
                "rounds": data.get("rounds"),
            }
        except (OSError, ValueError):
            coverage = None
    master_md = topic_dir / MASTER_MD
    master_pdf = topic_dir / MASTER_PDF
    return {
        "topic": str(topic_dir),
        "title": meta.title,
        "course": meta.course,
        "sources": [
            {
                "id": r.id,
                "kind": r.kind,
                "title": r.title,
                "status": r.status,
                "units": dict(r.units or {}),
                "notes": len((r.quality or {}).get("notes") or []),
                "error": r.error,
            }
            for r in records
        ],
        "review": review_state(settings, topic_dir, records),
        "stages": stages,
        "coverage": coverage,
        "master_md": str(master_md) if master_md.is_file() else None,
        "master_pdf": str(master_pdf) if master_pdf.is_file() else None,
    }


# ---------------------------------------------------------------- printing

_STATE_STYLE = {
    "готово": "green",
    "из кэша": "cyan",
    "устарело": "yellow",
    "ошибка": "red",
    "не выполнялась": "dim",
}
_SOURCE_STATUS = {
    "added": ("добавлен", "yellow"),
    "extracting": ("извлекается", "yellow"),
    "extracted": ("извлечён", "green"),
    "failed": ("ошибка", "red"),
    "skipped": ("пропущен", "dim"),
}


def print_status(status: dict[str, Any], *, console: Console) -> None:
    from rich.markup import escape
    from rich.table import Table

    console.print(f"[bold]{escape(status['title'])}[/bold] — {escape(status['course'])}")
    console.print(f"[dim]{escape(status['topic'])}[/dim]")

    table = Table(title="Источники", title_justify="left", show_lines=False)
    for col in ("ID", "Вид", "Название", "Статус", "Замечаний"):
        table.add_column(col, overflow="fold")
    for s in status["sources"]:
        label, style = _SOURCE_STATUS.get(s["status"], (s["status"], ""))
        table.add_row(
            escape(s["id"]),
            escape(s["kind"]),
            escape(s["title"]),
            f"[{style}]{label}[/{style}]" if style else label,
            str(s["notes"] or "—"),
        )
    if status["sources"]:
        console.print(table)
    else:
        console.print("Источников нет: добавьте их командой h0lon add.")

    r = status["review"]
    if not r["gate"]:
        console.print("Review gate: [dim]выключен[/dim]")
    elif r["valid"]:
        console.print(
            f"Review gate: [green]извлечение одобрено[/green] ({escape(str(r['approved_at']))})"
        )
    elif r["approved"]:
        changed = ", ".join(r["changed"])
        console.print(
            f"Review gate: [yellow]одобрение устарело[/yellow] (изменилось: {escape(changed)})"
        )
    else:
        console.print("Review gate: [yellow]извлечение не одобрено[/yellow] — h0lon approve")

    stages = Table(title="Стадии", title_justify="left", show_lines=False)
    for col in ("Стадия", "Состояние", "Прогонов агента", "Время"):
        stages.add_column(col, overflow="fold")
    for name, info in status["stages"].items():
        state = info["state"]
        style = _STATE_STYLE.get(state, "")
        dur = info.get("duration_s")
        stages.add_row(
            STAGE_TITLES.get(name, name),
            f"[{style}]{state}[/{style}]" if style else state,
            str(info["agent_runs"]) if info.get("agent_runs") is not None else "—",
            _fmt_duration(dur) if isinstance(dur, int | float) else "—",
        )
    console.print(stages)

    cov = status["coverage"]
    if cov and cov.get("total") is not None:
        pct = percent(int(cov.get("covered") or 0), int(cov.get("total") or 0))
        console.print(
            f"Покрытие: [bold]{pct}[/bold] ({cov['covered']} из {cov['total']} блоков), "
            f"непокрытых: {cov['uncovered']}, служебных и дублей: {cov.get('excluded') or 0}, "
            f"кругов дополнения: {cov.get('rounds') or 0}"
        )
    else:
        console.print("Покрытие: [dim]ещё не считалось[/dim]")
    for label, key in (("master.md", "master_md"), ("master.pdf", "master_pdf")):
        value = status[key]
        console.print(f"{label}: " + (escape(value) if value else "[dim]нет[/dim]"))


def print_build(result: BuildResult, *, console: Console) -> None:
    from rich.markup import escape
    from rich.table import Table

    table = Table(title="Сборка мастера", title_justify="left", show_lines=False)
    for col in ("Стадия", "Итог", "Прогонов агента", "Время"):
        table.add_column(col, overflow="fold")
    for s in result.stages:
        title = {**STAGE_TITLES, "review": REVIEW_TITLE, "git": GIT_TITLE}.get(s.stage, s.stage)
        if s.cached:
            verdict = "[cyan]из кэша[/cyan]"
        elif s.ok:
            verdict = "[green]готово[/green]"
        elif s.stage == "review":
            verdict = "[yellow]нужно одобрение[/yellow]"
        else:
            verdict = "[red]ошибка[/red]"
        note = s.details.get("note") if isinstance(s.details, dict) else None
        table.add_row(
            title,
            verdict + (f" ({escape(str(note))})" if note else ""),
            str(s.agent_runs) if s.agent_runs else "—",
            _fmt_duration(s.duration_s) if s.duration_s else "—",
        )
    console.print(table)
    for s in result.stages:
        for w in s.warnings[:MAX_WARNINGS_PRINTED]:
            console.print(f"[yellow]{escape(s.stage)}:[/yellow] {escape(w)}")
        if len(s.warnings) > MAX_WARNINGS_PRINTED:
            rest = len(s.warnings) - MAX_WARNINGS_PRINTED
            console.print(f"[yellow]{escape(s.stage)}:[/yellow] … и ещё {rest} (все — в --json)")
        for e in s.errors:
            if e not in result.message:  # the stop message below already says it
                console.print(f"[red]{escape(s.stage)}:[/red] {escape(e)}")
    if result.ok:
        console.print(f"[bold green]{escape(result.message)}[/bold green]")
        if result.master_md:
            console.print(f"master.md: {escape(str(result.master_md))}")
        if result.master_pdf:
            console.print(f"master.pdf: {escape(str(result.master_pdf))}")
    elif result.stopped_at == "review":
        console.print(
            f"[bold yellow]Остановка на review gate.[/bold yellow] {escape(result.message)}"
        )
    else:
        console.print(f"[bold red]{escape(result.message)}[/bold red]")
    console.print(f"[dim]Время: {_fmt_duration(result.duration_s)}[/dim]")
