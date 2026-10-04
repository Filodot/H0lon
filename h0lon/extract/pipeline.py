"""Extraction of a topic's sources into Source Docs (docs/ARCHITECTURE.md, «Конвейер»).

For every source: extractor → body.md → blocks (source.md + blocks.jsonl) → summary.md,
with a cache keyed by the source file, the extractor version, the prompt versions and the
models (the light tier of the summary; the extractor's own tier too when it is not light —
handwriting is read by the strong tier). One failing source never stops the others;
statuses, quality signals and cache keys go to topic.yaml through `h0lon.sources.ingest`.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from h0lon import __version__, tools
from h0lon.extract import blocks as bl
from h0lon.extract import summary as sm
from h0lon.extract.model import (
    EventCallback,
    ExtractContext,
    Extractor,
    ExtractPlan,
    ExtractResult,
)
from h0lon.extract.registry import ExtractError, get_extractor, resolve_extractor

if TYPE_CHECKING:
    from h0lon.config import Settings
    from h0lon.sources.models import SourceRecord

__all__ = ["extract_topic", "get_extractor", "print_plans", "print_results"]

EXTRACTED_DIR = "extracted"
SOURCE_FILE = "source.md"
BLOCKS_FILE = "blocks.jsonl"
META_FILE = "meta.json"
ERROR_LOG = "error.log"
META_FORMAT = 1
VISION_PROMPT = "vision_pages"
# Kinds whose extractors may send pages to vision agents (their prompt is part of the key).
VISION_KINDS = ("pdf-text", "pdf-scan", "slides")
# Share of Cyrillic letters from which a source counts as Russian.
RU_THRESHOLD = 0.3

UpdateSource = Callable[[Path, "SourceRecord"], None]


# ---------------------------------------------------------------- public API


def extract_topic(
    settings: Settings,
    topic_dir: Path,
    *,
    source_ids: Sequence[str] | None = None,
    force: bool = False,
    use_vision: bool = True,
    dry_run: bool = False,
    backend: str | None = None,
    on_event: EventCallback | None = None,
) -> list[ExtractResult] | list[ExtractPlan]:
    """Extract the topic's sources (all or `source_ids`); `dry_run` returns plans only."""
    from h0lon.sources.ingest import list_sources, update_source

    topic_dir = Path(topic_dir).resolve()
    records = list_sources(topic_dir)
    by_id = {r.id: r for r in records}
    wanted = list(dict.fromkeys(source_ids)) if source_ids else [r.id for r in records]
    unknown = [sid for sid in wanted if sid not in by_id]

    def emit(message: str) -> None:
        _safe_emit(on_event, message)

    if dry_run:
        plans: list[ExtractPlan] = []
        for sid in wanted:
            if sid in by_id:
                plans.append(
                    _plan_one(settings, topic_dir, by_id[sid], force, use_vision, backend, on_event)
                )
            else:
                plans.append(ExtractPlan(source_id=sid, notes=[_unknown_message(sid, records)]))
        return plans

    results: list[ExtractResult] = []
    for sid in wanted:
        if sid in by_id:
            results.append(
                _extract_one(
                    settings,
                    topic_dir,
                    by_id[sid],
                    force=force,
                    use_vision=use_vision,
                    backend=backend,
                    on_event=on_event,
                    update_source=update_source,
                )
            )
        else:
            results.append(
                ExtractResult(ok=False, source_id=sid, errors=[_unknown_message(sid, records)])
            )
    if not wanted:
        emit("В теме нет источников: добавьте их командой h0lon add")
    elif unknown:
        emit("Не найдены источники: " + ", ".join(unknown))
    return results


def _unknown_message(sid: str, records: Sequence[SourceRecord]) -> str:
    known = ", ".join(r.id for r in records) or "нет"
    return f"Источник {sid} не найден в теме (есть: {known})"


# ---------------------------------------------------------------- helpers


def _safe_emit(on_event: EventCallback | None, message: str) -> None:
    if on_event is not None:
        try:
            on_event(message)
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_file(topic_dir: Path, rec: SourceRecord) -> Path | None:
    if not rec.file:
        return None
    path = Path(rec.file)
    return path if path.is_absolute() else topic_dir / path


def _extractor_label(extractor: Extractor) -> str:
    module = type(extractor).__module__.rsplit(".", 1)[-1]
    return f"{module}@{getattr(extractor, 'version', '?')}"


def _prompt_ids(kind: str, extractor: Extractor, use_vision: bool) -> list[str]:
    """Prompts whose text can change the result (part of the cache key)."""
    if not use_vision:
        return []
    names: list[str] = list(getattr(extractor, "prompts", ()) or ())
    if not names and kind in VISION_KINDS:
        names.append(VISION_PROMPT)
    ids = []
    for name in names:  # an extractor may declare "vision_pages@1.0" or just "vision_pages"
        if "@" in name:
            ids.append(name)
            continue
        try:
            ids.append(sm.prompt_id(name))
        except FileNotFoundError:
            ids.append(f"{name}@?")
    ids.append(sm.prompt_id(sm.PROMPT_NAME))
    return ids


def _vision_agent(
    settings: Settings, extractor: Extractor, backend: str | None
) -> tuple[str, str] | None:
    """(backend, model) of the extractor's own agent tier; None for the light tier.

    The light tier is the one the summary uses (`sm.light_model`); an extractor that reads
    pages with the strong tier declares `tier = "strong"` and its bundle `stage`.
    """
    tier = getattr(extractor, "tier", "light")
    if tier == "light":
        return None
    from h0lon.extract import vision

    stage = getattr(extractor, "stage", vision.STAGE)
    return vision.agent_model(settings, backend, tier=tier, stage=stage)


def cache_key(
    settings: Settings,
    rec: SourceRecord,
    extractor: Extractor,
    *,
    file_sha256: str | None,
    use_vision: bool,
    backend: str | None,
) -> tuple[str, dict[str, Any]]:
    """(key, its parts): file hash + extractor version + prompt versions + models."""
    agent, model = sm.light_model(settings, backend) if use_vision else (None, None)
    parts: dict[str, Any] = {
        "sha256": file_sha256 or rec.sha256 or rec.url,
        "extractor": _extractor_label(extractor),
        "prompts": _prompt_ids(rec.kind, extractor, use_vision),
        "backend": agent,
        "model": model,
        "vision": use_vision,  # a --no-vision result must not satisfy a full run
    }
    own = _vision_agent(settings, extractor, backend) if use_vision else None
    if own is not None:  # light-tier extractors keep their keys
        parts["vision_backend"], parts["vision_model"] = own
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:32], parts


def read_meta(out_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _outputs_exist(out_dir: Path) -> bool:
    return all((out_dir / name).is_file() for name in (SOURCE_FILE, BLOCKS_FILE, sm.SUMMARY_FILE))


def _language(ratio: float | None, settings: Settings) -> str:
    if ratio is None:
        return settings.general.language
    return "ru" if ratio >= RU_THRESHOLD else "en"


def _ingest_quality(rec: SourceRecord, meta: dict[str, Any]) -> dict[str, Any]:
    """Quality signals written by ingest, without those of an earlier extraction.

    topic.yaml keeps the merged signals, so the ingest part is remembered in meta.json.
    For an extraction recorded before that (no `ingest_quality`), the notes that were its
    warnings are dropped; keys need nothing, the new extractor output overrides them.
    """
    stored = meta.get("ingest_quality")
    if isinstance(stored, dict) and meta.get("ingest_sha256") == rec.sha256:
        return dict(stored)
    current = dict(rec.quality or {})
    old_warnings = meta.get("warnings") if rec.extracted_key else None
    if not isinstance(old_warnings, list) or not isinstance(current.get("notes"), list):
        return current
    notes = [n for n in current["notes"] if n not in old_warnings]
    if notes:
        current["notes"] = notes
    else:
        current.pop("notes")
    return current


def _merge_quality(*parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    notes: list[str] = []
    for part in parts:
        for key, value in (part or {}).items():
            if key == "notes":
                for note in value if isinstance(value, list) else [value]:
                    if note and note not in notes:
                        notes.append(str(note))
            else:
                out[key] = value
    if notes:
        out["notes"] = notes
    return out


def _update(
    update_source: UpdateSource, topic_dir: Path, rec: SourceRecord, **changes: Any
) -> str | None:
    """Write the changed record to topic.yaml; a failure becomes a warning."""
    try:
        update_source(topic_dir, rec.model_copy(update=changes))
    except Exception as exc:
        return f"Не удалось обновить topic.yaml: {type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------- plans


def _plan_one(
    settings: Settings,
    topic_dir: Path,
    rec: SourceRecord,
    force: bool,
    use_vision: bool,
    backend: str | None,
    on_event: EventCallback | None,
) -> ExtractPlan:
    res = resolve_extractor(rec.kind)
    if res.extractor is None:
        return ExtractPlan(source_id=rec.id, notes=[res.reason or "нет экстрактора"])
    extractor = res.extractor
    out_dir = topic_dir / EXTRACTED_DIR / rec.id
    ctx = ExtractContext(
        topic_dir=topic_dir,
        source=rec,
        settings=settings,
        out_dir=out_dir,
        force=force,
        use_vision=use_vision,
        backend=backend,
        on_event=on_event,
    )
    try:
        plan = extractor.plan(ctx)
    except Exception as exc:
        plan = ExtractPlan(
            source_id=rec.id, notes=[f"план не построен: {type(exc).__name__}: {exc}"]
        )
    plan.source_id = rec.id
    src = _source_file(topic_dir, rec)
    sha = _file_sha256(src) if src is not None and src.is_file() else None
    key, _parts = cache_key(
        settings, rec, extractor, file_sha256=sha, use_vision=use_vision, backend=backend
    )
    meta = read_meta(out_dir)
    if not force and meta.get("key") == key and _outputs_exist(out_dir):
        if (meta.get("summary") or {}).get("mode") == "fallback":
            plan.pages_vision, plan.agent_runs = 0, 1
            plan.notes.append("кэш актуален; аннотация будет повторена агентом (1 прогон)")
        else:
            plan.pages_vision, plan.agent_runs = 0, 0
            plan.notes.append("кэш актуален — источник будет пропущен")
        return plan
    if use_vision:
        agent, model = sm.light_model(settings, backend)
        plan.agent_runs += 1
        plan.notes.append(f"аннотация: 1 прогон агента {agent} ({model}, лёгкий уровень)")
    else:
        plan.notes.append("аннотация без агента: оглавление и термины по заголовкам")
    return plan


# ---------------------------------------------------------------- extraction


def _extract_one(
    settings: Settings,
    topic_dir: Path,
    rec: SourceRecord,
    *,
    force: bool,
    use_vision: bool,
    backend: str | None,
    on_event: EventCallback | None,
    update_source: UpdateSource,
) -> ExtractResult:
    t0 = time.monotonic()
    sid = rec.id

    def emit(message: str) -> None:
        _safe_emit(on_event, f"{sid}: {message}")

    def done(result: ExtractResult) -> ExtractResult:
        result.duration_s = round(time.monotonic() - t0, 3)
        return result

    res = resolve_extractor(rec.kind)
    if res.skipped:
        reason = res.reason or "источник пропущен"
        emit(reason)
        warnings = [reason]
        if rec.status != "skipped" or rec.error != reason:
            problem = _update(update_source, topic_dir, rec, status="skipped", error=reason)
            warnings += [problem] if problem else []
        return done(ExtractResult(ok=True, source_id=sid, warnings=warnings))
    if res.extractor is None:
        reason = res.reason or f"Вид «{rec.kind}» не поддерживается"
        emit(reason)
        errors = [reason]
        problem = _update(update_source, topic_dir, rec, status="failed", error=reason)
        return done(
            ExtractResult(
                ok=False, source_id=sid, errors=errors, warnings=[problem] if problem else []
            )
        )
    extractor = res.extractor

    out_dir = topic_dir / EXTRACTED_DIR / sid
    ctx = ExtractContext(
        topic_dir=topic_dir,
        source=rec,
        settings=settings,
        out_dir=out_dir,
        force=force,
        use_vision=use_vision,
        backend=backend,
        on_event=on_event,
    )
    warnings: list[str] = []
    try:
        src = _source_file(topic_dir, rec)
        sha = _file_sha256(src) if src is not None and src.is_file() else None
        key, key_parts = cache_key(
            settings, rec, extractor, file_sha256=sha, use_vision=use_vision, backend=backend
        )
        meta = read_meta(out_dir)
        if sha and rec.sha256 and sha != rec.sha256:
            warnings.append(
                f"Файл {rec.file} изменился после добавления (sha256 не совпадает с "
                "topic.yaml) — извлекается текущая версия"
            )
        if not force and meta.get("key") == key and _outputs_exist(out_dir):
            if (meta.get("summary") or {}).get("mode") != "fallback":
                return done(_cached_result(rec, meta, out_dir, key, update_source, topic_dir, emit))
            return done(_summary_again(ctx, meta, out_dir, key, update_source, warnings, emit))
        return done(
            _run_extraction(
                ctx,
                extractor,
                key,
                key_parts,
                sha,
                update_source,
                warnings,
                emit,
                previous=meta,
                t0=t0,
            )
        )
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, ExtractError | bl.BlocksError)
            else (f"Внутренняя ошибка извлечения: {type(exc).__name__}: {exc}")
        )
        emit(f"ошибка — {message}")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            bl.atomic_write_text(out_dir / ERROR_LOG, traceback.format_exc())
        except OSError:
            pass
        problem = _update(update_source, topic_dir, rec, status="failed", error=message)
        if problem:
            warnings.append(problem)
        return done(ExtractResult(ok=False, source_id=sid, errors=[message], warnings=warnings))


def _cached_result(
    rec: SourceRecord,
    meta: dict[str, Any],
    out_dir: Path,
    key: str,
    update_source: UpdateSource,
    topic_dir: Path,
    emit: Callable[[str], None],
) -> ExtractResult:
    emit("кэш актуален — пропуск")
    warnings: list[str] = []
    quality = meta.get("quality") or rec.quality
    if rec.status != "extracted" or rec.extracted_key != key or rec.error:
        problem = _update(
            update_source,
            topic_dir,
            rec,
            status="extracted",
            extracted_key=key,
            error=None,
            quality=quality,
            units=meta.get("units") or rec.units,
        )
        if problem:
            warnings.append(problem)
    return ExtractResult(
        ok=True,
        source_id=rec.id,
        source_md=out_dir / SOURCE_FILE,
        blocks=int(meta.get("blocks") or 0),
        pages_total=int(meta.get("pages_total") or 0),
        pages_vision=int(meta.get("pages_vision") or 0),
        agent_runs=0,
        cached=True,
        quality=quality,
        warnings=warnings,
    )


def _summary_again(
    ctx: ExtractContext,
    meta: dict[str, Any],
    out_dir: Path,
    key: str,
    update_source: UpdateSource,
    warnings: list[str],
    emit: Callable[[str], None],
) -> ExtractResult:
    """The extraction is cached, but last time the summary agent failed: retry only it."""
    rec = ctx.source
    emit("кэш актуален, но аннотация была построена без агента — повтор аннотации")
    source_md = out_dir / SOURCE_FILE
    doc = bl.build_source_doc(
        source_md.read_text(encoding="utf-8"), rec.id, pandoc=_pandoc(ctx.settings)
    )
    summary = sm.summarize(ctx, doc, source_md, use_agent=ctx.use_vision)
    warnings += summary.warnings
    meta = dict(meta)
    meta["summary"] = summary.to_dict()
    meta["agent_runs"] = int(meta.get("agent_runs") or 0) + summary.agent_runs
    meta["summary_retried_at"] = _now_iso()
    bl.atomic_write_text(out_dir / META_FILE, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    quality = meta.get("quality") or rec.quality
    problem = _update(
        update_source,
        ctx.topic_dir,
        rec,
        status="extracted",
        extracted_key=key,
        error=None,
        quality=quality,
        units=meta.get("units") or rec.units,
    )
    if problem:
        warnings.append(problem)
    return ExtractResult(
        ok=True,
        source_id=rec.id,
        source_md=source_md,
        blocks=int(meta.get("blocks") or len(doc.blocks)),
        pages_total=int(meta.get("pages_total") or 0),
        pages_vision=0,
        agent_runs=summary.agent_runs,
        cached=False,
        quality=quality,
        warnings=warnings,
    )


def _pandoc(settings: Settings) -> Path:
    pandoc = tools.find_pandoc(settings.render.pandoc)
    if pandoc is None:
        raise ExtractError(
            "Pandoc не найден: установите пакет pypandoc-binary (uv sync) или укажите "
            "render.pandoc в h0lon.toml"
        )
    return pandoc


def _run_extraction(
    ctx: ExtractContext,
    extractor: Extractor,
    key: str,
    key_parts: dict[str, Any],
    sha: str | None,
    update_source: UpdateSource,
    warnings: list[str],
    emit: Callable[[str], None],
    *,
    previous: dict[str, Any],
    t0: float,
) -> ExtractResult:
    rec, settings, out_dir = ctx.source, ctx.settings, ctx.out_dir
    ingest_quality = _ingest_quality(rec, previous)
    started_at = _now_iso()
    problem = _update(update_source, ctx.topic_dir, rec, status="extracting", error=None)
    if problem:
        warnings.append(problem)
    pandoc = _pandoc(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ERROR_LOG).unlink(missing_ok=True)
    emit(f"извлечение ({rec.kind}, {_extractor_label(extractor)})")

    output = extractor.extract(ctx)
    warnings += output.warnings
    body_path = Path(output.body_md)
    if not body_path.is_file():
        raise ExtractError(f"Экстрактор не создал {body_path}")
    doc = bl.build_source_doc(body_path.read_text(encoding="utf-8"), rec.id, pandoc=pandoc)
    warnings += doc.warnings
    if not doc.blocks:
        raise ExtractError("После извлечения не осталось ни одного блока содержания")
    emit(f"блоков: {len(doc.blocks)} (якоря: {'места' if doc.anchor_mode == 'location' else '§'})")

    ratio = bl.cyrillic_ratio(b.text for b in doc.blocks)
    quality = _merge_quality(ingest_quality, output.quality, {"cyrillic_ratio": ratio})
    units: dict[str, int | float] = dict(rec.units)
    if output.pages_total and not units:
        units = {("slides" if rec.kind == "slides" else "pages"): output.pages_total}
    vision_used = output.agent_runs > 0
    agent, model = (None, None)
    if vision_used:
        agent, model = _vision_agent(settings, extractor, ctx.backend) or sm.light_model(
            settings, ctx.backend
        )
    vision_prompts = [p for p in key_parts["prompts"] if not p.startswith(sm.PROMPT_NAME + "@")]
    front: dict[str, Any] = {
        "id": rec.id,
        "kind": rec.kind,
        "title": rec.title,
        "origin": rec.file or rec.url,
        "original_name": rec.original_name,
    }
    if rec.url:
        front["url"] = rec.url
    front.update(
        {
            "sha256": sha or rec.sha256,
            "units": units,
            "quality": quality,
            "extracted_by": {
                "extractor": _extractor_label(extractor),
                "backend": agent,
                "model": model,
                "prompts": vision_prompts if vision_used else [],
                "date": date.today().isoformat(),
                "h0lon": __version__,
            },
            "language": _language(ratio, settings),
        }
    )
    source_md = out_dir / SOURCE_FILE
    bl.atomic_write_text(source_md, bl.compose_source_md(front, doc.body))
    bl.atomic_write_text(out_dir / BLOCKS_FILE, bl.blocks_jsonl(doc.blocks))

    summary = sm.summarize(ctx, doc, source_md, use_agent=ctx.use_vision)
    warnings += summary.warnings
    agent_runs = output.agent_runs + summary.agent_runs
    meta = {
        "format": META_FORMAT,
        "source_id": rec.id,
        "kind": rec.kind,
        "key": key,
        "key_parts": key_parts,
        "extractor": _extractor_label(extractor),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "duration_s": round(time.monotonic() - t0, 3),
        "agent_runs": agent_runs,
        "extract_agent_runs": output.agent_runs,
        "blocks": len(doc.blocks),
        "pages_total": output.pages_total,
        "pages_vision": output.pages_vision,
        "anchor_mode": doc.anchor_mode,
        "summary": summary.to_dict(),
        "quality": quality,
        "ingest_quality": ingest_quality,
        "ingest_sha256": rec.sha256,
        "units": units,
        "warnings": warnings,
        "h0lon": __version__,
    }
    bl.atomic_write_text(out_dir / META_FILE, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    problem = _update(
        update_source,
        ctx.topic_dir,
        rec,
        status="extracted",
        quality=quality,
        units=units,
        extracted_key=key,
        error=None,
    )
    if problem:
        warnings.append(problem)
    how = {"agent": "агентом", "deterministic": "без агента", "fallback": "без агента (сбой)"}
    emit(f"готово: блоков {len(doc.blocks)}, аннотация {how.get(summary.mode, summary.mode)}")
    return ExtractResult(
        ok=True,
        source_id=rec.id,
        source_md=source_md,
        blocks=len(doc.blocks),
        pages_total=output.pages_total,
        pages_vision=output.pages_vision,
        agent_runs=agent_runs,
        cached=False,
        quality=quality,
        warnings=warnings,
    )


# ---------------------------------------------------------------- printing


def _status(r: ExtractResult) -> str:
    if not r.ok:
        return "[red]ошибка[/red]"
    if r.cached:
        return "[cyan]из кэша[/cyan]"
    if r.source_md is None:
        return "[yellow]пропущен[/yellow]"
    return "[green]готово[/green]"


def print_results(results: Sequence[ExtractResult], *, console: Any) -> None:
    from rich.markup import escape
    from rich.table import Table

    if not results:
        console.print("Источников для извлечения нет.")
        return
    table = Table(title="Извлечение источников", show_lines=False)
    for col in (
        "Источник",
        "Статус",
        "Блоков",
        "Страниц (агент/всего)",
        "Прогонов агента",
        "Время",
    ):
        table.add_column(col, overflow="fold")
    for r in results:
        pages = f"{r.pages_vision}/{r.pages_total}" if r.pages_total else "—"
        table.add_row(
            escape(r.source_id),
            _status(r),
            str(r.blocks) if r.ok and r.source_md else "—",
            pages,
            str(r.agent_runs),
            f"{r.duration_s:.1f} с",
        )
    console.print(table)
    for r in results:
        prefix = f"{r.source_id}: "
        for w in r.warnings:
            w = w.removeprefix(prefix)  # extractors may prefix their messages themselves
            console.print(f"[yellow]{escape(r.source_id)}:[/yellow] {escape(w)}")
        for e in r.errors:
            e = e.removeprefix(prefix)
            console.print(f"[red]{escape(r.source_id)}:[/red] {escape(e)}")
    counts = {
        "готово": sum(1 for r in results if r.ok and not r.cached and r.source_md),
        "из кэша": sum(1 for r in results if r.cached),
        "пропущено": sum(1 for r in results if r.ok and not r.cached and r.source_md is None),
        "с ошибкой": sum(1 for r in results if not r.ok),
    }
    console.print("Итого: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    files = [r.source_md for r in results if r.ok and r.source_md]
    if files:
        console.print(
            f"Source Docs: [bold]{escape(str(files[0].parent.parent))}[/bold] "
            "(source.md, blocks.jsonl, summary.md)"
        )


def print_plans(plans: Sequence[ExtractPlan], *, console: Any) -> None:
    from rich.markup import escape
    from rich.table import Table

    if not plans:
        console.print("Источников для извлечения нет.")
        return
    table = Table(title="План извлечения (--dry-run)", show_lines=False)
    for col in ("Источник", "Страниц", "Через агента", "Прогонов агента", "Примечания"):
        table.add_column(col, overflow="fold")
    for p in plans:
        table.add_row(
            escape(p.source_id),
            str(p.pages_total) if p.pages_total else "—",
            str(p.pages_vision),
            str(p.agent_runs),
            escape("; ".join(p.notes)),
        )
    console.print(table)
    console.print(
        f"Всего прогонов агента: {sum(p.agent_runs for p in plans)}, "
        f"страниц через агента: {sum(p.pages_vision for p in plans)}"
    )
