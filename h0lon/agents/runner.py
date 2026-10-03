"""Agent runner: attempts, validation retries with feedback, fallback, cooling, run.json, usage.

Policy (docs/ARCHITECTURE.md, «Агенты»):
- auth / not_found / refusal → the next backend at once (auth also marks the backend cooling);
- rate_limit → the next backend, the failed one is cooling until the process ends;
- validation → up to `agents.max_validation_retries` retries with feedback, then the next backend;
- timeout / crash / unknown → one retry, then the next backend.

A crash/unknown outcome whose files and final message nevertheless pass validation counts as a
success (the contract is about the output); the error text is kept in the attempt record.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from h0lon.agents.base import (
    HARD_ERRORS,
    AgentBackend,
    AttemptRecord,
    BackendOutcome,
    Tier,
    Usage,
    format_limits,
    safe_emit,
    short,
)
from h0lon.agents.bundle import (
    Feedback,
    TaskBundle,
    build_prompt,
    system_prompt_version,
    write_text,
)
from h0lon.agents.usage import append_usage
from h0lon.agents.validate import check_contract, validate_final_message, validate_outputs
from h0lon.config import Settings

_KIND_RU = {
    "auth": "не выполнен вход",
    "rate_limit": "лимит подписки",
    "timeout": "таймаут",
    "crash": "сбой CLI",
    "validation": "результат не прошёл проверку",
    "not_found": "CLI не найден",
    "refusal": "отказ модели",
    "unknown": "ошибка",
}

# ---------------------------------------------------------------- backends and cooling

_cooling: dict[str, str] = {}
_cooling_lock = threading.Lock()


def mark_cooling(name: str, reason: str) -> None:
    with _cooling_lock:
        _cooling[name] = reason


def cooling_reason(name: str) -> str | None:
    with _cooling_lock:
        return _cooling.get(name)


def cooling_backends() -> dict[str, str]:
    with _cooling_lock:
        return dict(_cooling)


def reset_cooling(name: str | None = None) -> None:
    """Forget cooling marks (all or one), e.g. after the user logged in again."""
    with _cooling_lock:
        if name is None:
            _cooling.clear()
        else:
            _cooling.pop(name, None)


def get_backend(name: str, settings: Settings) -> AgentBackend:
    if name == "claude":
        from h0lon.agents.claude import ClaudeBackend

        return ClaudeBackend(settings)
    if name == "codex":
        from h0lon.agents.codex import CodexBackend

        return CodexBackend(settings)
    raise ValueError(f"Неизвестный агент «{name}» (допустимо: claude, codex)")


def candidate_backends(
    settings: Settings, *, stage: str, backend: str | None, fallback: bool
) -> list[str]:
    """Primary backend for the stage (or the explicit one) plus at most one fallback."""
    agents = settings.agents
    primary = backend or settings.stages.backend_for(stage, agents)
    order = [primary]
    if fallback and agents.fallback:
        for alt in (agents.fallback, agents.default):
            if alt and alt != primary:
                order.append(alt)
                break
    return order


# ---------------------------------------------------------------- result


@dataclass
class RunResult:
    ok: bool
    bundle: TaskBundle
    backend_used: str | None
    attempts: list[AttemptRecord]
    usage_total: Usage
    final_text: str
    problems: list[str]
    started_at: str = ""
    duration_s: float = 0.0
    final_data: Any = None  # parsed final message when the contract has a schema
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle": self.bundle.to_dict(),
            "backend_used": self.backend_used,
            "attempts": [a.to_dict() for a in self.attempts],
            "usage_total": self.usage_total.to_dict(),
            "final_text": self.final_text,
            "final_data": self.final_data,
            "problems": list(self.problems),
            "skipped": list(self.skipped),
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "system_prompt_version": system_prompt_version(),
        }

    def print(self, console: Any) -> None:
        # Paths, problems and CLI errors may contain `[...]`: escape everything dynamic.
        from rich.markup import escape
        from rich.table import Table

        status = "[bold green]OK[/bold green]" if self.ok else "[bold red]ОШИБКА[/bold red]"
        console.print(f"Прогон агента: {status}")
        console.print(f"Бандл: [bold]{escape(str(self.bundle.root))}[/bold]")
        console.print(
            f"Этап: {escape(self.bundle.stage)} · агент: {escape(self.backend_used or '—')} · "
            f"попыток: {len(self.attempts)} · {self.duration_s:.1f} с"
        )
        if self.attempts:
            table = Table(show_lines=False)
            for col in ("№", "Агент", "Модель", "Время", "Статус", "Токены вх/кэш/вых", "Детали"):
                table.add_column(col, overflow="fold")
            for a in self.attempts:
                kind_ru = _KIND_RU.get(a.error_kind or "", a.error_kind or "ошибка")
                st = "[green]ok[/green]" if a.ok else f"[red]{escape(kind_ru)}[/red]"
                details = a.error or ""
                if a.validation_problems:
                    details = (details + "; " if details else "") + "; ".join(
                        a.validation_problems[:3]
                    )
                    if len(a.validation_problems) > 3:
                        details += f" … (+{len(a.validation_problems) - 3})"
                u = a.usage
                table.add_row(
                    str(a.n),
                    escape(a.backend),
                    escape(a.model or "по умолчанию"),
                    f"{a.duration_s:.1f} с",
                    st,
                    f"{u.input_tokens}/{u.cached_input_tokens}/{u.output_tokens}",
                    escape(short(details, 240)),
                )
            console.print(table)
        t = self.usage_total
        line = (
            f"Токены всего: вход {t.input_tokens} (кэш {t.cached_input_tokens}), "
            f"выход {t.output_tokens}"
        )
        if t.reasoning_tokens:
            line += f", рассуждения {t.reasoning_tokens}"
        if t.cost_usd is not None:
            line += f", оценка CLI ${t.cost_usd:.4f}"
        console.print(line)
        limits = next((a.limits for a in reversed(self.attempts) if a.limits), None)
        limit_text = format_limits(limits)
        if limit_text:
            console.print(f"Лимиты подписки: {escape(limit_text)}")
        for name in self.skipped:
            console.print(f"[yellow]Пропущен:[/yellow] {escape(name)}")
        if self.problems:
            console.print("[red]Проблемы:[/red]")
            for p in self.problems:
                console.print(f"  - {p}", markup=False)
        if self.final_text:
            console.print("Финальное сообщение агента:")
            console.print(short(self.final_text, 600), markup=False)


# ---------------------------------------------------------------- helpers


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    write_text(tmp, json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: the target may be open in a reader for a moment
            if i == 9:
                raise
            time.sleep(0.05 * (i + 1))


def _next_attempt_number(bundle: TaskBundle) -> int:
    n = 0
    if bundle.attempts_dir.is_dir():
        for p in bundle.attempts_dir.iterdir():
            head = p.name.split("-", 1)[0]
            if p.is_dir() and head.isdigit():
                n = max(n, int(head))
    return n + 1


def _latest_attempt_dir(bundle: TaskBundle) -> Path | None:
    best: tuple[int, Path] | None = None
    if bundle.attempts_dir.is_dir():
        for p in bundle.attempts_dir.iterdir():
            head = p.name.split("-", 1)[0]
            if p.is_dir() and head.isdigit() and (best is None or int(head) > best[0]):
                best = (int(head), p)
    return best[1] if best else None


def archive_out(bundle: TaskBundle) -> Path | None:
    """Move leftovers of out/ into <latest attempt>/out/ and leave out/ empty."""
    bundle.out_dir.mkdir(parents=True, exist_ok=True)
    entries = list(bundle.out_dir.iterdir())
    if not entries:
        return None
    target_parent = _latest_attempt_dir(bundle) or (bundle.attempts_dir / "0-before")
    target = target_parent / "out"
    target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        dest = target / entry.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(entry), str(dest))
    return target


def _validate(bundle: TaskBundle, outcome: BackendOutcome) -> tuple[list[str], Any]:
    problems = validate_outputs(bundle.out_dir, bundle.contract)
    final_problems, data = validate_final_message(
        bundle.contract, final_text=outcome.final_text, structured=outcome.structured_output
    )
    return problems + final_problems, data


# ---------------------------------------------------------------- main entry


def run_task(
    bundle: TaskBundle,
    *,
    settings: Settings,
    tier: Tier = "strong",
    backend: str | None = None,
    fallback: bool = True,
    on_event: Callable[[str], None] | None = None,
) -> RunResult:
    """Run the bundle through the backend chain until the output passes validation."""

    def emit(msg: str) -> None:
        safe_emit(on_event, msg)

    started_at, t0 = _now_iso(), time.monotonic()
    state_dir = settings.general.state_path
    max_val = max(0, settings.agents.max_validation_retries)
    candidates = candidate_backends(
        settings, stage=bundle.stage, backend=backend, fallback=fallback
    )

    attempts: list[AttemptRecord] = []
    problems: list[str] = []
    skipped: list[str] = []
    feedback: Feedback | None = None
    ok = False
    backend_used: str | None = None
    final_text, final_data = "", None
    total = Usage()

    def snapshot() -> RunResult:
        return RunResult(
            ok=ok,
            bundle=bundle,
            backend_used=backend_used,
            attempts=list(attempts),
            usage_total=total,
            final_text=final_text,
            problems=list(problems),
            started_at=started_at,
            duration_s=time.monotonic() - t0,
            final_data=final_data,
            skipped=list(skipped),
        )

    contract_problems = check_contract(bundle.contract)
    if contract_problems:
        # Our bug, not the agent's: no backend can satisfy a broken contract.
        problems = ["Некорректный контракт вывода — агент не запускался", *contract_problems]
        result = snapshot()
        atomic_write_json(bundle.run_json, result.to_dict())
        return result

    stop_all = False
    for idx, name in enumerate(candidates):
        reason = cooling_reason(name)
        if reason:
            msg = f"{name}: пропущен до перезапуска приложения ({reason})"
            skipped.append(msg)
            problems.append(msg)
            emit(msg)
            continue
        try:
            be = get_backend(name, settings)
        except ValueError as exc:
            problems.append(str(exc))
            continue

        validation_retries = crash_retries = 0
        last_problems: list[str] = []
        while True:
            archive_out(bundle)
            n = _next_attempt_number(bundle)
            attempt_dir = bundle.attempts_dir / f"{n}-{name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            prompt = build_prompt(bundle, feedback=feedback, include_system=be.inline_system_prompt)
            write_text(attempt_dir / "prompt.md", prompt)
            emit(f"Попытка {n}: агент {name}" + (" (с обратной связью)" if feedback else ""))
            attempt_started = _now_iso()
            try:
                outcome = be.invoke(
                    bundle, tier=tier, prompt=prompt, attempt_dir=attempt_dir, on_event=on_event
                )
            except Exception as exc:  # a backend bug must not lose the run record
                outcome = BackendOutcome(
                    backend=name,
                    model=None,
                    argv=[],
                    exit_code=None,
                    duration_s=0.0,
                    error_kind="crash",
                    error=f"Внутренняя ошибка бэкенда: {exc!r}",
                    transcript=attempt_dir / "transcript.jsonl",
                )
            if outcome.argv:
                # The exact command line (the prompt itself goes through stdin).
                write_text(
                    attempt_dir / "argv.json",
                    json.dumps(
                        {"argv": outcome.argv, "cwd": str(bundle.root)},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )

            vproblems: list[str] = []
            data: Any = None
            kind = outcome.error_kind
            error = outcome.error
            internal_error: str | None = None
            if kind not in HARD_ERRORS:
                try:
                    vproblems, data = _validate(bundle, outcome)
                except Exception as exc:  # a checker bug must not lose the attempt record
                    internal_error = f"внутренняя ошибка проверки результата: {exc!r}"
                    vproblems, data = [internal_error], None
                if internal_error is not None:
                    kind, error = "validation", internal_error
                elif kind is None and vproblems:
                    kind, error = "validation", f"проверка не пройдена: {len(vproblems)} проблем"
                elif kind is not None and not vproblems:
                    error = f"предупреждение: {error}" if error else None
                    kind = None
            attempt_ok = kind is None
            record = AttemptRecord(
                n=n,
                backend=name,
                model=outcome.model,
                started_at=attempt_started,
                duration_s=round(outcome.duration_s, 3),
                exit_code=outcome.exit_code,
                ok=attempt_ok,
                error_kind=kind,
                error=error,
                usage=outcome.usage,
                validation_problems=vproblems,
                transcript=str(outcome.transcript or attempt_dir / "transcript.jsonl"),
                limits=outcome.limits,
            )
            attempts.append(record)
            total = total + outcome.usage
            for note in outcome.notes:
                emit(f"{name}: {note}")
            try:
                append_usage(
                    state_dir,
                    {
                        "ts": _now_iso(),
                        "bundle_id": bundle.id,
                        "bundle": str(bundle.root),
                        "stage": bundle.stage,
                        "attempt": n,
                        "backend": name,
                        "model": outcome.model,
                        "tier": tier,
                        "ok": attempt_ok,
                        "error_kind": kind,
                        "duration_s": round(outcome.duration_s, 3),
                        **outcome.usage.to_dict(),
                        "limits": outcome.limits,
                    },
                )
            except OSError as exc:
                emit(f"Не удалось записать журнал расхода: {exc}")

            if attempt_ok:
                ok, backend_used = True, name
                final_text, final_data = outcome.final_text, data
                problems = []
                emit(f"Попытка {n}: результат принят")
                atomic_write_json(bundle.run_json, snapshot().to_dict())
                break

            final_text = outcome.final_text
            summary = f"{name}, попытка {n}: {_KIND_RU.get(kind or '', kind)}"
            if error and kind != "validation":
                summary += f" — {error}"
            emit(summary)
            atomic_write_json(bundle.run_json, snapshot().to_dict())

            if internal_error is not None:
                # Retrying or switching agents cannot fix the checker itself.
                problems.append(f"{name}: {internal_error}")
                stop_all = True
                break
            if kind == "validation":
                last_problems = vproblems
                if validation_retries < max_val:
                    validation_retries += 1
                    feedback = Feedback(
                        problems=vproblems,
                        attempt_label=attempt_dir.name,
                        previous_out=attempt_dir / "out",
                    )
                    continue
            elif kind in ("timeout", "crash", "unknown") and crash_retries < 1:
                crash_retries += 1
                continue

            # This backend is done.
            if kind in ("auth", "rate_limit"):
                mark_cooling(name, _KIND_RU[kind] + (f": {short(error, 120)}" if error else ""))
            if kind == "validation":
                problems.append(
                    f"{name}: результат не прошёл проверку (попыток: {validation_retries + 1})"
                )
                problems.extend(last_problems)
                feedback = Feedback(
                    problems=last_problems,
                    attempt_label=attempt_dir.name,
                    previous_out=attempt_dir / "out",
                )
            else:
                problems.append(
                    f"{name}: {_KIND_RU.get(kind or '', kind)}" + (f" — {error}" if error else "")
                )
            if idx + 1 < len(candidates):
                emit(f"Переключение на запасной агент: {candidates[idx + 1]}")
            break
        if ok or stop_all:
            break

    if not ok and not attempts and not problems:
        problems.append("Нет доступных агентов для запуска")
    if not ok:
        # Keep the last attempt's files where the attempt record points.
        archive_out(bundle)
    result = snapshot()
    atomic_write_json(bundle.run_json, result.to_dict())
    return result
