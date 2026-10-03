"""Codex CLI headless backend: `codex exec --json`, prompt on stdin via the `-` positional."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from h0lon import procutil, tools
from h0lon.agents.base import (
    EXTRA_SESSION_VARS,
    AuthStatus,
    BackendOutcome,
    ErrorKind,
    EventCallback,
    Tier,
    Usage,
    classify_error_text,
    safe_emit,
    short,
)
from h0lon.agents.bundle import SCHEMA_FILE, write_text
from h0lon.config import Settings

if TYPE_CHECKING:
    from h0lon.agents.bundle import TaskBundle

API_KEY_VARS = ("CODEX_API_KEY", "OPENAI_API_KEY")
# Codex has no "max" reasoning effort.
_EFFORT_MAP = {"max": "xhigh"}


@dataclass
class CodexStream:
    """Incremental parser of `codex exec --json` events."""

    thread_id: str | None = None
    usage: Usage = field(default_factory=Usage)
    turn_completed: bool = False
    turn_failed: bool = False
    errors: list[str] = field(default_factory=list)
    last_agent_message: str = ""
    bad_lines: int = 0

    def feed(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            self.bad_lines += 1
            return None
        if not isinstance(ev, dict):
            return None
        etype = ev.get("type")
        if etype == "thread.started":
            self.thread_id = ev.get("thread_id")
            return "codex: сессия начата"
        if etype == "turn.completed":
            self.turn_completed = True
            self.usage = self.usage + usage_from_event(ev.get("usage") or {})
            return "codex: ход завершён"
        if etype == "turn.failed":
            self.turn_failed = True
            err = ev.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            if msg:
                self.errors.append(str(msg))
            return f"codex: ход завершился ошибкой: {short(str(msg or '?'), 200)}"
        if etype == "error":
            msg = str(ev.get("message") or "")
            self.errors.append(msg)
            return f"codex: {short(msg, 200)}"
        if etype in ("item.started", "item.completed"):
            item = ev.get("item") or {}
            itype = item.get("type")
            if itype == "agent_message" and etype == "item.completed":
                self.last_agent_message = str(item.get("text") or "")
                return f"codex: {short(self.last_agent_message, 120)}"
            if itype == "command_execution" and etype == "item.started":
                return f"codex: $ {short(str(item.get('command') or ''), 160)}"
            if itype == "file_change" and etype == "item.completed":
                paths = [
                    str(c.get("path"))
                    for c in item.get("changes") or []
                    if isinstance(c, dict) and c.get("path")
                ]
                return f"codex: изменены файлы: {short(', '.join(paths), 200)}"
            if itype == "error":
                msg = str(item.get("message") or "")
                return f"codex: предупреждение: {short(msg, 200)}"
        return None


def usage_from_event(u: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(u.get("input_tokens") or 0),
        cached_input_tokens=int(u.get("cached_input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        reasoning_tokens=int(u.get("reasoning_output_tokens") or 0),
        cost_usd=None,
    )


def image_arg(bundle: TaskBundle, image: Path) -> str:
    """`--image` value: relative to the bundle root (= cwd and -C of the process).

    codex splits `--image` values on commas (clap value_delimiter), so the path must not
    contain one; create_bundle renames such images, and a relative path keeps commas in
    state_dir or workspace paths out of the argument. Raises ValueError otherwise.
    """
    try:
        value = str(image.resolve().relative_to(bundle.root.resolve()))
    except ValueError:
        value = str(image)
    if "," in value:
        raise ValueError(
            f"путь изображения содержит запятую, codex разрезал бы его на части: {value} — "
            "переименуйте файл"
        )
    return value


def classify(stream: CodexStream, proc: procutil.ProcResult) -> tuple[ErrorKind | None, str | None]:
    if proc.error is not None:
        return "not_found", f"Не удалось запустить codex: {proc.error}"
    if proc.timed_out:
        return "timeout", f"Превышено время ожидания ({proc.duration_s:.0f} с), процесс остановлен"
    if stream.turn_completed and not stream.turn_failed and proc.exit_code == 0:
        return None, None
    errors = " | ".join(stream.errors)
    text = " ".join([errors, proc.stderr[-3000:]])
    kind = classify_error_text(text)
    detail = short(errors or proc.stderr.strip() or f"код выхода {proc.exit_code}", 300)
    if kind:
        return kind, detail
    if stream.turn_failed or stream.errors:
        return "unknown", detail
    if proc.exit_code == 0:
        return "unknown", "codex завершился без события turn.completed"
    return "crash", f"codex завершился с кодом {proc.exit_code}: {detail}"


class CodexBackend:
    name = "codex"
    inline_system_prompt = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.agents.codex

    def resolve(self) -> list[str] | None:
        return tools.resolve_agent_argv(self.cfg.bin, "codex")

    def env(self) -> dict[str, str]:
        return procutil.clean_env(drop_names=EXTRA_SESSION_VARS + API_KEY_VARS)

    def model_and_effort(self, tier: Tier) -> tuple[str, str]:
        if tier == "strong":
            model, effort = self.cfg.model_strong, self.cfg.effort_strong
        else:
            model, effort = self.cfg.model_light, self.cfg.effort_light
        return model, _EFFORT_MAP.get(effort, effort)

    def build_argv(
        self, prefix: list[str], bundle: TaskBundle, *, tier: Tier, last_message: Path
    ) -> list[str]:
        model, effort = self.model_and_effort(tier)
        argv = [
            *prefix,
            "exec",
            "--skip-git-repo-check",
            "-s",
            self.cfg.sandbox,
            "--ephemeral",
            "--json",
            "--color",
            "never",
            "-C",
            str(bundle.root),
            "-o",
            str(last_message),
        ]
        if model:
            argv += ["-m", model]
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']
        if bundle.contract.final_message_schema:
            argv += ["--output-schema", str(bundle.root / SCHEMA_FILE)]
        argv += list(self.cfg.extra_args)
        # `--image` takes several values, so images go last, each with its own flag, and
        # `--` ends option parsing before the `-` positional (prompt from stdin).
        for img in bundle.images:
            argv += ["--image", image_arg(bundle, img)]
        argv += ["--", "-"]
        return argv

    def auth_status(self) -> AuthStatus:
        prefix = self.resolve()
        if prefix is None:
            return AuthStatus(None, None, "CLI codex не найден")
        res = procutil.run([*prefix, "login", "status"], timeout=30, env=self.env())
        if res.error or res.timed_out:
            return AuthStatus(None, None, short(res.error or "нет ответа за 30 с"))
        text = (res.stdout + "\n" + res.stderr).strip()
        lower = text.lower()
        if "not logged in" in lower:
            return AuthStatus(False, None, "не выполнен вход: codex login")
        if "logged in" in lower:
            method = None
            for line in text.splitlines():
                if "logged in using" in line.lower():
                    method = line.split("using", 1)[1].strip() or None
                    break
            return AuthStatus(True, method, short(text, 200))
        return AuthStatus(None, None, short(text or f"код выхода {res.exit_code}", 200))

    def invoke(
        self,
        bundle: TaskBundle,
        *,
        tier: Tier,
        prompt: str,
        attempt_dir: Path,
        on_event: EventCallback | None,
    ) -> BackendOutcome:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        transcript = attempt_dir / "transcript.jsonl"
        last_message = attempt_dir / "last_message.md"
        model, effort = self.model_and_effort(tier)
        prefix = self.resolve()
        if prefix is None:
            write_text(transcript, "")
            return BackendOutcome(
                backend=self.name,
                model=model or None,
                argv=[],
                exit_code=None,
                duration_s=0.0,
                error_kind="not_found",
                error="CLI codex не найден — npm install -g @openai/codex или agents.codex.bin",
                transcript=transcript,
            )
        schema = bundle.schema_path
        if schema is not None and not schema.is_file():
            write_text(
                schema,
                json.dumps(bundle.contract.final_message_schema, ensure_ascii=False, indent=2),
            )
        try:
            argv = self.build_argv(prefix, bundle, tier=tier, last_message=last_message)
        except ValueError as exc:
            write_text(transcript, "")
            return BackendOutcome(
                backend=self.name,
                model=model or None,
                argv=[],
                exit_code=None,
                duration_s=0.0,
                error_kind="crash",
                error=f"codex не запущен: {exc}",
                transcript=transcript,
            )
        stream = CodexStream()
        shown_model = model or "из config.toml"
        safe_emit(
            on_event, f"codex: запуск (модель {shown_model}, effort {effort or 'по умолчанию'})"
        )
        with transcript.open("w", encoding="utf-8", newline="\n") as tfh:

            def on_line(line: str) -> None:
                tfh.write(line + "\n")
                tfh.flush()
                msg = stream.feed(line)
                if msg:
                    safe_emit(on_event, msg)

            proc = procutil.run(
                argv,
                cwd=bundle.root,
                input_text=prompt,
                timeout=self.cfg.timeout_s,
                env=self.env(),
                on_stdout_line=on_line,
            )
        write_text(attempt_dir / "stderr.txt", proc.stderr)
        kind, error = classify(stream, proc)
        final_text = ""
        if last_message.is_file():
            final_text = last_message.read_text(encoding="utf-8", errors="replace").strip()
        if not final_text:
            final_text = stream.last_agent_message.strip()
            write_text(last_message, final_text + ("\n" if final_text else ""))
        notes = [f"ошибка/предупреждение codex: {short(e, 200)}" for e in stream.errors]
        return BackendOutcome(
            backend=self.name,
            model=model or None,
            argv=argv,
            exit_code=proc.exit_code,
            duration_s=proc.duration_s,
            error_kind=kind,
            error=error,
            usage=stream.usage,
            final_text=final_text,
            structured_output=None,
            transcript=transcript,
            timed_out=proc.timed_out,
            notes=notes,
        )
