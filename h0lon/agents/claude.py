"""Claude Code headless backend: `claude -p` with stream-json output, prompt on stdin."""

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
from h0lon.agents.bundle import system_prompt_text, write_text
from h0lon.config import Settings

if TYPE_CHECKING:
    from h0lon.agents.bundle import TaskBundle

TOOLS = "Read,Write,Edit,Glob,Grep"
API_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# api_retry error categories after which further retries cannot help.
FATAL_RETRY_ERRORS = frozenset(
    {"authentication_failed", "oauth_org_not_allowed", "account_on_hold", "billing_error"}
)


@dataclass
class ClaudeStream:
    """Incremental parser of `--output-format stream-json --verbose` lines."""

    model: str | None = None
    session_id: str | None = None
    result: dict[str, Any] | None = None
    retries: list[dict[str, Any]] = field(default_factory=list)
    assistant_stop_reasons: list[str] = field(default_factory=list)
    last_text: str = ""
    bad_lines: int = 0
    limits: dict[str, Any] | None = None
    # Set when the run cannot succeed any more (auth failure, rejected limit window):
    # the backend then stops the CLI instead of waiting for its internal retries.
    fatal: str | None = None

    def feed(self, line: str) -> str | None:
        """Consume one stdout line; return a human-readable event or None."""
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
        etype, sub = ev.get("type"), ev.get("subtype")
        if etype == "system" and sub == "init":
            self.model = ev.get("model") or self.model
            self.session_id = ev.get("session_id")
            return f"claude: сессия начата (модель {self.model or '?'})"
        if etype == "system" and sub == "api_retry":
            self.retries.append(ev)
            if str(ev.get("error") or "") in FATAL_RETRY_ERRORS:
                self.fatal = str(ev.get("error"))
            return (
                f"claude: повтор запроса к API {ev.get('attempt', '?')}/"
                f"{ev.get('max_retries', '?')} ({ev.get('error') or ev.get('error_status') or '?'})"
            )
        if etype == "assistant":
            msg = ev.get("message") or {}
            if msg.get("stop_reason"):
                self.assistant_stop_reasons.append(str(msg["stop_reason"]))
            events = []
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    target = inp.get("file_path") or inp.get("path") or inp.get("pattern") or ""
                    events.append(f"claude: {block.get('name', '?')} {short(str(target), 120)}")
                elif block.get("type") == "text" and block.get("text"):
                    self.last_text = block["text"]
                    events.append(f"claude: {short(block['text'], 120)}")
            return "\n".join(events) or None
        if etype == "rate_limit_event":
            self.limits = parse_rate_limit_info(ev.get("rate_limit_info"))
            if self.limits is None:
                return None
            if self.limits.get("status") == "rejected" and not self.limits.get("using_overage"):
                self.fatal = "rate_limit"
            hot = [
                f"{name} {round(w['utilization'] * 100)} %"
                for name, w in self.limits["windows"].items()
                if (w.get("utilization") or 0) >= 0.8
            ]
            if self.limits.get("status") not in (None, "allowed") or hot:
                state = self.limits.get("status") or "?"
                return f"claude: лимит подписки — {state}; " + (", ".join(hot) or "без деталей")
            return None
        if etype == "result":
            self.result = ev
            status = "ошибка" if ev.get("is_error") else "готово"
            return f"claude: {status} ({ev.get('num_turns', '?')} ходов)"
        return None


def _iso_from_epoch(value: Any) -> str | None:
    from datetime import UTC, datetime

    if not isinstance(value, int | float) or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(value), UTC).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def parse_rate_limit_info(info: Any) -> dict[str, Any] | None:
    """Normalise Claude's `rate_limit_event.rate_limit_info` (subscription windows).

    Observed shape (Claude Code 2.1.288): {"status": "allowed", "rateLimitType": "five_hour",
    "resetsAt": <epoch s>, "overageStatus": "rejected", "isUsingOverage": false,
    "unifiedWindows": {"five_hour": {"utilization": 0.49, "resetsAt": <epoch s>}, "seven_day": …}}.
    """
    if not isinstance(info, dict):
        return None
    windows: dict[str, dict[str, Any]] = {}
    for name, win in (info.get("unifiedWindows") or {}).items():
        if not isinstance(win, dict):
            continue
        util = win.get("utilization")
        windows[str(name)] = {
            "utilization": float(util) if isinstance(util, int | float) else None,
            "resets_at": _iso_from_epoch(win.get("resetsAt")),
        }
    return {
        "status": info.get("status"),
        "type": info.get("rateLimitType"),
        "resets_at": _iso_from_epoch(info.get("resetsAt")),
        "using_overage": bool(info.get("isUsingOverage")),
        "windows": windows,
    }


def usage_from_result(result: dict[str, Any] | None) -> Usage:
    if not result:
        return Usage()
    u = result.get("usage") or {}
    base = int(u.get("input_tokens") or 0)
    created = int(u.get("cache_creation_input_tokens") or 0)
    read = int(u.get("cache_read_input_tokens") or 0)
    cost = result.get("total_cost_usd")
    return Usage(
        input_tokens=base + created + read,
        cached_input_tokens=read,
        output_tokens=int(u.get("output_tokens") or 0),
        reasoning_tokens=0,
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
    )


def classify(
    stream: ClaudeStream, proc: procutil.ProcResult, *, max_turns: int
) -> tuple[ErrorKind | None, str | None]:
    if proc.error is not None:
        return "not_found", f"Не удалось запустить claude: {proc.error}"
    if proc.timed_out:
        return "timeout", f"Превышено время ожидания ({proc.duration_s:.0f} с), процесс остановлен"
    res = stream.result
    retry_errors = " ".join(str(r.get("error") or "") for r in stream.retries)
    stop_reasons = list(stream.assistant_stop_reasons)
    if res is not None and res.get("stop_reason"):
        stop_reasons.append(str(res["stop_reason"]))
    if "refusal" in stop_reasons:
        return "refusal", "Модель отказалась выполнять задание (stop_reason = refusal)"
    if res is None:
        if stream.fatal == "rate_limit":
            return "rate_limit", "Окно лимита подписки исчерпано (rate_limit_event: rejected)"
        text = " ".join([proc.stderr[-2000:], retry_errors])
        kind = classify_error_text(text) or ("auth" if stream.fatal in FATAL_RETRY_ERRORS else None)
        detail = short(proc.stderr.strip() or "нет итогового события result", 300)
        if kind:
            return kind, detail
        return "crash", f"claude завершился с кодом {proc.exit_code} без результата: {detail}"
    if res.get("is_error"):
        text = " ".join(
            str(x)
            for x in (
                res.get("result"),
                res.get("terminal_reason"),
                res.get("api_error_status"),  # HTTP status, e.g. 401 / 429
                " ".join(map(str, res.get("errors") or [])),
                retry_errors,
            )
            if x
        )
        message = short(str(res.get("result") or res.get("terminal_reason") or text), 300)
        kind = classify_error_text(text)
        if kind is None and (stream.limits or {}).get("status") == "rejected":
            kind = "rate_limit"
        if kind is None and stream.retries:
            last = str(stream.retries[-1].get("error") or "")
            kind = classify_error_text(last.replace("_", " ")) or (
                "rate_limit" if last in ("rate_limit", "overloaded") else None
            )
        if kind:
            return kind, message
        if res.get("subtype") == "error_max_turns":
            return "unknown", f"Достигнут лимит ходов (--max-turns {max_turns})"
        return "unknown", message or "claude сообщил об ошибке"
    if proc.exit_code not in (0, None):
        return "unknown", f"claude завершился с кодом {proc.exit_code}"
    return None, None


class ClaudeBackend:
    name = "claude"
    inline_system_prompt = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.agents.claude

    def resolve(self) -> list[str] | None:
        return tools.resolve_agent_argv(self.cfg.bin, "claude")

    def env(self) -> dict[str, str]:
        drop = EXTRA_SESSION_VARS + (() if self.cfg.allow_api_key else API_KEY_VARS)
        return procutil.clean_env(drop_names=drop)

    def model_and_effort(self, tier: Tier) -> tuple[str, str]:
        if tier == "strong":
            return self.cfg.model_strong, self.cfg.effort_strong
        return self.cfg.model_light, self.cfg.effort_light

    def build_argv(
        self,
        prefix: list[str],
        bundle: TaskBundle,
        *,
        tier: Tier,
        system_prompt_file: Path,
    ) -> list[str]:
        model, effort = self.model_and_effort(tier)
        argv = [
            *prefix,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "acceptEdits",
            "--permission-prompts",
            "none",
            "--tools",
            TOOLS,
            "--allowedTools",
            TOOLS,
            "--disallowedTools",
            "mcp__*",
        ]
        if bundle.workspace:
            argv += ["--add-dir", str(bundle.workspace)]
        argv += ["--append-system-prompt-file", str(system_prompt_file)]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        argv += ["--max-turns", str(self.cfg.max_turns), "--no-session-persistence"]
        if bundle.contract.final_message_schema:
            argv += [
                "--json-schema",
                json.dumps(
                    bundle.contract.final_message_schema,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        argv += list(self.cfg.extra_args)
        return argv

    def auth_status(self) -> AuthStatus:
        prefix = self.resolve()
        if prefix is None:
            return AuthStatus(None, None, "CLI claude не найден")
        res = procutil.run([*prefix, "auth", "status"], timeout=30, env=self.env())
        if res.error or res.timed_out:
            return AuthStatus(None, None, short(res.error or "нет ответа за 30 с"))
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return AuthStatus(None, None, short(res.stdout or res.stderr or "пустой ответ"))
        logged = data.get("loggedIn")
        method = data.get("authMethod")
        detail = "вход выполнен" if logged else "не выполнен вход: claude auth login"
        if logged and method:
            detail += f" ({method})"
        return AuthStatus(bool(logged) if logged is not None else None, method, detail)

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
                error="CLI claude не найден — установите Claude Code или задайте agents.claude.bin",
                transcript=transcript,
            )
        system_file = attempt_dir / "system.md"
        write_text(system_file, system_prompt_text() + "\n")
        argv = self.build_argv(prefix, bundle, tier=tier, system_prompt_file=system_file)
        stream = ClaudeStream()
        safe_emit(on_event, f"claude: запуск (модель {model or 'по умолчанию'}, effort {effort})")

        with transcript.open("w", encoding="utf-8", newline="\n") as tfh:

            def on_line(line: str) -> str | None:
                tfh.write(line + "\n")
                tfh.flush()
                msg = stream.feed(line)
                if msg:
                    for part in msg.split("\n"):
                        safe_emit(on_event, part)
                if stream.fatal:
                    safe_emit(on_event, f"claude: остановлен досрочно ({stream.fatal})")
                    return procutil.STOP
                return None

            proc = procutil.run(
                argv,
                cwd=bundle.root,
                input_text=prompt,
                timeout=self.cfg.timeout_s,
                env=self.env(),
                on_stdout_line=on_line,
            )
        write_text(attempt_dir / "stderr.txt", proc.stderr)
        kind, error = classify(stream, proc, max_turns=self.cfg.max_turns)
        res = stream.result or {}
        final_text = str(res.get("result") or stream.last_text or "")
        structured = res.get("structured_output")
        notes = [
            f"повтор API {r.get('attempt', '?')}: {r.get('error') or r.get('error_status')}"
            for r in stream.retries
        ]
        for denial in res.get("permission_denials") or []:
            tool = denial.get("tool_name") if isinstance(denial, dict) else denial
            notes.append(f"отказано в разрешении: {short(str(tool), 120)}")
        last = final_text
        if structured is not None:
            last = json.dumps(structured, ensure_ascii=False, indent=2)
        write_text(attempt_dir / "last_message.md", last + ("\n" if last else ""))
        return BackendOutcome(
            backend=self.name,
            model=stream.model or model or None,
            argv=argv,
            exit_code=proc.exit_code,
            duration_s=proc.duration_s,
            error_kind=kind,
            error=error,
            usage=usage_from_result(stream.result),
            final_text=final_text,
            structured_output=structured,
            transcript=transcript,
            timed_out=proc.timed_out,
            notes=notes,
            limits=stream.limits,
        )
