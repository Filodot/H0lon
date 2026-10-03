"""Subprocess helpers: UTF-8 capture, live line callbacks, timeouts with tree kill, clean env."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

# Variables a parent Claude Code / desktop session sets for its own children. A nested
# `claude -p` must not inherit them, otherwise it may think it runs inside that session.
_SESSION_ENV_PREFIXES = ("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_")
_SESSION_ENV_NAMES = (
    "CLAUDECODE",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "CLAUDE_PREVIEW_CLASSIFIER_FLOOR",
)
# User-chosen auth for Claude Code (`claude setup-token`); must survive the session cleanup.
_SESSION_ENV_KEEP = ("CLAUDE_CODE_OAUTH_TOKEN",)
# Return this from `on_stdout_line` to stop the process early (its tree is killed).
STOP = "stop"

API_KEY_ENV_NAMES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CODEX_API_KEY", "OPENAI_API_KEY")


@dataclass
class ProcResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    error: str | None = None  # OSError text when the process could not be started
    stopped: bool = False  # the on_stdout_line callback asked to stop the process
    stdout_lines: list[str] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


def clean_env(
    base: Mapping[str, str] | None = None,
    *,
    drop_session_vars: bool = True,
    drop_names: Iterable[str] = (),
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy of the environment without parent-agent session variables (and optional names)."""
    env = dict(os.environ if base is None else base)
    drop = {n.upper() for n in drop_names}
    for key in list(env):
        upper = key.upper()
        if upper in drop or (
            drop_session_vars
            and upper not in _SESSION_ENV_KEEP
            and (upper in _SESSION_ENV_NAMES or upper.startswith(_SESSION_ENV_PREFIXES))
        ):
            env.pop(key)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra:
        env.update(extra)
    return env


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()


def run(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    on_stdout_line: Callable[[str], object] | None = None,
    encoding: str = "utf-8",
) -> ProcResult:
    """Run a process, capturing stdout/stderr as text. Never raises for process failures.

    stdin is closed (or fed `input_text` and closed) so tools that wait on stdin do not hang.
    """
    args = [os.fspath(a) for a in argv]
    start = time.monotonic()
    creationflags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=encoding,
            errors="replace",
            creationflags=creationflags,
        )
    except OSError as exc:
        return ProcResult(args, None, "", "", time.monotonic() - start, error=str(exc))

    out_lines: list[str] = []
    err_chunks: list[str] = []
    stop_requested = threading.Event()

    def _read_out() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out_lines.append(line)
            if on_stdout_line is None or stop_requested.is_set():
                continue
            try:
                verdict = on_stdout_line(line.rstrip("\r\n"))
            except Exception:  # callbacks must never break the reader
                verdict = None
            if verdict == STOP:
                stop_requested.set()
                _kill_tree(proc)

    def _read_err() -> None:
        assert proc.stderr is not None
        err_chunks.append(proc.stderr.read())

    t_out = threading.Thread(target=_read_out, daemon=True)
    t_err = threading.Thread(target=_read_err, daemon=True)
    t_out.start()
    t_err.start()

    if input_text is not None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(input_text)
        except (BrokenPipeError, OSError):
            pass
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    t_out.join(timeout=10)
    t_err.join(timeout=10)
    return ProcResult(
        argv=args,
        exit_code=proc.returncode,
        stdout="".join(out_lines),
        stderr="".join(err_chunks),
        duration_s=time.monotonic() - start,
        timed_out=timed_out,
        stopped=stop_requested.is_set(),
        stdout_lines=[line.rstrip("\r\n") for line in out_lines],
    )
