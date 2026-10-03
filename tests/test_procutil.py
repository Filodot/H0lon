"""procutil: early stop from the line callback, clean_env rules."""

from __future__ import annotations

import sys
import time

from h0lon import procutil

LOOPER = (
    "import sys, time\n"
    "for i in range(600):\n"
    "    print(f'line {i}', flush=True)\n"
    "    time.sleep(0.05)\n"
)


def test_stop_from_callback_kills_process_quickly() -> None:
    seen: list[str] = []

    def on_line(line: str) -> str | None:
        seen.append(line)
        return procutil.STOP if line == "line 2" else None

    start = time.monotonic()
    res = procutil.run([sys.executable, "-c", LOOPER], on_stdout_line=on_line, timeout=60)
    assert res.stopped is True
    assert res.timed_out is False
    assert time.monotonic() - start < 15
    assert seen[:3] == ["line 0", "line 1", "line 2"]
    # the callback is not called again after STOP
    assert seen[-1] == "line 2"


def test_callback_exception_does_not_break_reading() -> None:
    def boom(line: str) -> None:
        raise RuntimeError("callback failure")

    res = procutil.run([sys.executable, "-c", "print('a'); print('b')"], on_stdout_line=boom)
    assert res.ok
    assert res.stdout_lines == ["a", "b"]
    assert res.stopped is False


def test_clean_env_drops_session_vars_but_keeps_user_oauth_token() -> None:
    base = {
        "CLAUDECODE": "1",
        "CLAUDE_CODE_SESSION_ID": "x",
        "CLAUDE_AGENT_SDK_VERSION": "1",
        "CLAUDE_PID": "1",
        "CLAUDE_EFFORT": "xhigh",
        "CLAUDE_PREVIEW_CLASSIFIER_FLOOR": "1",
        "CLAUDE_CODE_OAUTH_TOKEN": "user-token",
        "PATH": "p",
    }
    env = procutil.clean_env(base)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "user-token"
    assert env["PATH"] == "p"
    for gone in (
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_PID",
        "CLAUDE_EFFORT",
        "CLAUDE_PREVIEW_CLASSIFIER_FLOOR",
    ):
        assert gone not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in procutil.clean_env(
        base, drop_names=["CLAUDE_CODE_OAUTH_TOKEN"]
    )
