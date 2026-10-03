"""Fake `claude` CLI emulating `-p --output-format stream-json --verbose` (see _fakecommon)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fakecommon as fc

CLI = "claude"
ZERO_USAGE = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
}


def _result(**kw: object) -> dict:
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 10,
        "num_turns": 2,
        "result": "",
        "session_id": "fake-session",
        "total_cost_usd": 0.0,
        "usage": dict(ZERO_USAGE),
        "permission_denials": [],
        "terminal_reason": "completed",
    }
    base.update(kw)
    return base


def main() -> int:
    argv = sys.argv[1:]
    if argv[:2] == ["auth", "status"]:
        print(json.dumps({"loggedIn": False, "authMethod": "none"}, indent=2))
        return 1
    mode = fc.current_mode(CLI)
    stdin = fc.read_stdin()
    fc.record_call(CLI, argv=argv, stdin=stdin, mode=mode)
    root = Path.cwd()
    init = {
        "type": "system",
        "subtype": "init",
        "model": "claude-fake-1",
        "session_id": "fake-session",
        "tools": ["Read", "Write", "Edit", "Glob", "Grep"],
    }

    if mode == "unauth":
        fc.emit(
            _result(
                is_error=True,
                terminal_reason="api_error",
                result="Not logged in · Please run /login",
            )
        )
        return 1
    if mode == "rate_limit":
        fc.emit(init)
        for i in (1, 2):
            fc.emit(
                {
                    "type": "system",
                    "subtype": "api_retry",
                    "attempt": i,
                    "max_retries": 2,
                    "retry_delay_ms": 1,
                    "error_status": 429,
                    "error": "rate_limit",
                }
            )
        fc.emit(
            _result(
                is_error=True, terminal_reason="api_error", result="API Error: Rate limit reached"
            )
        )
        return 1
    if mode == "crash":
        sys.stdout.write("this is not json\n")
        sys.stderr.write("fatal: something exploded\n")
        return 3
    if mode == "timeout":
        fc.emit(init)
        fc.hang(CLI)
        return 0
    if mode == "refusal":
        fc.emit(init)
        fc.emit(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Не могу помочь."}],
                    "stop_reason": "refusal",
                },
            }
        )
        fc.emit(_result(result="Не могу помочь.", stop_reason="refusal"))
        return 0

    good = mode == "ok"
    fc.emit(init)
    written = fc.write_outputs(root, good=good)
    fc.emit(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": str(root / "out" / p)},
                    }
                    for p in written
                ],
                "stop_reason": "tool_use",
            },
        }
    )
    data, text = fc.final_message(root)
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 100,
        "output_tokens": 20,
    }
    extra = {"structured_output": data} if "--json-schema" in argv and data is not None else {}
    fc.emit(_result(result=text, usage=usage, total_cost_usd=0.0123, **extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
