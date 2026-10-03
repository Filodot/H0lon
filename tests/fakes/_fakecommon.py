"""Shared helpers of the fake agent CLIs (imported from the fake scripts' own directory).

Behaviour is driven by environment variables:
- H0LON_FAKE_STATE   directory for call records (calls-<cli>.jsonl) and heartbeats (required);
- H0LON_FAKE_CLAUDE  / H0LON_FAKE_CODEX  comma-separated modes, one per invocation (the last
  one repeats): ok | unauth | rate_limit | bad_output | crash | timeout | refusal.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WATCHED_ENV = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_AGENT_SDK_VERSION",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
)


def state_dir() -> Path:
    path = Path(os.environ["H0LON_FAKE_STATE"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _calls_file(cli: str) -> Path:
    return state_dir() / f"calls-{cli}.jsonl"


def current_mode(cli: str) -> str:
    modes = [m.strip() for m in os.environ.get(f"H0LON_FAKE_{cli.upper()}", "ok").split(",")]
    modes = [m for m in modes if m] or ["ok"]
    path = _calls_file(cli)
    done = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
    return modes[min(done, len(modes) - 1)]


def record_call(cli: str, *, argv: list[str], stdin: str, mode: str, **extra: Any) -> None:
    rec = {
        "argv": argv,
        "stdin": stdin,
        "cwd": os.getcwd(),
        "mode": mode,
        "env": {k: os.environ.get(k) for k in WATCHED_ENV if k in os.environ},
        **extra,
    }
    with _calls_file(cli).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_stdin() -> str:
    data = sys.stdin.buffer.read()
    return data.decode("utf-8")


def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def hang(cli: str, seconds: float = 60.0) -> None:
    """Simulate a stuck agent: write heartbeats so a test can see whether it was killed."""
    beat = state_dir() / f"heartbeat-{cli}.txt"
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        with beat.open("a", encoding="utf-8") as fh:
            fh.write(".")
        time.sleep(0.1)


def instance_from_schema(schema: dict[str, Any]) -> Any:
    """Minimal instance that satisfies the simple schemas used by H0lon contracts."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = typ[0]
    if typ == "object" or "properties" in schema:
        props = schema.get("properties", {})
        return {k: instance_from_schema(v) for k, v in props.items()}
    if typ == "array":
        n = int(schema.get("minItems", 0))
        return [instance_from_schema(schema.get("items", {})) for _ in range(n)]
    if typ == "string":
        return "x" * max(1, int(schema.get("minLength", 1)))
    if typ == "integer":
        return int(schema.get("minimum", 0))
    if typ == "number":
        return float(schema.get("minimum", 0))
    if typ == "boolean":
        return True
    return None


def _markdown(spec: dict[str, Any]) -> str:
    lines = list(spec.get("required_headings") or []) + list(spec.get("required_text") or [])
    if not lines:
        lines = ["# Ответ"]
    text = "\n\n".join(lines) + "\n\nФейковый агент выполнил задание.\n"
    while len(text.strip()) < int(spec.get("min_chars", 1)):
        text += "Дополнительный текст для длины.\n"
    return text


def write_outputs(bundle_root: Path, *, good: bool) -> list[str]:
    """Write files for the bundle's contract into out/; return their relative paths."""
    meta = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    out = bundle_root / meta.get("out", "out")
    written = []
    for spec in meta["contract"]["files"]:
        path = out / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "json":
            if good:
                data = instance_from_schema(spec.get("json_schema") or {"type": "object"})
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            else:
                path.write_text("{not json", encoding="utf-8")
        elif good:
            path.write_text(_markdown(spec), encoding="utf-8")
        else:
            path.write_text("без заголовков\n", encoding="utf-8")
        written.append(spec["path"])
    return written


def final_message(bundle_root: Path) -> tuple[Any, str]:
    meta = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    schema = meta["contract"].get("final_message_schema")
    if not schema:
        return None, "Готово: файлы записаны."
    data = instance_from_schema(schema)
    return data, json.dumps(data, ensure_ascii=False)
