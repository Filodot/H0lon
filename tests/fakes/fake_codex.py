"""Fake `codex` CLI emulating `codex exec --json` (see _fakecommon).

Argument parsing mimics clap: `-i/--image` takes one or more values and greedily swallows
following non-option tokens (a lone `-` included); `--` ends option parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fakecommon as fc

CLI = "codex"
VALUE_OPTS = {
    "-c": "config",
    "--config": "config",
    "-m": "model",
    "--model": "model",
    "-s": "sandbox",
    "--sandbox": "sandbox",
    "-C": "cd",
    "--cd": "cd",
    "--add-dir": "add_dir",
    "--output-schema": "output_schema",
    "-o": "output_last_message",
    "--output-last-message": "output_last_message",
    "--color": "color",
    "-p": "profile",
    "--profile": "profile",
}
FLAGS = {"--json", "--ephemeral", "--skip-git-repo-check", "--oss", "--strict-config"}


def _is_option(tok: str) -> bool:
    return tok.startswith("-") and tok != "-"


def parse(argv: list[str]) -> dict:
    opts: dict = {"config": [], "image": [], "flags": [], "prompt": None, "add_dir": []}
    i = 0
    positional_only = False
    while i < len(argv):
        tok = argv[i]
        if positional_only or not _is_option(tok):
            if opts["prompt"] is not None:
                raise SystemExit(f"error: unexpected argument '{tok}'")
            opts["prompt"] = tok
            i += 1
            continue
        if tok == "--":
            positional_only = True
            i += 1
            continue
        if tok in ("-i", "--image") or tok.startswith("--image="):
            if tok.startswith("--image="):
                opts["image"] += tok.split("=", 1)[1].split(",")
                i += 1
                continue
            i += 1
            taken = 0
            while i < len(argv) and not _is_option(argv[i]):
                opts["image"] += argv[i].split(",")
                i += 1
                taken += 1
            if not taken:
                raise SystemExit("error: a value is required for '--image <FILE>...'")
            continue
        if tok in VALUE_OPTS:
            key = VALUE_OPTS[tok]
            value = argv[i + 1]
            if key in ("config", "add_dir"):
                opts[key].append(value)
            else:
                opts[key] = value
            i += 2
            continue
        if tok in FLAGS:
            opts["flags"].append(tok)
            i += 1
            continue
        raise SystemExit(f"error: unexpected argument '{tok}' found")
    return opts


def main() -> int:
    argv = sys.argv[1:]
    if argv[:2] == ["login", "status"]:
        print("Logged in using ChatGPT", file=sys.stderr)
        return 0
    if not argv or argv[0] != "exec":
        print("fake codex supports only `exec` and `login status`", file=sys.stderr)
        return 2
    try:
        opts = parse(argv[1:])
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    stdin = fc.read_stdin() if opts["prompt"] in (None, "-") else ""
    mode = fc.current_mode(CLI)
    fc.record_call(CLI, argv=argv, stdin=stdin, mode=mode, parsed=opts)

    for img in opts["image"]:
        if not Path(img).is_file():
            print(f"Error: failed to load image {img}", file=sys.stderr)
            return 1
    root = Path(opts.get("cd") or ".").resolve()
    fc.emit({"type": "thread.started", "thread_id": "fake-thread"})
    fc.emit({"type": "turn.started"})

    if mode == "unauth":
        msg = "unexpected status 401 Unauthorized: Missing bearer or basic authentication"
        fc.emit({"type": "error", "message": msg})
        fc.emit({"type": "turn.failed", "error": {"message": msg}})
        return 1
    if mode == "rate_limit":
        msg = "You've hit your usage limit. Try again in 3 hours."
        fc.emit({"type": "error", "message": msg})
        fc.emit({"type": "turn.failed", "error": {"message": msg}})
        return 1
    if mode == "crash":
        sys.stderr.write("thread 'main' panicked at fake\n")
        return 101
    if mode == "timeout":
        fc.hang(CLI)
        return 0

    written = fc.write_outputs(root, good=mode == "ok")
    fc.emit(
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "file_change",
                "changes": [{"path": str(root / "out" / p), "kind": "add"} for p in written],
                "status": "completed",
            },
        }
    )
    _, text = fc.final_message(root)
    if opts.get("output_schema") is None:
        text = "Готово."
    fc.emit(
        {"type": "item.completed", "item": {"id": "item_2", "type": "agent_message", "text": text}}
    )
    if opts.get("output_last_message"):
        Path(opts["output_last_message"]).write_text(text, encoding="utf-8")
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 600,
        "cache_write_input_tokens": 0,
        "output_tokens": 50,
        "reasoning_output_tokens": 7,
    }
    fc.emit({"type": "turn.completed", "usage": usage})
    return 0


if __name__ == "__main__":
    sys.exit(main())
