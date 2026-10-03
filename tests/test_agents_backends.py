"""Claude/Codex backends against the fake CLIs: argv, stdin, environment, parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes.agentkit import fake_modes, make_settings, read_calls

from h0lon import procutil
from h0lon.agents import (
    ExpectedFile,
    OutputContract,
    build_prompt,
    classify_error_text,
    create_bundle,
)
from h0lon.agents.bundle import system_prompt_text
from h0lon.agents.claude import ClaudeBackend, ClaudeStream, classify, usage_from_result
from h0lon.agents.codex import CodexBackend, CodexStream

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")


def _bundle(tmp_path: Path, *, images: int = 0, workspace: bool = False, schema: bool = True):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    imgs = []
    for i in range(images):
        p = src / f"img{i}.png"
        p.write_bytes(b"\x89PNG fake")
        imgs.append(p)
    ws = None
    if workspace:
        ws = tmp_path / "topic"
        ws.mkdir()
    contract = OutputContract(
        files=[ExpectedFile(path="result.md", kind="markdown", required_headings=["# Итог"])],
        final_message_schema={"type": "object", "properties": {"status": {"const": "ok"}}}
        if schema
        else None,
    )
    return create_bundle(
        tmp_path / "runs",
        stage="selftest",
        task="Сделай.",
        contract=contract,
        images=imgs,
        workspace=ws,
    )


@pytest.fixture(autouse=True)
def _session_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend we run inside a Claude Code session with API keys exported.
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "desktop")
    monkeypatch.setenv("CLAUDE_AGENT_SDK_VERSION", "9.9")
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-test")


def test_claude_argv_stdin_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path, claude={"extra_args": ["--fallback-model", "haiku"]})
    b = _bundle(tmp_path, workspace=True)
    be = ClaudeBackend(settings)
    prompt = build_prompt(b) + "\nЗапрос: «кириллица» ✓\n"
    out = be.invoke(
        b, tier="light", prompt=prompt, attempt_dir=b.attempts_dir / "1-claude", on_event=None
    )
    assert out.error_kind is None, out.error
    (call,) = read_calls(state, "claude")
    argv = call["argv"]
    assert "--bare" not in argv
    assert argv[0] == "-p"
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert pairs["--permission-prompts"] == "none"
    assert pairs["--permission-mode"] == "acceptEdits"
    assert pairs["--output-format"] == "stream-json" and "--verbose" in argv
    assert pairs["--tools"] == "Read,Write,Edit,Glob,Grep"
    assert pairs["--allowedTools"] == "Read,Write,Edit,Glob,Grep"
    assert pairs["--disallowedTools"] == "mcp__*"
    assert pairs["--add-dir"] == str(b.workspace)
    assert pairs["--model"] == "sonnet" and pairs["--effort"] == "low"
    assert pairs["--max-turns"] == "40" and "--no-session-persistence" in argv
    assert json.loads(pairs["--json-schema"]) == b.contract.final_message_schema
    sys_file = Path(pairs["--append-system-prompt-file"])
    assert sys_file.read_text(encoding="utf-8").strip() == system_prompt_text()
    assert argv[-2:] == ["--fallback-model", "haiku"]
    # The task text is not on the command line; it comes through stdin intact.
    assert all("Сделай." not in a for a in argv)
    assert call["stdin"].replace("\r\n", "\n") == prompt
    assert Path(call["cwd"]).resolve() == b.root
    # Parent-session variables and API keys are not inherited.
    assert call["env"] == {"OPENAI_API_KEY": "sk-openai-test", "CODEX_API_KEY": "sk-codex-test"}
    # Parsing of the stream: usage, structured output, model.
    assert out.model == "claude-fake-1"
    assert out.structured_output == {"status": "ok"}
    assert out.usage.input_tokens == 115 and out.usage.cached_input_tokens == 100
    assert out.usage.output_tokens == 20 and out.usage.cost_usd == pytest.approx(0.0123)
    adir = b.attempts_dir / "1-claude"
    assert (adir / "transcript.jsonl").read_text(encoding="utf-8").count("\n") >= 3
    assert json.loads((adir / "last_message.md").read_text(encoding="utf-8")) == {"status": "ok"}


def test_claude_strong_tier_and_allow_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path, claude={"allow_api_key": True})
    b = _bundle(tmp_path, schema=False)
    ClaudeBackend(settings).invoke(
        b, tier="strong", prompt="p", attempt_dir=b.attempts_dir / "1-claude", on_event=None
    )
    (call,) = read_calls(state, "claude")
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"
    assert "--json-schema" not in argv and "--add-dir" not in argv
    assert call["env"].get("ANTHROPIC_API_KEY") == "sk-ant-test"
    assert "CLAUDECODE" not in call["env"] and "CLAUDE_CODE_ENTRYPOINT" not in call["env"]


def test_codex_argv_images_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path, images=2)
    be = CodexBackend(settings)
    prompt = build_prompt(b, include_system=True)
    events: list[str] = []
    out = be.invoke(
        b,
        tier="light",
        prompt=prompt,
        attempt_dir=b.attempts_dir / "1-codex",
        on_event=events.append,
    )
    assert out.error_kind is None, out.error
    (call,) = read_calls(state, "codex")
    argv, parsed = call["argv"], call["parsed"]
    assert argv[0] == "exec"
    # Images reach codex as images (relative to cwd = -C = bundle root) and the prompt
    # still comes from stdin.
    assert not any(Path(p).is_absolute() for p in parsed["image"])
    assert [(b.root / p).resolve() for p in parsed["image"]] == b.images
    assert parsed["prompt"] == "-"
    stdin = call["stdin"].replace("\r\n", "\n")
    assert stdin == prompt
    assert stdin.startswith(system_prompt_text()[:40])
    assert parsed["sandbox"] == "workspace-write"
    assert {"--json", "--ephemeral", "--skip-git-repo-check"} <= set(parsed["flags"])
    assert Path(parsed["cd"]) == b.root
    assert Path(parsed["output_last_message"]) == b.attempts_dir / "1-codex" / "last_message.md"
    assert parsed["config"] == ['model_reasoning_effort="low"']
    assert "model" not in parsed  # empty model_* → codex uses ~/.codex/config.toml
    assert Path(parsed["output_schema"]) == b.root / "schema.json"
    assert all("Сделай." not in a for a in argv)
    for name in (
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_EFFORT",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ):
        assert name not in call["env"]
    assert out.final_text == '{"status": "ok"}'
    assert out.usage.input_tokens == 1000 and out.usage.cached_input_tokens == 600
    assert out.usage.reasoning_tokens == 7
    assert any("изменены файлы" in e for e in events)


def test_codex_image_with_commas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # codex splits --image values on commas: neither the file name nor a comma in the
    # bundle path (state_dir) may reach the argument.
    state = fake_modes(monkeypatch, tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    img = src / "Скан 1, стр 2.png"
    img.write_bytes(b"fake png")
    contract = OutputContract(
        files=[ExpectedFile(path="result.md", kind="markdown", required_headings=["# Итог"])]
    )
    b = create_bundle(
        tmp_path / "runs, 2", stage="selftest", task="Сделай.", contract=contract, images=[img]
    )
    assert [p.name for p in b.images] == ["Скан 1_ стр 2.png"]
    assert (b.images[0]).read_bytes() == b"fake png"
    be = CodexBackend(make_settings(tmp_path))
    out = be.invoke(
        b, tier="light", prompt="p", attempt_dir=b.attempts_dir / "1-codex", on_event=None
    )
    assert out.error_kind is None, out.error
    (call,) = read_calls(state, "codex")
    assert call["parsed"]["image"] == [str(Path("inputs") / "Скан 1_ стр 2.png")]
    assert call["parsed"]["prompt"] == "-"


def test_codex_refuses_image_path_with_comma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A bundle assembled by hand (not via create_bundle) may still carry such a name.
    state = fake_modes(monkeypatch, tmp_path)
    b = _bundle(tmp_path)
    bad = b.inputs_dir / "a,b.png"
    bad.write_bytes(b"fake png")
    b.images.append(bad.resolve())
    out = CodexBackend(make_settings(tmp_path)).invoke(
        b, tier="light", prompt="p", attempt_dir=b.attempts_dir / "1-codex", on_event=None
    )
    assert out.error_kind == "crash" and "запятую" in (out.error or "")
    assert read_calls(state, "codex") == []


def test_codex_model_and_max_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path, codex={"model_strong": "gpt-x", "effort_strong": "max"})
    b = _bundle(tmp_path, schema=False)
    CodexBackend(settings).invoke(
        b, tier="strong", prompt="p", attempt_dir=b.attempts_dir / "1-codex", on_event=None
    )
    (call,) = read_calls(state, "codex")
    assert call["parsed"]["model"] == "gpt-x"
    assert call["parsed"]["config"] == ['model_reasoning_effort="xhigh"']
    assert "output_schema" not in call["parsed"] and call["parsed"]["image"] == []


def test_not_found(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        claude={"bin": [str(tmp_path / "no-such" / "claude.exe")]},
        codex={"bin": [str(tmp_path / "no-such" / "codex.exe")]},
    )
    b = _bundle(tmp_path)
    for be in (ClaudeBackend(settings), CodexBackend(settings)):
        out = be.invoke(
            b, tier="light", prompt="p", attempt_dir=b.attempts_dir / be.name, on_event=None
        )
        assert out.error_kind == "not_found"
        assert be.resolve() is None
        assert be.auth_status().logged_in is None


def test_auth_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    st = ClaudeBackend(settings).auth_status()
    assert st.logged_in is False and st.method == "none"
    st = CodexBackend(settings).auth_status()
    assert st.logged_in is True and st.method == "ChatGPT"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Not logged in · Please run /login", "auth"),
        ("unexpected status 401 Unauthorized", "auth"),
        ("authentication_failed", "auth"),
        ("API Error: Rate limit reached", "rate_limit"),
        ("You've hit your usage limit. Try again later", "rate_limit"),
        ("429 Too Many Requests", "rate_limit"),
        ("Claude AI usage limit reached|1760000000", "rate_limit"),
        ("API Error: 529 overloaded_error", "rate_limit"),
        ("segmentation fault", None),
        ("", None),
    ],
)
def test_classify_error_text(text: str, kind: str | None) -> None:
    assert classify_error_text(text) == kind


def _proc(code: int | None, *, stderr: str = "", timed_out: bool = False) -> procutil.ProcResult:
    return procutil.ProcResult(["claude"], code, "", stderr, 1.0, timed_out=timed_out)


def test_claude_classification() -> None:
    def stream(*lines: dict) -> ClaudeStream:
        s = ClaudeStream()
        for ev in lines:
            s.feed(json.dumps(ev))
        return s

    unauth = stream(
        {
            "type": "result",
            "is_error": True,
            "terminal_reason": "api_error",
            "result": "Not logged in · Please run /login",
        }
    )
    assert classify(unauth, _proc(1), max_turns=5)[0] == "auth"
    retry = stream(
        {"type": "system", "subtype": "api_retry", "attempt": 1, "error": "rate_limit"},
        {"type": "result", "is_error": True, "result": "API Error", "terminal_reason": "x"},
    )
    assert classify(retry, _proc(1), max_turns=5)[0] == "rate_limit"
    turns = stream({"type": "result", "is_error": True, "subtype": "error_max_turns"})
    kind, msg = classify(turns, _proc(1), max_turns=5)
    assert kind == "unknown" and "--max-turns 5" in (msg or "")
    assert classify(ClaudeStream(), _proc(2, stderr="boom"), max_turns=5)[0] == "crash"
    assert classify(ClaudeStream(), _proc(None, timed_out=True), max_turns=5)[0] == "timeout"
    ok = stream({"type": "result", "is_error": False, "result": "done"})
    assert classify(ok, _proc(0), max_turns=5) == (None, None)
    refusal = stream({"type": "result", "is_error": False, "stop_reason": "refusal"})
    assert classify(refusal, _proc(0), max_turns=5)[0] == "refusal"


def test_usage_mapping() -> None:
    u = usage_from_result(
        {
            "usage": {
                "input_tokens": 3,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 2,
                "output_tokens": 4,
            },
            "total_cost_usd": 0.5,
        }
    )
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.cost_usd) == (12, 7, 4, 0.5)
    s = CodexStream()
    s.feed(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 37902,
                    "cached_input_tokens": 24960,
                    "output_tokens": 146,
                    "reasoning_output_tokens": 0,
                },
            }
        )
    )
    assert s.turn_completed and s.usage.input_tokens == 37902 and s.usage.output_tokens == 146
