"""Runner policy on the fake CLIs: retries with feedback, fallback, cooling, run.json, usage."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fakes.agentkit import fake_modes, make_settings, read_calls

from h0lon.agents import (
    ExpectedFile,
    OutputContract,
    cooling_backends,
    create_bundle,
    reset_cooling,
    run_task,
)
from h0lon.agents.usage import period_totals, read_usage

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")


@pytest.fixture(autouse=True)
def _fresh_cooling():
    reset_cooling()
    yield
    reset_cooling()


def _bundle(tmp_path: Path, stage: str = "selftest"):
    contract = OutputContract(
        files=[
            ExpectedFile(
                path="result.md",
                kind="markdown",
                required_headings=["# Проверка H0lon"],
                required_text=["Статус: OK"],
            ),
            ExpectedFile(
                path="data.json",
                kind="json",
                json_schema={
                    "type": "object",
                    "properties": {"ok": {"const": True}},
                    "required": ["ok"],
                },
            ),
        ],
        final_message_schema={
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )
    return create_bundle(tmp_path / "runs", stage=stage, task="Проверка.", contract=contract)


def _attempt_dirs(bundle) -> list[str]:
    return sorted(p.name for p in bundle.attempts_dir.iterdir())


def test_success_first_try_writes_run_json_and_usage(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    events: list[str] = []
    res = run_task(b, settings=settings, tier="light", on_event=events.append)
    assert res.ok and res.backend_used == "claude" and len(res.attempts) == 1
    assert res.problems == [] and res.final_data == {"status": "ok"}
    assert (b.out_dir / "result.md").is_file()
    adir = b.attempts_dir / "1-claude"
    for name in ("prompt.md", "transcript.jsonl", "stderr.txt", "last_message.md", "argv.json"):
        assert (adir / name).is_file(), name
    run = json.loads(b.run_json.read_text(encoding="utf-8"))
    assert run["ok"] is True and run["backend_used"] == "claude"
    assert run["attempts"][0]["usage"]["input_tokens"] == 115
    assert run["usage_total"]["output_tokens"] == 20
    assert run["system_prompt_version"] == "1.0"
    recs = read_usage(settings.general.state_path)
    assert len(recs) == 1
    assert recs[0]["bundle_id"] == b.id and recs[0]["backend"] == "claude"
    assert recs[0]["ok"] is True and recs[0]["input_tokens"] == 115 and recs[0]["tier"] == "light"
    assert period_totals(settings.general.state_path)["day"].usage.output_tokens == 20
    assert any("Попытка 1" in e for e in events)
    assert not list(b.root.glob("*.tmp"))


def test_auth_failure_switches_to_codex_and_cools(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="unauth", codex="ok")
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    events: list[str] = []
    res = run_task(b, settings=settings, tier="light", on_event=events.append)
    assert res.ok and res.backend_used == "codex"
    assert [(a.backend, a.ok, a.error_kind) for a in res.attempts] == [
        ("claude", False, "auth"),
        ("codex", True, None),
    ]
    assert "Not logged in" in (res.attempts[0].error or "")
    assert any("Переключение на запасной агент: codex" in e for e in events)
    assert "claude" in cooling_backends()
    # A second task in the same process does not touch the cooling backend.
    b2 = _bundle(tmp_path)
    res2 = run_task(b2, settings=settings, tier="light")
    assert res2.ok and [a.backend for a in res2.attempts] == ["codex"]
    assert len(read_calls(state, "claude")) == 1
    assert res2.skipped and "claude" in res2.skipped[0]
    assert len(read_usage(settings.general.state_path)) == 3


def test_invalid_output_retried_with_feedback(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="bad_output,ok")
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    res = run_task(b, settings=settings, tier="light")
    assert res.ok and res.backend_used == "claude"
    assert [(a.backend, a.error_kind) for a in res.attempts] == [
        ("claude", "validation"),
        ("claude", None),
    ]
    first = res.attempts[0]
    assert any("нет обязательных заголовков" in p for p in first.validation_problems)
    assert any("некорректный JSON" in p for p in first.validation_problems)
    calls = read_calls(state, "claude")
    assert "Обратная связь по предыдущей попытке" not in calls[0]["stdin"]
    second_prompt = calls[1]["stdin"]
    assert "# Обратная связь по предыдущей попытке" in second_prompt
    assert "некорректный JSON" in second_prompt
    assert "attempts/1-claude/out/" in second_prompt
    # The rejected output was moved away; out/ now holds the accepted files only.
    assert (b.attempts_dir / "1-claude" / "out" / "data.json").read_text(
        encoding="utf-8"
    ) == "{not json"
    assert json.loads((b.out_dir / "data.json").read_text(encoding="utf-8")) == {"ok": True}
    prompt_md = (b.attempts_dir / "2-claude" / "prompt.md").read_text(encoding="utf-8")
    assert "# Обратная связь по предыдущей попытке" in prompt_md


def test_validation_retries_exhausted_then_fallback(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="bad_output", codex="ok")
    settings = make_settings(tmp_path, max_validation_retries=1)
    b = _bundle(tmp_path)
    res = run_task(b, settings=settings, tier="light")
    assert res.ok and res.backend_used == "codex"
    assert [a.backend for a in res.attempts] == ["claude", "claude", "codex"]
    assert _attempt_dirs(b) == ["1-claude", "2-claude", "3-codex"]
    # The fallback agent also gets the feedback of the last failed attempt.
    assert "Обратная связь по предыдущей попытке" in read_calls(state, "codex")[0]["stdin"]
    assert "claude" not in cooling_backends()


def test_rate_limit_goes_to_fallback(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="rate_limit", codex="ok")
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    res = run_task(b, settings=settings, tier="light")
    assert res.ok and res.backend_used == "codex"
    assert res.attempts[0].error_kind == "rate_limit"
    assert "claude" in cooling_backends()


def test_codex_rate_limit_when_primary(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="ok", codex="rate_limit")
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    res = run_task(b, settings=settings, tier="light", backend="codex")
    assert res.ok and [a.backend for a in res.attempts] == ["codex", "claude"]
    assert res.attempts[0].error_kind == "rate_limit"


def test_both_fail(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="unauth", codex="crash")
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    res = run_task(b, settings=settings, tier="light")
    assert not res.ok and res.backend_used is None
    assert [(a.backend, a.error_kind) for a in res.attempts] == [
        ("claude", "auth"),
        ("codex", "crash"),
        ("codex", "crash"),
    ]
    assert any(p.startswith("claude: не выполнен вход") for p in res.problems)
    assert any(p.startswith("codex: сбой CLI") for p in res.problems)
    run = json.loads(b.run_json.read_text(encoding="utf-8"))
    assert run["ok"] is False and len(run["attempts"]) == 3 and run["problems"]
    assert len(read_usage(settings.general.state_path)) == 3


def test_no_fallback(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="unauth")
    settings = make_settings(tmp_path)
    res = run_task(_bundle(tmp_path), settings=settings, tier="light", fallback=False)
    assert not res.ok and [a.backend for a in res.attempts] == ["claude"]
    assert read_calls(state, "codex") == []


def test_refusal_goes_to_fallback(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="refusal", codex="ok")
    res = run_task(_bundle(tmp_path), settings=make_settings(tmp_path), tier="light")
    assert res.ok and [(a.backend, a.error_kind) for a in res.attempts] == [
        ("claude", "refusal"),
        ("codex", None),
    ]


def test_missing_cli_goes_to_fallback(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path)
    # A list spec: a bare name would fall back to ~/.local/bin/claude (tools.resolve_agent_argv).
    settings = make_settings(tmp_path, claude={"bin": [str(tmp_path / "missing" / "claude.exe")]})
    res = run_task(_bundle(tmp_path), settings=settings, tier="light")
    assert res.ok and [(a.backend, a.error_kind) for a in res.attempts] == [
        ("claude", "not_found"),
        ("codex", None),
    ]


def test_stage_mapping_selects_backend(tmp_path, monkeypatch) -> None:
    fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    settings.stages.synthesis = "codex"
    res = run_task(_bundle(tmp_path, stage="synthesis"), settings=settings, tier="light")
    assert res.backend_used == "codex"


def test_timeout_kills_process(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="timeout")
    settings = make_settings(tmp_path, claude={"timeout_s": 2})
    b = _bundle(tmp_path)
    t0 = time.monotonic()
    res = run_task(b, settings=settings, tier="light", fallback=False)
    elapsed = time.monotonic() - t0
    assert not res.ok
    assert [a.error_kind for a in res.attempts] == ["timeout", "timeout"]  # one retry
    assert elapsed < 30
    beat = state / "heartbeat-claude.txt"
    assert beat.exists()
    size = beat.stat().st_size
    time.sleep(0.6)
    assert beat.stat().st_size == size, "fake agent is still running after the timeout"


def test_broken_contract_never_starts_an_agent(tmp_path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)
    # Bundles reopened or assembled by hand bypass create_bundle's contract check.
    b.contract.files[1].json_schema = {"$ref": "#/definitions/nope"}
    res = run_task(b, settings=settings, tier="light")
    assert not res.ok and res.attempts == []
    assert "агент не запускался" in res.problems[0]
    assert any("«#/definitions/nope»" in p for p in res.problems)
    assert read_calls(state, "claude") == [] and read_calls(state, "codex") == []
    assert json.loads(b.run_json.read_text(encoding="utf-8"))["ok"] is False


def test_checker_crash_still_records_attempt(tmp_path, monkeypatch) -> None:
    from h0lon.agents import runner

    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    b = _bundle(tmp_path)

    def broken(bundle, outcome):
        raise RuntimeError("checker bug")

    monkeypatch.setattr(runner, "_validate", broken)
    res = run_task(b, settings=settings, tier="light")
    assert not res.ok
    # No retries and no fallback: neither can fix the checker.
    assert [(a.backend, a.error_kind) for a in res.attempts] == [("claude", "validation")]
    assert "checker bug" in (res.attempts[0].error or "")
    assert len(read_calls(state, "claude")) == 1 and read_calls(state, "codex") == []
    assert any("внутренняя ошибка проверки" in p for p in res.problems)
    run = json.loads(b.run_json.read_text(encoding="utf-8"))
    assert run["ok"] is False and len(run["attempts"]) == 1
    recs = [r for r in read_usage(settings.general.state_path) if r["bundle_id"] == b.id]
    assert len(recs) == 1 and recs[0]["error_kind"] == "validation"


def test_settings_ignore_user_environment(tmp_path, monkeypatch) -> None:
    # The autouse clean_h0lon_env removes H0LON_* overrides of the user's shell.
    import os

    assert not [k for k in os.environ if k.upper().startswith("H0LON_AGENTS")]
    assert make_settings(tmp_path).agents.default == "claude"
