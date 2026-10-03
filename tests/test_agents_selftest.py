"""`agent_test` (h0lon agent-test) on the fake CLIs, plus an opt-in live run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fakes.agentkit import fake_modes, make_settings, read_calls
from rich.console import Console

from h0lon.agents import reset_cooling
from h0lon.agents.selftest import agent_test, selftest_runs_dir


@pytest.fixture(autouse=True)
def _fresh_cooling():
    reset_cooling()
    yield
    reset_cooling()


@pytest.mark.usefixtures("clean_h0lon_env")
def test_selftest_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    res = agent_test(settings, backend="codex")
    assert res.ok, res.problems
    assert res.bundle.root.parent == selftest_runs_dir(settings).resolve()
    assert [f.path for f in res.bundle.contract.files] == ["result.md", "data.json"]
    result_md = (res.bundle.out_dir / "result.md").read_text(encoding="utf-8")
    assert "# Проверка H0lon" in result_md and "Статус: OK" in result_md
    assert json.loads((res.bundle.out_dir / "data.json").read_text(encoding="utf-8")) == {
        "ok": True,
        "language": "ru",
    }
    assert res.final_data == {"status": "ok", "files": []}
    json.dumps(res.to_dict(), ensure_ascii=False)  # serializable for --json


@pytest.mark.usefixtures("clean_h0lon_env")
def test_selftest_prompt_and_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_modes(monkeypatch, tmp_path, claude="unauth")
    settings = make_settings(tmp_path)
    img_dir = tmp_path / "pics"
    (img_dir / "x").mkdir(parents=True)
    a, b = img_dir / "scan.png", img_dir / "x" / "scan.png"
    a.write_bytes(b"\x89PNG a")
    b.write_bytes(b"\x89PNG b")
    res = agent_test(settings, prompt="Сколько будет 2+2?", images=[a, b])
    assert res.ok and res.backend_used == "codex"
    files = {f.path: f for f in res.bundle.contract.files}
    assert set(files) == {"result.md", "data.json", "answer.md", "images.md"}
    assert files["images.md"].required_headings == ["## scan.png", "## scan-2.png"]
    meta = json.loads((res.bundle.root / "bundle.json").read_text(encoding="utf-8"))
    assert [f["path"] for f in meta["contract"]["files"]][-1] == "images.md"
    task = res.bundle.task_path.read_text(encoding="utf-8")
    assert "Сколько будет 2+2?" in task and "out/images.md" in task
    (call,) = read_calls(state, "codex")
    assert [Path(p).name for p in call["parsed"]["image"]] == ["scan.png", "scan-2.png"]


@pytest.mark.usefixtures("clean_h0lon_env")
def test_result_print(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="bad_output,ok")
    settings = make_settings(tmp_path)
    res = agent_test(settings)
    console = Console(record=True, width=160, color_system=None)
    res.print(console)
    text = console.export_text()
    assert "Прогон агента: OK" in text
    assert str(res.bundle.root) in text.replace("\n", "")
    assert "claude" in text and "результат не прошёл проверку" in text
    assert "Токены всего: вход 230 (кэш 200), выход 40" in text


@pytest.mark.usefixtures("clean_h0lon_env")
def test_result_print_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modes(monkeypatch, tmp_path, claude="unauth", codex="unauth")
    res = agent_test(make_settings(tmp_path))
    assert not res.ok
    console = Console(record=True, width=160, color_system=None)
    res.print(console)
    text = console.export_text()
    assert "ОШИБКА" in text and "Проблемы:" in text and "не выполнен вход" in text


@pytest.mark.live_agent
@pytest.mark.skipif(os.environ.get("H0LON_LIVE_AGENT") != "1", reason="H0LON_LIVE_AGENT != 1")
@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_live_agent(backend: str) -> None:
    from h0lon.agents import get_backend
    from h0lon.config import load_settings

    settings = load_settings()
    be = get_backend(backend, settings)
    if be.resolve() is None:
        pytest.skip(f"{backend} не найден")
    if be.auth_status().logged_in is False:
        pytest.skip(f"{backend}: не выполнен вход")
    res = agent_test(settings, backend=backend, fallback=False, tier="light")
    assert res.ok, res.problems


@pytest.mark.usefixtures("clean_h0lon_env")
def test_result_print_escapes_markup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from h0lon.agents import AttemptRecord, Usage
    from h0lon.agents.runner import RunResult

    fake_modes(monkeypatch, tmp_path)
    # Square brackets in the state path, problems and CLI errors must print verbatim.
    settings = make_settings(tmp_path / "st [b]")
    res = agent_test(settings, backend="codex")
    bad = AttemptRecord(
        n=2,
        backend="codex",
        model="m[/i]",
        started_at="",
        duration_s=1.0,
        exit_code=1,
        ok=False,
        error_kind="crash",
        error="codex завершился с кодом 1: [/tmp] something",
        usage=Usage(),
        validation_problems=['out/data.json: «$.status»: значение "[/b] готово"'],
        transcript="",
    )
    broken = RunResult(
        ok=False,
        bundle=res.bundle,
        backend_used=None,
        attempts=[*res.attempts, bad],
        usage_total=res.usage_total,
        final_text="[/red]",
        problems=["[/x] проблема"],
        skipped=["[/y] claude: пропущен"],
    )
    console = Console(record=True, width=200, color_system=None)
    broken.print(console)
    text = console.export_text()
    for fragment in ("[/tmp] something", "[/b] готово", "m[/i]", "[/x] проблема", "[/y] claude"):
        assert fragment in text, fragment
    assert "st [b]" in text.replace("\n", "")


@pytest.mark.usefixtures("clean_h0lon_env")
def test_selftest_missing_image_leaves_no_bundle(tmp_path: Path, monkeypatch) -> None:
    state = fake_modes(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="Входной файл не найден"):
        agent_test(settings, images=[tmp_path / "нет такого.png"])
    runs = selftest_runs_dir(settings)
    assert not runs.exists() or not any(runs.iterdir())
    assert read_calls(state, "claude") == [] and read_calls(state, "codex") == []
