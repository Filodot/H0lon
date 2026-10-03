"""`h0lon serve`: the command, the in-process server start, the browser opening, the port check."""

from __future__ import annotations

import re
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from h0lon.cli import app as cli_app
from h0lon.config import Settings
from h0lon.web import app as web_app

runner = CliRunner()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def config(tmp_path: Path, clean_h0lon_env: None) -> Path:
    path = tmp_path / "h0lon.toml"
    workspaces = (tmp_path / "ws").as_posix()
    state = (tmp_path / "state").as_posix()
    path.write_text(
        f'[general]\nworkspaces = "{workspaces}"\nstate_dir = "{state}"\ngit_per_topic = false\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(application: Any, **kwargs: Any) -> None:
        calls.append({"app": application, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


def test_serve_starts_uvicorn_in_process_on_loopback(
    config: Path, tmp_path: Path, uvicorn_calls: list[dict[str, Any]]
) -> None:
    port = free_port()
    result = runner.invoke(cli_app, ["--config", str(config), "serve", "--port", str(port)])
    assert result.exit_code == 0, result.output
    (call,) = uvicorn_calls
    assert call["host"] == "127.0.0.1" and call["port"] == port
    assert isinstance(call["app"], FastAPI)
    assert call["app"].state.settings.general.workspaces_dir == tmp_path / "ws"
    assert call["timeout_graceful_shutdown"] == 3  # Ctrl+C must not wait for event streams
    assert f"http://127.0.0.1:{port}/" in result.output
    assert "Ctrl+C" in result.output
    assert "без авторизации" not in result.output  # no warning for the default host


def test_serve_default_port_is_8765(config: Path, uvicorn_calls: list[dict[str, Any]]) -> None:
    raw = runner.invoke(
        cli_app, ["serve", "--help"], env={"NO_COLOR": "1", "COLUMNS": "200"}
    ).output
    # CI terminals make rich colour the help; strip ANSI codes before looking for the words
    help_text = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    assert "8765" in help_text and "127.0.0.1" in help_text and "--open" in help_text
    # the real default is only used when it is free on this machine: the command must pass it on
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8765))
    except OSError:
        pytest.skip("порт 8765 занят на этом компьютере")
    result = runner.invoke(cli_app, ["--config", str(config), "serve"])
    assert result.exit_code == 0, result.output
    assert uvicorn_calls[0]["port"] == 8765


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::"])
def test_serve_warns_about_a_non_loopback_host(
    config: Path, uvicorn_calls: list[dict[str, Any]], host: str
) -> None:
    port = free_port()
    result = runner.invoke(
        cli_app, ["--config", str(config), "serve", "--host", host, "--port", str(port)]
    )
    # the port check may fail for an address this computer does not have: the warning comes first
    assert "Предупреждение" in result.output
    assert "интерфейс без авторизации" in result.output
    assert "не открывайте его в сеть" in result.output


def test_serve_on_all_interfaces_accepts_any_host_header(
    config: Path, uvicorn_calls: list[dict[str, Any]]
) -> None:
    port = free_port()
    result = runner.invoke(
        cli_app, ["--config", str(config), "serve", "--host", "0.0.0.0", "--port", str(port)]
    )
    assert result.exit_code == 0, result.output
    with TestClient(uvicorn_calls[0]["app"], base_url="http://192.168.1.20:8765") as c:
        assert c.get("/").status_code == 200  # LAN names cannot be listed in advance


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_serve_loopback_hosts_get_no_warning(
    config: Path, uvicorn_calls: list[dict[str, Any]], host: str
) -> None:
    port = free_port()
    result = runner.invoke(
        cli_app, ["--config", str(config), "serve", "--host", host, "--port", str(port)]
    )
    if result.exit_code != 0:
        pytest.skip(f"адрес {host} недоступен на этом компьютере: {result.output}")
    assert "без авторизации" not in result.output


def test_serve_app_refuses_foreign_host_headers_on_loopback(
    config: Path, uvicorn_calls: list[dict[str, Any]]
) -> None:
    port = free_port()
    runner.invoke(cli_app, ["--config", str(config), "serve", "--port", str(port)])
    with TestClient(uvicorn_calls[0]["app"], base_url=f"http://127.0.0.1:{port}") as c:
        assert c.get("/").status_code == 200
        assert c.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_serve_rejects_a_bad_port(config: Path, uvicorn_calls: list[dict[str, Any]]) -> None:
    for bad in ("0", "70000", "-1"):
        result = runner.invoke(cli_app, ["--config", str(config), "serve", "--port", bad])
        assert result.exit_code == 2, bad
    assert uvicorn_calls == []


def test_serve_reports_a_busy_port_in_russian(
    config: Path, uvicorn_calls: list[dict[str, Any]]
) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        result = runner.invoke(cli_app, ["--config", str(config), "serve", "--port", str(port)])
    assert result.exit_code == 2
    assert "Ошибка:" in result.output and str(port) in result.output
    assert f"--port {port + 1}" in result.output.replace("\n", "")
    assert "Traceback" not in result.output
    assert uvicorn_calls == []


def test_serve_with_broken_config_exits_2(tmp_path: Path, clean_h0lon_env: None) -> None:
    result = runner.invoke(cli_app, ["--config", str(tmp_path / "net.toml"), "serve"])
    assert result.exit_code == 2


def test_serve_open_starts_the_browser_thread(
    settings: Settings, uvicorn_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[str, str, int]] = []
    started = threading.Event()

    def fake_open(url: str, host: str, port: int, **kwargs: Any) -> bool:
        opened.append((url, host, port))
        started.set()
        return True

    monkeypatch.setattr(web_app, "_open_when_ready", fake_open)
    port = free_port()
    web_app.serve(settings, host="127.0.0.1", port=port, open_browser=True)
    assert started.wait(5)
    assert opened == [(f"http://127.0.0.1:{port}/", "127.0.0.1", port)]
    assert uvicorn_calls[0]["port"] == port


def test_serve_without_open_does_not_touch_the_browser(
    settings: Settings, uvicorn_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("browser must not be opened")

    monkeypatch.setattr("webbrowser.open", explode)
    monkeypatch.setattr(web_app, "_open_when_ready", explode)
    web_app.serve(settings, port=free_port())
    assert len(uvicorn_calls) == 1


def test_open_when_ready_waits_for_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert web_app._open_when_ready(f"http://127.0.0.1:{port}/", "127.0.0.1", port) is True
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_open_when_ready_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    port = free_port()  # nobody listens
    assert web_app._open_when_ready("http://x/", "127.0.0.1", port, timeout_s=0.6) is False
    assert opened == []


def test_url_and_allowed_hosts_helpers() -> None:
    assert web_app._url_for("127.0.0.1", 8765) == "http://127.0.0.1:8765/"
    assert web_app._url_for("0.0.0.0", 80) == "http://127.0.0.1:80/"
    assert web_app._url_for("::1", 9) == "http://[::1]:9/"
    assert web_app._allowed_hosts("127.0.0.1") == {"127.0.0.1", "localhost", "::1"}
    assert web_app._allowed_hosts("LOCALHOST") == {"127.0.0.1", "localhost", "::1"}
    assert web_app._allowed_hosts("0.0.0.0") is None
    assert web_app._allowed_hosts("192.168.1.5") is None


def test_ensure_port_free_message() -> None:
    web_app._ensure_port_free("127.0.0.1", free_port())  # does not raise
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        with pytest.raises(OSError, match=f"h0lon serve --port {port + 1}"):
            web_app._ensure_port_free("127.0.0.1", port)


def test_serve_warns_when_jobs_were_running_at_shutdown(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from h0lon.synth.model import BuildResult
    from h0lon.workspace import create_topic

    release = threading.Event()

    def slow_build(settings: Settings, topic_dir: Path, **kwargs: Any) -> BuildResult:
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    def run_and_return(application: Any, **kwargs: Any) -> None:
        topic = create_topic(settings.model_copy(deep=True), title="Тема", course="Курс")
        application.state.jobs.start("kurs/tema", topic, "build")  # still running at «Ctrl+C»

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow_build)
    monkeypatch.setattr("uvicorn.run", run_and_return)
    try:
        with caplog.at_level("WARNING", logger="h0lon.web"):
            web_app.serve(settings, port=free_port())
    finally:
        release.set()
    assert "Сервер остановлен, пока выполнялись задачи" in caplog.text
    assert "Сборка мастер-конспекта: kurs/tema" in caplog.text


def test_serve_open_flag_reaches_the_server_start(
    config: Path, uvicorn_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = threading.Event()
    seen: list[str] = []

    def fake_open(url: str, host: str, port: int, **kwargs: Any) -> bool:
        seen.append(url)
        opened.set()
        return True

    monkeypatch.setattr(web_app, "_open_when_ready", fake_open)
    port = free_port()
    result = runner.invoke(
        cli_app, ["--config", str(config), "serve", "--port", str(port), "--open"]
    )
    assert result.exit_code == 0, result.output
    assert opened.wait(5)
    assert seen == [f"http://127.0.0.1:{port}/"]
