from __future__ import annotations

import sys
from pathlib import Path

import pytest

from h0lon import tools


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def no_home_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide a real ~/.local/bin/claude so 'not found' cases are deterministic."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)


# ---------------------------------------------------------------- resolve_agent_argv


def test_resolve_agent_argv_list_keeps_extra_args() -> None:
    argv = tools.resolve_agent_argv([sys.executable, "-X", "utf8"], "claude")
    assert argv is not None
    assert Path(argv[0]) == Path(sys.executable)
    assert argv[1:] == ["-X", "utf8"]


def test_resolve_agent_argv_list_head_from_path(tmp_path: Path, monkeypatch) -> None:
    exe = _touch(tmp_path / "bin" / "fake-agent")
    monkeypatch.setattr(tools, "which", lambda name: exe if name == "fake-agent" else None)
    assert tools.resolve_agent_argv(["fake-agent", "--flag"], "claude") == [str(exe), "--flag"]


def test_resolve_agent_argv_explicit_file(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "claude-custom.exe")
    assert tools.resolve_agent_argv(str(exe), "claude") == [str(exe)]


@pytest.mark.usefixtures("no_home_claude")
@pytest.mark.parametrize("kind", ["claude", "codex"])
def test_resolve_agent_argv_missing(tmp_path: Path, kind: str) -> None:
    missing = str(tmp_path / "nowhere" / "agent.exe")
    assert tools.resolve_agent_argv(missing, kind) is None
    assert tools.resolve_agent_argv([missing, "--x"], kind) is None
    assert tools.resolve_agent_argv([], kind) is None
    assert tools.resolve_agent_argv("h0lon-no-such-agent-xyz", kind) is None


def test_resolve_claude_falls_back_to_local_bin(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(tools, "which", lambda name: None)
    exe = _touch(home / ".local" / "bin" / ("claude.exe" if tools.IS_WINDOWS else "claude"))
    assert tools.resolve_agent_argv("claude", "claude") == [str(exe)]


@pytest.fixture
def as_windows_x64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the codex shim logic behave as on Windows x64 regardless of the host."""
    import platform

    monkeypatch.setattr(tools, "IS_WINDOWS", True)
    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")


_TRIPLE = Path("vendor") / "x86_64-pc-windows-msvc" / "bin" / "codex.exe"


@pytest.mark.usefixtures("as_windows_x64")
@pytest.mark.parametrize(
    "native_rel",
    [
        # npm >= 7 nested optional dependency (the layout on the reference PC)
        Path("node_modules/@openai/codex/node_modules/@openai/codex-win32-x64") / _TRIPLE,
        # hoisted optional dependency
        Path("node_modules/@openai/codex-win32-x64") / _TRIPLE,
        # binary vendored into the main package (older releases)
        Path("node_modules/@openai/codex") / _TRIPLE,
    ],
)
@pytest.mark.parametrize("shim_name", ["codex.cmd", "codex.ps1", "codex"])
def test_codex_npm_shim_resolves_to_native(tmp_path: Path, native_rel: Path, shim_name: str):
    npm = tmp_path / "npm"
    shim = _touch(npm / shim_name)
    native = _touch(npm / native_rel)
    assert tools.resolve_agent_argv(str(shim), "codex") == [str(native)]


@pytest.mark.usefixtures("as_windows_x64")
def test_codex_shim_without_native_kept(tmp_path: Path) -> None:
    shim = _touch(tmp_path / "npm" / "codex.cmd")
    assert tools.resolve_agent_argv(str(shim), "codex") == [str(shim)]


@pytest.mark.usefixtures("as_windows_x64")
def test_codex_native_exe_used_directly(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "codex.exe")
    assert tools.resolve_agent_argv(str(exe), "codex") == [str(exe)]


@pytest.mark.usefixtures("as_windows_x64")
def test_claude_shim_not_rewritten(tmp_path: Path) -> None:
    npm = tmp_path / "npm"
    shim = _touch(npm / "claude.cmd")
    _touch(npm / "node_modules/@openai/codex-win32-x64" / _TRIPLE)
    assert tools.resolve_agent_argv(str(shim), "claude") == [str(shim)]


# ---------------------------------------------------------------- browser, pandoc, misc


def test_find_browser_explicit_path(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "Browsers" / "my-chromium.exe")
    assert tools.find_browser(str(exe)) == exe


def test_find_browser_explicit_missing(tmp_path: Path) -> None:
    assert tools.find_browser(str(tmp_path / "nope" / "chrome.exe")) is None


def test_find_browser_kind_uses_path(tmp_path: Path, monkeypatch) -> None:
    exe = _touch(tmp_path / "chromium")
    monkeypatch.setattr(tools, "which", lambda name: exe if name == "chromium" else None)
    monkeypatch.setattr(tools, "_browser_paths", lambda kind: [])
    assert tools.find_browser("chromium") == exe
    assert tools.find_browser("edge") is None
    assert tools.find_browser("auto") == exe


def test_find_pandoc_bundled_or_path() -> None:
    # pypandoc-binary is a hard dependency, so pandoc is always resolvable.
    found = tools.find_pandoc("")
    assert found is not None and found.is_file()


def test_find_pandoc_override(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "pandoc-custom.exe")
    assert tools.find_pandoc(str(exe)) == exe
    assert tools.find_pandoc(str(tmp_path / "missing.exe")) is None


def test_find_xelatex_override(tmp_path: Path) -> None:
    exe = _touch(tmp_path / "xelatex.exe")
    assert tools.find_xelatex(str(exe)) == exe
    assert tools.find_xelatex(str(tmp_path / "missing.exe")) is None


def test_find_tex_tool_prefers_sibling(tmp_path: Path) -> None:
    xelatex = _touch(tmp_path / "bin" / ("xelatex.exe" if tools.IS_WINDOWS else "xelatex"))
    kpse = _touch(tmp_path / "bin" / ("kpsewhich.exe" if tools.IS_WINDOWS else "kpsewhich"))
    assert tools.find_tex_tool(xelatex, "kpsewhich") == kpse


def test_missing_tex_files_empty_list(tmp_path: Path) -> None:
    assert tools.missing_tex_files(tmp_path / "xelatex.exe", []) == []


def test_tool_version_real_process() -> None:
    version = tools.tool_version([sys.executable])
    assert version is not None and version.startswith("Python 3.")


def test_tool_version_missing_binary(tmp_path: Path) -> None:
    assert tools.tool_version([str(tmp_path / "no-such-tool.exe")]) is None


def test_tool_info_found() -> None:
    assert tools.ToolInfo("x", Path("x")).found
    assert not tools.ToolInfo("x", None).found
