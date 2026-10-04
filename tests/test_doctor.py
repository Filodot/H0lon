from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from h0lon import doctor, procutil, tools
from h0lon.doctor import Check, exit_code, print_report, run_checks
from h0lon.extract import asr

EXPECTED_IDS = [
    "python",
    "config",
    "workspaces",
    "claude",
    "codex",
    "agents",
    "pandoc",
    "template",
    "xelatex",
    "latex-packages",
    "fonts",
    "browser",
    "git",
    "git-identity",
    "ffmpeg",
    "yt-dlp",
    "gpu",
    "asr",
]
VALID_STATUSES = {"ok", "warn", "missing", "error", "info"}

CLAUDE_OK = json.dumps(
    {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
)
CLAUDE_LOGGED_OUT = json.dumps(
    {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}
)
NVIDIA_SMI_TABLE = "| NVIDIA-SMI 596.49   Driver Version: 596.49   CUDA Version: 13.2 |\n"
REAL_LOAD_TEMPLATE = doctor._load_template
REAL_RUN = procutil.run


class FakeMachine:
    """Fake tool discovery + fake process runner. Tests mutate `tools` and `responses`.

    `delays` (by stem or by (stem, first argument)) honour the timeout like a real hung
    process; stems in `real` are run for real through procutil.run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tools: dict[str, Path | None] = {
            name: root / "bin" / f"{name}.exe"
            for name in (
                "claude",
                "codex",
                "pandoc",
                "browser",
                "ffmpeg",
                "ffprobe",
                "yt-dlp",
                "git",
                "nvidia-smi",
                "kpsewhich",
            )
        }
        self.tools["xelatex"] = root / "miktex" / "bin" / "xelatex.exe"
        # (tool stem, first argument) -> (exit code, stdout, stderr)
        self.responses: dict[tuple[str, str | None], tuple[int, str, str]] = {
            ("claude", "--version"): (0, "2.1.288 (Claude Code)\n", ""),
            ("claude", "auth"): (0, CLAUDE_OK, ""),
            ("codex", "--version"): (0, "codex-cli 0.160.0\n", ""),
            ("codex", "login"): (0, "", "Logged in using ChatGPT\n"),
            ("pandoc", "--version"): (0, "pandoc 3.9\nFeatures: +server +lua\n", ""),
            ("xelatex", "--version"): (0, "MiKTeX-XeTeX 4.16 (MiKTeX 25.12)\n", ""),
            ("git", "--version"): (0, "git version 2.55.0\n", ""),
            ("git", "config"): (0, "Someone\n", ""),
            ("ffmpeg", "-version"): (0, "ffmpeg version 9.0-full Copyright (c) 2000-2026\n", ""),
            ("yt-dlp", "--version"): (0, "2026.09.01\n", ""),
            ("nvidia-smi", "--query-gpu"): (
                0,
                "NVIDIA GeForce RTX 4050 Laptop GPU, 596.49, 6141\n",
                "",
            ),
            ("nvidia-smi", None): (0, NVIDIA_SMI_TABLE, ""),
        }
        self.delays: dict[str | tuple[str, str | None], float] = {}
        self.real: set[str] = set()
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.timeouts: list[tuple[str, float | None]] = []
        self.packages = ["fontspec.sty", "polyglossia.sty", "mdframed.sty"]
        # what is installed for the speech recognition (faster-whisper and CUDA for it)
        self.asr = asr.AsrProbe(
            installed=True, version="1.2.1", ctranslate2="4.8.2", cuda_devices=1
        )
        self.ytdlp_module: str | None = None  # the Python package yt-dlp (None: not installed)
        self.missing_tex: list[str] = []  # files kpsewhich does not find (.sty or font files)
        self.fonts: dict[str, str] | None = {
            "times new roman": "system",
            "arial": "system",
            "consolas": "system",
            "cambria": "system",
            "cambria math": "system",
        }

    def run(self, argv, *, env=None, cwd=None, timeout=None, **kw: Any) -> procutil.ProcResult:
        args = [str(a) for a in argv]
        self.calls.append((args, dict(env) if env is not None else None))
        stem = Path(args[0]).stem.lower()
        self.timeouts.append((stem, timeout))
        if stem in self.real:
            return REAL_RUN(argv, env=env, cwd=cwd, timeout=timeout, **kw)
        first = args[1].split("=")[0] if len(args) > 1 else None
        delay = self.delays.get((stem, first), self.delays.get(stem, 0.0))
        if timeout is not None and delay > timeout:
            time.sleep(timeout)
            return procutil.ProcResult(args, None, "", "", timeout, timed_out=True)
        time.sleep(delay)
        if stem == "kpsewhich":
            found = [f"/texmf/{f}" for f in args[1:] if f not in self.missing_tex]
            code = 1 if len(found) < len(args) - 1 else 0
            return procutil.ProcResult(args, code, "\n".join(found) + "\n", "", 0.01)
        code, out, err = self.responses.get((stem, first), (1, "", f"unexpected call {args}"))
        return procutil.ProcResult(args, code, out, err, 0.01)

    def env_of(self, stem: str) -> list[dict[str, str] | None]:
        return [env for args, env in self.calls if Path(args[0]).stem.lower() == stem]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeMachine:
    m = FakeMachine(tmp_path)

    def agent(spec, kind):
        path = m.tools[kind]
        return [str(path)] if path else None

    monkeypatch.setattr(tools, "resolve_agent_argv", agent)
    monkeypatch.setattr(tools, "find_pandoc", lambda override="": m.tools["pandoc"])
    monkeypatch.setattr(tools, "find_xelatex", lambda override="": m.tools["xelatex"])
    monkeypatch.setattr(tools, "find_browser", lambda preference="auto": m.tools["browser"])
    monkeypatch.setattr(tools, "find_simple", lambda name: m.tools.get(name))
    monkeypatch.setattr(tools, "find_tex_tool", lambda xelatex, name: m.tools.get(name))
    monkeypatch.setattr(procutil, "run", m.run)
    monkeypatch.setattr(
        doctor,
        "_load_template",
        lambda s: SimpleNamespace(
            name=s.render.template, title="Fake", dir=m.root, latex_packages=m.packages
        ),
    )
    monkeypatch.setattr(doctor, "_asr_probe", lambda: m.asr)
    monkeypatch.setattr(doctor, "_ytdlp_module_version", lambda: m.ytdlp_module)
    monkeypatch.setattr(doctor, "_installed_font_families", lambda: (m.fonts, "fake"))
    monkeypatch.setattr(doctor, "_windows_font_dirs", lambda: [m.root / "winfonts"])
    return m


def by_id(checks: list[Check]) -> dict[str, Check]:
    return {c.id: c for c in checks}


# ---------------------------------------------------------------- overall outcomes


def test_all_found_exit_zero(fake: FakeMachine, settings) -> None:
    checks = run_checks(settings)
    assert [c.id for c in checks] == EXPECTED_IDS
    c = by_id(checks)
    assert exit_code(checks) == 0
    assert all(ch.status == "ok" for ch in checks if ch.required)
    assert {ch.id for ch in checks if ch.required} >= {
        "python",
        "agents",
        "pandoc",
        "template",
        "git",
    }
    assert c["template"].status == "ok" and "«a4-notes»" in c["template"].detail
    assert "подписка max" in c["claude"].detail
    assert c["codex"].detail.endswith("вход: ChatGPT")
    assert c["agents"].status == "ok" and "claude — готов" in c["agents"].detail
    assert c["xelatex"].status == "ok" and c["xelatex"].data["distribution"] == "MiKTeX"
    assert c["latex-packages"].status == "ok"
    assert c["fonts"].status == "ok"
    assert c["gpu"].status == "info"
    assert c["asr"].status == "ok" and not c["asr"].required


def test_no_pandoc_exit_one(fake: FakeMachine, settings) -> None:
    fake.tools["pandoc"] = None
    checks = run_checks(settings)
    pandoc = by_id(checks)["pandoc"]
    assert pandoc.status == "missing" and pandoc.required and pandoc.hint
    assert exit_code(checks) == 1


def test_old_pandoc_exit_one(fake: FakeMachine, settings) -> None:
    fake.responses[("pandoc", "--version")] = (0, "pandoc 2.19.2\n", "")
    checks = run_checks(settings)
    assert by_id(checks)["pandoc"].status == "error"
    assert "3.1" in by_id(checks)["pandoc"].detail
    assert exit_code(checks) == 1


def test_no_agents_exit_one(fake: FakeMachine, settings) -> None:
    fake.tools["claude"] = None
    fake.tools["codex"] = None
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["claude"].status == "missing" and c["claude"].hint == doctor._claude_install()
    assert c["codex"].status == "missing" and "@openai/codex" in (c["codex"].hint or "")
    assert c["agents"].status == "missing" and c["agents"].required
    assert exit_code(checks) == 1


def test_agents_not_logged_in(fake: FakeMachine, settings) -> None:
    fake.responses[("claude", "auth")] = (1, CLAUDE_LOGGED_OUT, "")
    fake.responses[("codex", "login")] = (1, "", "Not logged in\n")
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["claude"].status == "warn" and c["claude"].hint == "claude auth login"
    assert c["codex"].status == "warn" and c["codex"].hint == "codex login"
    assert c["agents"].status == "error"
    assert "claude auth login" in (c["agents"].hint or "")
    assert exit_code(checks) == 1

    fake.calls.clear()
    unchecked = run_checks(settings, check_auth=False)
    assert exit_code(unchecked) == 0
    assert "вход не проверялся" in by_id(unchecked)["claude"].detail
    assert not any(args[1:2] in (["auth"], ["login"]) for args, _ in fake.calls)


def test_fallback_agent_is_enough(fake: FakeMachine, settings) -> None:
    fake.tools["claude"] = None
    c = by_id(run_checks(settings))
    assert c["agents"].status == "ok"
    assert "работать будет codex" in c["agents"].detail


def test_ready_agent_outside_config_is_suggested(fake: FakeMachine, make_settings) -> None:
    s = make_settings(agents={"fallback": ""})
    fake.responses[("claude", "auth")] = (1, CLAUDE_LOGGED_OUT, "")
    checks = run_checks(s)
    agents = by_id(checks)["agents"]
    assert agents.status == "error"
    assert 'agents.default = "codex"' in (agents.hint or "")
    assert exit_code(checks) == 1


def test_unknown_auth_is_not_ok(fake: FakeMachine, settings) -> None:
    fake.responses[("claude", "auth")] = (1, "", "error: unknown command 'auth'")
    fake.responses[("codex", "login")] = (2, "", "something odd")
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["claude"].status == "warn" and c["claude"].data["logged_in"] is None
    assert c["codex"].status == "warn"
    assert c["agents"].status == "warn"
    assert exit_code(checks) == 1


def test_agent_binary_that_does_not_run(fake: FakeMachine, settings) -> None:
    fake.responses[("claude", "--version")] = (1, "", "boom")
    c = by_id(run_checks(settings))
    assert c["claude"].status == "error" and "boom" in c["claude"].detail
    assert c["agents"].status == "ok"  # codex is ready
    assert "claude — не запускается" in c["agents"].detail


def test_claude_api_key_warns(fake: FakeMachine, make_settings) -> None:
    fake.responses[("claude", "auth")] = (
        0,
        json.dumps({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"}),
        "",
    )
    claude = by_id(run_checks(make_settings()))["claude"]
    assert claude.status == "warn"
    assert "allow_api_key" in claude.detail
    allowed = by_id(run_checks(make_settings(agents={"claude": {"allow_api_key": True}})))
    assert allowed["claude"].status == "ok"


def test_claude_env_is_cleaned(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk")
    c = by_id(run_checks(settings))
    envs = fake.env_of("claude")
    assert envs and all(env is not None for env in envs)
    for env in envs:
        assert env is not None
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env
    assert "не передаётся" in c["claude"].detail
    assert "sk-test" not in json.dumps([ch.to_dict() for ch in c.values()])


def test_codex_api_key_warns_without_leaking(fake: FakeMachine, settings) -> None:
    fake.responses[("codex", "login")] = (0, "", "Logged in using an API key - sk-proj-***ABCDE\n")
    codex = by_id(run_checks(settings))["codex"]
    assert codex.status == "warn"
    assert "API-ключу" in codex.detail
    assert "sk-" not in json.dumps(codex.to_dict())


# ---------------------------------------------------------------- render toolchain


def test_xelatex_missing_with_browser_is_warning(fake: FakeMachine, settings) -> None:
    fake.tools["xelatex"] = None
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["xelatex"].status == "warn" and not c["xelatex"].required
    assert c["xelatex"].hint == doctor._xelatex_hint()
    assert c["latex-packages"].status == "warn"
    assert exit_code(checks) == 0


@pytest.mark.parametrize(
    ("windows", "claude", "xelatex", "ytdlp"),
    [
        (True, "install.ps1", "MiKTeX", "winget install yt-dlp.yt-dlp"),
        (False, "install.sh", "texlive-xetex", "uv tool install yt-dlp"),
    ],
)
def test_install_hints_follow_platform(
    fake: FakeMachine, settings, monkeypatch, windows, claude, xelatex, ytdlp
) -> None:
    monkeypatch.setattr(doctor, "IS_WINDOWS", windows)  # read at check time, not import time
    fake.tools.update({"claude": None, "xelatex": None, "yt-dlp": None})
    c = by_id(run_checks(settings))
    assert claude in (c["claude"].hint or "")
    assert xelatex in (c["xelatex"].hint or "")
    assert ytdlp in (c["yt-dlp"].hint or "")


def test_xelatex_and_browser_missing_exit_one(fake: FakeMachine, settings) -> None:
    fake.tools["xelatex"] = None
    fake.tools["browser"] = None
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["xelatex"].status == "missing" and c["xelatex"].required
    assert c["browser"].status == "warn"
    assert exit_code(checks) == 1


def test_xelatex_required_when_fallback_disabled(fake: FakeMachine, make_settings) -> None:
    fake.tools["xelatex"] = None
    checks = run_checks(make_settings(render={"fallback_html": False}))
    assert by_id(checks)["xelatex"].required
    assert exit_code(checks) == 1


def test_html_engine_requires_browser_only(fake: FakeMachine, make_settings) -> None:
    fake.tools["xelatex"] = None
    s = make_settings(render={"engine": "html"})
    checks = run_checks(s)
    c = by_id(checks)
    assert not c["xelatex"].required and c["browser"].required
    assert exit_code(checks) == 0
    fake.tools["browser"] = None
    assert exit_code(run_checks(s)) == 1


def test_missing_packages_required_without_browser(fake: FakeMachine, settings) -> None:
    fake.tools["browser"] = None
    fake.missing_tex = ["mdframed.sty"]
    checks = run_checks(settings)
    pkgs = by_id(checks)["latex-packages"]
    assert pkgs.status == "missing" and pkgs.required
    assert "mdframed.sty" in pkgs.detail
    assert "miktex packages install mdframed" in (pkgs.hint or "")
    assert exit_code(checks) == 1


def test_missing_packages_tex_live_hint(fake: FakeMachine, settings, monkeypatch) -> None:
    fake.tools["xelatex"] = fake.root / "texlive" / "2026" / "bin" / "windows" / "xelatex.exe"
    fake.responses[("xelatex", "--version")] = (0, "XeTeX 3.141592653 (TeX Live 2026)\n", "")
    fake.packages.append("tcolorbox.sty")
    fake.missing_tex = ["mdframed.sty", "tcolorbox.sty"]
    c = by_id(run_checks(settings))
    assert c["xelatex"].data["distribution"] == "TeX Live"
    assert (c["latex-packages"].hint or "").startswith("tlmgr install mdframed tcolorbox (")
    assert "tlmgr search --global --file /mdframed.sty" in (c["latex-packages"].hint or "")
    assert c["latex-packages"].status == "warn"  # browser fallback available


def test_package_hint_uses_real_package_names() -> None:
    files = ["graphicx.sty", "longtable.sty", "calc.sty", "array.sty", "amssymb.sty"]
    assert doctor._package_hint(files, "TeX Live") == "tlmgr install graphics tools amsfonts"
    miktex = doctor._package_hint(files, "MiKTeX")
    assert miktex.startswith(
        "miktex packages install graphics; miktex packages install tools; "
        "miktex packages install amsfonts ("
    )
    assert "MiKTeX Console" not in miktex  # every name is known
    guessed = doctor._package_hint(["subcaption.sty", "mdframed.sty"], "MiKTeX")
    assert "install caption; miktex packages install mdframed" in guessed
    assert "MiKTeX Console" in guessed
    unknown = doctor._package_hint(["amsthm.sty", "xurl.sty"], "")
    assert "tlmgr install amscls xurl" in unknown and "--file /xurl.sty" in unknown


def test_kpsewhich_timeout_is_not_a_failure(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 1.0)
    fake.tools["browser"] = None  # XeLaTeX is required now
    fake.delays["kpsewhich"] = 60.0
    checks = run_checks(settings)
    pkgs = by_id(checks)["latex-packages"]
    assert pkgs.status == "warn" and "kpsewhich не ответил" in pkgs.detail
    assert not pkgs.required
    assert exit_code(checks) == 0


def test_template_unknown_is_required(fake: FakeMachine, make_settings, monkeypatch) -> None:
    def no_template(s):
        raise KeyError(s.render.template)

    monkeypatch.setattr(doctor, "_load_template", no_template)
    monkeypatch.setattr(doctor, "_template_names", lambda s: ["a4-notes"])
    checks = run_checks(make_settings(render={"template": "a4-note"}))
    c = by_id(checks)
    assert c["template"].status == "error" and c["template"].required
    assert "«a4-note» не найден" in c["template"].detail
    assert "доступны: a4-notes" in (c["template"].hint or "")
    assert c["latex-packages"].status == "warn" and not c["latex-packages"].required
    assert c["browser"].status == "ok"  # the HTML fallback does not save a missing template
    assert exit_code(checks) == 1


def test_template_broken_meta_is_required(fake: FakeMachine, settings, monkeypatch) -> None:
    def broken(s):
        raise ValueError("meta.yaml: latex_packages must be a list")

    monkeypatch.setattr(doctor, "_load_template", broken)
    checks = run_checks(settings)
    tpl = by_id(checks)["template"]
    assert tpl.status == "error" and tpl.required and "latex_packages" in tpl.detail
    assert exit_code(checks) == 1


def test_render_module_unavailable(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_load_template", REAL_LOAD_TEMPLATE)
    monkeypatch.setitem(sys.modules, "h0lon.render", None)  # import now raises ImportError
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["template"].status == "error" and "h0lon.render" in c["template"].detail
    assert c["latex-packages"].status == "warn"
    assert exit_code(checks) == 1  # no render module = no PDF at all


def test_real_template_is_found(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_load_template", REAL_LOAD_TEMPLATE)
    c = by_id(run_checks(settings))
    assert c["template"].status == "ok", c["template"].detail
    assert c["latex-packages"].data["packages"]  # a4-notes declares its .sty files


def test_fonts_missing(fake: FakeMachine, settings) -> None:
    assert fake.fonts is not None
    del fake.fonts["cambria math"]
    fonts = by_id(run_checks(settings))["fonts"]
    assert fonts.status == "warn"  # browser fallback exists
    assert "Cambria Math (render.math_font)" in fonts.detail
    assert "math_font" in (fonts.hint or "")


def test_fonts_undetectable(fake: FakeMachine, settings) -> None:
    fake.fonts = None
    assert by_id(run_checks(settings))["fonts"].status == "warn"


def test_fonts_undetectable_is_not_required(fake: FakeMachine, settings) -> None:
    fake.fonts = None  # e.g. Linux without fc-list
    fake.tools["browser"] = None  # XeLaTeX is required, so real font problems would fail
    checks = run_checks(settings)
    fonts = by_id(checks)["fonts"]
    assert fonts.status == "warn" and not fonts.required
    assert "не проверены: Times New Roman" in fonts.detail
    assert exit_code(checks) == 0


def test_font_given_by_file_name(fake: FakeMachine, make_settings) -> None:
    s = make_settings(render={"math_font": "latinmodern-math.otf"})
    fonts = by_id(run_checks(s))["fonts"]
    assert fonts.status == "ok" and "latinmodern-math.otf" in fonts.detail
    kpse_calls = [args for args, _ in fake.calls if Path(args[0]).stem == "kpsewhich"]
    assert ["latinmodern-math.otf"] in [args[1:] for args in kpse_calls]

    fake.missing_tex = ["latinmodern-math.otf"]
    fake.tools["browser"] = None
    checks = run_checks(s)
    fonts = by_id(checks)["fonts"]
    assert fonts.status == "missing" and fonts.required
    assert "latinmodern-math.otf (render.math_font)" in fonts.detail
    assert "kpsewhich" in (fonts.hint or "")
    assert exit_code(checks) == 1


def test_font_file_unverifiable_without_kpsewhich(fake: FakeMachine, make_settings) -> None:
    s = make_settings(render={"math_font": "latinmodern-math.otf"})
    fake.tools["kpsewhich"] = None
    fake.tools["browser"] = None
    checks = run_checks(s)
    fonts = by_id(checks)["fonts"]
    assert fonts.status == "warn" and not fonts.required
    assert "не проверены: latinmodern-math.otf" in fonts.detail
    assert exit_code(checks) == 0
    # On Windows the user and system font folders are a fallback without kpsewhich.
    (fake.root / "winfonts").mkdir()
    (fake.root / "winfonts" / "latinmodern-math.otf").write_bytes(b"")
    fonts = by_id(run_checks(s))["fonts"]
    assert fonts.status == ("ok" if doctor.IS_WINDOWS else "warn")


def test_font_file_by_path(fake: FakeMachine, make_settings, tmp_path) -> None:
    font = tmp_path / "MyMath.otf"
    font.write_bytes(b"")
    fonts = by_id(run_checks(make_settings(render={"math_font": str(font)})))["fonts"]
    assert fonts.status == "ok"
    gone = by_id(run_checks(make_settings(render={"math_font": str(tmp_path / "No.otf")})))
    assert gone["fonts"].status == "warn"  # missing, softened by the browser fallback
    assert "No.otf (render.math_font)" in gone["fonts"].detail


def test_registry_font_names() -> None:
    names = [
        "Cambria & Cambria Math (TrueType)",
        "Arial Bold (TrueType)",
        "Arial Narrow (TrueType)",
        "Consolas",
        "Arial Narrow Полужирный (TrueType)",
    ]
    fams = doctor._registry_families(names)
    assert fams == [
        "Cambria",
        "Cambria Math",
        "Arial Bold",
        "Arial Narrow",
        "Consolas",
        "Arial Narrow Полужирный",
    ]
    installed = {f.lower(): "system" for f in fams}
    assert doctor._font_scope("Cambria Math", installed) == "system"
    assert doctor._font_scope("arial", installed) == "system"  # via "Arial Bold"
    assert doctor._font_scope("Times New Roman", installed) is None
    assert doctor._font_scope("Arial Narrow", installed) == "system"
    only_narrow = {"arial narrow": "system"}
    assert doctor._font_scope("Arial", only_narrow) is None  # a different family
    assert doctor._font_scope("Consolas", {"consolas": "user"}) == "user"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows font registry")
def test_real_windows_font_registry() -> None:
    installed, reason = doctor._installed_font_families()
    assert installed, reason
    assert doctor._font_scope("Arial", installed) is not None


def test_user_only_font_noted(fake: FakeMachine, settings) -> None:
    assert fake.fonts is not None
    fake.fonts["consolas"] = "user"
    fonts = by_id(run_checks(settings))["fonts"]
    assert fonts.status == "ok" and "только для текущего пользователя" in fonts.detail


# ---------------------------------------------------------------- git, media, GPU


def test_git_required_for_git_per_topic(fake: FakeMachine, make_settings) -> None:
    fake.tools["git"] = None
    checks = run_checks(make_settings())
    git = by_id(checks)["git"]
    assert git.status == "missing" and git.required
    assert "git-identity" not in by_id(checks)
    assert exit_code(checks) == 1

    relaxed = run_checks(make_settings(general={"git_per_topic": False}))
    assert by_id(relaxed)["git"].status == "info"
    assert exit_code(relaxed) == 0


def test_git_identity_missing(fake: FakeMachine, settings) -> None:
    fake.responses[("git", "config")] = (1, "", "")
    checks = run_checks(settings)
    ident = by_id(checks)["git-identity"]
    assert ident.status == "warn" and not ident.required
    assert "user.email" in (ident.hint or "")
    assert exit_code(checks) == 0


def test_git_config_hang_is_cut(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 1.0)
    fake.delays[("git", "config")] = 30.0
    start = time.monotonic()
    checks = run_checks(settings)
    assert time.monotonic() - start < 1.5
    c = by_id(checks)
    assert c["git"].status == "ok" and c["git"].required
    assert c["git-identity"].status == "warn" and "не удалось проверить" in c["git-identity"].detail


def test_media_tools_missing_are_warnings(fake: FakeMachine, settings) -> None:
    fake.tools["ffmpeg"] = None
    fake.tools["yt-dlp"] = None
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["ffmpeg"].status == "warn" and "для видео и аудио" in c["ffmpeg"].detail
    assert "M5" not in c["ffmpeg"].detail and "M5" not in c["yt-dlp"].detail
    assert c["yt-dlp"].status == "warn" and c["yt-dlp"].hint
    assert exit_code(checks) == 0


def test_broken_media_tools_are_not_ok(fake: FakeMachine, settings) -> None:
    fake.responses[("yt-dlp", "--version")] = (
        1,
        "",
        "Fatal Python error: failed to load python313.dll\n",
    )
    fake.responses[("ffmpeg", "-version")] = (3221225781, "", "")
    checks = run_checks(settings)
    c = by_id(checks)
    assert c["yt-dlp"].status == "warn" and c["yt-dlp"].hint
    assert "не запускается: код 1: Fatal Python error" in c["yt-dlp"].detail
    assert "version" not in c["yt-dlp"].data
    assert c["ffmpeg"].status == "warn" and "не запускается" in c["ffmpeg"].detail
    assert exit_code(checks) == 0


def test_ytdlp_python_package_is_enough(fake: FakeMachine, settings) -> None:
    # the extractor runs `python -m yt_dlp`: the package counts even without an exe on PATH
    fake.tools["yt-dlp"] = None
    fake.ytdlp_module = "2026.8.19"
    c = by_id(run_checks(settings))["yt-dlp"]
    assert c.status == "ok" and "2026.8.19" in c.detail and "python -m yt_dlp" in c.detail
    assert c.data["module"] is True and c.data["version"] == "2026.8.19"
    assert not any(Path(args[0]).stem.lower() == "yt-dlp" for args, _env in fake.calls)


def test_ffprobe_missing(fake: FakeMachine, settings) -> None:
    fake.tools["ffprobe"] = None
    ff = by_id(run_checks(settings))["ffmpeg"]
    assert ff.status == "warn" and "ffprobe" in ff.detail


def test_gpu_detail(fake: FakeMachine, settings) -> None:
    gpu = by_id(run_checks(settings))["gpu"]
    assert gpu.status == "info"
    assert "RTX 4050" in gpu.detail and "6,0 ГБ" in gpu.detail and "CUDA 13.2" in gpu.detail
    assert gpu.data["gpus"][0]["vram_mib"] == 6141


def test_no_gpu(fake: FakeMachine, settings) -> None:
    fake.tools["nvidia-smi"] = None
    gpu = by_id(run_checks(settings))["gpu"]
    assert gpu.status == "info" and "CPU или в Colab" in gpu.detail


def test_asr_ready_on_gpu(fake: FakeMachine, settings) -> None:
    c = by_id(run_checks(settings))["asr"]
    assert c.status == "ok" and not c.required and c.hint is None
    assert "faster-whisper 1.2.1" in c.detail and "ctranslate2 4.8.2" in c.detail
    assert "распознавание на GPU, модель large-v3" in c.detail
    assert c.data["installed"] and c.data["cuda_devices"] == 1 and c.data["cuda_missing"] == []


def test_asr_not_installed_is_a_warning(fake: FakeMachine, settings) -> None:
    fake.asr = asr.AsrProbe(installed=False)
    checks = run_checks(settings)
    c = by_id(checks)["asr"]
    assert c.status == "warn" and not c.required and "группа зависимостей video" in c.detail
    assert "uv sync --extra video" in (c.hint or "") and "--extra video-gpu" in (c.hint or "")
    assert exit_code(checks) == 0


def test_asr_gpu_without_cuda_libraries(fake: FakeMachine, settings) -> None:
    fake.asr = asr.AsrProbe(
        installed=True,
        version="1.2.1",
        ctranslate2="4.8.2",
        cuda_devices=1,
        cuda_missing=["cublas64_12.dll", "cudnn_ops64_9.dll"],
    )
    checks = run_checks(settings)
    c = by_id(checks)["asr"]
    assert c.status == "warn" and not c.required
    assert "cublas64_12.dll, cudnn_ops64_9.dll" in c.detail and "на CPU" in c.detail
    assert "--extra video-gpu" in (c.hint or "") and c.data["cuda_missing"][0] == "cublas64_12.dll"
    assert exit_code(checks) == 0


def test_asr_without_a_gpu_is_info(fake: FakeMachine, settings) -> None:
    fake.asr = asr.AsrProbe(installed=True, version="1.2.1", ctranslate2="4.8.2", cuda_devices=0)
    c = by_id(run_checks(settings))["asr"]
    assert c.status == "info" and "на CPU" in c.detail and "small" in c.detail


def test_asr_probe_error_is_a_warning(fake: FakeMachine, settings) -> None:
    fake.asr = asr.AsrProbe(installed=True, version="1.2.1", error="ImportError: DLL load failed")
    c = by_id(run_checks(settings))["asr"]
    assert c.status == "warn" and "не удалось проверить CUDA" in c.detail
    assert "DLL load failed" in c.detail


def test_asr_local_cpu_mode_is_mentioned(fake: FakeMachine, make_settings) -> None:
    c = by_id(run_checks(make_settings(compute={"asr": "local-cpu"})))["asr"]
    assert c.status == "ok" and "compute.asr = local-cpu" in c.detail


# ---------------------------------------------------------------- config / workspaces


def test_config_and_workspaces(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setenv("H0LON_AGENTS__DEFAULT", "claude")
    c = by_id(run_checks(settings))
    assert c["config"].status == "info" and c["config"].hint
    assert "H0LON_AGENTS__DEFAULT" in c["config"].detail
    assert c["workspaces"].status == "warn" and "h0lon init" in (c["workspaces"].hint or "")
    settings.general.workspaces_dir.mkdir(parents=True)
    (settings.general.workspaces_dir / "theorver").mkdir()
    ws = by_id(run_checks(settings))["workspaces"]
    assert ws.status == "ok" and "курсов: 1" in ws.detail


# ---------------------------------------------------------------- robustness


def test_slow_command_is_cut_by_the_budget(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 1.0)
    fake.delays["yt-dlp"] = 30.0
    start = time.monotonic()
    checks = run_checks(settings)
    assert time.monotonic() - start < 1.5
    ytdlp = by_id(checks)["yt-dlp"]
    assert ytdlp.status == "warn" and "не ответил" in ytdlp.detail
    assert by_id(checks)["pandoc"].status == "ok"


def test_commands_never_outlive_the_deadline(fake: FakeMachine, settings) -> None:
    run_checks(settings)
    assert fake.timeouts
    for stem, timeout in fake.timeouts:
        assert timeout is not None and 0 < timeout <= doctor.DEADLINE_S, stem
        if stem != "kpsewhich":
            assert timeout <= doctor.PROC_TIMEOUT_S, stem


def test_command_after_deadline_is_not_started(fake: FakeMachine, monkeypatch) -> None:
    monkeypatch.setattr(doctor._budget, "deadline", time.monotonic() - 1, raising=False)
    res = doctor._run(["git", "--version"])
    assert res.timed_out and not res.ok
    assert doctor._failure(res) == "не запускался: время проверки истекло"
    assert fake.calls == []


def test_stuck_check_gets_a_stub(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 0.5)
    monkeypatch.setattr(doctor, "_check_gpu", _sleepy(3, Check("gpu", "GPU", "info", "late")))
    start = time.monotonic()
    checks = run_checks(settings)
    assert time.monotonic() - start < 1.0
    gpu = by_id(checks)["gpu"]
    assert gpu.status == "error" and "не завершилась" in gpu.detail and not gpu.required
    assert exit_code(checks) == 0


def _sleepy(seconds: float, result: Any = None) -> Any:
    def fn(*args: Any, **kwargs: Any) -> Any:
        time.sleep(seconds)
        return result

    return fn


def test_stuck_required_check_fails(fake: FakeMachine, make_settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 0.5)
    monkeypatch.setattr(tools, "find_pandoc", _sleepy(3))
    checks = run_checks(make_settings())
    pandoc = by_id(checks)["pandoc"]
    assert pandoc.status == "error" and "не завершилась" in pandoc.detail and pandoc.required
    assert exit_code(checks) == 1


def test_stuck_git_is_required_only_with_git_per_topic(
    fake: FakeMachine, make_settings, monkeypatch
) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 0.5)
    find = tools.find_simple
    monkeypatch.setattr(
        tools, "find_simple", lambda name: _sleepy(3)() if name == "git" else find(name)
    )
    checks = run_checks(make_settings())
    git = by_id(checks)["git"]
    assert git.status == "error" and git.required
    assert exit_code(checks) == 1
    relaxed = run_checks(make_settings(general={"git_per_topic": False}))
    assert not by_id(relaxed)["git"].required
    assert exit_code(relaxed) == 0


def test_crashed_required_check_fails(fake: FakeMachine, settings, monkeypatch) -> None:
    def boom(override=""):
        raise RuntimeError("pypandoc exploded")

    monkeypatch.setattr(tools, "find_pandoc", boom)
    checks = run_checks(settings)
    pandoc = by_id(checks)["pandoc"]
    assert pandoc.status == "error" and "pypandoc exploded" in pandoc.detail
    assert pandoc.required
    assert exit_code(checks) == 1


def test_broken_check_is_isolated(fake: FakeMachine, settings, monkeypatch) -> None:
    def boom(s):
        raise RuntimeError("kaput")

    monkeypatch.setattr(doctor, "_check_gpu", boom)
    gpu = by_id(run_checks(settings))["gpu"]
    assert gpu.status == "error" and "kaput" in gpu.detail


def _hanging_tool(directory: Path, name: str) -> Path:
    """An executable `name` that sleeps for 60 s (a .bat on Windows, a shell script elsewhere)."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "hang.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    if sys.platform == "win32":
        exe = directory / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}"\r\n', encoding="utf-8")
    else:
        exe = directory / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


def test_real_hung_process_is_killed_in_time(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "DEADLINE_S", 3.0)  # commands stop at 1.5 s, then the kill
    fake.tools["yt-dlp"] = _hanging_tool(fake.root / "hang", "yt-dlp")
    fake.real.add("yt-dlp")
    start = time.monotonic()
    checks = run_checks(settings)
    elapsed = time.monotonic() - start
    assert elapsed < 3.5
    ytdlp = by_id(checks)["yt-dlp"]
    assert ytdlp.status in ("warn", "error") and not ytdlp.required
    assert "не ответил" in ytdlp.detail or "не завершилась" in ytdlp.detail


def test_stuck_thread_does_not_keep_the_process_alive(tmp_path: Path) -> None:
    code = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from h0lon import doctor, procutil
        from h0lon.config import Settings

        procutil.run = lambda argv, **kw: procutil.ProcResult([str(a) for a in argv], 1, "", "", 0)
        doctor.DEADLINE_S = 1.0
        doctor._check_gpu = lambda s: time.sleep(60)
        root = Path({str(tmp_path)!r})
        s = Settings(general={{"workspaces": root / "ws", "state_dir": root / "state"}})
        checks = doctor.run_checks(s, check_auth=False)
        print([c.status for c in checks if c.id == "gpu"][0])
        """
    )
    env = {k: v for k, v in procutil.clean_env().items() if not k.upper().startswith("H0LON_")}
    start = time.monotonic()
    res = REAL_RUN([sys.executable, "-c", code], env=env, timeout=90)
    elapsed = time.monotonic() - start
    assert res.ok, res.stderr
    assert res.stdout.strip() == "error"
    assert elapsed < 30  # a joined worker would keep the interpreter alive for 60 s


def test_to_dict(fake: FakeMachine, settings) -> None:
    checks = run_checks(settings)
    dumped = [c.to_dict() for c in checks]
    text = json.dumps(dumped, ensure_ascii=False)
    assert json.loads(text) == dumped
    first = dumped[EXPECTED_IDS.index("pandoc")]
    assert set(first) == {"id", "title", "status", "detail", "hint", "required", "data"}
    assert isinstance(first["data"]["path"], str)
    assert Check("x", "X", "warn", "d", hint="h").to_dict() == {
        "id": "x",
        "title": "X",
        "status": "warn",
        "detail": "d",
        "hint": "h",
        "required": False,
        "data": {},
    }


def test_exit_code_rules() -> None:
    assert exit_code([]) == 0
    assert exit_code([Check("a", "A", "warn", "", required=False)]) == 0
    assert exit_code([Check("a", "A", "info", "", required=True)]) == 1
    assert exit_code([Check("a", "A", "ok", "", required=True)]) == 0


def test_print_report(fake: FakeMachine, settings, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "IS_WINDOWS", True)
    fake.tools["pandoc"] = None
    fake.tools["yt-dlp"] = None
    checks = run_checks(settings)
    console = Console(record=True, width=140, color_system=None)
    print_report(checks, console=console, settings=settings)
    out = console.export_text()
    assert "Pandoc *" in out
    assert "Что сделать:" in out and "winget install yt-dlp.yt-dlp" in out
    assert "Не пройдено обязательных проверок: 1 (Pandoc)" in out


def test_print_report_all_ok_with_markup_like_text(fake: FakeMachine, settings) -> None:
    checks = run_checks(settings)
    checks.append(Check("extra", "Extra", "warn", "[render] main_font", hint="[bold]x[/bold]"))
    console = Console(record=True, width=140, color_system=None)
    print_report(checks, console=console, settings=settings)
    out = console.export_text()
    assert "[render] main_font" in out and "[bold]x[/bold]" in out  # no markup interpretation
    assert "Все обязательные проверки пройдены" in out


# ---------------------------------------------------------------- real machine


def test_real_run_checks_without_auth(settings) -> None:
    start = time.monotonic()
    checks = run_checks(settings, check_auth=False)
    elapsed = time.monotonic() - start
    ids = [c.id for c in checks]
    assert ids[:6] == EXPECTED_IDS[:6]
    assert set(ids) <= set(EXPECTED_IDS)
    assert all(c.status in VALID_STATUSES for c in checks)
    assert by_id(checks)["python"].status == "ok"
    assert by_id(checks)["pandoc"].status == "ok"  # pypandoc-binary is a dependency
    assert by_id(checks)["template"].status == "ok"  # built-in a4-notes
    assert elapsed < doctor.DEADLINE_S + 1
    json.dumps([c.to_dict() for c in checks], ensure_ascii=False)
    print_report(checks, console=Console(record=True, width=120), settings=settings)
