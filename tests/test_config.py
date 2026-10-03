from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import pytest
from pydantic import BaseModel

from h0lon import config
from h0lon.config import (
    Settings,
    example_config_text,
    find_config_file,
    load_settings,
    write_user_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_EXAMPLE = REPO_ROOT / "h0lon.example.toml"

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the per-user config location into tmp_path (file not created)."""
    target = tmp_path / "appdata" / "H0lon" / "h0lon.toml"
    monkeypatch.setattr(config, "user_config_path", lambda: target)
    return target


# ---------------------------------------------------------------- defaults and loading


def test_defaults() -> None:
    s = Settings()
    assert s.source_path is None
    assert s.general.language == "ru"
    assert s.general.workspaces is None
    assert s.general.workspaces_dir == Path.home() / "Konspekty"
    assert s.general.review_gate is True
    assert s.general.git_per_topic is True
    assert s.agents.default == "claude"
    assert s.agents.fallback == "codex"
    assert s.agents.parallel_runs == 2
    assert s.agents.claude.bin == "claude"
    assert s.agents.claude.allow_api_key is False
    assert s.agents.codex.sandbox == "workspace-write"
    assert s.stages.backend_for("synthesis", s.agents) == "claude"
    assert s.render.engine == "xelatex"
    assert s.render.template == "a4-notes"
    assert s.render.main_font == "Times New Roman"
    assert s.render.math_font == "Cambria Math"
    assert s.api.enabled is False


def test_load_toml(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "custom.toml",
        """
[general]
workspaces = "D:/Notes"
review_gate = false

[agents]
default = "codex"
fallback = ""

[agents.claude]
bin = ["node", "cli.js"]

[stages]
synthesis = "claude"

[render]
template = "a4-compact"
""",
    )
    s = load_settings(path)
    assert s.source_path == path
    assert s.general.workspaces_dir == Path("D:/Notes")
    assert s.general.review_gate is False
    assert s.agents.default == "codex"
    assert s.agents.fallback == ""
    assert s.agents.claude.bin == ["node", "cli.js"]
    assert s.stages.backend_for("synthesis", s.agents) == "claude"
    assert s.stages.backend_for("variants", s.agents) == "codex"
    assert s.render.template == "a4-compact"
    assert s.render.engine == "xelatex"  # untouched keys keep defaults


def test_empty_paths_mean_default(tmp_path: Path) -> None:
    path = _write(tmp_path / "h0lon.toml", '[general]\nworkspaces = ""\nstate_dir = ""\n')
    s = load_settings(path)
    assert s.general.workspaces is None and s.general.state_dir is None
    assert s.general.workspaces_dir == Path.home() / "Konspekty"


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path / "h0lon.toml",
        '[agents]\ndefault = "claude"\n[general]\nworkspaces = "D:/from-toml"\n',
    )
    monkeypatch.setenv("H0LON_AGENTS__DEFAULT", "codex")
    monkeypatch.setenv("H0LON_GENERAL__WORKSPACES", str(tmp_path / "from-env"))
    s = load_settings(path)
    assert s.agents.default == "codex"
    assert s.general.workspaces_dir == tmp_path / "from-env"
    assert s.agents.fallback == "codex"  # sibling keys are not wiped by the override


def test_invalid_value_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "h0lon.toml", '[agents]\ndefault = "gpt"\n')
    with pytest.raises(ValueError):
        load_settings(path)


def test_load_explicit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")


# ---------------------------------------------------------------- config file lookup


def test_find_config_none(tmp_path: Path, monkeypatch, user_config: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert find_config_file() is None
    assert load_settings().source_path is None


def test_find_config_order(tmp_path: Path, monkeypatch, user_config: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    _write(user_config, '[agents]\ndefault = "codex"\n')
    assert find_config_file() == user_config
    assert load_settings().agents.default == "codex"

    local = _write(cwd / "h0lon.toml", "[agents]\nparallel_runs = 3\n")
    assert find_config_file().resolve() == local.resolve()
    assert load_settings().agents.parallel_runs == 3
    assert load_settings().agents.default == "claude"  # user file is not merged in

    env_file = _write(tmp_path / "env.toml", "[agents]\nparallel_runs = 4\n")
    monkeypatch.setenv("H0LON_CONFIG", str(env_file))
    assert find_config_file() == env_file
    assert load_settings().agents.parallel_runs == 4

    explicit = _write(tmp_path / "explicit.toml", "[agents]\nparallel_runs = 5\n")
    assert find_config_file(explicit) == explicit
    assert load_settings(explicit).agents.parallel_runs == 5


def test_find_config_env_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("H0LON_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError):
        find_config_file()


def test_find_config_explicit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_config_file(tmp_path / "missing.toml")


# ---------------------------------------------------------------- example config


def test_example_copies_identical() -> None:
    packaged = resources.files("h0lon.resources").joinpath("h0lon.example.toml").read_bytes()
    assert ROOT_EXAMPLE.read_bytes() == packaged
    assert example_config_text() == packaged.decode("utf-8")


def test_example_parses_and_matches_defaults() -> None:
    s = load_settings(ROOT_EXAMPLE)
    assert s.source_path == ROOT_EXAMPLE
    assert s.model_dump() == Settings().model_dump()


def _unknown_keys(data: dict, model: type[BaseModel], prefix: str = "") -> list[str]:
    unknown: list[str] = []
    for key, value in data.items():
        name = f"{prefix}{key}"
        field = model.model_fields.get(key)
        if field is None:
            unknown.append(name)
            continue
        annotation = field.annotation
        if (
            isinstance(value, dict)
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            unknown += _unknown_keys(value, annotation, f"{name}.")
    return unknown


def test_example_has_no_unknown_keys() -> None:
    # extra="ignore" would silently drop a typo, so check the example against the schema.
    data = tomllib.loads(ROOT_EXAMPLE.read_text(encoding="utf-8"))
    assert _unknown_keys(data, Settings) == []


# ---------------------------------------------------------------- write_user_config


def test_write_user_config(tmp_path: Path) -> None:
    target = tmp_path / "cfg" / "h0lon.toml"
    assert write_user_config(target) == target
    assert target.read_text(encoding="utf-8") == example_config_text()

    target.write_text("# edited by the user\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_user_config(target)
    assert target.read_text(encoding="utf-8") == "# edited by the user\n"

    write_user_config(target, force=True)
    assert target.read_text(encoding="utf-8") == example_config_text()


def test_write_user_config_default_location(user_config: Path) -> None:
    assert write_user_config() == user_config
    assert load_settings(user_config).source_path == user_config
