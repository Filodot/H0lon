"""Shared fixtures: settings isolated from the user's config, environment and home directories."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from h0lon.config import Settings

# Test switches that must survive environment cleaning.
_KEEP_ENV = {"H0LON_LIVE_AGENT"}


@pytest.fixture
def clean_h0lon_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove H0LON_* variables (config overrides and H0LON_CONFIG) for the test."""
    for name in list(os.environ):
        if name.upper().startswith("H0LON_") and name.upper() not in _KEEP_ENV:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_settings(tmp_path: Path, clean_h0lon_env: None) -> Callable[..., Settings]:
    """Factory: Settings with workspaces/state_dir in tmp_path plus per-section overrides.

    `make_settings(render={"template": "x"}, general={"git_per_topic": False})`.
    Directories are not created; code under test creates what it needs.
    """

    def factory(**sections: dict[str, Any]) -> Settings:
        general = {
            "workspaces": tmp_path / "workspaces",
            "state_dir": tmp_path / "state",
            **sections.pop("general", {}),
        }
        return Settings(general=general, **sections)

    return factory


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    """Default settings with workspaces in tmp_path/workspaces, state in tmp_path/state."""
    return make_settings()
