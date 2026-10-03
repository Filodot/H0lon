"""Test helpers for the agent runner: settings pointing at the fake CLIs, call records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from h0lon.config import Settings

FAKES = Path(__file__).resolve().parent
FAKE_CLAUDE = FAKES / "fake_claude.py"
FAKE_CODEX = FAKES / "fake_codex.py"


def make_settings(
    tmp_path: Path,
    *,
    claude: dict[str, Any] | None = None,
    codex: dict[str, Any] | None = None,
    **agents: Any,
) -> Settings:
    claude_cfg = {"bin": [sys.executable, str(FAKE_CLAUDE)], "timeout_s": 60, **(claude or {})}
    codex_cfg = {"bin": [sys.executable, str(FAKE_CODEX)], "timeout_s": 60, **(codex or {})}
    return Settings(
        general={"state_dir": str(tmp_path / "state")},
        agents={"claude": claude_cfg, "codex": codex_cfg, **agents},
    )


def fake_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, claude: str = "ok", codex: str = "ok"
) -> Path:
    """Configure the fakes' behaviour; returns their state directory."""
    state = tmp_path / "fake-state"
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("H0LON_FAKE_STATE", str(state))
    monkeypatch.setenv("H0LON_FAKE_CLAUDE", claude)
    monkeypatch.setenv("H0LON_FAKE_CODEX", codex)
    return state


def read_calls(state: Path, cli: str) -> list[dict[str, Any]]:
    path = state / f"calls-{cli}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
