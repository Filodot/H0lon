"""Seed files: placed into out/ before every attempt so agents can edit them in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from h0lon.agents import ExpectedFile, OutputContract, create_bundle
from h0lon.agents.runner import archive_out, restore_seed


def _contract() -> OutputContract:
    return OutputContract(files=[ExpectedFile(path="sections/s01.md", kind="markdown")])


def test_seed_is_copied_and_restored_after_archive(tmp_path: Path) -> None:
    original = tmp_path / "s01.md"
    original.write_text("# Раздел {#sec:s01}\n\nисходный текст\n", encoding="utf-8")
    bundle = create_bundle(
        tmp_path / "runs",
        stage="global",
        task="Правь out/sections/s01.md на месте.",
        contract=_contract(),
        seed={"sections/s01.md": original},
    )
    assert (bundle.seed_dir / "sections" / "s01.md").is_file()
    assert not (bundle.out_dir / "sections" / "s01.md").exists()

    assert restore_seed(bundle) == 1
    target = bundle.out_dir / "sections" / "s01.md"
    assert target.read_text(encoding="utf-8").endswith("исходный текст\n")

    target.write_text("испорчено агентом\n", encoding="utf-8")
    archive_out(bundle)
    assert not target.exists()
    restore_seed(bundle)
    assert "исходный текст" in target.read_text(encoding="utf-8")


def test_seed_rejects_paths_outside_out(tmp_path: Path) -> None:
    src = tmp_path / "a.md"
    src.write_text("x", encoding="utf-8")
    for bad in ("../a.md", "C:/a.md", ""):
        with pytest.raises(ValueError):
            create_bundle(
                tmp_path / "runs", stage="t", task="t", contract=_contract(), seed={bad: src}
            )


def test_missing_seed_file_raises_before_directory_appears(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(FileNotFoundError):
        create_bundle(
            runs, stage="t", task="t", contract=_contract(), seed={"a.md": tmp_path / "nope.md"}
        )
    assert not runs.exists() or not any(runs.iterdir())


def test_bundle_without_seed_restores_nothing(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path / "runs", stage="t", task="t", contract=_contract())
    assert restore_seed(bundle) == 0
