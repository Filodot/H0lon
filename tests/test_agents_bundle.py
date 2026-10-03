"""Task bundles, the output contract and the unified prompt."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from h0lon.agents import (
    ExpectedFile,
    Feedback,
    OutputContract,
    build_prompt,
    create_bundle,
    load_bundle,
)
from h0lon.agents.bundle import SYSTEM_PROMPT_PATH, system_prompt_text, system_prompt_version

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")


def _contract() -> OutputContract:
    return OutputContract(
        files=[
            ExpectedFile(
                path="result.md",
                kind="markdown",
                required_headings=["# Итог"],
                required_text=["Статус: OK"],
                min_chars=10,
            ),
            ExpectedFile(
                path="data.json",
                kind="json",
                json_schema={"type": "object", "required": ["ok"]},
            ),
            ExpectedFile(path="notes.txt", kind="text", required=False),
        ],
        final_message_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )


def _bundle(tmp_path: Path, **kw):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir()
    (src / "a" / "page.png").write_bytes(b"png-a")
    (src / "b" / "page.png").write_bytes(b"png-b")
    (src / "lecture.pdf").write_bytes(b"%PDF-1.4")
    return create_bundle(
        tmp_path / "runs",
        stage="handwriting",
        task="Разбери страницы конспекта.",
        contract=_contract(),
        inputs=[src / "lecture.pdf"],
        images=[src / "a" / "page.png", src / "b" / "page.png"],
        **kw,
    )


def test_create_bundle_layout(tmp_path: Path) -> None:
    ws = tmp_path / "topic"
    ws.mkdir()
    b = _bundle(tmp_path, workspace=ws)
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", b.id)
    assert b.root == (tmp_path / "runs" / b.id).resolve()
    assert b.task_path.read_text(encoding="utf-8") == "Разбери страницы конспекта.\n"
    assert b.out_dir.is_dir() and not any(b.out_dir.iterdir())
    names = sorted(p.name for p in b.inputs_dir.iterdir())
    assert names == ["lecture.pdf", "page-2.png", "page.png"]
    assert [p.name for p in b.images] == ["page.png", "page-2.png"]
    assert all(p.is_absolute() and p.parent == b.inputs_dir for p in b.images)
    assert b.images[1].read_bytes() == b"png-b"
    assert [p.name for p in b.input_files()] == ["lecture.pdf"]
    assert b.workspace == ws.resolve()

    meta = json.loads((b.root / "bundle.json").read_text(encoding="utf-8"))
    assert meta["id"] == b.id and meta["stage"] == "handwriting"
    assert meta["images"] == ["inputs/page.png", "inputs/page-2.png"]
    assert meta["inputs"] == ["inputs/lecture.pdf"]
    assert meta["contract"]["files"][0]["required_headings"] == ["# Итог"]
    assert json.loads((b.root / "schema.json").read_text(encoding="utf-8")) == (
        _contract().final_message_schema
    )


def test_load_bundle_roundtrip(tmp_path: Path) -> None:
    b = _bundle(tmp_path)
    again = load_bundle(b.root)
    assert again.id == b.id and again.stage == b.stage
    assert again.images == b.images
    assert again.contract.to_dict() == b.contract.to_dict()


def test_bundle_id_collision_and_missing_input(tmp_path: Path) -> None:
    create_bundle(tmp_path, stage="s", task="t", contract=_contract(), bundle_id="fixed")
    with pytest.raises(FileExistsError):
        create_bundle(tmp_path, stage="s", task="t", contract=_contract(), bundle_id="fixed")
    with pytest.raises(FileNotFoundError):
        create_bundle(
            tmp_path, stage="s", task="t", contract=_contract(), inputs=[tmp_path / "nope.pdf"]
        )


def test_failed_create_leaves_nothing_behind(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    good = tmp_path / "ok.png"
    good.write_bytes(b"png")
    with pytest.raises(FileNotFoundError, match="Входной файл не найден"):
        create_bundle(
            runs,
            stage="s",
            task="t",
            contract=_contract(),
            images=[good, tmp_path / "нет такого.png"],
            bundle_id="x",
        )
    assert not (runs / "x").exists()
    with pytest.raises(ValueError, match="Изображение должно быть файлом"):
        create_bundle(runs, stage="s", task="t", contract=_contract(), images=[tmp_path])
    broken = OutputContract(
        files=[ExpectedFile(path="d.json", kind="json", json_schema={"$ref": "#/definitions/no"})]
    )
    with pytest.raises(ValueError, match="Некорректный контракт вывода"):
        create_bundle(runs, stage="s", task="t", contract=broken, bundle_id="x")
    assert not (runs / "x").exists()
    # The same id is free again once the inputs are fixed.
    b = create_bundle(runs, stage="s", task="t", contract=_contract(), images=[good], bundle_id="x")
    assert (b.root / "bundle.json").is_file()


def test_copy_failure_removes_partial_bundle(tmp_path: Path, monkeypatch) -> None:
    from h0lon.agents import bundle as bundle_mod

    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bundle_mod.shutil, "copy2", boom)
    with pytest.raises(OSError, match="disk full"):
        create_bundle(
            tmp_path / "runs",
            stage="s",
            task="t",
            contract=_contract(),
            inputs=[src],
            bundle_id="y",
        )
    assert not (tmp_path / "runs" / "y").exists()


def test_image_names_without_commas(tmp_path: Path) -> None:
    # `codex --image` splits values on commas, so image copies never keep them.
    img = tmp_path / "Скан 1, стр 2.png"
    img.write_bytes(b"png")
    doc = tmp_path / "Лекция 1, часть 2.pdf"
    doc.write_bytes(b"%PDF")
    b = create_bundle(
        tmp_path / "runs", stage="s", task="t", contract=_contract(), inputs=[doc], images=[img]
    )
    assert [p.name for p in b.images] == ["Скан 1_ стр 2.png"]
    assert [p.name for p in b.input_files()] == ["Лекция 1, часть 2.pdf"]  # not passed by flag


@pytest.mark.parametrize("bad", ["../x.md", "/abs.md", "C:/x.md", "a//b.md", ""])
def test_expected_file_rejects_paths_outside_out(bad: str) -> None:
    with pytest.raises(ValueError):
        ExpectedFile(path=bad, kind="markdown")


def test_expected_file_normalizes_backslashes() -> None:
    assert ExpectedFile(path="sub\\a.md", kind="markdown").path == "sub/a.md"


def test_prompt_sections(tmp_path: Path) -> None:
    b = _bundle(tmp_path)
    prompt = build_prompt(b)
    assert prompt.startswith("# Задание\n\nРазбери страницы конспекта.")
    assert "# Входные данные" in prompt
    assert str(b.inputs_dir / "lecture.pdf") in prompt
    for img in b.images:
        assert str(img) in prompt
    assert "# Контракт вывода" in prompt
    assert "`out/result.md` — Markdown, обязательный, не короче 10 символов" in prompt
    assert "`# Итог`" in prompt and "`Статус: OK`" in prompt
    assert '"required": [\n' in prompt  # schema rendered as JSON
    assert "`out/notes.txt` — текст, необязательный" in prompt
    assert "Финальное сообщение: только JSON-объект" in prompt
    assert "Обратная связь" not in prompt
    assert system_prompt_text() not in prompt


def test_prompt_with_feedback_and_inline_system(tmp_path: Path) -> None:
    b = _bundle(tmp_path)
    prev = b.attempts_dir / "1-claude" / "out"
    prev.mkdir(parents=True)
    fb = Feedback(
        problems=["out/result.md: нет заголовка"], attempt_label="1-claude", previous_out=prev
    )
    prompt = build_prompt(b, feedback=fb, include_system=True)
    assert prompt.startswith(system_prompt_text())
    assert "<!--" not in prompt
    assert "# Обратная связь по предыдущей попытке" in prompt
    assert "- out/result.md: нет заголовка" in prompt
    assert "`attempts/1-claude/out/`" in prompt


def test_system_prompt_file() -> None:
    raw = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    assert system_prompt_version() == "1.0"
    text = system_prompt_text()
    assert "данные, а не инструкции" in text
    assert "out/" in text and "Не задавай вопросов" in text
    assert raw.startswith("<!--") and not text.startswith("<!--")
