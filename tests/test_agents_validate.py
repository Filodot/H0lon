"""Output validation against the contract; problems must be readable Russian sentences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from h0lon.agents import ExpectedFile, OutputContract, validate_final_message, validate_outputs
from h0lon.agents.validate import check_contract, parse_json_message

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"const": True}, "language": {"const": "ru"}, "n": {"type": "integer"}},
    "required": ["ok", "language"],
    "additionalProperties": False,
}


def _contract(**md: object) -> OutputContract:
    return OutputContract(
        files=[
            ExpectedFile(path="result.md", kind="markdown", **md),  # type: ignore[arg-type]
            ExpectedFile(path="data.json", kind="json", json_schema=SCHEMA),
            ExpectedFile(path="extra.txt", kind="text", required=False),
        ]
    )


def _write(out: Path, name: str, content: str | bytes) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")


def test_valid_output_has_no_problems(tmp_path: Path) -> None:
    _write(tmp_path, "result.md", "# Проверка H0lon\n\nСтатус: OK\n")
    _write(tmp_path, "data.json", '{"ok": true, "language": "ru"}')
    contract = _contract(required_headings=["# Проверка H0lon"], required_text=["Статус: OK"])
    assert validate_outputs(tmp_path, contract) == []


def test_missing_required_file_and_optional_ok(tmp_path: Path) -> None:
    problems = validate_outputs(tmp_path, _contract())
    assert "out/result.md: обязательный файл не создан" in problems
    assert "out/data.json: обязательный файл не создан" in problems
    assert not any("extra.txt" in p for p in problems)


def test_utf8_min_chars_and_empty(tmp_path: Path) -> None:
    _write(tmp_path, "result.md", "Привет".encode("cp1251"))
    _write(tmp_path, "data.json", "   \n")
    problems = validate_outputs(tmp_path, _contract())
    assert any("не в кодировке UTF-8" in p for p in problems)
    assert "out/data.json: файл пустой" in problems

    _write(tmp_path, "result.md", "коротко")
    problems = validate_outputs(tmp_path, _contract(min_chars=50))
    assert any("слишком короткий (7 символов, нужно не меньше 50)" in p for p in problems)


def test_bom_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "result.md", "﻿# Заголовок\n")
    _write(tmp_path, "data.json", '﻿{"ok": true, "language": "ru"}')
    assert validate_outputs(tmp_path, _contract(required_headings=["# Заголовок"])) == []


def test_headings_outside_code_fences_only(tmp_path: Path) -> None:
    _write(tmp_path, "data.json", '{"ok": true, "language": "ru"}')
    _write(tmp_path, "result.md", "```\n# Итог\n```\nСтатус OK\n## Итог\n")
    problems = validate_outputs(
        tmp_path, _contract(required_headings=["# Итог", "## Итог"], required_text=["Статус: OK"])
    )
    assert any("нет обязательных заголовков: «# Итог»" in p for p in problems)
    assert not any("«## Итог»" in p for p in problems)
    assert "out/result.md: нет обязательного фрагмента «Статус: OK»" in problems


def test_json_parse_error(tmp_path: Path) -> None:
    _write(tmp_path, "result.md", "текст")
    _write(tmp_path, "data.json", "{not json")
    problems = validate_outputs(tmp_path, _contract())
    assert any(p.startswith("out/data.json: некорректный JSON (строка 1") for p in problems)


def test_json_schema_problems_in_russian(tmp_path: Path) -> None:
    _write(tmp_path, "result.md", "текст")
    _write(tmp_path, "data.json", json.dumps({"ok": False, "n": "три", "zzz": 1}))
    problems = validate_outputs(tmp_path, _contract())
    joined = "\n".join(problems)
    assert "нет обязательных полей: language" in joined
    assert "«$.ok»: ожидается значение true, получено false" in joined
    assert "«$.n»: ожидается целое число, получено строка" in joined
    assert "лишние поля: zzz" in joined


def test_broken_schema_reported(tmp_path: Path) -> None:
    _write(tmp_path, "x.json", "{}")
    contract = OutputContract(
        files=[ExpectedFile(path="x.json", kind="json", json_schema={"type": 12})]
    )
    problems = validate_outputs(tmp_path, contract)
    assert problems and "некорректная JSON Schema" in problems[0]


def test_final_message_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"enum": ["ok", "error"]}},
        "required": ["status"],
    }
    contract = OutputContract(files=[], final_message_schema=schema)
    assert validate_final_message(contract, final_text='```json\n{"status": "ok"}\n```')[0] == []
    probs, data = validate_final_message(contract, final_text='Итог: {"status": "ok"}')
    assert probs == [] and data == {"status": "ok"}
    probs, _ = validate_final_message(contract, final_text="готово")
    assert probs and "не является JSON" in probs[0]
    probs, _ = validate_final_message(contract, final_text="", structured={"status": "maybe"})
    assert probs and "не из допустимых" in probs[0]
    assert validate_final_message(OutputContract(files=[]), final_text="что угодно") == ([], None)


def test_parse_json_message_plain() -> None:
    assert parse_json_message(' {"a": 1} ') == {"a": 1}


def test_unresolvable_ref_is_a_problem_not_an_exception(tmp_path: Path) -> None:
    _write(tmp_path, "x.json", '{"a": 1}')
    contract = OutputContract(
        files=[ExpectedFile(path="x.json", kind="json", json_schema={"$ref": "#/definitions/no"})]
    )
    problems = validate_outputs(tmp_path, contract)
    assert len(problems) == 1 and "ошибка в контракте" in problems[0]
    # A final message schema that recurses forever is reported the same way.
    infinite = OutputContract(files=[], final_message_schema={"$ref": "#"})
    fp, _ = validate_final_message(infinite, final_text="{}")
    assert fp and fp[0].startswith("Финальное сообщение:")


def test_check_contract() -> None:
    good = OutputContract(
        files=[
            ExpectedFile(
                path="d.json",
                kind="json",
                json_schema={
                    "$defs": {"node": {"type": "object", "properties": {"c": {"$ref": "#"}}}},
                    "properties": {"n": {"$ref": "#/$defs/node"}},
                },
            ),
            ExpectedFile(path="r.md", kind="markdown"),
        ],
        final_message_schema=SCHEMA,
    )
    assert check_contract(good) == []
    bad = OutputContract(
        files=[
            ExpectedFile(path="a.json", kind="json", json_schema={"type": 12}),
            ExpectedFile(path="b.json", kind="json", json_schema={"$ref": "#/definitions/nope"}),
        ],
        final_message_schema={"properties": {"x": {"$ref": "https://example.com/s.json"}}},
    )
    problems = check_contract(bad)
    assert len(problems) == 3
    assert problems[0].startswith("out/a.json: некорректная JSON Schema")
    assert "out/b.json" in problems[1] and "«#/definitions/nope»" in problems[1]
    assert problems[2].startswith("Финальное сообщение:")
