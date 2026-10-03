"""Validation of agent output against an OutputContract. Problems are Russian sentences."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import SchemaError, ValidationError

from h0lon.agents.contract import ExpectedFile, OutputContract

MAX_SCHEMA_ERRORS = 8

_TYPE_RU = {
    "object": "объект",
    "array": "массив",
    "string": "строка",
    "integer": "целое число",
    "number": "число",
    "boolean": "логическое значение",
    "null": "null",
}


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _dump(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:79] + "…"


def _location(err: ValidationError) -> str:
    loc = "$"
    for part in err.absolute_path:
        loc += f"[{part}]" if isinstance(part, int) else f".{part}"
    return loc


def schema_error_ru(err: ValidationError) -> str:
    """Human-readable Russian description of one jsonschema error."""
    v, vv, inst = err.validator, err.validator_value, err.instance
    where = _location(err)
    if v == "required" and isinstance(inst, dict):
        missing = [k for k in vv if k not in inst]
        return f"в «{where}» нет обязательных полей: {', '.join(missing)}"
    if v == "type":
        expected = vv if isinstance(vv, list) else [vv]
        exp = " или ".join(_TYPE_RU.get(t, str(t)) for t in expected)
        got = _TYPE_RU.get(_json_type(inst), _json_type(inst))
        return f"«{where}»: ожидается {exp}, получено {got}"
    if v == "const":
        return f"«{where}»: ожидается значение {_dump(vv)}, получено {_dump(inst)}"
    if v == "enum":
        allowed = ", ".join(_dump(x) for x in vv)
        return f"«{where}»: значение {_dump(inst)} не из допустимых ({allowed})"
    if v == "additionalProperties" and isinstance(inst, dict):
        known = set((err.schema or {}).get("properties", {}))
        extra = [k for k in inst if k not in known]
        return f"«{where}»: лишние поля: {', '.join(extra) or '?'}"
    if v in ("minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"):
        names = {
            "minLength": "минимальная длина",
            "maxLength": "максимальная длина",
            "minItems": "минимум элементов",
            "maxItems": "максимум элементов",
            "minimum": "минимум",
            "maximum": "максимум",
        }
        return f"«{where}»: нарушено ограничение «{names[v]} = {vv}» (значение {_dump(inst)})"
    if v == "pattern":
        return f"«{where}»: строка {_dump(inst)} не соответствует шаблону {vv}"
    return f"«{where}»: {err.message}"


def _exc_text(exc: BaseException, limit: int = 200) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


_INSTANCE_KEYWORDS = ("const", "enum", "default", "examples")


def _collect_refs(schema: dict) -> tuple[list[str], bool]:
    """All `$ref` strings of a schema, and whether a nested `$id` changes their base URI."""
    refs: list[str] = []
    nested_id = False

    def walk(node: Any, top: bool) -> None:
        nonlocal nested_id
        if isinstance(node, dict):
            if not top and isinstance(node.get("$id"), str):
                nested_id = True
            ref = node.get("$ref")
            if isinstance(ref, str) and ref not in refs:
                refs.append(ref)
            for key, value in node.items():
                if key not in _INSTANCE_KEYWORDS:  # those hold instance data, not subschemas
                    walk(value, False)
        elif isinstance(node, list):
            for value in node:
                walk(value, False)

    walk(schema, True)
    return refs, nested_id


def schema_check(schema: Any, *, label: str) -> list[str]:
    """Problems of a contract JSON Schema itself: invalid schema, unresolvable `$ref`."""
    if not isinstance(schema, dict):
        return [f"{label}: JSON Schema в контракте должна быть объектом"]
    try:
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
    except SchemaError as exc:
        return [f"{label}: некорректная JSON Schema в контракте ({exc.message})"]
    except Exception as exc:  # e.g. an unknown $schema dialect
        return [f"{label}: некорректная JSON Schema в контракте ({_exc_text(exc)})"]
    refs, nested_id = _collect_refs(schema)
    if nested_id:
        return []  # refs are relative to nested $id scopes; checked at validation time
    validator = cls(schema)
    problems: list[str] = []
    for ref in refs:
        try:
            # Resolves the reference against the root schema (the instance is irrelevant).
            for _ in validator.descend(None, {"$ref": ref}):
                pass
        except Exception as exc:
            problems.append(
                f"{label}: в JSON Schema контракта неразрешимая ссылка «{ref}» ({_exc_text(exc)})"
            )
    return problems


def check_contract(contract: OutputContract) -> list[str]:
    """Validate the contract's own schemas once, before any agent runs."""
    problems: list[str] = []
    for ef in contract.files:
        if ef.kind == "json" and ef.json_schema is not None:
            problems += schema_check(ef.json_schema, label=f"out/{ef.path}")
    if contract.final_message_schema is not None:
        problems += schema_check(contract.final_message_schema, label="Финальное сообщение")
    return problems


def schema_problems(data: Any, schema: dict, *, label: str) -> list[str]:
    """Validate `data` against `schema`; return Russian problems prefixed with `label`.

    Never raises: a broken contract schema becomes a problem string as well.
    """
    try:
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
    except SchemaError as exc:
        return [f"{label}: некорректная JSON Schema в контракте ({exc.message})"]
    except Exception as exc:
        return [f"{label}: некорректная JSON Schema в контракте ({_exc_text(exc)})"]
    try:
        validator = cls(schema)
        errors = sorted(
            validator.iter_errors(data), key=lambda e: [str(p) for p in e.absolute_path]
        )
    except Exception as exc:  # unresolvable $ref, infinite recursion, …
        return [
            f"{label}: не удалось применить JSON Schema контракта ({_exc_text(exc)}) — "
            "ошибка в контракте, а не в результате агента"
        ]
    problems = [f"{label}: {schema_error_ru(e)}" for e in errors[:MAX_SCHEMA_ERRORS]]
    if len(errors) > MAX_SCHEMA_ERRORS:
        problems.append(f"{label}: … и ещё {len(errors) - MAX_SCHEMA_ERRORS} нарушений схемы")
    return problems


_FENCE = re.compile(r"^\s*(```|~~~)")


def markdown_heading_lines(text: str) -> set[str]:
    """Heading lines (ATX `#`…) outside fenced code blocks, stripped."""
    headings: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and line.lstrip().startswith("#"):
            headings.add(line.strip())
    return headings


def _decode(path: Path, label: str) -> tuple[str | None, list[str]]:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError as exc:
        return None, [f"{label}: файл не в кодировке UTF-8 (ошибка в байте {exc.start})"]


def validate_file(out_dir: Path, ef: ExpectedFile) -> list[str]:
    label = f"out/{ef.path}"
    path = out_dir / ef.path
    if not path.exists():
        return [f"{label}: обязательный файл не создан"] if ef.required else []
    if not path.is_file():
        return [f"{label}: ожидался файл, а найден каталог"]
    text, problems = _decode(path, label)
    if text is None:
        return problems
    stripped = text.strip()
    if len(stripped) < max(ef.min_chars, 1):
        if not stripped:
            return [f"{label}: файл пустой"]
        problems.append(
            f"{label}: слишком короткий ({len(stripped)} символов, нужно не меньше {ef.min_chars})"
        )
    if ef.kind == "markdown" and ef.required_headings:
        present = markdown_heading_lines(text)
        missing = [h for h in ef.required_headings if h.strip() not in present]
        if missing:
            problems.append(
                f"{label}: нет обязательных заголовков: " + "; ".join(f"«{h}»" for h in missing)
            )
    for fragment in ef.required_text:
        if fragment not in text:
            problems.append(f"{label}: нет обязательного фрагмента «{fragment}»")
    if ef.kind == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(
                f"{label}: некорректный JSON (строка {exc.lineno}, столбец {exc.colno}: {exc.msg})"
            )
            return problems
        if ef.json_schema:
            problems += schema_problems(data, ef.json_schema, label=label)
    return problems


def validate_outputs(out_dir: Path, contract: OutputContract) -> list[str]:
    problems: list[str] = []
    for ef in contract.files:
        problems += validate_file(out_dir, ef)
    return problems


_CODE_FENCE_JSON = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse_json_message(text: str) -> Any:
    """Parse the agent's final message as JSON (tolerating a ```json fence). Raises ValueError."""
    candidate = text.strip()
    m = _CODE_FENCE_JSON.match(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("финальное сообщение не является JSON")


def validate_final_message(
    contract: OutputContract, *, final_text: str, structured: Any = None
) -> tuple[list[str], Any]:
    """Check the final message against `final_message_schema`. Returns (problems, parsed)."""
    schema = contract.final_message_schema
    if not schema:
        return [], None
    label = "Финальное сообщение"
    data = structured
    if data is None:
        if not final_text.strip():
            return [f"{label}: пустое, а требуется JSON по схеме контракта"], None
        try:
            data = parse_json_message(final_text)
        except ValueError:
            return [
                f"{label}: не является JSON-объектом, а требуется JSON по схеме контракта"
            ], None
    return schema_problems(data, schema, label=label), data
