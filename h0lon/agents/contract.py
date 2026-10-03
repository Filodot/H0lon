"""Output contract of an agent task: which files must appear in out/ and how they are checked."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

FileKind = Literal["markdown", "json", "text"]
_KINDS = ("markdown", "json", "text")
_KIND_RU = {"markdown": "Markdown", "json": "JSON", "text": "текст"}


@dataclass
class ExpectedFile:
    path: str  # relative to out/, forward slashes
    kind: FileKind
    required: bool = True
    min_chars: int = 1
    required_headings: list[str] = field(default_factory=list)  # markdown: exact heading lines
    json_schema: dict | None = None  # json: JSON Schema of the file content
    # Extension of the ARCHITECTURE contract: exact fragments that must occur in the text.
    required_text: list[str] = field(default_factory=list)
    description: str = ""  # what the file is for (shown to the agent)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Неизвестный тип файла «{self.kind}» (ожидается {', '.join(_KINDS)})")
        normalized = self.path.replace("\\", "/").strip()
        pure = PurePosixPath(normalized)
        if (
            not normalized
            or pure.is_absolute()
            or ":" in normalized
            or any(part in ("..", "") for part in normalized.split("/"))
        ):
            raise ValueError(
                f"Путь файла контракта должен быть относительным внутри out/: {self.path}"
            )
        self.path = str(pure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "required": self.required,
            "min_chars": self.min_chars,
            "required_headings": list(self.required_headings),
            "json_schema": self.json_schema,
            "required_text": list(self.required_text),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedFile:
        return cls(
            path=data["path"],
            kind=data["kind"],
            required=bool(data.get("required", True)),
            min_chars=int(data.get("min_chars", 1)),
            required_headings=list(data.get("required_headings") or []),
            json_schema=data.get("json_schema"),
            required_text=list(data.get("required_text") or []),
            description=data.get("description") or "",
        )


@dataclass
class OutputContract:
    files: list[ExpectedFile]
    final_message_schema: dict | None = None  # JSON Schema of the agent's final message

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [f.to_dict() for f in self.files],
            "final_message_schema": self.final_message_schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputContract:
        return cls(
            files=[ExpectedFile.from_dict(f) for f in data.get("files") or []],
            final_message_schema=data.get("final_message_schema"),
        )


def _json_block(data: Any) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"


def contract_prompt_section(contract: OutputContract, *, out_dir_abs: str | None = None) -> str:
    """Russian description of the contract for the prompt (section «Контракт вывода»)."""
    lines = ["# Контракт вывода", ""]
    where = f" (абсолютный путь: `{out_dir_abs}`)" if out_dir_abs else ""
    lines.append(
        f"Результат — файлы в каталоге `out/`{where}. Создай следующие файлы в кодировке UTF-8:"
    )
    lines.append("")
    for i, ef in enumerate(contract.files, 1):
        need = "обязательный" if ef.required else "необязательный"
        head = f"{i}. `out/{ef.path}` — {_KIND_RU[ef.kind]}, {need}"
        if ef.min_chars > 1:
            head += f", не короче {ef.min_chars} символов"
        head += "."
        if ef.description:
            head += f" {ef.description}"
        lines.append(head)
        if ef.required_headings:
            lines.append("   Обязательные заголовки (строки должны совпадать точно):")
            lines += [f"   - `{h}`" for h in ef.required_headings]
        if ef.required_text:
            lines.append("   Обязательные фрагменты текста (точно, с учётом регистра):")
            lines += [f"   - `{t}`" for t in ef.required_text]
        if ef.kind == "json":
            lines.append(
                "   Чистый JSON без комментариев и без Markdown-обёртки ```."
                + (" Должен проходить JSON Schema:" if ef.json_schema else "")
            )
            if ef.json_schema:
                lines += ["   " + ln for ln in _json_block(ef.json_schema).splitlines()]
    lines.append("")
    lines.append("Не создавай в `out/` других файлов и не пиши никуда, кроме `out/`.")
    lines.append("")
    if contract.final_message_schema:
        lines.append(
            "Финальное сообщение: только JSON-объект (без пояснений и без обёртки ```), "
            "соответствующий схеме:"
        )
        lines.append(_json_block(contract.final_message_schema))
    else:
        lines.append("Финальное сообщение: одно-два предложения о том, что сделано.")
    return "\n".join(lines)
