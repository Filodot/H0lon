"""Task bundles (runs/<id>/) and the backend-independent prompt."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from h0lon.agents.contract import OutputContract, contract_prompt_section
from h0lon.agents.validate import check_contract

BUNDLE_FILE = "bundle.json"
SCHEMA_FILE = "schema.json"
BUNDLE_FORMAT = 1
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"

_HTML_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)
_VERSION = re.compile(r"<!--\s*h0lon:system@([0-9][\w.\-]*)")


@dataclass
class TaskBundle:
    id: str
    root: Path
    stage: str
    task_path: Path
    inputs_dir: Path
    out_dir: Path
    images: list[Path]
    workspace: Path | None
    contract: OutputContract

    @property
    def attempts_dir(self) -> Path:
        return self.root / "attempts"

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def schema_path(self) -> Path | None:
        """schema.json with the final message schema (written by save_bundle), if any."""
        return self.root / SCHEMA_FILE if self.contract.final_message_schema else None

    def input_files(self) -> list[Path]:
        """Top-level entries of inputs/ that are not images."""
        if not self.inputs_dir.is_dir():
            return []
        images = {p.resolve() for p in self.images}
        return sorted(p for p in self.inputs_dir.iterdir() if p.resolve() not in images)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root": str(self.root),
            "stage": self.stage,
            "task": str(self.task_path),
            "inputs_dir": str(self.inputs_dir),
            "out_dir": str(self.out_dir),
            "images": [str(p) for p in self.images],
            "workspace": str(self.workspace) if self.workspace else None,
            "contract": self.contract.to_dict(),
        }


def write_text(path: Path, text: str) -> None:
    """UTF-8 with LF line endings (agents and git prefer them on every OS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def new_bundle_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _unique_target(directory: Path, name: str, taken: set[str]) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, i = name, 1
    while candidate.lower() in taken or (directory / candidate).exists():
        i += 1
        candidate = f"{stem}-{i}{suffix}"
    taken.add(candidate.lower())
    return directory / candidate


def safe_image_name(name: str) -> str:
    """File name for an image copy: `codex --image` splits its value on commas."""
    return name.replace(",", "_")


def _copy_in(src: Path, inputs_dir: Path, taken: set[str], *, image: bool = False) -> Path:
    name = safe_image_name(src.name) if image else src.name
    target = _unique_target(inputs_dir, name, taken)
    if src.is_dir():
        shutil.copytree(src, target)
    else:
        shutil.copy2(src, target)
    return target.resolve()


def _rel(bundle_root: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(bundle_root.resolve()).as_posix()
    except ValueError:
        return str(p)


def save_bundle(bundle: TaskBundle, *, created_at: str | None = None) -> None:
    """Write bundle.json (and schema.json when the contract has a final message schema)."""
    meta_path = bundle.root / BUNDLE_FILE
    previous: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    meta = {
        "format": BUNDLE_FORMAT,
        "id": bundle.id,
        "stage": bundle.stage,
        "created_at": created_at or previous.get("created_at") or _now_iso(),
        "task": _rel(bundle.root, bundle.task_path),
        "inputs": [_rel(bundle.root, p) for p in bundle.input_files()],
        "images": [_rel(bundle.root, p) for p in bundle.images],
        "out": _rel(bundle.root, bundle.out_dir),
        "workspace": str(bundle.workspace) if bundle.workspace else None,
        "contract": bundle.contract.to_dict(),
    }
    write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    schema = bundle.root / SCHEMA_FILE
    if bundle.contract.final_message_schema:
        write_text(
            schema,
            json.dumps(bundle.contract.final_message_schema, ensure_ascii=False, indent=2) + "\n",
        )
    elif schema.exists():
        schema.unlink()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_bundle(
    runs_dir: Path,
    *,
    stage: str,
    task: str,
    contract: OutputContract,
    inputs: Sequence[Path] = (),
    images: Sequence[Path] = (),
    workspace: Path | None = None,
    bundle_id: str | None = None,
) -> TaskBundle:
    """Create runs/<id>/ with task.md, inputs/ (copies), out/ and bundle.json.

    Everything is checked before the directory appears: a missing input raises
    FileNotFoundError, a broken contract schema or a non-file image raises ValueError; a
    failure while copying removes the half-built directory.
    """
    contract_problems = check_contract(contract)
    if contract_problems:
        raise ValueError("Некорректный контракт вывода: " + "; ".join(contract_problems))
    input_srcs = [Path(src).expanduser() for src in inputs]
    image_srcs = [Path(src).expanduser() for src in images]
    missing = [src for src in (*input_srcs, *image_srcs) if not src.exists()]
    if missing:
        label = "Входной файл не найден" if len(missing) == 1 else "Входные файлы не найдены"
        raise FileNotFoundError(f"{label}: " + ", ".join(str(p) for p in missing))
    not_files = [src for src in image_srcs if not src.is_file()]
    if not_files:
        raise ValueError("Изображение должно быть файлом: " + ", ".join(map(str, not_files)))

    bundle_id = bundle_id or new_bundle_id()
    root = Path(runs_dir).expanduser().resolve() / bundle_id
    if root.exists():
        raise FileExistsError(f"Каталог задания уже существует: {root}")
    inputs_dir, out_dir = root / "inputs", root / "out"
    inputs_dir.mkdir(parents=True)
    try:
        out_dir.mkdir()
        task_path = root / "task.md"
        write_text(task_path, task if task.endswith("\n") else task + "\n")

        taken: set[str] = set()
        for src in input_srcs:
            _copy_in(src, inputs_dir, taken)
        copied_images = [_copy_in(src, inputs_dir, taken, image=True) for src in image_srcs]

        bundle = TaskBundle(
            id=bundle_id,
            root=root,
            stage=stage,
            task_path=task_path,
            inputs_dir=inputs_dir,
            out_dir=out_dir,
            images=copied_images,
            workspace=Path(workspace).expanduser().resolve() if workspace else None,
            contract=contract,
        )
        save_bundle(bundle, created_at=_now_iso())
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return bundle


def load_bundle(root: Path) -> TaskBundle:
    """Reopen a bundle from its bundle.json."""
    root = Path(root).resolve()
    meta = json.loads((root / BUNDLE_FILE).read_text(encoding="utf-8"))
    return TaskBundle(
        id=meta["id"],
        root=root,
        stage=meta["stage"],
        task_path=root / meta.get("task", "task.md"),
        inputs_dir=root / "inputs",
        out_dir=root / meta.get("out", "out"),
        images=[(root / p).resolve() for p in meta.get("images") or []],
        workspace=Path(meta["workspace"]) if meta.get("workspace") else None,
        contract=OutputContract.from_dict(meta["contract"]),
    )


# ---------------------------------------------------------------- prompt


def system_prompt_raw() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def system_prompt_text() -> str:
    """System prompt without HTML comments (for backends that get it inline)."""
    return _HTML_COMMENT.sub("", system_prompt_raw()).strip()


def system_prompt_version() -> str | None:
    m = _VERSION.search(system_prompt_raw())
    return m.group(1) if m else None


@dataclass
class Feedback:
    """Problems of the previous attempt, shown to the agent on a retry."""

    problems: list[str]
    attempt_label: str | None = None  # e.g. "2-claude"
    previous_out: Path | None = None  # where the previous out/ was moved
    extra: list[str] = field(default_factory=list)


def _feedback_section(bundle: TaskBundle, fb: Feedback) -> str:
    who = f" ({fb.attempt_label})" if fb.attempt_label else ""
    lines = [
        "# Обратная связь по предыдущей попытке",
        "",
        f"Предыдущая попытка{who} не прошла автоматическую проверку. Проблемы:",
        "",
    ]
    lines += [f"- {p}" for p in fb.problems] or ["- (подробности не сохранились)"]
    lines.append("")
    lines += fb.extra
    note = "Каталог `out/` очищен."
    if fb.previous_out is not None and fb.previous_out.exists():
        note += (
            f" Файлы прошлой попытки сохранены в `{_rel(bundle.root, fb.previous_out)}/` — "
            "их можно взять за основу, но итог нужно заново записать в `out/`."
        )
    lines.append(note)
    lines.append("Исправь все перечисленные проблемы и создай в `out/` все файлы из контракта.")
    return "\n".join(lines)


def build_prompt(
    bundle: TaskBundle, *, feedback: Feedback | None = None, include_system: bool = False
) -> str:
    """The single prompt every backend receives (system text inline only when asked)."""
    parts: list[str] = []
    if include_system:
        parts.append(system_prompt_text())
        parts.append("---")
    task = bundle.task_path.read_text(encoding="utf-8").strip()
    parts.append("# Задание\n\n" + task)

    inp = ["# Входные данные", ""]
    inp.append(f"Каталог задания (текущий рабочий каталог): `{bundle.root}`")
    files = bundle.input_files()
    if files:
        inp += ["", "Входные файлы (только чтение):"]
        inp += [f"- `{p}`" for p in files]
    if bundle.images:
        inp += [
            "",
            "Изображения (рассмотри каждое; если изображение не приложено к сообщению, "
            "открой файл инструментом чтения):",
        ]
        inp += [f"- `{p}`" for p in bundle.images]
    if bundle.workspace:
        inp += ["", f"Папка темы (только чтение): `{bundle.workspace}`"]
    if not files and not bundle.images and not bundle.workspace:
        inp += ["", "Входных файлов нет — всё необходимое есть в задании."]
    parts.append("\n".join(inp))

    parts.append(contract_prompt_section(bundle.contract, out_dir_abs=str(bundle.out_dir)))
    if feedback is not None:
        parts.append(_feedback_section(bundle, feedback))
    return "\n\n".join(parts).rstrip() + "\n"
