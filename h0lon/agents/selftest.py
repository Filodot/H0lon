"""`h0lon agent-test`: a tiny end-to-end task that exercises bundle → agent → validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from h0lon.agents.base import Tier
from h0lon.agents.bundle import create_bundle, save_bundle
from h0lon.agents.contract import ExpectedFile, OutputContract
from h0lon.agents.runner import RunResult, run_task
from h0lon.config import Settings

RESULT_HEADING = "# Проверка H0lon"
RESULT_LINE = "Статус: OK"

DATA_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ok": {"const": True},
        "language": {"const": "ru"},
    },
    "required": ["ok", "language"],
    "additionalProperties": False,
}

# Strict-mode compatible (all properties required, no extra keys) so Codex accepts it.
FINAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "error"]},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "files"],
    "additionalProperties": False,
}


def selftest_runs_dir(settings: Settings) -> Path:
    return settings.general.state_path / "selftest" / "runs"


def _task_text(prompt: str | None, has_images: bool) -> str:
    lines = [
        "Это проверочный прогон headless-агента H0lon: нужно убедиться, что агент запускается,",
        "читает задание и записывает файлы строго по контракту вывода.",
        "",
        "1. Создай файл `out/result.md`. Первая строка — заголовок `# Проверка H0lon`, затем",
        "   отдельная строка `Статус: OK`, затем одно предложение о том, что ты сделал.",
        "2. Создай файл `out/data.json` ровно с таким содержимым:",
        '   `{"ok": true, "language": "ru"}`.',
    ]
    step = 3
    if prompt:
        lines += [
            f"{step}. Выполни дополнительное задание пользователя (текст ниже) и запиши ответ",
            "   на русском языке в `out/answer.md`.",
        ]
        step += 1
    if has_images:
        lines += [
            f"{step}. Рассмотри каждое изображение из раздела «Входные данные» и запиши в",
            "   `out/images.md` для каждого раздел с заголовком `## <имя файла>`: что изображено",
            "   и весь различимый текст дословно.",
        ]
    lines += [
        "",
        "В финальном сообщении перечисли созданные файлы (пути относительно `out/`).",
    ]
    if prompt:
        lines += ["", "## Дополнительное задание пользователя", "", prompt.strip()]
    return "\n".join(lines) + "\n"


def agent_test(
    settings: Settings,
    *,
    prompt: str | None = None,
    backend: str | None = None,
    fallback: bool = True,
    tier: Tier = "light",
    images: Sequence[Path] = (),
    on_event: Callable[[str], None] | None = None,
) -> RunResult:
    files = [
        ExpectedFile(
            path="result.md",
            kind="markdown",
            required_headings=[RESULT_HEADING],
            required_text=[RESULT_LINE],
            min_chars=len(RESULT_HEADING) + len(RESULT_LINE) + 2,
            description="Отчёт о проверке.",
        ),
        ExpectedFile(
            path="data.json",
            kind="json",
            json_schema=DATA_SCHEMA,
            description="Машиночитаемый результат проверки.",
        ),
    ]
    if prompt:
        files.append(
            ExpectedFile(
                path="answer.md",
                kind="markdown",
                min_chars=1,
                description="Ответ на дополнительное задание пользователя.",
            )
        )
    contract = OutputContract(files=files, final_message_schema=FINAL_SCHEMA)
    bundle = create_bundle(
        selftest_runs_dir(settings),
        stage="selftest",
        task=_task_text(prompt, bool(images)),
        contract=contract,
        images=list(images),
    )
    if bundle.images:
        # Headings use the names the images got inside inputs/ (collisions are renamed).
        bundle.contract.files.append(
            ExpectedFile(
                path="images.md",
                kind="markdown",
                required_headings=[f"## {p.name}" for p in bundle.images],
                min_chars=20 * len(bundle.images),
                description="Описание каждого изображения.",
            )
        )
        save_bundle(bundle)
    return run_task(
        bundle,
        settings=settings,
        tier=tier,
        backend=backend,
        fallback=fallback,
        on_event=on_event,
    )
