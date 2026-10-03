"""Topic workspaces: <workspaces>/<course>/<topic>/ with topic.yaml, standard dirs and git."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from h0lon import procutil, tools
from h0lon.config import Settings
from h0lon.names import slugify

TOPIC_DIRS = ("sources", "extracted", "synthesis", "variants", "runs")
TOPIC_FILE = "topic.yaml"
INITIAL_COMMIT_MESSAGE = "Тема создана"
GITIGNORE_PATTERNS = ("*.mp4", "*.mkv", "*.webm", "*.m4a", "*.wav", "*.mp3")

_TOPIC_YAML_HEADER = (
    "# Метаданные темы H0lon. Файл ведёт приложение; review_gate: null — как в настройках.\n"
)
_GITIGNORE_TEXT = (
    "# Видео и аудио не хранятся в истории темы: файлы слишком большие.\n"
    + "\n".join(GITIGNORE_PATTERNS)
    + "\n"
)
# A parent process (git hook, IDE) may point git at another repository through these.
_GIT_ENV_DROP = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_NAMESPACE",
)
_GIT_TIMEOUT_S = 60


class TopicWarning(UserWarning):
    """Non-fatal problem while creating a topic (e.g. the first git commit failed)."""


class TopicMeta(BaseModel):
    """Contents of topic.yaml. Unknown keys written by later stages are preserved."""

    model_config = ConfigDict(extra="allow")

    title: str
    course: str
    slug: str
    language: str = "ru"
    created: str  # ISO 8601, UTC
    sources: list[dict[str, Any]] = Field(default_factory=list)
    review_gate: bool | None = None  # None = inherit general.review_gate
    backends: dict[str, str] = Field(default_factory=dict)  # stage -> claude | codex

    @field_validator("created", mode="before")
    @classmethod
    def _created_to_iso(cls, v: object) -> object:
        # Hand-edited YAML may contain an unquoted timestamp that PyYAML parses itself.
        if isinstance(v, datetime | date):
            return v.isoformat()
        return v


@dataclass
class TopicCreation:
    path: Path
    meta: TopicMeta
    git_repo: bool = False
    committed: bool = False
    warnings: list[str] = field(default_factory=list)


def topic_path(settings: Settings, *, title: str, course: str, slug: str | None = None) -> Path:
    """Where `create_topic` puts the topic: <workspaces>/<slug(course)>/<slug or slug(title)>."""
    topic_slug = slugify(slug) if slug else slugify(title)
    return settings.general.workspaces_dir / slugify(course) / topic_slug


def create_topic(settings: Settings, *, title: str, course: str, slug: str | None = None) -> Path:
    """Create a topic workspace and return its path.

    Raises FileExistsError if the directory exists and is not empty. Non-fatal problems
    (no git, failed first commit) are emitted as `TopicWarning`; use
    `create_topic_detailed` to get them as a list instead.
    """
    result = create_topic_detailed(settings, title=title, course=course, slug=slug)
    for message in result.warnings:
        warnings.warn(message, TopicWarning, stacklevel=2)
    return result.path


def create_topic_detailed(
    settings: Settings, *, title: str, course: str, slug: str | None = None
) -> TopicCreation:
    """Same as `create_topic`, but returns the path together with git status and warnings."""
    if not title.strip():
        raise ValueError("Название темы не может быть пустым")
    if not course.strip():
        raise ValueError("Курс не может быть пустым")

    path = topic_path(settings, title=title, course=course, slug=slug)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Тема уже существует: {path}")

    path.mkdir(parents=True, exist_ok=True)
    for name in TOPIC_DIRS:
        (path / name).mkdir(exist_ok=True)

    meta = TopicMeta(
        title=title.strip(),
        course=course.strip(),
        slug=path.name,
        language=settings.general.language,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    save_topic(path, meta)
    (path / ".gitignore").write_text(_GITIGNORE_TEXT, encoding="utf-8", newline="\n")

    result = TopicCreation(path=path, meta=meta)
    if settings.general.git_per_topic:
        _init_git(result)
    return result


def save_topic(path: Path, meta: TopicMeta) -> Path:
    """Write topic.yaml (`path` is the topic directory or the file itself). Returns the file."""
    target = path / TOPIC_FILE if path.is_dir() else path
    body = yaml.safe_dump(
        meta.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    target.write_text(_TOPIC_YAML_HEADER + body, encoding="utf-8", newline="\n")
    return target


def load_topic(path: Path) -> TopicMeta:
    """Read topic.yaml (`path` is the topic directory or the file itself)."""
    target = path / TOPIC_FILE if path.is_dir() else path
    if not target.is_file():
        raise FileNotFoundError(f"Не найден {TOPIC_FILE}: {target}")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Не удалось прочитать {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{target}: ожидался YAML-словарь с полями темы")
    return TopicMeta.model_validate(data)


# ---------------------------------------------------------------- git


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _init_git(result: TopicCreation) -> None:
    git = tools.find_simple("git")
    if git is None:
        result.warnings.append(
            "git не найден — тема создана без репозитория. "
            "Установите git (Windows: winget install Git.Git) или выключите general.git_per_topic."
        )
        return
    env = procutil.clean_env(drop_names=_GIT_ENV_DROP)

    def git_run(*args: str) -> procutil.ProcResult:
        return procutil.run([str(git), *args], cwd=result.path, env=env, timeout=_GIT_TIMEOUT_S)

    res = git_run("init", "-q", "-b", "main")
    if not res.ok:  # git < 2.28 has no -b
        res = git_run("init", "-q")
        if res.ok:
            git_run("symbolic-ref", "HEAD", "refs/heads/main")
    if not res.ok:
        detail = _first_line(res.stderr) or res.error or f"код {res.exit_code}"
        result.warnings.append(f"git init не удался ({detail}) — тема создана без репозитория.")
        return
    result.git_repo = True

    res = git_run("add", "-A")
    if res.ok:
        res = git_run("commit", "-q", "-m", INITIAL_COMMIT_MESSAGE)
    if res.ok:
        result.committed = True
        return

    stderr = res.stderr or res.stdout
    lowered = stderr.lower()
    if "user.email" in lowered or "identity" in lowered or "who you are" in lowered:
        result.warnings.append(
            "Репозиторий темы создан, но первый коммит не сделан: не настроена git-идентичность. "
            'Выполните git config --global user.name "Ваше имя" и '
            'git config --global user.email "you@example.com", затем в каталоге темы: '
            f'git add -A; git commit -m "{INITIAL_COMMIT_MESSAGE}".'
        )
    else:
        detail = _first_line(stderr) or res.error or f"код {res.exit_code}"
        result.warnings.append(f"Репозиторий темы создан, но первый коммит не сделан: {detail}.")
