"""Configuration: TOML file + environment overrides (H0LON_<SECTION>__<KEY>)."""

from __future__ import annotations

import os
from contextvars import ContextVar
from importlib import resources
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

APP_NAME = "H0lon"
CONFIG_FILENAME = "h0lon.toml"
ENV_CONFIG = "H0LON_CONFIG"

Effort = Literal["low", "medium", "high", "xhigh", "max"]
AgentName = Literal["claude", "codex"]


class GeneralConfig(BaseModel):
    language: str = "ru"
    workspaces: Path | None = None
    state_dir: Path | None = None
    review_gate: bool = True
    git_per_topic: bool = True
    keep_video: bool = False

    @field_validator("workspaces", "state_dir", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        return None if v in ("", None) else v

    @property
    def workspaces_dir(self) -> Path:
        return Path(self.workspaces).expanduser() if self.workspaces else Path.home() / "Konspekty"

    @property
    def state_path(self) -> Path:
        if self.state_dir:
            return Path(self.state_dir).expanduser()
        return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


class ClaudeConfig(BaseModel):
    bin: str | list[str] = "claude"
    model_strong: str = "opus"
    model_light: str = "sonnet"
    effort_strong: Effort = "high"
    effort_light: Effort = "low"
    max_turns: int = 40
    timeout_s: int = 1800
    allow_api_key: bool = False
    extra_args: list[str] = Field(default_factory=list)


class CodexConfig(BaseModel):
    bin: str | list[str] = "codex"
    model_strong: str = ""
    model_light: str = ""
    effort_strong: Effort = "high"
    effort_light: Effort = "low"
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    timeout_s: int = 1800
    extra_args: list[str] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    default: AgentName = "claude"
    fallback: AgentName | Literal[""] = "codex"
    parallel_runs: int = 2
    max_validation_retries: int = 2
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    codex: CodexConfig = Field(default_factory=CodexConfig)


StageChoice = Literal["auto", "claude", "codex"]


class StagesConfig(BaseModel):
    handwriting: StageChoice = "auto"
    slides_frames: StageChoice = "auto"
    transcript_fix: StageChoice = "auto"
    synthesis: StageChoice = "auto"
    coverage: StageChoice = "auto"
    variants: StageChoice = "auto"

    def backend_for(self, stage: str, agents: AgentsConfig) -> AgentName:
        """Agent for a bundle stage; synthesis and service stages map onto the config keys."""
        field_name = STAGE_TO_FIELD.get(stage, stage)
        choice = getattr(self, field_name, "auto")
        if choice not in ("auto", "claude", "codex"):
            choice = "auto"
        return agents.default if choice == "auto" else choice


# Bundle stage names (agents.create_bundle(stage=…)) → StagesConfig field.
STAGE_TO_FIELD: dict[str, str] = {
    "outline": "synthesis",
    "sections": "synthesis",
    "global": "synthesis",
    "supplement": "synthesis",
    "coverage": "coverage",
    "fixlatex": "coverage",
    # M1 agent runs (page transcription, source annotation) follow the extraction setting.
    "extract": "slides_frames",
    "summary": "slides_frames",
}


class QueueConfig(BaseModel):
    unattended: bool = False
    keep_awake: bool = True


class ComputeConfig(BaseModel):
    asr: Literal["auto", "local-gpu", "local-cpu", "colab", "api"] = "auto"
    asr_model: str = "large-v3"
    colab_url: str = ""


class RenderConfig(BaseModel):
    engine: Literal["xelatex", "html"] = "xelatex"
    template: str = "a4-notes"
    fallback_html: bool = True
    passes: int = 3
    browser: str = "auto"
    pandoc: str = ""
    xelatex: str = ""
    main_font: str = "Times New Roman"
    sans_font: str = "Arial"
    mono_font: str = "Consolas"
    math_font: str = "Cambria Math"


class ApiConfig(BaseModel):
    enabled: bool = False
    base_url: str = "https://api.proxyapi.ru/v1"


_config_file: ContextVar[Path | None] = ContextVar("h0lon_config_file", default=None)


class Settings(BaseSettings):
    """Effective settings. Priority: init kwargs > env (H0LON_*) > TOML file > defaults."""

    model_config = SettingsConfigDict(
        env_prefix="H0LON_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = _config_file.get()
        if path is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
        return tuple(sources)

    # Filled by load_settings(); not part of the TOML schema.
    _source_path: Path | None = None

    @property
    def source_path(self) -> Path | None:
        """Config file the settings were loaded from (None = defaults only)."""
        return self._source_path


def user_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False, roaming=True)) / (
        CONFIG_FILENAME
    )


def find_config_file(explicit: Path | None = None) -> Path | None:
    """Resolve the config file: explicit > $H0LON_CONFIG > ./h0lon.toml > user config dir."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Файл конфигурации не найден: {explicit}")
        return explicit
    env = os.environ.get(ENV_CONFIG)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{ENV_CONFIG} указывает на несуществующий файл: {p}")
        return p
    local = Path.cwd() / CONFIG_FILENAME
    if local.is_file():
        return local
    user = user_config_path()
    if user.is_file():
        return user
    return None


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from the resolved config file (if any) with env overrides."""
    resolved = find_config_file(path)
    token = _config_file.set(resolved)
    try:
        settings = Settings()
    finally:
        _config_file.reset(token)
    settings._source_path = resolved
    return settings


def example_config_text() -> str:
    return (
        resources.files("h0lon.resources")
        .joinpath("h0lon.example.toml")
        .read_text(encoding="utf-8")
    )


def write_user_config(target: Path | None = None, *, force: bool = False) -> Path:
    """Write the example config to `target` (default: user config dir). Returns the path."""
    target = target or user_config_path()
    if target.exists() and not force:
        raise FileExistsError(f"Конфигурация уже существует: {target} (используйте --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_config_text(), encoding="utf-8")
    return target
