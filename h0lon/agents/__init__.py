"""Headless agents (Claude Code, Codex CLI): task bundles, runner, validation, usage journal."""

from h0lon.agents.base import (
    AgentBackend,
    AttemptRecord,
    AuthStatus,
    BackendOutcome,
    ErrorKind,
    Tier,
    Usage,
    classify_error_text,
)
from h0lon.agents.bundle import (
    Feedback,
    TaskBundle,
    build_prompt,
    create_bundle,
    load_bundle,
    save_bundle,
)
from h0lon.agents.contract import ExpectedFile, OutputContract
from h0lon.agents.runner import (
    RunResult,
    cooling_backends,
    get_backend,
    mark_cooling,
    reset_cooling,
    run_task,
)
from h0lon.agents.validate import check_contract, validate_final_message, validate_outputs

__all__ = [
    "AgentBackend",
    "AttemptRecord",
    "AuthStatus",
    "BackendOutcome",
    "ErrorKind",
    "ExpectedFile",
    "Feedback",
    "OutputContract",
    "RunResult",
    "TaskBundle",
    "Tier",
    "Usage",
    "build_prompt",
    "check_contract",
    "classify_error_text",
    "cooling_backends",
    "create_bundle",
    "get_backend",
    "load_bundle",
    "mark_cooling",
    "reset_cooling",
    "run_task",
    "save_bundle",
    "validate_final_message",
    "validate_outputs",
]
