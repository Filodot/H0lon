"""Shared agent types: usage, outcomes, attempt records, backend protocol, error classification."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from h0lon.agents.bundle import TaskBundle

Tier = Literal["light", "strong"]
ErrorKind = Literal[
    "auth", "rate_limit", "timeout", "crash", "validation", "not_found", "refusal", "unknown"
]
EventCallback = Callable[[str], None]

# Variables of a parent Claude Code session that procutil.clean_env does not catch (they lack
# the CLAUDE_CODE_ prefix); CLAUDE_EFFORT would override the child's --effort.
EXTRA_SESSION_VARS = ("CLAUDE_EFFORT", "CLAUDE_PREVIEW_CLASSIFIER_FLOOR")

# Errors after which the produced files are not even looked at: nothing useful was done.
HARD_ERRORS: frozenset[str] = frozenset({"auth", "rate_limit", "not_found", "timeout", "refusal"})


@dataclass
class Usage:
    """Token usage of one or more attempts.

    `input_tokens` is the full prompt size including cached tokens (OpenAI convention);
    `cached_input_tokens` is the part served from the prompt cache.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        cost = data.get("cost_usd")
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            cached_input_tokens=int(data.get("cached_input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
            cost_usd=float(cost) if cost is not None else None,
        )


@dataclass
class AuthStatus:
    logged_in: bool | None  # None = could not determine
    method: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendOutcome:
    """What a backend reports about one CLI invocation (before output validation)."""

    backend: str
    model: str | None
    argv: list[str]
    exit_code: int | None
    duration_s: float
    error_kind: ErrorKind | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    final_text: str = ""
    structured_output: Any = None  # parsed final message if the CLI validated it itself
    transcript: Path | None = None
    timed_out: bool = False
    notes: list[str] = field(default_factory=list)  # retries, permission denials, …
    # Subscription window utilisation reported by the CLI (Claude: rate_limit_event).
    limits: dict[str, Any] | None = None


@dataclass
class AttemptRecord:
    n: int
    backend: str
    model: str | None
    started_at: str
    duration_s: float
    exit_code: int | None
    ok: bool
    error_kind: ErrorKind | None
    error: str | None
    usage: Usage
    validation_problems: list[str]
    transcript: str  # path to the attempt's transcript.jsonl
    limits: dict[str, Any] | None = None  # subscription window utilisation, if reported

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


class AgentBackend(Protocol):
    name: str
    # True when the backend cannot take a system prompt file and the system text must be
    # placed at the top of the user prompt (Codex).
    inline_system_prompt: bool

    def resolve(self) -> list[str] | None: ...

    def auth_status(self) -> AuthStatus: ...

    def invoke(
        self,
        bundle: TaskBundle,
        *,
        tier: Tier,
        prompt: str,
        attempt_dir: Path,
        on_event: EventCallback | None,
    ) -> BackendOutcome: ...


# ---------------------------------------------------------------- error classification

_AUTH_STRONG = re.compile(
    r"not logged in|\b401\b|unauthori[sz]ed|authentication[_ ]failed|invalid api key|"
    r"invalid x-api-key|please run /login|oauth token|token (?:has )?expired|"
    r"not authenticated|login required",
    re.IGNORECASE,
)
_RATE = re.compile(
    r"rate[_ -]?limit|\b429\b|usage limit|limit reached|hit your (?:usage )?limit|"
    r"too many requests|quota|overloaded|\b529\b",
    re.IGNORECASE,
)
_AUTH_WEAK = re.compile(r"\blog ?in\b|credentials", re.IGNORECASE)


def classify_error_text(text: str) -> ErrorKind | None:
    """Map an error message to `auth` / `rate_limit`, or None if it is neither."""
    if not text:
        return None
    if _AUTH_STRONG.search(text):
        return "auth"
    if _RATE.search(text):
        return "rate_limit"
    if _AUTH_WEAK.search(text):
        return "auth"
    return None


_WINDOW_NAMES = {"five_hour": "5 ч", "seven_day": "7 дн", "seven_day_opus": "7 дн Opus"}


def format_limits(limits: dict[str, Any] | None) -> str | None:
    """'5 ч — 49 % (сброс 18:00 UTC), 7 дн — 22 %' or None when nothing is known."""
    if not limits:
        return None
    parts = []
    for name, win in (limits.get("windows") or {}).items():
        util = win.get("utilization")
        if util is None:
            continue
        label = _WINDOW_NAMES.get(name, name)
        text = f"{label} — {round(float(util) * 100)} %"
        resets = win.get("resets_at")
        if isinstance(resets, str) and len(resets) >= 16:
            text += f" (сброс {resets[5:10]} {resets[11:16]} UTC)"
        parts.append(text)
    status = limits.get("status")
    if status and status != "allowed":
        parts.append(f"статус {status}")
    return ", ".join(parts) or None


def short(text: str, limit: int = 300) -> str:
    """One-line, length-limited version of a message for logs and reports."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def safe_emit(on_event: EventCallback | None, message: str) -> None:
    if on_event is None:
        return
    try:
        on_event(message)
    except Exception:  # a broken progress callback must not break the run
        pass
