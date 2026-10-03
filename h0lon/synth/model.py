"""Shared types of the synthesis stage (M2). Contracts: docs/ARCHITECTURE.md, «Синтез (M2)»."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from h0lon.config import Settings

# Stages in execution order. `extract` reuses M1, `render` is M0 + the LaTeX auto-fix (S6).
STAGES: tuple[str, ...] = (
    "extract",
    "outline",  # S1
    "sections",  # S2
    "global",  # S3
    "coverage",  # S4 (+ S5 supplement loop, at most 2 rounds)
    "assemble",  # master.md = intro + sections + appendices
    "render",  # S6: master.pdf with LaTeX auto-fix, HTML fallback
)
Stage = Literal["extract", "outline", "sections", "global", "coverage", "assemble", "render"]

SYNTH_DIR = "synthesis"  # <topic>/synthesis/
SECTIONS_DIR = "sections"  # <topic>/synthesis/sections/<section id>.md
MASTER_MD = "master.md"
MASTER_PDF = "master.pdf"

# `<!-- src: P1.b007 S1.b012 -->` after every paragraph/block of the master: the coverage map.
SRC_COMMENT_PREFIX = "src:"

EventCallback = Callable[[str], None]


@dataclass
class OutlineSection:
    """One section of the unified topic structure (S1)."""

    id: str  # stable, ASCII: "s01", "s01-02"; used as file name and as {#sec:<id>} anchor
    title: str  # Russian
    level: int  # 1 = chapter (#), 2 = section (##)
    summary: str = ""  # one line: what the section covers
    blocks: list[str] = field(default_factory=list)  # block ids assigned to this section
    parent: str | None = None  # id of the level-1 section for level-2 sections

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Outline:
    title: str  # topic title for the master
    sections: list[OutlineSection]
    unassigned: list[dict[str, str]] = field(default_factory=list)  # {block, reason}
    conflict_hints: list[dict[str, Any]] = field(default_factory=list)  # {topic, blocks, note}

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "unassigned": list(self.unassigned),
            "conflict_hints": list(self.conflict_hints),
        }


# Records the synthesis agents emit next to the text; rendered into the master appendices.
@dataclass
class Correction:  # Приложение Б. Журнал правок
    section: str
    anchor: str  # [[P1:p3]] without brackets: "P1:p3"
    block: str | None
    as_written: str
    corrected: str
    reason: str


@dataclass
class Conflict:  # Приложение А. Расхождения между источниками
    section: str
    topic: str
    variants: list[dict[str, str]]  # [{anchor, text}]
    resolution: str | None  # chosen version for the body, or None if undecided
    reason: str


@dataclass
class Editorial:  # Приложение В. Редакторские дополнения
    section: str
    kind: Literal["answer", "reconstruction", "clarification"]
    refers_to: str  # block id or anchor
    text: str  # Markdown


@dataclass
class CoverageReport:
    total: int  # blocks of non-admin types
    covered: int
    by_source: dict[str, dict[str, int]]  # source id -> {total, covered}
    uncovered: list[dict[str, str]]  # {block, verdict: duplicate|admin|missing|unknown, note}
    rounds: int = 0  # supplement rounds performed

    @property
    def ratio(self) -> float:
        return self.covered / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ratio"] = round(self.ratio, 4)
        return d


@dataclass
class StageResult:
    stage: str
    ok: bool
    cached: bool = False
    agent_runs: int = 0
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildContext:
    topic_dir: Path
    settings: Settings
    force: bool = False  # ignore stage caches
    review: bool = True  # honour the review gate
    backend: str | None = None  # agent override
    on_event: EventCallback | None = None

    @property
    def synth_dir(self) -> Path:
        return self.topic_dir / SYNTH_DIR

    def emit(self, message: str) -> None:
        if self.on_event is not None:
            try:
                self.on_event(message)
            except Exception:
                pass


@dataclass
class BuildResult:
    ok: bool
    topic_dir: Path
    stopped_at: str | None = None  # stage that failed or the review gate ("review")
    master_md: Path | None = None
    master_pdf: Path | None = None
    stages: list[StageResult] = field(default_factory=list)
    coverage: CoverageReport | None = None
    message: str = ""  # human summary (Russian), e.g. what to do at the review gate
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "topic_dir": str(self.topic_dir),
            "stopped_at": self.stopped_at,
            "master_md": str(self.master_md) if self.master_md else None,
            "master_pdf": str(self.master_pdf) if self.master_pdf else None,
            "stages": [s.to_dict() for s in self.stages],
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "message": self.message,
            "duration_s": round(self.duration_s, 1),
        }
