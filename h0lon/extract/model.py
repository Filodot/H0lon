"""Shared types of the extraction stage (contracts in docs/ARCHITECTURE.md, section M1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from h0lon.config import Settings
    from h0lon.sources.models import SourceRecord

BlockType = Literal[
    "heading",
    "paragraph",
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "example",
    "problem",
    "solution",
    "remark",
    "formula",
    "list",
    "table",
    "figure",
    "code",
    "quote",
    "author-question",
    "editorial",
    "uncertain",
    "admin",
]

# Fenced-div classes that map 1:1 to block types (same vocabulary as the master, M0 render).
DIV_BLOCK_TYPES: tuple[str, ...] = (
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "example",
    "problem",
    "solution",
    "remark",
    "author-question",
    "editorial",
    "uncertain",
    "admin",
)

EventCallback = Callable[[str], None]


@dataclass
class Block:
    """Minimal unit of source content; the basis of the coverage map (PRD 8)."""

    id: str  # "<source id>.b<NNN>", e.g. P1.b007 — 1-based, zero-padded to 3 digits
    source: str  # source id, e.g. P1
    type: BlockType
    anchor: str  # location in the source, e.g. "P1:p3", "S1:s12", "W1:§2"
    text: str  # plain text (formulas kept as LaTeX), used for coverage and search
    md: str  # the block as Pandoc Markdown
    title: str | None = None  # div title attribute or heading text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageImage:
    """One page / slide prepared for vision transcription."""

    number: int  # 1-based page or slide number in the source
    image: Path  # PNG, long side ≤ 2000 px
    text_hint: str = ""  # text layer of the page (may be garbled for formulas), may be empty
    reason: str = ""  # why the page goes to vision: "math", "scan", "graphic", "low-text"


@dataclass
class ExtractContext:
    topic_dir: Path
    source: SourceRecord
    settings: Settings
    out_dir: Path  # extracted/<ID>/ (created by the pipeline)
    force: bool = False  # ignore caches
    use_vision: bool = True  # False: deterministic part only (pages marked as gaps)
    backend: str | None = None  # agent override for vision/summary runs
    on_event: EventCallback | None = None

    def emit(self, message: str) -> None:
        if self.on_event is not None:
            try:
                self.on_event(message)
            except Exception:
                pass


@dataclass
class ExtractResult:
    ok: bool
    source_id: str
    source_md: Path | None = None  # extracted/<ID>/source.md
    blocks: int = 0
    pages_total: int = 0
    pages_vision: int = 0  # pages/slides transcribed by an agent
    agent_runs: int = 0
    cached: bool = False  # nothing recomputed
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_md"] = str(self.source_md) if self.source_md else None
        return d


class Extractor(Protocol):
    """Turns one source into extracted/<ID>/{source.md, pages/, figures/, …}.

    It writes `body.md` — Pandoc Markdown of the whole source with location headings
    (`## [[P1:p3]] Страница 3`, `## [[S1:s12]] Слайд 12`, `## [[W1:§2]] …`) — and returns
    quality signals. The pipeline then builds source.md (front matter + block id comments),
    blocks.jsonl and summary.md from body.md, so extractors never number blocks themselves.
    """

    kinds: tuple[str, ...]
    version: str  # bump when the output for the same input changes
    # Optional: prompt files the extractor uses, e.g. ("vision_pages@1.0",); part of the cache
    # key (the pipeline reads it with getattr, default ()).
    # prompts: tuple[str, ...]

    def plan(self, ctx: ExtractContext) -> ExtractPlan: ...

    def extract(self, ctx: ExtractContext) -> ExtractOutput: ...


@dataclass
class ExtractPlan:
    """What extraction of one source will do (for `h0lon extract --dry-run`)."""

    source_id: str
    pages_total: int = 0
    pages_vision: int = 0
    agent_runs: int = 0  # vision batches + summary
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractOutput:
    body_md: Path  # extracted/<ID>/body.md
    pages_total: int = 0
    pages_vision: int = 0
    agent_runs: int = 0
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
