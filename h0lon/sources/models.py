"""Source records stored in topic.yaml (`sources:` list) and the source kind vocabulary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Kinds a source can have. `pdf-text` has a usable text layer, `pdf-scan` is a scanned PDF
# without one (printed or unknown), `handwritten` is a handwritten manuscript (photos or a
# scanned PDF; full pipeline in M4), `slides` is a presentation (PPTX or a slide-shaped PDF).
SourceKind = Literal[
    "pdf-text", "pdf-scan", "handwritten", "slides", "video", "audio", "web", "docx", "md", "tex"
]
SOURCE_KINDS: tuple[str, ...] = (
    "pdf-text",
    "pdf-scan",
    "handwritten",
    "slides",
    "video",
    "audio",
    "web",
    "docx",
    "md",
    "tex",
)

# Letter of the source id per kind: H1, P2, S1, V1, A1, W1, D1 …
ID_PREFIX: dict[str, str] = {
    "handwritten": "H",
    "pdf-text": "P",
    "pdf-scan": "P",
    "slides": "S",
    "video": "V",
    "audio": "A",
    "web": "W",
    "docx": "D",
    "md": "D",
    "tex": "D",
}

SourceStatus = Literal["added", "extracting", "extracted", "failed", "skipped"]


class SourceRecord(BaseModel):
    """One entry of topic.yaml `sources`. Unknown keys written by later stages are preserved."""

    model_config = ConfigDict(extra="allow")

    id: str  # H1, P2, S1 … — stable for the lifetime of the topic
    kind: SourceKind
    title: str  # human title (from the file name or the page title)
    file: str | None = None  # path relative to the topic dir, e.g. sources/P1_lektsiya-3.pdf
    original_name: str | None = None  # file name as the user gave it (may contain Cyrillic)
    url: str | None = None  # for web pages and videos given by link
    sha256: str | None = None  # of the file in sources/ (None for not-yet-downloaded URLs)
    size: int | None = None  # bytes
    added: str  # ISO 8601 UTC
    units: dict[str, int | float] = Field(default_factory=dict)  # pages / slides / minutes
    quality: dict[str, Any] = Field(default_factory=dict)  # signals, see ARCHITECTURE «M1»
    status: SourceStatus = "added"
    extracted_key: str | None = None  # cache key of the last successful extraction
    error: str | None = None  # last extraction error, human readable
