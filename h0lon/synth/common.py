"""Helpers shared by the synthesis stages: blocks, src comments, stage cache, prompts, agents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from h0lon.agents import OutputContract, Tier, create_bundle, run_task
from h0lon.agents.runner import RunResult
from h0lon.extract.model import Block
from h0lon.synth.model import SECTIONS_DIR, BuildContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# `<!-- src: P1.b007 S1.b012 -->` (ids separated by spaces and/or commas)
SRC_RE = re.compile(r"<!--\s*src:\s*([^>]*?)\s*-->")
BLOCK_ID_RE = re.compile(r"\b[A-Z][0-9]+\.b[0-9]{3,}\b")
SECTION_ID_RE = re.compile(r"^s\d{2}(-\d{2})?$")


# ---------------------------------------------------------------- blocks


def load_blocks(topic_dir: Path, source_ids: Iterable[str] | None = None) -> dict[str, Block]:
    """All blocks of the extracted sources (extracted/<ID>/blocks.jsonl), keyed by block id.

    Order: sources in `source_ids` order (default: sorted directory names), blocks in file order.
    """
    root = topic_dir / "extracted"
    ids = (
        list(source_ids)
        if source_ids is not None
        else sorted(p.name for p in root.iterdir() if p.is_dir())
        if root.is_dir()
        else []
    )
    blocks: dict[str, Block] = {}
    for sid in ids:
        path = root / sid / "blocks.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            block = Block(**{k: data.get(k) for k in Block.__dataclass_fields__})
            blocks[block.id] = block
    return blocks


def is_admin(block: Block) -> bool:
    return block.type == "admin"


# ---------------------------------------------------------------- src comments


def parse_src_ids(markdown: str) -> set[str]:
    """Block ids mentioned in `<!-- src: … -->` comments of a Markdown text."""
    ids: set[str] = set()
    for m in SRC_RE.finditer(markdown):
        ids.update(BLOCK_ID_RE.findall(m.group(1)))
    return ids


def src_ids_in_files(paths: Iterable[Path]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        if p.is_file():
            ids |= parse_src_ids(p.read_text(encoding="utf-8"))
    return ids


def text_size(markdown: str) -> int:
    """Length of the text without HTML comments and whitespace (for «nothing was lost» checks)."""
    return len(re.sub(r"\s+", "", re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)))


# ---------------------------------------------------------------- stage cache


def stage_key(*parts: str | bytes | Path | Mapping[str, Any] | Sequence[Any] | None) -> str:
    """sha256 over the given parts (file contents for existing paths, JSON for structures)."""
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            h.update(b"\x00none")
        elif isinstance(part, bytes):
            h.update(part)
        elif isinstance(part, Path):
            h.update(part.read_bytes() if part.is_file() else str(part).encode())
        elif isinstance(part, str):
            h.update(part.encode("utf-8"))
        else:
            h.update(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str).encode())
        h.update(b"\x1f")
    return h.hexdigest()


def meta_path(ctx: BuildContext, stage: str) -> Path:
    return ctx.synth_dir / f"{stage}.meta.json"


def read_meta(ctx: BuildContext, stage: str) -> dict[str, Any] | None:
    path = meta_path(ctx, stage)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_meta(ctx: BuildContext, stage: str, data: Mapping[str, Any]) -> None:
    write_json_atomic(meta_path(ctx, stage), dict(data))


def cached(ctx: BuildContext, stage: str, key: str, outputs: Iterable[Path]) -> bool:
    """True when the stage ran with the same key and all its outputs still exist."""
    if ctx.force:
        return False
    meta = read_meta(ctx, stage)
    return bool(meta and meta.get("key") == key and all(p.exists() for p in outputs))


def sections_dir(ctx: BuildContext) -> Path:
    return ctx.synth_dir / SECTIONS_DIR


# ---------------------------------------------------------------- prompts


def prompt_path(name: str) -> Path:
    """h0lon/prompts/<name>@<version>.md (the highest version if several)."""
    found = sorted(PROMPTS_DIR.glob(f"{name}@*.md"))
    if not found:
        raise FileNotFoundError(f"Не найден промпт {name}@<версия>.md в {PROMPTS_DIR}")
    return found[-1]


def prompt_id(name: str) -> str:
    """'outline@1.0' — part of cache keys and of the extracted_by / synthesis metadata."""
    return prompt_path(name).stem


def prompt_body(name: str) -> str:
    """Prompt text without the leading HTML comment (that comment is for developers)."""
    text = prompt_path(name).read_text(encoding="utf-8")
    return re.sub(r"\A\s*<!--.*?-->\s*", "", text, flags=re.DOTALL).strip() + "\n"


# ---------------------------------------------------------------- agents


def run_agent(
    ctx: BuildContext,
    *,
    stage: str,
    task: str,
    contract: OutputContract,
    inputs: Sequence[Path] = (),
    seed: Mapping[str, Path] | None = None,
    tier: Tier = "strong",
) -> RunResult:
    """One agent task of a synthesis stage: bundle in <topic>/runs/, run with retries/fallback."""
    bundle = create_bundle(
        ctx.topic_dir / "runs",
        stage=stage,
        task=task,
        contract=contract,
        inputs=inputs,
        seed=seed,
    )
    return run_task(
        bundle, settings=ctx.settings, tier=tier, backend=ctx.backend, on_event=ctx.on_event
    )


def model_for(ctx: BuildContext, tier: Tier) -> str:
    """Configured model name of the default backend for the tier (part of cache keys)."""
    agents = ctx.settings.agents
    name = ctx.backend or agents.default
    cfg = agents.claude if name == "claude" else agents.codex
    model = cfg.model_strong if tier == "strong" else cfg.model_light
    effort = cfg.effort_strong if tier == "strong" else cfg.effort_light
    return f"{name}:{model or 'default'}:{effort}"
