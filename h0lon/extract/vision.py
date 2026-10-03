"""Page transcription by a vision agent (docs/ARCHITECTURE.md, «Распознавание страниц»).

Pages go to the agent in batches: one task bundle per batch (`<topic>/runs/<id>/`, stage
`extract`) with the page PNGs and their text layers as `inputs/p<NNNN>.png|txt`; the
contract asks for `out/p<NNNN>.md` per page. Results are normalized (no page headings, no
```markdown fences, headings `#`/`##` → `###`, fenced divs balanced within the page, raw
TeX outside math made literal) and cached in `extracted/<ID>/pages/`: `p<NNNN>.md` +
`p<NNNN>.key`, the key being sha256 of the PNG, of the hint, of the prompt version, of the
flavor and of the task format.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from h0lon.agents import ExpectedFile, OutputContract, Tier, Usage, create_bundle, run_task
from h0lon.extract.model import ExtractContext, PageImage

Flavor = Literal["document", "slides", "scan"]
FLAVORS: tuple[str, ...] = ("document", "slides", "scan")

PROMPT_FILE = "vision_pages@1.0.md"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / PROMPT_FILE
PROMPT_NAME, PROMPT_VERSION = PROMPT_FILE.removesuffix(".md").split("@", 1)
PROMPT_REF = f"{PROMPT_NAME}@{PROMPT_VERSION}"
# Version of what this module adds around the prompt (page list, normalization). Part of
# the page cache key: bump it when the task text or `normalize_page_markdown` changes.
TASK_FORMAT = "2"
STAGE = "extract"
DEFAULT_BATCH = 6
SLIDES_BATCH = 10
EMPTY_PAGE = "<!-- пустая страница -->"

_LABEL = {"document": "Страница", "slides": "Слайд", "scan": "Страница"}
_LABEL_PLURAL = {"document": "страниц", "slides": "слайдов", "scan": "страниц"}


@dataclass
class VisionResult:
    pages: dict[int, str] = field(default_factory=dict)  # number → Markdown (no place heading)
    failed: dict[int, str] = field(default_factory=dict)  # number → reason
    agent_runs: int = 0  # run_task calls (one per batch)
    usage: Usage = field(default_factory=Usage)
    cached_pages: int = 0
    attempts: int = 0  # agent attempts over all runs (retries and fallbacks included)
    bundles: list[str] = field(default_factory=list)  # bundle directories of the runs
    warnings: list[str] = field(default_factory=list)  # repairs of the agent's Markdown


# ---------------------------------------------------------------- prompt


def _split_prompt(raw: str) -> tuple[str, dict[str, str]]:
    # Only comments that start a line (the file header); `<!-- пустая страница -->` inside
    # backticks is part of the instructions.
    text = re.sub(r"^<!--.*?-->[ \t]*\n?", "", raw, flags=re.DOTALL | re.MULTILINE)
    parts = re.split(r"^##[ \t]+Вариант:[ \t]*(\S+)[ \t]*$", text, flags=re.MULTILINE)
    common = parts[0].strip()
    variants = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return common, variants


def prompt_text(flavor: str) -> str:
    """Common part of the prompt + the section of one flavor (no comment, no «## Вариант»)."""
    common, variants = _split_prompt(PROMPT_PATH.read_text(encoding="utf-8"))
    if flavor not in variants:
        raise ValueError(f"Неизвестный вариант распознавания «{flavor}»")
    return common + "\n\n" + variants[flavor]


def _page_name(number: int) -> str:
    return f"p{number:04d}"


def batch_task(flavor: str, numbers: Sequence[int]) -> str:
    label = _LABEL[flavor]
    lines = [prompt_text(flavor), "", "## Страницы этого задания", ""]
    for n in numbers:
        name = _page_name(n)
        lines.append(
            f"- {label} {n}: изображение `inputs/{name}.png`, текстовый слой "
            f"`inputs/{name}.txt` → `out/{name}.md`"
        )
    return "\n".join(lines) + "\n"


def batch_contract(flavor: str, numbers: Sequence[int]) -> OutputContract:
    label = _LABEL[flavor].lower()
    return OutputContract(
        files=[
            ExpectedFile(
                path=f"{_page_name(n)}.md",
                kind="markdown",
                min_chars=1,
                description=(
                    f"Транскрипция: {label} {n} (`inputs/{_page_name(n)}.png`); "
                    f"пустая по содержанию страница — строка `{EMPTY_PAGE}`."
                ),
            )
            for n in numbers
        ]
    )


# ---------------------------------------------------------------- normalization

_FENCE_ALL = re.compile(
    r"\A\s*(?P<f>`{3,}|~{3,})[ \t]*(?P<lang>[\w+-]*)[ \t]*\n(?P<body>.*?)\n[ \t]*(?P=f)[ \t]*\s*\Z",
    re.DOTALL,
)
_UNWRAP_LANGS = {"markdown", "md", "pandoc", "commonmark", "gfm"}
# Structures of Markdown that code does not have: a bare ``` around the whole answer is
# unwrapped only when its body has one (a slide may be just a listing).
_MARKDOWN_ONLY = re.compile(r"^[ \t]*(?::{3,}|\$\$|\|?[ \t]*:?-{3,}:?[ \t]*\|)", re.MULTILINE)
_DIV_OPEN = re.compile(r"^[ \t]*:{3,}[ \t]*(?:\{[^}\n]*\}|[\w-]+)[ \t]*(?::{3,}[ \t]*)?$")
_DIV_CLOSE = re.compile(r"^[ \t]*:{3,}[ \t]*$")
_PLACE_HEADING = re.compile(r"^\s*#{1,6}\s*\[\[[A-Z][A-Za-z0-9]*:[^\]\s]+\]\]")
# A heading or a bold line naming the page («## Страница 3», «**Слайд 12. Итоги**») or a
# bare «Страница 3» line: the place heading of the Source Doc already says it.
_PAGE_TITLE = re.compile(
    r"^\s*(?:#{1,6}[ \t]*|\*\*|__)[ \t]*(?:\[\[[^\]]*\]\][ \t]*)?"
    r"(?:страница|стр\.|page|слайд|slide)[ \t]*№?[ \t]*\d+\b[^\n]{0,80}$",
    re.IGNORECASE,
)
_BARE_PAGE_TITLE = re.compile(
    r"^\s*(?:страница|стр\.|page|слайд|slide)[ \t]*№?[ \t]*\d+[ \t]*[.:]?[ \t]*$",
    re.IGNORECASE,
)
_TOP_HEADING = re.compile(r"^(#{1,2})(?=[ \t])")
_CODE_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def normalize_page_markdown(text: str) -> str:
    """Clean an agent's page: fences around the whole answer, page headings, `#`/`##` levels."""
    return normalize_page(text)[0]


def normalize_page(text: str) -> tuple[str, list[str]]:
    """`normalize_page_markdown` + the list of repairs worth a warning (Russian)."""
    issues: list[str] = []
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    m = _FENCE_ALL.match(text)
    if m:
        lang = m.group("lang").lower()
        if lang in _UNWRAP_LANGS or (not lang and _MARKDOWN_ONLY.search(m.group("body"))):
            text = m.group("body")
    lines = text.split("\n")
    # Leading page titles («## Страница 3», «**Слайд 12**», «## [[P1:p3]] Страница 3»).
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and (
        _PAGE_TITLE.match(lines[0])
        or _BARE_PAGE_TITLE.match(lines[0])
        or _PLACE_HEADING.match(lines[0])
    ):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    out: list[str] = []
    code = _closed_fences(lines)  # an unclosed fence is text for Pandoc
    math_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i in code:
            out.append(line.rstrip())
            continue
        opened_in_math = math_block
        if stripped.count("$$") % 2:
            math_block = not math_block
        if not opened_in_math and not stripped.startswith("$$"):
            if _PLACE_HEADING.match(line):
                continue  # the place heading belongs to the extractor
            line = _TOP_HEADING.sub("###", line)
        out.append(line.rstrip())
    out, opened, stray = balance_divs(out)
    if opened:
        issues.append(f"агент не закрыл блок «:::» ({opened} шт.) — закрыт в конце страницы")
    if stray:
        issues.append(f"лишние закрывающие «:::» ({stray} шт.) удалены")
    text = escape_raw_tex("\n".join(out))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text or EMPTY_PAGE), issues


def _closed_fences(lines: list[str]) -> set[int]:
    """Indices of lines inside closed code fences (an unclosed fence is text for Pandoc)."""
    inside: set[int] = set()
    i = 0
    while i < len(lines):
        m = _CODE_FENCE.match(lines[i])
        if m:
            mark = m.group(1)
            for j in range(i + 1, len(lines)):
                close = _CODE_FENCE.match(lines[j])
                if close and close.group(1)[0] == mark[0] and len(close.group(1)) >= len(mark):
                    inside.update(range(i, j + 1))
                    i = j
                    break
        i += 1
    return inside


def balance_divs(lines: list[str]) -> tuple[list[str], int, int]:
    """Close fenced divs left open by the agent and drop stray closing fences.

    The page is one place of the Source Doc: an open `::: {.theorem}` would swallow every
    following page (and its place anchor) into one block. Returns (lines, closed, dropped).
    """
    code = _closed_fences(lines)
    depth = stray = 0
    in_math = False
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in code:
            out.append(line)
            continue
        was_math = in_math
        if line.count("$$") % 2:
            in_math = not in_math
        if not was_math and not line.strip().startswith("$$"):
            if _DIV_CLOSE.match(line):
                if depth == 0:
                    stray += 1
                    continue
                depth -= 1
            elif _DIV_OPEN.match(line):
                depth += 1
        out.append(line)
    if depth:
        while out and not out[-1].strip():
            out.pop()
        out += [":::"] * depth
    return out, depth, stray


_FENCED_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
# Math and code spans by the rules of Pandoc (`tex_math_dollars`): `$$…$$`; `$…$` whose
# opening `$` is not followed by a space and whose closing `$` is not preceded by a space
# nor followed by a digit (so «5$ за штуку» and «$5 и $10» are not math); inline math may
# wrap onto the next line but not across a blank line; `\$` is not a delimiter.
_MATH_OR_CODE = re.compile(
    r"(?<!\\)\$\$.+?(?<!\\)\$\$"
    r"|(?<![\\$])\$(?![\s$])(?:\\.|[^$\\\n]|\n(?![ \t]*\n))+?(?<![\s\\])\$(?!\d)"
    r"|(`+)[^\n]+?(?<!`)\1(?!`)",
    re.DOTALL,
)
_BRACKET_MATH = re.compile(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]|(?<!\\)\\\((.+?)(?<!\\)\\\)", re.DOTALL)
_TEX_OUTSIDE_MATH = re.compile(r"(?<!\\)\\(?=[A-Za-z])")


def _plain(segment: str) -> str:
    r"""Text outside math and code: `\[…\]`/`\(…\)` become `$$…$$`/`$…$`, other `\cmd` literal."""
    out: list[str] = []
    pos = 0
    for m in _BRACKET_MATH.finditer(segment):
        out.append(_TEX_OUTSIDE_MATH.sub(r"\\\\", segment[pos : m.start()]))
        if m.group(1) is not None:
            out.append("$$" + m.group(1) + "$$")
        else:
            out.append("$" + m.group(2).strip() + "$")
        pos = m.end()
    out.append(_TEX_OUTSIDE_MATH.sub(r"\\\\", segment[pos:]))
    return "".join(out)


def escape_raw_tex(text: str) -> str:
    r"""`\mu` outside `$…$` and code becomes literal text; `\(…\)`, `\[…\]` become math.

    Pandoc would pass a bare `\mu` to LaTeX as raw TeX outside math mode (a compile
    error). Agents copy such fragments from pages where a formula failed to render in the
    source itself.
    """

    def outside_code(segment: str) -> str:
        out: list[str] = []
        pos = 0
        for m in _MATH_OR_CODE.finditer(segment):
            out += [_plain(segment[pos : m.start()]), m.group(0)]
            pos = m.end()
        out.append(_plain(segment[pos:]))
        return "".join(out)

    if "\\" not in text:
        return text
    pieces: list[str] = []
    pos = 0
    for m in _FENCED_CODE.finditer(text):
        pieces += [outside_code(text[pos : m.start()]), m.group(0)]
        pos = m.end()
    pieces.append(outside_code(text[pos:]))
    return "".join(pieces)


# ---------------------------------------------------------------- cache


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def page_key(page: PageImage, flavor: str) -> str:
    payload = {
        "png": _sha256_file(page.image),
        "hint": hashlib.sha256(page.text_hint.encode("utf-8")).hexdigest(),
        "prompt": PROMPT_REF,
        "flavor": flavor,
        "task": TASK_FORMAT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cache_dir(ctx: ExtractContext) -> Path:
    return ctx.out_dir / "pages"


def cached_page(ctx: ExtractContext, page: PageImage, flavor: str) -> str | None:
    """Cached Markdown of the page if its key matches (None when absent, stale or forced)."""
    if ctx.force or not page.image.is_file():
        return None
    base = cache_dir(ctx) / _page_name(page.number)
    md, key = base.with_suffix(".md"), base.with_suffix(".key")
    try:
        if key.read_text(encoding="utf-8").strip() != page_key(page, flavor):
            return None
        return md.read_text(encoding="utf-8")
    except OSError:
        return None


def _store(ctx: ExtractContext, page: PageImage, flavor: str, text: str) -> None:
    base = cache_dir(ctx) / _page_name(page.number)
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.with_suffix(".md").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    with base.with_suffix(".key").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(page_key(page, flavor) + "\n")


# ---------------------------------------------------------------- concurrency

_slots_lock = threading.Lock()
_slots: tuple[int, threading.BoundedSemaphore] | None = None


def _run_slots(n: int) -> threading.BoundedSemaphore:
    """Process-wide limit of simultaneous vision runs (`agents.parallel_runs`)."""
    global _slots
    n = max(1, n)
    with _slots_lock:
        if _slots is None or _slots[0] != n:
            _slots = (n, threading.BoundedSemaphore(n))
        return _slots[1]


# ---------------------------------------------------------------- main entry


@dataclass
class _BatchOutcome:
    pages: dict[int, str] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0
    ran: bool = False
    bundle: str | None = None
    warnings: list[str] = field(default_factory=list)


def _span(numbers: Sequence[int]) -> str:
    nums = sorted(numbers)
    if len(nums) == 1:
        return str(nums[0])
    if nums[-1] - nums[0] == len(nums) - 1:
        return f"{nums[0]}–{nums[-1]}"
    return ", ".join(map(str, nums))


def _run_batch(
    ctx: ExtractContext, batch: Sequence[PageImage], *, flavor: str, tier: Tier
) -> _BatchOutcome:
    outcome = _BatchOutcome()
    numbers = [p.number for p in batch]
    what = f"{ctx.source.id}: {_LABEL_PLURAL[flavor]} {_span(numbers)}"
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".vision-", dir=ctx.out_dir))
    try:
        images: list[Path] = []
        hints: list[Path] = []
        for p in batch:
            name = _page_name(p.number)
            img = staging / f"{name}.png"
            shutil.copyfile(p.image, img)
            hint = staging / f"{name}.txt"
            with hint.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(p.text_hint)
            images.append(img)
            hints.append(hint)
        bundle = create_bundle(
            ctx.topic_dir / "runs",
            stage=STAGE,
            task=batch_task(flavor, numbers),
            contract=batch_contract(flavor, numbers),
            inputs=hints,
            images=images,
        )
    except Exception as exc:  # unreadable image, disk full …
        reason = f"не удалось подготовить задание агенту: {exc}"
        outcome.failed = dict.fromkeys(numbers, reason)
        ctx.emit(f"{what}: {reason}")
        return outcome
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    outcome.bundle = str(bundle.root)
    ctx.emit(f"{what}: распознавание агентом ({len(batch)} шт.)")
    with _run_slots(ctx.settings.agents.parallel_runs):
        try:
            result = run_task(
                bundle,
                settings=ctx.settings,
                tier=tier,
                backend=ctx.backend,
                on_event=ctx.on_event,
            )
        except Exception as exc:  # the runner records failures itself; this is a bug guard
            reason = f"сбой запуска агента: {exc!r}"
            outcome.failed = dict.fromkeys(numbers, reason)
            ctx.emit(f"{what}: {reason}")
            return outcome
    outcome.ran = True
    outcome.usage = result.usage_total
    outcome.attempts = len(result.attempts)
    problem = "; ".join(result.problems[:3]) or "агент не выполнил задание"
    for p in batch:
        path = bundle.out_dir / f"{_page_name(p.number)}.md"
        text: str | None = None
        try:
            if path.is_file():
                text = path.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            text = None
        if text is None or not text.strip():
            outcome.failed[p.number] = problem if not result.ok else "агент не создал файл страницы"
            continue
        normalized, issues = normalize_page(text)
        outcome.warnings += [
            f"{ctx.source.id}: {_LABEL[flavor].lower()} {p.number} — {issue}" for issue in issues
        ]
        outcome.pages[p.number] = normalized
        try:
            _store(ctx, p, flavor, normalized)
        except OSError as exc:
            ctx.emit(f"{what}: кэш страницы {p.number} не записан ({exc})")
    status = "готово" if not outcome.failed else f"не распознано: {len(outcome.failed)}"
    ctx.emit(f"{what}: {status}")
    return outcome


def transcribe_pages(
    ctx: ExtractContext,
    pages: Sequence[PageImage],
    *,
    flavor: Flavor,
    tier: Tier = "light",
    batch_size: int = DEFAULT_BATCH,
) -> VisionResult:
    """Transcribe page images with the vision agent; cached pages are not sent again."""
    if flavor not in FLAVORS:
        raise ValueError(f"Неизвестный вариант распознавания «{flavor}»")
    numbers = [p.number for p in pages]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Номера страниц для распознавания повторяются")
    result = VisionResult()
    pending: list[PageImage] = []
    for p in sorted(pages, key=lambda x: x.number):
        cached = cached_page(ctx, p, flavor)
        if cached is not None:
            result.pages[p.number] = normalize_page_markdown(cached)
            result.cached_pages += 1
        else:
            pending.append(p)
    if result.cached_pages:
        ctx.emit(f"{ctx.source.id}: из кэша {result.cached_pages} {_LABEL_PLURAL[flavor]}")
    if not pending:
        return result

    size = max(1, batch_size)
    batches = [pending[i : i + size] for i in range(0, len(pending), size)]
    workers = max(1, min(ctx.settings.agents.parallel_runs, len(batches)))
    if workers == 1:
        outcomes = [_run_batch(ctx, b, flavor=flavor, tier=tier) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h0lon-vision") as pool:
            futures = [pool.submit(_run_batch, ctx, b, flavor=flavor, tier=tier) for b in batches]
            outcomes = [f.result() for f in futures]
    for o in outcomes:
        result.pages.update(o.pages)
        result.failed.update(o.failed)
        result.usage = result.usage + o.usage
        result.attempts += o.attempts
        result.warnings += o.warnings
        result.agent_runs += 1 if o.ran else 0
        if o.bundle:
            result.bundles.append(o.bundle)
    return result


def batches_needed(pending: int, batch_size: int) -> int:
    return -(-pending // max(1, batch_size)) if pending > 0 else 0
