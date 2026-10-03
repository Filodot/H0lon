"""PDF extractor (`pdf-text`, `pdf-scan`): text pages by code, the rest by the vision agent.

Every page is profiled and classified (`pages.classify_page`): `text` pages go through
pymupdf4llm one page at a time after running headers/footers are redacted from a working
copy; `math`, `scan` and `graphic` pages are rendered to `extracted/<ID>/pages/p<NNNN>.png`
and transcribed by `vision.transcribe_pages` with the cleaned text layer as a hint
(flavor `scan` for scans and for `pdf-scan` sources, `document` otherwise). `body.md` has a
place heading per page: `## [[P1:p3]] Страница 3`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from h0lon.agents import Usage
from h0lon.extract import pages as pg
from h0lon.extract import vision
from h0lon.extract.model import ExtractContext, ExtractOutput, ExtractPlan, PageImage
from h0lon.extract.registry import ExtractError

VERSION = f"1.1+mupdf{pymupdf.VersionBind}"
AGENT_OFF_TITLE = "не распознано: агент отключён"
FAILED_TITLE = "Страница не распознана агентом"
TEXT_LAYER_INTRO = "Текстовый слой страницы (формулы и рисунки в нём могут быть искажены):"
LOW_DPI = 150
REPAIRED_NOTE = (
    "PDF повреждён и был восстановлен при открытии — возможны потери страниц или их "
    "содержимого, сверьте с оригиналом"
)


def source_path(ctx: ExtractContext) -> Path:
    if not ctx.source.file:
        raise ExtractError(f"{ctx.source.id}: у источника нет файла в sources/")
    path = ctx.topic_dir / ctx.source.file
    if not path.is_file():
        raise ExtractError(f"{ctx.source.id}: файл не найден: {path}")
    return path


def write_body(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.rstrip("\n") + "\n")
    tmp.replace(path)
    return path


def page_name(number: int) -> str:
    return f"p{number:04d}"


def pages_list(numbers: Sequence[int]) -> str:
    nums = sorted(numbers)
    if len(nums) > 12:
        return ", ".join(map(str, nums[:12])) + f" … (всего {len(nums)})"
    return ", ".join(map(str, nums))


# ---------------------------------------------------------------- shared PDF machinery


@dataclass
class Analysis:
    """Profiles of all pages, running lines and what goes where."""

    profiles: list[pg.PageProfile]
    running: pg.RunningLines
    repaired: bool = False  # MuPDF had to repair the file (broken xref, truncated file)

    @property
    def vision(self) -> list[pg.PageProfile]:
        return [p for p in self.profiles if p.needs_vision]

    @property
    def text(self) -> list[pg.PageProfile]:
        return [p for p in self.profiles if not p.needs_vision]

    def drop(self, number: int) -> list[pg.LineInfo]:
        return self.running.drop.get(number, [])

    def count(self, kind: str) -> int:
        return sum(1 for p in self.profiles if p.kind == kind)


def analyze(doc: pymupdf.Document, *, slides: bool = False) -> Analysis:
    profiles = pg.profile_document(doc, slides=slides)
    return Analysis(
        profiles=profiles,
        running=pg.find_running_lines(profiles),
        repaired=bool(getattr(doc, "is_repaired", False)),
    )


def repaired_warning(sid: str) -> str:
    return f"{sid}: {REPAIRED_NOTE}"


def flavor_of(profile: pg.PageProfile, *, source_kind: str, slides: bool = False) -> str:
    if slides:
        return "slides"
    if source_kind == "pdf-scan" or (profile.kind == "scan" and profile.reason == "scan"):
        return "scan"
    return "document"


def text_pages_markdown(
    src: Path,
    analysis: Analysis,
    profiles: Sequence[pg.PageProfile],
    *,
    figures_dir: Path,
    figure_prefix: str,
    extra_drop: dict[int, list[pg.LineInfo]] | None = None,
    warnings: list[str],
) -> dict[int, str]:
    """Markdown of text pages: redact running lines in a working copy, pymupdf4llm per page."""
    out: dict[int, str] = {}
    todo = [p for p in profiles if p.reason != "empty"]
    for p in profiles:
        if p.reason == "empty":
            out[p.number] = vision.EMPTY_PAGE
    if not todo:
        return out
    work = pg.open_pdf(src)
    try:
        for p in todo:
            drop = analysis.drop(p.number) + (extra_drop or {}).get(p.number, [])
            md = ""
            try:
                pg.redact_lines(work[p.number - 1], drop)
                result = pg.text_page_markdown(
                    work,
                    p,
                    figures_dir=figures_dir,
                    figure_prefix=f"{figure_prefix}{page_name(p.number)}",
                    fixes=pg.line_fixes(p, drop),
                )
                md = result.markdown
            except Exception as exc:  # pymupdf4llm on an odd page: keep the text layer
                warnings.append(
                    f"Страница {p.number}: разметка не построена ({exc}); взят текстовый слой"
                )
            if not md.strip():
                plain = pg.page_text(p, drop, reflow=True)
                md = pg.escape_markdown(plain) if plain else vision.EMPTY_PAGE
            out[p.number] = md
    finally:
        work.close()
    return out


def render_vision_pages(
    doc: pymupdf.Document,
    analysis: Analysis,
    profiles: Sequence[pg.PageProfile],
    *,
    pages_dir: Path,
    extra_drop: dict[int, list[pg.LineInfo]] | None = None,
) -> dict[int, PageImage]:
    images: dict[int, PageImage] = {}
    for p in profiles:
        png = pg.render_page_png(doc[p.number - 1], pages_dir / f"{page_name(p.number)}.png")
        drop = analysis.drop(p.number) + (extra_drop or {}).get(p.number, [])
        images[p.number] = PageImage(
            number=p.number, image=png, text_hint=pg.page_text(p, drop), reason=p.reason
        )
    return images


@dataclass
class VisionOutcome:
    pages: dict[int, str] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)
    agent_runs: int = 0
    cached: int = 0
    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # repairs of the agent's Markdown


def run_vision(
    ctx: ExtractContext,
    groups: dict[str, list[PageImage]],
    *,
    batch_size: dict[str, int],
) -> VisionOutcome:
    out = VisionOutcome()
    usage = Usage()
    for flavor, images in groups.items():
        if not images:
            continue
        res = vision.transcribe_pages(
            ctx, images, flavor=flavor, tier="light", batch_size=batch_size.get(flavor, 6)
        )
        out.pages.update(res.pages)
        out.failed.update(res.failed)
        out.agent_runs += res.agent_runs
        out.cached += res.cached_pages
        out.attempts += res.attempts
        out.warnings += res.warnings
        usage = usage + res.usage
    out.usage = usage.to_dict()
    return out


def fallback_texts(
    analysis: Analysis,
    profiles: Sequence[pg.PageProfile],
    *,
    extra_drop: dict[int, list[pg.LineInfo]] | None = None,
) -> dict[int, str]:
    """Reflowed text layer of pages that go to the agent (used when it cannot help)."""
    return {
        p.number: pg.page_text(
            p, analysis.drop(p.number) + (extra_drop or {}).get(p.number, []), reflow=True
        )
        for p in profiles
    }


def fallback_block(text: str, *, agent_off: bool) -> str:
    title = AGENT_OFF_TITLE if agent_off else FAILED_TITLE
    return pg.uncertain_block(title, text, intro=TEXT_LAYER_INTRO)


def planned_runs(
    ctx: ExtractContext,
    analysis: Analysis,
    *,
    flavor_for: Any,
    batch_size: dict[str, int],
    pages_dir: Path,
    extra_drop: dict[int, list[pg.LineInfo]] | None = None,
) -> tuple[int, int, int]:
    """(pending vision pages, cached pages, agent runs) without rendering anything new."""
    pending: dict[str, int] = {}
    cached = 0
    for p in analysis.vision:
        flavor = flavor_for(p)
        png = pages_dir / f"{page_name(p.number)}.png"
        drop = analysis.drop(p.number) + (extra_drop or {}).get(p.number, [])
        image = PageImage(number=p.number, image=png, text_hint=pg.page_text(p, drop))
        if png.is_file() and vision.cached_page(ctx, image, flavor) is not None:
            cached += 1
        else:
            pending[flavor] = pending.get(flavor, 0) + 1
    runs = sum(vision.batches_needed(n, batch_size.get(f, 6)) for f, n in pending.items())
    return sum(pending.values()), cached, runs


def base_quality(
    analysis: Analysis,
    *,
    contents: dict[int, str],
    vision_done: int,
    failed: Sequence[int],
) -> dict[str, Any]:
    profiles = analysis.profiles
    n = len(profiles)
    # A short text page («Глава 3») has a text layer; a scan with a stray glyph has not.
    with_text = sum(
        1 for p in profiles if p.chars >= pg.SCAN_MAX_CHARS or (p.kind == "text" and p.chars)
    )
    scan_dpis = [
        p.scan_dpi
        for p in profiles
        if p.kind == "scan" and p.scan_dpi and p.vector_items < pg.SCAN_MIN_STROKES
    ]
    return {
        "text_layer": round(with_text / n, 3) if n else 0.0,
        "chars_per_page": round(pg.median_or_zero([p.chars for p in profiles])),
        "pages_text": analysis.count("text"),
        "pages_math": analysis.count("math"),
        "pages_scan": analysis.count("scan"),
        "pages_graphic": analysis.count("graphic"),
        "pages_vision": vision_done,
        "pages_failed": len(failed),
        "cyrillic_ratio": pg.cyrillic_ratio(contents.values()),
        "scan_dpi": round(pg.median_or_zero(scan_dpis)) if scan_dpis else None,
    }


def common_notes(
    analysis: Analysis,
    quality: dict[str, Any],
    *,
    use_vision: bool,
    failed: Sequence[int],
    unit: str = "стр.",
) -> list[str]:
    notes: list[str] = []
    if analysis.repaired:
        notes.append(REPAIRED_NOTE)
    vision_numbers = [p.number for p in analysis.vision]
    if not use_vision and vision_numbers:
        notes.append(
            f"Агент отключён: {unit} {pages_list(vision_numbers)} перенесены текстовым слоем "
            "без распознавания — формулы, сканы и рисунки нужно проверить"
        )
    if failed:
        notes.append(
            f"Не распознаны агентом: {unit} {pages_list(failed)} — вставлен текстовый слой, "
            "проверьте вручную"
        )
    if use_vision and quality.get("pages_math"):
        notes.append(
            f"Формулы перенесены агентом по изображению ({unit} с формулами: "
            f"{quality['pages_math']}) — проверьте выборочно индексы и знаки"
        )
    if quality.get("scan_dpi") and quality["scan_dpi"] < LOW_DPI:
        notes.append(
            f"Сканы низкого разрешения (≈{quality['scan_dpi']} dpi) — возможны ошибки чтения"
        )
    vector_scans = [
        p.number
        for p in analysis.profiles
        if p.kind == "scan" and p.vector_items >= pg.SCAN_MIN_STROKES
    ]
    if vector_scans:
        notes.append(
            f"Рукописные или векторные страницы без текстового слоя: {unit} "
            f"{pages_list(vector_scans)}"
        )
    small = [p.number for p in analysis.text if p.small_pictures]
    if small:
        notes.append(
            f"Мелкие картинки (возможно, формулы или значки) не перенесены: {unit} "
            f"{pages_list(small)} — сверьте с оригиналом"
        )
    if analysis.running.texts:
        shown = ", ".join(f"«{t}»" for t in analysis.running.texts[:4])
        more = len(analysis.running.texts) - 4
        notes.append("Удалены колонтитулы: " + shown + (f" и ещё {more}" if more > 0 else ""))
    kinds = [line.fix_kind for p in analysis.profiles for line in p.lines if line.fix_kind]
    if kinds.count("despaced"):
        notes.append(
            f"Склеен разреженный текст («М А Т Е М А Т И К А»): строк — {kinds.count('despaced')}"
        )
    if kinds.count("respaced"):
        notes.append(
            "Восстановлены пробелы между словами (текстовый слой без пробелов): строк — "
            f"{kinds.count('respaced')}"
        )
    return notes


# ---------------------------------------------------------------- extractor


class PdfExtractor:
    kinds: tuple[str, ...] = ("pdf-text", "pdf-scan")
    version: str = VERSION
    prompts: tuple[str, ...] = (vision.PROMPT_REF,)
    tier: str = "light"
    batch_size: int = vision.DEFAULT_BATCH

    def _batch_sizes(self) -> dict[str, int]:
        return {"document": self.batch_size, "scan": self.batch_size}

    def plan(self, ctx: ExtractContext) -> ExtractPlan:
        src = source_path(ctx)
        with pg.opened(src) as doc:
            analysis = analyze(doc)
        plan = ExtractPlan(source_id=ctx.source.id, pages_total=len(analysis.profiles))
        plan.notes.append(
            f"Страниц: {plan.pages_total} (текст: {analysis.count('text')}, формулы: "
            f"{analysis.count('math')}, сканы: {analysis.count('scan')}, рисунки: "
            f"{analysis.count('graphic')})"
        )
        if not analysis.vision:
            return plan
        if not ctx.use_vision:
            plan.notes.append(
                f"Агент отключён: {len(analysis.vision)} стр. пойдут текстовым слоем без проверки"
            )
            return plan
        pending, cached, runs = planned_runs(
            ctx,
            analysis,
            flavor_for=lambda p: flavor_of(p, source_kind=ctx.source.kind),
            batch_size=self._batch_sizes(),
            pages_dir=ctx.out_dir / "pages",
        )
        plan.pages_vision = pending
        plan.agent_runs = runs
        if pending:
            plan.notes.append(
                f"Распознавание агентом: {pending} стр., прогонов: {runs} "
                f"(батчи по {self.batch_size})"
            )
        if cached:
            plan.notes.append(f"Из кэша распознавания: {cached} стр.")
        return plan

    def extract(self, ctx: ExtractContext) -> ExtractOutput:
        t0 = time.monotonic()
        src = source_path(ctx)
        sid = ctx.source.id
        warnings: list[str] = []
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            doc = pg.open_pdf(src)
        except Exception as exc:
            raise ExtractError(f"{sid}: не удалось открыть PDF: {exc}") from exc
        try:
            analysis = analyze(doc)
            ctx.emit(
                f"{sid}: {len(analysis.profiles)} стр. — текст {analysis.count('text')}, "
                f"формулы {analysis.count('math')}, сканы {analysis.count('scan')}, "
                f"рисунки {analysis.count('graphic')}"
            )
            if analysis.repaired:
                warnings.append(repaired_warning(sid))
            contents = text_pages_markdown(
                src,
                analysis,
                analysis.text,
                figures_dir=ctx.out_dir / "figures",
                figure_prefix=f"{sid}_",
                warnings=warnings,
            )
            # Page renders are only for the agent: none without it.
            images = (
                render_vision_pages(doc, analysis, analysis.vision, pages_dir=ctx.out_dir / "pages")
                if ctx.use_vision
                else {}
            )
        finally:
            doc.close()

        outcome = VisionOutcome()
        if images:
            groups: dict[str, list[PageImage]] = {}
            for p in analysis.vision:
                flavor = flavor_of(p, source_kind=ctx.source.kind)
                groups.setdefault(flavor, []).append(images[p.number])
            outcome = run_vision(ctx, groups, batch_size=self._batch_sizes())
            warnings += outcome.warnings
        fallback = fallback_texts(analysis, analysis.vision)
        ratio_texts = dict(contents)
        for n in (p.number for p in analysis.vision):
            if n in outcome.pages:
                contents[n] = ratio_texts[n] = outcome.pages[n]
            else:
                contents[n] = fallback_block(fallback[n], agent_off=not ctx.use_vision)
                ratio_texts[n] = fallback[n]
        failed = sorted(outcome.failed) if ctx.use_vision else []
        if failed:
            warnings.append(
                f"{sid}: не распознаны агентом стр. {pages_list(failed)} — вставлен "
                "текстовый слой в блоке uncertain"
            )
        if analysis.vision and not ctx.use_vision:
            warnings.append(
                f"{sid}: распознавание агентом отключено — {len(analysis.vision)} стр. "
                "перенесены текстовым слоем без проверки"
            )

        body = []
        for p in analysis.profiles:
            body.append(f"## [[{sid}:p{p.number}]] Страница {p.number}")
            body.append(contents.get(p.number) or vision.EMPTY_PAGE)
        body_md = write_body(ctx.out_dir / "body.md", "\n\n".join(body))

        quality = base_quality(
            analysis, contents=ratio_texts, vision_done=len(outcome.pages), failed=failed
        )
        quality["notes"] = common_notes(analysis, quality, use_vision=ctx.use_vision, failed=failed)
        ctx.emit(f"{sid}: body.md готов за {time.monotonic() - t0:.1f} с")
        return ExtractOutput(
            body_md=body_md,
            pages_total=len(analysis.profiles),
            pages_vision=len(outcome.pages),
            agent_runs=outcome.agent_runs,
            quality=quality,
            warnings=warnings,
        )


def page_profiles(path: Path, *, slides: bool = False) -> list[dict[str, Any]]:
    """Profiles of every page as dicts (diagnostics: `python -m h0lon.extract.pdf file.pdf`)."""
    with pg.opened(path) as doc:
        return [p.to_dict() for p in analyze(doc, slides=slides).profiles]


if __name__ == "__main__":  # pragma: no cover - diagnostics helper
    import json
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps(page_profiles(Path(arg)), ensure_ascii=False, indent=1))
