"""Handwritten notes extractor (`handwritten`, docs/ARCHITECTURE.md, «Рукописные конспекты (M4)»).

A source is a photo (`.jpg/.jpeg/.png`, `.heic` with `pillow-heif`) or a PDF scan added with
`--kind handwritten`. Per page:

1. Preparation by code (Pillow): EXIF rotation, uniform margins cropped, long side ≤ 2000 px.
   Two files go to `extracted/<ID>/pages/`: `p<NNNN>.png` (colour: red marks and underlines
   mean something) and `p<NNNN>.enh.png` (grayscale, autocontrast, mild sharpening). The
   source file is never written to. PDF pages are rendered at ~200 dpi.
2. Transcription by the strong-tier agent (`vision.transcribe_pages` with the prompt
   `handwritten_pages`, batches of 4, bundle stage `handwriting`): both pictures of a page
   go to the bundle, the text layer of a PDF page is a hint.
3. `body.md` with a place heading per page (`## [[H1:p3]] Страница 3`); a page the agent
   did not return is replaced by an `uncertain` block (text layer if there is one).

`quality`: `pages`, `pages_vision`, `pages_failed`, `uncertain_marks` (`[неразборчиво]` and
`[?]`), `author_questions`, `scan_dpi`, `notes`. `pages/prep.json` remembers what was
prepared from which file, so a repeated run (and `--dry-run`) does not touch the pictures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from h0lon.extract import pages as pg
from h0lon.extract import pdf as pdfx
from h0lon.extract import vision
from h0lon.extract.model import ExtractContext, ExtractOutput, ExtractPlan, PageImage
from h0lon.extract.registry import ExtractError

# Version of the output for the same file: bump together with any change of the page
# preparation (it is part of the cache key and of `pages/prep.json`).
VERSION = "1.0"
PROMPT_NAME = "handwritten_pages"
PROMPT_REF = vision.prompt_ref(PROMPT_NAME)
FLAVOR = "scan"  # only labels the pages («Страница»); the prompt has no variants
TIER = "strong"
STAGE = "handwriting"  # bundle stage: the agent follows `stages.handwriting`
BATCH_SIZE = 4

MAX_SIDE = 2000
PREP_FILE = "prep.json"
ENHANCED_SUFFIX = ".enh.png"

# Margin cropping. A pixel is content when it differs from the colour of the picture's
# border by more than CROP_THRESHOLD (of 255) in any channel. The mask is measured on a
# probe of at most PROBE_SIDE px and counted by cells of CROP_CELL px: a cell is content from
# CROP_CELL_MIN lit pixels, so single specks do not hold the border back. The crop is
# applied only if it saves CROP_MIN_SAVING of the area and keeps CROP_MIN_KEEP of each side
# (an almost empty page stays whole: wrongly cropped writing is worse than white margins).
PROBE_SIDE = 1600
CROP_THRESHOLD = 30
CROP_CELL = 4
CROP_CELL_MIN = 2
CROP_PAD = 0.02  # of the long side
CROP_MIN_SAVING = 0.04
CROP_MIN_KEEP = 0.25

# Photos have no physical size: the resolution is estimated as if an A4 sheet filled the frame.
A4_LONG_SIDE_IN = 11.69
LOW_DPI = 150

HEIC_SUFFIXES = (".heic", ".heif")
HEIC_HINT = (
    "Формат HEIC открывается только с пакетом pillow-heif: выполните «uv add pillow-heif» "
    "или сохраните фото как JPG и добавьте файл заново"
)
TEXT_LAYER_NOTE = "Текстовый слой страницы (может быть искажён):"

_UNCERTAIN_RE = re.compile(r"\[неразборчиво\]|\[\?\]", re.IGNORECASE)
_UNREADABLE_RE = re.compile(r"\[неразборчиво\]", re.IGNORECASE)
_RESTORED_RE = re.compile(r"\[восстановлено по контексту\]", re.IGNORECASE)
_QUESTION_RE = re.compile(
    r"^[ \t]*:{3,}[ \t]*(?:author-question|\{[^}\n]*\.author-question\b[^}\n]*\})[ \t]*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------- pictures


@dataclass
class PreparedPicture:
    """A picture after rotation, cropping and scaling (RGB)."""

    image: Image.Image
    source_size: tuple[int, int]  # pixels after rotation, before cropping and scaling
    rotated: bool = False  # an EXIF orientation was applied
    cropped: bool = False


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB copy-or-same of a picture; transparency goes onto white paper."""
    if img.mode == "RGB":
        return img
    if "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        paper.alpha_composite(rgba)
        return paper.convert("RGB")
    return img.convert("RGB")


def _border_color(probe: Image.Image) -> tuple[int, int, int]:
    """Colour of the picture's border (median over the four edge strips and over channels)."""
    w, h = probe.size
    t = max(2, round(min(w, h) * 0.012))
    strips = (
        probe.crop((0, 0, w, t)),
        probe.crop((0, h - t, w, h)),
        probe.crop((0, 0, t, h)),
        probe.crop((w - t, 0, w, h)),
    )
    medians = [ImageStat.Stat(s).median for s in strips]
    out = []
    for c in range(3):
        values = sorted(m[c] for m in medians)
        out.append(round((values[1] + values[2]) / 2))
    return out[0], out[1], out[2]


_THRESHOLD_LUT = [0] * (CROP_THRESHOLD + 1) + [255] * (255 - CROP_THRESHOLD)
# `reduce` averages a cell of the 0/255 mask: CROP_CELL_MIN lit pixels give about this value.
_CELL_MIN_VALUE = int(255 * CROP_CELL_MIN / CROP_CELL**2) - 1
_CELL_LUT = [0] * _CELL_MIN_VALUE + [255] * (256 - _CELL_MIN_VALUE)


def find_content_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Box (in pixels of `img`) around everything that differs from the border colour, with
    a small pad; None when the picture is uniform. `img` must be RGB."""
    w, h = img.size
    probe = _scaled(img, PROBE_SIDE)
    pw, ph = probe.size
    bg = _border_color(probe)
    r, g, b = ImageChops.difference(probe, Image.new("RGB", probe.size, bg)).split()
    diff = ImageChops.lighter(ImageChops.lighter(r, g), b)
    mask = diff.point(_THRESHOLD_LUT)
    cells = mask.reduce(CROP_CELL).point(_CELL_LUT)
    box = cells.getbbox()
    if box is None:
        return None
    pad = round(CROP_PAD * max(pw, ph))
    x0 = max(0, box[0] * CROP_CELL - pad)
    y0 = max(0, box[1] * CROP_CELL - pad)
    x1 = min(pw, box[2] * CROP_CELL + pad)
    y1 = min(ph, box[3] * CROP_CELL + pad)
    sx, sy = w / pw, h / ph
    return (
        max(0, math.floor(x0 * sx)),
        max(0, math.floor(y0 * sy)),
        min(w, math.ceil(x1 * sx)),
        min(h, math.ceil(y1 * sy)),
    )


def crop_margins(img: Image.Image) -> tuple[Image.Image, bool]:
    """Crop uniform margins (see the constants above); the picture itself is not changed."""
    box = find_content_box(img)
    if box is None:
        return img, False
    w, h = img.size
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw < CROP_MIN_KEEP * w or bh < CROP_MIN_KEEP * h:
        return img, False
    if bw * bh > (1 - CROP_MIN_SAVING) * w * h:
        return img, False
    return img.crop(box), True


def _scaled(img: Image.Image, max_side: int) -> Image.Image:
    """`img` itself when its long side fits, else a smaller copy (never scaled up)."""
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.Resampling.LANCZOS, reducing_gap=2.0)


def fit_long_side(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """The picture scaled down to a long side ≤ `max_side` (never scaled up)."""
    return _scaled(img, max_side)


def prepare_picture(img: Image.Image, *, max_side: int = MAX_SIDE) -> PreparedPicture:
    """EXIF rotation → RGB → crop of uniform margins → scale to a long side ≤ `max_side`."""
    orientation = 1
    try:
        orientation = int(img.getexif().get(0x0112, 1) or 1)
    except Exception:
        pass
    try:
        rotated = ImageOps.exif_transpose(img)
    except Exception:  # a broken EXIF block must not lose the page
        rotated = None
    work = to_rgb(rotated if rotated is not None else img)
    source_size = work.size
    work, cropped = crop_margins(work)
    return PreparedPicture(
        image=fit_long_side(work, max_side),
        source_size=source_size,
        rotated=orientation not in (0, 1) and rotated is not None,
        cropped=cropped,
    )


def enhance(img: Image.Image) -> Image.Image:
    """Grayscale, autocontrast, mild sharpening — a new picture, `img` stays as it was."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    return gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=2))


def save_png(img: Image.Image, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    img.save(tmp, format="PNG", compress_level=6)
    tmp.replace(target)
    return target


def estimate_photo_dpi(size: tuple[int, int]) -> int:
    """Resolution as if an A4 sheet filled the frame (photos have no physical size)."""
    return round(max(size) / A4_LONG_SIDE_IN)


# ---------------------------------------------------------------- pages


@dataclass
class PreparedPage:
    number: int
    image: Path  # p<NNNN>.png (colour)
    enhanced: Path  # p<NNNN>.enh.png (grayscale)
    text_hint: str = ""  # text layer of a PDF page, empty for photos
    width: int = 0
    height: int = 0
    dpi: float | None = None  # resolution estimate (scans of a PDF, photos)
    rotated: bool = False
    cropped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "rotated": self.rotated,
            "cropped": self.cropped,
            "hint": self.text_hint,
        }

    def page_image(self) -> PageImage:
        return PageImage(
            number=self.number,
            image=self.image,
            text_hint=self.text_hint,
            reason="handwritten",
            extra_images=(self.enhanced,),
        )


def page_paths(pages_dir: Path, number: int) -> tuple[Path, Path]:
    name = f"p{number:04d}"
    return pages_dir / f"{name}.png", pages_dir / f"{name}{ENHANCED_SUFFIX}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _register_heif() -> None:
    try:
        import pillow_heif
    except ImportError as exc:
        raise ExtractError(HEIC_HINT) from exc
    pillow_heif.register_heif_opener()


def check_openable(path: Path) -> None:
    """Raise `ExtractError` early for a picture format that cannot be opened here."""
    if not is_pdf(path) and path.suffix.lower() in HEIC_SUFFIXES:
        _register_heif()


def open_picture(path: Path) -> Image.Image:
    """A picture loaded into memory; readable errors in Russian."""
    check_openable(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                im.load()
                return im.copy()  # keeps `info`, so the EXIF orientation survives
    except Image.DecompressionBombError as exc:
        raise ExtractError(f"Изображение слишком большое для обработки: {exc}") from exc
    except Image.UnidentifiedImageError as exc:
        raise ExtractError(
            f"Не удалось прочитать изображение {path.name}: формат не опознан"
        ) from exc
    except OSError as exc:
        raise ExtractError(f"Не удалось прочитать изображение {path.name}: {exc}") from exc


def _render_pdf_page(page: pymupdf.Page) -> Image.Image:
    zoom = pg.render_zoom(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _pdf_hint(page: pymupdf.Page, number: int) -> tuple[str, float | None]:
    """(text layer, scan resolution) of a PDF page; ("", None) when it cannot be measured."""
    try:
        profile = pg.profile_page(page, number)
        hint = pg.page_text(profile)
    except Exception:
        return "", None
    dpi = None
    if profile.scan_dpi and (
        profile.image_frac_all >= pg.SCAN_MIN_IMAGE and profile.vector_items < pg.SCAN_MIN_STROKES
    ):
        dpi = float(profile.scan_dpi)  # strokes drawn as paths have no resolution
    return hint, dpi


def _save_pair(picture: PreparedPicture, image: Path, enhanced: Path, source: Path) -> None:
    """Write both versions of a page; the source file can never be one of the targets."""
    for target in (image, enhanced):
        if target.resolve() == source.resolve():
            raise ExtractError(f"Исходный файл не перезаписывается: {source}")
    save_png(picture.image, image)
    save_png(enhance(picture.image), enhanced)


def page_count(path: Path) -> int:
    """Pages of a source: PDF pages, or 1 for a picture."""
    if not is_pdf(path):
        check_openable(path)
        return 1
    try:
        with pg.opened(path) as doc:
            return doc.page_count
    except Exception as exc:
        raise ExtractError(f"Не удалось открыть PDF: {exc}") from exc


def prepare_pages(path: Path, pages_dir: Path, *, on_progress: Any = None) -> list[PreparedPage]:
    """Prepare both pictures of every page of the source into `pages_dir`."""
    pages: list[PreparedPage] = []
    if is_pdf(path):
        try:
            doc = pg.open_pdf(path)
        except Exception as exc:
            raise ExtractError(f"Не удалось открыть PDF: {exc}") from exc
        try:
            if doc.page_count == 0:
                raise ExtractError("В PDF нет страниц")
            for index in range(doc.page_count):
                number = index + 1
                page = doc[index]
                hint, dpi = _pdf_hint(page, number)
                picture = prepare_picture(_render_pdf_page(page))
                image, enhanced = page_paths(pages_dir, number)
                _save_pair(picture, image, enhanced, path)
                width, height = picture.image.size
                pages.append(
                    PreparedPage(
                        number=number,
                        image=image,
                        enhanced=enhanced,
                        text_hint=hint,
                        width=width,
                        height=height,
                        dpi=dpi,
                        rotated=picture.rotated,
                        cropped=picture.cropped,
                    )
                )
                if on_progress:
                    on_progress(number, doc.page_count)
        finally:
            doc.close()
    else:
        picture = prepare_picture(open_picture(path))
        image, enhanced = page_paths(pages_dir, 1)
        _save_pair(picture, image, enhanced, path)
        width, height = picture.image.size
        pages.append(
            PreparedPage(
                number=1,
                image=image,
                enhanced=enhanced,
                width=width,
                height=height,
                dpi=float(estimate_photo_dpi(picture.source_size)),
                rotated=picture.rotated,
                cropped=picture.cropped,
            )
        )
    return pages


def _write_prep(pages_dir: Path, sha: str, pages: Sequence[PreparedPage]) -> None:
    data = {
        "format": 1,
        "version": VERSION,
        "source_sha256": sha,
        "pages": [p.to_dict() for p in pages],
    }
    tmp = pages_dir / (PREP_FILE + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(pages_dir / PREP_FILE)


def load_prepared(pages_dir: Path, sha: str) -> list[PreparedPage] | None:
    """Pages prepared earlier from the same file by the same version; None if any is missing."""
    try:
        data = json.loads((pages_dir / PREP_FILE).read_text(encoding="utf-8"))
        if data.get("version") != VERSION or data.get("source_sha256") != sha:
            return None
        pages = []
        for entry in data["pages"]:
            image, enhanced = page_paths(pages_dir, int(entry["number"]))
            if not image.is_file() or not enhanced.is_file():
                return None
            pages.append(
                PreparedPage(
                    number=int(entry["number"]),
                    image=image,
                    enhanced=enhanced,
                    text_hint=str(entry.get("hint") or ""),
                    width=int(entry.get("width") or 0),
                    height=int(entry.get("height") or 0),
                    dpi=entry.get("dpi"),
                    rotated=bool(entry.get("rotated")),
                    cropped=bool(entry.get("cropped")),
                )
            )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return pages or None


def _drop_stale(pages_dir: Path, count: int) -> None:
    """Pictures and cached transcriptions of pages beyond the new page count."""
    for path in pages_dir.glob("p[0-9][0-9][0-9][0-9]*"):
        try:
            if int(path.name[1:5]) > count:
                path.unlink()
        except (ValueError, OSError):
            continue


# ---------------------------------------------------------------- statistics of the text


@dataclass
class PageStats:
    unreadable: int = 0  # [неразборчиво]
    unsure: int = 0  # [?]
    restored: int = 0  # [восстановлено по контексту]
    questions: int = 0  # ::: author-question

    @property
    def uncertain(self) -> int:
        return self.unreadable + self.unsure


def page_stats(markdown: str) -> PageStats:
    """Marks of uncertainty and author's questions in the Markdown of one page."""
    unreadable = len(_UNREADABLE_RE.findall(markdown))
    return PageStats(
        unreadable=unreadable,
        unsure=len(_UNCERTAIN_RE.findall(markdown)) - unreadable,
        restored=len(_RESTORED_RE.findall(markdown)),
        questions=len(_QUESTION_RE.findall(markdown)),
    )


def _pages_text(numbers: Sequence[int], limit: int = 12) -> str:
    return pdfx.pages_list(list(numbers)[:limit]) + (
        f" … (всего {len(numbers)})" if len(numbers) > limit else ""
    )


def build_notes(
    stats: dict[int, PageStats],
    *,
    use_vision: bool,
    vision_done: int,
    failed: Sequence[int],
    empty: Sequence[int],
    dpis: Sequence[float],
    photo: bool,
) -> list[str]:
    """Russian notes for the review gate: what to check first."""
    notes: list[str] = []
    if not use_vision:
        notes.append(
            "Агент отключён: рукопись не распознана, вместо страниц стоят заглушки — "
            "запустите извлечение с агентом"
        )
    if failed:
        notes.append(
            f"Не распознаны агентом: стр. {_pages_text(failed)} — проверьте вручную по "
            "картинкам страниц (extracted/<ID>/pages/)"
        )
    if vision_done:
        notes.append(
            f"Рукопись перенесена агентом по изображению (стр.: {vision_done}): сверьте с "
            "оригиналом формулы, знаки и сокращения — они оставлены как у автора, без "
            "исправлений"
        )
    unreadable = sum(s.unreadable for s in stats.values())
    unsure = sum(s.unsure for s in stats.values())
    if unreadable + unsure:
        top = sorted(
            ((n, s.uncertain) for n, s in stats.items() if s.uncertain),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        where = ", ".join(f"стр. {n} ({count})" for n, count in top)
        notes.append(
            f"Неуверенных мест: {unreadable + unsure} («[неразборчиво]» — {unreadable}, "
            f"«[?]» — {unsure}); проверьте в первую очередь {where}"
        )
    restored = sum(s.restored for s in stats.values())
    if restored:
        notes.append(f"Формул, восстановленных по контексту, а не прочитанных целиком: {restored}")
    question_pages = [n for n, s in sorted(stats.items()) if s.questions]
    if question_pages:
        count = sum(s.questions for s in stats.values())
        notes.append(
            f"Вопросов автора («?», «??» на полях): {count}, стр. {_pages_text(question_pages)} — "
            "оформлены блоками author-question, ответов на них в источнике нет"
        )
    if empty:
        notes.append(f"Пустые страницы: {_pages_text(empty)}")
    if dpis:
        dpi = round(pg.median_or_zero(dpis))
        if dpi < LOW_DPI:
            unit = "Фото низкого разрешения" if photo else "Сканы низкого разрешения"
            how = " в пересчёте на лист A4" if photo else ""
            notes.append(f"{unit} (≈{dpi} dpi{how}) — возможны ошибки чтения")
    return notes


# ---------------------------------------------------------------- fallback pages


def fallback_content(page: PreparedPage, *, agent_off: bool) -> str:
    """What stands in `body.md` for a page without a transcription."""
    title = pdfx.AGENT_OFF_TITLE if agent_off else pdfx.FAILED_TITLE
    if page.text_hint.strip():
        return pg.uncertain_block(title, page.text_hint, intro=TEXT_LAYER_NOTE)
    where = f"pages/{page.image.name}"
    return pg.uncertain_note(
        title, f"Рукописная страница не перенесена в текст: см. изображение {where}."
    )


# ---------------------------------------------------------------- extractor


class HandwrittenExtractor:
    kinds: tuple[str, ...] = ("handwritten",)
    version: str = VERSION
    prompts: tuple[str, ...] = (PROMPT_REF,)
    tier: str = TIER
    stage: str = STAGE
    batch_size: int = BATCH_SIZE

    def _source(self, ctx: ExtractContext) -> Path:
        src = pdfx.source_path(ctx)
        check_openable(src)
        return src

    def plan(self, ctx: ExtractContext) -> ExtractPlan:
        src = self._source(ctx)
        total = page_count(src)
        plan = ExtractPlan(source_id=ctx.source.id, pages_total=total)
        plan.notes.append(
            f"Страниц: {total} ({'PDF' if is_pdf(src) else 'фото'}); подготовка изображений "
            "кодом, без агента"
        )
        if not ctx.use_vision:
            plan.notes.append("Агент отключён: рукопись не будет распознана (заглушки)")
            return plan
        pending, cached = total, 0
        prepared = None if ctx.force else load_prepared(ctx.out_dir / "pages", file_sha256(src))
        if prepared is not None:
            cached = sum(
                1
                for p in prepared
                if vision.cached_page(ctx, p.page_image(), FLAVOR, PROMPT_NAME) is not None
            )
            pending = total - cached
        plan.pages_vision = pending
        plan.agent_runs = vision.batches_needed(pending, self.batch_size)
        backend, model = vision.agent_model(ctx.settings, ctx.backend, tier=TIER, stage=STAGE)
        if pending:
            plan.notes.append(
                f"Распознавание агентом {backend} ({model}, сильный уровень): {pending} стр., "
                f"прогонов: {plan.agent_runs} (батчи по {self.batch_size})"
            )
        if cached:
            plan.notes.append(f"Из кэша распознавания: {cached} стр.")
        return plan

    def extract(self, ctx: ExtractContext) -> ExtractOutput:
        t0 = time.monotonic()
        src = self._source(ctx)
        sid = ctx.source.id
        warn: list[str] = []
        pages_dir = ctx.out_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        sha = file_sha256(src)
        prepared = None if ctx.force else load_prepared(pages_dir, sha)
        if prepared is None:
            ctx.emit(f"{sid}: подготовка изображений страниц")

            def progress(n: int, total: int) -> None:
                if total > 1:
                    ctx.emit(f"{sid}: страница {n}/{total} готова")

            try:
                prepared = prepare_pages(src, pages_dir, on_progress=progress)
            except ExtractError as exc:
                raise ExtractError(f"{sid}: {exc}") from exc
            _drop_stale(pages_dir, len(prepared))
            _write_prep(pages_dir, sha, prepared)
        else:
            ctx.emit(f"{sid}: изображения страниц уже подготовлены ({len(prepared)} стр.)")

        outcome = vision.VisionResult()
        if ctx.use_vision:
            outcome = vision.transcribe_pages(
                ctx,
                [p.page_image() for p in prepared],
                flavor=FLAVOR,
                tier=TIER,
                batch_size=self.batch_size,
                prompt_name=PROMPT_NAME,
                stage=STAGE,
            )
            warn += outcome.warnings

        contents: dict[int, str] = {}
        stats: dict[int, PageStats] = {}
        empty: list[int] = []
        for p in prepared:
            text = outcome.pages.get(p.number)
            if text is None:
                contents[p.number] = fallback_content(p, agent_off=not ctx.use_vision)
                continue
            contents[p.number] = text
            stats[p.number] = page_stats(text)
            if text.strip() == vision.EMPTY_PAGE:
                empty.append(p.number)
        failed = sorted(outcome.failed) if ctx.use_vision else []
        if failed:
            reasons = "; ".join(dict.fromkeys(outcome.failed.values()))
            suffix = f" ({reasons})" if reasons else ""
            warn.append(
                f"{sid}: не распознаны агентом стр. {_pages_text(failed)}{suffix} — на их "
                "месте блок uncertain"
            )
        if not ctx.use_vision:
            warn.append(
                f"{sid}: распознавание агентом отключено — {len(prepared)} стр. заменены заглушками"
            )

        body: list[str] = []
        for p in prepared:
            body.append(f"## [[{sid}:p{p.number}]] Страница {p.number}")
            body.append(contents[p.number] or vision.EMPTY_PAGE)
        body_md = pdfx.write_body(ctx.out_dir / "body.md", "\n\n".join(body))

        dpis = [p.dpi for p in prepared if p.dpi]
        quality: dict[str, Any] = {
            "pages": len(prepared),
            "pages_vision": len(outcome.pages),
            "pages_failed": len(failed),
            "uncertain_marks": sum(s.uncertain for s in stats.values()),
            "author_questions": sum(s.questions for s in stats.values()),
            "scan_dpi": round(pg.median_or_zero(dpis)) if dpis else None,
        }
        quality["notes"] = build_notes(
            stats,
            use_vision=ctx.use_vision,
            vision_done=len(outcome.pages),
            failed=failed,
            empty=empty,
            dpis=dpis,
            photo=not is_pdf(src),
        )
        ctx.emit(f"{sid}: body.md готов за {time.monotonic() - t0:.1f} с")
        return ExtractOutput(
            body_md=body_md,
            pages_total=len(prepared),
            pages_vision=len(outcome.pages),
            agent_runs=outcome.agent_runs,
            quality=quality,
            warnings=warn,
        )
