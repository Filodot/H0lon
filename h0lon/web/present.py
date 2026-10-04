"""Presentation helpers of the web interface: formatting, quality signals, topic cards, safe paths.

Everything here is pure (no FastAPI): the routes in `views.py` assemble pages from these pieces,
and the unit tests exercise them directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from h0lon.sources.ingest import KIND_LABELS, STATUS_LABELS
from h0lon.sources.models import SourceRecord

if TYPE_CHECKING:
    from h0lon.config import Settings
    from h0lon.web.jobs import JobManager

MASTER_PDF = "master.pdf"
MASTER_MD = "master.md"
TOPIC_FILE = "topic.yaml"

# CSS class of a source status badge.
STATUS_CLASSES = {
    "added": "info",
    "extracting": "warn",
    "extracted": "ok",
    "failed": "bad",
    "skipped": "muted",
}
STAGE_STATE_CLASSES = {
    "готово": "ok",
    "из кэша": "info",
    "устарело": "warn",
    "ошибка": "bad",
    "не выполнялась": "muted",
}
APPENDICES = (
    ("app:conflicts", "Расхождения между источниками"),
    ("app:corrections", "Журнал правок"),
    ("app:editorial", "Редакторские дополнения"),
    ("app:coverage", "Карта покрытия"),
)


# ---------------------------------------------------------------- formatting


def format_size(size: int | float | None) -> str:
    if not size:
        return "—"
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} КБ"
    return f"{size / (1024 * 1024):.1f} МБ".replace(".", ",")


def format_ts(ts: float | None) -> str:
    """Local date and time of a POSIX timestamp."""
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")


def format_iso(value: object) -> str:
    """Local date and time of an ISO 8601 string (`…Z` or with an offset); unknown → as is."""
    if not value:
        return "—"
    text = str(value)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%d.%m.%Y %H:%M")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = round(seconds)
    if total < 60:
        return f"{seconds:.1f} с".replace(".", ",") if seconds < 10 else f"{total} с"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {sec:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


def volume(record: SourceRecord) -> str:
    """«31 сл.», «12 стр.», «45 мин» or the file size."""
    units = record.units or {}
    if record.kind == "slides" and units.get("slides"):
        return f"{int(units['slides'])} сл."
    if units.get("pages"):
        return f"{int(units['pages'])} стр."
    if units.get("slides"):
        return f"{int(units['slides'])} сл."
    if units.get("minutes"):
        return f"{units['minutes']:g} мин"
    return format_size(record.size)


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind)


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, (status, ""))[0]


# ---------------------------------------------------------------- quality signals


@dataclass
class Signal:
    label: str
    value: str
    level: str = "info"  # ok | warn | bad | info


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def quality_signals(record: SourceRecord) -> list[Signal]:
    """Short signals for chips: what a reviewer should look at first."""
    q = record.quality or {}
    unit = "слайд" if record.kind == "slides" else "стр."
    out: list[Signal] = []
    layer = _num(q.get("text_layer"))
    if layer is not None and record.kind in ("pdf-text", "pdf-scan", "slides", "handwritten"):
        level = "ok" if layer >= 0.9 else "warn" if layer >= 0.5 else "bad"
        out.append(Signal("Текстовый слой", f"{round(layer * 100)} %", level))
    chars = _num(q.get("chars_per_page"))
    if chars is not None:
        out.append(Signal(f"Знаков на {unit}", f"{round(chars)}"))
    for key, label, level in (
        ("pages_math", "С формулами", "warn"),
        ("pages_scan", "Сканы", "warn"),
        ("pages_graphic", "Графика", "info"),
        ("pages_vision", "Распознано агентом", "info"),
    ):
        count = _num(q.get(key))
        if count:
            out.append(Signal(label, f"{int(count)}", level))
    failed = _num(q.get("pages_failed"))
    if failed:
        out.append(Signal("Не распознано", f"{int(failed)}", "bad"))
    cyr = _num(q.get("cyrillic_ratio"))
    if cyr is not None:
        out.append(Signal("Кириллица", f"{round(cyr * 100)} %"))
    dpi = _num(q.get("scan_dpi"))
    if dpi:
        out.append(Signal("Скан", f"{round(dpi)} dpi", "warn" if dpi < 150 else "info"))
    if q.get("truncated"):
        out.append(Signal("Файл", "обрезан", "bad"))
    elif q.get("repaired") or q.get("broken_pages"):
        out.append(Signal("Файл", "повреждён", "bad"))
    if q.get("encoding") and record.kind == "web":
        out.append(Signal("Кодировка", str(q["encoding"])))
    notes = q.get("notes")
    if isinstance(notes, list) and notes:
        out.append(Signal("Замечаний", str(len(notes)), "warn"))
    return out


def quality_notes(record: SourceRecord) -> list[str]:
    notes = (record.quality or {}).get("notes")
    if isinstance(notes, list):
        return [str(n) for n in notes]
    return [str(notes)] if notes else []


def quality_items(record: SourceRecord) -> list[tuple[str, str]]:
    """Every quality key with a printable value (notes are shown separately)."""
    items: list[tuple[str, str]] = []
    for key, value in (record.quality or {}).items():
        if key == "notes":
            continue
        if isinstance(value, dict | list):
            value = json.dumps(value, ensure_ascii=False)
        items.append((str(key), "—" if value is None else str(value)))
    return items


# ---------------------------------------------------------------- topics


@dataclass
class TopicCard:
    course_dir: str
    slug: str
    title: str
    course: str
    sources: int = 0
    extraction: str = ""
    extraction_class: str = "muted"
    master_pdf: bool = False
    built: str = ""
    job: str = ""
    error: str = ""

    @property
    def url(self) -> str:
        from urllib.parse import quote

        return f"/t/{quote(self.course_dir)}/{quote(self.slug)}"


@dataclass
class CourseGroup:
    name: str  # directory name (what the URLs use)
    title: str  # display name from topic.yaml
    topics: list[TopicCard] = field(default_factory=list)


def _extraction_summary(sources: list[dict[str, Any]]) -> tuple[str, str]:
    if not sources:
        return "нет источников", "muted"
    statuses = [str(s.get("status") or "added") for s in sources]
    failed = statuses.count("failed")
    relevant = [s for s in statuses if s != "skipped"]
    done = relevant.count("extracted")
    if failed:
        return f"ошибка извлечения: {failed}", "bad"
    if relevant and done == len(relevant):
        return "извлечено", "ok"
    if done:
        return f"извлечено {done} из {len(relevant)}", "warn"
    return "не извлечено", "muted"


def topic_card(
    settings: Settings, jobs: JobManager | None, course_dir: Path, topic_dir: Path
) -> TopicCard:
    from h0lon.workspace import load_topic

    card = TopicCard(
        course_dir=course_dir.name,
        slug=topic_dir.name,
        title=topic_dir.name,
        course=course_dir.name,
    )
    try:
        meta = load_topic(topic_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        card.error = str(exc)
        return card
    card.title, card.course = meta.title, meta.course
    sources = [s for s in meta.sources if isinstance(s, dict)]
    card.sources = len(sources)
    card.extraction, card.extraction_class = _extraction_summary(sources)
    pdf = topic_dir / MASTER_PDF
    try:
        stat = pdf.stat()
    except OSError:
        pass
    else:
        card.master_pdf = True
        card.built = format_ts(stat.st_mtime)
    if jobs is not None:
        active = jobs.active_for(topic_dir)
        if active is not None:
            card.job = active.title
    return card


def scan_topics(settings: Settings, jobs: JobManager | None = None) -> list[CourseGroup]:
    """Topics under `workspaces_dir`: <course>/<topic>/topic.yaml, grouped by course directory."""
    root = settings.general.workspaces_dir
    groups: list[CourseGroup] = []
    try:
        course_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name.casefold(),
        )
    except OSError:
        return groups
    for course_dir in course_dirs:
        try:
            topic_dirs = sorted(
                (p for p in course_dir.iterdir() if p.is_dir() and (p / TOPIC_FILE).is_file()),
                key=lambda p: p.name.casefold(),
            )
        except OSError:
            continue
        if not topic_dirs:
            continue
        cards = [topic_card(settings, jobs, course_dir, d) for d in topic_dirs]
        title = next((c.course for c in cards if not c.error), course_dir.name)
        groups.append(CourseGroup(name=course_dir.name, title=title, topics=cards))
    return groups


# ---------------------------------------------------------------- safe paths and names

_BAD_SEGMENT = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_BAD_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)}
_RESERVED_NAMES |= {f"lpt{i}" for i in range(1, 10)}
MAX_FILENAME_CHARS = 150


def valid_segment(segment: str) -> bool:
    """A single path component that cannot leave its directory (course, topic, source id)."""
    return (
        bool(segment)
        and segment == segment.strip()
        and not segment.startswith(".")
        and _BAD_SEGMENT.search(segment) is None
    )


def resolve_topic_file(topic_dir: Path, rel: str) -> Path | None:
    """`rel` (URL path inside the topic) → an existing file inside `topic_dir`, or None.

    Refused: empty and absolute paths, `..`, backslashes, drive letters and stream names (`:`),
    hidden entries (`.git`, `.topic.yaml.lock`), and anything that resolves (symlinks included)
    outside the topic directory.
    """
    if not rel or rel.startswith(("/", "\\")):
        return None
    parts = PurePosixPath(rel).parts
    if not parts or not all(valid_segment(p) for p in parts):
        return None
    try:
        base = topic_dir.resolve()
        target = (base.joinpath(*parts)).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return None
    except (OSError, ValueError):
        return None
    return target


_TEXT_TYPES = {
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".tex",
    ".log",
    ".csv",
    ".bib",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".css",
    ".js",
}
_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def file_delivery(path: Path) -> tuple[str, bool, dict[str, str]]:
    """(media type, as attachment, extra headers) for a topic file.

    Only PDF and raster images are served as themselves; text-like files (including HTML and
    XML of web sources) are served as plain text, so a page taken from the Internet never runs in
    this origin; SVG is an image that may not run scripts; everything else is a download.
    """
    ext = path.suffix.lower()
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-cache"}
    if ext == ".pdf":
        return "application/pdf", False, headers
    if ext in _IMAGE_TYPES:
        return _IMAGE_TYPES[ext], False, headers
    if ext == ".svg":
        headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; style-src 'unsafe-inline'"
        )
        return "image/svg+xml", False, headers
    if ext in _TEXT_TYPES:
        return "text/plain; charset=utf-8", False, headers
    return "application/octet-stream", True, headers


def safe_upload_name(raw: str | None, taken: set[str]) -> str:
    """File name for an uploaded file: the original name without any path, no characters that
    Windows forbids, no reserved device names, unique among `taken` (case-insensitive)."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _BAD_FILENAME_CHARS.sub("_", name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    ext = ext[:16]
    if not stem:
        stem = "upload"
    if stem.casefold() in _RESERVED_NAMES:
        stem = "_" + stem
    stem = stem[: max(1, MAX_FILENAME_CHARS - len(ext) - 1)]
    candidate = f"{stem}.{ext}" if ext else stem
    n = 2
    while candidate.casefold() in taken:
        candidate = f"{stem} ({n}).{ext}" if ext else f"{stem} ({n})"
        n += 1
    taken.add(candidate.casefold())
    return candidate


def split_links(text: str) -> list[str]:
    """Links textarea: one per line, blank lines ignored."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


# ---------------------------------------------------------------- variations (M7)

# What a preset gives, for the form of the topic page (value, name, hint).
VARIANT_PRESET_HINTS: dict[str, str] = {
    "brief": "Только главное: определения, ключевые теоремы, карта связей. 2–4 страницы.",
    "study": "Учебный конспект: мотивация, интуиция, формализм, примеры, типичные ошибки, "
    "вопросы для самопроверки.",
    "cheatsheet": "Шпаргалка на 1–2 листа: формулы и условия применимости, мелкий шрифт в две "
    "колонки.",
    "custom": "Документ по вашему запросу: опишите формат, объём, акцент.",
    "template": "Тот же текст мастера в другом шаблоне, без агента.",
}


def snippet(text: object, limit: int = 120) -> str:
    """One line of at most `limit` characters (for a request shown in a list)."""
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def variant_state(stale: bool) -> tuple[str, str]:
    """(label, badge class) of a variation: current or stale after a new master."""
    return ("устарело", "warn") if stale else ("актуально", "ok")


def appendix_links(master_md: Path) -> list[tuple[str, str]]:
    """(anchor id, title) of the appendices that master.md really has."""
    try:
        text = master_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [(anchor, title) for anchor, title in APPENDICES if "{#" + anchor in text]


_PAGES_CACHE: dict[tuple[str, int, int], dict[str, int]] = {}
_PAGES_CACHE_SIZE = 16


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def appendix_pages(pdf: Path, appendices: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Page of each appendix heading in `pdf` (1-based), by anchor id.

    The named destinations that LaTeX writes do not carry the ids of the Markdown (appendices
    are numbered A, B, C …), so the page comes from the bookmarks; a PDF without bookmarks (the
    HTML fallback engine) is searched for the heading text from its end. An appendix that
    cannot be found is left out. The result is cached by file mtime and size.
    """
    try:
        stat = pdf.stat()
    except OSError:
        return {}
    wanted = tuple(appendices)
    key = (f"{pdf.resolve()}|{wanted}", stat.st_mtime_ns, stat.st_size)
    cached = _PAGES_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    found: dict[str, int] = {}
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            toc = [(_squash(title), page) for _lvl, title, page in doc.get_toc(simple=True)]
            for anchor, title in wanted:
                needle = _squash(title)
                page = next((p for t, p in toc if needle in t and p > 0), None)
                if page is None:
                    for number in range(doc.page_count, 0, -1):
                        if needle in _squash(doc[number - 1].get_text()):
                            page = number
                            break
                if page is not None:
                    found[anchor] = page
    except Exception:  # a damaged PDF only means «no links»
        found = {}
    if len(_PAGES_CACHE) >= _PAGES_CACHE_SIZE:
        _PAGES_CACHE.pop(next(iter(_PAGES_CACHE)))
    _PAGES_CACHE[key] = dict(found)
    return found
