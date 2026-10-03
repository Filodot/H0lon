"""Routes of the web interface (docs/ARCHITECTURE.md, «Страницы и действия»).

Pages are rendered on the server; actions are plain form posts that answer with a redirect
(POST/redirect/GET) and leave a short message for the next page (`FlashStore`). Long work is
never done in a request: extraction, build and approval are jobs (`h0lon.web.jobs`) whose
progress the page follows over Server-Sent Events. Errors are shown as a readable Russian
message, never as a traceback.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from h0lon import __version__, tools
from h0lon.sources.ingest import IngestError, add_sources, list_sources, topic_lock
from h0lon.sources.models import SOURCE_KINDS, SourceRecord
from h0lon.synth.model import STAGES
from h0lon.web import present
from h0lon.web.jobs import STATUS_LABELS as JOB_STATUS_LABELS
from h0lon.web.jobs import Job, JobManager, TopicBusyError
from h0lon.web.render_md import SourceDocError, source_doc_html

router = APIRouter()

LOG_TAIL_LINES = 500  # lines of a job log put into the page; the rest streams over SSE
KEEPALIVE_S = 15.0
SSE_POLL_S = 0.3
MAX_FLASH_LINES = 8
MAX_RAW_SOURCE_CHARS = 200_000
BACKEND_CHOICES = (("", "по настройкам"), ("claude", "Claude"), ("codex", "Codex"))
PAGE_PNG_RE = re.compile(r"^p(\d+)\.png$", re.IGNORECASE)
SOURCE_ID_RE = re.compile(r"^[A-Z]\d+$")
_YES = {"1", "true", "on", "yes", "да"}


# ---------------------------------------------------------------- errors and messages


class WebError(Exception):
    """A request that cannot be served; shown as a readable page (status, title, message)."""

    def __init__(
        self,
        status: int,
        title: str,
        message: str = "",
        *,
        hint: str = "",
        back_url: str = "/",
        back_label: str = "К списку тем",
    ) -> None:
        super().__init__(title)
        self.status = status
        self.title = title
        self.message = message
        self.hint = hint
        self.back_url = back_url
        self.back_label = back_label


@dataclass
class Flash:
    level: str  # ok | info | warn | error
    text: str


class FlashStore:
    """Messages for the next page, kept in memory under a short token (put into the redirect
    URL). The messages are not stored in the URL: they can be long and contain file names."""

    def __init__(self, ttl_s: float = 600.0, limit: int = 200) -> None:
        self._items: OrderedDict[str, tuple[float, list[Flash]]] = OrderedDict()
        self._ttl = ttl_s
        self._limit = limit
        self._lock = threading.Lock()

    def put(self, messages: list[Flash]) -> str:
        token = secrets.token_hex(6)
        now = time.monotonic()
        with self._lock:
            self._items[token] = (now, messages)
            while self._items and (
                len(self._items) > self._limit
                or now - next(iter(self._items.values()))[0] > self._ttl
            ):
                self._items.popitem(last=False)
        return token

    def pop(self, token: str | None) -> list[Flash]:
        if not token:
            return []
        with self._lock:
            entry = self._items.pop(token, None)
        return entry[1] if entry else []


def flash(level: str, text: str) -> Flash:
    return Flash(level, text)


def _compact_flashes(level: str, lines: list[str]) -> list[Flash]:
    shown = [flash(level, line) for line in lines[:MAX_FLASH_LINES]]
    if len(lines) > MAX_FLASH_LINES:
        shown.append(flash(level, f"… и ещё {len(lines) - MAX_FLASH_LINES}"))
    return shown


def redirect(request: Request, url: str, *messages: Flash, fragment: str = "") -> RedirectResponse:
    if messages:
        token = request.app.state.flash.put(list(messages))
        url += ("&" if "?" in url else "?") + "flash=" + token
    if fragment:
        url += "#" + fragment
    return RedirectResponse(url, status_code=303)


def render(request: Request, name: str, *, status_code: int = 200, **context: Any) -> HTMLResponse:
    state = request.app.state
    base = {
        "flashes": [
            *state.flash.pop(request.query_params.get("flash")),
            *context.pop("flashes_extra", []),
        ],
        "version": __version__,
        "nav": "",
        "workspaces": str(state.settings.general.workspaces_dir),
    }
    return state.templates.TemplateResponse(
        request, name, {**base, **context}, status_code=status_code
    )


def error_response(request: Request, err: WebError) -> Response:
    if request.url.path.startswith("/jobs"):
        return JSONResponse({"error": err.title, "message": err.message}, status_code=err.status)
    return render(
        request,
        "error.html",
        status_code=err.status,
        status=err.status,
        title=err.title,
        message=err.message,
        hint=err.hint,
        back_url=err.back_url,
        back_label=err.back_label,
    )


# ---------------------------------------------------------------- topics


@dataclass(frozen=True)
class TopicRef:
    course: str  # directory name under workspaces_dir
    slug: str  # topic directory name
    path: Path

    @property
    def key(self) -> str:
        return f"{self.course}/{self.slug}"

    @property
    def url(self) -> str:
        return topic_url(self.course, self.slug)

    def file_url(self, rel: str) -> str:
        return f"/files/{quote(self.course)}/{quote(self.slug)}/{quote(rel, safe='/')}"


def topic_url(course: str, slug: str) -> str:
    return f"/t/{quote(course)}/{quote(slug)}"


def get_topic(request: Request, course: str, slug: str) -> TopicRef:
    root: Path = request.app.state.settings.general.workspaces_dir
    if present.valid_segment(course) and present.valid_segment(slug):
        path = root / course / slug
        if (path / present.TOPIC_FILE).is_file():
            return TopicRef(course, slug, path)
    raise WebError(
        404,
        "Тема не найдена",
        f"В каталоге рабочих областей нет темы «{course}/{slug}».",
        hint="Темы лежат в каталоге рабочих областей: <курс>/<тема>/topic.yaml.",
    )


def _read_topic(ref: TopicRef) -> tuple[Any, list[SourceRecord]]:
    from h0lon.workspace import load_topic

    try:
        return load_topic(ref.path), list_sources(ref.path)
    except (ValueError, FileNotFoundError, IngestError) as exc:
        raise WebError(
            422,
            "Не удалось прочитать тему",
            str(exc),
            hint="Файл topic.yaml повреждён или занят другим процессом h0lon.",
            back_url="/",
        ) from exc


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in _YES


def _save_topic_atomic(topic_dir: Path, meta: Any) -> None:
    from h0lon.workspace import save_topic

    target = topic_dir / present.TOPIC_FILE
    fd, tmp = tempfile.mkstemp(prefix=f".{present.TOPIC_FILE}.", suffix=".tmp", dir=topic_dir)
    os.close(fd)
    try:
        save_topic(Path(tmp), meta)
        for attempt in range(20):  # os.replace fails on Windows while another process reads
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------- index and new topic


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    state = request.app.state
    root: Path = state.settings.general.workspaces_dir
    groups = present.scan_topics(state.settings, state.jobs)
    return render(
        request,
        "index.html",
        nav="topics",
        groups=groups,
        root=str(root),
        root_exists=root.is_dir(),
        course_names=[g.title for g in groups],
        total=sum(len(g.topics) for g in groups),
    )


@router.post("/topics")
def create_topic(
    request: Request,
    title: Annotated[str, Form()] = "",
    course: Annotated[str, Form()] = "",
    slug: Annotated[str, Form()] = "",
) -> Response:
    from h0lon.workspace import create_topic_detailed

    settings = request.app.state.settings
    try:
        created = create_topic_detailed(
            settings, title=title, course=course, slug=slug.strip() or None
        )
    except FileExistsError as exc:
        return redirect(request, "/", flash("warn", str(exc)))
    except ValueError as exc:
        return redirect(request, "/", flash("error", str(exc)))
    except OSError as exc:
        return redirect(request, "/", flash("error", f"Не удалось создать тему: {exc}"))
    parts = created.path.relative_to(settings.general.workspaces_dir).parts
    if len(parts) != 2:  # a course name that has no letters or digits
        return redirect(
            request, "/", flash("warn", f"Тема создана: {created.path}, но не внутри курса.")
        )
    messages = [flash("ok", f"Тема «{created.meta.title}» создана.")]
    messages += [flash("warn", w) for w in created.warnings]
    return redirect(request, topic_url(*parts), *messages)


# ---------------------------------------------------------------- topic page


def _source_row(ref: TopicRef, record: SourceRecord) -> dict[str, Any]:
    q = record.quality or {}
    original = ""
    if record.url:
        original = record.url
    elif record.original_name and record.original_name != record.title:
        original = record.original_name
    file_url = ""
    if record.file and present.resolve_topic_file(ref.path, record.file) is not None:
        file_url = ref.file_url(record.file)
    has_doc = (ref.path / "extracted" / record.id / "source.md").is_file()
    return {
        "id": record.id,
        "kind": record.kind,
        "kind_label": present.kind_label(record.kind),
        "title": record.title,
        "original": original,
        "url": record.url,
        "volume": present.volume(record),
        "status": record.status,
        "status_label": present.status_label(record.status),
        "status_class": present.STATUS_CLASSES.get(record.status, "info"),
        "signals": present.quality_signals(record),
        "notes": len(present.quality_notes(record)),
        "error": record.error if record.status == "failed" else "",
        "review_url": f"{ref.url}/source/{quote(record.id)}" if has_doc else "",
        "file_url": file_url,
        "added": present.format_iso(record.added),
        "has_quality": bool(q),
    }


def _approve_state(records: list[SourceRecord]) -> tuple[bool, str]:
    """Whether «Одобрить» can work (mirrors `approve_topic`)."""
    if not records:
        return False, "Сначала добавьте источники."
    pending = [r.id for r in records if r.status not in ("extracted", "skipped")]
    if pending:
        return False, "Не все источники извлечены: " + ", ".join(pending) + "."
    if not any(r.status == "extracted" for r in records):
        return False, "Ни один источник не извлечён."
    return True, ""


def _gate_view(meta: Any, status: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    review = status["review"]
    if not review["gate"]:
        text, level = "Review gate выключен: сборка не ждёт проверки извлечения.", "muted"
    elif review["valid"]:
        text = f"Извлечение одобрено ({present.format_iso(review['approved_at'])})."
        level = "ok"
    elif review["approved"]:
        text = (
            "Одобрение устарело: изменились "
            + ", ".join(review["changed"])
            + ". Проверьте извлечение и одобрите его снова."
        )
        level = "warn"
    else:
        text, level = "Извлечение ещё не одобрено: сборка остановится на проверке.", "warn"
    flagged = [r for r in rows if r["status"] == "extracted" and (r["notes"] or _low_quality(r))]
    return {
        "enabled": bool(review["gate"]),
        "explicit": meta.review_gate,
        "text": text,
        "level": level,
        "valid": bool(review["valid"]),
        "flagged": [(r["id"], r["notes"], r["review_url"]) for r in flagged],
    }


def _low_quality(row: dict[str, Any]) -> bool:
    return any(s.level == "bad" for s in row["signals"])


def _stage_rows(status: dict[str, Any]) -> list[dict[str, Any]]:
    from h0lon.synth.build import STAGE_TITLES

    rows = []
    for name in STAGES:
        info = status["stages"].get(name) or {"state": "не выполнялась"}
        state = info.get("state", "не выполнялась")
        rows.append(
            {
                "name": name,
                "title": STAGE_TITLES.get(name, name),
                "state": state,
                "state_class": present.STAGE_STATE_CLASSES.get(state, "muted"),
                "duration": present.format_duration(info.get("duration_s")),
                "agent_runs": info.get("agent_runs"),
                "finished": present.format_iso(info.get("finished_at"))
                if info.get("finished_at")
                else "",
            }
        )
    return rows


def _master_view(ref: TopicRef, status: dict[str, Any]) -> dict[str, Any] | None:
    pdf = ref.path / present.MASTER_PDF
    md = ref.path / present.MASTER_MD
    if not pdf.is_file() and not md.is_file():
        return None
    view: dict[str, Any] = {"pdf_url": "", "md_url": "", "built": "", "appendices": []}
    if pdf.is_file():
        view["pdf_url"] = f"{ref.file_url(present.MASTER_PDF)}?v={int(pdf.stat().st_mtime)}"
        view["built"] = present.format_ts(pdf.stat().st_mtime)
    if md.is_file():
        view["md_url"] = ref.file_url(present.MASTER_MD)
        if pdf.is_file():
            wanted = present.appendix_links(md)
            pages = present.appendix_pages(pdf, wanted)
            view["appendices"] = [
                (
                    f"{ref.file_url(present.MASTER_PDF)}?v={int(pdf.stat().st_mtime)}#page={pages[a]}",
                    t,
                )
                for a, t in wanted
                if a in pages
            ]
    return view


def _coverage_view(status: dict[str, Any]) -> dict[str, Any] | None:
    from h0lon.synth.common import percent

    cov = status.get("coverage")
    if not cov or cov.get("total") is None:
        return None
    total, covered = int(cov.get("total") or 0), int(cov.get("covered") or 0)
    return {
        "percent": percent(covered, total),
        "value": round(100.0 * covered / total, 1) if total else 0,
        "covered": covered,
        "total": total,
        "uncovered": cov.get("uncovered") or 0,
        "rounds": cov.get("rounds") or 0,
    }


def _result_rows(job: Job) -> list[dict[str, Any]]:
    """Per-stage / per-source lines of a finished job for the «Итог» block."""
    result = job.result or {}
    rows: list[dict[str, Any]] = []
    if job.kind == "build":
        from h0lon.synth.build import GIT_TITLE, REVIEW_TITLE, STAGE_TITLES

        titles = {**STAGE_TITLES, "review": REVIEW_TITLE, "git": GIT_TITLE}
        for st in result.get("stages") or []:
            if st.get("cached"):
                verdict, cls = "из кэша", "info"
            elif st.get("ok"):
                verdict, cls = "готово", "ok"
            elif st.get("stage") == "review":
                verdict, cls = "нужно одобрение", "warn"
            else:
                verdict, cls = "ошибка", "bad"
            rows.append(
                {
                    "title": titles.get(st.get("stage"), st.get("stage")),
                    "verdict": verdict,
                    "class": cls,
                    "errors": list(st.get("errors") or [])[:5],
                    "warnings": list(st.get("warnings") or [])[:5],
                }
            )
    elif job.kind == "extract":
        for res in result.get("results") or []:
            if not res.get("ok"):
                verdict, cls = "ошибка", "bad"
            elif res.get("cached"):
                verdict, cls = "без изменений", "info"
            elif not res.get("source_md"):
                verdict, cls = "пропущен", "muted"
            else:
                verdict, cls = "готово", "ok"
            rows.append(
                {
                    "title": res.get("source_id"),
                    "verdict": verdict,
                    "class": cls,
                    "errors": list(res.get("errors") or [])[:5],
                    "warnings": list(res.get("warnings") or [])[:5],
                }
            )
    return rows


def job_view(job: Job) -> dict[str, Any]:
    lines, _first, total = job.read_events(0)
    shown = lines[-LOG_TAIL_LINES:]
    result = job.result or {}
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "label": JOB_STATUS_LABELS.get(job.status, job.status),
        "active": job.active,
        "cancel_requested": job.cancel_requested,
        "log": "\n".join(shown),
        "total": total,
        "shortened": len(shown) < total,
        "message": result.get("message") or job.error or "",
        "reason": result.get("reason") or "",
        "rows": _result_rows(job) if not job.active else [],
        "duration": present.format_duration(job.duration_s),
        "started": present.format_ts(job.started),
    }


def _topic_context(request: Request, ref: TopicRef) -> dict[str, Any]:
    from h0lon.synth.build import topic_status

    state = request.app.state
    settings = state.settings
    meta, records = _read_topic(ref)
    try:
        status = topic_status(settings, ref.path)
    except (ValueError, FileNotFoundError, IngestError) as exc:
        raise WebError(422, "Не удалось прочитать состояние темы", str(exc)) from exc
    rows = [_source_row(ref, r) for r in records]
    can_approve, approve_reason = _approve_state(records)
    active = state.jobs.active_for(ref.path)
    last = active or state.jobs.latest_for(ref.path)
    parallel = max(1, int(settings.agents.parallel_runs))
    return {
        "nav": "topics",
        "ref": ref,
        "title": meta.title,
        "course_title": meta.course,
        "path": str(ref.path),
        "rows": rows,
        "gate": _gate_view(meta, status, rows),
        "can_approve": can_approve,
        "approve_reason": approve_reason,
        "stages": _stage_rows(status),
        "coverage": _coverage_view(status),
        "master": _master_view(ref, status),
        "job": job_view(last) if last is not None else None,
        "busy": active is not None,
        "parallel": parallel,
        "kinds": [(k, present.kind_label(k)) for k in SOURCE_KINDS],
        "stage_choices": [("", "с начала, по кэшу")] + [(s, s) for s in STAGES],
        "backends": BACKEND_CHOICES,
        "plan": None,
    }


@router.get("/t/{course}/{slug}", response_class=HTMLResponse)
def topic_page(request: Request, course: str, slug: str) -> Response:
    ref = get_topic(request, course, slug)
    return render(request, "topic.html", **_topic_context(request, ref))


# ---------------------------------------------------------------- sources


def _merge_report(report: Any, added: list[str], warnings: list[str]) -> None:
    added.extend(f"{r.id} «{r.title}»" for r in report.added)
    warnings.extend(report.warnings)


@router.post("/t/{course}/{slug}/sources")
def add_topic_sources(
    request: Request,
    course: str,
    slug: str,
    files: Annotated[list[UploadFile] | None, File()] = None,
    links: Annotated[str, Form()] = "",
    kind: Annotated[str, Form()] = "",
) -> Response:
    state = request.app.state
    ref = get_topic(request, course, slug)
    url = ref.url
    kind = kind.strip().lower()
    if kind and kind not in SOURCE_KINDS:
        return redirect(request, url, flash("error", f"Неизвестный вид источника: {kind}."))
    active = state.jobs.active_for(ref.path)
    if active is not None:
        return redirect(
            request,
            url,
            flash("warn", f"Источники можно добавить после задачи «{active.title}»."),
            fragment="sources",
        )
    urls = present.split_links(links)
    uploads = [f for f in (files or []) if f.filename]
    if not uploads and not urls:
        return redirect(
            request, url, flash("warn", "Выберите файлы или вставьте ссылки."), fragment="sources"
        )

    added: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h0lon-upload-", ignore_cleanup_errors=True) as tmp:
        taken: set[str] = set()
        paths: list[str] = []
        for upload in uploads:
            dest = Path(tmp) / present.safe_upload_name(upload.filename, taken)
            with dest.open("wb") as out:
                shutil.copyfileobj(upload.file, out, 1 << 20)
            paths.append(str(dest))
        # Files and links go separately: `kind` that fits a file (pdf-text, …) is an error
        # for a link, and one bad item must not hide the result for the others.
        batches = []
        if paths:
            batches.append((paths, kind or None))
        if urls:
            batches.append((urls, kind if kind in ("web", "video", "audio") else None))
        for items, batch_kind in batches:
            try:
                report = add_sources(state.settings, ref.path, items, kind=batch_kind)
            except IngestError as exc:
                errors.append(str(exc))
            except (ValueError, OSError) as exc:
                errors.append(f"Не удалось добавить: {exc}")
            else:
                _merge_report(report, added, warnings)

    messages: list[Flash] = []
    if added:
        messages.append(flash("ok", f"Добавлено источников: {len(added)} — " + "; ".join(added)))
    messages += _compact_flashes("error", errors)
    messages += _compact_flashes("warn", warnings)
    if not messages:
        messages.append(flash("info", "Новых источников нет."))
    return redirect(request, url, *messages, fragment="sources")


@router.post("/t/{course}/{slug}/review-gate")
def set_review_gate(
    request: Request, course: str, slug: str, enabled: Annotated[str, Form()] = ""
) -> Response:
    from h0lon.workspace import load_topic

    ref = get_topic(request, course, slug)
    value = _flag(enabled)
    try:
        with topic_lock(ref.path):
            meta = load_topic(ref.path)
            meta.review_gate = value
            _save_topic_atomic(ref.path, meta)
    except (ValueError, FileNotFoundError, IngestError, OSError) as exc:
        return redirect(
            request,
            ref.url,
            flash("error", f"Не удалось изменить review gate: {exc}"),
            fragment="review",
        )
    text = (
        "Review gate включён: сборка остановится, пока вы не одобрите извлечение."
        if value
        else "Review gate выключен: сборка не будет ждать проверки извлечения."
    )
    return redirect(request, ref.url, flash("ok", text), fragment="review")


# ---------------------------------------------------------------- plan and jobs


@router.post("/t/{course}/{slug}/plan", response_class=HTMLResponse)
def extract_plan(
    request: Request,
    course: str,
    slug: str,
    no_vision: Annotated[str, Form()] = "",
    force: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "",
) -> Response:
    from h0lon.extract.pipeline import extract_topic

    state = request.app.state
    ref = get_topic(request, course, slug)
    context = _topic_context(request, ref)
    try:
        plans = extract_topic(
            state.settings,
            ref.path,
            force=_flag(force),
            use_vision=not _flag(no_vision),
            dry_run=True,
            backend=_backend(backend),
        )
    except (ValueError, FileNotFoundError, IngestError) as exc:
        context["flashes_extra"] = [flash("error", str(exc))]
        return render(request, "topic.html", **context)
    runs = sum(p.agent_runs for p in plans)
    parallel = context["parallel"]
    waves = -(-runs // parallel) if runs else 0  # ceil
    context["plan"] = {
        "rows": [
            {
                "id": p.source_id,
                "pages": p.pages_total,
                "vision": p.pages_vision,
                "runs": p.agent_runs,
                "notes": list(p.notes),
            }
            for p in plans
        ],
        "runs": runs,
        "vision": sum(p.pages_vision for p in plans),
        "pages": sum(p.pages_total for p in plans),
        "eta": f"{waves * 3}–{waves * 6} мин" if waves else "",
    }
    return render(request, "topic.html", **context)


def _backend(value: str | None) -> str | None:
    value = (value or "").strip().lower()
    return value if value in ("claude", "codex") else None


def _start_job(request: Request, ref: TopicRef, kind: str, **params: Any) -> Response:
    try:
        job = request.app.state.jobs.start(ref.key, ref.path, kind, **params)
    except TopicBusyError as exc:
        return redirect(request, ref.url, flash("warn", str(exc)), fragment="job")
    return redirect(
        request, ref.url, flash("info", f"Задача «{job.title}» запущена."), fragment="job"
    )


@router.post("/t/{course}/{slug}/extract")
def start_extract(
    request: Request,
    course: str,
    slug: str,
    no_vision: Annotated[str, Form()] = "",
    force: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "",
) -> Response:
    ref = get_topic(request, course, slug)
    return _start_job(
        request,
        ref,
        "extract",
        force=_flag(force),
        use_vision=not _flag(no_vision),
        backend=_backend(backend),
    )


@router.post("/t/{course}/{slug}/build")
def start_build(
    request: Request,
    course: str,
    slug: str,
    from_stage: Annotated[str, Form()] = "",
    force: Annotated[str, Form()] = "",
    no_review: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "",
) -> Response:
    ref = get_topic(request, course, slug)
    from_stage = from_stage.strip()
    if from_stage and from_stage not in STAGES:
        return redirect(request, ref.url, flash("error", f"Неизвестная стадия: {from_stage}."))
    return _start_job(
        request,
        ref,
        "build",
        review=not _flag(no_review),
        from_stage=from_stage or None,
        force=_flag(force),
        backend=_backend(backend),
    )


@router.post("/t/{course}/{slug}/approve")
def start_approve(request: Request, course: str, slug: str) -> Response:
    ref = get_topic(request, course, slug)
    return _start_job(request, ref, "approve")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> Response:
    jobs: JobManager = request.app.state.jobs
    job = jobs.get(job_id)
    if job is None:
        raise WebError(404, "Задача не найдена", "Журнал задач живёт, пока работает сервер.")
    base = "/t/" + "/".join(quote(p) for p in job.topic.split("/"))
    if jobs.cancel(job_id):
        return redirect(request, base, flash("info", "Остановка запрошена."), fragment="job")
    return redirect(request, base, flash("info", "Задача уже завершена."), fragment="job")


@router.get("/jobs/{job_id}")
def job_json(request: Request, job_id: str) -> Response:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise WebError(404, "Задача не найдена", "Журнал задач живёт, пока работает сервер.")
    return JSONResponse(jsonable_encoder(job.to_dict()))


def _sse(data: str, *, event: str | None = None, event_id: int | None = None) -> str:
    out = ""
    if event_id is not None:
        out += f"id: {event_id}\n"
    if event:
        out += f"event: {event}\n"
    return out + f"data: {data}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(
    request: Request, job_id: str, from_: Annotated[int, Query(alias="from", ge=0)] = 0
) -> Response:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise WebError(404, "Задача не найдена", "Журнал задач живёт, пока работает сервер.")
    start = from_
    last_id = request.headers.get("last-event-id", "")
    if last_id.isdigit():  # EventSource reconnect: continue after the last delivered line
        start = max(start, int(last_id))

    async def stream() -> Any:
        pos = start
        status = None
        idle = 0.0
        yield "retry: 3000\n\n"
        while True:
            active = job.active  # read first: events added before the status flip are not lost
            if job.status != status:
                status = job.status
                payload = {"status": status, "label": JOB_STATUS_LABELS.get(status, status)}
                yield _sse(json.dumps(payload, ensure_ascii=False), event="status")
            lines, first, pos = job.read_events(pos)
            for i, line in enumerate(lines):
                yield _sse(line, event_id=first + i + 1)
            if not active:
                payload = {
                    "status": job.status,
                    "label": JOB_STATUS_LABELS.get(job.status, job.status),
                    "message": (job.result or {}).get("message") or job.error or "",
                    "total": pos,
                }
                yield _sse(json.dumps(payload, ensure_ascii=False), event="done")
                return
            if lines:
                idle = 0.0
            else:
                idle += SSE_POLL_S
                if idle >= KEEPALIVE_S:
                    yield ": keep-alive\n\n"
                    idle = 0.0
            await asyncio.sleep(SSE_POLL_S)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- source review


@router.get("/t/{course}/{slug}/source/{source_id}", response_class=HTMLResponse)
def source_review(request: Request, course: str, slug: str, source_id: str) -> Response:
    from h0lon.synth.build import review_state

    state = request.app.state
    ref = get_topic(request, course, slug)
    if not SOURCE_ID_RE.match(source_id):
        raise WebError(404, "Источник не найден", back_url=ref.url, back_label="К теме")
    meta, records = _read_topic(ref)
    record = next((r for r in records if r.id == source_id), None)
    if record is None:
        raise WebError(
            404,
            "Источник не найден",
            f"В теме нет источника {source_id}.",
            back_url=ref.url,
            back_label="К теме",
        )
    out_dir = ref.path / "extracted" / source_id
    source_md = out_dir / "source.md"
    files_base = ref.file_url(f"extracted/{source_id}/")

    pages: list[dict[str, Any]] = []
    pages_dir = out_dir / "pages"
    if pages_dir.is_dir():
        for entry in pages_dir.iterdir():
            m = PAGE_PNG_RE.match(entry.name)
            if m and entry.is_file():
                pages.append(
                    {
                        "number": int(m.group(1)),
                        "url": ref.file_url(f"extracted/{source_id}/pages/{entry.name}"),
                    }
                )
        pages.sort(key=lambda p: p["number"])

    original: dict[str, str] | None = None
    if record.file:
        target = present.resolve_topic_file(ref.path, record.file)
        if target is not None:
            suffix = target.suffix.lower()
            original = {
                "url": ref.file_url(record.file),
                "name": target.name,
                "type": "pdf"
                if suffix == ".pdf"
                else "image"
                if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
                else "file",
            }

    doc_html = doc_error = raw_text = None
    if source_md.is_file():
        try:
            doc_html = source_doc_html(
                source_md,
                files_base=files_base,
                pandoc=tools.find_pandoc(state.settings.render.pandoc),
            )
        except SourceDocError as exc:
            doc_error = str(exc)
        except OSError as exc:
            doc_error = f"Не удалось прочитать source.md: {exc}"
        if doc_html is None and doc_error:
            try:
                raw_text = source_md.read_text(encoding="utf-8", errors="replace")[
                    :MAX_RAW_SOURCE_CHARS
                ]
            except OSError:
                raw_text = None

    extracted = [r.id for r in records if (ref.path / "extracted" / r.id / "source.md").is_file()]
    neighbours: dict[str, str] = {}
    if source_id in extracted:
        i = extracted.index(source_id)
        if i > 0:
            neighbours["prev"] = extracted[i - 1]
        if i + 1 < len(extracted):
            neighbours["next"] = extracted[i + 1]
    can_approve, approve_reason = _approve_state(records)
    return render(
        request,
        "source.html",
        nav="topics",
        ref=ref,
        title=meta.title,
        record=_source_row(ref, record),
        notes=present.quality_notes(record),
        quality_items=present.quality_items(record),
        pages=pages,
        original=original,
        doc_html=doc_html,
        doc_error=doc_error,
        raw_text=raw_text,
        has_doc=source_md.is_file(),
        neighbours=neighbours,
        can_approve=can_approve,
        approve_reason=approve_reason,
        busy=state.jobs.active_for(ref.path) is not None,
        gate_enabled=bool(review_state(state.settings, ref.path, records)["gate"]),
    )


# ---------------------------------------------------------------- files and doctor


@router.get("/files/{course}/{slug}/{path:path}")
def topic_file(request: Request, course: str, slug: str, path: str) -> Response:
    ref = get_topic(request, course, slug)
    target = present.resolve_topic_file(ref.path, path)
    if target is None:
        raise WebError(404, "Файл не найден", back_url=ref.url, back_label="К теме")
    media_type, attachment, headers = present.file_delivery(target)
    return FileResponse(
        target,
        media_type=media_type,
        headers=headers,
        filename=target.name if attachment else None,
        content_disposition_type="attachment" if attachment else "inline",
    )


DOCTOR_STATUS = {
    "ok": ("готово", "ok"),
    "warn": ("внимание", "warn"),
    "missing": ("нет", "bad"),
    "error": ("ошибка", "bad"),
    "info": ("инфо", "info"),
}


@router.get("/doctor", response_class=HTMLResponse)
def doctor_page(request: Request, no_auth: int = 0) -> Response:
    from h0lon import doctor as doctor_mod

    settings = request.app.state.settings
    checks = doctor_mod.run_checks(settings, check_auth=not no_auth)
    rows = []
    for c in checks:
        label, cls = DOCTOR_STATUS.get(c.status, (c.status, "info"))
        rows.append(
            {
                "title": c.title,
                "status": c.status,
                "label": label,
                "class": cls,
                "detail": c.detail,
                "hint": c.hint if c.status != "ok" else "",
                "required": c.required,
            }
        )
    failed = [c.title for c in checks if c.required and c.status != "ok"]
    return render(
        request,
        "doctor.html",
        nav="doctor",
        rows=rows,
        failed=failed,
        warnings=sum(
            1 for c in checks if not c.required and c.status in ("warn", "missing", "error")
        ),
        config=str(settings.source_path) if settings.source_path else "значения по умолчанию",
        no_auth=bool(no_auth),
        state_dir=str(settings.general.state_path),
    )
