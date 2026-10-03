"""Background jobs of the web interface (docs/ARCHITECTURE.md, «Веб-интерфейс (M3)»).

`extract`, `build` and `approve` run in worker threads of the server process, through the same
functions as the CLI (`extract_topic`, `build_topic`, `approve_topic`). Rules:

- one active (queued or running) job per topic; a second `start` raises `TopicBusyError`;
- at most `agents.parallel_runs` jobs run at once, the rest wait in a FIFO queue («queued»);
- progress lines from `on_event` accumulate in `Job.events` (the newest `MAX_EVENTS` are kept;
  `Job.dropped` is the absolute index of `events[0]`, so a reader that counts events from the
  start still gets consistent indexes);
- the outcome is `Job.status` plus `Job.result` (`to_dict()` of the library results):
  `done` (the pipeline succeeded), `failed` (an error result or an exception, text in
  `Job.error`), `stopped` (cancelled by the user, or `build` stopped at the review gate);
- `cancel` is soft. A queued job is dropped at once. A running `build` stops at the start of the
  next stage (the stage that is running finishes); a running `extract` stops between sources.
  An agent run that is in flight is never interrupted, so caches stay consistent.

The library functions are imported when a job runs, so tests may replace them with fakes
(`monkeypatch.setattr("h0lon.synth.build.build_topic", fake)`).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from h0lon.config import Settings

log = logging.getLogger("h0lon.web")

JOB_KINDS = ("extract", "build", "approve")
ACTIVE_STATUSES = ("queued", "running")
FINISHED_STATUSES = ("done", "failed", "stopped")
KIND_TITLES = {
    "extract": "Извлечение источников",
    "build": "Сборка мастер-конспекта",
    "approve": "Одобрение извлечения",
}
STATUS_LABELS = {
    "queued": "в очереди",
    "running": "выполняется",
    "done": "готово",
    "failed": "ошибка",
    "stopped": "остановлена",
}
MAX_EVENTS = 5000
MAX_EVENT_CHARS = 2000
MAX_JOBS = 100  # finished jobs beyond this are forgotten, oldest first
_QUEUE_POLL_S = 1.0


class TopicBusyError(Exception):
    """The topic already has an active job (`job`)."""

    def __init__(self, job: Job) -> None:
        super().__init__(
            f"Для этой темы уже выполняется задача «{job.title}» ({STATUS_LABELS[job.status]})."
        )
        self.job = job


class _JobCancelled(BaseException):
    """Raised from the progress callback at a safe point to stop a job. Library code that
    guards its callback with `except Exception` lets it through."""


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Job:
    """One background task. Attributes other than `events` are written only by the manager."""

    def __init__(
        self,
        *,
        id: str,
        topic: str,
        topic_dir: Path,
        kind: str,
        params: dict[str, Any],
        max_events: int = MAX_EVENTS,
    ) -> None:
        self.id = id
        self.topic = topic  # «<курс>/<тема>», as in the page URLs
        self.topic_dir = topic_dir
        self.kind = kind
        self.params = dict(params)
        self.status = "queued"
        self.events: list[str] = []
        self.dropped = 0  # absolute index of events[0]
        self.result: dict[str, Any] = {}
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.cancel_requested = False
        self._max_events = max(1, max_events)
        self._lock = threading.Lock()

    # ---- events

    def add_event(self, text: object) -> None:
        """Append a progress message (one event per line, long lines are cut)."""
        lines = [ln.rstrip() for ln in str(text).splitlines()] or [""]
        with self._lock:
            for line in lines:
                if len(line) > MAX_EVENT_CHARS:
                    line = line[: MAX_EVENT_CHARS - 1] + "…"
                self.events.append(line)
            overflow = len(self.events) - self._max_events
            if overflow > 0:
                del self.events[:overflow]
                self.dropped += overflow

    def read_events(self, start: int = 0) -> tuple[list[str], int, int]:
        """(lines from absolute index `start`, absolute index of the first line, next index).

        A `start` below the retained range begins at the oldest retained event.
        """
        with self._lock:
            first = max(start, self.dropped, 0)
            lines = self.events[first - self.dropped :]
            return list(lines), first, self.dropped + len(self.events)

    @property
    def total_events(self) -> int:
        with self._lock:
            return self.dropped + len(self.events)

    # ---- state

    @property
    def title(self) -> str:
        return KIND_TITLES.get(self.kind, self.kind)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def duration_s(self) -> float | None:
        if self.started is None:
            return None
        return round((self.finished or time.time()) - self.started, 1)

    def to_dict(self, *, tail: int | None = 100) -> dict[str, Any]:
        """JSON view of the job. `tail`: how many of the newest events to include (None = all
        retained); `events_total` is the absolute count, so a client can resume with `from=`."""
        lines, first, total = self.read_events(0)
        if tail is not None and len(lines) > tail:
            first += len(lines) - tail
            lines = lines[-tail:]
        return {
            "id": self.id,
            "topic": self.topic,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "status_label": STATUS_LABELS.get(self.status, self.status),
            "cancel_requested": self.cancel_requested,
            "created": _iso(self.created),
            "started": _iso(self.started),
            "finished": _iso(self.finished),
            "duration_s": self.duration_s,
            "error": self.error,
            "result": self.result,
            "events": lines,
            "events_from": first,
            "events_total": total,
        }


class JobManager:
    """Registry and scheduler of the jobs of one server process."""

    def __init__(self, settings: Settings, *, max_events: int = MAX_EVENTS) -> None:
        self.settings = settings
        self.max_events = max_events
        self._cond = threading.Condition()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []  # creation order
        self._queue: deque[Job] = deque()
        self._threads: dict[str, threading.Thread] = {}
        self._running = 0

    @property
    def limit(self) -> int:
        return max(1, int(self.settings.agents.parallel_runs))

    # ---- queries

    def get(self, job_id: str) -> Job | None:
        with self._cond:
            return self._jobs.get(job_id)

    def recent(self, topic_dir: Path | None = None) -> list[Job]:
        """Jobs, newest first (all, or only those of one topic)."""
        with self._cond:
            jobs = [self._jobs[i] for i in reversed(self._order)]
        if topic_dir is not None:
            key = _topic_key(topic_dir)
            jobs = [j for j in jobs if _topic_key(j.topic_dir) == key]
        return jobs

    def active_for(self, topic_dir: Path) -> Job | None:
        return next((j for j in self.recent(topic_dir) if j.active), None)

    def latest_for(self, topic_dir: Path) -> Job | None:
        jobs = self.recent(topic_dir)
        return jobs[0] if jobs else None

    # ---- control

    def start(self, topic: str, topic_dir: Path, kind: str, **params: Any) -> Job:
        """Queue a job and start its thread. Raises `TopicBusyError` / `ValueError`."""
        if kind not in JOB_KINDS:
            raise ValueError(f"Неизвестный вид задачи: {kind}")
        topic_dir = Path(topic_dir)
        with self._cond:
            busy = next(
                (
                    self._jobs[i]
                    for i in reversed(self._order)
                    if self._jobs[i].active
                    and _topic_key(self._jobs[i].topic_dir) == _topic_key(topic_dir)
                ),
                None,
            )
            if busy is not None:
                raise TopicBusyError(busy)
            job = Job(
                id=uuid.uuid4().hex[:12],
                topic=topic,
                topic_dir=topic_dir,
                kind=kind,
                params=params,
                max_events=self.max_events,
            )
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._queue.append(job)
            self._prune()
            thread = threading.Thread(
                target=self._worker, args=(job,), name=f"h0lon-job-{job.id}", daemon=True
            )
            self._threads[job.id] = thread
        thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        """Request a soft stop. Returns False if the job is unknown or already finished."""
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or not job.active:
                return False
            job.cancel_requested = True
            if job.status == "queued":
                if job in self._queue:
                    self._queue.remove(job)
                job.status = "stopped"
                job.finished = time.time()
                job.result = {"ok": False, "reason": "cancelled", "message": "Отменена до начала."}
                job.add_event("Задача отменена до начала.")
            else:
                job.add_event(
                    "Остановка запрошена: задача остановится перед следующей стадией "
                    "(выполняемая стадия будет завершена)."
                )
            self._cond.notify_all()
            return True

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Block until the job finished (for tests and scripts). Raises TimeoutError."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        deadline = time.monotonic() + timeout
        with self._cond:
            while job.active:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"Задача {job_id} не завершилась за {timeout:g} с")
                self._cond.wait(min(left, 0.2))
        return job

    # ---- internals

    def _prune(self) -> None:
        """Forget the oldest finished jobs beyond MAX_JOBS (called under the lock)."""
        excess = len(self._order) - MAX_JOBS
        if excess <= 0:
            return
        for job_id in list(self._order):
            if excess <= 0:
                break
            if not self._jobs[job_id].active:
                self._order.remove(job_id)
                del self._jobs[job_id]
                self._threads.pop(job_id, None)
                excess -= 1

    def _worker(self, job: Job) -> None:
        with self._cond:
            while True:
                if job.status != "queued":  # cancelled while waiting
                    return
                if self._queue and self._queue[0] is job and self._running < self.limit:
                    break
                self._cond.wait(_QUEUE_POLL_S)
            self._queue.popleft()
            first = self._running == 0
            self._running += 1
            job.status = "running"
            job.started = time.time()
            self._cond.notify_all()
        status, result, error = "failed", {"ok": False}, None
        try:
            if first:
                _reset_cooling()
            job.add_event(f"Задача «{job.title}» запущена.")
            status, result = _RUNNERS[job.kind](self.settings, job)
        except _JobCancelled as exc:
            stage = str(exc)
            status = "stopped"
            result = {
                "ok": False,
                "reason": "cancelled",
                "message": f"Остановлено по запросу перед стадией «{stage}»."
                if stage
                else "Остановлено по запросу.",
            }
            job.add_event(result["message"])
        except BaseException as exc:  # a failing job is shown on the page, never raised
            error = _describe_error(exc)
            result = {"ok": False, "message": error}
            job.add_event(error)
            if not _is_user_error(exc):
                log.exception("Задача %s (%s) завершилась исключением", job.id, job.kind)
        finally:
            if status == "failed" and error is None:
                error = str(result.get("message") or "Задача завершилась с ошибкой.")
            with self._cond:
                job.result = result
                job.error = error
                job.finished = time.time()
                job.status = status
                self._running -= 1
                self._cond.notify_all()


def _topic_key(topic_dir: Path) -> str:
    try:
        return os.path.normcase(str(Path(topic_dir).resolve()))
    except OSError:
        return os.path.normcase(str(topic_dir))


def _reset_cooling() -> None:
    """A long-lived server must not keep an agent «cooling» for hours (the CLI starts a new
    process every time): the next job tries it again."""
    try:
        from h0lon.agents import reset_cooling

        reset_cooling()
    except Exception:  # bookkeeping only
        log.debug("reset_cooling failed", exc_info=True)


def _is_user_error(exc: BaseException) -> bool:
    from h0lon.sources.ingest import IngestError

    return isinstance(exc, ValueError | FileNotFoundError | IngestError)


def _describe_error(exc: BaseException) -> str:
    """Readable text for the page: library errors are already Russian sentences."""
    if _is_user_error(exc):
        return str(exc) or type(exc).__name__
    return (
        f"Внутренняя ошибка: {type(exc).__name__}: {exc}. "
        "Подробности — в журнале сервера (окно, где запущен h0lon serve)."
    )


# ---------------------------------------------------------------- runners
# Each returns (status, result). The result dict always has `ok` and `message`.


def _fmt_seconds(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " с"


def _run_extract(settings: Settings, job: Job) -> tuple[str, dict[str, Any]]:
    from h0lon.extract.pipeline import extract_topic
    from h0lon.sources.ingest import list_sources

    params = job.params
    wanted = list(params.get("source_ids") or []) or [r.id for r in list_sources(job.topic_dir)]
    if not wanted:
        message = "В теме нет источников: добавьте файлы или ссылки."
        job.add_event(message)
        return "failed", {"ok": False, "results": [], "message": message}
    results = []
    cancelled = False
    # One call per source: the same work as `h0lon extract`, but a stop request is honoured
    # between sources.
    for sid in wanted:
        if job.cancel_requested:
            cancelled = True
            break
        results.extend(
            extract_topic(
                settings,
                job.topic_dir,
                source_ids=[sid],
                force=bool(params.get("force")),
                use_vision=bool(params.get("use_vision", True)),
                backend=params.get("backend") or None,
                on_event=job.add_event,
            )
        )
    for r in results:
        if not r.ok:
            job.add_event(f"{r.source_id}: ошибка — " + "; ".join(r.errors or ["не удалось"]))
        elif r.cached:
            job.add_event(f"{r.source_id}: без изменений (кэш актуален)")
        elif r.source_md is None:
            job.add_event(f"{r.source_id}: пропущен")
        else:
            extra = f", прогонов агента: {r.agent_runs}" if r.agent_runs else ""
            job.add_event(
                f"{r.source_id}: готово — блоков {r.blocks}{extra}, {_fmt_seconds(r.duration_s)}"
            )
    ok = all(r.ok for r in results)
    failed = [r.source_id for r in results if not r.ok]
    if cancelled:
        done_ids = ", ".join(r.source_id for r in results) or "ни одного"
        message = f"Остановлено по запросу. Обработано источников: {done_ids}."
        job.add_event(message)
        return "stopped", {
            "ok": False,
            "reason": "cancelled",
            "results": [r.to_dict() for r in results],
            "message": message,
        }
    if ok:
        message = f"Извлечение завершено: источников {len(results)}."
    else:
        message = "Не удалось извлечь: " + ", ".join(failed) + "."
    job.add_event(message)
    return ("done" if ok else "failed"), {
        "ok": ok,
        "results": [r.to_dict() for r in results],
        "message": message,
    }


def _build_stage_starts() -> frozenset[str]:
    """Messages `build_topic` emits when a stage starts: `<title>…`."""
    from h0lon.synth import build as build_mod

    titles = (*build_mod.STAGE_TITLES.values(), build_mod.REVIEW_TITLE, build_mod.GIT_TITLE)
    return frozenset(f"{title}…" for title in titles)


def _run_build(settings: Settings, job: Job) -> tuple[str, dict[str, Any]]:
    from h0lon.synth import build as build_mod

    params = job.params
    stage_starts = _build_stage_starts()

    def on_event(message: str) -> None:
        job.add_event(message)
        # Raised from the main thread of build_topic between stages: no stage is interrupted.
        if job.cancel_requested and message in stage_starts:
            raise _JobCancelled(message.removesuffix("…"))

    result = build_mod.build_topic(
        settings,
        job.topic_dir,
        review=bool(params.get("review", True)),
        from_stage=params.get("from_stage") or None,
        force=bool(params.get("force")),
        backend=params.get("backend") or None,
        on_event=on_event,
    )
    data = result.to_dict()
    data["message"] = result.message
    job.add_event(result.message)
    if result.ok:
        return "done", data
    if result.stopped_at == "review":
        data["reason"] = "review"
        return "stopped", data
    return "failed", data


def _run_approve(settings: Settings, job: Job) -> tuple[str, dict[str, Any]]:
    from h0lon.synth.build import approve_topic
    from h0lon.web.present import format_iso

    info = approve_topic(settings, job.topic_dir)
    message = (
        f"Извлечение одобрено: {', '.join(info['sources'])} ({format_iso(info['approved_at'])})."
    )
    job.add_event(message)
    return "done", {"ok": True, "message": message, **info}


_RUNNERS: dict[str, Callable[[Settings, Job], tuple[str, dict[str, Any]]]] = {
    "extract": _run_extract,
    "build": _run_build,
    "approve": _run_approve,
}
