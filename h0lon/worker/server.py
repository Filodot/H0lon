"""The speech recognition worker (docs/ARCHITECTURE.md, «Colab-worker и оценка времени (M6)»).

A small FastAPI application that runs in Google Colab (`colab/h0lon_worker.ipynb`) and on this
PC alike (`python -m h0lon.worker`):

- `GET /health` — application, protocol version, model, device, whether the worker is busy.
  Open to everybody; it only says whether the token that came with the request is right.
- `POST /asr` — multipart: `audio` (WAV or M4A/AAC, anything ffmpeg reads) and the fields
  `language`, `initial_prompt`, `model` → `{"job_id": …}`.
- `GET /jobs/<id>` — status (`queued`, `running`, `done`, `error`), progress and, when done, the
  `result` in the format of `extract.asr.Transcript.to_dict()`.
- `DELETE /jobs/<id>` — stops the job (a running one at the next segment) and forgets it.

Everything but `/health` needs `Authorization: Bearer <token>`, the token being the environment
variable `H0LON_WORKER_TOKEN`; a worker without a token answers 401 to everything else. The token
is checked before the body of a request is read. Recognition is `extract.asr.transcribe_local`:
the same code as on this PC, one job at a time (one GPU). The audio of a job lives in a folder of
its own and is deleted when the job ends.
"""

from __future__ import annotations

import contextlib
import hmac
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from h0lon import __version__, procutil, tools
from h0lon.config import Settings
from h0lon.extract import asr
from h0lon.extract.registry import ExtractError

APP_NAME = "h0lon-worker"
PROTOCOL = asr.WORKER_PROTOCOL
TOKEN_ENV = "H0LON_WORKER_TOKEN"
MAX_UPLOAD_ENV = "H0LON_WORKER_MAX_MB"
DEFAULT_MAX_UPLOAD_MB = 512
KEEP_FINISHED_S = 3600.0  # a finished job is remembered this long
MAX_JOBS = 200
MAX_EVENTS = 40  # messages of a job kept for `GET /jobs/<id>`
MAX_PROMPT_CHARS = 4000
AUDIO_SUFFIXES = {".wav", ".m4a", ".mp4", ".aac", ".mp3", ".ogg", ".opus", ".flac", ".webm"}
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(/[A-Za-z0-9][A-Za-z0-9._\-]*)?$")
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")

TranscribeFn = Callable[..., asr.Transcript]


# ---------------------------------------------------------------- recognition


def _is_whisper_wav(path: Path) -> bool:
    """16 kHz mono 16-bit WAV: what `asr.read_wav` takes as it is."""
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getsampwidth() == 2
                and w.getnchannels() == 1
                and w.getframerate() == asr.SAMPLE_RATE
            )
    except (wave.Error, EOFError, OSError):
        return False


def decode_to_wav(src: Path, target: Path) -> Path:
    """The sound of `src` as 16 kHz mono 16-bit WAV (ffmpeg). `ExtractError` when impossible."""
    ffmpeg = tools.find_simple("ffmpeg")
    if ffmpeg is None:
        raise ExtractError("на worker нет ffmpeg: не удаётся прочитать аудио не в формате WAV")
    res = procutil.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(asr.SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        timeout=3600,
    )
    if not res.ok or not target.is_file() or target.stat().st_size < 100:
        why = (res.stderr or res.error or "").strip().splitlines()
        raise ExtractError(
            "ffmpeg не смог прочитать присланное аудио: " + (why[-1][:200] if why else "пусто")
        )
    return target


def default_transcribe(
    audio: Path,
    *,
    language: str | None,
    initial_prompt: str,
    model: str,
    progress: Callable[[float, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> asr.Transcript:
    """What a job does: the audio to 16 kHz WAV, then `asr.transcribe_local`."""
    wav = audio if _is_whisper_wav(audio) else decode_to_wav(audio, audio.with_suffix(".16k.wav"))
    # Never `compute.asr = colab` here (the worker would send the job to itself), and no config
    # file: the model comes from the request.
    settings = Settings(compute={"asr": "auto", "asr_model": model})
    return asr.transcribe_local(
        wav,
        settings=settings,
        language=language,
        initial_prompt=initial_prompt,
        on_event=on_event,
        progress=progress,
        cancel=cancel,
        label="worker",
    )


def _gpu_name() -> str | None:
    smi = tools.find_simple("nvidia-smi")
    if smi is None:
        return None
    res = procutil.run(
        [str(smi), "--query-gpu=name", "--format=csv,noheader"],
        timeout=5,
    )
    if not res.ok:
        return None
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def probe_device() -> dict[str, Any]:
    """Device facts for `/health`: `device` is `cuda` only when ctranslate2 can really use it."""
    state = asr.probe()
    cuda = state.cuda_ready
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": _gpu_name() if cuda else None,
        "cuda_devices": state.cuda_devices,
        "faster_whisper": state.version,
        "ready": state.installed,
        "problem": state.error,
    }


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    folder: Path
    audio: Path
    language: str | None
    prompt: str
    model: str
    status: str = "queued"  # queued | running | done | error
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    done_s: float = 0.0
    total_s: float = 0.0
    events: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def view(self, *, with_result: bool = True) -> dict[str, Any]:
        total = self.total_s or 0.0
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "progress": {
                "done": round(self.done_s, 2),
                "total": round(total, 2),
                "fraction": round(min(1.0, self.done_s / total), 4) if total else 0.0,
            },
            "message": self.events[-1] if self.events else "",
            "events": list(self.events),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "seconds": (
                round((self.finished or time.time()) - self.started, 2) if self.started else None
            ),
        }
        if self.error:
            data["error"] = self.error
        if with_result and self.result is not None:
            data["result"] = self.result
        return data


class JobManager:
    """Jobs of the worker, one runs at a time (one GPU); the rest wait in the order they came."""

    def __init__(
        self,
        work_dir: Path,
        transcribe_fn: TranscribeFn,
        *,
        keep_finished_s: float = KEEP_FINISHED_S,
    ) -> None:
        self.work_dir = work_dir
        self._transcribe = transcribe_fn
        self._keep_s = keep_finished_s
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._thread: threading.Thread | None = None
        self._closed = False

    # -- queries
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge()
            return self._jobs.get(job_id)

    def counts(self) -> tuple[int, int]:
        """(running, queued)."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            return running, len(self._pending)

    # -- changes
    def new_folder(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:16]
        folder = self.work_dir / job_id
        folder.mkdir(parents=True, exist_ok=True)
        return job_id, folder

    def submit(self, job: Job) -> None:
        with self._wakeup:
            if self._closed:
                raise RuntimeError("worker останавливается")
            self._purge()
            self._jobs[job.id] = job
            self._pending.append(job.id)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, name="h0lon-worker-jobs", daemon=True
                )
                self._thread.start()
            self._wakeup.notify()

    def delete(self, job_id: str) -> bool:
        """Forget a job; a running one is stopped at its next segment. False: no such job."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            job.cancel.set()
            with contextlib.suppress(ValueError):
                self._pending.remove(job_id)
            running = job.status == "running"
        if not running:  # a running job removes its folder itself when it stops
            shutil.rmtree(job.folder, ignore_errors=True)
        return True

    def close(self) -> None:
        with self._wakeup:
            self._closed = True
            for job in self._jobs.values():
                job.cancel.set()
            self._wakeup.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)

    # -- the worker thread
    def _purge(self) -> None:
        """Forget old finished jobs (the lock is held)."""
        now = time.time()
        for job_id, job in list(self._jobs.items()):
            if job.finished is not None and now - job.finished > self._keep_s:
                self._jobs.pop(job_id, None)
        if len(self._jobs) > MAX_JOBS:
            finished = sorted(
                (j for j in self._jobs.values() if j.finished is not None),
                key=lambda j: j.finished or 0.0,
            )
            for job in finished[: len(self._jobs) - MAX_JOBS]:
                self._jobs.pop(job.id, None)

    def _loop(self) -> None:
        while True:
            with self._wakeup:
                while not self._pending and not self._closed:
                    self._wakeup.wait(timeout=1.0)
                if self._closed:
                    return
                job = self._jobs.get(self._pending.popleft())
                if job is None or job.cancel.is_set():
                    continue
                job.status = "running"
                job.started = time.time()
            self._run(job)

    def _run(self, job: Job) -> None:
        def progress(done: float, total: float) -> None:
            job.done_s, job.total_s = done, total

        def event(message: str) -> None:
            job.events.append(str(message)[:300])

        outcome: tuple[str, str | None, Any]
        try:
            transcript = self._transcribe(
                job.audio,
                language=job.language,
                initial_prompt=job.prompt,
                model=job.model,
                progress=progress,
                cancel=job.cancel.is_set,
                on_event=event,
            )
            if job.cancel.is_set():
                raise asr.AsrCancelled("задача остановлена")
            job.done_s = job.total_s or transcript.duration
            outcome = ("done", None, transcript.to_dict())
        except asr.AsrCancelled:
            outcome = ("cancelled", None, None)
        except ExtractError as exc:
            outcome = ("error", str(exc), None)
        except Exception as exc:  # a failing job must not stop the worker
            outcome = ("error", f"{type(exc).__name__}: {exc}", None)
        shutil.rmtree(job.folder, ignore_errors=True)  # the audio is not kept
        status, error, result = outcome
        with self._lock:  # one step: a reader never sees `done` without `finished`
            job.error, job.result = error, result
            job.finished = time.time()
            job.status = status


# ---------------------------------------------------------------- the application


def _suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in AUDIO_SUFFIXES else ".bin"


def create_app(
    *,
    token: str | None = None,
    work_dir: Path | None = None,
    transcribe_fn: TranscribeFn | None = None,
    probe_fn: Callable[[], dict[str, Any]] | None = None,
    default_model: str | None = None,
    max_upload_mb: int | None = None,
    keep_finished_s: float = KEEP_FINISHED_S,
) -> FastAPI:
    """The worker application.

    `token`: None reads `H0LON_WORKER_TOKEN`; an empty one means none (everything but `/health`
    is refused). `transcribe_fn` and `probe_fn` replace the recognition and the device probe
    (tests); `work_dir` is where the uploads live (a temporary folder, removed at exit, when not
    given).
    """
    secret = (os.environ.get(TOKEN_ENV, "") if token is None else token).strip()
    model_default = (default_model or asr.DEFAULT_MODEL).strip()
    limit_mb = max_upload_mb or int(os.environ.get(MAX_UPLOAD_ENV, "") or DEFAULT_MAX_UPLOAD_MB)
    limit = limit_mb * 1024 * 1024
    own_dir = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="h0lon-worker-")) if work_dir is None else Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = JobManager(root, transcribe_fn or default_transcribe, keep_finished_s=keep_finished_s)
    probe_lock = threading.Lock()
    probed: dict[str, Any] = {}
    probe = probe_fn or probe_device

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            jobs.close()
            if own_dir:
                shutil.rmtree(root, ignore_errors=True)

    app = FastAPI(
        title="H0lon worker", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.jobs = jobs
    app.state.root = root

    def authorized(request: Request) -> bool:
        if not secret:
            return False
        scheme, _, value = (request.headers.get("authorization") or "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            value.strip().encode("utf-8"), secret.encode("utf-8")
        )

    def require_auth(request: Request) -> None:
        if not secret:
            raise HTTPException(
                401,
                f"worker запущен без токена ({TOKEN_ENV}): он отвечает только на /health",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not authorized(request):
            raise HTTPException(
                401, "токен не подходит или не указан", headers={"WWW-Authenticate": "Bearer"}
            )

    def device_facts() -> dict[str, Any]:
        with probe_lock:
            if not probed:
                try:
                    probed.update(probe())
                except Exception as exc:  # a broken install must not break /health
                    probed.update({"device": "cpu", "ready": False, "problem": repr(exc)[:200]})
            return dict(probed)

    @app.get("/", response_class=PlainTextResponse)
    def index() -> str:
        return (
            "H0lon worker: работает. Состояние — /health; клиент — h0lon (compute.asr = colab).\n"
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        facts = device_facts()
        running, queued = jobs.counts()
        return {
            "ok": True,
            "app": APP_NAME,
            "protocol": PROTOCOL,
            "version": __version__,
            "model": model_default,
            "device": facts.get("device", "cpu"),
            "gpu": facts.get("gpu"),
            "cuda_devices": facts.get("cuda_devices", 0),
            "faster_whisper": facts.get("faster_whisper"),
            "ready": bool(facts.get("ready", True)),
            "problem": facts.get("problem"),
            "busy": bool(running or queued),
            "running": running,
            "queued": queued,
            "auth_configured": bool(secret),
            "authorized": authorized(request),
        }

    @app.post("/asr")
    async def submit(request: Request) -> JSONResponse:
        require_auth(request)  # before a byte of the body is read
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit + 65536:
            raise HTTPException(413, f"аудио больше {limit_mb} МБ")
        if not device_facts().get("ready", True) and transcribe_fn is None:
            raise HTTPException(503, "на worker не установлен faster-whisper")
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(400, f"не удалось разобрать запрос: {type(exc).__name__}") from None
        upload = form.get("audio")
        if upload is None or not hasattr(upload, "file"):
            raise HTTPException(422, "в запросе нет файла «audio»")
        language = str(form.get("language") or "").strip()
        prompt = str(form.get("initial_prompt") or "")
        model = str(form.get("model") or "").strip() or model_default
        if language in ("", "auto"):
            language = ""
        elif not LANGUAGE_RE.match(language):
            raise HTTPException(422, "language: ожидается код языка, например ru")
        if not MODEL_RE.match(model):
            raise HTTPException(422, "model: допустимо имя модели, например large-v3")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise HTTPException(422, f"initial_prompt длиннее {MAX_PROMPT_CHARS} знаков")

        job_id, folder = jobs.new_folder()
        target = folder / f"audio{_suffix(getattr(upload, 'filename', None))}"

        def save() -> int:
            size = 0
            with target.open("wb") as out:
                while True:
                    chunk = upload.file.read(1 << 20)
                    if not chunk:
                        return size
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"аудио больше {limit_mb} МБ")
                    out.write(chunk)

        try:
            size = await run_in_threadpool(save)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            with contextlib.suppress(Exception):
                await form.close()
        if size < 100:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, "аудио пустое")
        job = Job(
            id=job_id,
            folder=folder,
            audio=target,
            language=language or None,
            prompt=prompt,
            model=model,
        )
        job.events.append(f"принято аудио: {size / 1e6:.1f} МБ")
        jobs.submit(job)
        return JSONResponse({"job_id": job_id, "status": job.status}, status_code=202)

    @app.get("/jobs/{job_id}")
    def job_status(request: Request, job_id: str) -> dict[str, Any]:
        require_auth(request)
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "задача не найдена")
        return job.view()

    @app.delete("/jobs/{job_id}")
    def job_delete(request: Request, job_id: str) -> dict[str, Any]:
        require_auth(request)
        if not jobs.delete(job_id):
            raise HTTPException(404, "задача не найдена")
        return {"ok": True}

    return app
