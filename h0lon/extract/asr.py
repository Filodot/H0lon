"""Speech recognition with faster-whisper (docs/ARCHITECTURE.md, «Видео и аудио (M5)», ASR).

`transcribe(audio, settings=…)` turns `audio.wav` (16 kHz, mono) into a `Transcript`: segments
with word timestamps, the voice activity filter on, the glossary of the topic as the Whisper
`initial_prompt`. The device follows `compute.asr`: `auto` — CUDA when the GPU and its
libraries are there, otherwise the CPU (int8; the `small` model unless `compute.asr_model` is
set explicitly). The time a run takes goes to `<state_dir>/calibration.json` (seconds per
minute of audio for a device × model pair): it sharpens the estimates of the next plans.

The heavy packages (`faster_whisper`, `ctranslate2`, `numpy`) are imported inside functions,
so this module loads without the `video` group; a missing group is an `ExtractError` with the
install command. The audio goes to Whisper as an array read from the WAV file by us: the
decoder of faster-whisper (PyAV) is not used.

Where the recognition runs (`compute.asr`): on this PC (`transcribe_local`, the code above) or
on the Colab worker (`h0lon/worker`, protocol — docs/ARCHITECTURE.md «Colab-worker и оценка
времени (M6)»). `colab` always goes to the worker; `auto` goes there when `compute.colab_url`
is set and this PC has no usable CUDA. Only the audio leaves the PC (compressed to AAC: the
tunnel of the worker accepts about 100 MB per request). A worker that cannot be reached is a
`RemoteError`; with `compute.colab_fallback_local` (the default) `transcribe` then recognizes
the speech here. The token never goes into messages or into the URL: only the `Authorization`
header, and only to the address from the settings (redirects are not followed).
"""

from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import importlib.util
import ipaddress
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from h0lon import procutil, tools
from h0lon.extract.registry import ExtractError

if TYPE_CHECKING:
    from h0lon.config import Settings

IS_WINDOWS = sys.platform == "win32"

SAMPLE_RATE = 16000
TRANSCRIPT_FORMAT = 1
CALIBRATION_FILE = "calibration.json"
INSTALL_HINT = "uv sync --extra video (для видеокарты NVIDIA ещё --extra video-gpu)"
GPU_INSTALL_HINT = "uv sync --extra video --extra video-gpu"
DEFAULT_MODEL = "large-v3"
CPU_DEFAULT_MODEL = "small"

# Libraries ctranslate2 4.x needs on a NVIDIA card (CUDA 12, cuDNN 9). Checked on Windows
# only: elsewhere the loader finds them (or does not) without our help.
CUDA_LIBS_WINDOWS = ("cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll")
_CUDA_ERROR_RE = re.compile(
    r"cuda|cublas|cudnn|out of memory|cannot be loaded|is not found|device-side|gpu", re.IGNORECASE
)

# Whisper takes at most 223 tokens of a prompt; Russian words cost 2-4 tokens each.
PROMPT_MAX_TOKENS = 200
PROMPT_MAX_CHARS = 520

# Whisper's well-known leaks in silence and at the end of Russian recordings.
_JUNK_RE = re.compile(
    r"субтитры\s+(сделал|создавал|подогнал|делал)|редактор\s+субтитров|корректор\s+[а-яa-z]\.|"
    r"продолжение\s+следует|спасибо\s+за\s+(просмотр|внимание)\s*[.!]*$|dimatorzok|"
    r"подписывайтесь\s+на\s+канал",
    re.IGNORECASE,
)

# Starting estimates (seconds of work per minute of audio) until `calibration.json` has a
# measurement: PRD 13.1. «cuda» is the local GPU (RTX 4050, large-v3: 8.1 s per minute was
# measured in M5; the PRD range is 6-12 minutes per 90-minute lecture), «colab» is a T4 behind
# the worker (8-15 minutes per 90-minute lecture, without the upload of the audio).
DEFAULT_SECONDS_PER_MINUTE: dict[tuple[str, str], float] = {
    ("cuda", "large-v3"): 8.0,
    ("cuda", "large-v3-turbo"): 4.0,
    ("cuda", "medium"): 5.0,
    ("cuda", "small"): 2.5,
    ("colab", "large-v3"): 6.0,
    ("colab", "large-v3-turbo"): 3.0,
    ("colab", "medium"): 4.0,
    ("colab", "small"): 2.0,
    ("cpu", "large-v3"): 80.0,
    ("cpu", "large-v3-turbo"): 40.0,
    ("cpu", "medium"): 30.0,
    ("cpu", "small"): 15.0,  # measured in M6 on an i5-12500H: 15.2 s per minute (int8)
}

# The way to the Colab worker (see the module docstring).
WORKER_PROTOCOL = 1
UPLOAD_KBPS = 48  # AAC bitrate of the audio that goes to the worker (16 kHz, mono)
UPLOAD_MIN_KBPS = 24  # a very long recording is squeezed down to this
UPLOAD_MAX_BYTES = 90_000_000  # the tunnel (Cloudflare) refuses requests over 100 MB
DEFAULT_UPLOAD_BYTES_PER_S = 500_000  # PRD 13.1: ~45 MB take 1-2 minutes
HEALTH_TIMEOUT_S = 10.0
POLL_FAILURES_S = 120.0  # polling that fails for this long means the worker is gone
_MODEL_SIZE_TEXT = {
    "large-v3": "около 3 ГБ",
    "large-v2": "около 3 ГБ",
    "large-v3-turbo": "около 1,6 ГБ",
    "turbo": "около 1,6 ГБ",
    "medium": "около 1,5 ГБ",
    "small": "около 0,5 ГБ",
    "base": "около 150 МБ",
    "tiny": "около 75 МБ",
}


# ---------------------------------------------------------------- data


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "words": [
                [
                    round(w.start, 2),
                    round(w.end, 2),
                    w.text,
                    None if w.prob is None else round(w.prob, 3),
                ]
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        words = [
            Word(float(w[0]), float(w[1]), str(w[2]), None if w[3] is None else float(w[3]))
            for w in data.get("words") or []
        ]
        return cls(float(data["start"]), float(data["end"]), str(data["text"]), words)


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None = None
    duration: float = 0.0  # seconds of audio
    model: str = ""
    device: str = ""
    compute_type: str = ""
    seconds: float = 0.0  # wall time of the recognition
    key: str = ""  # cache key (audio, model, language, glossary prompt)
    prompt: str = ""  # the initial prompt that was used
    dropped: int = 0  # segments removed as hallucinations

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TRANSCRIPT_FORMAT,
            "key": self.key,
            "language": self.language,
            "duration": round(self.duration, 2),
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "seconds": round(self.seconds, 2),
            "prompt": self.prompt,
            "dropped": self.dropped,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        return cls(
            segments=[Segment.from_dict(s) for s in data["segments"]],
            language=data.get("language"),
            duration=float(data.get("duration") or 0.0),
            model=str(data.get("model") or ""),
            device=str(data.get("device") or ""),
            compute_type=str(data.get("compute_type") or ""),
            seconds=float(data.get("seconds") or 0.0),
            key=str(data.get("key") or ""),
            prompt=str(data.get("prompt") or ""),
            dropped=int(data.get("dropped") or 0),
        )

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def format_srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: Sequence[Segment]) -> str:
    lines: list[str] = []
    for n, seg in enumerate(segments, start=1):
        lines += [
            str(n),
            f"{format_srt_time(seg.start)} --> {format_srt_time(max(seg.end, seg.start))}",
            seg.text.strip(),
            "",
        ]
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    tmp.replace(path)


def write_transcript(out_dir: Path, transcript: Transcript) -> None:
    """`transcript.json` (segments with words) and `transcript.srt` in `out_dir`."""
    _write_atomic(
        out_dir / "transcript.json",
        json.dumps(transcript.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    _write_atomic(out_dir / "transcript.srt", to_srt(transcript.segments))


def read_transcript(out_dir: Path) -> Transcript | None:
    """The transcript of `out_dir`, None when it is absent or unreadable."""
    try:
        data = json.loads((out_dir / "transcript.json").read_text(encoding="utf-8"))
        if data.get("format") != TRANSCRIPT_FORMAT:
            return None
        return Transcript.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


# ---------------------------------------------------------------- packages and CUDA


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def have_faster_whisper() -> bool:
    return package_version("faster-whisper") is not None


def require_faster_whisper() -> None:
    if not have_faster_whisper() or importlib.util.find_spec("numpy") is None:
        raise ExtractError(
            "Для видео и аудио нужна группа зависимостей video (faster-whisper, numpy): "
            f"выполните «{INSTALL_HINT}»"
        )


def require_asr(settings: Settings) -> None:
    """What video and audio need on this PC for the way `compute.asr` recognizes the speech.

    With a Colab worker in sight (`colab`, or `auto` with `compute.colab_url`) faster-whisper is
    not required here: numpy still is (key frames), and a missing package only matters when the
    worker is unreachable and the recognition falls back to this PC.
    """
    if not worker_configured(settings):
        require_faster_whisper()
        return
    if importlib.util.find_spec("numpy") is None:
        raise ExtractError(
            f"Для видео и аудио нужен numpy (группа зависимостей video): выполните «{INSTALL_HINT}»"
        )


def nvidia_lib_dirs() -> list[Path]:
    """Directories with the DLLs (Windows `bin`, elsewhere `lib`) of the pip packages
    `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (group `video-gpu`)."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    roots = list(spec.submodule_search_locations or []) if spec else []
    sub = "bin" if IS_WINDOWS else "lib"
    dirs: list[Path] = []
    for root in roots:
        with contextlib.suppress(OSError):
            dirs += [d for d in sorted(Path(root).glob(f"*/{sub}")) if d.is_dir()]
    return dirs


def prepare_cuda_path() -> list[Path]:
    """Make the CUDA libraries of the pip packages loadable by ctranslate2.

    On Windows ctranslate2 loads cuBLAS and cuDNN itself, with the process search path: the
    directories must be on `PATH` (`os.add_dll_directory` alone is not enough for it).
    Idempotent; returns the directories.
    """
    dirs = nvidia_lib_dirs()
    if not dirs:
        return []
    if IS_WINDOWS:
        for d in dirs:
            with contextlib.suppress(OSError, AttributeError):
                os.add_dll_directory(str(d))
    current = os.environ.get("PATH", "")
    known = {p.lower() for p in current.split(os.pathsep)}
    fresh = [str(d) for d in dirs if str(d).lower() not in known]
    if fresh:
        os.environ["PATH"] = (
            os.pathsep.join([*fresh, current]) if current else os.pathsep.join(fresh)
        )
    return dirs


def missing_cuda_libs() -> list[str]:
    """CUDA libraries ctranslate2 needs that cannot be found (Windows; [] elsewhere)."""
    if not IS_WINDOWS:
        return []
    search = [
        *nvidia_lib_dirs(),
        *(Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p),
    ]
    missing = []
    for lib in CUDA_LIBS_WINDOWS:
        if not any((d / lib).is_file() for d in search):
            missing.append(lib)
    return missing


@dataclass
class AsrProbe:
    """What is installed for recognition (`h0lon doctor`, device choice)."""

    installed: bool
    version: str | None = None
    ctranslate2: str | None = None
    cuda_devices: int = 0
    cuda_missing: list[str] = field(default_factory=list)  # CUDA libraries not found
    error: str | None = None

    @property
    def cuda_ready(self) -> bool:
        return self.installed and self.cuda_devices > 0 and not self.cuda_missing


def probe() -> AsrProbe:
    """Installed faster-whisper / ctranslate2 and the state of CUDA for them."""
    version = package_version("faster-whisper")
    if version is None:
        return AsrProbe(installed=False)
    out = AsrProbe(installed=True, version=version)
    try:
        prepare_cuda_path()
        import ctranslate2

        out.ctranslate2 = getattr(ctranslate2, "__version__", None)
        out.cuda_devices = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # a broken install must not break doctor or the plan
        out.error = f"{type(exc).__name__}: {exc}"
    if out.cuda_devices:
        out.cuda_missing = missing_cuda_libs()
    return out


# ---------------------------------------------------------------- device and model


@dataclass
class DeviceChoice:
    device: str  # "cuda" | "cpu"
    compute_type: str
    model: str
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.model} на {'CUDA' if self.device == 'cuda' else 'CPU'} ({self.compute_type})"


def explicit_model(settings: Settings) -> bool:
    """`compute.asr_model` was set by the user (config file or environment)."""
    return "asr_model" in settings.compute.model_fields_set


def choose_device(
    settings: Settings, found: AsrProbe | None = None, *, force_cpu: bool = False
) -> DeviceChoice:
    """Device, compute type and model by `compute.asr` and what is installed.

    `auto` (and `colab`, whose local fallback is `auto`): CUDA when the GPU and its libraries
    are usable, otherwise the CPU; on the CPU the `small` model replaces the default `large-v3`
    (unless the user named a model) with a warning about quality. `local-gpu` without a usable
    CUDA is an error.
    """
    mode = settings.compute.asr
    if mode == "api":
        raise ExtractError(
            "Распознавание речи через «api» пока не поддерживается (API — M8): задайте "
            "compute.asr = auto, local-gpu, local-cpu или colab"
        )
    if mode == "colab":
        mode = "auto"  # the worker is tried first (`transcribe`); this is the fallback to this PC
    model = (settings.compute.asr_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    explicit = explicit_model(settings)
    warnings: list[str] = []
    state = found if found is not None else probe()
    use_cuda = False
    if mode != "local-cpu" and not force_cpu:
        if state.cuda_ready:
            use_cuda = True
        elif mode == "local-gpu":
            raise ExtractError(
                "Распознавание на GPU запрошено (compute.asr = local-gpu), но "
                + cuda_problem(state)
            )
        elif state.installed and state.cuda_devices > 0:
            warnings.append(cuda_problem(state, capital=True) + " — распознавание на CPU")
    if use_cuda:
        return DeviceChoice("cuda", "float16", model, warnings)
    if not explicit and model == DEFAULT_MODEL:
        model = CPU_DEFAULT_MODEL
        warnings.append(
            "Распознавание на CPU: взята модель small (large-v3 на процессоре в разы дольше "
            "длительности записи) — термины и формулы будут искажаться заметнее; для качества "
            "нужна видеокарта NVIDIA (группа video-gpu) или compute.asr_model = large-v3"
        )
    elif model in ("large-v3", "large-v2"):
        warnings.append(
            f"Модель {model} на CPU работает дольше длительности записи (около часа на час "
            "лекции и больше)"
        )
    return DeviceChoice("cpu", "int8", model, warnings)


def cuda_problem(state: AsrProbe, *, capital: bool = False) -> str:
    if not state.installed:
        text = f"faster-whisper не установлен («{INSTALL_HINT}»)"
    elif state.cuda_devices <= 0:
        text = "видеокарта NVIDIA с CUDA не найдена"
    elif state.cuda_missing:
        text = (
            "не найдены библиотеки CUDA ("
            + ", ".join(state.cuda_missing)
            + f"): «{GPU_INSTALL_HINT}»"
        )
    else:
        text = "CUDA недоступна" + (f" ({state.error})" if state.error else "")
    return text[0].upper() + text[1:] if capital else text


def model_size_text(model: str) -> str:
    return _MODEL_SIZE_TEXT.get(model, "размер зависит от модели")


def model_cached(model: str) -> bool:
    """The model is in the local Hugging Face cache (or `model` is a local directory)."""
    if Path(model).is_dir():
        return True
    try:
        from faster_whisper.utils import download_model

        download_model(model, local_files_only=True)
        return True
    except Exception:
        return False


def _hf_cache_bytes(model: str) -> int | None:
    """Bytes in the cache directory of the model while it downloads (None: unknown)."""
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import constants

        repo = _MODELS.get(model) or model
        folder = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        if not folder.is_dir():
            return 0
        return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    except Exception:
        return None


@contextlib.contextmanager
def _download_watch(model: str, emit: Callable[[str], None]):
    """Reports the growth of the model download once in a while (it has no progress of ours)."""
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(20.0):
            size = _hf_cache_bytes(model)
            if size:
                emit(f"Загрузка модели {model}: {size / 1e6:.0f} МБ")

    thread = threading.Thread(target=watch, daemon=True, name="h0lon-model-download")
    thread.start()
    try:
        yield
    finally:
        stop.set()


# ---------------------------------------------------------------- prompt and audio


def build_prompt(title: str, terms: Sequence[str], tokenizer: Any = None) -> str:
    """Initial prompt for Whisper: the subject and the glossary of the topic in running text.

    Whisper uses the end of an over-long prompt, so the text is cut by whole terms from the
    end of the list; the limit is checked with the model's tokenizer when there is one.
    """
    head = f"Лекция: {title.strip()}." if title.strip() else "Лекция."
    seen: set[str] = set()
    clean: list[str] = []
    for term in terms:
        t = " ".join(str(term).split()).strip(" .,;:")
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            clean.append(t)

    def render(items: Sequence[str]) -> str:
        return head + (" Термины: " + ", ".join(items) + "." if items else "")

    def too_long(text: str) -> bool:
        if len(text) > PROMPT_MAX_CHARS:
            return True
        if tokenizer is None:
            return False
        try:
            return len(tokenizer.encode(" " + text.strip()).ids) > PROMPT_MAX_TOKENS
        except Exception:
            return False

    items = list(clean)
    while items and too_long(render(items)):
        items.pop()
    return render(items)


def read_wav(path: Path) -> Any:
    """float32 samples in [-1, 1] of a 16-bit mono WAV (what `audio.wav` is)."""
    import numpy as np

    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
                raise ExtractError(
                    f"{path.name}: ожидается WAV 16 кГц, моно, 16 бит — "
                    f"получено {w.getframerate()} Гц, каналов {w.getnchannels()}"
                )
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise ExtractError(f"Не удалось прочитать аудио {path.name}: {exc}") from exc
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


# ---------------------------------------------------------------- cleaning


def _norm_text(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def clean_segments(segments: Sequence[Segment]) -> tuple[list[Segment], int]:
    """Drop empty segments, Whisper's subtitle credits and runs of identical segments.

    Returns (segments, number dropped). Runs of three or more identical segments are the
    well-known looping hallucination: the first one stays.
    """
    out: list[Segment] = []
    dropped = 0
    run = 0
    previous = ""
    for seg in segments:
        text = seg.text.strip()
        norm = _norm_text(text)
        if not norm or (len(text) < 90 and _JUNK_RE.search(text)):
            dropped += 1
            continue
        if norm == previous:
            run += 1
            if run >= 2:  # the third identical segment in a row and later ones
                dropped += 1
                continue
        else:
            run, previous = 0, norm
        seg.text = text
        out.append(seg)
    return out, dropped


# ---------------------------------------------------------------- calibration


def calibration_path(settings: Settings) -> Path:
    return settings.general.state_path / CALIBRATION_FILE


def _read_calibration(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record_calibration(
    settings: Settings, *, device: str, model: str, audio_seconds: float, wall_seconds: float
) -> dict[str, Any] | None:
    """Remember how many seconds a minute of audio took for the device × model pair."""
    if audio_seconds < 20 or wall_seconds <= 0:
        return None  # too short to say anything
    path = calibration_path(settings)
    data = _read_calibration(path)
    asr = data.setdefault("asr", {})
    key = f"{device}|{model}"
    minutes = audio_seconds / 60.0
    current = wall_seconds / minutes
    previous = asr.get(key)
    if isinstance(previous, dict) and previous.get("seconds_per_minute"):
        # long runs weigh more than short ones
        w_old = float(previous.get("audio_minutes") or 0.0)
        spm = (float(previous["seconds_per_minute"]) * w_old + current * minutes) / (
            w_old + minutes
        )
        runs = int(previous.get("runs") or 0) + 1
        total_minutes = w_old + minutes
    else:
        spm, runs, total_minutes = current, 1, minutes
    entry = {
        "seconds_per_minute": round(spm, 3),
        "last_seconds_per_minute": round(current, 3),
        "runs": runs,
        "audio_minutes": round(total_minutes, 2),
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    asr[key] = entry
    data["format"] = 1
    try:
        _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    except OSError:
        return None
    return entry


def calibration_entry(settings: Settings, device: str, model: str) -> dict[str, Any] | None:
    """The measurement for a device ("cuda", "cpu", "colab") × model pair, if there is one."""
    entry = (_read_calibration(calibration_path(settings)).get("asr") or {}).get(
        f"{device}|{model}"
    )
    if isinstance(entry, dict) and entry.get("seconds_per_minute"):
        return entry
    return None


def estimate_seconds_per_minute(settings: Settings, device: str, model: str) -> tuple[float, bool]:
    """(seconds of work per minute of audio, measured?) — calibration or the PRD estimate."""
    entry = calibration_entry(settings, device, model)
    if entry is not None:
        return float(entry["seconds_per_minute"]), True
    default = DEFAULT_SECONDS_PER_MINUTE.get((device, model))
    if default is None:
        family = device if device in ("cuda", "colab") else "cpu"
        default = DEFAULT_SECONDS_PER_MINUTE[(family, "medium")]
    return default, False


def record_upload(settings: Settings, *, size: int, seconds: float) -> dict[str, Any] | None:
    """Remember how fast the audio went to the worker (bytes per second, weighted by size)."""
    if size < 500_000 or seconds <= 0:
        return None  # too small: the connection set-up dominates
    path = calibration_path(settings)
    data = _read_calibration(path)
    uploads = data.setdefault("upload", {})
    previous = uploads.get("colab")
    total_bytes, total_seconds, runs = float(size), float(seconds), 1
    if isinstance(previous, dict) and previous.get("bytes") and previous.get("seconds"):
        total_bytes += float(previous["bytes"])
        total_seconds += float(previous["seconds"])
        runs += int(previous.get("runs") or 0)
    entry = {
        "bytes_per_second": round(total_bytes / total_seconds),
        "last_bytes_per_second": round(size / seconds),
        "runs": runs,
        "bytes": round(total_bytes),
        "seconds": round(total_seconds, 2),
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    uploads["colab"] = entry
    data["format"] = 1
    try:
        _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    except OSError:
        return None
    return entry


def upload_bytes_per_second(settings: Settings) -> tuple[float, bool]:
    """(bytes per second towards the worker, measured?) — calibration or the PRD estimate."""
    entry = (_read_calibration(calibration_path(settings)).get("upload") or {}).get("colab")
    if isinstance(entry, dict) and entry.get("bytes_per_second"):
        return float(entry["bytes_per_second"]), True
    return float(DEFAULT_UPLOAD_BYTES_PER_S), False


# ---------------------------------------------------------------- recognition (this PC)


class _CudaFailure(Exception):
    """A CUDA attempt failed before the first segment: the next attempt may help."""


class AsrCancelled(ExtractError):
    """The recognition was stopped on request (the worker's `DELETE /jobs/<id>`)."""


def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _attempts(settings: Settings, found: AsrProbe) -> list[DeviceChoice]:
    first = choose_device(settings, found)
    if first.device != "cuda":
        return [first]
    int8 = DeviceChoice("cuda", "int8_float16", first.model, [])
    cpu = choose_device(settings, found, force_cpu=True)
    return [first, int8, cpu]


def _transcribe_once(
    samples: Any,
    duration: float,
    choice: DeviceChoice,
    *,
    language: str | None,
    title: str,
    terms: Sequence[str],
    emit: Callable[[str], None],
    label: str,
    initial_prompt: str | None = None,
    progress: Callable[[float, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    from faster_whisper import WhisperModel

    cached = model_cached(choice.model)
    if not cached:
        emit(
            f"{label}: загрузка модели {choice.model} с Hugging Face (первый запуск, "
            f"{model_size_text(choice.model)})"
        )
    t_load = time.monotonic()
    try:
        with _download_watch(choice.model, emit) if not cached else contextlib.nullcontext():
            model = WhisperModel(
                choice.model, device=choice.device, compute_type=choice.compute_type
            )
    except Exception as exc:
        if choice.device == "cuda" and _CUDA_ERROR_RE.search(str(exc)):
            raise _CudaFailure(str(exc)) from exc
        raise ExtractError(f"Не удалось загрузить модель {choice.model}: {exc}") from exc
    emit(f"{label}: модель {choice.label} загружена за {time.monotonic() - t_load:.0f} с")

    tokenizer = getattr(model, "hf_tokenizer", None)
    # A prompt that came ready (the worker gets it from the client) is used as it is.
    prompt = initial_prompt if initial_prompt is not None else build_prompt(title, terms, tokenizer)
    segments: list[Segment] = []
    started = time.monotonic()
    last_report = started
    try:
        iterator, info = model.transcribe(
            samples,
            language=language or None,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            # The glossary prompt then works for every window (with True it scrolls out of the
            # context after the first windows), and one wrong passage cannot start a loop.
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        for seg in iterator:
            if cancel is not None and cancel():
                raise AsrCancelled("распознавание остановлено")
            words = [
                Word(
                    float(w.start),
                    float(w.end),
                    str(w.word).strip(),
                    getattr(w, "probability", None),
                )
                for w in (seg.words or [])
                if str(w.word).strip()
            ]
            segments.append(Segment(float(seg.start), float(seg.end), str(seg.text), words))
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(float(seg.end), duration)
            now = time.monotonic()
            if now - last_report >= 20.0:
                last_report = now
                speed = seg.end / max(now - started, 1e-6)
                emit(
                    f"{label}: распознано {_fmt(seg.end)} из {_fmt(duration)} "
                    f"({speed:.1f}× быстрее записи)"
                )
    except AsrCancelled:
        raise
    except Exception as exc:
        if choice.device == "cuda" and not segments and _CUDA_ERROR_RE.search(str(exc)):
            raise _CudaFailure(str(exc)) from exc
        raise ExtractError(f"Ошибка распознавания речи: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            del model
        gc.collect()
    wall = time.monotonic() - started
    cleaned, dropped = clean_segments(segments)
    return Transcript(
        segments=cleaned,
        language=getattr(info, "language", None) or language,
        duration=duration,
        model=choice.model,
        device=choice.device,
        compute_type=choice.compute_type,
        seconds=wall,
        prompt=prompt,
        dropped=dropped,
    )


def transcribe_local(
    audio: Path,
    *,
    settings: Settings,
    language: str | None = None,
    title: str = "",
    terms: Sequence[str] = (),
    on_event: Callable[[str], None] | None = None,
    label: str = "ASR",
    initial_prompt: str | None = None,
    progress: Callable[[float, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Recognize `audio` (16 kHz mono WAV) on this PC. Raises `ExtractError` (Russian).

    A CUDA failure before the first segment (missing library, no memory) falls back to
    `int8_float16`, then to the CPU, with a message each time. `initial_prompt` replaces the
    prompt built from `title` and `terms`; `progress(done_s, total_s)` is called per segment and
    `cancel()` returning True stops the run with `AsrCancelled` (both are for the worker).
    """

    def emit(message: str) -> None:
        if on_event is not None:
            with contextlib.suppress(Exception):
                on_event(message)

    require_faster_whisper()
    found = probe()
    samples = read_wav(audio)
    duration = len(samples) / SAMPLE_RATE
    attempts = _attempts(settings, found)
    for warning in attempts[0].warnings:
        emit(f"{label}: {warning}")
    if attempts[0].device == "cuda":
        prepare_cuda_path()
    problems: list[str] = []
    for n, choice in enumerate(attempts):
        if n:
            for warning in choice.warnings:
                emit(f"{label}: {warning}")
        try:
            return _transcribe_once(
                samples,
                duration,
                choice,
                language=language,
                title=title,
                terms=terms,
                emit=emit,
                label=label,
                initial_prompt=initial_prompt,
                progress=progress,
                cancel=cancel,
            )
        except _CudaFailure as exc:
            problems.append(str(exc))
            nxt = attempts[n + 1] if n + 1 < len(attempts) else None
            emit(
                f"{label}: сбой CUDA ({str(exc)[:160]})" + (f"; пробую {nxt.label}" if nxt else "")
            )
    raise ExtractError(
        "Распознавание речи не удалось ни на GPU, ни на CPU: " + "; ".join(problems)[:400]
    )


# ---------------------------------------------------------------- recognition (Colab worker)


class RemoteError(Exception):
    """The Colab worker cannot be used; the message (Russian) never contains the token.

    `fatal`: trying again will not help (wrong token, no such job, a recording too big).
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def worker_configured(settings: Settings) -> bool:
    """Recognition may go to the worker: `colab`, or `auto` with an address set."""
    mode = settings.compute.asr
    return mode == "colab" or (mode == "auto" and bool(settings.compute.colab_url.strip()))


def wants_worker(settings: Settings, found: AsrProbe | None = None) -> bool:
    """The worker is where the speech goes now: `colab`, or `auto` with an address and no
    usable CUDA on this PC (`found` is what `probe()` returned, taken when not given)."""
    mode = settings.compute.asr
    if mode == "colab":
        return True
    if mode == "auto" and settings.compute.colab_url.strip():
        state = found if found is not None else probe()
        return not state.cuda_ready
    return False


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def worker_url(settings: Settings) -> str:
    """The address of the worker from `compute.colab_url`, checked. https is required except
    for this PC itself: the token must not travel in clear text."""
    raw = settings.compute.colab_url.strip()
    if not raw:
        raise RemoteError(
            "не задан compute.colab_url: запустите colab/h0lon_worker.ipynb и впишите "
            "адрес из последней ячейки в h0lon.toml"
        )
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        host = parts.hostname
        _port = parts.port  # a wrong port raises here
    except ValueError as exc:
        raise RemoteError(f"compute.colab_url не похож на адрес: {exc}") from exc
    if parts.scheme not in ("http", "https") or not host:
        raise RemoteError("compute.colab_url должен выглядеть как https://имя.trycloudflare.com")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise RemoteError("в compute.colab_url не должно быть логина, пароля и параметров")
    if parts.scheme == "http" and not _is_loopback(host):
        raise RemoteError(
            "compute.colab_url должен начинаться с https://: токен нельзя отправлять по http"
        )
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the token to another host: it is an error instead."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _opener(url: str) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [_NoRedirect()]
    host = urlsplit(url).hostname or ""
    if _is_loopback(host):
        handlers.append(urllib.request.ProxyHandler({}))  # this PC is never behind the proxy
    return urllib.request.build_opener(*handlers)


def _status_text(code: int, body: bytes, what: str) -> RemoteError:
    detail = ""
    with contextlib.suppress(ValueError, TypeError, AttributeError):
        parsed = json.loads(body.decode("utf-8", errors="replace"))
        detail = str(parsed.get("detail") or parsed.get("error") or "")[:200]
    if code == 401:
        return RemoteError(
            "worker не принял токен: проверьте compute.colab_token (его печатает ноутбук Colab)",
            fatal=True,
        )
    if code == 404 and what == "health":
        return RemoteError(
            "по адресу compute.colab_url нет H0lon-worker (ответ 404): проверьте адрес", fatal=True
        )
    if code == 404:
        return RemoteError("задача не найдена на worker (он был перезапущен?)", fatal=True)
    if code == 413:
        return RemoteError("запись слишком велика для worker", fatal=True)
    if code in (502, 503, 504, 520, 521, 522, 523, 524, 530):
        return RemoteError(
            f"туннель или worker не отвечают (HTTP {code}): сессия Colab закрыта или ноутбук "
            "остановлен?"
        )
    return RemoteError(f"worker ответил HTTP {code}" + (f": {detail}" if detail else ""))


def _net_reason(exc: BaseException) -> str:
    reason = getattr(exc, "reason", exc)
    text = str(reason) or type(reason).__name__
    if "timed out" in text.lower():
        return "время ожидания истекло"
    return text[:200]


def _http(
    method: str,
    url: str,
    token: str,
    *,
    what: str,
    data: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> bytes:
    request = urllib.request.Request(url, data=data, method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with _opener(url).open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = b""
        with contextlib.suppress(Exception):
            body = exc.read(4096)
        raise _status_text(exc.code, body, what) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # `from None`: the chained exception would show the request (and so the header).
        raise RemoteError(f"нет связи с worker: {_net_reason(exc)}") from None
    except Exception as exc:  # http.client errors: a cut connection, a bad status line
        raise RemoteError(f"нет связи с worker: {type(exc).__name__}") from None


def _json(body: bytes, what: str) -> dict[str, Any]:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        data = None
    if not isinstance(data, dict):
        raise RemoteError(
            f"ответ на «{what}» не похож на ответ H0lon-worker (прокси или страница ошибки?)"
        )
    return data


def _check_health(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("app") != "h0lon-worker":
        raise RemoteError("по адресу compute.colab_url отвечает не H0lon-worker", fatal=True)
    protocol = data.get("protocol")
    if protocol != WORKER_PROTOCOL:
        raise RemoteError(
            f"версия протокола worker ({protocol}) не совпадает с клиентской ({WORKER_PROTOCOL}): "
            "обновите ноутбук worker или пакет h0lon",
            fatal=True,
        )
    if data.get("authorized") is False:
        raise RemoteError(
            "worker не принял токен: проверьте compute.colab_token (его печатает ноутбук Colab)",
            fatal=True,
        )
    if data.get("ready") is False:
        raise RemoteError("на worker не установлен faster-whisper", fatal=True)
    return data


def worker_health(settings: Settings, *, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any]:
    """`GET /health` of the worker, checked (app, protocol, token). Raises `RemoteError`.

    The answer: model, device (`cuda` / `cpu`), `gpu`, `busy`, `queued`, …
    """
    url = worker_url(settings)
    token = settings.compute.colab_token.strip()
    body = _http("GET", url + "/health", token, what="health", timeout=timeout)
    return _check_health(_json(body, "health"))


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or SAMPLE_RATE)
    except (wave.Error, EOFError, OSError) as exc:
        raise ExtractError(f"Не удалось прочитать аудио {path.name}: {exc}") from exc


def upload_plan(duration_s: float) -> tuple[int, int]:
    """(AAC bitrate in kbit/s, size in bytes) of the audio that goes to the worker.

    48 kbit/s is ~32 MB for 1.5 hours; a recording that would not fit in a request of 90 MB at
    this rate is squeezed (down to 24 kbit/s, ~8 hours); beyond that `RemoteError`.
    """
    seconds = max(duration_s, 1.0)
    kbps = UPLOAD_KBPS
    size = seconds * kbps * 1000 / 8
    if size > UPLOAD_MAX_BYTES:
        kbps = max(UPLOAD_MIN_KBPS, int(UPLOAD_MAX_BYTES * 8 / seconds / 1000))
        size = seconds * kbps * 1000 / 8
    if size > UPLOAD_MAX_BYTES:
        raise RemoteError(
            f"запись длиной {_fmt(duration_s)} не помещается в один запрос к worker "
            f"(предел {UPLOAD_MAX_BYTES // 1_000_000} МБ)",
            fatal=True,
        )
    return kbps, int(size)


def _encode_for_upload(wav: Path, folder: Path, kbps: int) -> Path | None:
    """The audio as AAC in an .m4a (mono, 16 kHz); None when ffmpeg cannot do it."""
    ffmpeg = tools.find_simple("ffmpeg")
    if ffmpeg is None:
        return None
    target = folder / "audio.m4a"
    res = procutil.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "aac",
            "-b:a",
            f"{kbps}k",
            str(target),
        ],
        timeout=3600,
    )
    if res.ok and target.is_file() and target.stat().st_size > 0:
        return target
    return None


class _UploadBody:
    """A multipart/form-data body streamed from a file: head + file + tail, with progress."""

    def __init__(
        self,
        head: bytes,
        path: Path,
        tail: bytes,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._head, self._tail = head, tail
        self._file = path.open("rb")
        self._file_size = path.stat().st_size
        self.length = len(head) + self._file_size + len(tail)
        self._pos = 0
        self._on_progress = on_progress

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.length - self._pos
        out = b""
        while n > len(out) and self._pos < self.length:
            want = n - len(out)
            if self._pos < len(self._head):
                chunk = self._head[self._pos : self._pos + want]
            elif self._pos < len(self._head) + self._file_size:
                chunk = self._file.read(want)
                if not chunk:  # the file got shorter: the request cannot be completed
                    raise OSError("файл с аудио изменился во время отправки")
            else:
                offset = self._pos - len(self._head) - self._file_size
                chunk = self._tail[offset : offset + want]
            out += chunk
            self._pos += len(chunk)
        if self._on_progress is not None:
            with contextlib.suppress(Exception):
                self._on_progress(self._pos, self.length)
        return out

    def close(self) -> None:
        self._file.close()


def _multipart(
    fields: dict[str, str], file_field: str, file_name: str, content_type: str
) -> tuple[bytes, bytes, str]:
    """(head, tail, content type) of a multipart body whose last part is the file."""
    boundary = "h0lon" + secrets.token_hex(16)
    head = b""
    for name, value in fields.items():
        head += (
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n').encode()
            + value.encode("utf-8")
            + b"\r\n"
        )
    head += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_name}"\r\nContent-Type: {content_type}\r\n\r\n'
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head, tail, f"multipart/form-data; boundary={boundary}"


# Pauses between polls of a job (module level: tests make them short).
POLL_FIRST_S = 0.5
POLL_MAX_S = 3.0
REPORT_EVERY_S = 20.0


def _wait_for_job(
    url: str,
    token: str,
    job_id: str,
    *,
    duration: float,
    emit: Callable[[str], None],
    label: str,
) -> Transcript:
    limit = max(1800.0, 2.5 * duration + 600.0)
    deadline = time.monotonic() + limit
    interval = POLL_FIRST_S
    failing_since: float | None = None
    last_report = time.monotonic()
    last_status = ""
    while True:
        try:
            data = _json(
                _http("GET", f"{url}/jobs/{job_id}", token, what="job", timeout=30.0), "job"
            )
        except RemoteError as exc:
            if exc.fatal:
                raise
            now = time.monotonic()
            failing_since = failing_since if failing_since is not None else now
            if now - failing_since > POLL_FAILURES_S:
                raise RemoteError(
                    f"связь с worker потеряна больше чем на {POLL_FAILURES_S:.0f} с ({exc})"
                ) from None
            time.sleep(min(interval, 5.0))
            continue
        failing_since = None
        status = str(data.get("status") or "")
        if status == "done":
            return _transcript_from(data.get("result"))
        if status == "error":
            raise RemoteError(f"worker не смог распознать запись: {data.get('error') or '?'}")
        if status == "cancelled":
            raise RemoteError("задача была остановлена на worker")
        now = time.monotonic()
        if status != last_status:
            last_status = status
            if status == "queued":
                emit(f"{label}: Colab-worker: задача в очереди (занят другой записью)")
        if now - last_report >= REPORT_EVERY_S:
            last_report = now
            prog = data.get("progress") or {}
            done_s, total_s = float(prog.get("done") or 0.0), float(prog.get("total") or duration)
            if status == "running" and done_s > 0:
                emit(f"{label}: Colab: распознано {_fmt(done_s)} из {_fmt(total_s)}")
            elif data.get("message"):
                emit(f"{label}: Colab: {str(data['message'])[:200]}")
        if now > deadline:
            raise RemoteError(f"worker не закончил за {limit / 60:.0f} мин: задача отменена")
        time.sleep(interval)
        interval = min(interval * 1.5, POLL_MAX_S)


def _transcript_from(result: Any) -> Transcript:
    if not isinstance(result, dict) or result.get("format") != TRANSCRIPT_FORMAT:
        raise RemoteError("worker вернул транскрипт неизвестного формата: обновите worker")
    try:
        return Transcript.from_dict(result)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RemoteError(f"worker вернул повреждённый транскрипт ({type(exc).__name__})") from None


def _delete_job(url: str, token: str, job_id: str) -> None:
    with contextlib.suppress(RemoteError):
        _http("DELETE", f"{url}/jobs/{job_id}", token, what="job", timeout=10.0)


def transcribe_remote(
    audio: Path,
    *,
    settings: Settings,
    language: str | None = None,
    title: str = "",
    terms: Sequence[str] = (),
    on_event: Callable[[str], None] | None = None,
    label: str = "ASR",
) -> Transcript:
    """Recognize `audio` (16 kHz mono WAV) on the Colab worker. Raises `RemoteError`.

    Steps: `GET /health` (address, token, protocol), the audio is compressed and sent with
    `POST /asr` (only the audio and the parameters leave the PC), `GET /jobs/<id>` is polled,
    the segments come back in the format of `Transcript.to_dict()`, the job is deleted. The
    result carries `device = "colab"` and `seconds` = the time on the worker.
    """

    def emit(message: str) -> None:
        if on_event is not None:
            with contextlib.suppress(Exception):
                on_event(message)

    url = worker_url(settings)
    token = settings.compute.colab_token.strip()
    if not token:
        raise RemoteError(
            "не задан compute.colab_token: токен печатает ноутбук Colab, впишите его в h0lon.toml"
        )
    emit(f"{label}: Colab-worker: проверка связи")
    health_body = _http("GET", url + "/health", token, what="health", timeout=HEALTH_TIMEOUT_S)
    health = _check_health(_json(health_body, "health"))
    gpu = health.get("gpu") or ("GPU" if health.get("device") == "cuda" else "")
    emit(
        f"{label}: Colab-worker на связи ({gpu or 'без GPU'}, модель {health.get('model') or '?'})"
    )
    if health.get("device") != "cuda":
        emit(
            f"{label}: Colab-worker работает без GPU — распознавание будет очень медленным; "
            "в Colab: «Среда выполнения → Сменить тип среды выполнения → T4 GPU»"
        )

    model = (settings.compute.asr_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    prompt = build_prompt(title, terms)
    duration = wav_duration(audio)
    kbps, _estimated = upload_plan(duration)
    fields = {"model": model, "language": language or "", "initial_prompt": prompt}
    with tempfile.TemporaryDirectory(prefix="h0lon-upload-") as tmp:
        encoded = _encode_for_upload(audio, Path(tmp), kbps)
        if encoded is not None:
            upload, name, ctype = encoded, "audio.m4a", "audio/mp4"
        else:
            upload, name, ctype = audio, "audio.wav", "audio/wav"
            if audio.stat().st_size > UPLOAD_MAX_BYTES:
                raise RemoteError(
                    "не удалось сжать аудио (ffmpeg не найден или завершился с ошибкой), а в "
                    "исходном виде оно не поместится в запрос к worker"
                )
        size = upload.stat().st_size
        emit(f"{label}: Colab: отправка аудио ({size / 1e6:.1f} МБ, {_fmt(duration)})")
        head, tail, content_type = _multipart(fields, "audio", name, ctype)
        last_pct = -1
        t_up = time.monotonic()

        def on_progress(sent: int, total: int) -> None:
            nonlocal last_pct
            pct = int(sent * 100 / max(total, 1)) // 25 * 25
            if pct > last_pct and 0 < pct < 100:
                last_pct = pct
                emit(f"{label}: Colab: отправлено {pct}%")

        body = _UploadBody(head, upload, tail, on_progress)
        try:
            reply = _http(
                "POST",
                url + "/asr",
                token,
                what="asr",
                data=body,
                headers={"Content-Type": content_type, "Content-Length": str(body.length)},
                timeout=300.0,
            )
        finally:
            body.close()
        up_seconds = time.monotonic() - t_up
    record_upload(settings, size=size, seconds=up_seconds)
    submitted = _json(reply, "asr")
    job_id = str(submitted.get("job_id") or "")
    if not job_id:
        raise RemoteError("worker не вернул номер задачи")
    emit(f"{label}: Colab: аудио отправлено за {up_seconds:.0f} с, идёт распознавание")
    try:
        transcript = _wait_for_job(url, token, job_id, duration=duration, emit=emit, label=label)
    except BaseException:
        _delete_job(url, token, job_id)  # a job that nobody waits for must not keep the GPU
        raise
    _delete_job(url, token, job_id)
    transcript.device = "colab"
    if not transcript.duration:
        transcript.duration = duration
    return transcript


# ---------------------------------------------------------------- recognition (entry point)


def transcribe(
    audio: Path,
    *,
    settings: Settings,
    language: str | None = None,
    title: str = "",
    terms: Sequence[str] = (),
    on_event: Callable[[str], None] | None = None,
    label: str = "ASR",
) -> Transcript:
    """Recognize `audio` (16 kHz mono WAV) where `compute.asr` says. Raises `ExtractError`.

    The worker first when it is the way (`wants_worker`); a worker that fails is a message and,
    with `compute.colab_fallback_local` (the default), the recognition on this PC (with its own
    fallbacks, `transcribe_local`).
    """

    def emit(message: str) -> None:
        if on_event is not None:
            with contextlib.suppress(Exception):
                on_event(message)

    found: AsrProbe | None = None
    if settings.compute.asr == "auto" and settings.compute.colab_url.strip():
        found = probe()
    if wants_worker(settings, found):
        try:
            return transcribe_remote(
                audio,
                settings=settings,
                language=language,
                title=title,
                terms=terms,
                on_event=on_event,
                label=label,
            )
        except RemoteError as exc:
            if not settings.compute.colab_fallback_local:
                raise ExtractError(
                    f"Colab-worker недоступен: {exc}. Распознавание на этом ПК выключено "
                    "(compute.colab_fallback_local = false)"
                ) from None
            emit(f"{label}: Colab-worker недоступен — {exc}; распознаю на этом ПК")
            try:
                require_faster_whisper()
            except ExtractError as local:
                raise ExtractError(
                    f"Colab-worker недоступен ({exc}), а на этом ПК распознавать нечем: {local}"
                ) from None
    return transcribe_local(
        audio,
        settings=settings,
        language=language,
        title=title,
        terms=terms,
        on_event=on_event,
        label=label,
    )
