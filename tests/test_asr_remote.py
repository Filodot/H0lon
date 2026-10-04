"""The client of the Colab worker (`extract/asr.py`), `doctor` and the fallback to this PC.

Two kinds of servers on localhost, nothing else on the network: a real worker application under
uvicorn in a thread (fake recognition, `test_worker.FakeRecognizer`) for the whole way, and a tiny
scripted HTTP server for answers a real worker never gives (errors, redirects, other protocols).
"""

from __future__ import annotations

import json
import math
import re
import socket
import struct
import threading
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from test_worker import TOKEN, FakeRecognizer

from h0lon import doctor, tools
from h0lon.config import Settings
from h0lon.extract import asr
from h0lon.extract import video as vd
from h0lon.extract.model import ExtractContext
from h0lon.extract.registry import ExtractError
from h0lon.sources.models import SourceRecord
from h0lon.worker import server as ws

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

FFMPEG = tools.find_simple("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не найден")


def write_wav(path: Path, seconds: float = 3.0) -> Path:
    """16 kHz mono 16-bit WAV with a tone (so that the AAC encoder has something to encode)."""
    rate = asr.SAMPLE_RATE
    frames = b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * 330 * n / rate)))
        for n in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


@pytest.fixture(autouse=True)
def quick_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "POLL_FIRST_S", 0.01)
    monkeypatch.setattr(asr, "POLL_MAX_S", 0.02)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ---------------------------------------------------------------- a real worker under uvicorn


@dataclass
class Worker:
    url: str
    app: Any
    recognizer: FakeRecognizer
    probe: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[Worker]:
    recognizer = FakeRecognizer()
    probe: dict[str, Any] = {"device": "cuda", "gpu": "Tesla T4", "ready": True}
    app = ws.create_app(
        token=TOKEN,
        work_dir=tmp_path / "worker",
        transcribe_fn=recognizer,
        probe_fn=lambda: dict(probe),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="test-worker"
    )
    thread.start()
    deadline = threading.Event()
    for _ in range(500):
        if server.started:
            break
        deadline.wait(0.01)
    assert server.started, "uvicorn did not start"
    yield Worker(f"http://127.0.0.1:{port}", app, recognizer, probe)
    server.should_exit = True
    thread.join(10)
    sock.close()


def settings_for(make_settings: Callable[..., Settings], url: str, **compute: Any) -> Settings:
    options = {"asr": "colab", "colab_url": url, "colab_token": TOKEN, **compute}
    return make_settings(compute=options)


# ---------------------------------------------------------------- a scripted server


@dataclass
class Reply:
    status: int = 200
    body: Any = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


class Scripted:
    """Answers the requests by a table `(method, path) → replies`; the last reply repeats."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[Reply]] = {}
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                owner.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "type": self.headers.get("Content-Type"),
                        "body": body,
                    }
                )
                replies = owner.routes.get((self.command, self.path.split("?")[0]))
                if not replies:
                    reply = Reply(404, {"detail": "нет такого пути"})
                else:
                    reply = replies.pop(0) if len(replies) > 1 else replies[0]
                data = (
                    reply.body
                    if isinstance(reply.body, bytes)
                    else json.dumps(reply.body, ensure_ascii=False).encode("utf-8")
                )
                self.send_response(reply.status)
                self.send_header("Content-Length", str(len(data)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def on(self, method: str, path: str, *replies: Reply) -> Scripted:
        self.routes[(method, path)] = list(replies)
        return self

    def calls(self, method: str, path: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == method and r["path"] == path]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def scripted() -> Iterator[Scripted]:
    server = Scripted()
    yield server
    server.close()


HEALTH_OK = {
    "ok": True,
    "app": "h0lon-worker",
    "protocol": asr.WORKER_PROTOCOL,
    "model": "large-v3",
    "device": "cuda",
    "gpu": "Tesla T4",
    "ready": True,
    "busy": False,
    "authorized": True,
}


def transcript_json(text: str = "Привет") -> dict[str, Any]:
    word = asr.Word(0.0, 2.0, text, 0.9)
    return asr.Transcript(
        segments=[asr.Segment(0.0, 2.0, text, [word])],
        language="ru",
        duration=3.0,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        seconds=2.5,
    ).to_dict()


def happy(scripted: Scripted, *job_replies: Reply) -> Scripted:
    scripted.on("GET", "/health", Reply(200, HEALTH_OK))
    scripted.on("POST", "/asr", Reply(202, {"job_id": "j1"}))
    scripted.on("GET", "/jobs/j1", *(job_replies or (Reply(200, done_job()),)))
    scripted.on("DELETE", "/jobs/j1", Reply(200, {"ok": True}))
    return scripted


def done_job(result: Any = "default") -> dict[str, Any]:
    return {
        "id": "j1",
        "status": "done",
        "progress": {"done": 3.0, "total": 3.0},
        "result": transcript_json() if result == "default" else result,
    }


def remote(
    settings: Settings, wav: Path, events: list[str] | None = None, **kwargs: Any
) -> asr.Transcript:
    return asr.transcribe_remote(
        wav,
        settings=settings,
        language="ru",
        title="Метрические методы",
        terms=["метод Парзена"],
        on_event=events.append if events is not None else None,
        label="V1",
        **kwargs,
    )


# ---------------------------------------------------------------- the whole way, a real worker


def test_the_audio_goes_to_the_worker_and_the_segments_come_back(
    worker: Worker,
    make_settings: Callable[..., Settings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: list[dict[str, Any]] = []
    monkeypatch.setattr(asr, "record_upload", lambda s, **kw: uploads.append(kw))
    settings = settings_for(make_settings, worker.url, asr_model="large-v3-turbo")
    events: list[str] = []
    transcript = remote(settings, write_wav(tmp_path / "audio.wav"), events)

    assert transcript.text == "Привет" and transcript.segments[0].words[0].text == "Привет"
    assert transcript.device == "colab"  # the calibration of the worker is its own pair
    assert transcript.model == "large-v3-turbo" and transcript.seconds == 1.5  # on the worker
    (call,) = worker.recognizer.calls
    assert call["language"] == "ru" and call["model"] == "large-v3-turbo"
    assert call["prompt"] == asr.build_prompt("Метрические методы", ["метод Парзена"])
    assert call["suffix"] == (".m4a" if FFMPEG else ".wav") and call["size"] > 100
    assert any("Colab-worker на связи (Tesla T4, модель large-v3)" in e for e in events)
    assert any("отправка аудио" in e for e in events) and any("отправлено за" in e for e in events)
    (upload,) = uploads
    assert upload["size"] > 100 and upload["seconds"] > 0
    assert worker.app.state.jobs._jobs == {}  # the client deleted the job


@needs_ffmpeg
def test_the_audio_is_compressed_before_it_is_sent(
    worker: Worker, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    wav = write_wav(tmp_path / "audio.wav", 10.0)
    settings = settings_for(make_settings, worker.url)
    remote(settings, wav)
    (call,) = worker.recognizer.calls
    assert call["suffix"] == ".m4a" and call["size"] < wav.stat().st_size / 4
    assert asr.upload_plan(10.0)[0] == asr.UPLOAD_KBPS


def test_without_ffmpeg_the_wav_goes_as_it_is(
    worker: Worker, make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tools, "find_simple", lambda name: None)
    wav = write_wav(tmp_path / "audio.wav", 2.0)
    remote(settings_for(make_settings, worker.url), wav)
    (call,) = worker.recognizer.calls
    assert call["suffix"] == ".wav" and call["size"] == wav.stat().st_size
    # a recording that cannot be compressed and does not fit is an error, not a failed upload
    monkeypatch.setattr(asr, "UPLOAD_MAX_BYTES", 20_000)
    with pytest.raises(asr.RemoteError, match="не удалось сжать"):
        remote(settings_for(make_settings, worker.url), wav)


def test_a_worker_without_a_gpu_is_said_to_be_slow(
    worker: Worker, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    worker.probe.update(device="cpu", gpu=None)
    events: list[str] = []
    remote(settings_for(make_settings, worker.url), write_wav(tmp_path / "a.wav", 1.0), events)
    assert any("без GPU" in e and "T4 GPU" in e for e in events)


def test_the_error_of_a_job_is_a_remote_error_and_the_job_is_deleted(
    worker: Worker, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    worker.recognizer.fail = ExtractError("Ошибка распознавания речи: битое аудио")
    with pytest.raises(asr.RemoteError, match="битое аудио"):
        remote(settings_for(make_settings, worker.url), write_wav(tmp_path / "a.wav", 1.0))
    assert worker.app.state.jobs._jobs == {}


def test_a_wrong_token_is_named_without_showing_it(
    worker: Worker, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = settings_for(make_settings, worker.url, colab_token="wrong-token-123")
    with pytest.raises(asr.RemoteError) as caught:
        remote(settings, write_wav(tmp_path / "a.wav", 1.0))
    assert caught.value.fatal and "токен" in str(caught.value)
    assert "wrong-token-123" not in str(caught.value) and TOKEN not in str(caught.value)
    assert worker.recognizer.calls == []


def test_a_missing_token_is_a_message_before_any_request(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(asr.RemoteError, match=r"compute\.colab_token"):
        remote(settings_for(make_settings, scripted.url, colab_token=""), tmp_path / "a.wav")
    assert scripted.requests == []


# ---------------------------------------------------------------- the entry point and the fallback


def local_stub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_local(audio: Path, **kwargs: Any) -> asr.Transcript:
        calls.append(kwargs)
        return asr.Transcript(segments=[], device="cuda", model="large-v3")

    monkeypatch.setattr(asr, "transcribe_local", fake_local)
    return calls


def remote_stub(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def fake_remote(audio: Path, **kwargs: Any) -> asr.Transcript:
        calls.append(audio)
        return asr.Transcript(segments=[], device="colab", model="large-v3")

    monkeypatch.setattr(asr, "transcribe_remote", fake_remote)
    return calls


def probe(**kwargs: Any) -> asr.AsrProbe:
    base: dict[str, Any] = {"installed": True, "version": "1.2.1", "cuda_devices": 1}
    return asr.AsrProbe(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("compute", "cuda", "goes_to_worker"),
    [
        ({"asr": "colab", "colab_url": "https://a.trycloudflare.com"}, True, True),
        ({"asr": "auto", "colab_url": "https://a.trycloudflare.com"}, False, True),
        ({"asr": "auto", "colab_url": "https://a.trycloudflare.com"}, True, False),
        ({"asr": "local-gpu", "colab_url": "https://a.trycloudflare.com"}, True, False),
        ({"asr": "local-cpu", "colab_url": "https://a.trycloudflare.com"}, False, False),
        ({"asr": "auto"}, False, False),
    ],
)
def test_the_way_follows_compute_asr(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compute: dict[str, Any],
    cuda: bool,
    goes_to_worker: bool,
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe(cuda_devices=1 if cuda else 0))
    local, far = local_stub(monkeypatch), remote_stub(monkeypatch)
    out = asr.transcribe(tmp_path / "a.wav", settings=make_settings(compute=compute))
    wanted = asr.wants_worker(make_settings(compute=compute), probe(cuda_devices=int(cuda)))
    assert wanted is goes_to_worker
    assert bool(far) is goes_to_worker and bool(local) is not goes_to_worker
    assert out.device == ("colab" if goes_to_worker else "cuda")


def test_auto_without_the_package_goes_to_the_worker(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: asr.AsrProbe(installed=False))
    far = remote_stub(monkeypatch)
    settings = make_settings(compute={"colab_url": "https://a.trycloudflare.com"})
    asr.transcribe(tmp_path / "a.wav", settings=settings)
    assert far


def test_an_unreachable_worker_falls_back_to_this_pc(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = local_stub(monkeypatch)
    monkeypatch.setattr(asr, "require_faster_whisper", lambda: None)
    url = f"http://127.0.0.1:{free_port()}"  # nobody listens
    events: list[str] = []
    out = asr.transcribe(
        write_wav(tmp_path / "a.wav", 1.0),
        settings=settings_for(make_settings, url),
        language="ru",
        title="Тема",
        terms=["ядро"],
        on_event=events.append,
        label="V1",
    )
    assert out.device == "cuda" and local[0]["language"] == "ru" and local[0]["terms"] == ["ядро"]
    assert any("Colab-worker недоступен — нет связи" in e and "на этом ПК" in e for e in events)
    assert TOKEN not in " ".join(events)


def test_without_the_fallback_an_unreachable_worker_is_an_error(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = local_stub(monkeypatch)
    url = f"http://127.0.0.1:{free_port()}"
    settings = settings_for(make_settings, url, colab_fallback_local=False)
    with pytest.raises(ExtractError, match=r"Colab-worker недоступен.*colab_fallback_local"):
        asr.transcribe(write_wav(tmp_path / "a.wav", 1.0), settings=settings)
    assert local == []


def test_a_fallback_without_faster_whisper_says_both(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = local_stub(monkeypatch)

    def missing() -> None:
        raise ExtractError("Для видео и аудио нужна группа зависимостей video")

    monkeypatch.setattr(asr, "require_faster_whisper", missing)
    url = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(ExtractError, match=r"worker недоступен.*распознавать нечем.*video"):
        asr.transcribe(
            write_wav(tmp_path / "a.wav", 1.0), settings=settings_for(make_settings, url)
        )
    assert local == []


def test_colab_without_an_address_falls_back_with_a_reason(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = local_stub(monkeypatch)
    monkeypatch.setattr(asr, "require_faster_whisper", lambda: None)
    events: list[str] = []
    settings = make_settings(compute={"asr": "colab"})
    asr.transcribe(tmp_path / "a.wav", settings=settings, on_event=events.append)
    assert local and any("не задан compute.colab_url" in e for e in events)


def test_choose_device_treats_colab_as_the_local_fallback(
    make_settings: Callable[..., Settings],
) -> None:
    with_gpu = asr.choose_device(make_settings(compute={"asr": "colab"}), probe())
    assert (with_gpu.device, with_gpu.model) == ("cuda", "large-v3")
    no_gpu = asr.choose_device(make_settings(compute={"asr": "colab"}), probe(cuda_devices=0))
    assert (no_gpu.device, no_gpu.model) == ("cpu", "small")
    with pytest.raises(ExtractError, match="api"):
        asr.choose_device(make_settings(compute={"asr": "api"}), probe())


def test_require_asr_needs_faster_whisper_only_when_there_is_no_worker(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asr, "package_version", lambda name: None)
    colab = make_settings(compute={"asr": "colab", "colab_url": "https://a.trycloudflare.com"})
    asr.require_asr(colab)  # the worker recognizes; numpy is there
    auto = make_settings(compute={"colab_url": "https://a.trycloudflare.com"})
    asr.require_asr(auto)
    with pytest.raises(ExtractError, match="uv sync --extra video"):
        asr.require_asr(make_settings())  # no worker: the old rule
    with pytest.raises(ExtractError, match="uv sync --extra video"):
        asr.require_asr(make_settings(compute={"asr": "local-gpu", "colab_url": "https://a.x"}))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ExtractError, match="numpy"):
        asr.require_asr(colab)


def test_the_plan_of_a_recording_names_the_worker(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe(cuda_devices=0))
    settings = settings_for(make_settings, "https://a.trycloudflare.com")
    rec = SourceRecord(
        id="V1", kind="video", title="Лекция", added="2026-10-04T00:00:00Z", units={"minutes": 60}
    )
    ctx = ExtractContext(topic_dir=tmp_path, source=rec, settings=settings, out_dir=tmp_path / "x")
    plan = vd.VideoExtractor().plan(ctx)
    text = "; ".join(plan.notes)
    assert "Распознавание речи (large-v3 в Colab): ≈ 6 мин (оценка)" in text


# ---------------------------------------------------------------- the address and the answers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://a-b.trycloudflare.com", "https://a-b.trycloudflare.com"),
        ("  https://a-b.trycloudflare.com/  ", "https://a-b.trycloudflare.com"),
        ("a-b.trycloudflare.com", "https://a-b.trycloudflare.com"),
        ("http://127.0.0.1:8790", "http://127.0.0.1:8790"),
        ("http://localhost:8790/", "http://localhost:8790"),
        ("http://[::1]:8790", "http://[::1]:8790"),
        ("https://host.example/prefix/", "https://host.example/prefix"),
    ],
)
def test_worker_url_is_normalized(
    make_settings: Callable[..., Settings], raw: str, expected: str
) -> None:
    assert asr.worker_url(make_settings(compute={"colab_url": raw})) == expected


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ("", "не задан compute.colab_url"),
        ("http://a.trycloudflare.com", "https://"),
        ("http://192.168.1.5:8790", "https://"),
        ("ftp://a.example", "https://имя"),
        ("https://user:pass@a.example", "логина"),
        ("https://a.example/?x=1", "параметров"),
        ("https://a.example:99999", "адрес"),
    ],
)
def test_worker_url_rejects_what_would_leak_the_token(
    make_settings: Callable[..., Settings], raw: str, text: str
) -> None:
    with pytest.raises(asr.RemoteError, match=text):
        asr.worker_url(make_settings(compute={"colab_url": raw}))


def test_health_answers_that_are_not_a_h0lon_worker(
    scripted: Scripted, make_settings: Callable[..., Settings]
) -> None:
    settings = settings_for(make_settings, scripted.url)
    cases = [
        (Reply(200, {"app": "something-else"}), "не H0lon-worker"),
        (Reply(200, {**HEALTH_OK, "protocol": 99}), r"версия протокола worker \(99\)"),
        (Reply(200, {**HEALTH_OK, "authorized": False}), "не принял токен"),
        (Reply(200, {**HEALTH_OK, "ready": False}), "не установлен faster-whisper"),
        (Reply(200, b"<html>Cloudflare</html>"), "не похож на ответ H0lon-worker"),
        (Reply(200, b"[1, 2]"), "не похож на ответ H0lon-worker"),
        (Reply(401, {"detail": "x"}), "не принял токен"),
        (Reply(404, {"detail": "x"}), "нет H0lon-worker"),
        (Reply(530, b"error code: 1033"), "туннель или worker не отвечают"),
        (Reply(500, {"detail": "внутренняя беда"}), "HTTP 500: внутренняя беда"),
    ]
    for reply, text in cases:
        scripted.on("GET", "/health", reply)
        with pytest.raises(asr.RemoteError, match=text):
            asr.worker_health(settings)
    scripted.on("GET", "/health", Reply(200, HEALTH_OK))
    assert asr.worker_health(settings)["gpu"] == "Tesla T4"
    assert {r["auth"] for r in scripted.requests} == {f"Bearer {TOKEN}"}


def test_nobody_home_is_a_readable_error(make_settings: Callable[..., Settings]) -> None:
    settings = settings_for(make_settings, f"http://127.0.0.1:{free_port()}")
    with pytest.raises(asr.RemoteError, match="нет связи с worker") as caught:
        asr.worker_health(settings, timeout=2.0)
    assert not caught.value.fatal and TOKEN not in str(caught.value)


def test_a_redirect_is_not_followed_so_the_token_stays_home(
    scripted: Scripted, make_settings: Callable[..., Settings]
) -> None:
    elsewhere = Scripted()
    try:
        elsewhere.on("GET", "/health", Reply(200, HEALTH_OK))
        scripted.on("GET", "/health", Reply(302, {}, {"Location": elsewhere.url + "/health"}))
        with pytest.raises(asr.RemoteError, match="HTTP 302"):
            asr.worker_health(settings_for(make_settings, scripted.url))
        assert elsewhere.requests == []
    finally:
        elsewhere.close()


def test_the_token_is_only_in_the_header(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    happy(scripted)
    remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))
    assert scripted.requests and all(r["auth"] == f"Bearer {TOKEN}" for r in scripted.requests)
    assert all(TOKEN not in r["path"] for r in scripted.requests)
    (post,) = scripted.calls("POST", "/asr")
    assert TOKEN.encode() not in post["body"]


# ---------------------------------------------------------------- the request and the polling


def field_of(body: bytes, name: str) -> str:
    match = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n(.*?)\r\n--h0lon', body, re.DOTALL)
    assert match, f"no field {name}"
    return match.group(1).decode("utf-8")


def test_the_multipart_request_carries_the_parameters_and_the_audio(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    happy(scripted)
    wav = write_wav(tmp_path / "a.wav", 2.0)
    remote(settings_for(make_settings, scripted.url, asr_model="medium"), wav)
    (post,) = scripted.calls("POST", "/asr")
    assert post["type"].startswith("multipart/form-data; boundary=h0lon")
    body = post["body"]
    assert field_of(body, "model") == "medium" and field_of(body, "language") == "ru"
    assert field_of(body, "initial_prompt") == asr.build_prompt(
        "Метрические методы", ["метод Парзена"]
    )
    assert re.search(rb'name="audio"; filename="audio\.(m4a|wav)"', body)
    assert body.rstrip().endswith(b"--")
    assert scripted.calls("DELETE", "/jobs/j1")  # cleaned up


def test_polling_survives_short_failures(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    running = {"id": "j1", "status": "running", "progress": {"done": 1.0, "total": 3.0}}
    queued = {"id": "j1", "status": "queued", "progress": {}}
    happy(
        scripted,
        Reply(503, b"bad gateway"),
        Reply(200, queued),
        Reply(530, b"tunnel"),
        Reply(200, running),
        Reply(200, done_job()),
    )
    events: list[str] = []
    transcript = remote(
        settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0), events
    )
    assert transcript.text == "Привет"
    assert len(scripted.calls("GET", "/jobs/j1")) == 5
    assert any("в очереди" in e for e in events)


def test_polling_gives_up_when_the_worker_stays_away(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(asr, "POLL_FAILURES_S", 0.05)
    happy(scripted, Reply(502, b"gone"))
    with pytest.raises(asr.RemoteError, match="связь с worker потеряна"):
        remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))
    assert scripted.calls("DELETE", "/jobs/j1")  # it tries to free the GPU


def test_a_lost_job_is_fatal_at_once(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    happy(scripted, Reply(404, {"detail": "задача не найдена"}))
    with pytest.raises(asr.RemoteError, match="перезапущен") as caught:
        remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))
    assert caught.value.fatal and len(scripted.calls("GET", "/jobs/j1")) == 1


@pytest.mark.parametrize(
    ("job", "text"),
    [
        ({"status": "error", "error": "CUDA out of memory"}, "не смог распознать запись: CUDA out"),
        ({"status": "cancelled"}, "остановлена на worker"),
        (
            {"status": "done", "result": {"format": 99, "segments": []}},
            "неизвестного формата",
        ),
        (
            {"status": "done", "result": {"format": asr.TRANSCRIPT_FORMAT, "segments": [{}]}},
            "повреждённый транскрипт",
        ),
        ({"status": "done"}, "неизвестного формата"),
    ],
)
def test_bad_job_results(
    scripted: Scripted,
    make_settings: Callable[..., Settings],
    tmp_path: Path,
    job: dict[str, Any],
    text: str,
) -> None:
    happy(scripted, Reply(200, {"id": "j1", **job}))
    with pytest.raises(asr.RemoteError, match=text):
        remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))
    assert scripted.calls("DELETE", "/jobs/j1")


def test_a_refused_upload_is_an_error_without_polling(
    scripted: Scripted, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    happy(scripted)
    scripted.on("POST", "/asr", Reply(413, {"detail": "аудио больше 512 МБ"}))
    with pytest.raises(asr.RemoteError, match="слишком велика"):
        remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))
    assert scripted.calls("GET", "/jobs/j1") == []
    scripted.on("POST", "/asr", Reply(202, {"status": "queued"}))
    with pytest.raises(asr.RemoteError, match="не вернул номер задачи"):
        remote(settings_for(make_settings, scripted.url), write_wav(tmp_path / "a.wav", 1.0))


# ---------------------------------------------------------------- pieces


def test_upload_plan_squeezes_long_recordings_and_refuses_the_longest() -> None:
    kbps, size = asr.upload_plan(90 * 60)
    assert kbps == 48 and 30e6 < size < 34e6  # ~32 MB for a 90-minute lecture
    kbps, size = asr.upload_plan(6 * 3600)
    assert kbps < 48 and size <= asr.UPLOAD_MAX_BYTES
    assert asr.upload_plan(8 * 3600)[0] == 25  # 90 MB over 8 hours
    with pytest.raises(asr.RemoteError, match="не помещается"):
        asr.upload_plan(9 * 3600)


def test_the_upload_body_streams_head_file_and_tail(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 40)
    seen: list[tuple[int, int]] = []
    body = asr._UploadBody(b"HEAD--", path, b"--TAIL", lambda a, b: seen.append((a, b)))
    try:
        assert body.length == 6 + 10240 + 6
        chunks = []
        while True:
            chunk = body.read(1000)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        body.close()
    data = b"".join(chunks)
    assert data == b"HEAD--" + path.read_bytes() + b"--TAIL" and len(chunks) == 11
    assert seen[-1] == (body.length, body.length) and seen[0][0] == 1000


def test_a_file_that_shrinks_during_the_upload_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 100)
    body = asr._UploadBody(b"", path, b"")
    try:
        # as if the file had been longer when the request began
        body._file_size = body.length = 200
        with pytest.raises(OSError, match="изменился"):
            body.read(-1)
    finally:
        body.close()


def test_calibration_of_the_worker(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()
    assert asr.estimate_seconds_per_minute(settings, "colab", "large-v3") == (6.0, False)
    assert asr.estimate_seconds_per_minute(settings, "colab", "unknown") == (4.0, False)
    assert asr.estimate_seconds_per_minute(settings, "cuda", "large-v3") == (8.0, False)
    asr.record_calibration(
        settings, device="colab", model="large-v3", audio_seconds=600, wall_seconds=70
    )
    assert asr.estimate_seconds_per_minute(settings, "colab", "large-v3") == (7.0, True)
    assert asr.estimate_seconds_per_minute(settings, "cuda", "large-v3")[1] is False  # separate
    assert asr.calibration_entry(settings, "colab", "large-v3")["runs"] == 1  # type: ignore[index]
    # the upload: bytes per second, longer uploads weigh more; tiny ones say nothing
    assert asr.upload_bytes_per_second(settings) == (asr.DEFAULT_UPLOAD_BYTES_PER_S, False)
    assert asr.record_upload(settings, size=100_000, seconds=1.0) is None
    first = asr.record_upload(settings, size=10_000_000, seconds=10.0)
    assert first is not None and first["bytes_per_second"] == 1_000_000
    second = asr.record_upload(settings, size=30_000_000, seconds=10.0)
    assert second is not None and second["bytes_per_second"] == 2_000_000 and second["runs"] == 2
    assert asr.upload_bytes_per_second(settings) == (2_000_000.0, True)
    data = json.loads(asr.calibration_path(settings).read_text("utf-8"))
    assert set(data) == {"asr", "upload", "format"} and "colab|large-v3" in data["asr"]


# ---------------------------------------------------------------- doctor


def check(make_settings: Callable[..., Settings], url: str, **compute: Any) -> doctor.Check:
    return doctor._check_colab(settings_for(make_settings, url, **compute))


def test_doctor_ok_with_a_gpu_worker(worker: Worker, make_settings) -> None:
    result = check(make_settings, worker.url)
    assert result.id == "colab" and result.status == "ok" and not result.required
    assert "на связи, токен принят" in result.detail and "Tesla T4" in result.detail
    assert f"протокол {asr.WORKER_PROTOCOL}" in result.detail and "127.0.0.1" in result.detail
    assert TOKEN not in json.dumps(result.to_dict())
    assert result.data["device"] == "cuda" and result.data["model"] == "large-v3"


def test_doctor_warns_about_a_worker_without_a_gpu(worker: Worker, make_settings) -> None:
    worker.probe.update(device="cpu", gpu=None)
    result = check(make_settings, worker.url)
    assert result.status == "warn" and "очень медленным" in result.detail
    assert result.hint is not None and "T4 GPU" in result.hint


def test_doctor_warns_when_the_worker_is_gone_and_says_what_follows(
    make_settings, worker: Worker
) -> None:
    url = f"http://127.0.0.1:{free_port()}"
    result = check(make_settings, url)
    assert result.status == "warn" and "нет связи" in result.detail
    assert "распознавание пойдёт на этом ПК" in result.detail
    assert result.hint is not None and "h0lon_worker.ipynb" in result.hint
    strict = check(make_settings, url, colab_fallback_local=False)
    assert "остановится с ошибкой" in strict.detail


def test_doctor_names_a_wrong_or_missing_token(worker: Worker, make_settings) -> None:
    wrong = check(make_settings, worker.url, colab_token="wrong-token-123")
    assert wrong.status == "warn" and "токен" in wrong.detail
    assert "wrong-token-123" not in json.dumps(wrong.to_dict())
    empty = check(make_settings, worker.url, colab_token="")
    assert empty.status == "warn" and "compute.colab_token" in empty.detail


def test_doctor_notes_that_a_local_mode_ignores_the_worker(worker: Worker, make_settings) -> None:
    result = check(make_settings, worker.url, asr="local-gpu")
    assert result.status == "ok" and "worker не используется" in result.detail


def test_the_check_exists_only_when_an_address_is_set(
    make_settings: Callable[..., Settings],
) -> None:
    plain = [job[0] for job in doctor._jobs(make_settings(), False)]
    assert "colab" not in plain and "asr" in plain
    url = "https://a.trycloudflare.com"
    with_url = [job[0] for job in doctor._jobs(make_settings(compute={"colab_url": url}), False)]
    assert with_url[-1] == "colab" and "colab" in doctor.ORDER


# ---------------------------------------------------------------- settings


def test_compute_settings_and_the_example_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tomllib

    from h0lon.config import load_settings

    compute = Settings().compute
    assert (compute.colab_url, compute.colab_token, compute.colab_fallback_local) == ("", "", True)
    monkeypatch.setenv("H0LON_COMPUTE__COLAB_TOKEN", "from-env")
    monkeypatch.setenv("H0LON_COMPUTE__COLAB_FALLBACK_LOCAL", "false")
    assert Settings().compute.colab_token == "from-env"
    assert Settings().compute.colab_fallback_local is False
    monkeypatch.delenv("H0LON_COMPUTE__COLAB_TOKEN")
    monkeypatch.delenv("H0LON_COMPUTE__COLAB_FALLBACK_LOCAL")

    root = Path(__file__).resolve().parent.parent
    example = (root / "h0lon.example.toml").read_bytes()
    assert example == (root / "h0lon" / "resources" / "h0lon.example.toml").read_bytes()
    section = tomllib.loads(example.decode("utf-8"))["compute"]
    assert section["colab_token"] == "" and section["colab_fallback_local"] is True
    assert section["colab_url"] == ""  # no address and no token in the repository
    loaded = load_settings(root / "h0lon.example.toml").compute
    assert (loaded.colab_token, loaded.colab_fallback_local) == ("", True)


def test_the_notebook_is_valid_and_keeps_no_secrets() -> None:
    root = Path(__file__).resolve().parent.parent
    notebook = json.loads((root / "colab" / "h0lon_worker.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["cells"]
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert all(not c["outputs"] for c in notebook["cells"] if c["cell_type"] == "code")
    assert "h0lon[video,video-gpu] @ {H0LON_SOURCE}" in code
    assert "git+https://github.com/Filodot/H0lon" in code
    assert "cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" in code
    assert "secrets.token_urlsafe" in code and "H0LON_WORKER_TOKEN" in code
    assert "h0lon.worker" in code and "trycloudflare" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":  # every code cell is Python (apart from the pip magic)
            source = "\n".join(
                line for line in "".join(cell["source"]).splitlines() if not line.startswith("!")
            )
            compile(source, cell["id"], "exec")
