"""The recognition worker (h0lon/worker): protocol, token, jobs, cancellation.

The application is driven through `TestClient`; the recognition is a fake (`transcribe_fn`) and
the device probe too: no model, no GPU, no network. `default_transcribe` is checked with a fake
`asr.transcribe_local` (and with the real ffmpeg for the audio that is not WAV).
"""

from __future__ import annotations

import io
import threading
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from h0lon import tools
from h0lon.config import Settings
from h0lon.extract import asr
from h0lon.extract.registry import ExtractError
from h0lon.worker import __main__ as worker_main
from h0lon.worker import server as ws

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

TOKEN = "tok-" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FFMPEG = tools.find_simple("ffmpeg")


def wav_bytes(seconds: float = 1.0, *, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(seconds * rate) * channels)
    return buf.getvalue()


class FakeRecognizer:
    """Stands in for `default_transcribe`; `hold` keeps a job running until `release`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.hold = False
        self.fail: Exception | None = None

    def __call__(
        self,
        audio: Path,
        *,
        language: str | None,
        initial_prompt: str,
        model: str,
        progress: Callable[[float, float], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> asr.Transcript:
        self.calls.append(
            {
                "audio": audio,
                "suffix": audio.suffix,
                "exists": audio.is_file(),
                "size": audio.stat().st_size,
                "language": language,
                "prompt": initial_prompt,
                "model": model,
            }
        )
        self.started.set()
        if progress:
            progress(30.0, 60.0)
        if on_event:
            on_event("модель загружена")
        while self.hold and not self.release.is_set():
            if cancel and cancel():
                raise asr.AsrCancelled("остановлено")
            time.sleep(0.005)
        if self.fail is not None:
            raise self.fail
        word = asr.Word(0.0, 2.0, "Привет", 0.9)
        return asr.Transcript(
            segments=[asr.Segment(0.0, 2.0, "Привет", [word])],
            language=language or "ru",
            duration=60.0,
            model=model,
            device="cuda",
            compute_type="float16",
            seconds=1.5,
            prompt=initial_prompt,
        )


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def make_client(tmp_path: Path, recognizer: FakeRecognizer) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**kwargs: Any) -> TestClient:
        options: dict[str, Any] = {
            "token": TOKEN,
            "work_dir": tmp_path / "work",
            "transcribe_fn": recognizer,
            "probe_fn": lambda: {"device": "cuda", "gpu": "Tesla T4", "ready": True},
        }
        client = TestClient(ws.create_app(**{**options, **kwargs}))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def submit(client: TestClient, data: bytes | None = None, **fields: str) -> Any:
    form = {"language": "ru", "initial_prompt": "Лекция: Метрические методы.", **fields}
    return client.post(
        "/asr",
        headers=AUTH,
        files={"audio": ("audio.wav", data if data is not None else wav_bytes(), "audio/wav")},
        data=form,
    )


def wait_for(client: TestClient, job_id: str, states: set[str], timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        view = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if view["status"] in states:
            return view
        assert time.monotonic() < deadline, f"job stays {view['status']}"
        time.sleep(0.01)


# ---------------------------------------------------------------- health and token


def test_health_is_open_and_describes_the_worker(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ok"] is True and body["app"] == "h0lon-worker"
    assert body["protocol"] == asr.WORKER_PROTOCOL == ws.PROTOCOL
    assert body["model"] == "large-v3" and body["device"] == "cuda" and body["gpu"] == "Tesla T4"
    assert body["ready"] is True and body["busy"] is False
    assert body["auth_configured"] is True and body["authorized"] is False  # no token sent
    assert client.get("/health", headers=AUTH).json()["authorized"] is True
    wrong = client.get("/health", headers={"Authorization": "Bearer nope"}).json()
    assert wrong["authorized"] is False
    assert TOKEN not in client.get("/health", headers=AUTH).text  # the token is never echoed


def test_everything_but_health_needs_the_token(client: TestClient) -> None:
    for method, url in (("get", "/jobs/abc"), ("delete", "/jobs/abc")):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic " + TOKEN}):
            response = getattr(client, method)(url, headers=headers)
            assert response.status_code == 401, (method, headers)
            assert response.headers["www-authenticate"] == "Bearer"
    refused = client.post("/asr", files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert refused.status_code == 401
    assert client.get("/jobs/abc", headers=AUTH).status_code == 404  # the token is right


def test_a_worker_without_a_token_refuses_everything_but_health(
    make_client: Callable[..., TestClient], recognizer: FakeRecognizer
) -> None:
    client = make_client(token="")
    assert client.get("/health").json()["auth_configured"] is False
    for headers in ({}, AUTH, {"Authorization": "Bearer "}):
        response = client.get("/jobs/abc", headers=headers)
        assert response.status_code == 401
        assert "H0LON_WORKER_TOKEN" in response.json()["detail"]
    assert submit(client).status_code == 401
    assert recognizer.calls == []


def test_the_token_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recognizer: FakeRecognizer
) -> None:
    monkeypatch.setenv("H0LON_WORKER_TOKEN", "  from-env-token  ")
    app = ws.create_app(work_dir=tmp_path, transcribe_fn=recognizer)
    with TestClient(app) as client:
        ok = client.get("/jobs/x", headers={"Authorization": "Bearer from-env-token"})
        assert ok.status_code == 404
        assert client.get("/jobs/x", headers=AUTH).status_code == 401


def test_the_body_is_not_read_before_the_token_is_checked(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    big = b"0" * 3_000_000
    response = client.post(
        "/asr", files={"audio": ("a.wav", big, "audio/wav")}, headers={"Authorization": "Bearer x"}
    )
    assert response.status_code == 401 and recognizer.calls == []


# ---------------------------------------------------------------- a job from start to end


def test_a_job_runs_and_returns_the_transcript(
    client: TestClient, recognizer: FakeRecognizer, tmp_path: Path
) -> None:
    response = submit(client, model="large-v3-turbo")
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    done = wait_for(client, job_id, {"done"})
    (call,) = recognizer.calls
    assert call["language"] == "ru" and call["model"] == "large-v3-turbo"
    assert call["prompt"] == "Лекция: Метрические методы." and call["exists"] and call["size"] > 100
    result = done["result"]
    assert result["format"] == asr.TRANSCRIPT_FORMAT
    transcript = asr.Transcript.from_dict(result)  # the format of extract/asr.py
    assert transcript.text == "Привет" and transcript.segments[0].words[0].prob == 0.9
    assert transcript.model == "large-v3-turbo" and transcript.device == "cuda"
    assert transcript.seconds == 1.5
    assert done["progress"] == {"done": 60.0, "total": 60.0, "fraction": 1.0}
    assert done["seconds"] is not None and done["started"] and done["finished"]
    assert any("принято аудио" in e for e in done["events"])
    # the audio is deleted when the job ends, and the job can be deleted by the client
    assert not call["audio"].exists() and not call["audio"].parent.exists()
    assert client.delete(f"/jobs/{job_id}", headers=AUTH).json() == {"ok": True}
    assert client.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
    assert client.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 404


def test_defaults_of_the_fields(client: TestClient, recognizer: FakeRecognizer) -> None:
    response = client.post(
        "/asr", headers=AUTH, files={"audio": ("lecture.m4a", b"x" * 500, "audio/mp4")}
    )
    assert response.status_code == 202
    wait_for(client, response.json()["job_id"], {"done"})
    (call,) = recognizer.calls
    assert call["language"] is None and call["model"] == "large-v3" and call["prompt"] == ""
    assert call["suffix"] == ".m4a"  # a known extension is kept for ffmpeg


def test_progress_and_messages_are_visible_while_the_job_runs(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    recognizer.hold = True
    job_id = submit(client).json()["job_id"]
    assert recognizer.started.wait(5)
    view = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert view["status"] == "running" and "result" not in view
    assert view["progress"] == {"done": 30.0, "total": 60.0, "fraction": 0.5}
    assert view["message"] == "модель загружена"
    health = client.get("/health").json()
    assert health["busy"] is True and health["running"] == 1
    recognizer.release.set()
    assert wait_for(client, job_id, {"done"})["result"]["segments"]
    assert client.get("/health").json()["busy"] is False


def test_jobs_run_one_at_a_time_in_order(client: TestClient, recognizer: FakeRecognizer) -> None:
    recognizer.hold = True
    first = submit(client, initial_prompt="первая").json()["job_id"]
    assert recognizer.started.wait(5)
    second = submit(client, initial_prompt="вторая").json()["job_id"]
    assert client.get(f"/jobs/{second}", headers=AUTH).json()["status"] == "queued"
    assert client.get("/health").json()["queued"] == 1
    assert len(recognizer.calls) == 1  # one GPU: the second waits
    recognizer.release.set()
    wait_for(client, first, {"done"})
    wait_for(client, second, {"done"})
    assert [c["prompt"] for c in recognizer.calls] == ["первая", "вторая"]


def test_a_failing_job_is_an_error_and_the_worker_goes_on(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    recognizer.fail = ExtractError("Ошибка распознавания речи: битое аудио")
    failed = wait_for(client, submit(client).json()["job_id"], {"error"})
    assert failed["error"] == "Ошибка распознавания речи: битое аудио" and "result" not in failed
    recognizer.fail = RuntimeError("boom")
    crashed = wait_for(client, submit(client).json()["job_id"], {"error"})
    assert crashed["error"] == "RuntimeError: boom"
    recognizer.fail = None
    assert wait_for(client, submit(client).json()["job_id"], {"done"})["result"]


def test_deleting_a_running_job_stops_it_and_removes_the_audio(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    recognizer.hold = True
    job_id = submit(client).json()["job_id"]
    assert recognizer.started.wait(5)
    audio = recognizer.calls[0]["audio"]
    assert audio.exists()
    assert client.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 200
    assert client.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
    deadline = time.monotonic() + 5
    while client.get("/health").json()["busy"] or audio.parent.exists():
        assert time.monotonic() < deadline, "the stopped job still holds the worker"
        time.sleep(0.01)
    recognizer.hold = False  # the worker is free again
    assert wait_for(client, submit(client).json()["job_id"], {"done"})["result"]


def test_deleting_a_queued_job_keeps_it_from_running(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    recognizer.hold = True
    submit(client, initial_prompt="первая")
    assert recognizer.started.wait(5)
    queued = submit(client, initial_prompt="вторая").json()["job_id"]
    assert client.delete(f"/jobs/{queued}", headers=AUTH).status_code == 200
    recognizer.release.set()
    deadline = time.monotonic() + 5
    while client.get("/health").json()["busy"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert [c["prompt"] for c in recognizer.calls] == ["первая"]


def test_finished_jobs_are_forgotten_after_a_while(
    make_client: Callable[..., TestClient], recognizer: FakeRecognizer
) -> None:
    client = make_client(keep_finished_s=0.05)
    job_id = submit(client).json()["job_id"]
    wait_for(client, job_id, {"done"})
    time.sleep(0.1)
    assert client.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404


# ---------------------------------------------------------------- bad requests


@pytest.mark.parametrize(
    ("fields", "text"),
    [
        ({"model": "../../etc/passwd"}, "model"),
        ({"model": "/abs/path"}, "model"),
        ({"model": "C:\\models\\x"}, "model"),
        ({"model": "a b"}, "model"),
        ({"language": "russian!"}, "language"),
        ({"initial_prompt": "я" * 4001}, "initial_prompt"),
    ],
)
def test_bad_fields_are_refused(
    client: TestClient, recognizer: FakeRecognizer, fields: dict[str, str], text: str
) -> None:
    response = submit(client, **fields)
    assert response.status_code == 422 and text in response.json()["detail"]
    assert recognizer.calls == []


def test_a_repo_style_model_name_is_accepted(client: TestClient) -> None:
    ok = submit(client, model="Systran/faster-whisper-large-v3")
    assert ok.status_code == 202


def test_missing_and_empty_audio(client: TestClient) -> None:
    none = client.post("/asr", headers=AUTH, data={"language": "ru"})
    assert none.status_code in (400, 422)
    empty = submit(client, data=b"")
    assert empty.status_code == 422 and "пусто" in empty.json()["detail"]


def test_too_big_audio_is_refused_and_leaves_nothing(
    make_client: Callable[..., TestClient], recognizer: FakeRecognizer, tmp_path: Path
) -> None:
    client = make_client(max_upload_mb=1)
    response = submit(client, data=b"0" * 1_200_000)
    assert response.status_code == 413 and "1 МБ" in response.json()["detail"]
    assert recognizer.calls == []
    assert list((tmp_path / "work").iterdir()) == []  # the folder of the job is gone


def test_a_worker_without_faster_whisper_says_so() -> None:
    app = ws.create_app(
        token=TOKEN, probe_fn=lambda: {"device": "cpu", "ready": False, "problem": "нет пакета"}
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["problem"] == "нет пакета"
        refused = submit(client)
        assert refused.status_code == 503 and "faster-whisper" in refused.json()["detail"]


def test_a_broken_probe_does_not_break_health(make_client: Callable[..., TestClient]) -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("DLL load failed")

    client = make_client(probe_fn=broken)
    body = client.get("/health").json()
    assert body["ok"] is True and body["device"] == "cpu" and body["ready"] is False
    assert "DLL load failed" in body["problem"]


# ---------------------------------------------------------------- recognition of a job


def test_default_transcribe_uses_the_same_code_and_never_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_local(wav: Path, **kwargs: Any) -> asr.Transcript:
        seen.update(kwargs, wav=wav)
        return asr.Transcript(segments=[], model=kwargs["settings"].compute.asr_model)

    monkeypatch.setattr(asr, "transcribe_local", fake_local)
    monkeypatch.setenv("H0LON_COMPUTE__ASR", "colab")  # must not make the worker call itself
    monkeypatch.setenv("H0LON_COMPUTE__COLAB_URL", "https://example.trycloudflare.com")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(wav_bytes(1.0))
    events: list[str] = []
    out = ws.default_transcribe(
        audio,
        language="ru",
        initial_prompt="Лекция.",
        model="medium",
        progress=lambda *a: None,
        cancel=lambda: False,
        on_event=events.append,
    )
    assert out.model == "medium"
    settings: Settings = seen["settings"]
    assert settings.compute.asr == "auto" and settings.compute.asr_model == "medium"
    assert asr.explicit_model(settings)  # the model of the request is not replaced on the CPU
    assert seen["wav"] == audio  # 16 kHz mono WAV is read as it is
    assert seen["initial_prompt"] == "Лекция." and seen["language"] == "ru"
    assert seen["label"] == "worker" and seen["on_event"] == events.append


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не найден")
def test_audio_that_is_not_a_16k_wav_is_decoded_by_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "stereo.wav"
    src.write_bytes(wav_bytes(1.0, rate=44100, channels=2))
    assert ws._is_whisper_wav(src) is False
    out = ws.decode_to_wav(src, tmp_path / "out.wav")
    assert ws._is_whisper_wav(out) and 0.9 < asr.wav_duration(out) < 1.1
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not audio at all" * 20)
    with pytest.raises(ExtractError, match="ffmpeg не смог прочитать"):
        ws.decode_to_wav(bad, tmp_path / "bad.wav")
    monkeypatch.setattr(tools, "find_simple", lambda name: None)
    with pytest.raises(ExtractError, match="нет ffmpeg"):
        ws.decode_to_wav(src, tmp_path / "x.wav")


# ---------------------------------------------------------------- python -m h0lon.worker


def test_main_starts_uvicorn_and_warns_about_a_missing_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    started: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.update(app=app, **kw))
    monkeypatch.delenv("H0LON_WORKER_TOKEN", raising=False)
    work = ["--work-dir", str(tmp_path / "work")]
    assert worker_main.main(["--port", "0", *work]) == 2  # port 0 is out of range for the worker
    assert worker_main.main(["--port", "18790", "--model", "small", *work]) == 0
    assert started["host"] == "127.0.0.1" and started["port"] == 18790
    captured = capsys.readouterr()
    assert "H0LON_WORKER_TOKEN не задана" in captured.err and "НЕ задан" in captured.out
    assert started["app"].state.jobs is not None


def test_main_never_prints_the_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: None)
    monkeypatch.setenv("H0LON_WORKER_TOKEN", TOKEN)
    work = ["--work-dir", str(tmp_path / "work")]
    assert worker_main.main(["--port", "18791", *work]) == 0
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err and "токен задан" in captured.out
    monkeypatch.setenv("H0LON_WORKER_TOKEN", "short")
    assert worker_main.main(["--port", "18791", "--host", "0.0.0.0", *work]) == 0
    err = capsys.readouterr().err
    assert "короче 16 знаков" in err and "не только этот ПК" in err and "short" not in err
