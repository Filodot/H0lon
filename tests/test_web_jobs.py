"""Background jobs of the web interface: scheduling, events, outcomes, cancellation.

The library functions (`extract_topic`, `build_topic`, `approve_topic`) are replaced by fakes:
no agent, no network.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from h0lon.agents import cooling_backends, mark_cooling
from h0lon.config import Settings
from h0lon.extract.model import ExtractResult
from h0lon.synth.model import BuildResult, StageResult
from h0lon.web.jobs import MAX_EVENTS, Job, JobManager, TopicBusyError
from h0lon.workspace import create_topic


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


@pytest.fixture
def topic(settings: Settings) -> Path:
    return create_topic(settings, title="Тема", course="Курс")


def make_manager(settings: Settings, **kwargs: Any) -> JobManager:
    return JobManager(settings, **kwargs)


def fake_build(emit_stages: tuple[str, ...] = (), ok: bool = True, stopped_at: str | None = None):
    calls: list[dict[str, Any]] = []

    def build_topic(settings, topic_dir, **kwargs):
        calls.append(kwargs)
        on_event = kwargs["on_event"]
        for title in emit_stages:
            on_event(f"{title}…")
        result = BuildResult(
            ok=ok,
            topic_dir=Path(topic_dir),
            stopped_at=stopped_at,
            stages=[StageResult(stage="extract", ok=True)],
            message="Готово: master.pdf." if ok else "Стадия остановлена",
        )
        return result

    build_topic.calls = calls  # type: ignore[attr-defined]
    return build_topic


def test_build_job_collects_events_and_result(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_build(("Структура темы (S1)", "Разделы (S2)"))
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake)
    jobs = make_manager(settings)

    job = jobs.start("kurs/tema", topic, "build", review=False, from_stage="outline", force=True)
    jobs.wait(job.id)

    assert job.status == "done"
    assert job.result["ok"] is True
    assert job.result["message"] == "Готово: master.pdf."
    assert [s["stage"] for s in job.result["stages"]] == ["extract"]
    assert "Структура темы (S1)…" in job.events
    assert job.events[0].startswith("Задача «Сборка мастер-конспекта» запущена")
    assert job.events[-1] == "Готово: master.pdf."
    assert job.started is not None and job.finished is not None and job.finished >= job.started
    kwargs = fake.calls[0]
    assert kwargs["review"] is False
    assert kwargs["from_stage"] == "outline"
    assert kwargs["force"] is True
    assert kwargs["backend"] is None
    data = job.to_dict()
    assert data["status"] == "done"
    assert data["status_label"] == "готово"
    assert data["events_total"] == len(job.events)
    assert data["created"].endswith("Z")


def test_build_stopped_at_review_gate_is_stopped_with_reason(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build(ok=False, stopped_at="review"))
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    jobs.wait(job.id)
    assert job.status == "stopped"
    assert job.result["reason"] == "review"
    assert job.error is None


def test_failed_build_result_is_failed(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build(ok=False, stopped_at="outline"))
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    jobs.wait(job.id)
    assert job.status == "failed"
    assert job.error == "Стадия остановлена"


def test_exception_becomes_failed_with_readable_text(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("что-то сломалось")

    monkeypatch.setattr("h0lon.synth.build.build_topic", boom)
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    jobs.wait(job.id)
    assert job.status == "failed"
    assert "Внутренняя ошибка: RuntimeError: что-то сломалось" in job.error
    assert "Traceback" not in job.error
    assert job.events[-1] == job.error
    assert job.result["ok"] is False


def test_library_value_error_is_shown_as_is(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any):
        raise ValueError("Неизвестная стадия «x»")

    monkeypatch.setattr("h0lon.synth.build.build_topic", refuse)
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    jobs.wait(job.id)
    assert job.status == "failed"
    assert job.error == "Неизвестная стадия «x»"


def test_one_active_job_per_topic(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    jobs = make_manager(settings)
    first = jobs.start("kurs/tema", topic, "build")
    with pytest.raises(TopicBusyError) as busy:
        jobs.start("kurs/tema", topic, "approve")
    assert busy.value.job is first
    assert "уже выполняется" in str(busy.value)
    assert jobs.active_for(topic) is first
    release.set()
    jobs.wait(first.id)
    assert jobs.active_for(topic) is None
    assert jobs.latest_for(topic) is first
    again = jobs.start("kurs/tema", topic, "build")  # free again
    jobs.wait(again.id)
    assert again.status == "done"


def test_global_limit_queues_the_rest(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limited = Settings(
        general={
            "workspaces": tmp_path / "ws",
            "state_dir": tmp_path / "state",
            "git_per_topic": False,
        },
        agents={"parallel_runs": 1},
    )
    t1 = create_topic(limited, title="Один", course="Курс")
    t2 = create_topic(limited, title="Два", course="Курс")
    started: list[str] = []
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        started.append(Path(topic_dir).name)
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    jobs = make_manager(limited)
    j1 = jobs.start("kurs/odin", t1, "build")
    deadline = time.monotonic() + 5
    while j1.status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    j2 = jobs.start("kurs/dva", t2, "build")
    time.sleep(0.3)
    assert j1.status == "running"
    assert j2.status == "queued"
    assert started == ["odin"]
    release.set()
    jobs.wait(j1.id)
    jobs.wait(j2.id)
    assert j2.status == "done"
    assert started == ["odin", "dva"]


def test_parallel_runs_two_run_together(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    topics = [create_topic(settings, title=f"Т{i}", course="Курс") for i in range(2)]
    barrier = threading.Barrier(2, timeout=10)

    def meet(settings, topic_dir, **kwargs):
        barrier.wait()  # both jobs must be running at once, or this raises BrokenBarrierError
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", meet)
    jobs = make_manager(settings)  # default parallel_runs = 2
    started = [jobs.start(f"kurs/{i}", t, "build") for i, t in enumerate(topics)]
    for job in started:
        jobs.wait(job.id)
        assert job.status == "done", job.error


def test_events_are_capped_with_absolute_indexes() -> None:
    job = Job(id="j", topic="a/b", topic_dir=Path("."), kind="build", params={}, max_events=50)
    for i in range(120):
        job.add_event(f"строка {i}")
    assert len(job.events) == 50
    assert job.dropped == 70
    assert job.total_events == 120
    lines, first, nxt = job.read_events(0)
    assert (first, nxt, lines[0], lines[-1]) == (70, 120, "строка 70", "строка 119")
    lines, first, nxt = job.read_events(100)
    assert (first, nxt, lines[0]) == (100, 120, "строка 100")
    assert job.read_events(120) == ([], 120, 120)
    assert job.to_dict(tail=5)["events"] == [f"строка {i}" for i in range(115, 120)]
    assert job.to_dict(tail=5)["events_from"] == 115


def test_default_cap_is_5000_and_long_lines_are_cut() -> None:
    job = Job(id="j", topic="a/b", topic_dir=Path("."), kind="build", params={})
    assert MAX_EVENTS == 5000
    job.add_event("x" * 5000)
    assert len(job.events[0]) <= 2000 and job.events[0].endswith("…")
    job.add_event("первая\nвторая\n")
    assert job.events[-2:] == ["первая", "вторая"]


def test_add_event_is_thread_safe() -> None:
    job = Job(id="j", topic="a/b", topic_dir=Path("."), kind="build", params={}, max_events=100000)

    def writer(n: int) -> None:
        for i in range(500):
            job.add_event(f"{n}:{i}")

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert job.total_events == 4000
    assert len(set(job.events)) == 4000


def test_cancel_queued_job_is_dropped(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limited = Settings(
        general={
            "workspaces": tmp_path / "ws",
            "state_dir": tmp_path / "state",
            "git_per_topic": False,
        },
        agents={"parallel_runs": 1},
    )
    t1 = create_topic(limited, title="Один", course="Курс")
    t2 = create_topic(limited, title="Два", course="Курс")
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    jobs = make_manager(limited)
    j1 = jobs.start("kurs/odin", t1, "build")
    deadline = time.monotonic() + 5
    while j1.status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    j2 = jobs.start("kurs/dva", t2, "build")
    assert jobs.cancel(j2.id) is True
    assert j2.status == "stopped"
    assert j2.result["reason"] == "cancelled"
    assert jobs.active_for(t2) is None
    assert jobs.cancel(j2.id) is False  # already finished
    assert jobs.cancel("unknown") is False
    release.set()
    jobs.wait(j1.id)
    assert j1.status == "done"


def test_cancel_running_build_stops_before_next_stage(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_stage = threading.Event()
    go_on = threading.Event()
    reached: list[str] = []

    def staged(settings, topic_dir, **kwargs):
        on_event = kwargs["on_event"]
        on_event("Извлечение источников…")
        reached.append("extract")
        in_stage.set()
        go_on.wait(10)  # the stage is running; cancel arrives meanwhile
        on_event("Извлечение источников: готово")  # not a stage start: must not stop
        on_event("Структура темы (S1)…")  # the next stage start: the job stops here
        reached.append("outline")  # never reached
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", staged)
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    assert in_stage.wait(5)
    assert jobs.cancel(job.id) is True
    assert job.cancel_requested
    go_on.set()
    jobs.wait(job.id)
    assert job.status == "stopped"
    assert reached == ["extract"]
    assert job.result["reason"] == "cancelled"
    assert "Структура темы (S1)" in job.result["message"]
    assert any("Остановка запрошена" in line for line in job.events)


def _result(sid: str, ok: bool = True, **kwargs: Any) -> ExtractResult:
    return ExtractResult(ok=ok, source_id=sid, source_md=Path(f"{sid}/source.md"), **kwargs)


def _topic_with_sources(settings: Settings, ids: tuple[str, ...]) -> Path:
    from h0lon.workspace import load_topic, save_topic

    path = create_topic(settings, title="Источники", course="Курс")
    meta = load_topic(path)
    meta.sources = [
        {"id": sid, "kind": "md", "title": sid, "added": "2026-10-04T00:00:00Z"} for sid in ids
    ]
    save_topic(path, meta)
    return path


def test_extract_job_runs_per_source_and_reports(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _topic_with_sources(settings, ("D1", "D2"))
    seen: list[dict[str, Any]] = []

    def fake_extract(settings, topic_dir, **kwargs):
        seen.append(kwargs)
        (sid,) = kwargs["source_ids"]
        kwargs["on_event"](f"{sid}: читаю")
        if sid == "D2":
            return [ExtractResult(ok=False, source_id=sid, errors=["нет текстового слоя"])]
        return [_result(sid, blocks=7, agent_runs=2)]

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract)
    jobs = make_manager(settings)
    job = jobs.start(
        "kurs/istochniki", path, "extract", force=True, use_vision=False, backend="codex"
    )
    jobs.wait(job.id)

    assert job.status == "failed"
    assert [k["source_ids"] for k in seen] == [["D1"], ["D2"]]
    assert all(k["force"] is True and k["use_vision"] is False for k in seen)
    assert all(k["backend"] == "codex" and not k.get("dry_run") for k in seen)
    assert "D1: читаю" in job.events
    assert any(line.startswith("D1: готово — блоков 7, прогонов агента: 2") for line in job.events)
    assert "D2: ошибка — нет текстового слоя" in job.events
    assert job.error == "Не удалось извлечь: D2."
    assert [r["source_id"] for r in job.result["results"]] == ["D1", "D2"]


def test_extract_job_cancel_between_sources(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _topic_with_sources(settings, ("D1", "D2", "D3"))
    first_done = threading.Event()
    go_on = threading.Event()
    extracted: list[str] = []

    def fake_extract(settings, topic_dir, **kwargs):
        (sid,) = kwargs["source_ids"]
        extracted.append(sid)
        if sid == "D1":
            first_done.set()
            go_on.wait(10)
        return [_result(sid)]

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract)
    jobs = make_manager(settings)
    job = jobs.start("kurs/istochniki", path, "extract")
    assert first_done.wait(5)
    jobs.cancel(job.id)
    go_on.set()
    jobs.wait(job.id)
    assert job.status == "stopped"
    assert extracted == ["D1"]
    assert "Обработано источников: D1" in job.result["message"]


def test_extract_without_sources_fails_politely(settings: Settings, topic: Path) -> None:
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "extract")
    jobs.wait(job.id)
    assert job.status == "failed"
    assert "нет источников" in job.error


def test_approve_job_with_real_library(settings: Settings) -> None:
    from h0lon.workspace import load_topic, save_topic

    path = create_topic(settings, title="Одобрение", course="Курс")
    meta = load_topic(path)
    meta.sources = [
        {
            "id": "D1",
            "kind": "md",
            "title": "D1",
            "added": "2026-10-04T00:00:00Z",
            "status": "extracted",
            "extracted_key": "k1",
        }
    ]
    save_topic(path, meta)
    jobs = make_manager(settings)
    job = jobs.start("kurs/odobrenie", path, "approve")
    jobs.wait(job.id)
    assert job.status == "done", job.error
    assert job.result["sources"] == ["D1"]
    assert job.events[-1].startswith("Извлечение одобрено: D1")
    review = (load_topic(path).model_extra or {}).get("review")
    assert review["extracted_keys"] == {"D1": "k1"}


def test_approve_job_fails_when_not_extracted(settings: Settings) -> None:
    path = _topic_with_sources(settings, ("D1",))
    jobs = make_manager(settings)
    job = jobs.start("kurs/istochniki", path, "approve")
    jobs.wait(job.id)
    assert job.status == "failed"
    assert "Не все источники извлечены (D1)" in job.error


def test_cooling_is_reset_when_the_server_is_idle(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark_cooling("claude", "лимит")
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build())
    jobs = make_manager(settings)
    job = jobs.start("kurs/tema", topic, "build")
    jobs.wait(job.id)
    assert cooling_backends() == {}


def test_unknown_kind_and_wait_timeout(settings: Settings, topic: Path) -> None:
    jobs = make_manager(settings)
    with pytest.raises(ValueError, match="Неизвестный вид"):
        jobs.start("kurs/tema", topic, "nope")
    with pytest.raises(KeyError):
        jobs.wait("missing")


def test_finished_jobs_beyond_the_limit_are_forgotten(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("h0lon.web.jobs.MAX_JOBS", 3)
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build())
    jobs = make_manager(settings)
    ids = []
    for _ in range(6):
        job = jobs.start("kurs/tema", topic, "build")
        jobs.wait(job.id)
        ids.append(job.id)
    remembered = [j.id for j in jobs.recent()]
    assert len(remembered) <= 4
    assert remembered[0] == ids[-1]
    assert jobs.get(ids[0]) is None
