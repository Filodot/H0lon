"""Web interface: the «Очередь» page, the «В очередь» buttons and the `queue` job.

`build_topic` and `extract_topic` are replaced by fakes that may wait on events, so a job can be
observed while it runs. No agent, no network, no real waiting for a limit to reset.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from h0lon import queue as Q
from h0lon.agents.usage import append_usage
from h0lon.config import Settings
from h0lon.extract.model import ExtractResult
from h0lon.synth import build as build_mod
from h0lon.synth.model import BuildResult
from h0lon.web.app import create_app
from h0lon.web.jobs import KIND_TITLES, Job, JobManager, TopicBusyError
from h0lon.workspace import create_topic

A = "/t/kurs/a"
B = "/t/kurs/b"


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture
def topics(settings: Settings) -> tuple[Path, Path]:
    return (
        create_topic(settings, title="Тема А", course="Курс", slug="a"),
        create_topic(settings, title="Тема Б", course="Курс", slug="b"),
    )


@pytest.fixture(autouse=True)
def no_real_keep_awake(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []

    @contextmanager
    def fake(enabled: bool = True, **_: Any) -> Iterator[bool]:
        calls.append(enabled)
        yield False

    monkeypatch.setattr(Q, "keep_awake", fake)
    return calls


class Gate:
    """A fake build: emits the stage starts, may hold the build on an event."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.hold = False
        self.results: dict[str, BuildResult] = {}

    def __call__(self, settings: Settings, topic_dir: Path, **kw: Any) -> BuildResult:
        name = Path(topic_dir).name
        self.calls.append(name)
        on_event = kw["on_event"]
        on_event(f"{build_mod.STAGE_TITLES['extract']}…")
        if self.hold:
            self.entered.set()
            assert self.release.wait(15), "the test never released the build"
        on_event(f"{build_mod.STAGE_TITLES['outline']}…")
        return self.results.get(name) or BuildResult(
            ok=True, topic_dir=Path(topic_dir), message=f"Готово: {name}."
        )


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Gate:
    fake = Gate()
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake)
    return fake


def flashes(client: TestClient, response) -> str:
    assert response.status_code == 303, response.text
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    return page.text


def items(settings: Settings) -> dict[str, Q.QueueItem]:
    return {i.topic: i for i in Q.list_items(settings)}


def wait_for_queue_job(app, timeout: float = 20.0) -> Job:
    jobs: JobManager = app.state.jobs
    job = jobs.queue_job()
    assert job is not None
    return jobs.wait(job.id, timeout)


def journal(settings: Settings, **windows: tuple[float, datetime]) -> None:
    """usage.jsonl with the given windows: name → (utilization, resets_at)."""
    now = datetime.now(UTC)
    limits = {
        "status": "allowed",
        "windows": {
            name: {"utilization": util, "resets_at": resets.isoformat(timespec="seconds")}
            for name, (util, resets) in windows.items()
        },
    }
    append_usage(
        settings.general.state_path,
        {"ts": now.isoformat(timespec="seconds"), "backend": "claude", "limits": limits},
    )


def post_run(client: TestClient, **form: str) -> str:
    response = client.post("/queue/run", data=form)
    return flashes(client, response)


# ---------------------------------------------------------------- the page


def test_empty_queue_page(client: TestClient) -> None:
    page = client.get("/queue")
    assert page.status_code == 200
    assert "<title>Очередь сборок — H0lon</title>" in page.text
    assert "Очередь пуста" in page.text and "Данных о лимитах пока нет" in page.text
    assert re.search(r'<a href="/queue" aria-current="page">Очередь</a>', page.text)
    run = re.search(
        r'<button class="btn btn-primary" type="submit"[^>]*>Запустить очередь', page.text
    )
    assert run and "disabled" in run.group(0)
    assert 'id="job"' not in page.text  # no job yet
    assert 'href="/queue"' in client.get("/").text  # the navigation of every page


def test_the_topic_page_has_queue_buttons_and_the_changelog_link(
    client: TestClient, topics: tuple[Path, Path]
) -> None:
    html = client.get(A).text
    assert 'name="action" value="build" formaction="/t/kurs/a/queue"' in html
    assert 'name="action" value="extract" formaction="/t/kurs/a/queue"' in html
    assert ">В очередь</button>" in html and "Тема в очереди" not in html
    assert "Журнал изменений" not in html
    (topics[0] / "synthesis").mkdir(exist_ok=True)
    (topics[0] / "synthesis" / "changelog.md").write_text("# Журнал\n", encoding="utf-8")
    (topics[0] / "master.md").write_text("# Тема\n", encoding="utf-8")
    html = client.get(A).text
    assert 'href="/files/kurs/a/synthesis/changelog.md"' in html and "Журнал изменений" in html
    assert client.get("/files/kurs/a/synthesis/changelog.md").status_code == 200


def test_job_kinds_know_the_queue() -> None:
    assert KIND_TITLES["queue"] == "Очередь сборок"


# ---------------------------------------------------------------- adding


def test_the_build_form_options_travel_with_the_item(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    response = client.post(
        f"{A}/queue",
        data={
            "action": "build",
            "force": "1",
            "no_review": "1",
            "from_stage": "outline",
            "backend": "codex",
        },
    )
    assert response.headers["location"].startswith(f"{A}?flash=")
    assert response.headers["location"].endswith("#build")
    assert "Тема поставлена в очередь (сборка)" in flashes(client, response)
    item = items(settings)["kurs/a"]
    assert item.action == "build" and item.status == "queued"
    assert item.params == {
        "backend": "codex",
        "from_stage": "outline",
        "review": False,
        "force": True,
    }
    again = client.post(f"{A}/queue", data={"action": "build"})
    assert "kurs/a: уже в очереди" in flashes(client, again)
    assert len(Q.list_items(settings)) == 1


def test_the_extract_form_options_travel_with_the_item(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    client.post(f"{A}/queue", data={"action": "extract", "no_vision": "1", "force": "1"})
    item = items(settings)["kurs/a"]
    assert item.action == "extract" and item.params == {"use_vision": False, "force": True}
    page = client.get(A).text
    assert "Тема в очереди: извлечение (в очереди)" in page and 'href="/queue"' in page


def test_bad_requests_to_the_queue_buttons(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    response = client.post(f"{A}/queue", data={"action": "publish"})
    assert "Неизвестное действие: publish" in flashes(client, response)
    response = client.post(f"{A}/queue", data={"action": "build", "from_stage": "bogus"})
    assert "Неизвестная стадия" in flashes(client, response)
    assert Q.list_items(settings) == []
    missing = client.post("/t/kurs/net/queue", data={"action": "build"})
    assert missing.status_code == 404 and "Тема не найдена" in missing.text


def test_the_page_lists_items_with_links_and_options(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    client.post(f"{A}/queue", data={"action": "build", "no_review": "1", "force": "1"})
    client.post(f"{B}/queue", data={"action": "extract"})
    Q._update(
        settings, items(settings)["kurs/b"].id, status="paused", message="Остановка на review gate"
    )
    html = client.get("/queue").text
    assert '<a href="/t/kurs/a">Тема А</a>' in html and '<a href="/t/kurs/b">Тема Б</a>' in html
    assert "без остановки на review gate, без кэша" in html
    assert "badge-queued" in html and "в очереди" in html
    assert "badge-warn" in html and "пауза" in html and "Остановка на review gate" in html
    assert html.count('action="/queue/') >= 3  # remove ×2, run
    assert "Запустить очередь (2)" in html
    assert re.search(r'<button class="btn btn-primary" type="submit">Запустить очередь', html)


def test_the_page_survives_a_deleted_topic_and_a_broken_file(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    client.post(f"{A}/queue", data={"action": "build"})
    (topics[0] / "topic.yaml").unlink()
    html = client.get("/queue").text
    assert "тема не найдена" in html and "kurs/a" in html
    Q.queue_path(settings).write_text("{не json", encoding="utf-8")
    broken = client.get("/queue")
    assert broken.status_code == 200 and "повреждён" in broken.text


def test_limits_panel(client: TestClient, settings: Settings) -> None:
    now = datetime.now(UTC)
    journal(
        settings,
        five_hour=(0.5, now + timedelta(hours=2)),
        seven_day=(0.2, now + timedelta(days=3)),
    )
    html = client.get("/queue").text
    assert "Лимиты подписки" in html and 'value="50"' in html and 'value="20"' in html
    assert "порог 90 %" in html and "порог 95 %" in html and "Лимиты позволяют работать" in html
    journal(settings, five_hour=(0.93, now + timedelta(hours=1)))
    html = client.get("/queue").text
    assert "Сейчас очередь ждала бы сброса окна: 5-часовое окно подписки занято на 93 %" in html
    assert "limit-hot" in html
    journal(settings, seven_day=(0.99, now + timedelta(days=1)))
    assert (
        "Сейчас очередь остановилась бы: недельный лимит подписки занят на 99 %"
        in client.get("/queue").text
    )


# ---------------------------------------------------------------- running


def test_run_with_nothing_queued(client: TestClient) -> None:
    assert "В очереди нет элементов" in post_run(client)
    assert client.get("/queue").text.count("Запустить очередь") >= 1


def test_the_queue_job_builds_the_topics(
    client: TestClient, app, settings: Settings, topics: tuple[Path, Path], gate: Gate
) -> None:
    client.post(f"{A}/queue", data={"action": "build", "no_review": "1"})
    client.post(f"{B}/queue", data={"action": "build"})
    text = post_run(client)
    assert "Задача «Очередь сборок» запущена." in text
    job = wait_for_queue_job(app)
    assert job.kind == "queue" and job.status == "done" and job.topic == "queue"
    assert gate.calls == ["a", "b"]
    assert job.result["done"] == 2 and job.result["message"] == "Очередь: готово: 2."
    assert any("kurs/a (сборка) — начало" in e for e in job.events)
    assert {i.status for i in Q.list_items(settings)} == {"done"}

    html = client.get("/queue").text
    assert 'id="job"' in html and "Очередь сборок" in html and "badge-done" in html
    assert "Очередь: готово: 2." in html and "kurs/a (build)" in html
    assert "Убрать выполненные" in html
    data = client.get(f"/jobs/{job.id}").json()
    assert data["kind"] == "queue" and data["status"] == "done"


def test_the_unattended_flag_reaches_the_runner(
    client: TestClient, app, topics: tuple[Path, Path], gate: Gate, no_real_keep_awake: list[bool]
) -> None:
    client.post(f"{A}/queue", data={"action": "build"})
    post_run(client, unattended="1")
    wait_for_queue_job(app)
    assert no_real_keep_awake == [True]  # unattended and queue.keep_awake


def test_a_failed_item_makes_the_job_failed(
    client: TestClient, app, settings: Settings, topics: tuple[Path, Path], gate: Gate
) -> None:
    gate.results["a"] = BuildResult(
        ok=False, topic_dir=topics[0], stopped_at="render", message="Стадия «PDF (S6)» остановлена"
    )
    client.post(f"{A}/queue", data={"action": "build"})
    client.post(f"{B}/queue", data={"action": "build"})
    post_run(client)
    job = wait_for_queue_job(app)
    assert job.status == "failed" and "остановлена после ошибки" in job.error.lower()
    assert items(settings)["kurs/b"].status == "queued"
    html = client.get("/queue").text
    assert "badge-failed" in html and "PDF (S6)" in html


def test_a_second_start_while_the_queue_runs(
    client: TestClient, app, settings: Settings, topics: tuple[Path, Path], gate: Gate
) -> None:
    gate.hold = True
    client.post(f"{A}/queue", data={"action": "build"})
    client.post(f"{B}/queue", data={"action": "build"})
    post_run(client)
    assert gate.entered.wait(10)
    try:
        assert "Очередь уже обрабатывается." in post_run(client)
        html = client.get("/queue").text
        assert "badge-running" in html and "Очередь уже обрабатывается" in html
        assert re.search(r'<button class="btn btn-sm"[^>]*>Остановить</button>', html)
        running = items(settings)["kurs/a"]
        assert running.status == "running"
        # a running item cannot be removed
        denied = client.post(f"/queue/{running.id}/remove")
        assert "Выполняемый элемент нельзя убрать" in flashes(client, denied)

        # the topic the queue works on is busy for every other job, and its page says so
        busy = client.post(f"{A}/build", data={})
        assert "уже выполняется задача «Очередь сборок»" in flashes(client, busy)
        page = client.get(A).text
        assert "Очередь сборок" in page and 'data-job-active="1"' in page
        assert app.state.jobs.active_for(topics[0]) is not None
        assert app.state.jobs.active_for(topics[1]) is None  # b has not started yet
    finally:
        gate.release.set()
    job = wait_for_queue_job(app)
    assert job.status == "done" and app.state.jobs.active_for(topics[0]) is None


def test_stopping_the_queue_job_between_stages(
    client: TestClient, app, settings: Settings, topics: tuple[Path, Path], gate: Gate
) -> None:
    gate.hold = True
    client.post(f"{A}/queue", data={"action": "build"})
    client.post(f"{B}/queue", data={"action": "build"})
    post_run(client)
    assert gate.entered.wait(10)
    job = app.state.jobs.queue_job()
    cancel = client.post(f"/jobs/{job.id}/cancel")
    assert cancel.status_code == 303 and cancel.headers["location"].startswith("/queue?flash=")
    assert "Остановка запрошена" in flashes(client, cancel)
    gate.release.set()
    job = wait_for_queue_job(app)
    assert job.status == "stopped" and job.result["reason"] == "cancelled"
    assert gate.calls == ["a"]
    assert items(settings)["kurs/a"].status == "paused"
    assert "Остановлено по запросу между стадиями" in items(settings)["kurs/a"].message
    assert items(settings)["kurs/b"].status == "queued"


def test_the_weekly_limit_stops_the_queue_job(
    client: TestClient, app, settings: Settings, topics: tuple[Path, Path], gate: Gate
) -> None:
    journal(settings, seven_day=(0.99, datetime.now(UTC) + timedelta(days=2)))
    client.post(f"{A}/queue", data={"action": "build"})
    post_run(client)
    job = wait_for_queue_job(app)
    assert job.status == "stopped" and job.result["reason"] == "limit" and gate.calls == []
    assert items(settings)["kurs/a"].status == "paused"
    html = client.get("/queue").text
    assert "недельный лимит подписки занят на 99 %" in html and "badge-stopped" in html


def test_a_topic_busy_with_another_job_is_skipped_and_stays_queued(
    client: TestClient,
    app,
    settings: Settings,
    topics: tuple[Path, Path],
    gate: Gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, release = threading.Event(), threading.Event()

    def slow_extract(settings: Settings, topic_dir: Path, **kw: Any) -> list[ExtractResult]:
        started.set()
        assert release.wait(15)
        return []

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", slow_extract)
    from h0lon.sources.ingest import add_sources

    note = topics[0].parent / "note.md"
    note.write_text("# Заметка\n\nТекст.\n", encoding="utf-8")
    add_sources(settings, topics[0], [str(note)])
    assert client.post(f"{A}/extract", data={}).status_code == 303
    assert started.wait(10)
    try:
        client.post(f"{A}/queue", data={"action": "build"})
        client.post(f"{B}/queue", data={"action": "build"})
        post_run(client)
        job = wait_for_queue_job(app)
        assert job.status == "done" and gate.calls == ["b"]
        assert items(settings)["kurs/a"].status == "queued"
        assert "Пропущено: для темы уже выполняется задача" in items(settings)["kurs/a"].message
        assert "пропущено (тема занята): 1" in job.result["message"]
        html = client.get("/queue").text
        assert "Пропущено: для темы уже выполняется задача" in html
    finally:
        release.set()


# ---------------------------------------------------------------- removing


def test_remove_and_clear(
    client: TestClient, settings: Settings, topics: tuple[Path, Path]
) -> None:
    client.post(f"{A}/queue", data={"action": "build"})
    client.post(f"{B}/queue", data={"action": "build"})
    first, second = Q.list_items(settings)
    Q._update(settings, first.id, status="done")
    removed = client.post(f"/queue/{second.id}/remove")
    assert "Элемент убран из очереди" in flashes(client, removed)
    again = client.post(f"/queue/{second.id}/remove")
    assert "Элемента уже нет в очереди" in flashes(client, again)
    assert client.post("/queue/не-id/remove").status_code == 404

    client.post(f"{B}/queue", data={"action": "extract"})
    cleared = client.post("/queue/clear", data={"done": "1"})
    assert "Убрано элементов: 1" in flashes(client, cleared)
    assert [i.action for i in Q.list_items(settings)] == ["extract"]
    everything = client.post("/queue/clear")
    assert "Убрано элементов: 1" in flashes(client, everything)
    assert Q.list_items(settings) == []


# ---------------------------------------------------------------- the manager


def test_claim_reserves_a_topic_against_other_jobs(
    settings: Settings, topics: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    hold, entered = threading.Event(), threading.Event()

    def blocked(settings: Settings, topic_dir: Path, **kw: Any) -> BuildResult:
        entered.set()
        assert hold.wait(15)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="Готово")

    monkeypatch.setattr("h0lon.synth.build.build_topic", blocked)
    manager = JobManager(settings)
    other = manager.start("kurs/b", topics[1], "build")
    assert entered.wait(10)
    queue_job = Job(
        id="q1",
        topic="queue",
        topic_dir=settings.general.state_path / "queue",
        kind="queue",
        params={},
    )
    manager._jobs["q1"] = queue_job  # not started: only its claims matter here
    manager._order.append("q1")
    queue_job.status = "running"
    try:
        reason = manager.claim_topic(queue_job, topics[1])
        assert reason is not None and "уже выполняется задача «Сборка мастер-конспекта»" in reason
        assert queue_job.extra_dirs == []
        assert manager.claim_topic(queue_job, topics[0]) is None
        assert queue_job.extra_dirs == [topics[0]]
        with pytest.raises(TopicBusyError) as busy:
            manager.start("kurs/a", topics[0], "build")
        assert busy.value.job is queue_job and "Очередь сборок" in str(busy.value)
        assert manager.active_for(topics[0]) is queue_job
        assert manager.recent(topics[1]) == [other]
        manager.release_topic(queue_job)
        assert queue_job.extra_dirs == [] and manager.active_for(topics[0]) is None
    finally:
        hold.set()
        manager.wait(other.id)
        queue_job.status = "done"


def test_queue_job_lookup(settings: Settings) -> None:
    manager = JobManager(settings)
    assert manager.queue_job() is None
    done = Job(id="1", topic="queue", topic_dir=Path("x"), kind="queue", params={})
    done.status = "done"
    live = Job(id="2", topic="queue", topic_dir=Path("x"), kind="queue", params={})
    live.status = "running"
    for job in (done, live):
        manager._jobs[job.id] = job
        manager._order.append(job.id)
    assert manager.queue_job() is live
    live.status = "done"
    assert manager.queue_job() is live  # the latest one when none is active
    assert json.dumps(live.to_dict())
