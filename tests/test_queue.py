"""Build queue: the file, pace by the subscription limits, the run, keep-awake, the CLI.

Builds, extraction and variations are replaced by fakes; the clock and `sleep` are injected, so no
test waits. The usage journal is written by hand with `limits` as the agents write them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from h0lon import queue as Q
from h0lon.agents.usage import append_usage
from h0lon.cli import app
from h0lon.config import Settings
from h0lon.synth import build as build_mod
from h0lon.synth.model import BuildResult
from h0lon.workspace import create_topic

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
REAL_KEEP_AWAKE = Q.keep_awake  # the autouse fixture replaces Q.keep_awake in every test


# ---------------------------------------------------------------- fakes


class Clock:
    """Injected `now` and `sleep`: sleeping moves the clock, nothing really waits."""

    def __init__(self, trace: list[Any] | None = None, start: datetime = T0) -> None:
        self.t = start
        self.sleeps: list[float] = []
        self.trace = trace if trace is not None else []

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.trace.append(("sleep", seconds))
        self.t += timedelta(seconds=seconds)

    @property
    def slept(self) -> float:
        return sum(self.sleeps)


def iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def journal(
    settings: Settings,
    *,
    five: float | None = None,
    five_resets: datetime | None = None,
    seven: float | None = None,
    seven_resets: datetime | None = None,
    status: str = "allowed",
    ts: datetime = T0,
    backend: str = "claude",
    limits_type: str = "five_hour",
) -> None:
    """One attempt of an agent in usage.jsonl, with the windows it reported."""
    windows: dict[str, Any] = {}
    if five is not None:
        windows["five_hour"] = {
            "utilization": five,
            "resets_at": iso(five_resets or ts + timedelta(hours=2)),
        }
    if seven is not None:
        windows["seven_day"] = {
            "utilization": seven,
            "resets_at": iso(seven_resets or ts + timedelta(days=3)),
        }
    limits = (
        {
            "status": status,
            "type": limits_type,
            "resets_at": windows.get(limits_type, {}).get("resets_at"),
            "using_overage": False,
            "windows": windows,
        }
        if windows
        else None
    )
    append_usage(
        settings.general.state_path,
        {"ts": iso(ts), "backend": backend, "ok": True, "stage": "sections", "limits": limits},
    )


class FakeBuild:
    """Replacement of `build_topic`: emits the stage-start messages like the real one."""

    STAGES = (
        "extract",
        "review",
        "outline",
        "sections",
        "global",
        "coverage",
        "assemble",
        "render",
        "git",
    )

    def __init__(self, trace: list[Any]) -> None:
        self.trace = trace
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: dict[str, BuildResult | BaseException] = {}
        self.before_stage: Callable[[str, str], None] | None = None  # (topic, stage name)

    def title(self, name: str) -> str:
        if name == "review":
            return build_mod.REVIEW_TITLE
        if name == "git":
            return build_mod.GIT_TITLE
        return build_mod.STAGE_TITLES[name]

    def __call__(self, settings: Settings, topic_dir: Path, **kw: Any) -> BuildResult:
        name = Path(topic_dir).name
        self.calls.append((name, kw))
        self.trace.append(("build", name))
        for stage in self.STAGES:
            if self.before_stage is not None:
                self.before_stage(name, stage)
            self.trace.append(("start", name, stage))
            kw["on_event"](f"{self.title(stage)}…")  # the queue may wait or stop right here
            self.trace.append(("run", name, stage))
        outcome = self.results.get(name)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome or BuildResult(
            ok=True, topic_dir=Path(topic_dir), message=f"Готово: {name}/master.pdf."
        )


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


@pytest.fixture
def trace() -> list[Any]:
    return []


@pytest.fixture
def clock(trace: list[Any]) -> Clock:
    return Clock(trace)


@pytest.fixture
def fake_build(monkeypatch: pytest.MonkeyPatch, trace: list[Any]) -> FakeBuild:
    fake = FakeBuild(trace)
    monkeypatch.setattr("h0lon.synth.build.build_topic", fake)
    return fake


@pytest.fixture(autouse=True)
def no_real_keep_awake(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """run_queue must never touch the real execution state of the machine running the tests."""
    calls: list[bool] = []

    @contextmanager
    def fake(enabled: bool = True, **_: Any) -> Iterator[bool]:
        calls.append(enabled)
        yield False

    monkeypatch.setattr(Q, "keep_awake", fake)
    return calls


def make_topic(settings: Settings, slug: str, title: str | None = None) -> Path:
    return create_topic(settings, title=title or f"Тема {slug}", course="Курс", slug=slug)


def run(settings: Settings, clock: Clock, **kw: Any) -> Q.QueueRunReport:
    events: list[str] = kw.pop("events", [])
    return Q.run_queue(settings, now=clock.now, sleep=clock.sleep, on_event=events.append, **kw)


def statuses(settings: Settings) -> dict[str, str]:
    return {i.topic: i.status for i in Q.list_items(settings)}


# ---------------------------------------------------------------- configuration


def test_thresholds_default_and_validation(make_settings: Callable[..., Settings]) -> None:
    cfg = Settings().queue
    assert cfg.pause_at == 0.9 and cfg.weekly_stop == 0.95
    assert make_settings(queue={"pause_at": 0.8, "weekly_stop": 1}).queue.pause_at == 0.8
    for bad in ({"pause_at": 0}, {"pause_at": 1.2}, {"weekly_stop": -0.1}):
        with pytest.raises(ValidationError):
            Settings(queue=bad)


# ---------------------------------------------------------------- the file


def test_add_list_dedupe_and_params(settings: Settings) -> None:
    a = make_topic(settings, "a")
    make_topic(settings, "b")
    report = Q.add_items(settings, [a, "kurs/b"], params={"review": False, "backend": "codex"})
    assert [i.topic for i in report.added] == ["kurs/a", "kurs/b"]
    assert all(i.status == "queued" and i.action == "build" for i in report.added)
    assert report.added[0].params == {"backend": "codex", "review": False}
    assert len({i.id for i in report.added}) == 2 and report.added[0].added.endswith("Z")

    again = Q.add_items(settings, [a])
    assert not again.added and "kurs/a: уже в очереди" in again.skipped[0]
    other = Q.add_items(settings, [a], action="extract")  # another action is another item
    assert [i.action for i in other.added] == ["extract"]

    raw = json.loads(Q.queue_path(settings).read_text("utf-8"))
    assert raw["version"] == 1 and [i["topic"] for i in raw["items"]] == [
        "kurs/a",
        "kurs/b",
        "kurs/a",
    ]
    assert set(raw["items"][0]) == {
        "id",
        "topic",
        "action",
        "params",
        "status",
        "added",
        "started",
        "finished",
        "message",
    }
    assert [i.id for i in Q.list_items(settings)] == [i["id"] for i in raw["items"]]


def test_add_is_all_or_nothing_and_checks_arguments(settings: Settings) -> None:
    a = make_topic(settings, "a")
    with pytest.raises(FileNotFoundError, match="Тема не найдена"):
        Q.add_items(settings, [a, "kurs/нет-такой"])
    assert Q.list_items(settings) == []
    with pytest.raises(ValueError, match="Неизвестное действие"):
        Q.add_items(settings, [a], action="publish")
    with pytest.raises(ValueError, match="Неизвестный агент"):
        Q.add_items(settings, [a], params={"backend": "gpt"})
    with pytest.raises(ValueError, match="Неизвестная стадия"):
        Q.add_items(settings, [a], params={"from_stage": "bogus"})


def test_topic_outside_the_workspaces_is_stored_by_path(
    settings: Settings, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    other = make_settings(general={"workspaces": tmp_path / "elsewhere", "git_per_topic": False})
    path = make_topic(other, "x")
    report = Q.add_items(settings, [path])  # the queue of `settings`, a topic of another place
    assert report.added[0].topic == str(path.resolve())
    assert Q.resolve_item_topic(settings, report.added[0]) == path.resolve()


def test_remove_and_clear(settings: Settings) -> None:
    topics = [make_topic(settings, s) for s in "abcd"]
    ids = [i.id for i in Q.add_items(settings, topics).added]
    Q._update(settings, ids[0], status="done")
    Q._update(settings, ids[1], status="failed")
    Q._update(settings, ids[2], status="running")
    assert Q.remove_item(settings, ids[3]) is True and Q.remove_item(settings, ids[3]) is False
    with pytest.raises(Q.QueueError, match="Выполняемый"):
        Q.remove_item(settings, ids[2])
    assert Q.clear_items(settings, done_only=True) == 1
    assert [i.status for i in Q.list_items(settings)] == ["failed", "running"]
    assert Q.clear_items(settings) == 1  # everything but the running one
    assert [i.status for i in Q.list_items(settings)] == ["running"]


def test_a_broken_file_is_reported_not_overwritten(settings: Settings) -> None:
    path = Q.queue_path(settings)
    path.parent.mkdir(parents=True)
    path.write_text("{ не json", "utf-8")
    with pytest.raises(Q.QueueError, match="повреждён"):
        Q.list_items(settings)
    with pytest.raises(Q.QueueError, match="повреждён"):
        Q.add_items(settings, [make_topic(settings, "a")])
    assert path.read_text("utf-8") == "{ не json"
    path.write_text(json.dumps([{"id": "1", "topic": "kurs/a"}, "мусор"]), "utf-8")
    (item,) = Q.list_items(settings)  # a bare list is accepted, a broken entry is skipped
    assert item.status == "queued" and item.action == "build"


def test_threads_adding_at_once_lose_nothing(settings: Settings) -> None:
    topics = [make_topic(settings, f"t{i}") for i in range(10)]
    errors: list[BaseException] = []

    def add(path: Path) -> None:
        try:
            Q.add_items(settings, [path])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(p,)) for p in topics]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(i.topic for i in Q.list_items(settings)) == sorted(
        f"kurs/t{i}" for i in range(10)
    )


def test_os_lock_excludes_a_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    first = Q._try_acquire(path)
    assert first is not None
    assert Q._try_acquire(path) is None  # another handle, as another process would have
    Q._unlock_fd(first)
    second = Q._try_acquire(path)
    assert second is not None
    Q._unlock_fd(second)


def test_the_store_lock_is_reentrant(settings: Settings) -> None:
    with Q._store(settings):
        assert Q.list_items(settings) == []  # a nested use in the same thread does not block


# ---------------------------------------------------------------- limits and pace


def test_read_limits_takes_the_newest_value_of_every_window(settings: Settings) -> None:
    journal(settings, five=0.2, seven=0.5, ts=T0 - timedelta(hours=1))
    journal(settings, five=0.6, ts=T0)  # no weekly window in this attempt
    (settings.general.state_path / "usage.jsonl").open("a", encoding="utf-8").write("{обрыв\n")
    journal(settings, ts=T0 + timedelta(minutes=1), backend="codex")  # no limits at all
    snap = Q.read_limits(settings)
    assert snap.known and snap.windows["five_hour"].utilization == 0.6
    assert snap.windows["seven_day"].utilization == 0.5
    assert snap.windows["five_hour"].seen_at == T0 and snap.limits is not None


def test_read_limits_reads_only_the_tail_of_a_big_journal(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal(settings, five=0.99, seven=0.99, ts=T0 - timedelta(days=1))  # far at the start
    for i in range(200):
        append_usage(settings.general.state_path, {"ts": iso(T0), "backend": "codex", "n": i})
    journal(settings, five=0.3, ts=T0)
    monkeypatch.setattr(Q, "USAGE_TAIL_BYTES", 4096)
    snap = Q.read_limits(settings)
    assert snap.windows["five_hour"].utilization == 0.3
    assert "seven_day" not in snap.windows  # beyond the tail: unknown, never a stale guess


def test_no_journal_means_no_limits(settings: Settings) -> None:
    snap = Q.read_limits(settings)
    assert not snap.known
    assert Q.decide_pace(settings, snap, T0).action == "go"
    info = Q.limits_summary(settings, T0)
    assert info["known"] is False and info["action"] == "go" and info["windows"] == []


def test_decide_pace(settings: Settings) -> None:
    def pace(**kw: Any) -> Q.Pace:
        journal(settings, **kw)
        return Q.decide_pace(settings, Q.read_limits(settings), T0)

    assert pace(five=0.89, seven=0.5).action == "go"
    wait = pace(five=0.9, five_resets=T0 + timedelta(minutes=30), seven=0.5, ts=T0)
    assert wait.action == "wait" and "5-часовое окно" in wait.reason and "90 %" in wait.reason
    assert wait.until == T0 + timedelta(minutes=30, seconds=Q.RESET_MARGIN_S)
    stop = pace(five=0.99, seven=0.95)
    assert stop.action == "stop" and "недельный лимит" in stop.reason and "95 %" in stop.reason


def test_a_window_that_has_reset_counts_as_empty(settings: Settings) -> None:
    journal(
        settings,
        five=0.99,
        five_resets=T0 - timedelta(minutes=1),
        seven=0.99,
        seven_resets=T0 - timedelta(hours=1),
        ts=T0 - timedelta(hours=6),
    )
    assert Q.decide_pace(settings, Q.read_limits(settings), T0).action == "go"
    info = Q.limits_summary(settings, T0)
    assert [w["active"] for w in info["windows"]] == [False, False]
    assert not any(w["hot"] for w in info["windows"])


def test_a_window_without_reset_time_ages_out(settings: Settings) -> None:
    append_usage(
        settings.general.state_path,
        {"ts": iso(T0), "limits": {"windows": {"five_hour": {"utilization": 0.97}}}},
    )
    snap = Q.read_limits(settings)
    wait = Q.decide_pace(settings, snap, T0 + timedelta(hours=1))
    assert wait.action == "wait" and wait.until == T0 + timedelta(hours=5, seconds=60)
    assert Q.decide_pace(settings, snap, T0 + timedelta(hours=5, minutes=2)).action == "go"


def test_a_rejected_window_blocks_whatever_its_utilization(settings: Settings) -> None:
    journal(settings, five=0.5, five_resets=T0 + timedelta(hours=1), status="rejected")
    pace = Q.decide_pace(settings, Q.read_limits(settings), T0)
    assert pace.action == "wait" and "100 %" in pace.reason


def test_thresholds_come_from_the_settings(make_settings: Callable[..., Settings]) -> None:
    custom = make_settings(
        general={"git_per_topic": False}, queue={"pause_at": 0.5, "weekly_stop": 0.6}
    )
    journal(custom, five=0.5, seven=0.55)
    assert Q.decide_pace(custom, Q.read_limits(custom), T0).action == "wait"
    journal(custom, five=0.1, seven=0.6)
    assert Q.decide_pace(custom, Q.read_limits(custom), T0).action == "stop"


def test_limits_summary_for_the_pages(settings: Settings) -> None:
    journal(settings, five=0.95, five_resets=T0 + timedelta(hours=1), seven=0.4)
    info = Q.limits_summary(settings, T0)
    assert info["known"] and info["action"] == "wait" and info["pause_at"] == 90
    five, seven = info["windows"]
    assert (five["name"], five["percent"], five["threshold"], five["hot"]) == (
        "five_hour",
        95,
        90,
        True,
    )
    assert (seven["percent"], seven["threshold"], seven["hot"]) == (40, 95, False)
    assert "5 ч — 95 %" in info["text"] and "7 дн — 40 %" in info["text"]


# ---------------------------------------------------------------- the run


def test_run_builds_the_queued_topics_in_order(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    topics = [make_topic(settings, s) for s in ("a", "b")]
    Q.add_items(settings, topics, params={"review": False, "force": True, "backend": "codex"})
    events: list[str] = []
    report = run(settings, clock, events=events)
    assert [c[0] for c in fake_build.calls] == ["a", "b"]
    kw = fake_build.calls[0][1]
    assert kw["review"] is False and kw["force"] is True and kw["backend"] == "codex"
    assert kw["from_stage"] is None
    assert report.ok and report.stopped is None and report.count("done") == 2
    assert statuses(settings) == {"kurs/a": "done", "kurs/b": "done"}
    first = Q.list_items(settings)[0]
    assert first.message == "Готово: a/master.pdf." and first.started and first.finished
    assert report.message == "Очередь: готово: 2."
    assert any("kurs/a (сборка) — начало" in e for e in events)
    assert clock.sleeps == []  # no limits known: no pauses
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_an_empty_queue_is_not_an_error(settings: Settings, clock: Clock) -> None:
    report = run(settings, clock)
    assert report.ok and report.outcomes == [] and "обрабатывать нечего" in report.message


def test_a_failed_item_stops_the_queue_unless_unattended(
    settings: Settings, clock: Clock, fake_build: FakeBuild, no_real_keep_awake: list[bool]
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b", "c")])
    fake_build.results["a"] = BuildResult(
        ok=False, topic_dir=Path("a"), stopped_at="render", message="Стадия «PDF (S6)» остановлена"
    )
    report = run(settings, clock)
    assert report.stopped == "error" and not report.ok
    assert statuses(settings) == {"kurs/a": "failed", "kurs/b": "queued", "kurs/c": "queued"}
    assert "PDF (S6)" in Q.list_items(settings)[0].message and "--unattended" in report.message
    assert no_real_keep_awake == [False]  # attended: the machine may sleep

    fake_build.results["b"] = RuntimeError("взрыв")  # a bug in one item
    report = run(settings, clock, unattended=True)  # a is failed and is not retried
    assert report.stopped is None and report.count("failed") == 1 and report.count("done") == 1
    assert statuses(settings) == {"kurs/a": "failed", "kurs/b": "failed", "kurs/c": "done"}
    assert "Внутренняя ошибка: RuntimeError: взрыв" in Q.list_items(settings)[1].message
    assert no_real_keep_awake == [False, True]  # unattended + queue.keep_awake


def test_the_unattended_setting_counts_too(
    make_settings: Callable[..., Settings], clock: Clock, fake_build: FakeBuild
) -> None:
    settings = make_settings(general={"git_per_topic": False}, queue={"unattended": True})
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    fake_build.results["a"] = RuntimeError("x")
    report = run(settings, clock)
    assert report.count("failed") == 1 and report.count("done") == 1


def test_keep_awake_follows_the_setting(
    make_settings: Callable[..., Settings],
    clock: Clock,
    fake_build: FakeBuild,
    no_real_keep_awake: list[bool],
) -> None:
    settings = make_settings(general={"git_per_topic": False}, queue={"keep_awake": False})
    run(settings, clock, unattended=True)
    assert no_real_keep_awake == [False]


def test_a_topic_stopped_at_the_review_gate_is_paused_and_the_queue_goes_on(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    fake_build.results["a"] = BuildResult(
        ok=False, topic_dir=Path("a"), stopped_at="review", message="Остановка: h0lon approve"
    )
    report = run(settings, clock)
    assert report.stopped is None and report.count("paused") == 1 and report.count("done") == 1
    assert statuses(settings) == {"kurs/a": "paused", "kurs/b": "done"}
    assert "на паузе: 1" in report.message
    # approved later: the paused item is taken again by the next run
    del fake_build.results["a"]
    report = run(settings, clock)
    assert statuses(settings) == {"kurs/a": "done", "kurs/b": "done"} and report.count("done") == 1


def test_a_deleted_topic_fails_its_item_only(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    a, b = make_topic(settings, "a"), make_topic(settings, "b")
    Q.add_items(settings, [a, b])
    (a / "topic.yaml").unlink()
    report = run(settings, clock, unattended=True)
    assert statuses(settings) == {"kurs/a": "failed", "kurs/b": "done"}
    assert "Тема не найдена" in Q.list_items(settings)[0].message and report.count("done") == 1


def test_extract_and_variant_items(
    settings: Settings, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from h0lon.extract.model import ExtractResult

    seen: dict[str, Any] = {}

    def fake_extract(settings: Settings, topic_dir: Path, **kw: Any) -> list[ExtractResult]:
        seen["extract"] = kw
        return [
            ExtractResult(ok=True, source_id="S1"),
            ExtractResult(ok=kw["force"] is not True, source_id="P1"),
        ]

    class FakeVariantResult:
        ok = True

        def message(self) -> str:
            return "Вариация готова."

    def fake_variant(settings: Settings, topic_dir: Path, **kw: Any) -> FakeVariantResult:
        seen["variant"] = kw
        return FakeVariantResult()

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract)
    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake_variant)
    a = make_topic(settings, "a")
    Q.add_items(settings, [a], action="extract", params={"use_vision": False})
    Q.add_items(settings, [a], action="variant", params={"preset": "brief", "force": True})
    report = run(settings, clock)
    assert seen["extract"]["use_vision"] is False and seen["extract"]["force"] is False
    assert seen["variant"]["preset"] == "brief" and seen["variant"]["force"] is True
    assert [o.status for o in report.outcomes] == ["done", "done"]
    assert Q.list_items(settings)[0].message == "Извлечение завершено: источников 2."
    assert Q.list_items(settings)[1].message == "Вариация готова."

    Q.add_items(settings, [a], action="extract", params={"force": True})
    report = run(settings, clock)
    assert report.outcomes[0].status == "failed" and "P1" in report.outcomes[0].message


# ---------------------------------------------------------------- pace in the run


def test_five_hour_window_full_waits_until_it_resets(
    settings: Settings, clock: Clock, fake_build: FakeBuild, trace: list[Any]
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    reset = T0 + timedelta(minutes=45)
    journal(settings, five=0.95, five_resets=reset, seven=0.3)
    events: list[str] = []
    report = run(settings, clock, events=events)
    assert report.ok and report.count("done") == 2
    # one wait, until the reset plus the margin, in pieces; nothing ran before it
    assert clock.slept == pytest.approx(45 * 60 + Q.RESET_MARGIN_S)
    assert max(clock.sleeps) <= Q.WAIT_CHUNK_S
    assert trace.index(("build", "a")) > max(i for i, e in enumerate(trace) if e[0] == "sleep")
    assert clock.now() >= reset + timedelta(seconds=Q.RESET_MARGIN_S)
    wait_event = next(e for e in events if e.startswith("Лимит подписки"))
    assert "5-часовое окно подписки занято на 95 %" in wait_event and "через 46 мин" in wait_event
    assert "Пауза закончена: продолжаем." in events
    assert report.waited_s == pytest.approx(45 * 60 + Q.RESET_MARGIN_S)
    assert "Ожидание сброса лимитов: 46 мин" in report.message
    assert sum(1 for e in events if e.startswith("Лимит подписки")) == 1  # b needs no new wait


def test_the_pace_is_checked_between_the_stages_of_a_build(
    settings: Settings, clock: Clock, fake_build: FakeBuild, trace: list[Any]
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.2, seven=0.2)

    def heat_up(topic: str, stage: str) -> None:
        if stage == "sections":  # the agents of the stages before took the window
            journal(
                settings,
                five=0.93,
                five_resets=clock.now() + timedelta(hours=1),
                ts=clock.now(),
            )

    fake_build.before_stage = heat_up
    report = run(settings, clock)
    assert report.ok
    start = trace.index(("start", "a", "sections"))
    ran = trace.index(("run", "a", "sections"))
    assert any(e[0] == "sleep" for e in trace[start:ran])  # waited right before S2
    assert not any(e[0] == "sleep" for e in trace[:start])
    assert clock.slept == pytest.approx(3600 + Q.RESET_MARGIN_S)


def test_only_stages_with_agents_are_paced(
    settings: Settings, clock: Clock, fake_build: FakeBuild, trace: list[Any]
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.2)

    def heat_up(topic: str, stage: str) -> None:
        if stage == "assemble":
            journal(
                settings, five=0.99, five_resets=clock.now() + timedelta(hours=1), ts=clock.now()
            )

    fake_build.before_stage = heat_up
    assert run(settings, clock).ok
    assert clock.sleeps == []  # assemble, render and git start no agents


def test_weekly_limit_stops_the_queue_before_an_item(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    journal(settings, five=0.3, seven=0.97, seven_resets=T0 + timedelta(days=2))
    events: list[str] = []
    report = run(settings, clock, events=events)
    assert report.stopped == "limit" and not report.ok
    assert fake_build.calls == [] and clock.sleeps == []
    assert statuses(settings) == {"kurs/a": "paused", "kurs/b": "queued"}
    item = Q.list_items(settings)[0]
    assert "недельный лимит подписки занят на 97 %" in item.message
    assert "Остановлена: недельный лимит" in report.message
    # the limit is gone (a new week): the next run takes the paused item first
    journal(settings, five=0.1, seven=0.1, ts=T0 + timedelta(days=3))
    clock.t = T0 + timedelta(days=3)
    report = run(settings, clock)
    assert report.ok and statuses(settings) == {"kurs/a": "done", "kurs/b": "done"}
    assert [c[0] for c in fake_build.calls] == ["a", "b"]


def test_weekly_limit_stops_a_build_between_stages(
    settings: Settings, clock: Clock, fake_build: FakeBuild, trace: list[Any]
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])

    def exhaust(topic: str, stage: str) -> None:
        if stage == "global":
            journal(
                settings, seven=0.96, seven_resets=clock.now() + timedelta(days=1), ts=clock.now()
            )

    fake_build.before_stage = exhaust
    report = run(settings, clock)
    assert report.stopped == "limit"
    assert ("run", "a", "sections") in trace and ("run", "a", "global") not in trace
    assert statuses(settings) == {"kurs/a": "paused", "kurs/b": "queued"}
    message = Q.list_items(settings)[0].message
    assert "Остановлено между стадиями" in message and "сборка продолжится" in message
    assert [c[0] for c in fake_build.calls] == ["a"]


def test_an_old_window_is_not_waited_for(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.99, five_resets=T0 - timedelta(hours=1), ts=T0 - timedelta(hours=6))
    assert run(settings, clock).ok and clock.sleeps == []


def test_waiting_resets_the_cooling_of_the_agents(
    settings: Settings, clock: Clock, fake_build: FakeBuild, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("h0lon.agents.reset_cooling", lambda *a: calls.append("reset"))
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.95, five_resets=T0 + timedelta(minutes=5))
    run(settings, clock)
    assert calls == ["reset"]


def test_a_long_wait_leaves_notes_in_the_log(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.95, five_resets=T0 + timedelta(hours=2))
    events: list[str] = []
    run(settings, clock, events=events)
    notes = [e for e in events if e.startswith("Ждём сброса окна лимита")]
    assert 8 <= len(notes) <= 13  # every ten minutes of two hours
    assert "осталось" in notes[0]


def test_a_stop_request_ends_a_wait_and_leaves_the_item_queued(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    journal(settings, five=0.95, five_resets=T0 + timedelta(hours=3))
    report = run(settings, clock, should_stop=lambda: len(clock.sleeps) >= 3)
    assert report.stopped == "cancelled" and "Остановлена по запросу" in report.message
    assert len(clock.sleeps) == 3 and fake_build.calls == []
    assert statuses(settings) == {"kurs/a": "queued"}


def test_a_stop_request_between_items_and_between_stages(
    settings: Settings, clock: Clock, fake_build: FakeBuild, trace: list[Any]
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    stop = threading.Event()
    fake_build.before_stage = lambda topic, stage: stop.set() if stage == "outline" else None
    report = run(settings, clock, should_stop=stop.is_set)
    assert report.stopped == "cancelled"
    assert ("run", "a", "extract") in trace and ("run", "a", "outline") not in trace
    assert statuses(settings) == {"kurs/a": "paused", "kurs/b": "queued"}
    assert "Остановлено по запросу между стадиями" in Q.list_items(settings)[0].message

    stop.clear()
    fake_build.before_stage = None

    def stop2() -> bool:
        return statuses(settings).get("kurs/a") == "done"

    report = run(settings, clock, should_stop=stop2)  # a finishes, then the stop is seen
    assert report.stopped == "cancelled" and statuses(settings) == {
        "kurs/a": "done",
        "kurs/b": "queued",
    }


def test_a_busy_topic_is_skipped_and_released(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    claimed: list[str] = []
    released: list[str] = []

    def claim(path: Path) -> str | None:
        claimed.append(path.name)
        return "для темы уже выполняется задача «Сборка»" if path.name == "a" else None

    report = run(settings, clock, claim=claim, release=lambda p: released.append(p.name))
    assert claimed == ["a", "b"] and released == ["b"]
    assert statuses(settings) == {"kurs/a": "queued", "kurs/b": "done"}
    assert "Пропущено: для темы уже выполняется" in Q.list_items(settings)[0].message
    assert report.count("queued") == 1 and "пропущено (тема занята): 1" in report.message


def test_only_one_runner_at_a_time_and_dead_runs_are_recovered(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, s) for s in ("a", "b")])
    nested: list[BaseException] = []

    def during_the_build(topic: str, stage: str) -> None:
        if topic == "a" and stage == "extract" and not nested:
            try:
                run(settings, clock)
            except Q.QueueBusyError as exc:
                nested.append(exc)

    fake_build.before_stage = during_the_build
    assert run(settings, clock).ok
    assert nested and "уже обрабатывается" in str(nested[0])

    # a runner that died leaves its item `running`; the next run queues it again and does it
    (first, _second) = Q.list_items(settings)
    Q._update(settings, first.id, status="running", message="x")
    fake_build.before_stage = None
    events: list[str] = []
    report = run(settings, clock, events=events)
    assert report.count("done") == 1 and statuses(settings)["kurs/a"] == "done"
    assert any("Незавершённых элементов с прошлого запуска: 1" in e for e in events)


def test_ctrl_c_leaves_the_item_resumable(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])
    fake_build.results["a"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(settings, clock)
    (item,) = Q.list_items(settings)
    assert item.status == "paused" and "Прервано пользователем" in item.message
    del fake_build.results["a"]
    assert run(settings, clock).ok


def test_the_progress_callback_may_fail(
    settings: Settings, clock: Clock, fake_build: FakeBuild
) -> None:
    Q.add_items(settings, [make_topic(settings, "a")])

    def broken(message: str) -> None:
        raise RuntimeError("сломанный обработчик")

    report = Q.run_queue(settings, now=clock.now, sleep=clock.sleep, on_event=broken)
    assert report.ok


# ---------------------------------------------------------------- keep awake


class FakeKernel32:
    def __init__(self, result: int = 0x80000000) -> None:
        self.calls: list[int] = []
        self.result = result

    def SetThreadExecutionState(self, flags: int) -> int:
        self.calls.append(flags)
        return self.result


def test_keep_awake_on_windows_sets_and_clears_the_state() -> None:
    kernel = FakeKernel32()
    with REAL_KEEP_AWAKE(True, platform="win32", kernel32=kernel) as applied:
        assert applied is True and kernel.calls == [Q.ES_CONTINUOUS | Q.ES_SYSTEM_REQUIRED]
    assert kernel.calls[-1] == Q.ES_CONTINUOUS  # the request is withdrawn


def test_keep_awake_clears_the_state_when_the_block_fails() -> None:
    kernel = FakeKernel32()
    with pytest.raises(RuntimeError), REAL_KEEP_AWAKE(True, platform="win32", kernel32=kernel):
        raise RuntimeError("x")
    assert kernel.calls[-1] == Q.ES_CONTINUOUS


def test_keep_awake_is_harmless_elsewhere_and_when_refused() -> None:
    kernel = FakeKernel32()
    with REAL_KEEP_AWAKE(True, platform="linux", kernel32=kernel) as applied:
        assert applied is False
    with REAL_KEEP_AWAKE(False, platform="win32", kernel32=kernel) as applied:
        assert applied is False
    assert kernel.calls == []
    refusing = FakeKernel32(result=0)  # SetThreadExecutionState returns 0 on failure
    with REAL_KEEP_AWAKE(True, platform="win32", kernel32=refusing) as applied:
        assert applied is False
    assert refusing.calls == [Q.ES_CONTINUOUS | Q.ES_SYSTEM_REQUIRED]  # and nothing to withdraw


# ---------------------------------------------------------------- printing


def render_text(fn: Callable[..., None], *args: Any, **kw: Any) -> str:
    console = Console(record=True, width=200)
    fn(*args, console=console, **kw)
    return console.export_text()


def test_print_queue_and_run(settings: Settings, clock: Clock, fake_build: FakeBuild) -> None:
    text = render_text(Q.print_queue, [], settings=settings)
    assert "Очередь пуста" in text and "Данных о лимитах" in text
    Q.add_items(settings, [make_topic(settings, "a")], params={"review": False})
    journal(settings, five=0.95, five_resets=T0 + timedelta(days=100), seven=0.2)  # no wait
    items = Q.list_items(settings)
    text = render_text(Q.print_queue, items, settings=settings)
    assert "kurs/a" in text and "в очереди" in text and "сборка" in text
    assert "Лимиты Claude:" in text and "пауза до сброса окна" in text and "90 %" in text
    report = run(settings, clock)
    text = render_text(Q.print_run, report)
    assert "Итог запуска очереди" in text and "готово" in text and "kurs/a" in text


# ---------------------------------------------------------------- the CLI


@pytest.fixture
def cli(settings: Settings, tmp_path: Path) -> Callable[..., Any]:
    config = tmp_path / "h0lon.toml"
    config.write_text(
        "[general]\n"
        f"workspaces = '{settings.general.workspaces_dir}'\n"
        f"state_dir = '{settings.general.state_path}'\n"
        "git_per_topic = false\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    def invoke(*args: str, **kw: Any) -> Any:
        return runner.invoke(app, ["-c", str(config), *args], **kw)

    return invoke


def test_cli_add_list_run_clear(
    settings: Settings, cli: Callable[..., Any], fake_build: FakeBuild
) -> None:
    make_topic(settings, "a")
    make_topic(settings, "b")
    res = cli("queue", "add", "kurs/a", "b", "--no-review")
    assert res.exit_code == 0, res.output
    assert "В очереди: kurs/a" in res.output and "h0lon queue run" in res.output
    res = cli("queue", "add", "kurs/a")  # again
    assert res.exit_code == 1 and "уже в очереди" in res.output
    res = cli("queue", "add", "нет/темы")
    assert res.exit_code == 2 and "Ошибка:" in res.output and "Тема не найдена" in res.output
    res = cli("queue", "add", "kurs/b", "--action", "publish")
    assert res.exit_code == 2

    res = cli("queue", "list", "--json")
    data = json.loads(res.stdout)
    assert [i["topic"] for i in data["items"]] == ["kurs/a", "kurs/b"]
    assert data["items"][0]["params"] == {"review": False} and data["limits"]["known"] is False
    res = cli("queue", "list")
    assert res.exit_code == 0 and "Очередь сборок" in res.output and "kurs/b" in res.output

    res = cli("queue", "run", "--json")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert (
        out["ok"]
        and out["done"] == 2
        and [i["topic"] for i in out["items"]] == ["kurs/a", "kurs/b"]
    )
    res = cli("queue", "run")  # nothing left
    assert res.exit_code == 0 and "обрабатывать нечего" in res.output

    res = cli("queue", "clear", "--done")
    assert res.exit_code == 0 and "Убрано элементов: 2" in res.output
    cli("queue", "add", "kurs/a")
    res = cli("queue", "clear", input="n\n")
    assert res.exit_code == 1 and len(Q.list_items(settings)) == 1
    res = cli("queue", "clear", "--yes")
    assert res.exit_code == 0 and Q.list_items(settings) == []


def test_cli_run_exit_codes(
    settings: Settings, cli: Callable[..., Any], fake_build: FakeBuild
) -> None:
    for slug in ("a", "b"):
        make_topic(settings, slug)
    cli("queue", "add", "kurs/a", "kurs/b")
    fake_build.results["a"] = BuildResult(
        ok=False, topic_dir=Path("a"), stopped_at="render", message="остановлена"
    )
    res = cli("queue", "run", "--json")
    assert res.exit_code == 1 and json.loads(res.stdout)["stopped"] == "error"
    res = cli("queue", "run", "--unattended", "--json")  # a is failed, b is done now
    assert res.exit_code == 0 and json.loads(res.stdout)["done"] == 1

    cli("queue", "clear", "--yes")
    cli("queue", "add", "kurs/a")
    journal(
        settings,
        seven=0.99,
        seven_resets=datetime.now(UTC) + timedelta(days=1),
        ts=datetime.now(UTC),
    )
    res = cli("queue", "run")
    assert res.exit_code == 3 and "Остановлена: недельный лимит" in res.output


def test_cli_reports_a_broken_queue_file_and_a_second_runner(
    settings: Settings, cli: Callable[..., Any]
) -> None:
    path = Q.queue_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("не json", "utf-8")
    res = cli("queue", "list")
    assert res.exit_code == 2 and "повреждён" in res.output
    path.unlink()
    lock = Q._try_acquire(path.with_name(Q.RUN_LOCK_FILE))
    assert lock is not None
    try:
        res = cli("queue", "run")
        assert res.exit_code == 2 and "уже обрабатывается" in res.output
    finally:
        Q._unlock_fd(lock)
