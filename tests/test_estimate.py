"""The time estimate (`h0lon/estimate.py`, `h0lon estimate`): units of work and the time per way.

Plans of the extraction are made by hand (`ExtractPlan`) unless a test says otherwise; the probe
of faster-whisper is a fake; the agent is never run and nothing touches the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from h0lon import estimate as est
from h0lon.agents.usage import append_usage
from h0lon.cli import app
from h0lon.config import Settings
from h0lon.extract import asr
from h0lon.extract.model import ExtractPlan
from h0lon.sources.ingest import add_sources
from h0lon.sources.models import SourceRecord
from h0lon.workspace import create_topic, load_topic, save_topic

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

CACHED = ["кэш актуален — источник будет пропущен"]
URL = "https://abc-def.trycloudflare.com"
TOKEN = "tok-" + "y" * 40


def probe(cuda: bool = True, installed: bool = True, **kw: Any) -> asr.AsrProbe:
    return asr.AsrProbe(installed=installed, version="1.2.1", cuda_devices=1 if cuda else 0, **kw)


@pytest.fixture(autouse=True)
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe())


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


def record(sid: str, kind: str, **kw: Any) -> dict[str, Any]:
    rec = SourceRecord(
        id=sid,
        kind=kind,
        title=kw.pop("title", f"Источник {sid}"),
        added="2026-10-04T00:00:00Z",
        **kw,
    )
    return rec.model_dump(mode="json")


def make_topic(settings: Settings, *records: dict[str, Any]) -> Path:
    path = create_topic(settings, title="Метрические методы", course="Курс", slug="metricheskie")
    meta = load_topic(path)
    meta.sources = list(records)
    save_topic(path, meta)
    return path


def lecture_topic(settings: Settings) -> tuple[Path, list[ExtractPlan]]:
    """A 90-minute video, a 31-slide presentation, a cached PDF with a Source Doc."""
    topic = make_topic(
        settings,
        record("V1", "video", units={"minutes": 90}),
        record("S1", "slides", units={"pages": 31, "slides": 31}),
        record("P1", "pdf-text", units={"pages": 10}, quality={"chars_per_page": 3000}),
    )
    doc = topic / "extracted" / "P1"
    doc.mkdir(parents=True)
    (doc / "source.md").write_text("текст " * 5000, encoding="utf-8")  # 30 000 characters
    plans = [
        ExtractPlan("V1", pages_vision=6, agent_runs=5, notes=["Длительность: 90.0 мин"]),
        ExtractPlan("S1", pages_total=31, pages_vision=21, agent_runs=4, notes=["аннотация"]),
        ExtractPlan("P1", notes=list(CACHED)),
    ]
    return topic, plans


def estimate_of(
    settings: Settings, topic: Path, plans: list[ExtractPlan], **kw: Any
) -> est.Estimate:
    return est.estimate_topic(settings, topic, plans=plans, **kw)


# ---------------------------------------------------------------- units of work


def test_units_of_work(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    video, slides, pdf = result.sources
    assert (video.cached, video.audio_minutes, video.agent_pages, video.agent_runs) == (
        False,
        90.0,
        6,
        5,
    )
    assert slides.chars == round(31 * est.CHARS_PER_PAGE["slides"])  # a guess by the slides
    assert pdf.cached and pdf.chars == 30_000  # its Source Doc is there: counted as it is
    assert pdf.audio_minutes is None
    assert result.asr_minutes == 90.0 and result.extraction_pending is True
    assert result.agent_pages == 27  # the cached source reads nothing
    assert result.source_chars == video.chars + slides.chars + pdf.chars
    assert video.chars == round(90 * est.CHARS_PER_AUDIO_MINUTE)
    assert result.title == "Метрические методы" and result.parallel == 2


def test_the_length_of_a_recording_respects_the_limit_and_unknown_lengths(
    make_settings: Callable[..., Settings],
) -> None:
    limited = make_settings(general={"git_per_topic": False}, compute={"video_max_minutes": 15})
    topic = make_topic(
        limited, record("V1", "video", units={"minutes": 90}), record("A1", "audio", units={})
    )
    plans = [ExtractPlan("V1", agent_runs=2), ExtractPlan("A1", agent_runs=2)]
    result = est.estimate_topic(limited, topic, plans=plans)
    assert result.asr_minutes == 15.0 and result.asr_unknown == ["A1"]
    assert any("A1" in n and "станет известна" in n for n in result.notes)


def test_a_cached_recording_costs_nothing(settings: Settings) -> None:
    topic = make_topic(settings, record("V1", "video", units={"minutes": 90}))
    result = estimate_of(settings, topic, [ExtractPlan("V1", notes=list(CACHED))])
    assert result.asr_minutes == 0.0 and result.sources[0].cached
    assert result.total.low == pytest.approx(sum(a.seconds.low for a in result.agents))


def test_the_cached_marker_is_what_the_pipeline_says(settings: Settings) -> None:
    from h0lon.extract import pipeline

    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert est.CACHED_MARK in source  # `_plan_one` must keep saying it


# ---------------------------------------------------------------- the ways to recognize


def way(result: est.Estimate, key: str) -> est.AsrWay:
    return next(w for w in result.asr if w.key == key)


def test_the_starting_values_of_the_prd_without_measurements(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    gpu, cpu, colab = way(result, "local-gpu"), way(result, "local-cpu"), way(result, "colab")
    assert (gpu.model, gpu.seconds_per_minute, gpu.measured) == ("large-v3", 8.0, False)
    assert gpu.seconds == 8.0 * 90 and gpu.upload_s == 0
    assert (cpu.model, cpu.seconds_per_minute) == ("small", 15.0)  # the CPU gets the small model
    assert cpu.seconds == 15.0 * 90
    assert (colab.model, colab.seconds_per_minute, colab.measured) == ("large-v3", 6.0, False)
    # the audio goes as AAC at 48 kbit/s (~32 MB for 90 minutes) at 0.5 MB/s
    assert 30 < colab.upload_mb < 34 and not colab.upload_measured
    assert colab.upload_s == pytest.approx(colab.upload_mb * 1e6 / 500_000, rel=0.01)
    assert colab.seconds == pytest.approx(6.0 * 90 + colab.upload_s)
    assert gpu.available and cpu.available and not colab.available
    assert "compute.colab_url" in colab.reason
    assert [w.selected for w in result.asr] == [True, False, False]  # auto → the GPU
    assert [w.fastest for w in result.asr] == [True, False, False]  # the Colab is not available


def test_a_configured_worker_is_available_and_chosen_by_compute_asr(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        general={"git_per_topic": False},
        compute={"asr": "colab", "colab_url": URL, "colab_token": TOKEN},
    )
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    colab = way(result, "colab")
    assert colab.available and colab.selected and colab.fastest  # 6 s/min + the upload < 8 s/min
    assert not way(result, "local-gpu").selected
    assert result.total.low >= colab.seconds  # the total follows the chosen way
    no_token = make_settings(
        general={"git_per_topic": False}, compute={"asr": "colab", "colab_url": URL}
    )
    assert "colab_token" in way(estimate_of(no_token, topic, plans), "colab").reason


def test_auto_with_a_worker_takes_it_only_without_cuda(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        general={"git_per_topic": False}, compute={"colab_url": URL, "colab_token": TOKEN}
    )
    topic, plans = lecture_topic(settings)
    assert way(estimate_of(settings, topic, plans), "local-gpu").selected
    monkeypatch.setattr(asr, "probe", lambda: probe(cuda=False))
    result = estimate_of(settings, topic, plans)
    assert way(result, "colab").selected and not way(result, "local-gpu").available
    assert "видеокарта nvidia" in way(result, "local-gpu").reason.lower()


def test_without_cuda_the_cpu_is_chosen_and_the_named_model_stays(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe(cuda=False))
    settings = make_settings(general={"git_per_topic": False})
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert way(result, "local-cpu").selected and way(result, "local-cpu").model == "small"
    assert not way(result, "local-gpu").available
    named = make_settings(general={"git_per_topic": False}, compute={"asr_model": "large-v3"})
    heavy = way(estimate_of(named, topic, plans), "local-cpu")
    assert (heavy.model, heavy.seconds_per_minute) == ("large-v3", 80.0)  # PRD: hours on a CPU


def test_a_missing_package_makes_both_local_ways_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe(installed=False, cuda=False))
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert not way(result, "local-gpu").available and not way(result, "local-cpu").available
    assert "faster-whisper не установлен" in way(result, "local-cpu").reason.lower()
    assert any("недоступен" in n for n in result.notes)  # the chosen way cannot work


def test_cuda_libraries_missing_is_the_reason_of_the_gpu(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: probe(cuda_missing=["cublas64_12.dll"]))
    topic, plans = lecture_topic(settings)
    gpu = way(estimate_of(settings, topic, plans), "local-gpu")
    assert not gpu.available and "cublas64_12.dll" in gpu.reason


def test_measurements_replace_the_starting_values(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        general={"git_per_topic": False}, compute={"colab_url": URL, "colab_token": TOKEN}
    )
    asr.record_calibration(
        settings, device="cuda", model="large-v3", audio_seconds=600, wall_seconds=50
    )
    asr.record_calibration(
        settings, device="colab", model="large-v3", audio_seconds=1200, wall_seconds=60
    )
    asr.record_calibration(
        settings, device="colab", model="large-v3", audio_seconds=1200, wall_seconds=180
    )
    asr.record_upload(settings, size=20_000_000, seconds=10.0)  # 2 MB/s
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    gpu, colab = way(result, "local-gpu"), way(result, "colab")
    assert (gpu.seconds_per_minute, gpu.measured, gpu.runs) == (5.0, True, 1)
    assert (colab.seconds_per_minute, colab.measured, colab.runs) == (6.0, True, 2)  # (3+9)/2
    assert colab.upload_measured and colab.upload_s == pytest.approx(colab.upload_mb / 2, rel=0.01)
    assert way(result, "local-cpu").measured is False  # nothing measured for the CPU yet


def test_the_api_is_not_estimated(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(general={"git_per_topic": False}, compute={"asr": "api"})
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert est.selected_way_key(settings, probe()) is None
    assert not any(w.selected for w in result.asr)
    assert any("api" in n for n in result.notes)


# ---------------------------------------------------------------- the agents


def journal(settings: Settings, stage: str, seconds: float, *, ok: bool = True) -> None:
    append_usage(
        settings.general.state_path,
        {"ts": "2026-10-04T10:00:00Z", "stage": stage, "ok": ok, "duration_s": seconds},
    )


def agent(result: est.Estimate, key: str) -> est.AgentRow:
    return next(a for a in result.agents if a.key == key)


def test_synthesis_runs_by_the_size_of_the_topic() -> None:
    assert est.synthesis_runs(0) == {"outline": 1, "sections": 1, "global": 1, "coverage": 1}
    assert est.synthesis_runs(115_466)["sections"] == 11  # a real topic of five sources
    big = est.synthesis_runs(1_000_000)
    assert (big["sections"], big["global"], big["coverage"]) == (96, 3, 3)


def test_agent_runs_without_a_journal_use_the_prd_numbers(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert [a.key for a in result.agents] == [
        "extraction",
        "outline",
        "sections",
        "global",
        "coverage",
    ]
    extraction = agent(result, "extraction")
    assert extraction.runs == 9 and not extraction.measured  # 5 + 4 runs of the plans
    assert (extraction.run_s.low, extraction.run_s.high) == (60.0, 60.0)  # one minute a run
    assert extraction.waves == 5 and extraction.seconds.low == 300.0  # 2 at a time
    sections = agent(result, "sections")
    assert sections.run_s.low == 180.0 and sections.run_s.high == 360.0  # 3-6 minutes
    chars = result.source_chars
    runs = -(-chars // est.SECTIONS_CHARS_PER_RUN)
    assert sections.runs == runs and sections.waves == -(-runs // 2)
    assert sections.seconds.low == 180.0 * sections.waves
    outline = agent(result, "outline")
    assert (outline.runs, outline.waves, outline.seconds.high) == (1, 1, 360.0)
    assert result.agent_runs == sum(a.runs for a in result.agents)


def test_the_journal_sets_the_time_of_a_run_per_stage(settings: Settings) -> None:
    journal(settings, "outline", 100.0)
    journal(settings, "outline", 200.0)
    journal(settings, "outline", 9999.0, ok=False)  # a failed run says nothing
    journal(settings, "sections", 60.0)
    journal(settings, "summary", 20.0)
    journal(settings, "transcript_fix", 40.0)
    journal(settings, "selftest", 500.0)  # not a stage of the work
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    outline = agent(result, "outline")
    assert outline.measured and outline.run_s.low == outline.run_s.high == 150.0
    assert agent(result, "sections").run_s.low == 60.0 and agent(result, "sections").measured
    assert agent(result, "extraction").run_s.low == 30.0  # the mean of summary and transcript_fix
    assert agent(result, "global").measured is False  # no runs of this stage yet
    assert agent(result, "global").run_s.high == 360.0


def test_parallel_runs_shorten_only_the_sections(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(general={"git_per_topic": False}, agents={"parallel_runs": 4})
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    sections = agent(result, "sections")
    assert result.parallel == 4 and sections.waves == -(-sections.runs // 4)
    assert agent(result, "coverage").waves == agent(result, "coverage").runs


def test_nothing_to_do_when_everything_is_cached_and_built(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    topic = make_topic(settings, record("P1", "pdf-text", units={"pages": 3}))
    states = {s: {"state": "готово"} for s in ("outline", "sections", "global", "coverage")}
    monkeypatch.setattr(
        "h0lon.synth.build.topic_status",
        lambda s, t: {"stages": {**states, "coverage": {"state": "из кэша"}}},
    )
    result = estimate_of(settings, topic, [ExtractPlan("P1", notes=list(CACHED))])
    assert result.agents == [] and result.extraction_pending is False
    assert result.total.high == 0 and result.asr_minutes == 0
    assert est.to_view(result)["nothing"] is True


def test_only_stale_stages_are_counted(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    topic = make_topic(settings, record("P1", "pdf-text", units={"pages": 3}))
    stages = {
        "outline": {"state": "готово"},
        "sections": {"state": "готово"},
        "global": {"state": "устарело"},
        "coverage": {"state": "не выполнялась"},
    }
    monkeypatch.setattr("h0lon.synth.build.topic_status", lambda s, t: {"stages": stages})
    result = estimate_of(settings, topic, [ExtractPlan("P1", notes=list(CACHED))])
    assert [a.key for a in result.agents] == ["global", "coverage"]
    # a source that has to be extracted makes every stage after it stale
    fresh = estimate_of(settings, topic, [ExtractPlan("P1", agent_runs=1)])
    assert [a.key for a in fresh.agents] == [
        "extraction",
        "outline",
        "sections",
        "global",
        "coverage",
    ]


def test_a_topic_that_was_never_built_is_estimated_in_full(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, [p for p in plans if p.source_id != "P1"])
    assert {a.key for a in result.agents} >= {"outline", "sections", "global", "coverage"}


def test_without_the_agent_no_extraction_runs_are_counted(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    for plan in plans:
        plan.agent_runs = 0
    result = estimate_of(settings, topic, plans, use_vision=False)
    assert "extraction" not in [a.key for a in result.agents]
    assert any("--no-vision" in n for n in result.notes)


# ---------------------------------------------------------------- the total


def test_the_total_is_the_chosen_way_plus_the_agents(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    gpu = way(result, "local-gpu")
    expected_low = gpu.seconds + sum(a.seconds.low for a in result.agents)
    expected_high = gpu.seconds + sum(a.seconds.high for a in result.agents)
    assert result.total.low == pytest.approx(expected_low)
    assert result.total.high == pytest.approx(expected_high)
    assert result.attention_s == est.ATTENTION_S


def test_a_chosen_way_that_cannot_work_falls_back_to_the_first_available_for_the_total(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(general={"git_per_topic": False}, compute={"asr": "colab"})
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert way(result, "colab").selected and not way(result, "colab").available
    gpu = way(result, "local-gpu")
    assert result.total.low == pytest.approx(
        gpu.seconds + sum(a.seconds.low for a in result.agents)
    )


# ---------------------------------------------------------------- text and the real dry run


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0 с"),
        (45, "45 с"),
        (80, "1 мин 20 с"),
        (180, "3 мин"),
        (660, "11 мин"),
        (3900, "1 ч 05 мин"),
    ],
)
def test_human_seconds(seconds: float, text: str) -> None:
    assert est.human_seconds(seconds) == text


@pytest.mark.parametrize(
    ("low", "high", "text"),
    [
        (720, None, "≈ 12 мин"),
        (180, 360, "3–6 мин"),
        (1080, 2160, "18–36 мин"),
        (2700, 5400, "45 мин – 1 ч 30 мин"),
        (20, 40, "20–40 с"),
        (0, 0, "0 с"),
    ],
)
def test_human_span(low: float, high: float | None, text: str) -> None:
    assert est.human_span(low, high) == text


def test_runs_word() -> None:
    assert [est.runs_word(n) for n in (1, 2, 5, 11, 21, 24)] == [
        "1 прогон",
        "2 прогона",
        "5 прогонов",
        "11 прогонов",
        "21 прогон",
        "24 прогона",
    ]


def test_the_view_is_plain_text_for_a_template(settings: Settings) -> None:
    topic, plans = lecture_topic(settings)
    view = est.to_view(estimate_of(settings, topic, plans))
    assert (view["sources"], view["to_extract"], view["cached"]) == (3, 2, 1)
    assert view["asr_minutes"] == "90,0" and view["agent_pages"] == 27
    gpu, cpu, colab = view["ways"]
    assert gpu["time"] == "≈ 12 мин" and gpu["selected"] and gpu["fastest"]
    assert gpu["speed"] == "8 с на минуту звука" and "оценка PRD" in gpu["speed_source"]
    assert gpu["upload"] == "—" and cpu["time"] == "≈ 22 мин"
    assert not colab["available"] and "compute.colab_url" in colab["reason"]
    assert colab["upload"].endswith("МБ)")
    assert view["agents"][0]["title"].startswith("Извлечение") and view["agents"][0]["runs"] == 9
    assert view["total"] and view["attention"] == "10 мин" and view["parallel"] == 2
    json.dumps(view)  # nothing exotic in it


def test_the_source_of_a_speed_is_said(settings: Settings) -> None:
    asr.record_calibration(
        settings, device="cuda", model="large-v3", audio_seconds=600, wall_seconds=50
    )
    topic, plans = lecture_topic(settings)
    result = estimate_of(settings, topic, plans)
    assert est.way_source_text(way(result, "local-gpu")) == "по замеру, 1 запуск"
    assert est.way_source_text(way(result, "local-cpu")) == "оценка PRD, замеров ещё нет"
    assert est.agent_source_text(agent(result, "outline")) == "ориентир PRD"


def test_a_real_dry_run_of_the_extraction(settings: Settings, tmp_path: Path) -> None:
    topic = create_topic(settings, title="Тема", course="Курс", slug="tema")
    note = tmp_path / "notes.md"
    note.write_text("# Метрики\n\nРасстояние симметрично.\n" * 5, encoding="utf-8")
    add_sources(settings, topic, [str(note)])
    result = est.estimate_topic(settings, topic)  # `extract_topic(dry_run=True)` inside
    (source,) = result.sources
    assert source.id == "D1" and not source.cached and source.chars > 0
    assert result.asr_minutes == 0 and result.extraction_pending
    assert {a.key for a in result.agents} >= {"outline", "sections"}
    data = result.to_dict()
    assert data["agent_runs"] == result.agent_runs and data["total"]["high"] > 0
    json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------- the command


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
    return lambda *args: runner.invoke(app, ["-c", str(config), *args])


def test_the_command_prints_tables(
    settings: Settings, cli: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    monkeypatch.setattr("h0lon.cli.console", Console(width=200))  # tables do not wrap
    _topic, plans = lecture_topic(settings)
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", lambda *a, **k: plans)
    result = cli("estimate", "metricheskie")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Оценка времени: Метрические методы" in out
    assert "Распознавание речи: 90,0 мин звука" in out
    assert "Этот ПК" in out and "видеокарта" in out and "Google" in out and "Colab" in out
    assert "по настройкам" in out and "быстрее всех" in out
    assert "Прогоны агента (параллельно до 2)" in out and "Итого ориентировочно:" in out
    assert "из кэша 1" in out and "Ваше внимание" in out


def test_the_command_json(
    settings: Settings, cli: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _topic, plans = lecture_topic(settings)
    seen: dict[str, Any] = {}

    def fake(settings_: Any, topic_dir: Any, **kwargs: Any) -> list[ExtractPlan]:
        seen.update(kwargs)
        return plans

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake)
    result = cli("estimate", "metricheskie", "--json", "--no-vision")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["asr_minutes"] == 90.0 and data["asr"][0]["key"] == "local-gpu"
    assert data["total"]["low"] > 0 and data["agent_runs"] > 0
    assert seen["use_vision"] is False and seen["dry_run"] is True


def test_the_command_reports_a_missing_topic(cli: Callable[..., Any]) -> None:
    result = cli("estimate", "net-takoy-temy")
    assert result.exit_code == 2 and "Ошибка" in result.output
