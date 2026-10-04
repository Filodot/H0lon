"""Web interface: the «Оценка» block of the topic page (next to the extraction plan).

`extract_topic` is a fake that returns plans (no files, no agents); the probe of faster-whisper
is a fake; nothing else is touched.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from h0lon import estimate as est
from h0lon.config import Settings
from h0lon.extract import asr
from h0lon.extract.model import ExtractPlan
from h0lon.sources.models import SourceRecord
from h0lon.web.app import create_app
from h0lon.workspace import create_topic, load_topic, save_topic

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

TOPIC = "/t/kurs/metricheskie-metody"
URL = "https://abc-def.trycloudflare.com"
TOKEN = "tok-" + "z" * 40


@pytest.fixture(autouse=True)
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        asr, "probe", lambda: asr.AsrProbe(installed=True, version="1.2.1", cuda_devices=1)
    )


def source(sid: str, kind: str, **kw: Any) -> dict[str, Any]:
    rec = SourceRecord(
        id=sid, kind=kind, title=f"Источник {sid}", added="2026-10-04T00:00:00Z", **kw
    )
    return rec.model_dump(mode="json")


def make_topic(settings: Settings, *records: dict[str, Any]) -> Path:
    path = create_topic(
        settings, title="Метрические методы", course="Курс", slug="metricheskie-metody"
    )
    meta = load_topic(path)
    meta.sources = list(records)
    save_topic(path, meta)
    return path


def lecture(settings: Settings) -> Path:
    return make_topic(
        settings,
        source("V1", "video", units={"minutes": 90}),
        source("S1", "slides", units={"pages": 31, "slides": 31}),
    )


PLANS = [
    ExtractPlan("V1", pages_vision=6, agent_runs=5, notes=["Длительность: 90.0 мин"]),
    ExtractPlan("S1", pages_total=31, pages_vision=21, agent_runs=4, notes=["аннотация"]),
]


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"count": 0}

    def fake_extract(settings: Settings, topic_dir: Path, **kwargs: Any) -> list[ExtractPlan]:
        calls["count"] += 1
        calls.update(kwargs)
        return [ExtractPlan(**vars(p)) for p in PLANS]

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract)
    return calls


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


def client_for(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings), follow_redirects=False) as c:
        yield c


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    lecture(settings)
    yield from client_for(settings)


def block(html: str) -> str:
    match = re.search(r'<div class="estimate" id="estimate">.*?\n    </div>\n', html, re.S)
    assert match, "no estimate block"
    return match.group(0)


# ---------------------------------------------------------------- the block


def test_the_page_offers_the_estimate_but_does_not_compute_it(
    client: TestClient, seen: dict[str, Any]
) -> None:
    html = client.get(TOPIC).text
    assert 'id="estimate"' not in html and seen["count"] == 0  # a dry run costs time
    assert f'formaction="{TOPIC}/estimate"' in html and "Оценить время" in html
    assert f'formaction="{TOPIC}/plan"' in html


def test_the_plan_comes_with_the_estimate(client: TestClient, seen: dict[str, Any]) -> None:
    html = client.post(f"{TOPIC}/plan").text
    assert "План извлечения" in html and seen["count"] == 1  # one dry run for both
    section = block(html)
    assert "<h3>Оценка</h3>" in section
    assert "Источников: 2 (к извлечению 2, из кэша 0)" in section
    assert "Страниц и кадров, которые читает агент: 27" in section
    assert "Распознавание речи: 90,0 мин звука" in section


def test_the_estimate_alone(client: TestClient, seen: dict[str, Any]) -> None:
    response = client.post(
        f"{TOPIC}/estimate", data={"no_vision": "1", "force": "1", "backend": "codex"}
    )
    assert response.status_code == 200
    assert seen["dry_run"] is True and seen["force"] is True
    assert seen["use_vision"] is False and seen["backend"] == "codex"
    assert "План извлечения" not in response.text
    assert 'id="estimate"' in response.text


def test_ways_to_recognize_the_speech(client: TestClient, seen: dict[str, Any]) -> None:
    section = block(client.post(f"{TOPIC}/estimate").text)
    rows = section.split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]
    gpu, cpu, colab = rows
    assert "Этот ПК, видеокарта" in gpu and "large-v3" in gpu and "≈ 12 мин" in gpu
    assert "8 с на минуту звука" in gpu and "оценка PRD, замеров ещё нет" in gpu
    assert "по настройкам" in gpu and "быстрее всех" in gpu and "недоступно" not in gpu
    assert "Этот ПК, процессор" in cpu and "small" in cpu and "≈ 22 мин" in cpu
    assert "по настройкам" not in cpu
    assert "Google Colab (worker)" in colab and "МБ)" in colab  # the audio is uploaded
    assert "недоступно: не задан compute.colab_url" in colab


def test_a_measured_speed_is_shown_as_measured(
    settings: Settings, client: TestClient, seen: dict[str, Any]
) -> None:
    asr.record_calibration(
        settings, device="cuda", model="large-v3", audio_seconds=600, wall_seconds=50
    )
    section = block(client.post(f"{TOPIC}/estimate").text)
    assert "по замеру, 1 запуск" in section and "5 с на минуту звука" in section
    assert "≈ 7 мин 30 с" in section  # 90 minutes × 5 s


def test_the_agent_table_and_the_total(client: TestClient, seen: dict[str, Any]) -> None:
    section = block(client.post(f"{TOPIC}/estimate").text)
    assert "Прогоны агента (параллельно до 2)" in section
    for title in ("Извлечение (аннотации", "S1 · структура", "S2 · написание", "S3 · глобальная"):
        assert title in section
    assert "ориентир PRD" in section and "3–6 мин" in section
    assert re.search(r"Итого ориентировочно: <strong>[^<]+</strong>", section)
    assert "ваше внимание ≈ 10 мин" in section and "параллелизме 2" in section


def test_a_configured_worker_is_offered_and_can_be_the_chosen_way(
    make_settings: Callable[..., Settings], seen: dict[str, Any]
) -> None:
    settings = make_settings(
        general={"git_per_topic": False},
        compute={"asr": "colab", "colab_url": URL, "colab_token": TOKEN},
    )
    lecture(settings)
    (client,) = list(client_for(settings))
    html = client.post(f"{TOPIC}/estimate").text
    rows = block(html).split("<tbody>")[1].split("</tbody>")[0].split("<tr>")[1:]
    gpu, _cpu, colab = rows
    assert "недоступно" not in colab and "по настройкам" in colab and "быстрее всех" in colab
    assert "по настройкам" not in gpu
    assert TOKEN not in html and URL not in html  # neither the token nor the address is shown


def test_nothing_to_estimate_when_everything_is_cached(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_topic(settings, source("P1", "pdf-text", units={"pages": 3}))
    monkeypatch.setattr(
        "h0lon.extract.pipeline.extract_topic",
        lambda *a, **k: [ExtractPlan("P1", notes=["кэш актуален — источник будет пропущен"])],
    )
    from h0lon.synth import build

    real_status = build.topic_status

    def built(settings_: Settings, topic_dir: Path) -> dict[str, Any]:
        info = real_status(settings_, topic_dir)  # the page needs the rest of it
        for stage in ("outline", "sections", "global", "coverage"):
            info["stages"][stage] = {"state": "готово"}
        return info

    monkeypatch.setattr("h0lon.synth.build.topic_status", built)
    (client,) = list(client_for(settings))
    section = block(client.post(f"{TOPIC}/estimate").text)
    assert "Распознавание речи не требуется" in section
    assert "Агент не понадобится: всё берётся из кэша" in section


def test_texts_are_escaped(
    client: TestClient, seen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = est.estimate_topic

    def with_a_note(*args: Any, **kwargs: Any) -> est.Estimate:
        result = real(*args, **kwargs)
        result.notes.append("<script>alert(1)</script>")
        return result

    monkeypatch.setattr("h0lon.estimate.estimate_topic", with_a_note)
    html = client.post(f"{TOPIC}/estimate").text
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_a_failing_estimate_is_a_message_and_the_plan_stays(
    client: TestClient, seen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> est.Estimate:
        raise RuntimeError("журнал повреждён")

    monkeypatch.setattr("h0lon.estimate.estimate_topic", broken)
    plan = client.post(f"{TOPIC}/plan")
    assert plan.status_code == 200 and "План извлечения" in plan.text
    assert "Оценку времени построить не удалось: журнал повреждён" in plan.text
    assert 'id="estimate"' not in plan.text
    alone = client.post(f"{TOPIC}/estimate")
    assert alone.status_code == 200 and "журнал повреждён" in alone.text


def test_the_estimate_needs_sources(settings: Settings) -> None:
    make_topic(settings)
    (client,) = list(client_for(settings))
    html = client.get(TOPIC).text
    assert re.search(r'formaction="[^"]+/estimate"[^>]* disabled', html)
    assert re.search(r'formaction="[^"]+/plan"[^>]* disabled', html)


def test_an_unknown_topic_is_a_page_not_a_traceback(client: TestClient) -> None:
    response = client.post("/t/kurs/net-takoj-temy/estimate")
    assert response.status_code == 404 and "Тема не найдена" in response.text
