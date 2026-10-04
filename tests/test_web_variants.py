"""Web interface: the «Вариации» block of the topic page and the `variant` job.

`run_variant` is replaced by a fake (jobs) or `run_agent` / `render_document` of the module are
(end to end): no agent, no XeLaTeX, no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_synth_variants import MASTER, FakeAgent, FakeRender

from h0lon.config import Settings
from h0lon.synth import variants as V
from h0lon.web.app import create_app
from h0lon.web.jobs import KIND_TITLES, Job, JobManager
from h0lon.workspace import create_topic

TOPIC = "/t/kurs/metricheskie-metody"
STATIC = Path(__file__).resolve().parent.parent / "h0lon" / "web" / "static"


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
def topic(settings: Settings) -> Path:
    path = create_topic(
        settings, title="Метрические методы", course="Курс", slug="metricheskie-metody"
    )
    (path / "master.md").write_text(MASTER, encoding="utf-8")
    return path


def make_variant(
    topic: Path,
    slug: str,
    *,
    preset: str = "brief",
    prompt: str | None = None,
    template: str = "a4-notes",
    template_requested: str | None = None,
    stale: bool = False,
    pdf: bool = True,
    created: str = "2026-10-04T10:00:00Z",
) -> Path:
    """A finished variation on disk, as `run_variant` leaves it."""
    folder = topic / "variants" / slug
    folder.mkdir(parents=True)
    master_sha = hashlib.sha256((topic / "master.md").read_bytes()).hexdigest()
    (folder / "variant.md").write_text("# Кратко\n\nТекст.\n", encoding="utf-8")
    if pdf:
        (folder / "variant.pdf").write_bytes(b"%PDF-1.4 stub")
    meta = {
        "slug": slug,
        "preset": preset,
        "prompt": prompt,
        "template": template,
        "template_requested": template_requested,
        "title": V.preset_title(preset, template_requested),
        "created": created,
        "master_sha": "0" * 64 if stale else master_sha,
        "engine": "xelatex",
        "pages": 3,
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return folder


def flashes(client: TestClient, response) -> str:
    assert response.status_code == 303, response.text
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    return page.text


def block(html: str) -> str:
    """The HTML of the «Вариации» section."""
    m = re.search(r'<section class="card step" id="variants".*?</section>', html, re.S)
    assert m, "no variants section"
    return m.group(0)


# ---------------------------------------------------------------- the block


def test_block_without_a_master_says_to_build_first(client: TestClient, topic: Path) -> None:
    (topic / "master.md").unlink()
    section = block(client.get(TOPIC).text)
    assert "сначала соберите его" in section and "data-variant-form" not in section


def test_block_shows_presets_templates_and_agents(client: TestClient, topic: Path) -> None:
    section = block(client.get(TOPIC).text)
    assert 'action="/t/kurs/metricheskie-metody/variant"' in section
    assert "data-variant-form" in section and "data-variant-prompt" in section
    for title in ("Кратко", "Учебный конспект", "Шпаргалка", "По запросу", "Другой шаблон"):
        assert title in section
    assert section.count('name="preset"') == 5
    assert re.search(r'name="preset" value="brief" checked', section)
    assert "сильный уровень агента" in section and "без агента" in section
    assert '<option value="a4-compact">a4-compact — ' in section
    assert '<option value="a4-notes">a4-notes — ' in section
    assert '<option value="codex">Codex</option>' in section
    assert "Вариаций пока нет." in section
    assert "<table" not in section  # no list yet
    assert "disabled" not in section  # nothing runs


def test_list_shows_state_links_and_one_button_rebuild(client: TestClient, topic: Path) -> None:
    make_variant(topic, "brief", created="2026-10-04T10:00:00Z")
    make_variant(
        topic,
        "custom-vopros",
        preset="custom",
        prompt="Сделай <b>вопросы</b> к экзамену",
        stale=True,
        template_requested="a4-compact",
        template="a4-compact",
    )
    section = block(client.get(TOPIC).text)
    rows = section.split("<tbody>")[1].split("<tr>")[1:]
    assert len(rows) == 2
    stale_row = next(r for r in rows if "variants/custom-vopros" in r)
    fresh_row = next(r for r in rows if "variants/brief" in r)

    assert "badge-ok" in fresh_row and "актуально" in fresh_row
    assert "badge-warn" in stale_row and "устарело" in stale_row
    assert "/files/kurs/metricheskie-metody/variants/brief/variant.pdf?v=" in fresh_row
    assert "/files/kurs/metricheskie-metody/variants/brief/variant.md" in fresh_row
    assert "Открыть PDF" in fresh_row and "Мастер изменился" in section

    # the request is shown escaped; the rebuild form carries everything the job needs
    assert "&lt;b&gt;вопросы&lt;/b&gt;" in stale_row and "<b>вопросы" not in section
    assert 'name="preset" value="custom"' in stale_row
    assert 'name="template" value="a4-compact"' in stale_row
    assert "Сделай &lt;b&gt;вопросы&lt;/b&gt; к экзамену" in stale_row
    assert 'name="force"' not in stale_row and "Пересобрать" in stale_row
    assert 'name="force" value="1"' in fresh_row and "Заново" in fresh_row
    assert 'name="template" value=""' in fresh_row  # no template was asked for explicitly


def test_a_variant_without_a_pdf_is_listed_without_a_link(client: TestClient, topic: Path) -> None:
    make_variant(topic, "brief", pdf=False)
    section = block(client.get(TOPIC).text)
    assert "PDF не собран" in section and "variant.pdf" not in section
    assert "variants/brief/variant.md" in section


def test_a_changed_master_makes_the_list_stale(client: TestClient, topic: Path) -> None:
    make_variant(topic, "brief")
    assert "актуально" in block(client.get(TOPIC).text)
    (topic / "master.md").write_text(MASTER + "\n# Ещё\n", encoding="utf-8")
    section = block(client.get(TOPIC).text)
    assert "устарело" in section and "актуально" not in section


def test_variant_files_are_served_from_the_topic(client: TestClient, topic: Path) -> None:
    make_variant(topic, "brief")
    pdf = client.get("/files/kurs/metricheskie-metody/variants/brief/variant.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    md = client.get("/files/kurs/metricheskie-metody/variants/brief/variant.md")
    assert md.status_code == 200 and "# Кратко" in md.text


def test_the_form_is_disabled_while_a_job_runs(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return V.VariantResult(ok=True, slug="brief", dir=Path(topic_dir), preset="brief")

    monkeypatch.setattr("h0lon.synth.variants.run_variant", slow)
    client.post(f"{TOPIC}/variant", data={"preset": "brief"})
    try:
        section = block(client.get(TOPIC).text)
        assert re.search(r"<button[^>]*type=\"submit\"[^>]*disabled[^>]*>Собрать вариацию", section)
        again = client.post(f"{TOPIC}/variant", data={"preset": "study"})
        assert "уже выполняется задача «Вариация: Кратко»" in flashes(client, again)
        assert len(app.state.jobs.recent()) == 1
    finally:
        release.set()
        app.state.jobs.wait(app.state.jobs.recent()[0].id)


# ---------------------------------------------------------------- the job


def test_post_starts_a_variant_job_with_the_right_parameters(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(settings, topic_dir, **kwargs):
        seen.update(kwargs)
        kwargs["on_event"](V.AGENT_START)
        kwargs["on_event"]("Вариация «Кратко»: лёгкий уровень агента.")
        pdf = Path(topic_dir) / "variants" / "brief" / "variant.pdf"
        return V.VariantResult(
            ok=True,
            slug="brief",
            dir=pdf.parent,
            preset="brief",
            variant_pdf=pdf,
            template="a4-notes",
            warnings=["Строки выходят за поле"],
        )

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    response = client.post(
        f"{TOPIC}/variant",
        data={"preset": "brief", "backend": "codex", "force": "1", "template": ""},
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith(f"{TOPIC}?flash=")
    assert response.headers["location"].endswith("#job")
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)

    assert job.kind == "variant" and job.status == "done" and job.title == "Вариация: Кратко"
    assert seen["preset"] == "brief" and seen["prompt"] is None and seen["template"] is None
    assert seen["backend"] == "codex" and seen["force"] is True
    assert job.params["label"] == "Кратко"
    assert job.result["ok"] and job.result["slug"] == "brief"
    assert job.result["message"].startswith("Вариация «Кратко» собрана:")
    assert V.AGENT_START in job.events

    html = client.get(TOPIC).text
    assert 'data-job-active="0"' in html and "Вариация: Кратко" in html
    assert "Вариация «Кратко» собрана" in html and "Строки выходят за поле" in html
    assert "badge-done" in html and "готово" in html


def test_custom_and_template_requests(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(settings, topic_dir, **kwargs):
        calls.append(kwargs)
        return V.VariantResult(ok=True, slug="x", dir=Path(topic_dir), preset=kwargs["preset"])

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    client.post(f"{TOPIC}/variant", data={"preset": "custom", "prompt": "  Вопросы к экзамену  "})
    app.state.jobs.wait(app.state.jobs.recent()[0].id)
    client.post(f"{TOPIC}/variant", data={"preset": "template", "template": "a4-compact"})
    app.state.jobs.wait(app.state.jobs.recent()[0].id)
    client.post(f"{TOPIC}/variant", data={"preset": "cheatsheet", "template": "a4-notes"})
    app.state.jobs.wait(app.state.jobs.recent()[0].id)

    assert [(c["preset"], c["prompt"], c["template"], c["force"]) for c in calls] == [
        ("custom", "Вопросы к экзамену", None, False),
        ("template", None, "a4-compact", False),
        ("cheatsheet", None, "a4-notes", False),
    ]
    assert [j.title for j in reversed(app.state.jobs.recent())] == [
        "Вариация: По запросу",
        "Вариация: Другой шаблон: a4-compact",
        "Вариация: Шпаргалка",
    ]


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        ({"preset": "bogus"}, "Неизвестный пресет «bogus»"),
        ({}, "Укажите пресет"),
        ({"preset": "custom"}, "нужен запрос"),
        ({"preset": "brief", "prompt": "запрос"}, "только для пресета custom"),
        ({"preset": "template"}, "укажите шаблон"),
        ({"preset": "template", "template": "net"}, "Шаблон «net» не найден"),
        ({"preset": "brief", "template": "net"}, "Шаблон «net» не найден"),
    ],
)
def test_wrong_requests_are_messages_not_jobs(
    client: TestClient, app, topic: Path, data: dict[str, str], needle: str
) -> None:
    response = client.post(f"{TOPIC}/variant", data=data)
    assert response.headers["location"].endswith("#variants")
    assert needle in flashes(client, response)
    assert app.state.jobs.recent() == []


def test_no_master_is_a_message(client: TestClient, app, topic: Path) -> None:
    (topic / "master.md").unlink()
    response = client.post(f"{TOPIC}/variant", data={"preset": "brief"})
    assert "сначала соберите его" in flashes(client, response)
    assert app.state.jobs.recent() == []


def test_unknown_topic_is_404(client: TestClient) -> None:
    assert client.post("/t/kurs/net/variant", data={"preset": "brief"}).status_code == 404


def test_a_failed_variant_is_shown_with_its_errors(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(settings, topic_dir, **kwargs):
        return V.VariantResult(
            ok=False,
            slug="brief",
            dir=Path(topic_dir),
            preset="brief",
            errors=["Агент не вернул результат: out/variant.md: обязательный файл не создан"],
        )

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    client.post(f"{TOPIC}/variant", data={"preset": "brief"})
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.status == "failed" and "Вариация «Кратко» не собрана" in job.error
    html = client.get(TOPIC).text
    assert "badge-failed" in html and "обязательный файл не создан" in html
    assert "badge-bad" in html  # the row of the result list


def test_a_library_error_is_shown_as_it_is(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(settings, topic_dir, **kwargs):
        raise ValueError("Шаблон «x» не найден. Доступны: a4-notes.")

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    client.post(f"{TOPIC}/variant", data={"preset": "brief"})
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.status == "failed" and job.error == "Шаблон «x» не найден. Доступны: a4-notes."
    assert "Шаблон «x» не найден" in client.get(TOPIC).text


def test_cancel_stops_a_variant_job_before_the_next_phase(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_agent = threading.Event()
    go_on = threading.Event()
    rendered = threading.Event()

    def fake(settings, topic_dir, **kwargs):
        kwargs["on_event"](V.AGENT_START)
        in_agent.set()
        go_on.wait(10)
        kwargs["on_event"](V.RENDER_START)  # raises here: the stop request is honoured
        rendered.set()
        return V.VariantResult(ok=True, slug="brief", dir=Path(topic_dir), preset="brief")

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    client.post(f"{TOPIC}/variant", data={"preset": "brief"})
    job = app.state.jobs.recent()[0]
    assert in_agent.wait(5)
    assert client.post(f"/jobs/{job.id}/cancel").status_code == 303
    go_on.set()
    app.state.jobs.wait(job.id)
    assert job.status == "stopped" and not rendered.is_set()
    assert job.result["reason"] == "cancelled" and "сборка PDF" in job.result["message"]


# ---------------------------------------------------------------- end to end with fakes


def test_variant_job_with_the_real_library_and_fake_agent_and_render(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, render = FakeAgent(), FakeRender()
    monkeypatch.setattr(V, "run_agent", agent)
    monkeypatch.setattr(V, "render_document", render)

    client.post(f"{TOPIC}/variant", data={"preset": "cheatsheet"})
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.status == "done", (job.error, job.events)
    assert agent.calls[0].tier == "light" and render.calls[0].template == "a4-compact"
    assert (topic / "variants" / "cheatsheet" / "variant.pdf").is_file()

    section = block(client.get(TOPIC).text)
    assert "Шпаргалка" in section and "a4-compact" in section and "актуально" in section
    assert "/variants/cheatsheet/variant.pdf?v=" in section

    # the same request again: nothing runs, the job says so
    client.post(f"{TOPIC}/variant", data={"preset": "cheatsheet"})
    again = app.state.jobs.recent()[0]
    app.state.jobs.wait(again.id)
    assert again.status == "done" and again.result["cached"] is True
    assert "из кэша" in again.result["message"] and len(agent.calls) == 1
    assert "из кэша" in client.get(TOPIC).text

    # the master is rebuilt: the variation is stale and one button builds it again
    (topic / "master.md").write_text(MASTER + "\n# Новая глава\n", encoding="utf-8")
    assert "устарело" in block(client.get(TOPIC).text)
    client.post(f"{TOPIC}/variant", data={"preset": "cheatsheet", "prompt": "", "template": ""})
    last = app.state.jobs.recent()[0]
    app.state.jobs.wait(last.id)
    assert last.status == "done" and not last.result["cached"] and len(agent.calls) == 2
    assert "актуально" in block(client.get(TOPIC).text)


# ---------------------------------------------------------------- jobs unit and static files


def test_job_manager_knows_the_variant_kind(
    settings: Settings, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert KIND_TITLES["variant"] == "Вариация"

    def fake(settings, topic_dir, **kwargs):
        kwargs["on_event"]("запрос к агенту")
        return V.VariantResult(ok=True, slug="brief", dir=Path(topic_dir), preset="brief")

    monkeypatch.setattr("h0lon.synth.variants.run_variant", fake)
    jobs = JobManager(settings)
    job = jobs.start("kurs/metricheskie-metody", topic, "variant", preset="brief", label="Кратко")
    jobs.wait(job.id)
    assert isinstance(job, Job) and job.title == "Вариация: Кратко" and job.status == "done"
    assert "запрос к агенту" in job.events and job.events[-1].startswith("Вариация «Кратко»")
    plain = jobs.start("kurs/metricheskie-metody", topic, "variant", preset="brief")
    jobs.wait(plain.id)
    assert plain.title == "Вариация"
    with pytest.raises(ValueError):
        jobs.start("kurs/metricheskie-metody", topic, "bogus")


def test_static_files_know_the_block(client: TestClient) -> None:
    js = client.get("/static/app.js").text
    assert "initVariants" in js and "data-variant-form" in js and "data-variant-prompt" in js
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert ".variant-form" in css and ".variant-presets" in css
