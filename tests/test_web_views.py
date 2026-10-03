"""Web interface: pages, forms, jobs over HTTP (incl. SSE), files, source review, doctor.

Agents and the network are never used: `extract_topic` / `build_topic` are replaced by fakes,
links go to a local http.server or to a fake `add_sources`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pymupdf
import pytest
from fastapi.testclient import TestClient

from h0lon import tools
from h0lon.config import Settings
from h0lon.doctor import Check
from h0lon.extract.model import ExtractPlan, ExtractResult
from h0lon.sources.ingest import IngestReport
from h0lon.sources.models import SourceRecord
from h0lon.synth.model import BuildResult, StageResult
from h0lon.web import render_md
from h0lon.web.app import create_app
from h0lon.workspace import create_topic, load_topic, save_topic

SAMPLE_PNG = Path(__file__).parent / "fixtures" / "img" / "sample.png"
TOPIC = "/t/kurs/metricheskie-metody"


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


def make_pdf(path: Path, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        text = f"Metric spaces, lecture page {n + 1}. " + "The distance is symmetric. " * 20
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path.read_bytes()


SOURCE_MD = """---
id: S1
kind: slides
title: Лекция
---

## [[S1:s1]] Слайд 1. Метрики

<!-- S1.b001 paragraph -->

Функция $\\rho(x,y)$ — метрика.

## [[S1:s2]] Слайд 2. Свойства

<!-- S1.b002 list -->

- симметрия
- неравенство треугольника

## [[S1:s3]] Слайд 3. Формула

<!-- S1.b003 formula -->

$$\\rho(a,b)=\\left(\\sum_i |a_i-b_i|^p\\right)^{1/p}$$
"""


def set_sources(path: Path, records: list[dict[str, Any]]) -> None:
    meta = load_topic(path)
    meta.sources = records
    save_topic(path, meta)


def record(sid: str, kind: str, title: str, **extra: Any) -> dict[str, Any]:
    base = {"id": sid, "kind": kind, "title": title, "added": "2026-10-04T10:00:00Z"}
    return SourceRecord.model_validate({**base, **extra}).model_dump(mode="json")


@pytest.fixture
def topic(settings: Settings) -> Path:
    """A synthetic extracted topic: S1 (slides, PDF + page images, notes) and D1 (Markdown)."""
    path = create_topic(
        settings, title="Метрические методы", course="Курс", slug="metricheskie-metody"
    )
    assert path == settings.general.workspaces_dir / "kurs" / "metricheskie-metody"
    make_pdf(path / "sources" / "S1_lektsiya.pdf", pages=3)
    (path / "sources" / "D1_zametki.md").write_text("# Заметки\n\nТекст.\n", encoding="utf-8")
    out = path / "extracted" / "S1"
    (out / "pages").mkdir(parents=True)
    (out / "source.md").write_text(SOURCE_MD, encoding="utf-8")
    for n in (1, 3):
        shutil.copy(SAMPLE_PNG, out / "pages" / f"p{n:04d}.png")
    (out / "pages" / "p0001.md").write_text("страница", encoding="utf-8")
    (path / "extracted" / "D1").mkdir()
    (path / "extracted" / "D1" / "source.md").write_text(
        "## [[D1:§1]] Заметки\n\nТекст.\n", encoding="utf-8"
    )
    set_sources(
        path,
        [
            record(
                "S1",
                "slides",
                "Лекция 1. Метрики",
                file="sources/S1_lektsiya.pdf",
                original_name="Лекция 1.pdf",
                units={"pages": 3, "slides": 3},
                status="extracted",
                extracted_key="key-s1",
                quality={
                    "text_layer": 0.97,
                    "chars_per_page": 212,
                    "pages_math": 2,
                    "pages_vision": 2,
                    "cyrillic_ratio": 0.9,
                    "notes": ["Формулы перенесены агентом — проверьте индексы"],
                },
            ),
            record(
                "D1",
                "md",
                "Заметки",
                file="sources/D1_zametki.md",
                status="extracted",
                extracted_key="key-d1",
            ),
        ],
    )
    return path


def flashes(client: TestClient, response) -> str:
    """Follow the redirect of a POST and return the text of the page that shows its message."""
    assert response.status_code == 303, response.text
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    return page.text


def sse_events(body: str) -> list[tuple[str, str]]:
    """[(event, data)] of an SSE body ("message" for lines without an event name)."""
    events: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        name, data = "message", []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if data:
            events.append((name, "\n".join(data)))
    return events


# ---------------------------------------------------------------- index and new topic


def test_index_empty_workspace(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Тем пока нет" in page.text
    assert "Новая тема" in page.text
    assert 'action="/topics"' in page.text


def test_index_lists_courses_and_topics(
    client: TestClient, topic: Path, settings: Settings
) -> None:
    other = create_topic(settings, title="Вероятность", course="Теорвер")
    make_pdf(topic / "master.pdf")
    page = client.get("/").text
    assert "Курс" in page and "Метрические методы" in page
    assert "Теорвер" in page and "Вероятность" in page
    assert 'href="/t/kurs/metricheskie-metody"' in page
    assert "Источников</span> 2" in page
    assert "извлечено" in page
    assert "master.pdf" in page and "Последняя сборка" in page
    assert "не собрано" in page  # the other topic
    assert other.exists()


def test_index_survives_a_broken_topic_yaml(
    client: TestClient, topic: Path, settings: Settings
) -> None:
    broken = settings.general.workspaces_dir / "kurs" / "slomano"
    broken.mkdir(parents=True)
    (broken / "topic.yaml").write_text("title: [незакрыто\n", encoding="utf-8")
    page = client.get("/")
    assert page.status_code == 200
    assert "Не удалось прочитать topic.yaml" in page.text
    assert "Метрические методы" in page.text  # the healthy topic is still listed


def test_index_shows_running_job(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    assert client.post(f"{TOPIC}/build").status_code == 303
    try:
        assert "Сборка мастер-конспекта…" in client.get("/").text
    finally:
        release.set()


def test_create_topic_redirects_to_the_new_topic(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/topics", data={"title": "Теорвер — лекция 3", "course": "Теория вероятностей"}
    )
    assert response.status_code == 303
    path = settings.general.workspaces_dir / "teoriya-veroyatnostey" / "teorver-lektsiya-3"
    assert (path / "topic.yaml").is_file()
    assert response.headers["location"].startswith(
        "/t/teoriya-veroyatnostey/teorver-lektsiya-3?flash="
    )
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    assert "Тема «Теорвер — лекция 3» создана." in page.text
    meta = load_topic(path)
    assert (meta.title, meta.course) == ("Теорвер — лекция 3", "Теория вероятностей")
    # the message is shown once
    assert "создана." not in client.get(response.headers["location"].split("?")[0]).text


def test_create_topic_with_explicit_slug(client: TestClient, settings: Settings) -> None:
    client.post("/topics", data={"title": "Лекция", "course": "Алгебра", "slug": "lecture-1"})
    assert (settings.general.workspaces_dir / "algebra" / "lecture-1" / "topic.yaml").is_file()


def test_create_topic_errors_are_messages(client: TestClient, topic: Path) -> None:
    empty = client.post("/topics", data={"title": "  ", "course": "Курс"})
    assert "Название темы не может быть пустым" in flashes(client, empty)
    no_course = client.post("/topics", data={"title": "Тема", "course": ""})
    assert "Курс не может быть пустым" in flashes(client, no_course)
    duplicate = client.post(
        "/topics",
        data={"title": "Метрические методы", "course": "Курс", "slug": "metricheskie-metody"},
    )
    text = flashes(client, duplicate)
    assert "Тема уже существует" in text
    assert "Traceback" not in text


def test_user_text_is_escaped(client: TestClient, settings: Settings) -> None:
    client.post(
        "/topics", data={"title": "<b>жирно</b><script>1</script>", "course": "<i>курс</i>"}
    )
    page = client.get("/").text
    assert "<script>1</script>" not in page
    assert "<b>жирно</b>" not in page
    assert "&lt;b&gt;жирно&lt;/b&gt;" in page


# ---------------------------------------------------------------- topic page


def test_topic_page_shows_sources_gate_stages(client: TestClient, topic: Path) -> None:
    page = client.get(TOPIC)
    assert page.status_code == 200
    html = page.text
    assert "Метрические методы" in html
    for needle in (
        "Лекция 1. Метрики",
        "Лекция 1.pdf",  # original name under the title
        "3 сл.",
        "извлечён",
        "Текстовый слой</span> 97 %",
        "Замечаний</span> 1",
        'href="/t/kurs/metricheskie-metody/source/S1"',
        'href="/files/kurs/metricheskie-metody/sources/S1_lektsiya.pdf"',
        "Review gate включён",
        "Извлечение ещё не одобрено",
        "Что проверить в первую очередь",
        "Структура темы (S1)",
        "Не выполнялась".lower(),
    ):
        assert needle in html, needle
    assert 'action="/t/kurs/metricheskie-metody/sources"' in html
    assert 'enctype="multipart/form-data"' in html
    assert 'name="links"' in html
    assert "Показать план" in html and "Собрать" in html and "Одобрить извлечение" in html
    assert "<iframe" not in html  # no master.pdf yet
    assert "data-job-id" not in html


def make_master_pdf(path: Path, headings: dict[int, str], *, bookmarks: bool) -> None:
    """A 5-page PDF whose pages carry the given headings (as text, and as bookmarks if asked)."""
    doc = pymupdf.open()
    for number in range(1, 6):
        page = doc.new_page()
        text = headings.get(number, f"Страница {number}")
        page.insert_htmlbox(pymupdf.Rect(50, 50, 550, 300), f"<h1>{text}</h1>")
    if bookmarks:
        doc.set_toc([[1, title, number] for number, title in headings.items()])
    doc.save(path)
    doc.close()


MASTER_MD_WITH_APPENDICES = (
    "# Введение\n\n# Расхождения между источниками {#app:conflicts .appendix}\n\n"
    "# Карта покрытия {#app:coverage}\n"
)


def test_topic_page_embeds_master_pdf_and_appendices(client: TestClient, topic: Path) -> None:
    make_master_pdf(
        topic / "master.pdf",
        {1: "Введение", 3: "Расхождения между источниками", 5: "Карта покрытия"},
        bookmarks=True,
    )
    (topic / "master.md").write_text(MASTER_MD_WITH_APPENDICES, encoding="utf-8")
    html = client.get(TOPIC).text
    assert '<iframe class="pdf-frame" src="/files/kurs/metricheskie-metody/master.pdf?v=' in html
    assert 'href="/files/kurs/metricheskie-metody/master.md"' in html
    assert re.search(
        r'href="[^"]*master\.pdf\?v=\d+#page=3"[^>]*>Расхождения между источниками', html
    )
    assert re.search(r'href="[^"]*master\.pdf\?v=\d+#page=5"[^>]*>Карта покрытия', html)
    assert "Журнал правок" not in html and "Редакторские дополнения" not in html


def test_appendix_pages_fall_back_to_the_text_of_pdf_without_bookmarks(
    client: TestClient, topic: Path
) -> None:
    make_master_pdf(
        topic / "master.pdf",
        {2: "Расхождения между источниками", 4: "Карта покрытия"},
        bookmarks=False,
    )
    (topic / "master.md").write_text(MASTER_MD_WITH_APPENDICES, encoding="utf-8")
    html = client.get(TOPIC).text
    assert "#page=2" in html and "#page=4" in html


def test_appendix_without_a_page_gets_no_link(client: TestClient, topic: Path) -> None:
    make_pdf(topic / "master.pdf")  # no such headings anywhere
    (topic / "master.md").write_text(MASTER_MD_WITH_APPENDICES, encoding="utf-8")
    html = client.get(TOPIC).text
    assert "Приложения в PDF" not in html
    assert '<iframe class="pdf-frame"' in html


def test_topic_page_for_unknown_topic_is_a_friendly_404(client: TestClient, topic: Path) -> None:
    for url in ("/t/kurs/net-takoy", "/t/net/net", "/t/..%2Fkurs/metricheskie-metody"):
        page = client.get(url)
        assert page.status_code == 404, url
        assert "не найдена" in page.text  # «Тема не найдена» or «Страница не найдена»
        assert "Traceback" not in page.text


def test_topic_page_with_unreadable_topic_yaml(client: TestClient, topic: Path) -> None:
    (topic / "topic.yaml").write_text("title: [незакрыто\n", encoding="utf-8")
    page = client.get(TOPIC)
    assert page.status_code == 422
    assert "Не удалось прочитать тему" in page.text
    assert "Traceback" not in page.text


# ---------------------------------------------------------------- adding sources


def test_upload_pdf_is_added_through_add_sources(
    client: TestClient, topic: Path, tmp_path: Path
) -> None:
    pdf = make_pdf(tmp_path / "лекция.pdf", pages=2)
    response = client.post(
        f"{TOPIC}/sources",
        files=[("files", ("Новая лекция.pdf", pdf, "application/pdf"))],
        data={"links": "", "kind": ""},
    )
    text = flashes(client, response)
    assert "Добавлено источников: 1" in text
    sources = load_topic(topic).sources
    added = sources[-1]
    assert added["id"] == "P1" and added["kind"] == "pdf-text"
    assert added["original_name"] == "Новая лекция.pdf"
    assert added["title"] == "Новая лекция"
    assert (topic / added["file"]).is_file()
    assert added["units"] == {"pages": 2}
    # the same file again is a warning, not an error and not a second record
    again = client.post(
        f"{TOPIC}/sources", files=[("files", ("Новая лекция.pdf", pdf, "application/pdf"))]
    )
    assert "уже добавлен как P1" in flashes(client, again)
    assert len(load_topic(topic).sources) == len(sources)


def test_upload_name_cannot_leave_the_temporary_directory(
    client: TestClient, topic: Path, tmp_path: Path
) -> None:
    pdf = make_pdf(tmp_path / "x.pdf")
    response = client.post(
        f"{TOPIC}/sources",
        files=[("files", ("..\\..\\..\\evil: name?.pdf", pdf, "application/pdf"))],
    )
    assert response.status_code == 303
    added = load_topic(topic).sources[-1]
    assert added["original_name"] == "evil_ name_.pdf"
    assert not list(tmp_path.glob("evil*"))
    assert not list(topic.parent.glob("evil*")) and not list(topic.glob("evil*"))


def test_upload_unsupported_file_is_an_error_message(client: TestClient, topic: Path) -> None:
    response = client.post(
        f"{TOPIC}/sources",
        files=[("files", ("архив.rar", b"Rar!\x1a\x07\x00 data", "application/x-rar"))],
    )
    text = flashes(client, response)
    assert "flash-error" in text
    assert "архив.rar" in text
    assert len(load_topic(topic).sources) == 2


def test_nothing_chosen_is_a_hint(client: TestClient, topic: Path) -> None:
    response = client.post(f"{TOPIC}/sources", data={"links": "  \n\n ", "kind": ""})
    assert "Выберите файлы или вставьте ссылки" in flashes(client, response)


def test_links_go_to_add_sources_one_per_line(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_add(settings, topic_dir, items, *, kind=None, title=None):
        calls.append((list(items), kind))
        assert Path(topic_dir) == topic
        return IngestReport(
            added=[
                SourceRecord(id=f"W{i}", kind="web", title=u, added="x")
                for i, u in enumerate(items, 1)
            ],
            warnings=["что-то странное"],
        )

    monkeypatch.setattr("h0lon.web.views.add_sources", fake_add)
    response = client.post(
        f"{TOPIC}/sources",
        data={"links": "https://a.example/x\n\n  https://b.example/y  \n", "kind": "web"},
    )
    text = flashes(client, response)
    assert calls == [(["https://a.example/x", "https://b.example/y"], "web")]
    assert "Добавлено источников: 2" in text
    assert "что-то странное" in text and "flash-warn" in text


def test_kind_for_files_is_not_forced_on_links(
    client: TestClient, topic: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_add(settings, topic_dir, items, *, kind=None, title=None):
        calls.append(([Path(i).name if "://" not in i else i for i in items], kind))
        return IngestReport()

    monkeypatch.setattr("h0lon.web.views.add_sources", fake_add)
    client.post(
        f"{TOPIC}/sources",
        files=[("files", ("фото.jpg", b"\xff\xd8\xff data", "image/jpeg"))],
        data={"links": "https://a.example/x", "kind": "handwritten"},
    )
    assert calls == [(["фото.jpg"], "handwritten"), (["https://a.example/x"], None)]


def test_unknown_kind_is_refused(client: TestClient, topic: Path) -> None:
    response = client.post(f"{TOPIC}/sources", data={"links": "https://a.example", "kind": "exe"})
    assert "Неизвестный вид источника" in flashes(client, response)


class _PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = (
            "<html><head><title>Метод ближайших соседей</title></head><body><article>"
            + "<p>Классификация объектов по ближайшим соседям. </p>" * 30
            + "</article></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def test_link_to_a_local_http_server_is_downloaded(client: TestClient, topic: Path) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/knn.html"
        response = client.post(f"{TOPIC}/sources", data={"links": url, "kind": ""})
        text = flashes(client, response)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert "Добавлено источников: 1" in text
    added = load_topic(topic).sources[-1]
    assert added["kind"] == "web" and added["id"] == "W1" and added["url"] == url
    assert (topic / added["file"]).is_file()


def test_sources_cannot_be_added_while_a_job_runs(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    client.post(f"{TOPIC}/build")
    try:
        response = client.post(f"{TOPIC}/sources", data={"links": "https://a.example/x"})
        assert "после задачи «Сборка мастер-конспекта»" in flashes(client, response)
        page = client.get(TOPIC).text
        assert "Пока идёт задача, источники добавить нельзя" in page
    finally:
        release.set()


# ---------------------------------------------------------------- review gate


def test_review_gate_switch_changes_topic_yaml(client: TestClient, topic: Path) -> None:
    before = load_topic(topic)
    assert before.review_gate is None  # inherits general.review_gate (on)
    page = client.get(TOPIC).text
    assert 'aria-checked="true"' in page and "Review gate включён" in page
    assert 'name="enabled" value="0"' in page  # the button turns it off
    assert "берётся из настроек" in page

    off = client.post(f"{TOPIC}/review-gate", data={"enabled": "0"})
    text = flashes(client, off)
    assert load_topic(topic).review_gate is False
    assert "Review gate выключен: сборка не будет ждать" in text
    page = client.get(TOPIC).text
    assert 'aria-checked="false"' in page and 'name="enabled" value="1"' in page
    assert "берётся из настроек" not in page

    client.post(f"{TOPIC}/review-gate", data={"enabled": "1"})
    assert load_topic(topic).review_gate is True
    after = load_topic(topic)
    assert [s["id"] for s in after.sources] == ["S1", "D1"]  # nothing else is lost
    assert after.title == before.title and after.created == before.created


def test_review_gate_keeps_unknown_keys(client: TestClient, topic: Path) -> None:
    meta = load_topic(topic)
    meta.review = {"approved_at": "2026-10-04T10:00:00Z", "extracted_keys": {"S1": "key-s1"}}  # type: ignore[attr-defined]
    save_topic(topic, meta)
    client.post(f"{TOPIC}/review-gate", data={"enabled": "0"})
    review = (load_topic(topic).model_extra or {}).get("review")
    assert review["extracted_keys"] == {"S1": "key-s1"}


def test_gate_setting_from_config_is_shown(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(general={"git_per_topic": False, "review_gate": False})
    create_topic(settings, title="Тема", course="Курс", slug="tema")
    with TestClient(create_app(settings), follow_redirects=False) as c:
        page = c.get("/t/kurs/tema").text
    assert "Review gate выключен" in page and 'aria-checked="false"' in page


# ---------------------------------------------------------------- plan


def test_plan_shows_pages_and_agent_runs(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_extract(settings, topic_dir, **kwargs):
        seen.update(kwargs)
        return [
            ExtractPlan(
                "S1", pages_total=31, pages_vision=21, agent_runs=5, notes=["аннотация: 1 прогон"]
            ),
            ExtractPlan("D1", pages_total=0, pages_vision=0, agent_runs=0, notes=["кэш актуален"]),
        ]

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract)
    response = client.post(f"{TOPIC}/plan", data={"force": "1", "backend": "codex"})
    assert response.status_code == 200
    assert seen["dry_run"] is True and seen["force"] is True
    assert seen["use_vision"] is True and seen["backend"] == "codex"
    html = response.text
    assert "План извлечения" in html
    assert "аннотация: 1 прогон" in html and "кэш актуален" in html
    assert "Прогонов агента: 5; параллельно до 2" in html  # default parallel_runs
    assert "9–18 мин" in html  # ceil(5 / 2) = 3 waves of 3-6 minutes
    # plan with --no-vision
    client.post(f"{TOPIC}/plan", data={"no_vision": "1"})
    assert seen["use_vision"] is False


def test_plan_error_is_shown_on_the_page(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any):
        raise ValueError("Нечего планировать")

    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", refuse)
    response = client.post(f"{TOPIC}/plan")
    assert response.status_code == 200
    assert "Нечего планировать" in response.text
    assert "План извлечения" not in response.text


# ---------------------------------------------------------------- jobs over HTTP


def fake_extract_ok(settings, topic_dir, **kwargs):
    kwargs["on_event"]("S1: читаю страницы")
    return [
        ExtractResult(ok=True, source_id=sid, source_md=Path("x/source.md"), blocks=5)
        for sid in kwargs["source_ids"]
    ]


def test_extract_job_and_its_page(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", fake_extract_ok)
    response = client.post(f"{TOPIC}/extract", data={"no_vision": "1", "backend": "claude"})
    assert response.status_code == 303
    assert response.headers["location"].startswith(f"{TOPIC}?flash=")
    assert response.headers["location"].endswith("#job")
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.kind == "extract" and job.status == "done"
    assert job.params["use_vision"] is False and job.params["backend"] == "claude"

    data = client.get(f"/jobs/{job.id}").json()
    assert data["status"] == "done" and data["kind"] == "extract"
    assert data["topic"] == "kurs/metricheskie-metody"
    assert "S1: читаю страницы" in data["events"]
    assert data["result"]["results"][0]["source_id"] == "S1"

    html = client.get(TOPIC).text
    assert f'data-job-id="{job.id}"' in html and 'data-job-active="0"' in html
    assert "S1: читаю страницы" in html  # the log is rendered into the page
    assert "Извлечение завершено: источников 2." in html
    assert "badge-done" in html


def test_extract_with_unknown_topic_is_404(client: TestClient) -> None:
    assert client.post("/t/kurs/net/extract").status_code == 404


def test_build_job_passes_options(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_build(settings, topic_dir, **kwargs):
        seen.update(kwargs)
        kwargs["on_event"]("Структура темы (S1)…")
        return BuildResult(
            ok=True,
            topic_dir=Path(topic_dir),
            stages=[
                StageResult(stage="extract", ok=True, cached=True),
                StageResult(stage="outline", ok=True, warnings=["тонкое место"]),
            ],
            message="Готово: master.pdf. Покрытие блоков источников: 99,7 %.",
        )

    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build)
    response = client.post(
        f"{TOPIC}/build",
        data={"from_stage": "sections", "force": "1", "no_review": "1", "backend": "codex"},
    )
    assert response.status_code == 303
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert (seen["review"], seen["from_stage"], seen["force"], seen["backend"]) == (
        False,
        "sections",
        True,
        "codex",
    )
    html = client.get(TOPIC).text
    assert "Готово: master.pdf. Покрытие блоков источников: 99,7 %." in html
    assert "тонкое место" in html
    assert "из кэша" in html


def test_build_defaults_follow_the_review_gate(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_build(settings, topic_dir, **kwargs):
        seen.update(kwargs)
        return BuildResult(
            ok=False, topic_dir=Path(topic_dir), stopped_at="review", message="Нужно одобрение"
        )

    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build)
    client.post(f"{TOPIC}/build")
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert seen["review"] is True and seen["from_stage"] is None and seen["force"] is False
    assert job.status == "stopped" and job.result["reason"] == "review"
    html = client.get(TOPIC).text
    assert "badge-stopped" in html and "Нужно одобрение" in html


def test_unknown_stage_is_refused_without_a_job(client: TestClient, app, topic: Path) -> None:
    response = client.post(f"{TOPIC}/build", data={"from_stage": "bogus"})
    assert "Неизвестная стадия: bogus" in flashes(client, response)
    assert app.state.jobs.recent() == []


def test_second_job_for_the_same_topic_is_refused(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow(settings, topic_dir, **kwargs):
        release.wait(10)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="ok")

    monkeypatch.setattr("h0lon.synth.build.build_topic", slow)
    client.post(f"{TOPIC}/build")
    try:
        response = client.post(f"{TOPIC}/extract")
        assert "уже выполняется задача «Сборка мастер-конспекта»" in flashes(client, response)
        assert len(app.state.jobs.recent()) == 1
        page = client.get(TOPIC).text
        assert 'data-job-active="1"' in page
        assert re.search(r"<button[^>]*disabled[^>]*>Собрать</button>", page)
        assert "Остановить" in page
    finally:
        release.set()


def test_cancel_over_http(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_stage = threading.Event()
    go_on = threading.Event()

    def staged(settings, topic_dir, **kwargs):
        kwargs["on_event"]("Извлечение источников…")
        in_stage.set()
        go_on.wait(10)
        kwargs["on_event"]("Структура темы (S1)…")
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="не дойдём")

    monkeypatch.setattr("h0lon.synth.build.build_topic", staged)
    client.post(f"{TOPIC}/build")
    job = app.state.jobs.recent()[0]
    assert in_stage.wait(5)
    response = client.post(f"/jobs/{job.id}/cancel")
    assert response.status_code == 303
    assert response.headers["location"].startswith(f"{TOPIC}?flash=")
    assert "Остановка запрошена" in client.get(response.headers["location"]).text
    go_on.set()
    app.state.jobs.wait(job.id)
    assert job.status == "stopped"
    again = client.post(f"/jobs/{job.id}/cancel")
    assert "Задача уже завершена" in flashes(client, again)
    assert client.post("/jobs/unknown/cancel").status_code == 404


def test_job_json_unknown_is_json_404(client: TestClient) -> None:
    for url in ("/jobs/nope", "/jobs/nope/events"):
        response = client.get(url)
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert "error" in response.json()


def test_events_stream_of_a_finished_job(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_build(settings, topic_dir, **kwargs):
        for line in ("Первая стадия…", "Вторая стадия…", "Итог"):
            kwargs["on_event"](line)
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="Готово")

    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build)
    client.post(f"{TOPIC}/build")
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)

    response = client.get(f"/jobs/{job.id}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in response.headers["cache-control"]
    events = sse_events(response.text)
    lines = [data for name, data in events if name == "message"]
    assert lines[0].startswith("Задача «Сборка мастер-конспекта» запущена")
    assert lines[1:4] == ["Первая стадия…", "Вторая стадия…", "Итог"]
    assert lines[-1] == "Готово"
    assert events[-1][0] == "done"
    done = json.loads(events[-1][1])
    assert done["status"] == "done" and done["label"] == "готово" and done["total"] == len(lines)
    assert any(name == "status" for name, _ in events)
    assert "id: 1\n" in response.text  # ids are absolute indexes: the next `from`

    resumed = sse_events(client.get(f"/jobs/{job.id}/events?from=3").text)
    assert [d for n, d in resumed if n == "message"] == lines[3:]
    by_header = sse_events(
        client.get(f"/jobs/{job.id}/events", headers={"Last-Event-ID": "4"}).text
    )
    assert [d for n, d in by_header if n == "message"] == lines[4:]
    assert client.get(f"/jobs/{job.id}/events?from=-1").status_code == 422


def test_events_stream_follows_a_running_job(
    client: TestClient, app, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got_a = threading.Event()
    release = threading.Event()

    def fake_build(settings, topic_dir, **kwargs):
        kwargs["on_event"]("A: началось")
        got_a.set()
        release.wait(10)
        kwargs["on_event"]("B: закончилось")
        return BuildResult(ok=True, topic_dir=Path(topic_dir), message="Готово")

    monkeypatch.setattr("h0lon.synth.build.build_topic", fake_build)
    client.post(f"{TOPIC}/build")
    job = app.state.jobs.recent()[0]
    assert got_a.wait(5)
    seen: list[str] = []
    with client.stream("GET", f"/jobs/{job.id}/events") as stream:
        for line in stream.iter_lines():
            seen.append(line)
            if line == "data: A: началось":
                release.set()  # the stream must go on and deliver B and the end
            if line == "event: done":
                break
    assert "data: B: закончилось" in seen
    assert seen.index("data: A: началось") < seen.index("data: B: закончилось")


def test_approve_job_over_http(client: TestClient, app, topic: Path) -> None:
    response = client.post(f"{TOPIC}/approve")
    assert response.status_code == 303
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.status == "done", job.error
    assert job.kind == "approve"
    review = (load_topic(topic).model_extra or {}).get("review")
    assert review["extracted_keys"] == {"S1": "key-s1", "D1": "key-d1"}
    html = client.get(TOPIC).text
    assert "Извлечение одобрено: S1, D1" in html
    assert "Извлечение одобрено (" in html  # the gate block


def test_approve_refuses_unextracted_sources_with_a_message(
    client: TestClient, app, topic: Path
) -> None:
    meta = load_topic(topic)
    meta.sources[1]["status"] = "added"
    save_topic(topic, meta)
    page = client.get(TOPIC).text
    assert "Не все источники извлечены: D1." in page
    assert re.search(r"<button[^>]*disabled[^>]*>Одобрить извлечение</button>", page)
    client.post(f"{TOPIC}/approve")  # a script can still try: the job explains
    job = app.state.jobs.recent()[0]
    app.state.jobs.wait(job.id)
    assert job.status == "failed"
    assert "Не все источники извлечены (D1)" in client.get(TOPIC).text


# ---------------------------------------------------------------- source review


def test_source_review_page(client: TestClient, topic: Path) -> None:
    page = client.get(f"{TOPIC}/source/S1")
    assert page.status_code == 200
    html = page.text
    # quality signals and notes on top
    assert "Текстовый слой</span> 97 %" in html
    assert "Формулы перенесены агентом — проверьте индексы" in html
    assert "Что стоит проверить" in html
    # page images on the left, from the protected files route
    assert 'id="pg-1"' in html and 'id="pg-3"' in html and 'id="pg-2"' not in html
    assert 'src="/files/kurs/metricheskie-metody/extracted/S1/pages/p0001.png"' in html
    assert 'href="/files/kurs/metricheskie-metody/sources/S1_lektsiya.pdf"' in html
    # Source Doc on the right: MathML, location headings, block ids
    assert "<math" in html and "application/x-tex" in html
    assert 'data-loc="S1:s2" data-page="2"' in html
    assert 'class="blk" id="S1.b001"' in html
    assert "<script" not in html.replace('<script src="/static/app.js', "")
    # approve button of the topic, navigation
    assert 'action="/t/kurs/metricheskie-metody/source/S1/approve"' not in html
    assert 'action="/t/kurs/metricheskie-metody/approve"' in html
    assert "Одобрить извлечение темы" in html
    assert 'href="/t/kurs/metricheskie-metody/source/D1"' in html  # next source


def test_source_review_without_page_images_shows_the_original_pdf(
    client: TestClient, topic: Path
) -> None:
    shutil.rmtree(topic / "extracted" / "S1" / "pages")
    html = client.get(f"{TOPIC}/source/S1").text
    assert 'class="pageimg"' not in html
    assert (
        '<iframe class="orig-frame" src="/files/kurs/metricheskie-metody/sources/S1_lektsiya.pdf"'
        in html
    )


def test_source_review_of_a_markdown_source_has_no_page_pane_images(
    client: TestClient, topic: Path
) -> None:
    html = client.get(f"{TOPIC}/source/D1").text
    assert "Для этого вида источника нет картинок страниц" in html
    assert 'data-loc="D1:§1"' in html
    assert "Замечаний к качеству извлечения нет" in html


def test_source_review_not_extracted_yet(client: TestClient, topic: Path) -> None:
    meta = load_topic(topic)
    meta.sources[1]["status"] = "added"
    save_topic(topic, meta)
    shutil.rmtree(topic / "extracted" / "D1")
    page = client.get(f"{TOPIC}/source/D1")
    assert page.status_code == 200
    assert "Source Doc ещё нет" in page.text
    assert "Не все источники извлечены: D1." in page.text  # the approve button explains


def test_source_review_unknown_and_malicious_ids(client: TestClient, topic: Path) -> None:
    for sid in ("S9", "s1", "S1x", "%2e%2e", "S1%2f..%2f..", "S1%5C", "S1%00"):
        page = client.get(f"{TOPIC}/source/{sid}")
        assert page.status_code == 404, sid
        assert "Traceback" not in page.text


def test_source_review_pandoc_failure_falls_back_to_plain_text(
    client: TestClient, topic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any):
        raise render_md.SourceDocError("Не найден Pandoc: поставьте его")

    monkeypatch.setattr("h0lon.web.views.source_doc_html", broken)
    page = client.get(f"{TOPIC}/source/S1")
    assert page.status_code == 200
    assert "Не найден Pandoc: поставьте его" in page.text
    assert '<pre class="rawdoc">' in page.text and "симметрия" in page.text


@pytest.mark.skipif(tools.find_pandoc("") is None, reason="Pandoc не найден")
def test_source_doc_content_cannot_script_the_page(client: TestClient, topic: Path) -> None:
    (topic / "extracted" / "D1" / "source.md").write_text(
        '## [[D1:§1]] Заметки\n\n<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n'
        "[x](javascript:alert(3))\n",
        encoding="utf-8",
    )
    html = client.get(f"{TOPIC}/source/D1").text
    assert "alert(1)" not in html and "onerror" not in html and "javascript:" not in html.lower()


# ---------------------------------------------------------------- files


@pytest.fixture
def secret(topic: Path) -> Path:
    path = topic.parents[2] / "secret.txt"  # next to the workspaces dir
    path.write_text("TOP-SECRET", encoding="utf-8")
    return path


def test_files_serve_pdf_images_and_text(client: TestClient, topic: Path) -> None:
    make_pdf(topic / "master.pdf")
    (topic / "master.md").write_text("# Мастер\n", encoding="utf-8")
    base = "/files/kurs/metricheskie-metody"

    pdf = client.get(f"{base}/master.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert pdf.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in pdf.headers  # would break the PDF viewer
    assert not pdf.headers.get("content-disposition", "").startswith("attachment")

    png = client.get(f"{base}/extracted/S1/pages/p0001.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert png.content == SAMPLE_PNG.read_bytes()

    md = client.get(f"{base}/master.md")
    assert md.headers["content-type"] == "text/plain; charset=utf-8"
    assert "# Мастер" in md.text

    ranged = client.get(f"{base}/master.pdf", headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206 and len(ranged.content) == 10

    non_ascii = client.get(f"{base}/sources/{quote('S1_lektsiya.pdf')}")
    assert non_ascii.status_code == 200


def test_files_never_run_in_our_origin(client: TestClient, topic: Path) -> None:
    base = "/files/kurs/metricheskie-metody"
    (topic / "sources" / "W1_page.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    (topic / "sources" / "pic.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>"
    )
    (topic / "sources" / "deck.pptx").write_bytes(b"PK\x03\x04 fake")
    page = client.get(f"{base}/sources/W1_page.html")
    assert page.headers["content-type"] == "text/plain; charset=utf-8"
    assert page.headers["x-content-type-options"] == "nosniff"
    svg = client.get(f"{base}/sources/pic.svg")
    assert svg.headers["content-type"] == "image/svg+xml"
    assert svg.headers["content-security-policy"].startswith("sandbox")
    deck = client.get(f"{base}/sources/deck.pptx")
    assert deck.headers["content-type"] == "application/octet-stream"
    assert deck.headers["content-disposition"].startswith("attachment")


@pytest.mark.parametrize(
    "path",
    [
        "../../secret.txt",
        "%2e%2e/%2e%2e/secret.txt",
        "..%2f..%2fsecret.txt",
        "..%2F..%2F..%2Fsecret.txt",
        "sources/../../../secret.txt",
        "sources%2F..%2F..%2F..%2Fsecret.txt",
        "sources%5C..%5C..%5C..%5Csecret.txt",
        "..%5C..%5C..%5Csecret.txt",
        "%2Fetc%2Fpasswd",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "C%3A%5CWindows%5Cwin.ini",
        "sources/S1_lektsiya.pdf:stream",
        ".git/config",
        ".topic.yaml.lock",
        ".gitignore",
        "sources",  # a directory
        "nothing/here.txt",
        "%00",
        "sources/%00.pdf",
    ],
)
def test_files_refuse_paths_outside_the_topic(
    client: TestClient, topic: Path, secret: Path, path: str
) -> None:
    response = client.get(f"/files/kurs/metricheskie-metody/{path}")
    assert response.status_code == 404, path
    assert "TOP-SECRET" not in response.text
    assert "Traceback" not in response.text
    assert "не найден" in response.text  # «Файл не найден» or «Страница не найдена»


def test_files_refuse_a_foreign_course_or_topic_segment(
    client: TestClient, topic: Path, secret: Path
) -> None:
    for url in (
        "/files/..%2Fkurs/metricheskie-metody/topic.yaml",
        "/files/kurs/..%2Fmetricheskie-metody/topic.yaml",
        "/files/kurs/../secret.txt",
        "/files/%2e%2e/secret.txt/x",
        "/files/kurs/metricheskie-metody%5C..%5C..%5Csecret.txt/x",
    ):
        response = client.get(url)
        assert response.status_code == 404, url
        assert "TOP-SECRET" not in response.text


def test_files_refuse_symlinks_that_leave_the_topic(
    client: TestClient, topic: Path, secret: Path
) -> None:
    link = topic / "sources" / "link.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("символические ссылки недоступны")
    response = client.get("/files/kurs/metricheskie-metody/sources/link.txt")
    assert response.status_code == 404
    assert "TOP-SECRET" not in response.text


def test_files_of_another_topic_are_not_reachable_through_this_one(
    client: TestClient, topic: Path, settings: Settings
) -> None:
    other = create_topic(settings, title="Другая", course="Курс", slug="drugaya")
    (other / "sources" / "private.txt").write_text("PRIVATE", encoding="utf-8")
    ok = client.get("/files/kurs/drugaya/sources/private.txt")
    assert ok.status_code == 200 and ok.text == "PRIVATE"
    leak = client.get("/files/kurs/metricheskie-metody/..%2Fdrugaya%2Fsources%2Fprivate.txt")
    assert leak.status_code == 404 and "PRIVATE" not in leak.text


# ---------------------------------------------------------------- doctor


def test_doctor_page_shows_checks(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_checks(settings, *, check_auth=True):
        calls.append(check_auth)
        return [
            Check("python", "Python", "ok", "3.13.7", required=True),
            Check("pandoc", "Pandoc", "ok", "pandoc 3.9", required=True),
            Check(
                "xelatex",
                "XeLaTeX",
                "missing",
                "не найден",
                hint="Установите MiKTeX",
                required=True,
            ),
            Check("ffmpeg", "ffmpeg", "warn", "старая версия", hint="Обновите ffmpeg"),
            Check("cuda", "CUDA", "info", "GPU не нужен"),
        ]

    monkeypatch.setattr("h0lon.doctor.run_checks", fake_checks)
    page = client.get("/doctor")
    assert page.status_code == 200
    html = page.text
    assert calls == [True]
    for needle in (
        "Python",
        "3.13.7",
        "XeLaTeX",
        "не найден",
        "Установите MiKTeX",
        "Обновите ffmpeg",
    ):
        assert needle in html, needle
    assert (
        "badge-ok" in html and "badge-bad" in html and "badge-warn" in html and "badge-info" in html
    )
    assert "Не пройдено обязательных проверок: 1 (XeLaTeX)" in html
    assert "Что сделать: GPU" not in html
    client.get("/doctor?no_auth=1")
    assert calls == [True, False]


def test_doctor_page_all_green(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "h0lon.doctor.run_checks",
        lambda settings, *, check_auth=True: [
            Check("python", "Python", "ok", "3.13", required=True)
        ],
    )
    html = client.get("/doctor").text
    assert "Все обязательные проверки пройдены" in html
    assert "Не пройдено" not in html


# ---------------------------------------------------------------- security and errors


def test_pages_carry_a_strict_csp_and_no_external_resources(
    client: TestClient, topic: Path
) -> None:
    for url in ("/", TOPIC, f"{TOPIC}/source/S1", "/nope"):
        response = client.get(url)
        csp = response.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "script-src 'self'" in csp
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "same-origin"
        assert not re.findall(r'(?:src|href)="(?:https?:)?//[^"]+"', response.text)
        assert "<style" not in response.text and ' style="' not in response.text


def test_static_files_and_no_api_docs(client: TestClient) -> None:
    css = client.get("/static/app.css")
    assert css.status_code == 200 and "prefers-color-scheme: dark" in css.text
    assert "http://" not in css.text.replace("http://www.w3.org", "")
    assert "@import" not in css.text and "url(http" not in css.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "EventSource" in js.text
    for url in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(url).status_code == 404


def test_cross_site_posts_are_refused(client: TestClient, topic: Path) -> None:
    evil = {"Origin": "http://evil.example"}
    for url, data in (
        ("/topics", {"title": "x", "course": "y"}),
        (f"{TOPIC}/build", {}),
        (f"{TOPIC}/review-gate", {"enabled": "0"}),
        (f"{TOPIC}/approve", {}),
    ):
        response = client.post(url, data=data, headers=evil)
        assert response.status_code == 403, url
        assert "другого сайта" in response.text
    assert client.post(f"{TOPIC}/build", headers={"Origin": "null"}).status_code == 403
    assert (
        client.post(f"{TOPIC}/build", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    )
    assert load_topic(topic).review_gate is None
    # the page's own origin is fine, and GET is never refused for its Origin
    same = client.post(
        f"{TOPIC}/review-gate", data={"enabled": "0"}, headers={"Origin": "http://testserver"}
    )
    assert same.status_code == 303
    assert client.get("/", headers=evil).status_code == 200


def test_foreign_host_header_is_refused_on_loopback(settings: Settings, topic: Path) -> None:
    app = create_app(settings, allowed_hosts={"127.0.0.1", "localhost", "::1"})
    with TestClient(app, base_url="http://127.0.0.1:8765", follow_redirects=False) as c:
        assert c.get("/").status_code == 200
        assert c.get("/", headers={"Host": "localhost:8765"}).status_code == 200
        assert c.get("/", headers={"Host": "[::1]:8765"}).status_code == 200
        rebinding = c.get("/", headers={"Host": "attacker.example:8765"})
        assert rebinding.status_code == 403
        assert "Host" in rebinding.text
        assert c.get("/", headers={"Host": ""}).status_code == 403


def test_unexpected_error_is_a_friendly_500(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("внутренняя поломка")

    monkeypatch.setattr("h0lon.web.views.present.scan_topics", boom)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        page = c.get("/")
    assert page.status_code == 500
    assert "Внутренняя ошибка" in page.text
    assert "RuntimeError: внутренняя поломка" in page.text
    assert "Traceback" not in page.text and 'File "' not in page.text


def test_unknown_routes_and_methods_are_friendly(client: TestClient, topic: Path) -> None:
    missing = client.get("/net-takoy-stranitsy")
    assert missing.status_code == 404 and "Страница не найдена" in missing.text
    wrong = client.get(f"{TOPIC}/extract")  # actions are POST only
    assert wrong.status_code == 405 and "Действие не поддерживается" in wrong.text


def test_invalid_form_is_a_friendly_422(client: TestClient, topic: Path) -> None:
    response = client.post(
        f"{TOPIC}/sources", content=b"{}", headers={"Content-Type": "application/json"}
    )
    assert response.status_code in (303, 422)
    assert "Traceback" not in response.text
