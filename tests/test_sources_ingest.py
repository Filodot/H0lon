from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from test_sources_kit import (
    WIDE,
    break_xref,
    make_docx,
    make_pdf,
    make_png,
    make_pptx,
    truncate_file,
)

from h0lon.config import Settings
from h0lon.sources import ingest
from h0lon.sources.detect import os_path
from h0lon.sources.ingest import (
    COUNTERS_KEY,
    LOCK_FILE,
    IngestError,
    add_sources,
    detect_kind,
    list_sources,
    print_sources,
    update_source,
)
from h0lon.sources.models import SourceRecord
from h0lon.workspace import create_topic, load_topic


@pytest.fixture
def topic(make_settings: Callable[..., Settings]) -> tuple[Settings, Path]:
    settings = make_settings(general={"git_per_topic": False})
    path = create_topic(settings, title="Теорвер — лекция 3", course="Теорвер")
    return settings, path


@pytest.fixture
def materials(tmp_path: Path) -> Path:
    d = tmp_path / "Материалы курса"
    d.mkdir()
    return d


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_sources(topic_dir: Path) -> list[dict]:
    data = yaml.safe_load((topic_dir / "topic.yaml").read_text(encoding="utf-8"))
    return data["sources"]


def _files(topic_dir: Path) -> list[str]:
    return sorted(p.name for p in (topic_dir / "sources").iterdir())


# ---------------------------------------------------------------- basic ingest


def test_add_pdfs_with_cyrillic_names(topic, materials: Path) -> None:
    settings, topic_dir = topic
    slides = make_pdf(materials / "Лекция 03. Слайды.pdf", size=WIDE, chars=250, pages=5)
    text = make_pdf(materials / "Конспект лекции (полный).pdf", chars=1500, seed="a")
    scan = make_pdf(materials / "скан_с_телефона 1.PDF", chars=0, image=True)

    report = add_sources(settings, topic_dir, [str(slides), str(text), str(scan)])

    assert report.warnings == []
    assert [r.id for r in report.added] == ["S1", "P1", "P2"]
    assert [r.kind for r in report.added] == ["slides", "pdf-text", "pdf-scan"]
    s1, p1, p2 = report.added
    assert s1.file == "sources/S1_lektsiya-03-slaydy.pdf"
    assert p1.file == "sources/P1_konspekt-lektsii-polnyy.pdf"
    assert p2.file == "sources/P2_skan-s-telefona-1.pdf"  # extension lower-cased
    assert s1.original_name == "Лекция 03. Слайды.pdf"
    assert s1.title == "Лекция 03. Слайды"
    assert p2.title == "скан с телефона 1"  # underscores read as spaces
    assert s1.units == {"pages": 5, "slides": 5}
    assert p1.units == {"pages": 3}
    assert set(p1.quality) >= {"text_layer", "chars_per_page", "aspect"}
    assert p1.status == "added" and p1.url is None

    for record in report.added:
        copy = topic_dir / record.file
        assert copy.is_file()
        assert record.sha256 == _sha(copy)
        assert record.size == copy.stat().st_size
        assert record.added.endswith("+00:00")

    raw = _raw_sources(topic_dir)
    assert [r["id"] for r in raw] == ["S1", "P1", "P2"]
    # exclude_none=False: every model field is present, None included.
    assert raw[0]["url"] is None and raw[0]["extracted_key"] is None and raw[0]["error"] is None
    assert list_sources(topic_dir) == report.added
    assert load_topic(topic_dir).model_extra[COUNTERS_KEY] == {"P": 2, "S": 1}
    assert _files(topic_dir) == [
        "P1_konspekt-lektsii-polnyy.pdf",
        "P2_skan-s-telefona-1.pdf",
        "S1_lektsiya-03-slaydy.pdf",
    ]
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_other_kinds_get_their_letters(topic, materials: Path) -> None:
    settings, topic_dir = topic
    md = materials / "Заметки.md"
    md.write_text("# Заметки\n", encoding="utf-8")
    tex = materials / "статья.tex"
    tex.write_text("\\documentclass{article}", encoding="utf-8")
    html = materials / "страница.html"
    html.write_text(
        "<html><head><title>Мера Лебега — Википедия</title></head></html>", encoding="utf-8"
    )
    video = materials / "lecture.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    audio = materials / "talk.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 100)
    items = [
        make_docx(materials / "Отчёт.docx"),
        md,
        tex,
        make_pptx(materials / "deck.pptx", slides=3),
        make_png(materials / "photo 1.png"),
        html,
        video,
        audio,
    ]
    report = add_sources(settings, topic_dir, [str(p) for p in items])
    assert report.warnings == []
    got = [(r.id, r.kind) for r in report.added]
    assert got == [
        ("D1", "docx"),
        ("D2", "md"),
        ("D3", "tex"),
        ("S1", "slides"),
        ("H1", "handwritten"),
        ("W1", "web"),
        ("V1", "video"),
        ("A1", "audio"),
    ]
    by_id = {r.id: r for r in report.added}
    assert by_id["S1"].units == {"slides": 3}
    assert by_id["H1"].units == {"pages": 1}
    assert by_id["W1"].title == "Мера Лебега — Википедия"  # from <title>
    assert by_id["W1"].file == "sources/W1_stranitsa.html"
    assert by_id["V1"].file == "sources/V1_lecture.mp4"


def test_kind_override_and_title(topic, materials: Path) -> None:
    settings, topic_dir = topic
    scan = make_pdf(materials / "рукопись.pdf", chars=0, image=True)
    report = add_sources(settings, topic_dir, [str(scan)], kind="handwritten", title="  Лекция  3 ")
    (record,) = report.added
    assert record.id == "H1" and record.kind == "handwritten"
    assert record.title == "Лекция 3"
    assert record.quality["detected_kind"] == "pdf-scan"
    assert record.units == {"pages": 3}

    # --kind also lets unknown formats in.
    odd = materials / "notes.xyz"
    odd.write_text("plain text", encoding="utf-8")
    (record,) = add_sources(settings, topic_dir, [str(odd)], kind="MD").added
    assert record.kind == "md" and record.file == "sources/D1_notes.xyz"


def test_html_without_title_uses_file_name(topic, materials: Path) -> None:
    settings, topic_dir = topic
    page = materials / "сохранённая_страница.htm"
    page.write_bytes("<html><body>Текст</body></html>".encode("cp1251"))
    (record,) = add_sources(settings, topic_dir, [str(page)]).added
    assert record.title == "сохранённая страница"
    assert (topic_dir / record.file).read_bytes() == page.read_bytes()  # a byte copy


# ---------------------------------------------------------------- ids


def test_ids_are_never_reused(topic, materials: Path) -> None:
    settings, topic_dir = topic
    a = make_pdf(materials / "a.pdf", seed="a")
    b = make_pdf(materials / "b.pdf", seed="b")
    c = make_pdf(materials / "c.pdf", seed="c")
    d = make_pdf(materials / "d.pdf", seed="d")
    add_sources(settings, topic_dir, [str(a), str(b)])

    # The user deletes P2 by hand (record and file): P2 must not come back.
    meta_path = topic_dir / "topic.yaml"
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    data["sources"] = [s for s in data["sources"] if s["id"] != "P2"]
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (topic_dir / "sources" / "P2_b.pdf").unlink()
    (record,) = add_sources(settings, topic_dir, [str(c)]).added
    assert record.id == "P3"

    # Without the counter, files and extracted/ dirs still block old numbers.
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    del data[COUNTERS_KEY]
    data["sources"] = [s for s in data["sources"] if s["id"] != "P3"]
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (topic_dir / "extracted" / "P4").mkdir()
    (record,) = add_sources(settings, topic_dir, [str(d)]).added
    assert record.id == "P5"


# ---------------------------------------------------------------- duplicates and errors


def test_duplicates_by_sha256_are_skipped(topic, materials: Path) -> None:
    settings, topic_dir = topic
    original = make_pdf(materials / "Лекция 1.pdf")
    renamed = materials / "копия лекции.pdf"
    renamed.write_bytes(original.read_bytes())
    add_sources(settings, topic_dir, [str(original)])

    report = add_sources(settings, topic_dir, [str(renamed), str(original)])
    assert report.added == []
    assert len(report.warnings) == 2
    assert all("уже добавлен как P1" in w for w in report.warnings)
    assert [r.id for r in list_sources(topic_dir)] == ["P1"]
    assert _files(topic_dir) == ["P1_lektsiya-1.pdf"]  # no leftover temporary copies

    # The same file twice in one call.
    other = make_pdf(materials / "other.pdf", seed="x")
    report = add_sources(settings, topic_dir, [str(other), str(other)])
    assert [r.id for r in report.added] == ["P2"]
    assert len(report.warnings) == 1


def test_missing_file_raises_before_anything_is_added(topic, materials: Path) -> None:
    settings, topic_dir = topic
    good = make_pdf(materials / "good.pdf")
    with pytest.raises(IngestError, match=r"Файл не найден: .*нет такого файла\.pdf"):
        add_sources(settings, topic_dir, [str(good), str(materials / "нет такого файла.pdf")])
    assert list_sources(topic_dir) == []
    assert _files(topic_dir) == []
    with pytest.raises(IngestError, match="https://"):
        add_sources(settings, topic_dir, ["www.example.org/page"])


@pytest.mark.parametrize(
    ("items", "kwargs", "match"),
    [
        ([], {}, "Не указаны"),
        (["  ", ""], {}, "Не указаны"),
        (["x.pdf"], {"kind": "presentation"}, "Неизвестный вид"),
        (["https://example.org"], {"kind": "pdf-text"}, "Для ссылки"),
        (["ftp://example.org/a.pdf"], {}, "http:// и https://"),
    ],
)
def test_invalid_requests(topic, items, kwargs, match) -> None:
    settings, topic_dir = topic
    with pytest.raises(IngestError, match=match):
        add_sources(settings, topic_dir, items, **kwargs)


def test_title_for_several_items_is_an_error(topic, materials: Path) -> None:
    settings, topic_dir = topic
    a, b = make_pdf(materials / "a.pdf", seed="a"), make_pdf(materials / "b.pdf", seed="b")
    with pytest.raises(IngestError, match="--title"):
        add_sources(settings, topic_dir, [str(a), str(b)], title="Одно название")
    assert list_sources(topic_dir) == []


def test_unsupported_named_file_is_an_error(topic, materials: Path) -> None:
    settings, topic_dir = topic
    doc = materials / "Старый отчёт.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(IngestError, match=r"\.doc не поддерживается — сохраните его в Word"):
        add_sources(settings, topic_dir, [str(make_pdf(materials / "a.pdf")), str(doc)])
    assert list_sources(topic_dir) == []


def test_missing_topic(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match=r"topic\.yaml"):
        add_sources(settings, tmp_path / "nope", ["a.pdf"])


def test_broken_and_empty_files_are_skipped_with_warning(topic, materials: Path) -> None:
    settings, topic_dir = topic
    broken = materials / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    empty = materials / "empty.md"
    empty.write_bytes(b"")
    fake_docx = materials / "fake.docx"
    fake_docx.write_text("plain text")
    good = make_pdf(materials / "good.pdf")
    report = add_sources(settings, topic_dir, [str(broken), str(empty), str(fake_docx), str(good)])
    assert [r.id for r in report.added] == ["P1"]
    assert len(report.warnings) == 3
    assert "broken.pdf: не удалось открыть PDF" in report.warnings[0]
    assert "пустой" in report.warnings[1]
    assert "DOCX" in report.warnings[2]
    assert _files(topic_dir) == ["P1_good.pdf"]


# ---------------------------------------------------------------- directories and wildcards


def test_directory_is_a_warning(topic, materials: Path) -> None:
    settings, topic_dir = topic
    make_pdf(materials / "inside.pdf")
    good = make_pdf(materials.parent / "outside.pdf")
    report = add_sources(settings, topic_dir, [str(materials), str(good)])
    assert [r.id for r in report.added] == ["P1"]
    (warning,) = report.warnings
    assert "каталоги не поддерживаются — перечислите файлы" in warning
    assert "*.pdf" in warning


def test_wildcards_are_expanded_in_natural_order(topic, materials: Path) -> None:
    settings, topic_dir = topic
    for n in (10, 2, 1):
        make_pdf(materials / f"Лекция {n}.pdf", seed=str(n))
    (materials / "readme.xyz").write_text("x")
    (materials / "sub").mkdir()
    report = add_sources(settings, topic_dir, [str(materials / "*")])
    assert [r.original_name for r in report.added] == [
        "Лекция 1.pdf",
        "Лекция 2.pdf",
        "Лекция 10.pdf",
    ]
    assert [r.id for r in report.added] == ["P1", "P2", "P3"]
    (warning,) = report.warnings
    assert "readme.xyz" in warning and "пропущен" in warning
    with pytest.raises(IngestError, match="ничего не найдено"):
        add_sources(settings, topic_dir, [str(materials / "*.docx")])


# ---------------------------------------------------------------- file system corner cases


def test_read_only_source_gives_writable_copy(topic, materials: Path) -> None:
    settings, topic_dir = topic
    src = make_pdf(materials / "только чтение.pdf")
    os.chmod(src, stat.S_IREAD)
    try:
        (record,) = add_sources(settings, topic_dir, [str(src)]).added
    finally:
        os.chmod(src, stat.S_IREAD | stat.S_IWRITE)
    copy = topic_dir / record.file
    assert os.access(copy, os.W_OK)
    copy.unlink()  # would fail on Windows for a read-only file


def test_long_source_path(topic, tmp_path: Path) -> None:
    settings, topic_dir = topic
    deep = tmp_path
    for n in range(6):
        deep = deep / f"очень длинное имя каталога с материалами курса номер {n}"
    os.makedirs(os_path(deep), exist_ok=True)
    src_name = "Лекция с очень длинным названием про случайные величины и их распределения.pdf"
    target = deep / src_name
    assert len(str(target)) > 300
    pdf = make_pdf(tmp_path / "short.pdf")
    with open(os_path(target), "wb") as fh:
        fh.write(pdf.read_bytes())

    (record,) = add_sources(settings, topic_dir, [str(target)]).added
    assert record.original_name == src_name
    assert record.file is not None
    slug = record.file.removeprefix("sources/P1_").removesuffix(".pdf")
    assert len(slug) <= ingest.SLUG_MAX_LEN
    assert record.sha256 == _sha(pdf)


def test_quoted_item_and_relative_path(
    topic, materials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, topic_dir = topic
    make_pdf(materials / "с пробелами.pdf")
    monkeypatch.chdir(materials)
    (record,) = add_sources(settings, topic_dir, ['"с пробелами.pdf"']).added
    assert record.original_name == "с пробелами.pdf"


def test_stale_incoming_copies_are_cleaned(topic, materials: Path) -> None:
    settings, topic_dir = topic
    stale = topic_dir / "sources" / ".incoming-dead.pdf"
    fresh = topic_dir / "sources" / ".incoming-live.pdf"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    old = time.time() - 3 * 24 * 3600
    os.utime(stale, (old, old))
    add_sources(settings, topic_dir, [str(make_pdf(materials / "a.pdf"))])
    assert not stale.exists()
    assert fresh.exists()  # may belong to a concurrent add


# ---------------------------------------------------------------- videos by link (no network)


def test_video_links_are_recorded_without_file(topic) -> None:
    settings, topic_dir = topic
    report = add_sources(
        settings,
        topic_dir,
        ["https://www.youtube.com/watch?v=abc123", "https://rutube.ru/video/xyz/"],
    )
    assert [(r.id, r.kind, r.file, r.sha256) for r in report.added] == [
        ("V1", "video", None, None),
        ("V2", "video", None, None),
    ]
    assert report.added[0].url == "https://www.youtube.com/watch?v=abc123"
    assert report.added[0].title == "youtube.com/watch?v=abc123"
    again = add_sources(settings, topic_dir, ["https://youtube.com/watch?v=abc123"])
    assert again.added == []
    assert "уже добавлен как V1" in again.warnings[0]
    titled = add_sources(settings, topic_dir, ["https://youtu.be/q"], title="Лекция 5")
    assert titled.added[0].title == "Лекция 5"


# ---------------------------------------------------------------- update_source


def test_update_source_keeps_foreign_fields(topic, materials: Path) -> None:
    settings, topic_dir = topic
    add_sources(settings, topic_dir, [str(make_pdf(materials / "a.pdf"))])
    (record,) = list_sources(topic_dir)
    record = record.model_copy(update={"error": "старая ошибка"})
    update_source(topic_dir, record)

    # Meanwhile another stage writes its own keys into the record and the topic.
    meta_path = topic_dir / "topic.yaml"
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    data["sources"][0]["review"] = {"checked": True}
    data["custom_topic_key"] = [1, 2]
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    record.status = "extracted"
    record.quality = {**record.quality, "pages_vision": 2}
    record.extracted_key = "abc"
    record.error = None
    update_source(topic_dir, record)

    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    stored = data["sources"][0]
    assert stored["status"] == "extracted"
    assert stored["extracted_key"] == "abc"
    assert stored["error"] is None  # None from the record wins
    assert stored["review"] == {"checked": True}  # unknown to the record: kept
    assert stored["quality"]["pages_vision"] == 2
    assert data["custom_topic_key"] == [1, 2]
    assert data["title"] == "Теорвер — лекция 3"
    assert not list(topic_dir.glob(".topic.yaml.*.tmp"))


def test_update_source_unknown_id(topic) -> None:
    _, topic_dir = topic
    record = SourceRecord(id="P9", kind="pdf-text", title="x", added="2026-10-03T00:00:00+00:00")
    with pytest.raises(KeyError, match="P9"):
        update_source(topic_dir, record)


def test_update_source_is_atomic(topic, materials: Path, monkeypatch) -> None:
    settings, topic_dir = topic
    add_sources(settings, topic_dir, [str(make_pdf(materials / "a.pdf"))])
    before = (topic_dir / "topic.yaml").read_bytes()
    (record,) = list_sources(topic_dir)
    record.status = "failed"

    def broken_save(path: Path, meta) -> Path:
        Path(path).write_text("sources: [trunc", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(ingest, "save_topic", broken_save)
    with pytest.raises(OSError, match="disk full"):
        update_source(topic_dir, record)
    assert (topic_dir / "topic.yaml").read_bytes() == before
    assert not list(topic_dir.glob(".topic.yaml.*.tmp"))


def test_concurrent_updates_and_adds_do_not_lose_writes(topic, materials: Path) -> None:
    settings, topic_dir = topic
    pdfs = [make_pdf(materials / f"src{n}.pdf", seed=f"s{n}", pages=1) for n in range(6)]
    add_sources(settings, topic_dir, [str(p) for p in pdfs[:4]])
    records = list_sources(topic_dir)
    errors: list[BaseException] = []

    def bump(record: SourceRecord) -> None:
        try:
            for n in range(15):
                record.quality = {**record.quality, "round": n}
                record.status = "extracting" if n < 14 else "extracted"
                update_source(topic_dir, record)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def add(path: Path) -> None:
        try:
            add_sources(settings, topic_dir, [str(path)])
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=bump, args=(r,)) for r in records]
    threads += [threading.Thread(target=add, args=(p,)) for p in pdfs[4:]]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    final = {r.id: r for r in list_sources(topic_dir)}
    assert sorted(final) == ["P1", "P2", "P3", "P4", "P5", "P6"]
    for r in records:
        assert final[r.id].status == "extracted"
        assert final[r.id].quality["round"] == 14


# ---------------------------------------------------------------- list / print


def test_list_sources_reports_bad_records(topic) -> None:
    _, topic_dir = topic
    meta_path = topic_dir / "topic.yaml"
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    data["sources"] = [{"id": "P1", "kind": "pdf", "title": "x", "added": "now"}]
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="источник №1 \\(P1\\) некорректен: поле kind"):
        list_sources(topic_dir)


def test_print_sources_table(topic, materials: Path) -> None:
    settings, topic_dir = topic
    add_sources(
        settings,
        topic_dir,
        [
            str(make_pdf(materials / "Слайды [3].pdf", size=WIDE, chars=250, pages=31)),
            str(make_pdf(materials / "скан.pdf", chars=0, image=True, pages=2)),
            str(make_docx(materials / "отчёт.docx")),
            "https://youtu.be/abc",
        ],
    )
    records = list_sources(topic_dir)
    records[1].status = "failed"
    records[1].error = "агент недоступен"
    console = Console(record=True, width=160, color_system=None)
    print_sources(records, console=console, title="Источники темы")
    out = console.export_text()
    assert "Источники темы" in out
    assert "Слайды [3]" in out  # brackets are not eaten as markup
    for needle in ("S1", "P1", "D1", "V1", "slides", "pdf-scan", "31 сл.", "2 стр.", "16:9"):
        assert needle in out, needle
    assert "текст 0%" in out and "ошибка" in out and "агент недоступен" in out
    assert "youtu.be/abc" in out and "https://youtu.be/abc" not in out  # title is the link
    assert "pdf-scan — скан PDF без текстового слоя" in out

    empty = Console(record=True, width=120, color_system=None)
    print_sources([], console=empty, title="Добавлены источники")
    assert empty.export_text().strip() == "Добавлены источники: нет."


# ---------------------------------------------------------------- CLI


def test_cli_add_and_sources(topic, materials: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from h0lon.cli import app

    settings, topic_dir = topic
    config = tmp_path / "h0lon.toml"
    config.write_text(
        "[general]\n"
        f"workspaces = '{settings.general.workspaces_dir}'\n"
        f"state_dir = '{tmp_path / 'state'}'\n"
        "git_per_topic = false\n",
        encoding="utf-8",
    )
    pdf = make_pdf(materials / "Лекция.pdf")
    runner = CliRunner()
    res = runner.invoke(app, ["-c", str(config), "add", str(topic_dir), str(pdf), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert [r["id"] for r in data["added"]] == ["P1"]

    res = runner.invoke(app, ["-c", str(config), "add", str(topic_dir), str(pdf)])
    assert res.exit_code == 1  # nothing added: the duplicate is only a warning
    assert "уже добавлен как P1" in res.output

    res = runner.invoke(app, ["-c", str(config), "add", str(topic_dir), "нет.pdf"])
    assert res.exit_code == 2
    assert "Файл не найден" in res.output

    res = runner.invoke(app, ["-c", str(config), "add", str(topic_dir), "http://example.com:abc/x"])
    assert res.exit_code == 2  # a readable error, not a ValueError traceback
    assert "Некорректная ссылка" in res.output and "Traceback" not in res.output

    res = runner.invoke(app, ["-c", str(config), "sources", "teorver/teorver-lektsiya-3"])
    assert res.exit_code == 0, res.output
    assert "P1" in res.output and "Лекция" in res.output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking")
def test_replace_retries_while_target_is_open(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    target.write_text("old")
    src = tmp_path / "s.txt"
    src.write_text("new")
    fh = open(target, "rb")  # noqa: SIM115 - held open on purpose
    timer = threading.Timer(0.2, fh.close)
    timer.start()
    try:
        ingest._replace(str(src), str(target))
    finally:
        timer.join()
        fh.close()
    assert target.read_text() == "new"


# ---------------------------------------------------------------- review fixes


def test_damaged_pdfs_are_added_with_warning(topic, materials: Path, capfd) -> None:
    settings, topic_dir = topic
    full = make_pdf(materials / "full.pdf", pages=20, chars=1500, seed="t")
    cut = truncate_file(full, materials / "недокачан.pdf", 0.5)
    broken = break_xref(make_pdf(materials / "x.pdf", seed="x"), materials / "битый.pdf")
    capfd.readouterr()

    report = add_sources(settings, topic_dir, [str(cut), str(broken)])

    p1, p2 = report.added
    assert p1.quality["repaired"] is True and p1.quality["truncated"] is True
    assert p2.quality["repaired"] is True and "truncated" not in p2.quality
    assert "обрезан" in p1.quality["notes"][0] and "скачайте файл заново" in p1.quality["notes"][0]
    assert "повреждён" in p2.quality["notes"][0]
    # Lost pages are not promised to the agent as scans.
    assert not any("распознает агент" in n for n in p1.quality["notes"])
    assert len(report.warnings) == 2
    assert report.warnings[0].startswith("недокачан.pdf (P1): PDF обрезан")
    assert report.warnings[1].startswith("битый.pdf (P2): PDF повреждён")
    out, err = capfd.readouterr()
    assert "format error" not in out + err  # MuPDF's own messages are not printed

    intact = add_sources(settings, topic_dir, [str(full)])
    assert intact.warnings == []
    assert "repaired" not in intact.added[0].quality

    console = Console(record=True, width=160, color_system=None)
    print_sources(list_sources(topic_dir), console=console, title="Источники темы")
    text = console.export_text()
    assert "обрезан" in text and "повреждён" in text


def test_wildcards_with_brackets_in_names(topic, tmp_path: Path) -> None:
    settings, topic_dir = topic
    folder = tmp_path / "[2026] Лекции [черновик]"
    for n in (2, 1):
        make_pdf(folder / f"Лекция [{n}].pdf", seed=str(n))

    report = add_sources(settings, topic_dir, [str(folder)])
    (warning,) = report.warnings
    suggested = warning.split("«")[1].split("»")[0]  # the pattern the warning recommends
    assert suggested == str(folder / "*.pdf")

    report = add_sources(settings, topic_dir, [suggested])
    assert [r.original_name for r in report.added] == ["Лекция [1].pdf", "Лекция [2].pdf"]
    assert report.warnings == []
    with pytest.raises(IngestError, match="ничего не найдено"):
        add_sources(settings, topic_dir, [str(folder / "*.docx")])


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://example.com:abc/page", "неверный номер порта"),
        ("http://example.com:99999/x", "неверный номер порта"),
        ("http://[abc", "IPv6"),
        ("https://", "нет адреса сайта"),
        ("https:///path", "нет адреса сайта"),
    ],
)
def test_malformed_links_are_errors(topic, url: str, reason: str) -> None:
    settings, topic_dir = topic
    with pytest.raises(IngestError, match=f"Некорректная ссылка: .* — .*{reason}"):
        add_sources(settings, topic_dir, ["https://youtu.be/ok", url])
    assert list_sources(topic_dir) == []  # checked before anything is added
    with pytest.raises(IngestError, match="Некорректная ссылка"):
        detect_kind(url)


def test_url_key_keeps_non_default_ports() -> None:
    key = ingest._url_key
    assert key("http://127.0.0.1:8001/page") != key("http://127.0.0.1:8002/page")
    assert key("http://example.org:80/a") == key("https://www.example.org:443/a/")
    assert key("http://[::1]:8000/x") == "//[::1]:8000/x"
    assert key("http://example.org:abc/") == "http://example.org:abc/"  # broken: verbatim


def test_dotted_names_without_extension_keep_their_tail(topic, materials: Path) -> None:
    settings, topic_dir = topic
    items = [
        make_pdf(materials / "Лекция 2024.10.03", seed="d"),
        make_pdf(materials / "Report v1.2", seed="v"),
        make_pdf(materials / "Lecture.final", seed="f"),
        make_pdf(materials / "Конспект.v2.pdf", seed="c"),
    ]
    report = add_sources(settings, topic_dir, [str(p) for p in items])
    assert [(r.title, r.file) for r in report.added] == [
        ("Лекция 2024.10.03", "sources/P1_lektsiya-2024-10-03.pdf"),
        ("Report v1.2", "sources/P2_report-v1-2.pdf"),
        ("Lecture.final", "sources/P3_lecture-final.pdf"),
        ("Конспект.v2", "sources/P4_konspekt-v2.pdf"),
    ]
    console = Console(record=True, width=160, color_system=None)
    print_sources(report.added, console=console, title="Добавлены")
    assert console.export_text().count("Лекция 2024.10.03") == 1  # no «original name» line


def test_kind_of_an_added_file_can_be_changed(topic, materials: Path) -> None:
    settings, topic_dir = topic
    scan = make_pdf(materials / "рукопись.pdf", chars=0, image=True)
    other = make_pdf(materials / "другое.pdf", seed="o")
    first = add_sources(settings, topic_dir, [str(scan), str(other)]).added
    assert [(r.id, r.kind) for r in first] == [("P1", "pdf-scan"), ("P2", "pdf-text")]
    stored = list_sources(topic_dir)[0]
    stored.status = "failed"
    stored.error = "агент недоступен"
    update_source(topic_dir, stored)

    report = add_sources(settings, topic_dir, [str(scan)], kind="handwritten")

    (record,) = report.added
    assert (record.id, record.kind, record.status, record.error) == (
        "H1",
        "handwritten",
        "added",
        None,
    )
    assert record.file == "sources/H1_rukopis.pdf"
    assert record.quality["detected_kind"] == "pdf-scan"
    assert record.added == first[0].added and record.title == "рукопись"
    (warning,) = report.warnings
    assert "уже был добавлен как P1" in warning
    assert "pdf-scan → handwritten" in warning and "новый id H1" in warning
    assert [r.id for r in list_sources(topic_dir)] == ["H1", "P2"]  # same place in the list
    assert _files(topic_dir) == ["H1_rukopis.pdf", "P2_drugoe.pdf"]

    # The same --kind again is an ordinary duplicate, and P1 is never issued again.
    again = add_sources(settings, topic_dir, [str(scan)], kind="handwritten")
    assert again.added == [] and "уже добавлен как H1" in again.warnings[0]
    (fresh,) = add_sources(settings, topic_dir, [str(make_pdf(materials / "н.pdf"))]).added
    assert fresh.id == "P3"

    # Same letter: the id and the file stay; --title renames.
    (rec,) = add_sources(settings, topic_dir, [str(other)], kind="pdf-scan", title="Скан").added
    assert (rec.id, rec.kind, rec.title, rec.file) == (
        "P2",
        "pdf-scan",
        "Скан",
        "sources/P2_drugoe.pdf",
    )
    assert (topic_dir / rec.file).is_file()


def test_kind_of_an_extracted_source_is_not_changed(topic, materials: Path) -> None:
    settings, topic_dir = topic
    scan = make_pdf(materials / "рукопись.pdf", chars=0, image=True)
    (record,) = add_sources(settings, topic_dir, [str(scan)]).added
    record.status = "extracted"
    update_source(topic_dir, record)

    report = add_sources(settings, topic_dir, [str(scan)], kind="handwritten")
    assert report.added == []
    (warning,) = report.warnings
    assert "уже извлечён" in warning and "вид не изменён" in warning
    assert "удалите запись P1" in warning and "--kind handwritten" in warning
    assert [(r.id, r.kind) for r in list_sources(topic_dir)] == [("P1", "pdf-scan")]


def test_utf16_text_files_are_stored_as_utf8(topic, materials: Path) -> None:
    settings, topic_dir = topic
    text = "# Определение\r\n\r\nСлучайная величина — функция.\r\n"
    notes = materials / "заметки.txt"
    notes.write_bytes(text.encode("utf-16"))  # with a BOM, like PowerShell 5.1 `>`

    (record,) = add_sources(settings, topic_dir, [str(notes)]).added
    copy = topic_dir / record.file
    assert copy.read_bytes() == text.encode("utf-8")
    assert record.sha256 == _sha(copy) and record.size == copy.stat().st_size
    assert record.quality["source_encoding"] == "UTF-16"
    assert "перекодирована в UTF-8" in record.quality["notes"][0]

    # The same text saved as UTF-8 is a duplicate of the converted copy.
    utf8 = materials / "заметки.md"
    utf8.write_bytes(text.encode("utf-8"))
    report = add_sources(settings, topic_dir, [str(utf8)])
    assert report.added == [] and "уже добавлен как D1" in report.warnings[0]

    # No BOM and no extension: UTF-16 BE LaTeX is still recognised and converted.
    tex = materials / "статья"
    tex.write_bytes(
        "\\documentclass{article}\n\\begin{document}Мера\\end{document}\n".encode("utf-16-be")
    )
    (record,) = add_sources(settings, topic_dir, [str(tex)]).added
    assert (record.kind, record.quality["source_encoding"]) == ("tex", "UTF-16 BE")
    assert (topic_dir / record.file).read_text(encoding="utf-8").startswith("\\documentclass")

    # NUL bytes that are not text: skipped with a warning.
    junk = materials / "junk.md"
    junk.write_bytes(b"\x00\x01\x02\x03" * 50)
    report = add_sources(settings, topic_dir, [str(junk)])
    assert report.added == [] and "нулевые байты" in report.warnings[0]
    assert not [n for n in _files(topic_dir) if n.startswith(".incoming-")]


def test_lock_timeout_is_a_readable_error(topic, materials: Path, monkeypatch) -> None:
    settings, topic_dir = topic
    add_sources(settings, topic_dir, [str(make_pdf(materials / "a.pdf", seed="a"))])
    (record,) = list_sources(topic_dir)
    lock_path = topic_dir / LOCK_FILE
    fd = ingest._acquire_file_lock(lock_path, 1.0)  # held by "another process"
    assert fd is not None
    monkeypatch.setattr(ingest, "LOCK_TIMEOUT_S", 0.3)
    try:
        with pytest.raises(IngestError, match="Тема занята другим процессом"):
            update_source(topic_dir, record)
        with pytest.raises(IngestError, match="Тема занята"):
            add_sources(settings, topic_dir, [str(make_pdf(materials / "b.pdf", seed="b"))])
    finally:
        ingest._release_file_lock(fd, lock_path)
    update_source(topic_dir, record)  # free again
    if sys.platform == "win32":
        assert not lock_path.exists()  # the topic folder stays clean


_WORKER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from h0lon.sources.ingest import add_sources

    topic, worker, mats, count = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    deadline = time.monotonic() + 60
    while not (mats / "go").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    for i in range(count):
        f = mats / f"w{worker}_{i}.md"
        f.write_text(f"# note {worker} {i}\\n", encoding="utf-8")
        report = add_sources(None, topic, [str(f)])
        assert len(report.added) == 1, report.warnings
    print("ok")
    """
)


def test_concurrent_processes_do_not_lose_records(topic, materials: Path) -> None:
    _, topic_dir = topic
    workers, count = 3, 6
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(topic_dir), str(n), str(materials), str(count)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        for n in range(workers)
    ]
    time.sleep(0.5)  # let the interpreters start, then release them together
    (materials / "go").write_text("")
    outputs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0] * workers, outputs
    records = list_sources(topic_dir)
    assert sorted(int(r.id[1:]) for r in records) == list(range(1, workers * count + 1))
    assert len(_files(topic_dir)) == workers * count  # no orphaned copies
    assert sorted(r.file for r in records) == sorted(f"sources/{n}" for n in _files(topic_dir))
