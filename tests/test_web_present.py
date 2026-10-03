"""Presentation helpers: formatting, quality signals, safe names and paths."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from h0lon.config import Settings
from h0lon.sources.models import SourceRecord
from h0lon.web import present


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


def rec(kind: str = "pdf-text", **extra: object) -> SourceRecord:
    return SourceRecord.model_validate(
        {"id": "P1", "kind": kind, "title": "t", "added": "2026-10-04T00:00:00Z", **extra}
    )


def test_format_size_and_duration() -> None:
    assert present.format_size(None) == "—"
    assert present.format_size(0) == "—"
    assert present.format_size(100) == "1 КБ"
    assert present.format_size(2048) == "2 КБ"
    assert present.format_size(1024 * 1024 * 3 // 2) == "1,5 МБ"
    assert present.format_duration(None) == "—"
    assert present.format_duration(3.25) == "3,2 с"
    assert present.format_duration(42.4) == "42 с"
    assert present.format_duration(65) == "1 мин 05 с"
    assert present.format_duration(3725) == "1 ч 02 мин"


def test_format_dates() -> None:
    assert present.format_ts(None) == "—"
    assert re.fullmatch(r"\d\d\.\d\d\.\d{4} \d\d:\d\d", present.format_ts(1_790_000_000))
    assert present.format_iso(None) == "—"
    assert present.format_iso("не дата") == "не дата"
    assert re.fullmatch(r"\d\d\.\d\d\.\d{4} \d\d:\d\d", present.format_iso("2026-10-04T10:00:00Z"))


def test_volume() -> None:
    assert present.volume(rec("slides", units={"pages": 31, "slides": 31})) == "31 сл."
    assert present.volume(rec(units={"pages": 12})) == "12 стр."
    assert present.volume(rec("video", units={"minutes": 45.5})) == "45.5 мин"
    assert present.volume(rec("md", size=3000)) == "3 КБ"
    assert present.volume(rec("web")) == "—"


def test_quality_signals_levels_and_content() -> None:
    signals = present.quality_signals(
        rec(
            "slides",
            quality={
                "text_layer": 0.42,
                "chars_per_page": 212.4,
                "pages_math": 18,
                "pages_scan": 1,
                "pages_graphic": 2,
                "pages_vision": 21,
                "pages_failed": 3,
                "cyrillic_ratio": 0.97,
                "scan_dpi": 120,
                "notes": ["a", "b"],
            },
        )
    )
    by_label = {s.label: s for s in signals}
    assert by_label["Текстовый слой"].value == "42 %" and by_label["Текстовый слой"].level == "bad"
    assert by_label["Знаков на слайд"].value == "212"
    assert by_label["С формулами"].level == "warn" and by_label["С формулами"].value == "18"
    assert by_label["Не распознано"].level == "bad"
    assert by_label["Скан"].value == "120 dpi" and by_label["Скан"].level == "warn"
    assert by_label["Замечаний"].value == "2"
    assert by_label["Кириллица"].value == "97 %"


def test_quality_signals_good_and_missing() -> None:
    good = {s.label: s for s in present.quality_signals(rec(quality={"text_layer": 1.0}))}
    assert good["Текстовый слой"].level == "ok"
    assert present.quality_signals(rec("md")) == []
    web = present.quality_signals(rec("web", quality={"encoding": "utf-8"}))
    assert [(s.label, s.value) for s in web] == [("Кодировка", "utf-8")]
    broken = present.quality_signals(rec(quality={"truncated": True}))
    assert [(s.label, s.level) for s in broken] == [("Файл", "bad")]
    repaired = present.quality_signals(rec(quality={"repaired": True}))
    assert repaired[0].value == "повреждён"
    junk = present.quality_signals(rec(quality={"text_layer": "много", "pages_math": True}))
    assert junk == []  # wrong types are ignored, not crashed on


def test_quality_notes_and_items() -> None:
    r = rec(
        quality={"notes": ["раз", "два"], "producer": "pdfTeX", "extra": {"a": 1}, "none": None}
    )
    assert present.quality_notes(r) == ["раз", "два"]
    assert present.quality_notes(rec(quality={"notes": "одна"})) == ["одна"]
    assert present.quality_notes(rec()) == []
    assert present.quality_items(r) == [
        ("producer", "pdfTeX"),
        ("extra", '{"a": 1}'),
        ("none", "—"),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("лекция.pdf", "лекция.pdf"),
        ("C:\\Users\\me\\Лекция 1.pdf", "Лекция 1.pdf"),
        ("../../etc/passwd", "passwd"),
        ("a:b*c?d.pdf", "a_b_c_d.pdf"),
        ("  .hidden. ", "hidden"),
        ("CON.pdf", "_CON.pdf"),
        ("nul", "_nul"),
        ("", "upload"),
        (None, "upload"),
        ("...", "upload"),
        ("tab\tname.md", "tab_name.md"),
    ],
)
def test_safe_upload_name(raw: str | None, expected: str) -> None:
    assert present.safe_upload_name(raw, set()) == expected


def test_safe_upload_name_is_unique_and_bounded() -> None:
    taken: set[str] = set()
    names = [present.safe_upload_name("Лекция.pdf", taken) for _ in range(3)]
    assert names == ["Лекция.pdf", "Лекция (2).pdf", "Лекция (3).pdf"]
    assert (
        present.safe_upload_name("ЛЕКЦИЯ.PDF", {"лекция.pdf"}) == "ЛЕКЦИЯ (2).PDF"
    )  # case-insens.
    long = present.safe_upload_name("я" * 400 + ".pdf", set())
    assert len(long) <= present.MAX_FILENAME_CHARS and long.endswith(".pdf")


def test_split_links() -> None:
    assert present.split_links("a\n\n  b  \r\nc") == ["a", "b", "c"]
    assert present.split_links("") == []
    assert present.split_links("   \n ") == []


@pytest.mark.parametrize(
    ("segment", "ok"),
    [
        ("kurs", True),
        ("metricheskie-metody", True),
        ("Лекция 3", True),
        ("S1", True),
        ("", False),
        (".", False),
        ("..", False),
        (".git", False),
        ("a/b", False),
        ("a\\b", False),
        ("C:", False),
        ("a:b", False),
        (" x", False),
        ("x ", False),
        ("a*b", False),
        ("a\x00b", False),
    ],
)
def test_valid_segment(segment: str, ok: bool) -> None:
    assert present.valid_segment(segment) is ok


def test_resolve_topic_file(tmp_path: Path) -> None:
    topic = tmp_path / "topic"
    (topic / "sources").mkdir(parents=True)
    (topic / "sources" / "a b.pdf").write_bytes(b"%PDF")
    (topic / ".git").mkdir()
    (topic / ".git" / "config").write_text("x")
    (tmp_path / "secret.txt").write_text("s")
    ok = present.resolve_topic_file(topic, "sources/a b.pdf")
    assert ok == (topic / "sources" / "a b.pdf").resolve()
    for bad in (
        "",
        "/sources/a b.pdf",
        "\\sources\\a b.pdf",
        "sources",
        "../secret.txt",
        "sources/../../secret.txt",
        "sources\\..\\..\\secret.txt",
        "C:/x",
        ".git/config",
        "sources/missing.pdf",
        "sources/a b.pdf:ads",
    ):
        assert present.resolve_topic_file(topic, bad) is None, bad
    link = topic / "sources" / "out.txt"
    try:
        os.symlink(tmp_path / "secret.txt", link)
    except (OSError, NotImplementedError):
        return
    assert present.resolve_topic_file(topic, "sources/out.txt") is None


def test_file_delivery_types() -> None:
    cases = {
        "a.pdf": ("application/pdf", False),
        "a.PNG": ("image/png", False),
        "a.jpg": ("image/jpeg", False),
        "a.md": ("text/plain; charset=utf-8", False),
        "a.html": ("text/plain; charset=utf-8", False),
        "a.json": ("text/plain; charset=utf-8", False),
        "a.svg": ("image/svg+xml", False),
        "a.docx": ("application/octet-stream", True),
        "a": ("application/octet-stream", True),
    }
    for name, (media, attachment) in cases.items():
        got_media, got_attachment, headers = present.file_delivery(Path(name))
        assert (got_media, got_attachment) == (media, attachment), name
        assert headers["X-Content-Type-Options"] == "nosniff"
    assert "sandbox" in present.file_delivery(Path("a.svg"))[2]["Content-Security-Policy"]
    assert "Content-Security-Policy" not in present.file_delivery(Path("a.pdf"))[2]


def test_appendix_links(tmp_path: Path) -> None:
    md = tmp_path / "master.md"
    md.write_text("# A {#app:corrections}\n\n# B {#app:coverage .x}\n", encoding="utf-8")
    assert present.appendix_links(md) == [
        ("app:corrections", "Журнал правок"),
        ("app:coverage", "Карта покрытия"),
    ]
    assert present.appendix_links(tmp_path / "nope.md") == []


def test_scan_topics_groups_by_course(settings: Settings) -> None:
    from h0lon.workspace import create_topic

    s = settings
    create_topic(s, title="Б тема", course="Алгебра")
    create_topic(s, title="А тема", course="Алгебра")
    create_topic(s, title="Тема", course="Геометрия")
    (s.general.workspaces_dir / "pusto").mkdir()  # a course without topics
    (s.general.workspaces_dir / ".hidden" / "x").mkdir(parents=True)
    (s.general.workspaces_dir / ".hidden" / "x" / "topic.yaml").write_text("title: x")
    groups = present.scan_topics(s)
    assert [g.name for g in groups] == ["algebra", "geometriya"]
    assert [g.title for g in groups] == ["Алгебра", "Геометрия"]
    assert [t.title for t in groups[0].topics] == ["А тема", "Б тема"]
    assert groups[0].topics[0].url == "/t/algebra/a-tema"


def test_appendix_pages_prefers_bookmarks_then_text(tmp_path: Path) -> None:
    import pymupdf

    wanted = [
        ("app:conflicts", "Расхождения между источниками"),
        ("app:coverage", "Карта покрытия"),
    ]
    with_toc = tmp_path / "toc.pdf"
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page()
    doc.set_toc([[1, "Расхождения  между\nисточниками", 2], [1, "Карта покрытия", 4]])
    doc.save(with_toc)
    doc.close()
    assert present.appendix_pages(with_toc, wanted) == {"app:conflicts": 2, "app:coverage": 4}

    plain = tmp_path / "plain.pdf"
    doc = pymupdf.open()
    for number in range(1, 4):
        page = doc.new_page()
        page.insert_htmlbox(
            pymupdf.Rect(50, 50, 500, 200),
            "<p>Карта покрытия</p>" if number == 3 else f"<p>Страница {number}</p>",
        )
    doc.save(plain)
    doc.close()
    assert present.appendix_pages(plain, wanted) == {"app:coverage": 3}
    assert present.appendix_pages(tmp_path / "missing.pdf", wanted) == {}
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert present.appendix_pages(broken, wanted) == {}
