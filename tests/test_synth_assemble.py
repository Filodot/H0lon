"""Assembly of master.md: structure, appendices, empty notes, escaping, renderability."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from h0lon import tools
from h0lon.config import Settings
from h0lon.render import render_document
from h0lon.render.pandoc import MARKDOWN_FORMAT, read_front_matter
from h0lon.sources.models import SourceRecord
from h0lon.synth import assemble
from h0lon.synth.common import parse_src_ids
from h0lon.synth.model import (
    BuildContext,
    CoverageReport,
    Outline,
    OutlineSection,
)
from h0lon.workspace import create_topic

TODAY = "2026-10-04"


def make_outline() -> Outline:
    return Outline(
        title="Метрические методы классификации",
        sections=[
            OutlineSection(id="s01", title="Классификация по похожести", level=1),
            OutlineSection(
                id="s01-01",
                title="Метрика и расстояния",
                level=2,
                blocks=["S1.b001"],
                parent="s01",
            ),
            OutlineSection(
                id="s01-02", title="Метод 1NN", level=2, blocks=["S1.b002"], parent="s01"
            ),
            OutlineSection(id="s02", title="Ядерное сглаживание", level=1, blocks=["P1.b001"]),
        ],
    )


SECTION_TEXT = {
    "s01-01": (
        "## Метрика и расстояния {#sec:s01-01}\n\n"
        "Функция $\\rho(a,b)$ называется метрикой.[[S1:s6]]\n"
        "<!-- src: S1.b001 -->\n"
    ),
    "s01-02": (
        "## Метод 1NN {#sec:s01-02}\n\n"
        "Объект получает метку ближайшего соседа.[[S1:s11]]\n"
        "<!-- src: S1.b002 -->\n"
    ),
    "s02": (
        "# Ядерное сглаживание {#sec:s02}\n\n"
        "Ядро $K(r)$ неотрицательно и чётно.[[P1:p2]]\n"
        "<!-- src: P1.b001 -->\n"
    ),
}


def make_sources() -> list[SourceRecord]:
    return [
        SourceRecord(
            id="S1",
            kind="slides",
            title="Лекция 03 | слайды",
            added="2026-10-03T00:00:00Z",
            units={"pages": 31, "slides": 31},
            status="extracted",
        ),
        SourceRecord(
            id="P1",
            kind="pdf-text",
            title="Летучка",
            added="2026-10-03T00:00:00Z",
            units={"pages": 16},
            status="extracted",
        ),
    ]


def make_coverage() -> CoverageReport:
    return CoverageReport(
        total=4,
        covered=3,
        by_source={"S1": {"total": 2, "covered": 2}, "P1": {"total": 2, "covered": 1}},
        uncovered=[
            {"block": "P1.b002", "verdict": "missing", "note": "условие сходимости"},
            {"block": "S1.b009", "verdict": "admin", "note": "титульный слайд"},
        ],
        rounds=1,
    )


@pytest.fixture
def topic_ctx(settings: Settings) -> BuildContext:
    topic = create_topic(
        Settings(general={**settings.general.model_dump(), "git_per_topic": False}),
        title="Метрические методы",
        course="Машинное обучение",
    )
    return BuildContext(topic_dir=topic, settings=settings)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_final(ctx: BuildContext, *, intro: bool = True, glossary: bool = True) -> None:
    final = ctx.synth_dir / "final"
    for sid, text in SECTION_TEXT.items():
        write(final / "sections" / f"{sid}.md", text)
    if intro:
        write(
            final / "intro.md",
            "# Введение {#sec:intro .unnumbered}\n\nТема охватывает метрические методы.\n",
        )
    if glossary:
        write(
            final / "glossary.md",
            "# Глоссарий терминов и обозначений {#sec:glossary .unnumbered}\n\n"
            "- **Метрика** (metric) — функция расстояния, [раздел](#sec:s01-01)\n",
        )


def write_notes(ctx: BuildContext) -> None:
    sections = ctx.synth_dir / "sections"
    write(
        sections / "s01-01.notes.json",
        json.dumps(
            {
                "corrections": [
                    {
                        "anchor": "S1:s6",
                        "block": "S1.b001",
                        "as_written": "метрика $\\rho(a,b)\\ge 0$",
                        "corrected": "$\\rho(a,b)\\ge 0$",
                        "reason": "лишнее слово",
                    }
                ],
                "conflicts": [
                    {
                        "topic": "Обозначение расстояния",
                        "variants": [
                            {"anchor": "S1:s6", "text": "$\\rho(a,b)$"},
                            {"anchor": "P1:p2", "text": "$d(a,b)$"},
                        ],
                        "resolution": "$\\rho(a,b)$",
                        "reason": "так в слайдах",
                    }
                ],
                "editorial": [
                    {
                        "kind": "answer",
                        "refers_to": "S1.b001",
                        "text": "Пометка «??»: $\\rho$ — метрика.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )
    write(
        sections / "s02.notes.json",
        json.dumps({"corrections": [], "conflicts": [], "editorial": []}),
    )


def build_master(ctx: BuildContext, **kwargs: Any) -> str:
    path = assemble.assemble_master(
        ctx, make_outline(), make_coverage(), make_sources(), today=TODAY, **kwargs
    )
    assert path == ctx.topic_dir / "master.md"
    return path.read_text(encoding="utf-8")


def headings(markdown: str) -> list[tuple[int, str]]:
    body = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.DOTALL)
    return [
        (len(m.group(1)), m.group(2))
        for m in re.finditer(r"^(#{1,6}) (.+)$", body, flags=re.MULTILINE)
    ]


def pandoc_blocks(markdown: str) -> list[dict[str, Any]]:
    """Blocks of the document as Pandoc reads it (needs pandoc)."""
    from h0lon import procutil

    pandoc = tools.find_pandoc()
    if pandoc is None:
        pytest.skip("pandoc not found")
    res = procutil.run(
        [str(pandoc), "-f", MARKDOWN_FORMAT, "-t", "json"], input_text=markdown, timeout=120
    )
    assert res.ok, res.stderr
    return json.loads(res.stdout)["blocks"]


# ---------------------------------------------------------------- notes


def test_collect_notes_dedup_and_sections(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_notes(ctx)
    write(
        ctx.synth_dir / "sections" / "s01-02.notes.json",
        json.dumps(
            {
                # the same correction as in s01-01 (other section) and a duplicate inside the file
                "corrections": [
                    {"anchor": "[[S1:s11]]", "as_written": "a", "corrected": "b", "reason": "r"},
                    {"anchor": "S1:s11", "as_written": "a", "corrected": "b", "reason": "r"},
                ],
                "conflicts": [],
                "editorial": [{"kind": "weird", "refers_to": "H1.b003", "text": "t"}],
            }
        ),
    )
    write(
        ctx.synth_dir / "global" / "global.notes.json",
        json.dumps(
            {
                "corrections": [
                    {
                        "section": "s02",
                        "anchor": "P1:p2",
                        "as_written": "x",
                        "corrected": "y",
                        "reason": "z",
                    }
                ],
                "conflicts": [],
                "editorial": [],
            }
        ),
    )
    write(ctx.synth_dir / "sections" / "s09.notes.json", "{not json")
    problems: list[str] = []
    notes = assemble.collect_notes(ctx, outline=make_outline(), warnings=problems)

    assert set(notes) == {"corrections", "conflicts", "editorial"}
    assert [(c["section"], c["anchor"]) for c in notes["corrections"]] == [
        ("s01-01", "S1:s6"),
        ("s01-02", "S1:s11"),  # brackets stripped, duplicate inside the file removed
        ("s02", "P1:p2"),  # global notes keep their own `section`
    ]
    assert notes["conflicts"][0]["section"] == "s01-01"
    assert notes["conflicts"][0]["variants"][1] == {"anchor": "P1:p2", "text": "$d(a,b)$"}
    assert notes["conflicts"][0]["resolution"] == "$\\rho(a,b)$"
    assert notes["editorial"][1]["kind"] == "clarification"  # unknown kind falls back
    assert len(problems) == 1 and "s09.notes.json" in problems[0]


def test_collect_notes_without_files(topic_ctx: BuildContext) -> None:
    assert assemble.collect_notes(topic_ctx) == {
        "corrections": [],
        "conflicts": [],
        "editorial": [],
    }


# ---------------------------------------------------------------- structure


def test_assemble_structure(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    write_notes(ctx)
    text = build_master(ctx)

    front = read_front_matter(text)
    assert front["title"] == "Метрические методы классификации"
    assert front["subtitle"] == "Мастер-конспект темы"
    assert front["course"] == "Машинное обучение"
    assert front["date"] == TODAY and isinstance(front["date"], str)
    assert front["author"] == "H0lon"
    assert front["sources"] == [
        {"id": "S1", "kind": "slides", "title": "Лекция 03 | слайды", "units": {"slides": 31}},
        {"id": "P1", "kind": "pdf-text", "title": "Летучка", "units": {"pages": 16}},
    ]

    heads = [h for _, h in headings(text) if h.startswith(("#", ""))]
    titles = [re.sub(r"\s*\{.*\}$", "", h) for h in heads]
    assert titles == [
        "Введение",
        "Классификация по похожести",  # chapter without a file: heading only
        "Метрика и расстояния",
        "Метод 1NN",
        "Ядерное сглаживание",
        "Глоссарий терминов и обозначений",
        "Расхождения между источниками",
        "Журнал правок",
        "Редакторские дополнения",
        "Карта покрытия",
        "Непокрытые блоки",
    ]
    assert "# Классификация по похожести {#sec:s01}" in text
    assert text.count(".appendix") == 1
    assert "# Расхождения между источниками {#app:conflicts .appendix}" in text
    assert "# Журнал правок {#app:corrections}" in text
    assert "# Редакторские дополнения {#app:editorial}" in text
    assert "# Карта покрытия {#app:coverage}" in text
    # body: intro first, glossary right before the appendices
    assert text.index("{#sec:intro") < text.index("{#sec:s01}") < text.index("{#sec:glossary")
    assert text.index("{#sec:glossary") < text.index("{#app:conflicts")
    assert parse_src_ids(text) == {"S1.b001", "S1.b002", "P1.b001"}  # appendices add none
    assert text.endswith("\n") and "\r" not in text


def test_assemble_appendices_content(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    write_notes(ctx)
    write(
        ctx.synth_dir / "global" / "global.notes.json",
        json.dumps(
            {
                "corrections": [],
                "conflicts": [
                    {
                        "section": "s02",
                        "topic": "Чётность ядра",
                        "variants": [{"anchor": "P1:p2", "text": "чётное"}],
                        "resolution": None,
                        "reason": "источник один",
                    }
                ],
                "editorial": [],
            }
        ),
    )
    text = build_master(ctx)

    # А: conflicts, with the chosen variant and the reason
    assert '::: {.conflict title="Обозначение расстояния"}' in text
    assert "Раздел [«Метрика и расстояния»](#sec:s01-01)." in text
    assert "- [[S1:s6]]: $\\rho(a,b)$" in text and "- [[P1:p2]]: $d(a,b)$" in text
    assert "В тексте принят вариант: $\\rho(a,b)$. Причина: так в слайдах" in text
    assert "В тексте дана нейтральная формулировка" in text  # resolution None
    # Б: corrections table
    assert "| Якорь | Как написано | Как исправлено | Причина |" in text
    assert "| [[S1:s6]] | метрика $\\rho(a,b)\\ge 0$ | $\\rho(a,b)\\ge 0$ | лишнее слово |" in text
    # В: editorial with a link to the section and the referenced block
    assert '::: {.editorial title="Ответ на пометку автора"}' in text
    assert "Раздел [«Метрика и расстояния»](#sec:s01-01); относится к `S1.b001`." in text
    assert "Пометка «??»: $\\rho$ — метрика." in text
    # Г: coverage map
    assert "| Источник | Блоков | Покрыто | % |" in text
    assert "| S1 — Лекция 03 \\| слайды | 2 | 2 | 100 % |" in text
    assert "| P1 — Летучка | 2 | 1 | 50 % |" in text
    assert "| **Всего** | **4** | **3** | **75 %** |" in text
    assert "- `P1.b002` — пропущен: условие сходимости" in text
    assert "- `S1.b009` — служебный: титульный слайд" in text
    assert text.index("пропущен") < text.index("служебный")  # missing blocks first


def test_assemble_empty_notes(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    text = build_master(ctx)
    assert "Прямых правок не было." in text
    assert "Расхождений между источниками не обнаружено." in text
    assert "Редакторских дополнений нет." in text
    assert "| Якорь |" not in text  # no empty table
    # the four appendices exist even when empty: lettering А–Г stays stable
    for ident in ("app:conflicts", "app:corrections", "app:editorial", "app:coverage"):
        assert f"{{#{ident}" in text
    only_empty = ctx.synth_dir / "sections"
    write(
        only_empty / "s02.notes.json",
        json.dumps({"corrections": [], "conflicts": [], "editorial": []}),
    )
    assert "Прямых правок не было." in build_master(ctx)


def test_assemble_minimal_intro_without_global_files(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx, intro=False, glossary=False)
    text = build_master(ctx)
    assert "# Введение {#sec:intro .unnumbered}" in text
    assert "**S1** (слайды, 31 сл.) — Лекция 03 \\| слайды" in text
    assert "**P1** (PDF, 16 стр.) — Летучка" in text
    assert "{#sec:glossary" not in text
    assert text.index("{#sec:intro") < text.index("{#sec:s01}")


def test_assemble_missing_section_file(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    (ctx.synth_dir / "final" / "sections" / "s01-02.md").unlink()
    with pytest.raises(assemble.AssembleError, match="s01-02"):
        build_master(ctx)
    # a leaf without blocks does not need a file: the heading alone is written
    outline = make_outline()
    outline.sections[2].blocks = []
    path = assemble.assemble_master(ctx, outline, make_coverage(), make_sources(), today=TODAY)
    assert "## Метод 1NN {#sec:s01-02}" in path.read_text(encoding="utf-8")


def test_assemble_adds_missing_heading(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    write(ctx.synth_dir / "final" / "sections" / "s01-02.md", "Текст без заголовка.\n")
    text = build_master(ctx)
    assert "## Метод 1NN {#sec:s01-02}\n\nТекст без заголовка." in text


def test_assemble_key_follows_inputs(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    write_notes(ctx)
    sources = make_sources()
    key = assemble.assemble_key(ctx, sources)
    assert key == assemble.assemble_key(ctx, sources)
    write(ctx.synth_dir / "final" / "sections" / "s02.md", SECTION_TEXT["s02"] + "Ещё абзац.\n")
    key2 = assemble.assemble_key(ctx, sources)
    assert key2 != key
    write(ctx.synth_dir / "sections" / "s02.notes.json", '{"corrections": []}')
    assert assemble.assemble_key(ctx, sources) != key2
    assert assemble.assemble_key(ctx, sources[:1]) != key2


# ---------------------------------------------------------------- escaping


HOSTILE = (
    "первая строка\n"
    "::: {.theorem}\n"
    "# Заголовок внутри заметки\n"
    "<!-- src: X9.b999 -->\n"
    "```python\n"
    "цена $5 и стоимость | таблица\n"
    "вторая | строка\n"
)


def test_clean_helpers() -> None:
    assert assemble.clean_inline("а | б\nв", table=True) == "а \\| б в"
    assert assemble.clean_inline("$|a|$ и | b", table=True) == "$|a|$ и \\| b"  # math keeps pipes
    assert assemble.clean_inline("`x|y` | z", table=True) == "`x|y` \\| z"
    assert assemble.clean_inline("::: fence") == "\\::: fence"
    assert assemble.clean_inline("# head") == "\\# head"
    assert assemble.clean_inline("- item") == "\\- item"
    assert assemble.clean_inline("цена $5 и $10") == "цена \\$5 и \\$10"  # not a formula
    assert assemble.clean_inline("$x$ и $5") == "$x$ и \\$5"
    assert assemble.clean_inline("a <b> c") == "a \\<b> c"
    assert assemble.clean_inline("до <!-- src: X1.b001 --> после") == "до после"
    assert assemble.attr_value('Он сказал "да" \\') == 'Он сказал \\"да\\"'
    block = assemble.clean_block(HOSTILE)
    assert "src: X9" not in block
    assert not any(ln.startswith((":::", "# ", "```")) for ln in block.splitlines())
    assert assemble.plain("a_b*c [x] $y") == "a\\_b\\*c \\[x\\] \\$y"
    assert assemble.snippet("Формула $x^2$ и текст", 100) == "Формула … и текст"
    long = assemble.snippet("слово " * 40, 30)
    assert len(long) <= 30 and long.endswith("…")


def test_hostile_notes_do_not_break_structure(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    write(
        ctx.synth_dir / "sections" / "s01-01.notes.json",
        json.dumps(
            {
                "corrections": [
                    {
                        "anchor": "S1:s6",
                        "as_written": HOSTILE,
                        "corrected": "a | b\n\nc",
                        "reason": "r |",
                    }
                ],
                "conflicts": [
                    {
                        "topic": 'Тема "в кавычках" $x',
                        "variants": [{"anchor": "S1:s6", "text": HOSTILE}],
                        "resolution": HOSTILE,
                        "reason": HOSTILE,
                    }
                ],
                "editorial": [{"kind": "answer", "refers_to": "??? | x", "text": HOSTILE}],
            },
            ensure_ascii=False,
        ),
    )
    text = build_master(ctx)
    assert parse_src_ids(text) == {"S1.b001", "S1.b002", "P1.b001"}  # no src from the notes
    assert "X9.b999" not in text

    # the document structure is exactly the expected one
    blocks = pandoc_blocks(text)
    levels = [b["c"][0] for b in blocks if b["t"] == "Header"]
    assert levels.count(1) == 2 + 1 + 1 + 4  # intro, two chapters... + glossary + appendices
    assert not any(b["t"] == "CodeBlock" for b in blocks)  # the fence was neutralised
    tables = [b for b in blocks if b["t"] == "Table"]
    assert len(tables) == 2  # corrections and coverage
    corrections = tables[0]
    head_cells = corrections["c"][3][1][0][1]
    assert len(head_cells) == 4
    body_rows = corrections["c"][4][0][3]
    assert len(body_rows) == 1 and len(body_rows[0][1]) == 4
    # fenced divs: only the two the assembler wrote (conflict, editorial)
    divs = [b for b in blocks if b["t"] == "Div"]
    assert {d["c"][0][1][0] for d in divs} == {"conflict", "editorial"}


# ---------------------------------------------------------------- coverage map


def test_coverage_appendix_with_blocks(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    extracted = ctx.topic_dir / "extracted" / "P1"
    extracted.mkdir(parents=True)
    block = {
        "id": "P1.b002",
        "source": "P1",
        "type": "paragraph",
        "anchor": "P1:p3",
        "text": "Условие сходимости $h_n\\to 0$ и $n h_n\\to\\infty$ для окна",
        "md": "…",
        "title": None,
    }
    write(extracted / "blocks.jsonl", json.dumps(block, ensure_ascii=False) + "\n")
    text = build_master(ctx)
    line = next(ln for ln in text.splitlines() if ln.startswith("- `P1.b002`"))
    assert "[[P1:p3]]" in line and "пропущен" in line
    assert "«Условие сходимости … и … для окна»" in line  # formulas dropped from the excerpt
    assert "\\to" not in line


def test_coverage_appendix_full(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    write_final(ctx)
    full = CoverageReport(
        total=4,
        covered=4,
        by_source={"S1": {"total": 2, "covered": 2}, "P1": {"total": 2, "covered": 2}},
        uncovered=[],
    )
    path = assemble.assemble_master(ctx, make_outline(), full, make_sources(), today=TODAY)
    text = path.read_text(encoding="utf-8")
    assert "| **Всего** | **4** | **4** | **100 %** |" in text
    assert "Непокрытых блоков нет" in text


# ---------------------------------------------------------------- renders


@pytest.mark.needs_xelatex
def test_assembled_master_renders_with_xelatex(
    topic_ctx: BuildContext, make_settings: Callable[..., Settings]
) -> None:
    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")
    ctx = topic_ctx
    write_final(ctx)
    write_notes(ctx)
    write(
        ctx.synth_dir / "sections" / "s02.notes.json",
        json.dumps(
            {
                "corrections": [
                    {
                        "anchor": "P1:p2",
                        "as_written": HOSTILE,
                        "corrected": "$\\|v\\|$ и | тест",
                        "reason": "r",
                    }
                ],
                "conflicts": [],
                "editorial": [],
            },
            ensure_ascii=False,
        ),
    )
    master = assemble.assemble_master(
        ctx, make_outline(), make_coverage(), make_sources(), today=TODAY
    )
    report = render_document(master, settings=ctx.settings, out_dir=ctx.topic_dir, engine="xelatex")
    assert report.ok, report.errors
    assert report.pdf == ctx.topic_dir / "master.pdf"
    assert report.checks is not None and report.checks.ok and report.checks.pages >= 3


def test_same_correction_in_two_sections_is_one_row(topic_ctx: BuildContext) -> None:
    ctx = topic_ctx
    correction = {"anchor": "S1:s6", "as_written": "a", "corrected": "b", "reason": "r"}
    conflict = {"topic": "T", "variants": [{"anchor": "S1:s6", "text": "x"}], "reason": "r"}
    for sid in ("s01-01", "s01-02"):
        write(
            ctx.synth_dir / "sections" / f"{sid}.notes.json",
            json.dumps(
                {"corrections": [correction], "conflicts": [conflict], "editorial": []},
                ensure_ascii=False,
            ),
        )
    notes = assemble.collect_notes(ctx, outline=make_outline())
    # the journal has no section column: one row; conflicts name their section: two
    assert len(notes["corrections"]) == 1 and notes["corrections"][0]["section"] == "s01-01"
    assert [c["section"] for c in notes["conflicts"]] == ["s01-01", "s01-02"]


def test_clean_block_keeps_code_fences_intact() -> None:
    text = (
        "Цена $5 и текст.\n\n```bash\n$ pip install h0lon\necho $HOME | wc\n```\n\n"
        "После <b> кода $y$."
    )
    cleaned = assemble.clean_block(text)
    assert "```bash\n$ pip install h0lon\necho $HOME | wc\n```" in cleaned  # code untouched
    assert r"Цена \$5 и текст." in cleaned and r"После \<b> кода $y$." in cleaned
    # an unbalanced fence cannot swallow the rest of the master
    broken = assemble.clean_block("до\n```python\nкод без закрытия")
    assert not any(ln.startswith("```") for ln in broken.splitlines())
