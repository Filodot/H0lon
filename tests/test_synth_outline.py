"""S1 of the synthesis: inputs of the agent, checks of the outline, retries, repairs, cache."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from h0lon.agents import RunResult, Usage, create_bundle
from h0lon.agents.runner import restore_seed
from h0lon.agents.validate import validate_outputs
from h0lon.config import Settings
from h0lon.extract.model import Block
from h0lon.sources.models import SourceRecord
from h0lon.synth import outline as ol
from h0lon.synth.common import load_blocks, read_meta
from h0lon.synth.model import BuildContext, Outline, OutlineSection

# The extracted topic of the owner's end-to-end check (a copy is taken, the original is never
# touched). Read at import: the `settings` fixture clears H0LON_* variables.
REAL_TOPIC = os.environ.get("H0LON_M2_REAL_TOPIC", "")

SUMMARY_P1 = """<!-- h0lon: summary@1.0, агент claude (claude-sonnet-5-5), 2026-10-04 -->

## Аннотация

Конспект по методу k ближайших соседей.

## Оглавление

- Метод k ближайших соседей [[P1:p1]]

## Термины и обозначения

- **kNN** — метод k ближайших соседей [[P1:p1]]
- $\\rho$ — метрика [[P1:p2]]
"""

SUMMARY_S1 = """<!-- h0lon: summary@1.0 -->

## Аннотация

Слайды.

## Оглавление

- Титульный слайд [[S1:s1]]
"""

LONG_TEXT = "Очень длинный абзац про расстояния. " * 12


def blk(
    type_: str, anchor: str, text: str, md: str | None = None, title: str | None = None
) -> dict:
    return {
        "type": type_,
        "anchor": anchor,
        "text": text,
        "md": text if md is None else md,
        "title": title,
    }


SOURCES: list[tuple[str, str, str, str | None, list[dict]]] = [
    (
        "P1",
        "pdf-text",
        "Лекция 3 (конспект)",
        SUMMARY_P1,
        [
            blk("heading", "P1:p1", "Метод k ближайших соседей", "### Метод k ближайших соседей"),
            blk("paragraph", "P1:p1", "Пусть задана выборка."),
            blk("definition", "P1:p2", "Метрика — функция расстояния."),
            blk("formula", "P1:p2", "$$\\rho(a,b)=|a-b|$$"),
            blk("admin", "P1:p3", "Дедлайн сдачи 10 октября"),
            blk("paragraph", "P1:p3", "Взвешенный kNN учитывает расстояния."),
            blk("example", "P1:p4", "Пример классификации точки."),
        ],
    ),
    (
        "S1",
        "slides",
        "Слайды",
        SUMMARY_S1,
        [
            blk("paragraph", "S1:s1", "Титульный слайд"),
            blk("list", "S1:s2", "kNN — это просто", "- kNN — это просто"),
            blk("definition", "S1:s3", "Метрика — расстояние."),
            blk("paragraph", "S1:s4", "Выводы лекции."),
        ],
    ),
    (
        "W1",
        "web",
        "Статья",
        None,
        [
            blk("heading", "W1:§0", "Статья", "# Статья {.source-title}", "Статья"),
            blk("paragraph", "W1:§1", LONG_TEXT.strip() + "\nвторая строка"),
            blk("figure", "W1:§1", "", "![](https://example.org/y.png)"),
        ],
    ),
]


def write_topic(tmp_path: Path, spec: list | None = None) -> tuple[Path, list[SourceRecord]]:
    """A topic with extracted/<ID>/{blocks.jsonl, summary.md}; returns (dir, source records)."""
    topic = tmp_path / "topic"
    records: list[SourceRecord] = []
    for sid, kind, title, summary, blocks in spec or SOURCES:
        out = topic / "extracted" / sid
        out.mkdir(parents=True, exist_ok=True)
        with (out / "blocks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i, b in enumerate(blocks, 1):
                row = {"id": f"{sid}.b{i:03d}", "source": sid, **b}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if summary is not None:
            (out / "summary.md").write_text(summary, encoding="utf-8")
        records.append(
            SourceRecord(
                id=sid,
                kind=kind,
                title=title,
                added="2026-10-04T00:00:00Z",  # type: ignore[arg-type]
            )
        )
    return topic, records


@pytest.fixture
def topic(tmp_path: Path) -> tuple[Path, list[SourceRecord], dict[str, Block]]:
    path, sources = write_topic(tmp_path)
    return path, sources, load_blocks(path, [s.id for s in sources])


@pytest.fixture
def ctx(tmp_path: Path, settings: Settings) -> BuildContext:
    return BuildContext(topic_dir=tmp_path / "topic", settings=settings)


def good_outline() -> dict[str, Any]:
    return {
        "title": "Метрические методы",
        "sections": [
            {
                "id": "s01",
                "title": "Метрики и kNN",
                "level": 1,
                "summary": "Основы",
                "blocks": [],
                "parent": None,
            },
            {
                "id": "s01-01",
                "title": "Определения",
                "level": 2,
                "summary": "Метрика",
                "blocks": ["P1.b001", "P1.b002", "P1.b003", "S1.b003"],
                "parent": "s01",
            },
            {
                "id": "s01-02",
                "title": "Примеры",
                "level": 2,
                "summary": "",
                "blocks": ["P1.b004", "P1.b006", "P1.b007", "S1.b002", "S1.b004"],
                "parent": "s01",
            },
            {
                "id": "s02",
                "title": "Из статьи",
                "level": 1,
                "summary": "Википедия",
                "blocks": ["W1.b001", "W1.b002", "W1.b003"],
                "parent": None,
            },
        ],
        "unassigned": [
            {"block": "P1.b005", "reason": "дедлайн (служебный блок)"},
            {"block": "S1.b001", "reason": "титульный слайд"},
        ],
        "conflict_hints": [
            {"topic": "определение метрики", "blocks": ["P1.b003", "S1.b003"], "note": "по-разному"}
        ],
    }


# ---------------------------------------------------------------- fake agent


@dataclass
class Call:
    n: int
    stage: str
    task: str
    input_names: list[str]
    seed: dict[str, Path]
    bundle: Any
    extra: dict[str, Any] = field(default_factory=dict)


class FakeAgent:
    """Stands in for `run_agent`: a real bundle, files written by `writer`, runner-like checks."""

    def __init__(self, writer: Callable[[Call], None]) -> None:
        self.writer = writer
        self.calls: list[Call] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        ctx: BuildContext,
        *,
        stage: str,
        task: str,
        contract: Any,
        inputs: Any = (),
        seed: Any = None,
        tier: str = "strong",
    ) -> RunResult:
        bundle = create_bundle(
            ctx.topic_dir / "runs",
            stage=stage,
            task=task,
            contract=contract,
            inputs=inputs,
            seed=seed,
        )
        restore_seed(bundle)
        with self._lock:
            call = Call(
                len(self.calls) + 1,
                stage,
                task,
                [Path(p).name for p in inputs],
                dict(seed or {}),
                bundle,
            )
            self.calls.append(call)
        self.writer(call)
        problems = validate_outputs(bundle.out_dir, contract)
        return RunResult(
            ok=not problems,
            bundle=bundle,
            backend_used="fake",
            attempts=[],
            usage_total=Usage(),
            final_text="",
            problems=problems,
        )


def write_outline(data: dict[str, Any], md: str = "# Структура\n\n1. Метрики и kNN — основы.\n"):
    def writer(call: Call) -> None:
        out = call.bundle.out_dir
        (out / "outline.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        (out / "outline.md").write_text(md, encoding="utf-8")

    return writer


def scripted(*writers: Callable[[Call], None]) -> FakeAgent:
    """One writer per call (the last one repeats)."""
    return FakeAgent(lambda call: writers[min(call.n, len(writers)) - 1](call))


def sec(id_: str, level: int, blocks: list[str], parent: str | None = None) -> OutlineSection:
    return OutlineSection(id=id_, title=f"Раздел {id_}", level=level, blocks=blocks, parent=parent)


def make_block(bid: str, text: str = "текст", type_: str = "paragraph") -> Block:
    return Block(
        id=bid, source=bid.split(".")[0], type=type_, anchor=f"{bid[:2]}:p1", text=text, md=text
    )  # type: ignore[arg-type]


# ---------------------------------------------------------------- inputs


def test_prepare_inputs_writes_three_files(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    paths = ol.prepare_inputs(ctx, blocks, sources)
    root = ctx.synth_dir / "inputs"
    assert paths == {
        "summaries": root / "summaries.md",
        "blocks_index": root / "blocks_index.md",
        "terms": root / "terms.md",
    }
    assert all(p.is_file() for p in paths.values())


def test_summaries_md_structure(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    text = ol.prepare_inputs(ctx, blocks, sources)["summaries"].read_text("utf-8")
    headers = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert headers == [
        "P1 — Лекция 3 (конспект) (pdf-text)",
        "S1 — Слайды (slides)",
        "W1 — Статья (web)",
    ]
    assert "h0lon: summary@" not in text  # the service comment is dropped
    assert "Конспект по методу k ближайших соседей." in text
    # the headings of summary.md go one level deeper under the source header
    assert "### Аннотация" in text and "### Термины и обозначения" in text
    assert not re.search(r"^## (Аннотация|Оглавление|Термины)", text, flags=re.MULTILINE)


def test_summaries_md_falls_back_to_block_headings(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    text = ol.prepare_inputs(ctx, blocks, sources)["summaries"].read_text("utf-8")
    w1 = text.split("## W1 — Статья (web)")[1]
    assert "summary.md нет" in w1
    assert "- Статья [[W1:§0]]" in w1


def test_blocks_index_one_line_per_block(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    lines = ol.prepare_inputs(ctx, blocks, sources)["blocks_index"].read_text("utf-8").splitlines()
    assert len(lines) == len(blocks) == 14
    pattern = re.compile(r"^[A-Z]\d+\.b\d{3} \[[a-z-]+\] \([^)\s]+\) \S.*$")
    assert all(pattern.match(line) for line in lines), lines
    assert [line.split()[0] for line in lines] == list(blocks)  # sources in topic order
    assert "P1.b003 [definition] (P1:p2) Метрика — функция расстояния." in lines


def test_blocks_index_marks_admin_cuts_and_flattens_text(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    lines = ol.prepare_inputs(ctx, blocks, sources)["blocks_index"].read_text("utf-8").splitlines()
    by_id = {line.split()[0]: line for line in lines}
    assert by_id["P1.b005"] == "P1.b005 [admin] (P1:p3) (служебный блок) Дедлайн сдачи 10 октября"
    long_line = by_id["W1.b002"]
    text = long_line.split(") ", 1)[1]
    assert len(text) == ol.INDEX_TEXT_CHARS + 1 and text.endswith("…")
    assert "\n" not in text and "вторая строка" not in text
    assert by_id["W1.b003"].endswith("![](https://example.org/y.png)")  # no text: the Markdown


def test_terms_md_collects_terms_with_source_headers(topic, ctx: BuildContext) -> None:
    _, sources, blocks = topic
    text = ol.prepare_inputs(ctx, blocks, sources)["terms"].read_text("utf-8")
    assert "## P1 — Лекция 3 (конспект) (pdf-text)" in text
    assert "- **kNN** — метод k ближайших соседей [[P1:p1]]" in text
    assert "## S1" not in text and "## W1" not in text  # no terms section there
    assert "Аннотация" not in text


def test_terms_md_without_any_terms(tmp_path: Path, ctx: BuildContext) -> None:
    spec = [(sid, kind, title, None, blocks) for sid, kind, title, _, blocks in SOURCES]
    path, sources = write_topic(tmp_path, spec)
    blocks = load_blocks(path, [s.id for s in sources])
    text = ol.prepare_inputs(ctx, blocks, sources)["terms"].read_text("utf-8")
    assert "не найдено" in text


def test_ensure_terms_builds_the_file_when_missing(topic, ctx: BuildContext) -> None:
    _, _, blocks = topic
    path = ol.ensure_terms(ctx, blocks)
    assert path == ctx.synth_dir / "inputs" / "terms.md"
    assert "**kNN**" in path.read_text("utf-8")


@pytest.mark.skipif(not REAL_TOPIC, reason="H0LON_M2_REAL_TOPIC не задан")
def test_prepare_inputs_on_a_copy_of_the_real_topic(tmp_path: Path, settings: Settings) -> None:
    from h0lon.sources.ingest import list_sources

    topic_dir = tmp_path / "real"
    shutil.copytree(REAL_TOPIC, topic_dir, ignore=shutil.ignore_patterns("runs", "synthesis"))
    sources = list_sources(topic_dir)
    blocks = load_blocks(topic_dir, [s.id for s in sources])
    ctx = BuildContext(topic_dir=topic_dir, settings=settings)
    paths = ol.prepare_inputs(ctx, blocks, sources)
    index = paths["blocks_index"].read_text("utf-8").splitlines()
    assert len(index) == len(blocks) > 100
    assert all(re.match(r"^[A-Z]\d+\.b\d{3} \[[a-z-]+\] \(\S+\) \S", line) for line in index)
    summaries = paths["summaries"].read_text("utf-8")
    for s in sources:
        assert f"## {s.id} — " in summaries
    assert "h0lon: summary@" not in summaries
    assert "Термины и обозначения" in paths["terms"].read_text("utf-8")


# ---------------------------------------------------------------- checks


def kinds(check: ol.OutlineCheck) -> list[str]:
    return sorted({p.kind for p in check.problems})


def test_check_accepts_a_valid_outline(topic) -> None:
    _, _, blocks = topic
    check = ol.check_outline(ol.parse_outline(good_outline()), blocks)
    assert check.ok
    assert check.missing == []


def test_check_does_not_require_admin_blocks(topic) -> None:
    _, _, blocks = topic
    data = good_outline()
    data["unassigned"] = [u for u in data["unassigned"] if u["block"] != "P1.b005"]
    assert not ol.check_outline(ol.parse_outline(data), blocks).problems


def mutate(fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data = copy.deepcopy(good_outline())
    fn(data)
    return data


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: d["sections"][2].update(id="s01-01"), "structure"),  # duplicate id
        (lambda d: d["sections"][1].update(id="s01-1"), "structure"),  # bad id format
        (lambda d: d["sections"][1].update(parent=None), "structure"),  # level 2 without parent
        (lambda d: d["sections"][1].update(parent="s09"), "structure"),  # parent does not exist
        (lambda d: d["sections"][1].update(parent="s02", id="s01-01"), "structure"),  # id prefix
        (lambda d: d["sections"][0].update(parent="s02"), "structure"),  # chapter with a parent
        (lambda d: d["sections"][0].update(blocks=["P1.b001"]), "structure"),  # chapter with kids
        (lambda d: d["sections"][1]["blocks"].append("P9.b001"), "unknown"),
        (lambda d: d["sections"][1]["blocks"].append("W1.b001"), "duplicate"),
        (lambda d: d["sections"][1]["blocks"].append("P1.b001"), "duplicate"),
        (lambda d: d["sections"][1]["blocks"].remove("P1.b001"), "missing"),
        (lambda d: d["unassigned"].append({"block": "P1.b001", "reason": "x"}), "both"),
        (lambda d: d["unassigned"][1].update(reason=""), "reason"),
        (lambda d: d["unassigned"].append({"block": "X1.b001", "reason": "x"}), "unknown"),
    ],
)
def test_check_finds_violations(topic, change, expected: str) -> None:
    _, _, blocks = topic
    check = ol.check_outline(ol.parse_outline(mutate(change)), blocks)
    assert expected in kinds(check), [p.text for p in check.problems]


def test_check_reports_missing_blocks_in_source_order(topic) -> None:
    _, _, blocks = topic
    data = good_outline()
    data["sections"][1]["blocks"] = []
    data["sections"][2]["blocks"].remove("S1.b002")
    check = ol.check_outline(ol.parse_outline(data), blocks)
    assert check.missing == ["P1.b001", "P1.b002", "P1.b003", "S1.b002", "S1.b003"]
    text = next(p.text for p in check.problems if p.kind == "missing")
    assert "P1.b001, P1.b002" in text and "(5)" in text


def test_check_empty_outline_has_a_structure_problem(topic) -> None:
    _, _, blocks = topic
    outline = Outline(title="T", sections=[sec("s01", 1, [])])
    assert "structure" in kinds(ol.check_outline(outline, blocks))


def test_normalize_puts_children_after_their_chapter_and_drops_empty_leaves() -> None:
    outline = Outline(
        title="T",
        sections=[
            sec("s01", 1, []),
            sec("s02", 1, []),
            sec("s01-01", 2, ["P1.b001"], "s01"),
            sec("s02-01", 2, ["P1.b002"], "s02"),
            sec("s01-02", 2, [], "s01"),  # empty leaf
            sec("s03", 1, []),  # empty chapter without sections
        ],
    )
    warnings = ol.normalize_outline(outline)
    assert [s.id for s in outline.sections] == ["s01", "s01-01", "s02", "s02-01"]
    assert len(warnings) == 2 and "s01-02" in warnings[1] and "s03" in warnings[1]


def test_normalize_removes_a_chapter_whose_last_section_was_empty() -> None:
    outline = Outline(title="T", sections=[sec("s01", 1, []), sec("s01-01", 2, [], "s01")])
    ol.normalize_outline(outline)
    assert outline.sections == []


def test_repair_drops_unknown_and_duplicates_and_conflicts(topic) -> None:
    _, _, blocks = topic
    data = good_outline()
    data["sections"][1]["blocks"] += ["P9.b001", "W1.b001"]  # unknown; duplicate of s02
    data["unassigned"][1]["reason"] = ""  # S1.b001 without a reason
    data["unassigned"] += [
        {"block": "P1.b001", "reason": "лишнее"},  # also assigned to a section
        {"block": "P1.b005", "reason": "повтор"},  # listed twice
    ]
    outline = ol.parse_outline(data)
    warnings = ol.repair_outline(outline, blocks)
    assert "P9.b001" not in outline.sections[1].blocks
    assert "W1.b001" in outline.sections[1].blocks  # the first section keeps it
    assert outline.sections[3].blocks == ["W1.b002", "W1.b003"]
    assert outline.unassigned == [
        {"block": "P1.b005", "reason": "дедлайн (служебный блок)"},
        {"block": "S1.b001", "reason": "причина не указана"},
    ]
    assert not ol.check_outline(outline, blocks).problems
    assert len(warnings) == 4


def test_autofill_puts_a_block_next_to_its_nearest_neighbour(topic) -> None:
    _, _, blocks = topic
    data = good_outline()
    data["sections"][1]["blocks"].remove("P1.b002")  # between P1.b001 and P1.b003, both in s01-01
    data["sections"][2]["blocks"].remove("P1.b006")  # P1.b004 before, P1.b007 after (s01-02)
    outline = ol.parse_outline(data)
    missing = ol.missing_blocks(outline, blocks)
    assert missing == ["P1.b002", "P1.b006"]
    warnings = ol.autofill_missing(outline, blocks, missing)
    assert outline.sections[1].blocks == ["P1.b001", "P1.b002", "P1.b003", "S1.b003"]
    assert outline.sections[2].blocks == ["P1.b004", "P1.b006", "P1.b007", "S1.b002", "S1.b004"]
    assert len(warnings) == 1 and "P1.b002 → `s01-01`" in warnings[0]


def test_autofill_uses_the_next_block_when_there_is_no_previous_one(topic) -> None:
    _, _, blocks = topic
    data = good_outline()
    data["sections"][3]["blocks"].remove("W1.b001")
    outline = ol.parse_outline(data)
    ol.autofill_missing(outline, blocks, ["W1.b001"])
    assert outline.sections[3].blocks == ["W1.b001", "W1.b002", "W1.b003"]


def test_autofill_limit_is_few_blocks(topic) -> None:
    _, _, blocks = topic
    assert ol.autofill_limit(blocks) == ol.AUTOFILL_MIN
    many = {f"P1.b{i:03d}": make_block(f"P1.b{i:03d}") for i in range(1, 1001)}
    assert ol.autofill_limit(many) == 30


# ---------------------------------------------------------------- the stage


def test_run_outline_success_writes_files_and_meta(topic, ctx: BuildContext, monkeypatch) -> None:
    _, sources, blocks = topic
    agent = scripted(write_outline(good_outline(), "# Мой список структуры\n\nтекст агента\n"))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and not result.cached and result.agent_runs == 1 and not result.warnings
    assert result.details["sections"] == 4 and result.details["leaves"] == 3
    assert result.details["blocks_assigned"] == 12 and result.details["unassigned"] == 2

    saved = json.loads((ctx.synth_dir / "outline.json").read_text("utf-8"))
    assert saved["title"] == "Метрические методы"
    assert [s["id"] for s in saved["sections"]] == ["s01", "s01-01", "s01-02", "s02"]
    # outline.md is always rendered by code (with section ids for S3–S5); the agent's own
    # outline.md is kept next to it when the code changed nothing
    assert (ctx.synth_dir / "outline.agent.md").read_text("utf-8").startswith("# Мой список")
    assert "(`s01`)" in (ctx.synth_dir / "outline.md").read_text("utf-8")

    meta = read_meta(ctx, "outline")
    assert meta and meta["ok"] is True and meta["prompt"] == "outline@1.0"
    assert meta["agent_runs"] == 1 and meta["model"].startswith("claude:") and meta["key"]

    (call,) = agent.calls
    assert call.stage == "outline" and call.input_names == ["summaries.md", "blocks_index.md"]
    assert "# Задача: единая структура мастер-конспекта темы" in call.task
    assert (
        "`inputs/summaries.md`" in call.task
        and "блоков: 14, из них служебных `admin`: 1" in call.task
    )
    assert "Обратная связь" not in call.task
    # the bundle contract: outline.json with a schema and outline.md
    assert [f.path for f in call.bundle.contract.files] == ["outline.json", "outline.md"]
    assert call.bundle.contract.files[0].json_schema is not None


def test_run_outline_cache_force_and_changed_input(topic, ctx: BuildContext, monkeypatch) -> None:
    path, sources, blocks = topic
    agent = scripted(write_outline(good_outline()))
    monkeypatch.setattr(ol, "run_agent", agent)
    assert not ol.run_outline(ctx, blocks, sources).cached
    again = ol.run_outline(ctx, blocks, sources)
    assert again.ok and again.cached and again.agent_runs == 0 and len(agent.calls) == 1
    assert again.details["sections"] == 4

    ctx.force = True
    assert not ol.run_outline(ctx, blocks, sources).cached
    ctx.force = False
    assert len(agent.calls) == 2

    (path / "extracted" / "P1" / "summary.md").write_text(
        SUMMARY_P1 + "\n- **новое** — термин\n", "utf-8"
    )
    assert not ol.run_outline(ctx, blocks, sources).cached  # summaries.md changed → new key
    assert len(agent.calls) == 3
    assert ol.run_outline(ctx, blocks, sources).cached


def test_a_failed_run_never_looks_like_a_cache_hit(topic, ctx: BuildContext, monkeypatch) -> None:
    path, sources, blocks = topic
    good = scripted(write_outline(good_outline()))
    monkeypatch.setattr(ol, "run_agent", good)
    assert ol.run_outline(ctx, blocks, sources).ok
    summary = path / "extracted" / "P1" / "summary.md"
    summary.write_text(SUMMARY_P1 + "\n- **x** — y\n", "utf-8")
    monkeypatch.setattr(ol, "run_agent", FakeAgent(lambda call: None))  # writes nothing
    failed = ol.run_outline(ctx, blocks, sources)
    assert not failed.ok and failed.errors
    assert read_meta(ctx, "outline")["key"] is None  # type: ignore[index]
    summary.write_text(SUMMARY_P1, "utf-8")  # back to the first inputs
    monkeypatch.setattr(ol, "run_agent", good)
    assert not ol.run_outline(ctx, blocks, sources).cached
    assert len(good.calls) == 2


def test_run_outline_retries_with_feedback_and_a_seed(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    first = good_outline()
    first["sections"][1]["blocks"].remove("P1.b002")  # one block forgotten

    def fixer(call: Call) -> None:
        out = call.bundle.out_dir
        data = json.loads((out / "outline.json").read_text("utf-8"))  # restored from the seed
        data["sections"][1]["blocks"].insert(1, "P1.b002")
        (out / "outline.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert (out / "outline.md").is_file()

    agent = scripted(write_outline(first), fixer)
    monkeypatch.setattr(ol, "run_agent", agent)
    events: list[str] = []
    ctx.on_event = events.append
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.agent_runs == 2 and not result.warnings

    one, two = agent.calls
    assert "Обратная связь по предыдущей попытке" not in one.task
    assert "# Обратная связь по предыдущей попытке" in two.task
    assert "Не назначены блоки (1): P1.b002" in two.task
    assert one.bundle.root != two.bundle.root  # a new bundle
    assert set(two.seed) == {"outline.json", "outline.md"}
    saved = json.loads((ctx.synth_dir / "outline.json").read_text("utf-8"))
    assert saved["sections"][1]["blocks"][:3] == ["P1.b001", "P1.b002", "P1.b003"]
    assert any("попытка 2" in e for e in events)


def test_run_outline_adds_a_few_missing_blocks_by_code(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    bad = good_outline()
    bad["sections"][2]["blocks"].remove("S1.b004")  # always forgotten
    agent = scripted(write_outline(bad))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and len(agent.calls) == 1 + ol.MAX_RETRIES
    assert result.agent_runs == 3
    # the nearest neighbour of S1.b004 in its source is S1.b003, which is in s01-01
    assert any("S1.b004 → `s01-01`" in w for w in result.warnings)
    saved = json.loads((ctx.synth_dir / "outline.json").read_text("utf-8"))
    assert saved["sections"][1]["blocks"][-2:] == ["S1.b003", "S1.b004"]
    # the code changed the structure, so outline.md is rendered from outline.json
    md = (ctx.synth_dir / "outline.md").read_text("utf-8")
    assert md == ol.render_outline_md(ol.load_outline(ctx))
    assert not ol.check_outline(ol.load_outline(ctx), blocks).problems


def test_run_outline_fails_when_too_many_blocks_are_missing(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    bad = good_outline()
    bad["sections"][1]["blocks"] = []  # four blocks...
    bad["sections"][2]["blocks"] = ["P1.b004"]  # ...and four more
    agent = scripted(write_outline(bad))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert not result.ok and len(agent.calls) == 3
    assert "не назначены блоки" in result.errors[0] and "допустимо до 5" in result.errors[0]
    assert not (ctx.synth_dir / "outline.json").exists()
    assert read_meta(ctx, "outline")["ok"] is False  # type: ignore[index]


def test_run_outline_fails_on_structure_problems_after_retries(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    bad = good_outline()
    bad["sections"][2]["id"] = "s01-01"  # duplicate id
    agent = scripted(write_outline(bad))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert not result.ok and len(agent.calls) == 3
    assert "повторяется" in result.errors[0]
    assert "повторяется" in agent.calls[1].task  # it was part of the feedback
    assert not (ctx.synth_dir / "outline.json").exists()


def test_run_outline_keeps_the_better_attempt(topic, ctx: BuildContext, monkeypatch) -> None:
    _, sources, blocks = topic
    nearly = good_outline()
    nearly["sections"][2]["blocks"].remove("S1.b004")  # one block missing
    worse = good_outline()
    worse["sections"][2]["id"] = "s01-01"  # structure broken
    agent = scripted(write_outline(nearly), write_outline(worse), write_outline(worse))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and any("S1.b004" in w for w in result.warnings)


def test_run_outline_uses_the_first_result_when_a_retry_fails(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    nearly = good_outline()
    nearly["sections"][2]["blocks"].remove("S1.b004")  # one block missing

    def drops_the_seed(call: Call) -> None:
        for name in ("outline.json", "outline.md"):
            (call.bundle.out_dir / name).unlink()

    agent = scripted(write_outline(nearly), drops_the_seed)
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and len(agent.calls) == 2 and result.agent_runs == 2
    assert any("Повторный прогон агента не удался" in w for w in result.warnings)
    assert any("S1.b004" in w for w in result.warnings)


def test_run_outline_repairs_duplicates_after_retries(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    bad = good_outline()
    bad["sections"][2]["blocks"].append("W1.b002")  # also in s02
    agent = scripted(write_outline(bad))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and any("оставлены в первом" in w for w in result.warnings)
    saved = ol.load_outline(ctx)
    assert sum(s.blocks.count("W1.b002") for s in saved.sections) == 1


def test_run_outline_drops_empty_sections_and_fixes_order(
    topic, ctx: BuildContext, monkeypatch
) -> None:
    _, sources, blocks = topic
    data = good_outline()
    data["sections"].append(
        {
            "id": "s01-03",
            "title": "Пустой",
            "level": 2,
            "summary": "",
            "blocks": [],
            "parent": "s01",
        }
    )
    agent = scripted(write_outline(data))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and len(agent.calls) == 1
    assert [s.id for s in ol.load_outline(ctx).sections] == ["s01", "s01-01", "s01-02", "s02"]
    assert any("Удалены разделы без блоков" in w for w in result.warnings)
    assert "Пустой" not in (ctx.synth_dir / "outline.md").read_text("utf-8")


def test_run_outline_agent_failure_is_a_stage_error(topic, ctx: BuildContext, monkeypatch) -> None:
    _, sources, blocks = topic
    agent = FakeAgent(lambda call: None)
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, blocks, sources)
    assert not result.ok and len(agent.calls) == 1 and result.agent_runs == 1
    assert "Агент не вернул допустимую структуру" in result.errors[0]
    assert "outline.json" in result.errors[0]


def test_run_outline_without_blocks(ctx: BuildContext, monkeypatch) -> None:
    agent = scripted(write_outline(good_outline()))
    monkeypatch.setattr(ol, "run_agent", agent)
    result = ol.run_outline(ctx, {}, [])
    assert not result.ok and not agent.calls and "извлечение" in result.errors[0]


def test_contract_schema_checks_the_format_of_ids(tmp_path: Path) -> None:
    contract = ol._contract()
    out = tmp_path / "out"
    out.mkdir()
    (out / "outline.md").write_text("# Структура темы\n\n1. Глава\n", "utf-8")
    bad = good_outline()
    bad["sections"][1]["id"] = "chapter1"
    bad["sections"][1]["level"] = 3
    bad["sections"][2]["blocks"] = ["блок-7"]
    (out / "outline.json").write_text(json.dumps(bad, ensure_ascii=False), "utf-8")
    problems = validate_outputs(out, contract)
    assert any("шаблону" in p for p in problems) and any("level" in p for p in problems)
    (out / "outline.json").write_text(json.dumps(good_outline(), ensure_ascii=False), "utf-8")
    assert validate_outputs(out, contract) == []
    # unassigned and conflict_hints may be absent
    minimal = {k: v for k, v in good_outline().items() if k in ("title", "sections")}
    (out / "outline.json").write_text(json.dumps(minimal, ensure_ascii=False), "utf-8")
    assert validate_outputs(out, contract) == []


# ---------------------------------------------------------------- load and render


def test_load_outline_roundtrip_and_errors(ctx: BuildContext) -> None:
    with pytest.raises(FileNotFoundError, match="Структура темы не найдена"):
        ol.load_outline(ctx)
    ctx.synth_dir.mkdir(parents=True)
    (ctx.synth_dir / "outline.json").write_text("{не json", "utf-8")
    with pytest.raises(ValueError, match="некорректный JSON"):
        ol.load_outline(ctx)
    (ctx.synth_dir / "outline.json").write_text(
        json.dumps(good_outline(), ensure_ascii=False), "utf-8"
    )
    outline = ol.load_outline(ctx)
    assert isinstance(outline, Outline) and outline.title == "Метрические методы"
    assert outline.sections[1] == OutlineSection(
        id="s01-01",
        title="Определения",
        level=2,
        summary="Метрика",
        blocks=["P1.b001", "P1.b002", "P1.b003", "S1.b003"],
        parent="s01",
    )
    assert outline.unassigned[0] == {"block": "P1.b005", "reason": "дедлайн (служебный блок)"}
    assert outline.conflict_hints[0]["blocks"] == ["P1.b003", "S1.b003"]


def test_render_outline_md_has_ids_numbers_and_counts() -> None:
    md = ol.render_outline_md(ol.parse_outline(good_outline()))
    assert md.startswith("# Метрические методы")
    assert "- **1. Метрики и kNN (`s01`) — Основы. Блоков: 9**" in md
    assert "  - 1.1. Определения (`s01-01`) — Метрика. Блоков: 4" in md
    assert "  - 1.2. Примеры (`s01-02`). Блоков: 5" in md
    assert "- **2. Из статьи (`s02`) — Википедия. Блоков: 3**" in md
    assert "## Возможные расхождения источников" in md and "определение метрики" in md
    assert "- P1.b005 — дедлайн (служебный блок)" in md
