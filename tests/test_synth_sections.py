"""S2 of the synthesis: groups, inputs of a section, agent runs, retries, cache, heading files."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
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
from h0lon.synth import sections as sc
from h0lon.synth.common import load_blocks, parse_src_ids, read_meta, text_size
from h0lon.synth.model import BuildContext, Outline, OutlineSection

# See test_synth_outline.py: a copy of the owner's extracted topic is taken, never the original.
REAL_TOPIC = os.environ.get("H0LON_M2_REAL_TOPIC", "")

BANNER_RE = re.compile(r"<!-- =+ КОНТЕКСТ")
BLOCK_HEAD_RE = re.compile(r"^<!-- (\S+) (\S+) \[\[([^\]]+)\]\] -->$", re.MULTILINE)


# ---------------------------------------------------------------- data


def make_block(
    bid: str, md: str = "текст", type_: str = "paragraph", anchor: str | None = None
) -> Block:
    source = bid.split(".")[0]
    return Block(
        id=bid,
        source=source,
        type=type_,  # type: ignore[arg-type]
        anchor=anchor or f"{source}:p1",
        text=md,
        md=md,
    )


def blocks_of(*ids: str, size: int = 100) -> dict[str, Block]:
    return {bid: make_block(bid, f"Блок {bid}. " + "x" * size) for bid in ids}


def sec(
    id_: str, level: int, blocks: list[str], parent: str | None = None, *, summary: str = ""
) -> OutlineSection:
    return OutlineSection(
        id=id_, title=f"Раздел {id_}", level=level, blocks=blocks, parent=parent, summary=summary
    )


def write_extracted(topic: Path, blocks: dict[str, Block]) -> None:
    by_source: dict[str, list[Block]] = {}
    for b in blocks.values():
        by_source.setdefault(b.source, []).append(b)
    for sid, items in by_source.items():
        out = topic / "extracted" / sid
        out.mkdir(parents=True, exist_ok=True)
        with (out / "blocks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for b in items:
                fh.write(json.dumps(b.to_dict(), ensure_ascii=False) + "\n")


def world_blocks() -> dict[str, Block]:
    ids = [f"P1.b{i:03d}" for i in range(1, 13)] + [f"S1.b{i:03d}" for i in range(1, 7)]
    return {
        bid: make_block(
            bid,
            f"Содержимое блока {bid}. " * 4,
            "definition" if bid.endswith("3") else "paragraph",
            anchor=f"{bid[:2]}:p{int(bid[-3:]) // 3 + 1}",
        )
        for bid in ids
    }


def world_outline() -> Outline:
    return Outline(
        title="Тема",
        sections=[
            sec("s01", 1, [], summary="Основы"),
            sec("s01-01", 2, ["P1.b001", "P1.b002", "S1.b001"], "s01"),
            sec("s01-02", 2, ["P1.b003", "P1.b004", "P1.b005", "S1.b002"], "s01"),
            sec("s02", 1, [], summary="Методы."),
            sec("s02-01", 2, ["P1.b006", "P1.b007", "S1.b003"], "s02"),
            sec("s02-02", 2, ["P1.b008", "P1.b009", "S1.b004"], "s02"),
            sec("s03", 1, ["P1.b010", "P1.b011", "S1.b005"]),
            sec("s04", 1, ["P1.b012", "S1.b006"]),
        ],
    )


@pytest.fixture
def ctx(tmp_path: Path, settings: Settings) -> BuildContext:
    return BuildContext(topic_dir=tmp_path / "topic", settings=settings)


@pytest.fixture(autouse=True)
def three_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """The world outline has 6 sections with blocks: two per group → three groups."""
    monkeypatch.setattr(sc, "MAX_SECTIONS", 2)


# ---------------------------------------------------------------- fake agent


@dataclass
class Call:
    n: int
    task: str
    section_ids: list[str]
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
        assert stage == "sections" and tier == "strong"
        bundle = create_bundle(
            ctx.topic_dir / "runs",
            stage=stage,
            task=task,
            contract=contract,
            inputs=inputs,
            seed=seed,
        )
        restore_seed(bundle)
        ids = [f.path[:-3] for f in contract.files if f.path.endswith(".md")]
        with self._lock:
            call = Call(
                len(self.calls) + 1,
                task,
                ids,
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

    def calls_for(self, section_id: str) -> list[Call]:
        return [c for c in self.calls if section_id in c.section_ids]

    def groups(self) -> list[list[str]]:
        return sorted(c.section_ids for c in self.calls)


EMPTY_NOTES = {"corrections": [], "conflicts": [], "editorial": []}


def read_input_blocks(call: Call, section_id: str) -> list[tuple[str, str, str, str]]:
    """(id, type, anchor, md) of the section blocks in the bundle input (without the context)."""
    text = (call.bundle.inputs_dir / f"{section_id}.blocks.md").read_text("utf-8")
    text = BANNER_RE.split(text)[0]
    heads = list(BLOCK_HEAD_RE.finditer(text))
    items = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append((m.group(1), m.group(2), m.group(3), text[m.end() : end].strip()))
    return items


def good_writer(
    outline: Outline,
    *,
    omit: Callable[[Call, str], set[str]] | None = None,
    notes: dict[str, Any] | None = None,
    heading: Callable[[OutlineSection], str] | None = None,
) -> Callable[[Call], None]:
    """A well-behaved agent: heading, one paragraph per block with anchor and src comment."""
    by_id = {s.id: s for s in outline.sections}

    def writer(call: Call) -> None:
        out = call.bundle.out_dir
        for sid in call.section_ids:
            s = by_id[sid]
            skipped = omit(call, sid) if omit else set()
            lines = [heading(s) if heading else f"{'#' * s.level} {s.title} {{#sec:{s.id}}}", ""]
            for bid, _type, anchor, md in read_input_blocks(call, sid):
                if bid in skipped:
                    continue
                lines += [f"{md}[[{anchor}]]", f"<!-- src: {bid} -->", ""]
            (out / f"{sid}.md").write_text("\n".join(lines), encoding="utf-8")
            (out / f"{sid}.notes.json").write_text(
                json.dumps(notes or EMPTY_NOTES, ensure_ascii=False), encoding="utf-8"
            )

    return writer


@pytest.fixture
def world(ctx: BuildContext) -> tuple[Outline, dict[str, Block]]:
    return world_outline(), world_blocks()


# ---------------------------------------------------------------- plan_groups


def test_plan_groups_limits_the_number_of_sections() -> None:
    leaves = [sec(f"s{i:02d}", 1, [f"P1.b{i:03d}"]) for i in range(1, 11)]
    blocks = blocks_of(*(b for s in leaves for b in s.blocks))
    groups = sc.plan_groups(Outline("T", leaves), blocks)
    assert [[s.id for s in g] for g in groups] == [
        ["s01", "s02", "s03", "s04"],
        ["s05", "s06", "s07", "s08"],
        ["s09", "s10"],
    ]
    one = sc.plan_groups(Outline("T", leaves[:3]), blocks, max_sections=1)
    assert [len(g) for g in one] == [1, 1, 1]


def test_plan_groups_limits_characters_per_group() -> None:
    leaves = [sec(f"s{i:02d}", 1, [f"P1.b{i:03d}"]) for i in range(1, 6)]
    blocks = {b: make_block(b, "x" * 30) for s in leaves for b in s.blocks}
    groups = sc.plan_groups(Outline("T", leaves), blocks, max_chars=60)
    assert [[s.id for s in g] for g in groups] == [
        ["s01", "s02"],
        ["s03", "s04"],
        ["s05"],
    ]  # 60 fits
    tight = sc.plan_groups(Outline("T", leaves), blocks, max_chars=59)
    assert [len(g) for g in tight] == [1, 1, 1, 1, 1]


def test_plan_groups_gives_a_big_section_a_group_of_its_own() -> None:
    sizes = {"s01": 10, "s02": 100, "s03": 10, "s04": 10}
    leaves = [sec(sid, 1, [f"P1.b{i:03d}"]) for i, sid in enumerate(sizes, 1)]
    blocks = {
        f"P1.b{i:03d}": make_block(f"P1.b{i:03d}", "x" * n) for i, n in enumerate(sizes.values(), 1)
    }
    groups = sc.plan_groups(Outline("T", leaves), blocks, max_chars=60)
    assert [[s.id for s in g] for g in groups] == [["s01"], ["s02"], ["s03", "s04"]]


def test_plan_groups_takes_sections_with_blocks_in_order_across_chapters(world) -> None:
    outline, blocks = world
    groups = sc.plan_groups(outline, blocks, max_sections=4)
    # chapters with sections (no blocks) are not in any group; the group crosses chapters
    assert [[s.id for s in g] for g in groups] == [
        ["s01-01", "s01-02", "s02-01", "s02-02"],
        ["s03", "s04"],
    ]


def test_plan_groups_ignores_unknown_blocks_and_empty_sections() -> None:
    outline = Outline(
        "T",
        [sec("s01", 1, ["P9.b001"]), sec("s02", 1, []), sec("s03", 1, ["P1.b001", "P9.b002"])],
    )
    groups = sc.plan_groups(outline, blocks_of("P1.b001"))
    assert [[s.id for s in g] for g in groups] == [["s03"]]


def test_plan_groups_defaults_follow_decision_a4() -> None:
    import inspect

    params = inspect.signature(sc.plan_groups).parameters
    assert params["max_chars"].default == 60_000 and params["max_sections"].default == 4


# ---------------------------------------------------------------- inputs of a section


def test_section_inputs_follow_the_order_of_the_outline(ctx: BuildContext) -> None:
    blocks = {
        "P1.b001": make_block("P1.b001", "первый", "heading", "P1:p1"),
        "P1.b002": make_block("P1.b002", "второй $x$", "definition", "P1:p2"),
        "P1.b003": make_block("P1.b003", "третий", "formula", "P1:p3"),
    }
    section = sec("s01-01", 2, ["P1.b003", "P1.b001", "P1.b002", "P1.b001"], "s01")
    path = sc.prepare_section_inputs(ctx, section, blocks)
    assert path == ctx.synth_dir / "inputs" / "s01-01.blocks.md"
    text = path.read_text("utf-8")
    heads = BLOCK_HEAD_RE.findall(text)
    assert heads == [
        ("P1.b003", "formula", "P1:p3"),
        ("P1.b001", "heading", "P1:p1"),
        ("P1.b002", "definition", "P1:p2"),
    ]
    assert "<!-- P1.b002 definition [[P1:p2]] -->\nвторой $x$\n" in text
    assert "КОНТЕКСТ" not in text  # all neighbours are inside the section


def test_section_inputs_add_marked_neighbours_without_ids(ctx: BuildContext) -> None:
    blocks = {f"P1.b{i:03d}": make_block(f"P1.b{i:03d}", f"текст {i}") for i in range(1, 8)}
    blocks.update({f"S1.b{i:03d}": make_block(f"S1.b{i:03d}", f"слайд {i}") for i in range(1, 4)})
    # runs P1.b002..b003 and P1.b005 (P1.b004 lies between them) and the first slide
    section = sec("s02", 1, ["P1.b002", "P1.b003", "P1.b005", "S1.b001"])
    text = sc.prepare_section_inputs(ctx, section, blocks).read_text("utf-8")
    body, context = BANNER_RE.split(text)
    assert [h[0] for h in BLOCK_HEAD_RE.findall(body)] == [
        "P1.b002",
        "P1.b003",
        "P1.b005",
        "S1.b001",
    ]
    assert "не переноси" in text and "не указывай в src" in text
    notes = re.findall(r"<!-- контекст, не переносить: источник (\w+), ([^>]*?) \[\[", context)
    assert [n[0] for n in notes] == ["P1", "P1", "P1", "S1"]
    # P1.b001 before the first run; P1.b004 between the runs (once, with both relations);
    # P1.b006 after the second run; S1.b002 after the slide; nothing before S1.b001
    assert "текст 1" in context and "текст 6" in context and "слайд 2" in context
    assert context.count("текст 4") == 1
    assert "следует за блоком P1.b003; предшествует блоку P1.b005" in context
    assert "текст 7" not in context and "слайд 3" not in context
    # context blocks have no id headers: they cannot be counted as covered in this section
    assert not BLOCK_HEAD_RE.findall(context)
    for context_id in ("P1.b001", "P1.b004", "P1.b006", "S1.b002"):
        assert context_id not in context


def test_section_inputs_cut_a_long_neighbour_to_plain_text(ctx: BuildContext) -> None:
    long_md = "::: {.example}\n" + "очень длинно " * 400 + "\n:::"
    long_block = make_block("P1.b001", long_md, "example")
    long_block.text = "очень длинно " * 400  # the plain text has no fences
    blocks = {"P1.b001": long_block, "P1.b002": make_block("P1.b002", "нужный")}
    text = sc.prepare_section_inputs(ctx, sec("s01", 1, ["P1.b002"]), blocks).read_text("utf-8")
    context = BANNER_RE.split(text)[1]
    assert ":::" not in context and context.rstrip().endswith("…")
    assert len(context) < sc.CONTEXT_CHARS + 400


def test_section_inputs_cap_the_context(ctx: BuildContext) -> None:
    ids = [f"P1.b{i:03d}" for i in range(1, 200)]
    blocks = {b: make_block(b, f"т {b}") for b in ids}
    section = sec("s01", 1, ids[1::2])  # every second block: ~100 neighbours
    text = sc.prepare_section_inputs(ctx, section, blocks).read_text("utf-8")
    assert text.count("<!-- контекст, не переносить") == sc.MAX_CONTEXT


def test_section_inputs_skip_unknown_blocks(ctx: BuildContext) -> None:
    blocks = blocks_of("P1.b001")
    text = sc.prepare_section_inputs(ctx, sec("s01", 1, ["P9.b001", "P1.b001"]), blocks).read_text(
        "utf-8"
    )
    assert [h[0] for h in BLOCK_HEAD_RE.findall(text)] == ["P1.b001"]


def test_heading_only_text() -> None:
    assert sc.heading_only_text(sec("s01", 1, [])) == "# Раздел s01 {#sec:s01}\n"
    with_summary = sc.heading_only_text(sec("s01", 1, [], summary="Основы метрик"))
    assert with_summary == "# Раздел s01 {#sec:s01}\n\nОсновы метрик.\n"
    done = sc.heading_only_text(sec("s02-01", 2, [], summary="Уже с точкой."))
    assert done == "## Раздел s02-01 {#sec:s02-01}\n\nУже с точкой.\n"


# ---------------------------------------------------------------- the stage


def test_run_sections_writes_files_for_every_section(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    events: list[str] = []
    ctx.on_event = events.append
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and not result.cached and not result.warnings and not result.errors
    assert result.agent_runs == 3 and len(agent.calls) == 3
    assert agent.groups() == [["s01-01", "s01-02"], ["s02-01", "s02-02"], ["s03", "s04"]]
    d = ctx.synth_dir / "sections"
    for s in outline.sections:
        assert (d / f"{s.id}.md").is_file(), s.id
        assert (d / f"{s.id}.notes.json").is_file() == bool(s.blocks), s.id
    # chapters with sections: the heading and the summary, written by code, no agent
    assert (d / "s01.md").read_text("utf-8") == "# Раздел s01 {#sec:s01}\n\nОсновы.\n"
    assert (d / "s02.md").read_text("utf-8") == "# Раздел s02 {#sec:s02}\n\nМетоды.\n"
    assert not agent.calls_for("s01") and not agent.calls_for("s02")
    # every block id is in the src comments of its section
    for s in outline.sections:
        if s.blocks:
            assert parse_src_ids((d / f"{s.id}.md").read_text("utf-8")) == set(s.blocks)
    assert (d / "s03.md").read_text("utf-8").startswith("# Раздел s03 {#sec:s03}\n")
    assert json.loads((d / "s03.notes.json").read_text("utf-8")) == EMPTY_NOTES
    assert result.details["groups"] == 3 and result.details["sections_written_by_code"] == 2
    assert any("групп 3" in e for e in events)

    meta = read_meta(ctx, "sections")
    assert meta and meta["ok"] is True and meta["prompt"] == "sections@1.0" and meta["key"]
    assert meta["agent_runs"] == 3 and len(meta["groups"]) == 3


def test_agents_get_the_prompt_the_section_list_and_named_inputs(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    sc.run_sections(ctx, outline, blocks)
    first = next(c for c in agent.calls if c.section_ids == ["s01-01", "s01-02"])
    # the prompt names its inputs outline.md, glossary.md and <id>.blocks.md
    assert first.input_names == [
        "outline.md",
        "glossary.md",
        "s01-01.blocks.md",
        "s01-02.blocks.md",
    ]
    assert "# Задача: написать разделы мастер-конспекта из блоков источников" in first.task
    assert (
        "`s01-01` — раздел (level 2). Первая строка файла: `## Раздел s01-01 {#sec:s01-01}`."
        in first.task
    )
    assert "Глава: `s01` «Раздел s01»." in first.task and "Блоков: 3." in first.task
    assert "`inputs/s01-02.blocks.md`" in first.task
    assert "Обратная связь" not in first.task
    chapter_leaf = next(c for c in agent.calls if "s03" in c.section_ids)
    assert (
        "`s03` — глава (level 1). Первая строка файла: `# Раздел s03 {#sec:s03}`."
        in chapter_leaf.task
    )
    # outline.md has ids for links, glossary.md is the terms file
    staged_outline = (first.bundle.inputs_dir / "outline.md").read_text("utf-8")
    assert "(`s02-01`)" in staged_outline and "(`s04`)" in staged_outline
    terms = (ctx.synth_dir / "inputs" / "terms.md").read_text("utf-8")
    assert (first.bundle.inputs_dir / "glossary.md").read_text("utf-8") == terms
    # contract: text with the anchor and notes with a schema for every section of the group
    files = {f.path: f for f in first.bundle.contract.files}
    assert set(files) == {"s01-01.md", "s01-01.notes.json", "s01-02.md", "s01-02.notes.json"}
    assert files["s01-02.md"].required_text == ["{#sec:s01-02}"]
    assert files["s01-02.notes.json"].json_schema is not None


def test_groups_run_in_parallel_up_to_the_limit(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    barrier = threading.Barrier(2, timeout=20)
    inner = good_writer(outline)
    waited = []

    def writer(call: Call) -> None:
        if len(waited) < 2:  # the first two groups must be in the agent at the same time
            waited.append(call.n)
            barrier.wait()
        inner(call)

    agent = FakeAgent(writer)
    monkeypatch.setattr(sc, "run_agent", agent)
    ctx.settings.agents.parallel_runs = 2
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and len(agent.calls) == 3 and len(waited) == 2


def test_parallelism_never_exceeds_parallel_runs(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    inner = good_writer(outline)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def writer(call: Call) -> None:
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.15)
        inner(call)
        with lock:
            state["now"] -= 1

    for limit in (1, 2):
        shutil.rmtree(ctx.synth_dir, ignore_errors=True)
        state["max"] = 0
        monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
        ctx.settings.agents.parallel_runs = limit
        assert sc.run_sections(ctx, outline, blocks).ok
        assert state["max"] == limit


def test_cache_skips_unchanged_groups(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    assert sc.run_sections(ctx, outline, blocks).agent_runs == 3
    d = ctx.synth_dir / "sections"
    before = {p.name: p.read_bytes() for p in d.iterdir()}

    again = sc.run_sections(ctx, outline, blocks)
    assert again.ok and again.cached and again.agent_runs == 0 and len(agent.calls) == 3

    # a title changes in one group: only that group is written again
    outline.sections[5].title = "Другое название"  # s02-02
    part = sc.run_sections(ctx, outline, blocks)
    assert part.ok and not part.cached and part.agent_runs == 1 and len(agent.calls) == 4
    assert agent.calls[-1].section_ids == ["s02-01", "s02-02"]
    assert part.details["groups_cached"] == 2 and part.details["groups_run"] == 1
    after = {p.name: p.read_bytes() for p in d.iterdir()}
    for name in ("s01-01.md", "s01-02.md", "s03.md", "s04.md", "s03.notes.json"):
        assert after[name] == before[name], name
    assert after["s02-02.md"] != before["s02-02.md"]
    assert sc.run_sections(ctx, outline, blocks).cached  # and again from the cache

    # a block moves from one group to another: both groups are written again
    outline.sections[1].blocks.remove("P1.b002")
    outline.sections[7].blocks.append("P1.b002")  # s04
    moved = sc.run_sections(ctx, outline, blocks)
    assert moved.agent_runs == 2
    assert sorted(c.section_ids for c in agent.calls[-2:]) == [["s01-01", "s01-02"], ["s03", "s04"]]


def test_cache_depends_on_blocks_glossary_and_force(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    sc.run_sections(ctx, outline, blocks)
    # the text of one block changes → only its group (its neighbours of the same group included)
    changed = dict(blocks)
    changed["P1.b008"] = make_block("P1.b008", "новый текст блока", anchor="P1:p3")
    assert sc.run_sections(ctx, outline, changed).agent_runs == 1
    assert agent.calls[-1].section_ids == ["s02-01", "s02-02"]
    # the terms change → every group
    terms = ctx.synth_dir / "inputs" / "terms.md"
    terms.write_text(terms.read_text("utf-8") + "\n- **новый** — термин\n", "utf-8")
    assert sc.run_sections(ctx, outline, changed).agent_runs == 3
    # force
    ctx.force = True
    assert sc.run_sections(ctx, outline, changed).agent_runs == 3
    ctx.force = False
    assert sc.run_sections(ctx, outline, changed).cached


def test_a_missing_output_file_invalidates_its_group(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    sc.run_sections(ctx, outline, blocks)
    (ctx.synth_dir / "sections" / "s03.notes.json").unlink()
    result = sc.run_sections(ctx, outline, blocks)
    assert result.agent_runs == 1 and agent.calls[-1].section_ids == ["s03", "s04"]


def test_missing_src_ids_are_retried_with_feedback_and_a_seed(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world

    def omit(call: Call, sid: str) -> set[str]:
        return {"P1.b004", "S1.b002"} if not call.seed and sid == "s01-02" else set()

    inner = good_writer(outline, omit=omit)

    def writer(call: Call) -> None:
        if "s01-02" in call.section_ids and call.seed:
            # the previous files are in out/ already: the agent edits them in place
            assert (call.bundle.out_dir / "s01-02.md").is_file()
            assert "P1.b003" in (call.bundle.out_dir / "s01-02.md").read_text("utf-8")
        inner(call)

    agent = FakeAgent(writer)
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and not result.warnings and result.agent_runs == 4
    retry = [c for c in agent.calls if c.seed and "s01-02" in c.section_ids]
    assert len(retry) == 1
    call = retry[0]
    # only the section with gaps goes again; the section that was fine is not repeated
    assert call.section_ids == ["s01-02"]
    assert call.input_names == ["outline.md", "glossary.md", "s01-02.blocks.md"]
    assert set(call.seed) == {"s01-02.md", "s01-02.notes.json"}
    assert "# Обратная связь по предыдущей попытке" in call.task
    assert (
        "Раздел `s01-02`: в комментариях `<!-- src: … -->` нет блоков (2): P1.b004, S1.b002."
        in call.task
    )
    assert (
        "`s01-01` —"
        not in call.task.split("# Разделы этого задания")[1].split("# Входные файлы")[0]
    )
    assert "Обратная связь" not in agent.calls[0].task
    final = (ctx.synth_dir / "sections" / "s01-02.md").read_text("utf-8")
    assert parse_src_ids(final) == set(outline.sections[2].blocks)


def test_gaps_after_the_retries_are_a_warning_not_an_error(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    agent = FakeAgent(
        good_writer(outline, omit=lambda call, sid: {"S1.b003"} if sid == "s02-01" else set())
    )
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and not result.errors
    assert len(agent.calls_for("s02-01")) == 1 + sc.MAX_RETRIES
    assert result.agent_runs == 2 + 1 + sc.MAX_RETRIES  # two other groups + three runs of this one
    assert len(result.warnings) == 1
    assert (
        "`s02-01`" in result.warnings[0]
        and "S1.b003" in result.warnings[0]
        and "S4/S5" in result.warnings[0]
    )
    assert result.details["missing"] == {"s02-01": ["S1.b003"]}
    assert (ctx.synth_dir / "sections" / "s02-01.md").is_file()
    # the warning is remembered with the group record
    again = sc.run_sections(ctx, outline, blocks)
    assert (
        again.cached and again.warnings == result.warnings and again.details["groups_cached"] == 3
    )


def test_group_failure_is_an_error_and_other_groups_are_kept(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    inner = good_writer(outline)

    def writer(call: Call) -> None:
        if "s03" not in call.section_ids:
            inner(call)  # the last group gets nothing

    agent = FakeAgent(writer)
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, blocks)
    assert not result.ok and result.agent_runs == 3
    assert len(result.errors) == 1 and "группа 3/3 (s03, s04)" in result.errors[0]
    assert (ctx.synth_dir / "sections" / "s01-01.md").is_file()
    assert read_meta(ctx, "sections")["key"] is None  # type: ignore[index]

    # the next run writes only the group that failed
    good = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", good)
    again = sc.run_sections(ctx, outline, blocks)
    assert again.ok and again.agent_runs == 1 and good.groups() == [["s03", "s04"]]
    assert sc.run_sections(ctx, outline, blocks).cached


def test_an_exception_in_a_group_does_not_lose_the_others(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    inner = good_writer(outline)

    def writer(call: Call) -> None:
        if "s02-01" in call.section_ids:
            raise RuntimeError("сбой фейка")
        inner(call)

    monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
    result = sc.run_sections(ctx, outline, blocks)
    assert not result.ok and "RuntimeError" in result.errors[0] and "s02-01" in result.errors[0]
    assert (ctx.synth_dir / "sections" / "s03.md").is_file()


def test_a_failed_retry_keeps_the_previous_result(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    inner = good_writer(outline, omit=lambda call, sid: {"P1.b008"} if sid == "s02-02" else set())

    def writer(call: Call) -> None:
        if call.seed and "s02-02" in call.section_ids:
            (call.bundle.out_dir / "s02-02.md").unlink()  # the retry drops the seeded file
            return
        inner(call)

    monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok
    assert any("повторный прогон не удался" in w for w in result.warnings)
    assert any("P1.b008" in w and "S4/S5" in w for w in result.warnings)
    assert (ctx.synth_dir / "sections" / "s02-02.md").is_file()


def test_a_retry_that_loses_text_is_rejected(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world

    def writer(call: Call) -> None:
        by = {s.id: s for s in outline.sections}
        out = call.bundle.out_dir
        for sid in call.section_ids:
            s = by[sid]
            full = not call.seed or sid != "s01-01"
            lines = [f"{'#' * s.level} {s.title} {{#sec:{sid}}}", ""]
            for bid, _t, anchor, md in read_input_blocks(call, sid):
                if bid == "P1.b002" and not call.seed:
                    continue  # missing on the first run
                lines += [(md if full else "кратко") + f"[[{anchor}]]", f"<!-- src: {bid} -->", ""]
            (out / f"{sid}.md").write_text("\n".join(lines), encoding="utf-8")
            (out / f"{sid}.notes.json").write_text(json.dumps(EMPTY_NOTES), encoding="utf-8")

    monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok
    assert any("ухудшил результат" in w for w in result.warnings)
    text = (ctx.synth_dir / "sections" / "s01-01.md").read_text("utf-8")
    assert "кратко" not in text and "P1.b002" not in parse_src_ids(text)


def test_heading_level_is_fixed_by_code(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    wrong = good_writer(
        outline, heading=lambda s: f"{'#' * (3 - s.level)} {s.title} {{#sec:{s.id}}}"
    )
    monkeypatch.setattr(sc, "run_agent", FakeAgent(wrong))
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok
    d = ctx.synth_dir / "sections"
    assert (d / "s01-01.md").read_text("utf-8").startswith("## Раздел s01-01 {#sec:s01-01}\n")
    assert (d / "s03.md").read_text("utf-8").startswith("# Раздел s03 {#sec:s03}\n")
    assert sum("уровень заголовка исправлен" in w for w in result.warnings) == 6


def test_a_heading_missing_in_the_file_is_added_by_code() -> None:
    section = sec("s01-01", 2, ["P1.b001"], "s01")
    text, note = sc.fix_heading(
        "Текст без заголовка, см. {#sec:s01-01}.\n\n<!-- src: P1.b001 -->\n", section
    )
    assert text.startswith("## Раздел s01-01 {#sec:s01-01}\n\nТекст без заголовка")
    assert note and "добавлен кодом" in note
    ok = "<!-- служебный комментарий -->\n## Раздел s01-01 {#sec:s01-01}\n\nТекст.\n"
    assert sc.fix_heading(ok, section) == (ok, None)
    text, note = sc.fix_heading(
        "Лишнее вступление.\n\n### Раздел {#sec:s01-01 .unnumbered}\n", section
    )
    assert "\n## Раздел {#sec:s01-01 .unnumbered}" in text and "посторонний текст" in note
    # `s01` must not be taken for `s01-01` and the other way round
    other = sec("s01", 1, ["P1.b001"])
    text, note = sc.fix_heading("## Раздел {#sec:s01-01}\n\nТекст.\n", other)
    assert text.startswith("# Раздел s01 {#sec:s01}\n\n## Раздел {#sec:s01-01}")


def test_quality_warnings(monkeypatch, ctx: BuildContext, world) -> None:
    outline, blocks = world
    base = good_writer(outline)

    def writer(call: Call) -> None:
        base(call)
        path = call.bundle.out_dir / f"{call.section_ids[0]}.md"
        if call.section_ids[0] == "s01-01":
            path.write_text(
                path.read_text("utf-8") + "\nСм. [тут](#sec:s09).\n<!-- src: P1.b099 -->\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok
    assert any("ссылки на несуществующие разделы: #sec:s09" in w for w in result.warnings)
    assert any("не назначены" in w and "P1.b099" in w for w in result.warnings)


def test_a_much_shorter_text_is_reported(monkeypatch, ctx: BuildContext) -> None:
    big = {f"P1.b{i:03d}": make_block(f"P1.b{i:03d}", "длинный текст " * 100) for i in range(1, 4)}
    outline = Outline("T", [sec("s01", 1, list(big))])

    def writer(call: Call) -> None:
        out = call.bundle.out_dir
        (out / "s01.md").write_text(
            "# Раздел s01 {#sec:s01}\n\nКоротко.\n<!-- src: P1.b001 P1.b002 P1.b003 -->\n", "utf-8"
        )
        (out / "s01.notes.json").write_text(json.dumps(EMPTY_NOTES), encoding="utf-8")

    monkeypatch.setattr(sc, "run_agent", FakeAgent(writer))
    result = sc.run_sections(ctx, outline, big)
    assert result.ok and len(result.warnings) == 1 and "короче источников" in result.warnings[0]


def test_stale_files_are_removed_and_other_files_stay(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    d = ctx.synth_dir / "sections"
    d.mkdir(parents=True)
    for name in (
        "s09.md",
        "s09.notes.json",
        "s03-01.md",
        "s01.notes.json",
        "README.txt",
        "s01.bak",
    ):
        (d / name).write_text("старое", "utf-8")
    inputs = ctx.synth_dir / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "s09.blocks.md").write_text("старое", "utf-8")
    (inputs / "terms.md").write_text("# Термины\n", "utf-8")
    monkeypatch.setattr(sc, "run_agent", FakeAgent(good_writer(outline)))
    assert sc.run_sections(ctx, outline, blocks).ok
    names = {p.name for p in d.iterdir()}
    assert not names & {"s09.md", "s09.notes.json", "s03-01.md", "s01.notes.json"}
    assert {"README.txt", "s01.bak", "s01.md", "s03.notes.json"} <= names
    assert not (inputs / "s09.blocks.md").exists() and (inputs / "s03.blocks.md").exists()
    assert (inputs / "terms.md").read_text("utf-8") == "# Термины\n"


def test_unknown_blocks_in_the_outline_are_skipped_with_a_warning(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    outline.sections[1].blocks.append("P9.b001")
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and result.agent_runs == 3
    assert any("неизвестные блоки" in w and "P9.b001" in w for w in result.warnings)


def test_a_section_with_only_unknown_blocks_gets_a_heading_file(
    monkeypatch, ctx: BuildContext, world
) -> None:
    outline, blocks = world
    outline.sections.append(sec("s05", 1, ["P9.b001"], summary="Пусто"))
    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    assert sc.run_sections(ctx, outline, blocks).ok
    assert (ctx.synth_dir / "sections" / "s05.md").read_text(
        "utf-8"
    ) == "# Раздел s05 {#sec:s05}\n\nПусто.\n"
    assert not agent.calls_for("s05")


def test_an_outline_of_headings_only_needs_no_agent(monkeypatch, ctx: BuildContext) -> None:
    outline = Outline("T", [sec("s01", 1, [], summary="Введение")])
    agent = FakeAgent(lambda call: None)
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, {})
    assert result.ok and not agent.calls and result.agent_runs == 0
    assert (ctx.synth_dir / "sections" / "s01.md").is_file()
    assert sc.run_sections(ctx, outline, {}).cached


def test_notes_must_follow_the_schema(tmp_path: Path) -> None:
    contract = sc._contract([sec("s01-01", 2, ["P1.b001"], "s01")])
    out = tmp_path / "out"
    out.mkdir()
    (out / "s01-01.md").write_text(
        "## Раздел {#sec:s01-01}\n\nТекст.\n<!-- src: P1.b001 -->\n", "utf-8"
    )
    full = {
        "corrections": [
            {
                "anchor": "P1:p3",
                "block": "P1.b007",
                "as_written": "σ-кольцо",
                "corrected": "σ-алгебра",
                "reason": "описка",
            }
        ],
        "conflicts": [
            {
                "topic": "строгость",
                "variants": [
                    {"anchor": "P1:p3", "text": "P(X < x)"},
                    {"anchor": "S1:s12", "text": "P(X ≤ x)"},
                ],
                "resolution": None,
                "reason": "нельзя выбрать",
            }
        ],
        "editorial": [{"kind": "answer", "refers_to": "H1.b003", "text": "Ответ."}],
    }
    notes = out / "s01-01.notes.json"
    notes.write_text(json.dumps(full, ensure_ascii=False), "utf-8")
    assert validate_outputs(out, contract) == []
    notes.write_text(json.dumps(EMPTY_NOTES), "utf-8")
    assert validate_outputs(out, contract) == []
    bad = {
        "corrections": [{"anchor": "P1:p3"}],
        "conflicts": [],
        "editorial": [{"kind": "хм", "refers_to": "x", "text": "y"}],
    }
    notes.write_text(json.dumps(bad, ensure_ascii=False), "utf-8")
    problems = validate_outputs(out, contract)
    assert any("as_written" in p for p in problems) and any("kind" in p for p in problems)
    notes.write_text(json.dumps({"corrections": []}), "utf-8")
    assert any("conflicts" in p for p in validate_outputs(out, contract))
    # the text must contain the anchor of its section
    (out / "s01-01.md").write_text("## Раздел без якоря\n\nТекст про раздел.\n", "utf-8")
    assert any("{#sec:s01-01}" in p for p in validate_outputs(out, contract))


# ---------------------------------------------------------------- a real topic


@pytest.mark.skipif(not REAL_TOPIC, reason="H0LON_M2_REAL_TOPIC не задан")
def test_sections_on_a_copy_of_the_real_topic(
    monkeypatch, tmp_path: Path, settings: Settings
) -> None:
    from h0lon.sources.ingest import list_sources
    from h0lon.synth import outline as ol

    topic_dir = tmp_path / "real"
    shutil.copytree(REAL_TOPIC, topic_dir, ignore=shutil.ignore_patterns("runs", "synthesis"))
    sources = list_sources(topic_dir)
    blocks = load_blocks(topic_dir, [s.id for s in sources])
    ctx = BuildContext(topic_dir=topic_dir, settings=settings)
    ol.prepare_inputs(ctx, blocks, sources)

    # an outline of consecutive blocks: chapters of three sections of ~25 blocks each
    ids = [b for b in blocks if blocks[b].type != "admin"]
    leaves = [ids[i : i + 25] for i in range(0, len(ids), 25)]
    sections: list[OutlineSection] = []
    for c, start in enumerate(range(0, len(leaves), 3), 1):
        sections.append(OutlineSection(f"s{c:02d}", f"Глава {c}", 1, "Кратко", []))
        for k, chunk in enumerate(leaves[start : start + 3], 1):
            sections.append(
                OutlineSection(f"s{c:02d}-{k:02d}", f"Раздел {c}.{k}", 2, "", chunk, f"s{c:02d}")
            )
    outline = Outline("Метрические методы классификации", sections)
    monkeypatch.setattr(sc, "MAX_SECTIONS", 4)  # the autouse fixture made it 2

    groups = sc.plan_groups(outline, blocks)
    leaf_sections = [s for s in sections if s.blocks]
    assert [s for g in groups for s in g] == leaf_sections
    assert all(len(g) <= 4 for g in groups)
    assert all(sum(sc.section_size(s, blocks) for s in g) <= 60_000 or len(g) == 1 for g in groups)

    agent = FakeAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and not result.errors, result.errors
    assert result.agent_runs == len(groups)
    d = ctx.synth_dir / "sections"
    written = [d / f"{s.id}.md" for s in sections]
    assert all(p.is_file() for p in written)
    covered = set()
    for s in leaf_sections:
        text = (d / f"{s.id}.md").read_text("utf-8")
        assert text.startswith(f"## {s.title} {{#sec:{s.id}}}")
        covered |= parse_src_ids(text)
        assert text_size(text) > 0
    assert covered == set(ids)
    assert sc.run_sections(ctx, outline, blocks).cached
