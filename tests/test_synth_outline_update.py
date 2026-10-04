"""Incremental update of the structure (S1 in update mode) and what it means for S2 and the build.

The agent is replaced by `FakeAgent` of test_synth_outline: it reads `inputs/outline.json` of the
bundle like the real one would and writes `out/outline.json`. S2 uses the fake agent of
test_synth_sections, the build uses the fakes of test_synth_build.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_synth_build import (  # noqa: F401 - `_git_identity` is an autouse fixture there
    Env,
    FakeExtract,
    FakeSynth,
    _git_identity,
    fake_render_master,
    git,
    make_topic,
    run_build,
)
from test_synth_outline import SOURCES, Call, FakeAgent, blk, good_outline, scripted, write_topic
from test_synth_sections import FakeAgent as SectionAgent
from test_synth_sections import good_writer, make_block, world_blocks, world_outline

from h0lon.config import Settings
from h0lon.extract.model import Block
from h0lon.sources.models import SourceRecord
from h0lon.synth import build as build_mod
from h0lon.synth import outline as ol
from h0lon.synth import sections as sc
from h0lon.synth.common import load_blocks, read_meta, write_json_atomic, write_meta
from h0lon.synth.model import BuildContext, Outline, OutlineSection, StageResult
from h0lon.workspace import load_topic, save_topic

SUMMARY_V1 = """<!-- h0lon: summary@1.0 -->

## Аннотация

Видеолекция.

## Оглавление

- Метрические методы [[V1:00:10]]

## Термины и обозначения

- **окно Парзена** — ядерная оценка плотности [[V1:05:00]]
"""
V1_BLOCKS = [
    blk("heading", "V1:00:10", "Метрические методы", "## Метрические методы"),
    blk("paragraph", "V1:00:10", "Лектор напоминает про метрику."),
    blk("definition", "V1:05:00", "Окно Парзена — ядерная оценка плотности."),
    blk("admin", "V1:09:00", "Запись остановлена"),
]
V1 = ("V1", "video", "Видеолекция", SUMMARY_V1, V1_BLOCKS)
UPDATE_TASK = "# Задача: встроить новые блоки в существующую структуру темы"
FULL_TASK = "# Задача: единая структура мастер-конспекта темы"


# ---------------------------------------------------------------- the topic and the agents


@pytest.fixture
def ctx(tmp_path: Path, settings: Settings) -> BuildContext:
    return BuildContext(topic_dir=tmp_path / "topic", settings=settings)


def blocks_of(path: Path, sources: list[SourceRecord]) -> dict[str, Block]:
    return load_blocks(path, [s.id for s in sources])


def old_outline_of(call: Call) -> dict[str, Any]:
    return json.loads((call.bundle.inputs_dir / "outline.json").read_text("utf-8"))


def put(out_dir: Path, data: dict[str, Any]) -> None:
    (out_dir / "outline.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def put_full(out_dir: Path, data: dict[str, Any]) -> None:
    """The two files the ordinary S1 has to write."""
    put(out_dir, data)
    (out_dir / "outline.md").write_text("# Структура\n\n1. Метрики.\n", encoding="utf-8")


def section(data: dict[str, Any], sid: str) -> dict[str, Any]:
    return next(s for s in data["sections"] if s["id"] == sid)


def updated(data: dict[str, Any]) -> dict[str, Any]:
    """The old outline with the new V1 blocks placed (what a good agent returns)."""
    out = copy.deepcopy(data)
    section(out, "s01-01")["blocks"].append("V1.b003")  # the definition of the Parzen window
    section(out, "s01-02")["blocks"] += ["V1.b001", "V1.b002"]
    out["unassigned"].append({"block": "V1.b004", "reason": "служебный блок"})
    return out


def good_update(call: Call) -> None:
    put(call.bundle.out_dir, updated(old_outline_of(call)))


def full_with_v1(call: Call) -> None:
    """What the ordinary S1 returns for the topic with V1."""
    put_full(call.bundle.out_dir, updated(good_outline()))


def dispatch(on_update: Callable[[Call], None], on_full: Callable[[Call], None] = full_with_v1):
    """A fake agent that tells the two prompts apart."""

    def writer(call: Call) -> None:
        (on_update if call.task.startswith(UPDATE_TASK) else on_full)(call)

    return FakeAgent(writer)


@pytest.fixture
def built(tmp_path: Path, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch):
    """A topic whose structure was built (P1, S1, W1), then V1 was added: (blocks, sources)."""
    path, sources = write_topic(tmp_path)
    first = scripted(lambda call: put_full(call.bundle.out_dir, good_outline()))
    monkeypatch.setattr(ol, "run_agent", first)
    assert ol.run_outline(ctx, blocks_of(path, sources), sources).ok
    assert len(first.calls) == 1
    path, sources = write_topic(tmp_path, [*SOURCES, V1])  # the new source is extracted
    return blocks_of(path, sources), sources


def agent_calls(monkeypatch: pytest.MonkeyPatch, agent: FakeAgent) -> FakeAgent:
    monkeypatch.setattr(ol, "run_agent", agent)
    return agent


# ---------------------------------------------------------------- the mode


def test_update_runs_the_update_prompt_on_the_new_blocks_only(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built
    agent = agent_calls(monkeypatch, dispatch(good_update))
    events: list[str] = []
    ctx.on_event = events.append
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and not result.cached and result.agent_runs == 1 and not result.warnings
    (call,) = agent.calls
    assert call.stage == "outline" and call.task.startswith(UPDATE_TASK)
    assert call.input_names == ["outline.json", "new_blocks_index.md", "summaries.md"]
    assert [f.path for f in call.bundle.contract.files] == ["outline.json"]  # no outline.md
    assert "разделов: 4, назначено блоков: 12" in call.task and "V1: 4" in call.task
    assert "блоков: 4, из них служебных `admin`: 1" in call.task
    assert any("обновление, новых блоков: 3" in e for e in events)

    index = (ctx.synth_dir / "inputs" / "new_blocks_index.md").read_text("utf-8").splitlines()
    assert [line.split()[0] for line in index] == ["V1.b001", "V1.b002", "V1.b003", "V1.b004"]
    assert index[3].endswith("(служебный блок) Запись остановлена")
    # the inputs of the agent: the old outline as it was, and the summaries of every source
    assert old_outline_of(call) == good_outline()
    assert "## V1 — Видеолекция (video)" in (call.bundle.inputs_dir / "summaries.md").read_text(
        "utf-8"
    )

    saved = ol.load_outline(ctx)
    assert section(saved.to_dict(), "s01-02")["blocks"][-2:] == ["V1.b001", "V1.b002"]
    assert not ol.check_outline(saved, blocks).problems
    assert (ctx.synth_dir / "outline.md").read_text("utf-8") == ol.render_outline_md(saved)
    meta = read_meta(ctx, "outline")
    assert meta["mode"] == "update" and meta["new_blocks"] == 3  # type: ignore[index]
    assert meta["update_prompt"] == "outline_update@1.0" and meta["prompt"] == "outline@1.0"  # type: ignore[index]
    assert result.details["mode"] == "update" and result.details["new_sections"] == []
    assert result.details["new_blocks"] == 3 and result.details["blocks_assigned"] == 12 + 3

    again = ol.run_outline(ctx, blocks, sources)  # the same inputs: the cache
    assert again.cached and again.agent_runs == 0 and again.details["mode"] == "cached"
    assert len(agent.calls) == 1


def test_update_adds_a_new_chapter_where_the_agent_puts_it(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def with_chapter(call: Call) -> None:
        data = old_outline_of(call)
        data["sections"].insert(
            3,  # between s01-02 and s02: after the first chapter
            {
                "id": "s03",
                "title": "Ядерные методы",
                "level": 1,
                "summary": "Окно Парзена",
                "blocks": ["V1.b001", "V1.b002", "V1.b003"],
                "parent": None,
            },
        )
        put(call.bundle.out_dir, data)

    agent_calls(monkeypatch, dispatch(with_chapter))
    result = ol.run_outline(ctx, blocks, sources)
    assert (
        result.ok
        and result.details["mode"] == "update"
        and result.details["new_sections"] == ["s03"]
    )
    ids = [s.id for s in ol.load_outline(ctx).sections]
    assert ids == ["s01", "s01-01", "s01-02", "s03", "s02"]  # old order intact, new in between


def test_update_puts_back_what_it_must_not_change(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def careless(call: Call) -> None:
        data = updated(old_outline_of(call))
        data["title"] = "Другое название темы"
        section(data, "s02")["summary"] = "Переписанное описание главы"  # no new block there
        section(data, "s01-02")["summary"] = "Теперь и про видео"  # got new blocks: allowed
        data["unassigned"] = [u for u in data["unassigned"] if u["block"] != "P1.b005"]
        data["conflict_hints"] = []
        put(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(careless))
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and len(agent.calls) == 1 and not result.warnings
    saved = ol.load_outline(ctx)
    assert saved.title == "Метрические методы"
    by_id = {s.id: s for s in saved.sections}
    assert by_id["s02"].summary == "Википедия"
    assert by_id["s01-02"].summary == "Теперь и про видео"
    assert {"block": "P1.b005", "reason": "дедлайн (служебный блок)"} in saved.unassigned
    assert saved.conflict_hints[0]["topic"] == "определение метрики"


def test_update_keeps_the_old_place_of_a_block_the_agent_repeated(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def repeats(call: Call) -> None:
        data = updated(old_outline_of(call))
        section(data, "s01-01")["blocks"].insert(0, "W1.b002")  # also lives in s02
        put(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(repeats))
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and len(agent.calls) == 1 and not result.warnings
    saved = ol.load_outline(ctx)
    assert "W1.b002" not in {s.id: s for s in saved.sections}["s01-01"].blocks
    assert "W1.b002" in {s.id: s for s in saved.sections}["s02"].blocks


# ---------------------------------------------------------------- retries and fallback


def test_update_retries_with_feedback_when_an_old_section_changes(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def renames(call: Call) -> None:
        data = updated(old_outline_of(call))
        section(data, "s01-01")["title"] = "Метрика и расстояния"
        data["sections"] = [s for s in data["sections"] if s["id"] != "s02"]  # and drops one
        put(call.bundle.out_dir, data)

    def fixes(call: Call) -> None:
        # the previous attempt is in out/ (the seed); the old outline is still in inputs/
        assert (call.bundle.out_dir / "outline.json").is_file()
        good_update(call)

    agent = agent_calls(
        monkeypatch,
        FakeAgent(lambda c: (renames if c.n == 1 else fixes)(c)),
    )
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.agent_runs == 2 and result.details["mode"] == "update"
    one, two = agent.calls
    assert "Обратная связь" not in one.task and "# Обратная связь по предыдущей попытке" in two.task
    assert "Удалены существующие разделы (1): `s02` «Из статьи»" in two.task
    assert "Изменены существующие разделы (1): `s01-01` (заголовок «Определения» →" in two.task
    assert two.task.startswith(UPDATE_TASK) and set(two.seed) == {"outline.json"}
    assert two.input_names == ["outline.json", "new_blocks_index.md", "summaries.md"]
    assert ol.load_outline(ctx).sections[1].title == "Определения"


def test_update_feedback_names_moved_and_reordered_blocks(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def shuffles(call: Call) -> None:
        data = updated(old_outline_of(call))
        section(data, "s01-01")["blocks"].remove("P1.b002")  # moved to the examples
        section(data, "s01-02")["blocks"].insert(0, "P1.b002")
        section(data, "s01-02")["blocks"].reverse()  # and the rest is turned round
        put(call.bundle.out_dir, data)

    agent = agent_calls(
        monkeypatch, FakeAgent(lambda c: (shuffles if c.n == 1 else good_update)(c))
    )
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.agent_runs == 2
    feedback = agent.calls[1].task
    assert "Блоки вышли из своих разделов (1): P1.b002 (был `s01-01`, стал `s01-02`)" in feedback
    assert "Изменён порядок старых блоков внутри разделов: `s01-02`" in feedback


def test_update_falls_back_to_the_ordinary_s1_after_failed_retries(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def breaks(call: Call) -> None:
        data = updated(old_outline_of(call))
        section(data, "s01-02")["id"] = "s01-09"  # an old section vanishes
        put(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(breaks))
    events: list[str] = []
    ctx.on_event = events.append
    result = ol.run_outline(ctx, blocks, sources)
    assert [c.task.startswith(UPDATE_TASK) for c in agent.calls] == [True, True, True, False]
    assert agent.calls[3].task.startswith(FULL_TASK)
    assert agent.calls[3].input_names == ["summaries.md", "blocks_index.md"]
    assert result.ok and result.agent_runs == 4 and result.details["mode"] == "full"
    assert result.details["update_failed"] is True
    assert result.warnings[0].startswith("Обновление структуры не удалось (")
    assert "структура пересобрана заново" in result.warnings[0]
    assert "Удалены существующие разделы" in result.warnings[0]
    assert any("Пересборка структуры" in e for e in events)
    assert not (ctx.synth_dir / "inputs" / "new_blocks_index.md").exists()
    assert not ol.check_outline(ol.load_outline(ctx), blocks).problems
    assert read_meta(ctx, "outline")["warnings"][0] == result.warnings[0]  # type: ignore[index]
    assert read_meta(ctx, "outline").get("mode") is None  # type: ignore[union-attr]


def test_update_falls_back_when_the_agent_gives_nothing(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built
    agent = agent_calls(monkeypatch, dispatch(lambda call: None))  # writes no file
    result = ol.run_outline(ctx, blocks, sources)
    assert [c.task[:30] for c in agent.calls] == [UPDATE_TASK[:30], FULL_TASK[:30]]
    assert result.ok and result.details["update_failed"] is True and result.agent_runs == 2
    assert "агент не вернул допустимую структуру" in result.warnings[0]


def test_update_fallback_failure_is_a_stage_error_and_keeps_the_old_outline(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built
    agent_calls(monkeypatch, FakeAgent(lambda call: None))
    result = ol.run_outline(ctx, blocks, sources)
    assert not result.ok and result.agent_runs == 2
    assert ol.load_outline(ctx).to_dict() == ol.parse_outline(good_outline()).to_dict()
    assert read_meta(ctx, "outline")["key"] is None  # type: ignore[index]


def test_update_adds_a_few_forgotten_blocks_by_code(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, sources = built

    def forgets(call: Call) -> None:
        data = updated(old_outline_of(call))
        section(data, "s01-02")["blocks"].remove("V1.b002")
        put(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(forgets))
    result = ol.run_outline(ctx, blocks, sources)
    assert len(agent.calls) == 1 + ol.MAX_RETRIES and all(
        c.task.startswith(UPDATE_TASK) for c in agent.calls
    )
    assert result.ok and result.details["mode"] == "update" and result.agent_runs == 3
    assert any("V1.b002 → `s01-02`" in w for w in result.warnings)  # next to V1.b001
    assert "V1.b002" in {s.id: s for s in ol.load_outline(ctx).sections}["s01-02"].blocks


def test_update_falls_back_when_too_many_new_blocks_are_left_out(
    tmp_path: Path, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sources = write_topic(tmp_path)
    agent_calls(monkeypatch, scripted(lambda call: put_full(call.bundle.out_dir, good_outline())))
    assert ol.run_outline(ctx, blocks_of(path, sources), sources).ok
    many = [blk("paragraph", f"V1:{i}", f"Абзац видео {i}.") for i in range(1, 9)]
    path, sources = write_topic(tmp_path, [*SOURCES, ("V1", "video", "Видео", None, many)])
    blocks = blocks_of(path, sources)

    def ignores(call: Call) -> None:
        put(call.bundle.out_dir, old_outline_of(call))  # nothing placed

    def full(call: Call) -> None:
        data = good_outline()
        data["sections"][3]["blocks"] += [f"V1.b{i:03d}" for i in range(1, 9)]
        put_full(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(ignores, full))
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.details["update_failed"] is True and result.agent_runs == 4
    assert "не назначены новые блоки (8; допустимо до 5)" in result.warnings[0]
    assert agent.calls[-1].task.startswith(FULL_TASK)


def test_update_warns_when_the_new_blocks_end_up_unassigned(
    tmp_path: Path, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sources = write_topic(tmp_path)
    agent_calls(monkeypatch, scripted(lambda call: put_full(call.bundle.out_dir, good_outline())))
    assert ol.run_outline(ctx, blocks_of(path, sources), sources).ok
    path, sources = write_topic(
        tmp_path, [*SOURCES, ("V1", "video", "Видео", None, [blk("paragraph", "V1:1", "Шум.")])]
    )

    def shrugs(call: Call) -> None:
        data = old_outline_of(call)
        data["unassigned"].append({"block": "V1.b001", "reason": "не по теме"})
        put(call.bundle.out_dir, data)

    agent_calls(monkeypatch, dispatch(shrugs))
    result = ol.run_outline(ctx, blocks_of(path, sources), sources)
    assert result.ok and result.details["mode"] == "update"
    assert any("В `unassigned` 1 из 1 новых содержательных блоков" in w for w in result.warnings)


# ---------------------------------------------------------------- when the mode does not apply


def full_run_fixture(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeAgent, dict[str, Block], list[SourceRecord]]:
    blocks, sources = built
    agent = agent_calls(monkeypatch, dispatch(good_update))
    return agent, blocks, sources


def test_update_is_not_used_when_forced(
    built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, blocks, sources = full_run_fixture(built, ctx, monkeypatch)
    ctx.force = True  # --force and --from outline mean «build the structure again»
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.details["mode"] == "full"
    assert agent.calls[0].task.startswith(FULL_TASK) and len(agent.calls) == 1


def test_update_is_not_used_when_old_blocks_are_gone(
    tmp_path: Path, built, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _blocks, _sources = built
    # the slides are removed from the topic and the video is there
    spec = [s for s in (*SOURCES, V1) if s[0] != "S1"]
    path, sources = write_topic(tmp_path, spec)
    blocks = blocks_of(path, sources)

    def full(call: Call) -> None:
        data = good_outline()
        for s in data["sections"]:
            s["blocks"] = [b for b in s["blocks"] if not b.startswith("S1.")]
        data["unassigned"] = [u for u in data["unassigned"] if not u["block"].startswith("S1.")]
        data["sections"][1]["blocks"] += ["V1.b003"]
        data["sections"][2]["blocks"] += ["V1.b001", "V1.b002"]
        put_full(call.bundle.out_dir, data)

    agent = agent_calls(monkeypatch, dispatch(good_update, full))
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.details["mode"] == "full" and "update_failed" not in result.details
    assert agent.calls[0].task.startswith(FULL_TASK)


def test_update_needs_new_content_blocks(
    tmp_path: Path, ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sources = write_topic(tmp_path)
    agent = agent_calls(
        monkeypatch, scripted(lambda call: put_full(call.bundle.out_dir, good_outline()))
    )
    blocks = blocks_of(path, sources)
    assert ol.run_outline(ctx, blocks, sources).details["mode"] == "full"  # no outline yet
    # a new source with service blocks only, and a changed summary: the ordinary S1 as before
    path, sources = write_topic(
        tmp_path, [*SOURCES, ("V1", "video", "Видео", None, [blk("admin", "V1:1", "Реклама")])]
    )
    blocks = blocks_of(path, sources)
    assert ol.plan_update(ctx, ol.previous_outline(ctx), blocks) is None
    result = ol.run_outline(ctx, blocks, sources)
    assert result.ok and result.details["mode"] == "full" and len(agent.calls) == 2
    assert all(c.task.startswith(FULL_TASK) for c in agent.calls)


def test_a_broken_old_outline_is_not_a_base(built, ctx: BuildContext) -> None:
    bad = good_outline()
    bad["sections"][2]["id"] = "s01-01"  # a duplicate id
    (ctx.synth_dir / "outline.json").write_text(json.dumps(bad, ensure_ascii=False), "utf-8")
    assert ol.previous_outline(ctx) is None
    (ctx.synth_dir / "outline.json").write_text("{не json", "utf-8")
    assert ol.previous_outline(ctx) is None
    (ctx.synth_dir / "outline.json").unlink()
    assert ol.previous_outline(ctx) is None


def test_plan_update_lists_new_blocks(built, ctx: BuildContext) -> None:
    blocks, _ = built
    plan = ol.plan_update(ctx, ol.previous_outline(ctx), blocks)
    assert plan is not None
    assert plan.new_ids == ["V1.b001", "V1.b002", "V1.b003", "V1.b004"]
    assert plan.new_content == ["V1.b001", "V1.b002", "V1.b003"]  # the admin block is optional


# ---------------------------------------------------------------- the preserve check


def old_and(mutate: Callable[[dict[str, Any]], None]) -> list[ol.Problem]:
    old = ol.parse_outline(good_outline())
    data = copy.deepcopy(good_outline())
    mutate(data)
    return ol.check_preserved(old, ol.parse_outline(data))


def test_check_preserved_accepts_additions() -> None:
    def add(data: dict[str, Any]) -> None:
        section(data, "s01-01")["blocks"].append("V1.b003")
        section(data, "s01-01")["blocks"].insert(1, "V1.b001")  # between the old ones
        data["sections"].insert(
            2,
            {"id": "s01-03", "title": "Новый", "level": 2, "blocks": ["V1.b002"], "parent": "s01"},
        )
        data["sections"].append(
            {"id": "s03", "title": "Глава", "level": 1, "blocks": ["V1.b004"], "parent": None}
        )

    assert old_and(add) == []


@pytest.mark.parametrize(
    ("mutate", "text"),
    [
        (lambda d: d["sections"].pop(2), "Удалены существующие разделы (1): `s01-02` «Примеры»"),
        (lambda d: section(d, "s02").update(title="Другое"), "заголовок «Из статьи» → «Другое»"),
        (lambda d: section(d, "s01-02").update(level=1, parent=None), "level 2 → 1"),
        (lambda d: section(d, "s01-02").update(parent="s02"), "parent s01 → s02"),
        (lambda d: d["sections"].reverse(), "Изменён порядок существующих разделов"),
        (lambda d: section(d, "s02")["blocks"].clear(), "Блоки вышли из своих разделов (3)"),
        (
            lambda d: section(d, "s01-01")["blocks"].reverse(),
            "Изменён порядок старых блоков внутри разделов: `s01-01`",
        ),
    ],
)
def test_check_preserved_finds_changes(mutate: Callable[[dict[str, Any]], None], text: str) -> None:
    problems = old_and(mutate)
    assert problems and all(p.kind == "preserve" for p in problems)
    assert any(text in p.text for p in problems), [p.text for p in problems]


def test_preserve_problems_are_structural(built, ctx: BuildContext) -> None:
    blocks, _ = built
    old = ol.parse_outline(good_outline())
    broken = ol.parse_outline(good_outline())
    broken.sections.pop(2)
    check = ol.check_update(old, broken, blocks)
    assert check.structural and check.score[0] >= 1 and not check.ok


# ---------------------------------------------------------------- terms for the group keys


TERMS = """# Термины и обозначения источников

## P1 — Лекция (pdf-text)

- **kNN** — метод

## S1 — Слайды (slides)

- **метрика** — расстояние

## V1 — Видео (video)

- **окно** — ядро
"""


def test_terms_for_sources_keeps_the_head_and_the_wanted_parts() -> None:
    both = ol.terms_for_sources(TERMS, {"P1", "V1"})
    assert both.startswith("# Термины и обозначения источников")
    assert "**kNN**" in both and "**окно**" in both and "**метрика**" not in both
    assert ol.terms_for_sources(TERMS, set()).strip() == "# Термины и обозначения источников"
    # a part keeps its text when another part is appended after it (blank lines do not count)
    old = TERMS.split("## V1")[0]
    for sid in ({"P1"}, {"S1"}, {"P1", "S1"}):
        assert ol.terms_for_sources(old, sid) == ol.terms_for_sources(TERMS, sid)
    plain = "# Термины\n\nВ источниках не найдено разделов.\n"
    assert ol.terms_for_sources(plain, {"P1"}) == plain.rstrip("\n")
    fenced = TERMS.replace("- **kNN** — метод", "```\n## V1 — не заголовок\n```")
    assert "не заголовок" in ol.terms_for_sources(fenced, {"P1"})  # a fence is not a heading


# ---------------------------------------------------------------- S2 after an update


@pytest.fixture
def s2(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sc, "MAX_SECTIONS", 2)  # the world outline: three groups
    ctx = BuildContext(topic_dir=tmp_path / "topic", settings=settings)
    terms = ctx.synth_dir / "inputs" / "terms.md"
    terms.parent.mkdir(parents=True)
    terms.write_text(TERMS.split("## V1")[0], encoding="utf-8")
    outline, blocks = world_outline(), world_blocks()
    agent = SectionAgent(good_writer(outline))
    monkeypatch.setattr(sc, "run_agent", agent)
    first = sc.run_sections(ctx, outline, blocks)
    assert (
        first.ok
        and first.agent_runs == 3
        and agent.groups()
        == [
            ["s01-01", "s01-02"],
            ["s02-01", "s02-02"],
            ["s03", "s04"],
        ]
    )
    return ctx, outline, blocks, agent, terms


def with_new_source(
    outline: Outline, blocks: dict[str, Block], terms: Path
) -> tuple[Outline, dict[str, Block]]:
    """V1 joins: one block in s04 (the last group) and its terms appended to the glossary."""
    new_outline = copy.deepcopy(outline)
    next(s for s in new_outline.sections if s.id == "s04").blocks.append("V1.b001")
    new_blocks = {**blocks, "V1.b001": make_block("V1.b001", "Блок видео.", anchor="V1:00:10")}
    terms.write_text(TERMS, encoding="utf-8")
    return new_outline, new_blocks


def test_a_new_source_rewrites_only_the_groups_that_got_its_blocks(s2) -> None:
    ctx, outline, blocks, agent, terms = s2
    new_outline, new_blocks = with_new_source(outline, blocks, terms)
    result = sc.run_sections(ctx, new_outline, new_blocks)
    assert result.ok and result.agent_runs == 1 and not result.cached
    assert agent.calls[-1].section_ids == ["s03", "s04"]
    assert result.details["groups"] == 3 and result.details["groups_cached"] == 2
    # the agent of that group still gets the whole glossary, the new terms included
    assert "окно" in (agent.calls[-1].bundle.inputs_dir / "glossary.md").read_text("utf-8")
    assert sc.run_sections(ctx, new_outline, new_blocks).cached


def test_terms_of_a_source_the_group_draws_on_still_rewrite_it(s2) -> None:
    ctx, outline, blocks, _agent, terms = s2
    terms.write_text(TERMS.split("## V1")[0].replace("**kNN** — метод", "**kNN** — иначе"), "utf-8")
    assert sc.run_sections(ctx, outline, blocks).agent_runs == 3  # P1 is in every group
    head = terms.read_text("utf-8").replace("# Термины и обозначения источников", "# Термины")
    terms.write_text(head, encoding="utf-8")
    assert sc.run_sections(ctx, outline, blocks).agent_runs == 3  # the head concerns everyone


def write_legacy_meta(ctx, outline, blocks, terms) -> None:
    """sections.meta.json as the version before the incremental update wrote it: the group
    records stored under keys that contain the whole glossary file."""
    groups = sc.plan_groups(outline, blocks, max_chars=sc.MAX_CHARS, max_sections=2)
    paths = {s.id: sc.prepare_section_inputs(ctx, s, blocks) for g in groups for s in g}
    meta = read_meta(ctx, "sections")
    assert meta is not None
    by_sections = {tuple(rec["sections"]): rec for rec in meta["groups"].values()}  # any order
    legacy = {
        sc._legacy_group_key(ctx, g, outline, paths, terms): by_sections[tuple(s.id for s in g)]
        for g in groups
    }
    write_meta(ctx, "sections", {**meta, "key": "старый ключ стадии", "groups": legacy})


def test_groups_stored_under_the_old_key_are_adopted(s2) -> None:
    ctx, outline, blocks, agent, terms = s2
    write_legacy_meta(ctx, outline, blocks, terms)
    result = sc.run_sections(ctx, outline, blocks)
    assert result.ok and result.cached and result.agent_runs == 0 and len(agent.calls) == 3
    assert result.details["groups_cached"] == 3
    assert len(read_meta(ctx, "sections")["groups"]) == 3  # type: ignore[index]
    assert sc.run_sections(ctx, outline, blocks).cached  # and now under the new keys


def test_old_groups_are_adopted_when_the_glossary_grew_with_the_new_source(s2) -> None:
    """The first update of a topic built before the incremental mode: S1 keeps the glossary of
    that build as terms.prev.md, and the groups that do not draw on V1 are still valid."""
    ctx, outline, blocks, agent, terms = s2
    write_legacy_meta(ctx, outline, blocks, terms)
    prev = ctx.synth_dir / "inputs" / ol.TERMS_PREV_FILE
    prev.write_text(terms.read_text("utf-8"), encoding="utf-8")
    new_outline, new_blocks = with_new_source(outline, blocks, terms)
    result = sc.run_sections(ctx, new_outline, new_blocks)
    assert result.ok and result.agent_runs == 1 and result.details["groups_cached"] == 2
    assert agent.calls[-1].section_ids == ["s03", "s04"]

    # not for a group whose own sources' terms changed in between
    write_legacy_meta(ctx, outline, blocks, prev)
    terms.write_text(
        terms.read_text("utf-8").replace("**kNN** — метод", "**kNN** — иначе"), "utf-8"
    )
    result = sc.run_sections(ctx, outline, blocks)
    assert result.agent_runs == 3


def test_prepare_inputs_keeps_the_previous_glossary(tmp_path: Path, ctx: BuildContext) -> None:
    path, sources = write_topic(tmp_path)
    blocks = blocks_of(path, sources)
    prev = ctx.synth_dir / "inputs" / ol.TERMS_PREV_FILE
    ol.prepare_inputs(ctx, blocks, sources)
    assert not prev.exists()  # nothing was there before
    first = (ctx.synth_dir / "inputs" / "terms.md").read_text("utf-8")
    ol.prepare_inputs(ctx, blocks, sources)
    assert not prev.exists()  # the glossary did not change
    path, sources = write_topic(tmp_path, [*SOURCES, V1])
    ol.prepare_inputs(ctx, blocks_of(path, sources), sources)
    assert prev.read_text("utf-8") == first
    assert "окно Парзена" in (ctx.synth_dir / "inputs" / "terms.md").read_text("utf-8")


# ---------------------------------------------------------------- stable groups


def flat_outline(n: int) -> tuple[Outline, dict[str, Block]]:
    """n chapters with one small block each: s01 … s<n>."""
    ids = [f"s{i:02d}" for i in range(1, n + 1)]
    blocks = {f"P1.b{i:03d}": make_block(f"P1.b{i:03d}", "x" * 10) for i in range(1, n + 1)}
    sections = [
        OutlineSection(id=sid, title=f"Раздел {sid}", level=1, blocks=[f"P1.b{i:03d}"])
        for i, sid in enumerate(ids, 1)
    ]
    return Outline("T", sections), blocks


def ids_of(groups: list[list[Any]]) -> list[list[str]]:
    return [[s.id for s in g] for g in groups]


def test_plan_groups_keeps_a_valid_previous_group_when_a_section_is_inserted() -> None:
    outline, blocks = flat_outline(10)
    plain = sc.plan_groups(outline, blocks)
    assert ids_of(plain) == [
        ["s01", "s02", "s03", "s04"],
        ["s05", "s06", "s07", "s08"],
        ["s09", "s10"],
    ]
    previous = ids_of(plain)
    blocks["V1.b001"] = make_block("V1.b001", "x" * 10)
    new = OutlineSection(id="s11", title="Новый", level=1, blocks=["V1.b001"])

    # between two groups: it forms a group of its own, the others stay
    outline.sections.insert(4, new)
    assert ids_of(sc.plan_groups(outline, blocks, keep=previous)) == [
        ["s01", "s02", "s03", "s04"],
        ["s11"],
        ["s05", "s06", "s07", "s08"],
        ["s09", "s10"],
    ]
    # without `keep` everything after it moves (the old behaviour)
    assert ids_of(sc.plan_groups(outline, blocks))[1:] == [
        ["s11", "s05", "s06", "s07"],
        ["s08", "s09", "s10"],
    ]

    # inside a group: only that group is planned again (with the section that is new)
    outline.sections.remove(new)
    outline.sections.insert(2, new)
    assert ids_of(sc.plan_groups(outline, blocks, keep=previous)) == [
        ["s01", "s02", "s11", "s03"],
        ["s04"],
        ["s05", "s06", "s07", "s08"],
        ["s09", "s10"],
    ]


def test_plan_groups_drops_groups_that_are_no_longer_valid() -> None:
    outline, blocks = flat_outline(8)
    previous = [["s01", "s02", "s03", "s04"], ["s05", "s06", "s07", "s08"]]
    # a section is gone
    gone = Outline("T", [s for s in outline.sections if s.id != "s02"])
    assert ids_of(sc.plan_groups(gone, blocks, keep=previous)) == [
        ["s01", "s03", "s04"],
        ["s05", "s06", "s07", "s08"],
    ]
    # a section moved
    moved = Outline("T", [*outline.sections[1:3], outline.sections[0], *outline.sections[3:]])
    assert ids_of(sc.plan_groups(moved, blocks, keep=previous))[1] == ["s05", "s06", "s07", "s08"]
    assert ids_of(sc.plan_groups(moved, blocks, keep=previous))[0] == ["s02", "s03", "s01", "s04"]
    # the group grew over the limit
    big = dict(blocks)
    big["P1.b003"] = make_block("P1.b003", "x" * 100)
    groups = sc.plan_groups(outline, big, max_chars=60, keep=previous)
    assert ids_of(groups)[0] == ["s01", "s02"] and ["s03"] in ids_of(groups)
    # the limit of sections became smaller
    assert [len(g) for g in sc.plan_groups(outline, blocks, max_sections=2, keep=previous)] == [
        2
    ] * 4
    # a group of one section stays whatever its size
    assert ids_of(sc.plan_groups(outline, big, max_chars=60, keep=[["s03"]])).count(["s03"]) == 1


def test_plan_groups_ignores_nonsense_in_keep() -> None:
    outline, blocks = flat_outline(6)
    canonical = ids_of(sc.plan_groups(outline, blocks))
    junk = [[], ["s99"], ["s01", "s99"], ["s03", "s02"], ["s01", "s03"]]
    assert ids_of(sc.plan_groups(outline, blocks, keep=junk)) == canonical
    overlapping = [["s01", "s02"], ["s02", "s03"]]  # the first wins
    assert ids_of(sc.plan_groups(outline, blocks, keep=overlapping))[:2] == [
        ["s01", "s02"],
        ["s03", "s04", "s05", "s06"],
    ]
    assert ids_of(sc.plan_groups(outline, blocks, keep=iter(canonical))) == canonical


def test_a_new_section_in_the_middle_rewrites_only_its_own_group(s2) -> None:
    ctx, outline, blocks, agent, _terms = s2
    new_outline = copy.deepcopy(outline)
    new_blocks = {**blocks, "V1.b001": make_block("V1.b001", "Блок видео.", anchor="V1:00:10")}
    # a new leaf at the end of the first chapter: the old plan would move every later group
    at = next(i for i, s in enumerate(new_outline.sections) if s.id == "s01-02") + 1
    new_outline.sections.insert(
        at,
        OutlineSection(
            id="s01-03", title="Новый раздел", level=2, blocks=["V1.b001"], parent="s01"
        ),
    )
    agent.writer = good_writer(new_outline)  # the agent knows the new section now
    result = sc.run_sections(ctx, new_outline, new_blocks)
    assert result.ok and result.agent_runs == 1 and agent.calls[-1].section_ids == ["s01-03"]
    assert result.details["groups"] == 4 and result.details["groups_cached"] == 3
    assert sc.run_sections(ctx, new_outline, new_blocks).cached

    # --force plans from scratch: [s01-01, s01-02], [s01-03, s02-01], [s02-02, s03], [s04]
    ctx.force = True
    forced = sc.run_sections(ctx, new_outline, new_blocks)
    assert forced.agent_runs == 4
    assert sorted(c.section_ids for c in agent.calls[-4:]) == [
        ["s01-01", "s01-02"],
        ["s01-03", "s02-01"],
        ["s02-02", "s03"],
        ["s04"],
    ]
    ctx.force = False
    assert sc.run_sections(ctx, new_outline, new_blocks).cached  # the canonical plan is stable too


# ---------------------------------------------------------------- the build: changelog


@pytest.fixture
def env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Env:
    """As in test_synth_build: a topic with two extracted sources, every stage faked."""
    topic = make_topic(settings)
    synth = FakeSynth().install(monkeypatch)
    extract = FakeExtract()
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", extract)
    environment = Env(settings, topic, synth, extract)
    monkeypatch.setattr("h0lon.synth.render.render_master", fake_render_master(environment))
    return environment


def add_source(
    env: Env, sid: str, title: str, blocks: list[dict[str, Any]], kind: str = "pdf-text"
):
    out = env.topic / "extracted" / sid
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"{sid}.b{i:03d}", "source": sid, "title": None, **b}
        for i, b in enumerate(blocks, 1)
    ]
    (out / "blocks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    (out / "source.md").write_text(f"# {title}\n", encoding="utf-8")
    meta = load_topic(env.topic)
    meta.sources.append(
        SourceRecord(
            id=sid,
            kind=kind,  # type: ignore[arg-type]
            title=title,
            added="2026-10-04T00:00:00Z",
            status="extracted",
            extracted_key=f"key-{sid}",
        ).model_dump(mode="json")
    )
    save_topic(env.topic, meta)


def set_key(env: Env, sid: str, key: str) -> None:
    meta = load_topic(env.topic)
    for row in meta.sources:
        if row["id"] == sid:
            row["extracted_key"] = key
    save_topic(env.topic, meta)


def changelog(env: Env) -> str:
    return (env.topic / "synthesis" / "changelog.md").read_text("utf-8")


def entries(text: str) -> list[str]:
    return [part for part in text.split("\n## ")[1:]]


NEW_BLOCKS = [
    {"type": "paragraph", "anchor": "P2:p1", "text": "Новый абзац.", "md": "Новый абзац."},
    {"type": "paragraph", "anchor": "P2:p2", "text": "Ещё абзац.", "md": "Ещё абзац."},
]


def test_the_first_build_starts_the_changelog(env: Env) -> None:
    result = run_build(env)
    assert result.ok
    text = changelog(env)
    assert text.startswith("# Журнал изменений мастер-конспекта\n\nТема: «Метрические методы».")
    (entry,) = entries(text)
    assert " — сборка мастера" in entry.splitlines()[0]
    assert "- **Источники:** добавлены: S1 «Слайды лекции» (слайды); P1 «Летучка» (" in entry
    assert "(PDF с текстовым слоем)." in entry
    assert "- **Структура (S1):** выполнена." in entry  # the fake S1 has no mode
    assert "- **Покрытие:** 100 % (9 из 9 блоков)." in entry and "- **Прогонов агента:**" in entry
    assert "synthesis/changelog.md" in git(env.topic, "ls-files").split()  # it is committed
    sources = json.loads((env.topic / "synthesis" / "build.json").read_text("utf-8"))["sources"]
    assert sorted(sources) == ["P1", "S1"] and sources["S1"]["extracted_key"] == "key-S1"


def test_a_build_that_changed_nothing_writes_no_entry(env: Env) -> None:
    run_build(env)
    commits = git(env.topic, "rev-list", "--count", "HEAD").strip()
    before = changelog(env)
    second = run_build(env)
    assert second.ok and changelog(env) == before
    assert git(env.topic, "rev-list", "--count", "HEAD").strip() == commits


def test_a_new_source_is_an_entry(env: Env) -> None:
    run_build(env)
    add_source(env, "P2", "Вторая летучка", NEW_BLOCKS)
    result = run_build(env)
    assert result.ok
    first, second = entries(changelog(env))
    assert "добавлены: P2 «Вторая летучка» (PDF с текстовым слоем)." in second
    assert "Источники:** добавлены: S1" in first  # the earlier entry stays
    assert "- **Разделы (S2):** выполнены." in second and "Прогонов агента" in second
    assert "100 % (11 из 11 блоков)" in second
    assert "synthesis/changelog.md" in git(env.topic, "show", "--name-only", "--format=", "HEAD")


def test_changed_and_removed_sources_are_named(env: Env) -> None:
    run_build(env)
    set_key(env, "S1", "key-S1-новый")
    meta = load_topic(env.topic)
    meta.sources = [s for s in meta.sources if s["id"] != "P1"]
    save_topic(env.topic, meta)
    run_build(env)
    last = entries(changelog(env))[-1]
    assert "изменены: S1 «Слайды лекции» (слайды)" in last
    assert "удалены: P1 «Летучка» (PDF с текстовым слоем)" in last


def test_a_topic_built_before_the_changelog_gets_a_neutral_first_entry(env: Env) -> None:
    run_build(env)
    (env.topic / "synthesis" / "changelog.md").unlink()
    data = json.loads((env.topic / "synthesis" / "build.json").read_text("utf-8"))
    data.pop("sources")
    write_json_atomic(env.topic / "synthesis" / "build.json", data)  # as an older version left it
    add_source(env, "P2", "Вторая летучка", NEW_BLOCKS)
    run_build(env)
    (entry,) = entries(changelog(env))
    assert "журнал ведётся с этой сборки, состав источников до неё неизвестен" in entry
    assert "добавлены" not in entry
    # the set of sources is remembered now: the next change is told exactly
    add_source(env, "P3", "Третья", NEW_BLOCKS)
    run_build(env)
    assert "добавлены: P3 «Третья»" in entries(changelog(env))[-1]


def test_the_sections_s2_rewrote_are_listed(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """S2 reports its groups in sections.meta.json; the entry names the groups written anew."""
    plain = env.synth.run_sections
    counter = {"n": 0}

    def with_groups(ctx: BuildContext, outline: Outline, blocks: dict[str, Block]) -> StageResult:
        groups = dict((read_meta(ctx, "sections") or {}).get("groups") or {})
        result = plain(ctx, outline, blocks)  # the fake rewrites the meta without the groups
        meta = read_meta(ctx, "sections") or {}
        leaves = [s.id for s in outline.sections if s.blocks]
        pairs = [leaves[i : i + 2] for i in range(0, len(leaves), 2)]
        written = 0
        for pair in pairs:
            key = ",".join(pair)
            text = "".join(blocks[b].md for s in outline.sections if s.id in pair for b in s.blocks)
            if key not in groups or groups[key]["text"] != text:
                counter["n"] += 1
                written += 1
                groups[key] = {"sections": pair, "bundles": [f"runs/b{counter['n']}"], "text": text}
        write_meta(ctx, "sections", {**meta, "groups": groups})
        result.details = {"groups": len(pairs), "groups_cached": len(pairs) - written}
        return result

    monkeypatch.setattr(sys.modules["h0lon.synth.sections"], "run_sections", with_groups)
    run_build(env)
    first = entries(changelog(env))[-1]
    assert "- **Разделы (S2):** переписаны разделы (3): `s01-01` «Раздел 1.1»" in first
    assert "групп 2, из кэша 0" in first

    # one more source: only the last group changes, the first one comes from the cache
    add_source(env, "P2", "Вторая летучка", NEW_BLOCKS)
    run_build(env)
    last = entries(changelog(env))[-1]
    assert (
        "- **Разделы (S2):** переписаны разделы (2): `s02-01` «Раздел 2.1», "
        "`s02-02` «Раздел 2.2»; групп 2, из кэша 1." in last
    )
    assert "s01-01" not in last


def test_the_s1_mode_is_told(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    fake: FakeSynth = env.synth
    plain = fake.run_outline
    runs = {"n": 0}

    def tagged(ctx: BuildContext, blocks: dict[str, Block], sources: list[SourceRecord]):
        result = plain(ctx, blocks, sources)
        if not result.cached:
            runs["n"] += 1
            result.details = {
                "mode": "update" if runs["n"] > 1 else "full",
                "new_blocks": len(NEW_BLOCKS),
                "new_sections": ["s09"],
            }
        return result

    monkeypatch.setattr(sys.modules["h0lon.synth.outline"], "run_outline", tagged)
    run_build(env)
    assert "- **Структура (S1):** построена заново." in entries(changelog(env))[0]
    add_source(env, "P2", "Вторая летучка", NEW_BLOCKS)
    run_build(env)
    last = entries(changelog(env))[-1]
    assert "- **Структура (S1):** обновлена по новым блокам (+2): существующие разделы и" in last
    assert "новые разделы: `s09`" in last

    def failed_update(ctx: BuildContext, blocks, sources):
        result = plain(ctx, blocks, sources)
        result.details = {"mode": "full", "update_failed": True}
        return result

    monkeypatch.setattr(sys.modules["h0lon.synth.outline"], "run_outline", failed_update)
    add_source(env, "P3", "Третья", NEW_BLOCKS)
    run_build(env)
    assert (
        "построена заново (обновление существующей структуры не удалось)"
        in entries(changelog(env))[-1]
    )


def test_a_changelog_failure_is_a_warning_not_a_failed_build(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("диск полон")

    monkeypatch.setattr(build_mod, "_changelog_entry", boom)
    result = run_build(env)
    assert result.ok
    git_stage = result.stages[-1]
    assert git_stage.stage == "git" and git_stage.details["committed"] is True
    assert (
        git_stage.warnings
        and "Журнал изменений не записан: RuntimeError: диск полон" in git_stage.warnings[0]
    )
    assert not (env.topic / "synthesis" / "changelog.md").exists()


def test_a_failed_build_leaves_the_changelog_and_the_memory_of_sources_alone(env: Env) -> None:
    run_build(env)
    snapshot = (env.topic / "synthesis" / "build.json").read_text("utf-8")
    text = changelog(env)
    add_source(env, "P2", "Вторая летучка", NEW_BLOCKS)
    env.synth.fail_stage = "global"
    assert not run_build(env).ok
    assert changelog(env) == text
    assert (
        json.loads((env.topic / "synthesis" / "build.json").read_text("utf-8"))["sources"]
        == json.loads(snapshot)["sources"]
    )
    env.synth.fail_stage = None
    run_build(env)  # the source added before the failure is told now
    assert "добавлены: P2" in entries(changelog(env))[-1]
