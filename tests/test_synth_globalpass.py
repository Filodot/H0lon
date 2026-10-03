"""S3: global pass (h0lon/synth/globalpass.py) with a fake agent instead of a real one."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from h0lon.agents import AttemptRecord, OutputContract, Usage, create_bundle
from h0lon.agents.bundle import TaskBundle
from h0lon.agents.runner import RunResult, restore_seed
from h0lon.agents.validate import validate_outputs
from h0lon.config import Settings
from h0lon.synth import globalpass
from h0lon.synth.common import parse_src_ids, text_size
from h0lon.synth.globalpass import heading_anchors, run_global
from h0lon.synth.model import BuildContext, Outline, OutlineSection
from h0lon.workspace import create_topic

# ---------------------------------------------------------------- fake agent


@dataclass
class Call:
    index: int
    stage: str
    task: str
    tier: str
    bundle: TaskBundle
    contract: OutputContract

    @property
    def out(self) -> Path:
        return self.bundle.out_dir


@dataclass
class FakeAgent:
    """Replaces common.run_agent: creates a real bundle, lets `handler` write out/, validates."""

    handler: Callable[[Call], str | None]
    calls: list[Call] = field(default_factory=list)

    def __call__(
        self, ctx: BuildContext, *, stage, task, contract, inputs=(), seed=None, tier="strong"
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
        call = Call(len(self.calls), stage, task, tier, bundle, contract)
        self.calls.append(call)
        failure = self.handler(call)  # "crash" simulates an agent that cannot run at all
        problems = validate_outputs(bundle.out_dir, contract)
        if failure:
            problems, kind = [f"сбой агента: {failure}"], "crash"
        else:
            kind = "validation" if problems else None
        attempt = AttemptRecord(
            n=1,
            backend="fake",
            model=None,
            started_at="",
            duration_s=0.0,
            exit_code=0,
            ok=not problems,
            error_kind=kind,
            error=None,
            usage=Usage(),
            validation_problems=list(problems),
            transcript="",
        )
        return RunResult(
            ok=not problems,
            bundle=bundle,
            backend_used="fake",
            attempts=[attempt],
            usage_total=Usage(),
            final_text="",
            problems=problems,
        )


# ---------------------------------------------------------------- the topic

FILLER = (
    "Метрика задаёт расстояние между объектами выборки и определяет, какие соседи считаются "
    "близкими; от её выбора зависит качество классификации и скорость поиска соседей. "
)
NOTES = {
    "corrections": [
        {
            "block": "S1.b002",
            "as_written": "σ-кольцо",
            "corrected": "σ-алгебра",
            "reason": "описка",
        }
    ],
    "conflicts": [],
    "editorial": [],
}


def para(src: list[str], word: str) -> str:
    refs = "".join(f"[[{b.split('.')[0]}:p1]]" for b in src[:1])
    return f"{word}. {FILLER}${{{word}}}$.{refs}\n<!-- src: {' '.join(src)} -->\n"


def section_md(sid: str, title: str, level: int, blocks: list[str]) -> str:
    out = [f"{'#' * level} {title} {{#sec:{sid}}}\n"]
    for i, b in enumerate(blocks):
        out.append(para([b], f"Утверждение {sid}-{i}"))
    return "\n".join(out)


OUTLINE = Outline(
    title="Метрические методы",
    sections=[
        OutlineSection(id="s01", title="Основы", level=1, summary="Введение в метрики"),
        OutlineSection(
            id="s01-01",
            title="Метрики",
            level=2,
            parent="s01",
            blocks=["S1.b001", "S1.b002", "S1.b003"],
        ),
        OutlineSection(
            id="s01-02",
            title="Метод kNN",
            level=2,
            parent="s01",
            blocks=["S1.b004", "S1.b005", "P1.b001"],
        ),
        OutlineSection(id="s02", title="Ядра", level=1, blocks=["P1.b002", "P1.b003", "P1.b004"]),
    ],
)


@pytest.fixture
def ctx(settings: Settings) -> BuildContext:
    topic = create_topic(settings, title="Метрические методы", course="Машинное обучение")
    events: list[str] = []
    context = BuildContext(topic_dir=topic, settings=settings, on_event=events.append)
    sdir = context.synth_dir / "sections"
    for s in OUTLINE.sections:
        if s.blocks:
            text = section_md(s.id, s.title, s.level, s.blocks)
            (sdir / f"{s.id}.md").parent.mkdir(parents=True, exist_ok=True)
            (sdir / f"{s.id}.md").write_text(text, encoding="utf-8", newline="\n")
            notes = {k: [dict(r) for r in v] for k, v in NOTES.items()} if s.id == "s01-01" else {}
            (sdir / f"{s.id}.notes.json").write_text(
                json.dumps(notes or {"corrections": [], "conflicts": [], "editorial": []}),
                encoding="utf-8",
            )
    (context.synth_dir / "outline.md").write_text("# Метрические методы\n", encoding="utf-8")
    (context.synth_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (context.synth_dir / "inputs" / "terms.md").write_text(
        "# Термины и обозначения\n\n- **Метрика** — расстояние\n", encoding="utf-8"
    )
    return context


def install(monkeypatch: pytest.MonkeyPatch, handler: Callable[[Call], str | None]) -> FakeAgent:
    agent = FakeAgent(handler)
    monkeypatch.setattr(globalpass, "run_agent", agent)
    return agent


def section_ids(call: Call) -> list[str]:
    return sorted(p.stem for p in (call.out / "sections").glob("*.md"))


def good_handler(call: Call) -> str | None:
    """A faithful pass: edits the sections a little and writes intro, glossary and notes."""
    for path in (call.out / "sections").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("Метрика задаёт", "Метрика определяет"), encoding="utf-8")
    contract_paths = {f.path for f in call.contract.files}
    if "intro.md" in contract_paths:
        (call.out / "intro.md").write_text(
            "# Введение {#sec:intro .unnumbered}\n\n"
            "Конспект охватывает метрические методы классификации; источники — S1 и P1.\n",
            encoding="utf-8",
        )
        (call.out / "glossary.md").write_text(
            "# Глоссарий терминов и обозначений {#sec:glossary .unnumbered}\n\n"
            "- **Метрика** (metric) — расстояние, [раздел](#sec:s01-01)\n",
            encoding="utf-8",
        )
    (call.out / "global.notes.json").write_text(
        json.dumps(
            {
                "corrections": [],
                "conflicts": [
                    {
                        "section": "s01-01",
                        "topic": "обозначение метрики",
                        "variants": [{"anchor": "S1:p1", "text": "ρ"}],
                        "resolution": "ρ",
                        "reason": "единая нотация",
                    }
                ],
                "editorial": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return None


def drop_src(call: Call, block: str = "S1.b002") -> None:
    path = call.out / "sections" / "s01-01.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(re.sub(rf"\b{re.escape(block)}\b", "", text), encoding="utf-8")


def read_sections(ctx: BuildContext, where: str) -> dict[str, str]:
    base = ctx.synth_dir / where / "sections"
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(base.glob("*.md"))}


# ---------------------------------------------------------------- success


def test_global_pass_success(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    originals = read_sections(ctx, "")
    seen: dict[str, Any] = {}

    def handler(call: Call) -> str | None:
        # The seed: copies of all sections are in out/ before the agent starts.
        seen["ids"] = section_ids(call)
        seen["seed_ok"] = all(
            (call.out / "sections" / f"{sid}.md").read_text(encoding="utf-8") == text
            for sid, text in originals.items()
        )
        seen["inputs"] = sorted(p.name for p in call.bundle.inputs_dir.iterdir())
        seen["glossary"] = (call.bundle.inputs_dir / "glossary.md").read_text(encoding="utf-8")
        seen["notes"] = json.loads((call.bundle.inputs_dir / "notes.json").read_text("utf-8"))
        return good_handler(call)

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)

    assert result.ok and not result.cached and result.stage == "global"
    assert result.agent_runs == 1 and len(agent.calls) == 1
    assert result.warnings == []
    assert agent.calls[0].stage == "global" and agent.calls[0].tier == "strong"
    assert seen["ids"] == ["s01-01", "s01-02", "s02"]  # s01 is a chapter without a file
    assert seen["seed_ok"]
    assert seen["inputs"] == ["glossary.md", "notes.json", "outline.md", "sources.md"]
    assert "Метрика" in seen["glossary"]
    # notes of S2 merged, every record names its section
    assert seen["notes"]["corrections"] == [{"section": "s01-01", **NOTES["corrections"][0]}]
    task = agent.calls[0].task
    assert "глобальная правка" in task and "`out/sections/s01-02.md`" in task

    final = read_sections(ctx, "final")
    assert sorted(final) == ["s01-01", "s01-02", "s02"]
    assert "Метрика определяет" in final["s01-01"]
    assert final == read_sections(ctx, "global")
    for name in ("intro.md", "glossary.md"):
        assert (ctx.synth_dir / "final" / name).is_file()
        assert (ctx.synth_dir / "global" / name).is_file()
    notes = json.loads((ctx.synth_dir / "global" / "global.notes.json").read_text("utf-8"))
    assert notes["conflicts"][0]["section"] == "s01-01"
    meta = json.loads((ctx.synth_dir / "global.meta.json").read_text("utf-8"))
    assert (
        meta["ok"]
        and meta["agent_runs"] == 1
        and meta["key"]
        and meta["prompt"] == ("globalpass@1.0")
    )
    # the sources of S2 are untouched
    assert read_sections(ctx, "") == originals


def test_terms_are_built_from_summaries_when_terms_file_is_missing(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ctx.synth_dir / "inputs" / "terms.md").unlink()
    summary = ctx.topic_dir / "extracted" / "S1" / "summary.md"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "## Аннотация\n\nТекст.\n\n## Термины и обозначения\n\n- **Ядро** — функция веса\n\n"
        "## Другое\n\nне сюда\n",
        encoding="utf-8",
    )
    seen: dict[str, str] = {}

    def handler(call: Call) -> str | None:
        seen["glossary"] = (call.bundle.inputs_dir / "glossary.md").read_text(encoding="utf-8")
        seen["sources"] = (call.bundle.inputs_dir / "sources.md").read_text(encoding="utf-8")
        return good_handler(call)

    install(monkeypatch, handler)
    assert run_global(ctx, OUTLINE).ok
    assert "Ядро" in seen["glossary"] and "Источник S1" in seen["glossary"]
    assert "не сюда" not in seen["glossary"]
    assert "S1" in seen["sources"]


# ---------------------------------------------------------------- failures and retries


def test_lost_src_is_retried_with_feedback_in_a_new_bundle(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    originals = read_sections(ctx, "")
    seeds: list[bool] = []

    def handler(call: Call) -> str | None:
        seeds.append(
            (call.out / "sections" / "s01-01.md").read_text("utf-8") == originals["s01-01"]
        )
        good_handler(call)
        if call.index == 0:
            drop_src(call)
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)

    assert result.ok and result.warnings == []
    assert result.agent_runs == 2
    assert len({c.bundle.id for c in agent.calls}) == 2  # a new bundle for the repeat
    assert seeds == [True, True]  # the repeat starts from the S2 texts again
    first, second = agent.calls
    assert "Замечания автоматической проверки" not in first.task
    assert "Замечания автоматической проверки" in second.task
    assert "S1.b002 (раздел s01-01)" in second.task and "потеряны ссылки" in second.task
    assert "S1.b002" in parse_src_ids(
        (ctx.synth_dir / "final" / "sections" / "s01-01.md").read_text("utf-8")
    )
    assert (ctx.synth_dir / "final" / "intro.md").is_file()


def test_persistent_loss_falls_back_to_s2_sections_decision_a5(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    originals = read_sections(ctx, "")

    def handler(call: Call) -> str | None:
        good_handler(call)
        drop_src(call)
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)

    assert result.ok  # the build goes on: A5
    assert result.agent_runs == 3 and len(agent.calls) == 3  # first run + 2 repeats
    assert any("S1.b002" in w and "без единой нотации" in w for w in result.warnings)
    assert result.details["fallback"] is True
    assert read_sections(ctx, "final") == originals  # exactly the S2 texts
    assert not (ctx.synth_dir / "final" / "intro.md").exists()
    assert not (ctx.synth_dir / "final" / "glossary.md").exists()
    assert not (ctx.synth_dir / "global").exists()
    meta = json.loads((ctx.synth_dir / "global.meta.json").read_text("utf-8"))
    assert meta["ok"] and meta["fallback"] is True


def test_shrunken_volume_is_rejected(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(call: Call) -> str | None:
        good_handler(call)
        if call.index == 0:  # keep every src comment, cut the text to a third
            for path in (call.out / "sections").glob("*.md"):
                lines = path.read_text(encoding="utf-8").splitlines()
                kept = [
                    re.sub(r"(\S.{40}).*", r"\1", ln) if not ln.startswith(("#", "<!--")) else ln
                    for ln in lines
                ]
                path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)

    assert result.ok and result.agent_runs == 2
    second = agent.calls[1].task
    assert "просел объём" in second and "нужно не меньше 90%" in second
    assert "s01-01:" in second  # the worst sections are named
    sizes = [text_size(t) for t in read_sections(ctx, "final").values()]
    assert sum(sizes) >= 0.9 * sum(text_size(t) for t in read_sections(ctx, "").values())


def test_changed_heading_is_rejected(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(call: Call) -> str | None:
        good_handler(call)
        if call.index == 0:
            path = call.out / "sections" / "s02.md"
            path.write_text(path.read_text("utf-8").replace("{#sec:s02}", "{#sec:s02x}"), "utf-8")
        return None

    agent = install(monkeypatch, handler)
    # the contract itself (required_text `{#sec:s02`) lets `{#sec:s02x}` through; the code check
    # must not
    result = run_global(ctx, OUTLINE)
    assert result.ok and result.agent_runs == 2
    assert "пропал заголовок {#sec:s02}" in agent.calls[1].task
    assert "{#sec:s02}" in (ctx.synth_dir / "final" / "sections" / "s02.md").read_text("utf-8")


def test_validation_failure_of_the_runner_is_retried(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(call: Call) -> str | None:
        good_handler(call)
        if call.index == 0:
            (call.out / "intro.md").unlink()  # contract violation: no introduction
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)
    assert result.ok and result.agent_runs == 2
    assert "out/intro.md" in agent.calls[1].task  # the validation problem is in the feedback
    assert (ctx.synth_dir / "final" / "intro.md").is_file()


def test_agent_that_cannot_run_stops_the_stage(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, lambda call: "не выполнен вход")
    result = run_global(ctx, OUTLINE)
    assert not result.ok and len(agent.calls) == 1  # no useless repeats
    assert "не выполнен вход" in result.errors[0]
    assert not (ctx.synth_dir / "final").exists()
    assert json.loads((ctx.synth_dir / "global.meta.json").read_text("utf-8"))["ok"] is False


def test_missing_section_text_is_an_error(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ctx.synth_dir / "sections" / "s02.md").unlink()
    agent = install(monkeypatch, good_handler)
    result = run_global(ctx, OUTLINE)
    assert not result.ok and "s02" in result.errors[0] and not agent.calls


def test_invalid_notes_schema_is_a_contract_violation(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(call: Call) -> str | None:
        good_handler(call)
        if call.index == 0:
            (call.out / "global.notes.json").write_text('{"corrections": "нет"}', "utf-8")
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)
    assert result.ok and result.agent_runs == 2
    assert "global.notes.json" in agent.calls[1].task


# ---------------------------------------------------------------- cache


def test_cache_hit_runs_no_agent_and_a_changed_section_invalidates(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, good_handler)
    assert run_global(ctx, OUTLINE).agent_runs == 1

    again = run_global(ctx, OUTLINE)
    assert again.ok and again.cached and again.agent_runs == 0
    assert len(agent.calls) == 1

    forced = BuildContext(topic_dir=ctx.topic_dir, settings=ctx.settings, force=True)
    assert not run_global(forced, OUTLINE).cached
    assert len(agent.calls) == 2

    path = ctx.synth_dir / "sections" / "s02.md"
    path.write_text(path.read_text("utf-8") + "\nДобавленный абзац.\n", "utf-8")
    changed = run_global(ctx, OUTLINE)
    assert not changed.cached and len(agent.calls) == 3

    # a deleted output also breaks the cache
    (ctx.synth_dir / "final" / "intro.md").unlink()
    assert not run_global(ctx, OUTLINE).cached
    assert len(agent.calls) == 4


def test_a5_outcome_is_cached_with_its_warning(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(call: Call) -> str | None:
        good_handler(call)
        drop_src(call)
        return None

    agent = install(monkeypatch, handler)
    first = run_global(ctx, OUTLINE)
    assert first.details["fallback"] and len(agent.calls) == 3
    second = run_global(ctx, OUTLINE)
    assert second.cached and len(agent.calls) == 3
    assert second.warnings and second.details["fallback"] is True


def test_changed_notes_or_terms_invalidate(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, good_handler)
    run_global(ctx, OUTLINE)
    (ctx.synth_dir / "inputs" / "terms.md").write_text("# Термины\n\n- **Новое**\n", "utf-8")
    assert not run_global(ctx, OUTLINE).cached
    assert len(agent.calls) == 2


# ---------------------------------------------------------------- big themes


def test_big_theme_is_split_by_chapters(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(globalpass, "MAX_RUN_CHARS", 1_500)  # chapter s01 ≈ 2 files, s02 alone
    seen: list[dict[str, Any]] = []

    def handler(call: Call) -> str | None:
        seen.append(
            {
                "ids": section_ids(call),
                "files": sorted(f.path for f in call.contract.files),
                "inputs": sorted(p.name for p in call.bundle.inputs_dir.iterdir()),
                "notes": json.loads((call.bundle.inputs_dir / "notes.json").read_text("utf-8")),
            }
        )
        return good_handler(call)

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)

    assert result.ok and result.agent_runs == 2 and result.details["parts"] == 2
    assert seen[0]["ids"] == ["s01-01", "s01-02"] and seen[1]["ids"] == ["s02"]
    # intro and glossary come from the last run only, and only that run gets the overview
    assert "intro.md" not in seen[0]["files"] and "glossary.md" not in seen[0]["files"]
    assert "intro.md" in seen[1]["files"] and "glossary.md" in seen[1]["files"]
    assert "overview.md" not in seen[0]["inputs"] and "overview.md" in seen[1]["inputs"]
    assert "часть 1 из 2" in agent.calls[0].task and "часть 2 из 2" in agent.calls[1].task
    assert "не нужны" in agent.calls[0].task
    # notes are split by part
    assert [r["section"] for r in seen[0]["notes"]["corrections"]] == ["s01-01"]
    assert seen[1]["notes"]["corrections"] == []
    # everything ends up in final/, the notes of both parts are merged
    assert sorted(read_sections(ctx, "final")) == ["s01-01", "s01-02", "s02"]
    assert (ctx.synth_dir / "final" / "intro.md").is_file()
    notes = json.loads((ctx.synth_dir / "global" / "global.notes.json").read_text("utf-8"))
    assert len(notes["conflicts"]) == 2


def test_big_theme_one_failing_part_means_a5_for_everything(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(globalpass, "MAX_RUN_CHARS", 1_500)
    originals = read_sections(ctx, "")

    def handler(call: Call) -> str | None:
        good_handler(call)
        if "s02" in section_ids(call):  # the last part loses a block every time
            path = call.out / "sections" / "s02.md"
            path.write_text(re.sub(r"P1\.b003", "", path.read_text("utf-8")), "utf-8")
        return None

    agent = install(monkeypatch, handler)
    result = run_global(ctx, OUTLINE)
    assert result.ok and result.details["fallback"]
    assert len(agent.calls) == 1 + 3  # part 1 once, part 2 three times
    assert read_sections(ctx, "final") == originals
    assert not (ctx.synth_dir / "final" / "intro.md").exists()


# ---------------------------------------------------------------- helpers


def test_heading_anchors_ignore_code_fences_and_read_levels() -> None:
    text = (
        "# Глава {#sec:s01}\n\n```\n## код {#sec:fake}\n```\n\n"
        "## Раздел {#sec:s01-01 .unnumbered}\n\n### Без якоря\n"
    )
    assert heading_anchors(text) == {"sec:s01": 1, "sec:s01-01": 2}


def test_outline_text_falls_back_to_rendered_outline(ctx: BuildContext) -> None:
    (ctx.synth_dir / "outline.md").unlink()
    text = globalpass.outline_text(ctx, OUTLINE)
    assert "`s01-02` Метод kNN" in text and "(3 блоков)" in text
