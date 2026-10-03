"""S4/S5: coverage and supplement (h0lon/synth/coverage.py) with a fake agent."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from h0lon.agents import AttemptRecord, OutputContract, Usage, create_bundle
from h0lon.agents.bundle import TaskBundle
from h0lon.agents.runner import RunResult, restore_seed
from h0lon.agents.validate import validate_outputs
from h0lon.config import Settings
from h0lon.extract.model import Block
from h0lon.synth import coverage
from h0lon.synth.coverage import compute_coverage, master_index, run_coverage
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

    def inp(self, name: str) -> str:
        return (self.bundle.inputs_dir / name).read_text(encoding="utf-8")


@dataclass
class FakeAgent:
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
        failure = self.handler(call)
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

    def stages(self) -> list[str]:
        return [c.stage for c in self.calls]


def install(monkeypatch: pytest.MonkeyPatch, handler: Callable[[Call], str | None]) -> FakeAgent:
    agent = FakeAgent(handler)
    monkeypatch.setattr(coverage, "run_agent", agent)
    return agent


# ---------------------------------------------------------------- the topic

FILLER = "Метрика задаёт расстояние между объектами и определяет, какие соседи близки. "

OUTLINE = Outline(
    title="Метрические методы",
    sections=[
        OutlineSection(id="s01", title="Основы", level=1),
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
            blocks=["S1.b004", "S1.b005", "S1.b006", "P1.b001"],
        ),
        OutlineSection(id="s02", title="Ядра", level=1, blocks=["P1.b002", "P1.b003", "P1.b004"]),
    ],
)


def make_block(bid: str, kind: str = "paragraph") -> Block:
    source = bid.split(".")[0]
    num = int(bid.split(".b")[1])
    text = f"Условие сходимости {bid} для ядра номер {num}"
    return Block(
        id=bid,
        source=source,
        type=kind,  # type: ignore[arg-type]
        anchor=f"{source}:s{num}",
        text=text,
        md=f"{text}: $\\sum_i w_i = 1$.",
    )


def make_blocks() -> dict[str, Block]:
    ids = [f"S1.b{n:03d}" for n in range(1, 7)] + [f"P1.b{n:03d}" for n in range(1, 5)]
    blocks = {bid: make_block(bid) for bid in ids}
    blocks["S1.b007"] = make_block("S1.b007", "admin")
    return blocks


def paragraph(block_ids: list[str], label: str) -> str:
    return f"{label}. {FILLER}[[S1:s1]]\n<!-- src: {' '.join(block_ids)} -->\n"


FINAL = {
    "s01-01": ("## Метрики {#sec:s01-01}", [["S1.b001"], ["S1.b002"]]),
    "s01-02": ("## Метод kNN {#sec:s01-02}", [["S1.b004", "S1.b005"]]),
    "s02": ("# Ядра {#sec:s02}", [["P1.b002"], ["P1.b003"]]),
}


def write_final(ctx: BuildContext) -> dict[str, str]:
    base = ctx.synth_dir / "final"
    texts = {}
    for sid, (heading, paras) in FINAL.items():
        text = (
            heading
            + "\n\n"
            + "\n".join(paragraph(p, f"Абзац {sid}-{i}") for i, p in enumerate(paras))
        )
        (base / "sections").mkdir(parents=True, exist_ok=True)
        (base / "sections" / f"{sid}.md").write_text(text, encoding="utf-8", newline="\n")
        texts[sid] = text
    (base / "glossary.md").write_text(
        "# Глоссарий {#sec:glossary .unnumbered}\n\n- **Ядро** — функция веса.\n"
        "<!-- src: S1.b006 -->\n",
        encoding="utf-8",
    )
    return texts


@pytest.fixture
def ctx(settings: Settings) -> BuildContext:
    topic = create_topic(settings, title="Метрические методы", course="Машинное обучение")
    context = BuildContext(topic_dir=topic, settings=settings)
    (context.synth_dir / "outline.md").write_text("# Метрические методы\n", encoding="utf-8")
    write_final(context)
    return context


def final_text(ctx: BuildContext, sid: str) -> str:
    return (ctx.synth_dir / "final" / "sections" / f"{sid}.md").read_text(encoding="utf-8")


def uncovered_ids(call: Call) -> list[str]:
    return re.findall(r"^<!-- ([A-Z]\d+\.b\d+) ", call.inp("uncovered.md"), re.MULTILINE)


def missing_by_section(call: Call) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current = None
    for line in call.inp("missing.md").splitlines():
        m = re.match(r"^## Раздел `([^`]+)`", line)
        if m:
            current = m.group(1)
            result[current] = []
        m = re.match(r"^### Блок (\S+)", line)
        if m and current:
            result[current].append(m.group(1))
    return result


Verdicts = dict[str, tuple[str, str | None, str]]


def s4_handler(verdicts: Verdicts) -> Callable[[Call], None]:
    def handle(call: Call) -> None:
        items = []
        for bid in uncovered_ids(call):
            kind, section, note = verdicts[bid]
            items.append({"block": bid, "verdict": kind, "section": section, "note": note})
        (call.out / "coverage.json").write_text(
            json.dumps({"verdicts": items}, ensure_ascii=False), encoding="utf-8"
        )

    return handle


def s5_add(call: Call, skip: Collection[str] = ()) -> None:
    """A faithful supplement: one new paragraph per missing block, with anchor and src."""
    for sid, bids in missing_by_section(call).items():
        path = call.out / "sections" / f"{sid}.md"
        text = path.read_text(encoding="utf-8")
        for bid in bids:
            if bid in skip:
                continue
            text += f"\nДобавлено из {bid}: условие сходимости. [[{bid.split('.')[0]}:s9]]\n"
            text += f"<!-- src: {bid} -->\n"
        path.write_text(text, encoding="utf-8")


def handler_for(verdicts: Verdicts, s5: Callable[[Call], None] = s5_add):
    def handle(call: Call) -> str | None:
        if call.stage == "coverage":
            s4_handler(verdicts)(call)
        else:
            s5(call)
        return None

    return handle


DEFAULT_VERDICTS: Verdicts = {
    "S1.b003": ("duplicate", "s01-01", "то же определение метрики, что в абзаце 0"),
    "P1.b001": ("missing", "s01-02", "условие сходимости нигде не приведено"),
    "P1.b004": ("admin", None, "титульный слайд"),
}


# ---------------------------------------------------------------- compute_coverage


def test_compute_coverage_counts_src_comments_of_sections_intro_and_glossary(
    ctx: BuildContext,
) -> None:
    report = compute_coverage(make_blocks(), ctx.synth_dir / "final" / "sections")
    # b001 b002 b004 b005 | P1: b002 b003 | b006 through the glossary
    assert (report.total, report.covered) == (10, 7)
    assert report.by_source == {
        "S1": {"total": 6, "covered": 5},
        "P1": {"total": 4, "covered": 2},
    }
    assert report.uncovered == [
        {"block": "S1.b003", "verdict": "unknown", "note": ""},
        {"block": "P1.b001", "verdict": "unknown", "note": ""},
        {"block": "P1.b004", "verdict": "unknown", "note": ""},
    ]
    assert report.ratio == pytest.approx(0.7) and report.rounds == 0
    assert report.to_dict()["ratio"] == 0.7

    (ctx.synth_dir / "final" / "intro.md").write_text(
        "# Введение {#sec:intro}\n\nТекст.\n<!-- src: S1.b003, P1.b001 -->\n", encoding="utf-8"
    )
    again = compute_coverage(make_blocks(), ctx.synth_dir / "final" / "sections")
    assert again.covered == 9 and [u["block"] for u in again.uncovered] == ["P1.b004"]


def test_admin_blocks_are_not_counted(ctx: BuildContext) -> None:
    blocks = make_blocks()
    assert "S1.b007" in blocks and blocks["S1.b007"].type == "admin"
    report = compute_coverage(blocks, ctx.synth_dir / "final" / "sections")
    assert report.total == 10
    assert "S1.b007" not in {u["block"] for u in report.uncovered}
    assert report.by_source["S1"]["total"] == 6


def test_compute_coverage_accepts_a_block_list_and_a_missing_directory(tmp_path: Path) -> None:
    report = compute_coverage(list(make_blocks().values()), tmp_path / "nope")
    assert (report.total, report.covered, len(report.uncovered)) == (10, 0, 10)
    empty = compute_coverage({}, tmp_path / "nope")
    assert empty.total == 0 and empty.ratio == 1.0 and empty.by_source == {}


# ---------------------------------------------------------------- run_coverage


def test_nothing_uncovered_means_no_agent_runs(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ctx.synth_dir / "final" / "sections" / "s02.md"
    path.write_text(
        path.read_text("utf-8") + "\nещё\n<!-- src: S1.b003 P1.b001 P1.b004 -->\n", "utf-8"
    )
    agent = install(monkeypatch, lambda call: pytest.fail("agent must not run"))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    assert result.ok and result.agent_runs == 0 and agent.calls == []
    assert report.ratio == 1.0 and report.uncovered == [] and report.rounds == 0
    data = json.loads((ctx.synth_dir / "coverage.json").read_text("utf-8"))
    assert data["covered"] == data["total"] == 10 and data["verdicts"] == []
    assert data["ratio"] == 1.0
    meta = json.loads((ctx.synth_dir / "coverage.meta.json").read_text("utf-8"))
    assert meta["ok"] and meta["agent_runs"] == 0


def test_duplicate_admin_and_missing_verdicts(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {sid: final_text(ctx, sid) for sid in FINAL}
    seen: dict[str, Any] = {}

    def handler(call: Call) -> str | None:
        if call.stage == "coverage":
            seen["uncovered"] = call.inp("uncovered.md")
            seen["index"] = call.inp("master_index.md")
            seen["inputs"] = sorted(p.name for p in call.bundle.inputs_dir.iterdir())
        else:
            seen["missing"] = call.inp("missing.md")
            seen["seed"] = (call.out / "sections" / "s01-02.md").read_text("utf-8")
            seen["inputs5"] = sorted(p.name for p in call.bundle.inputs_dir.iterdir())
        return handler_for(DEFAULT_VERDICTS)(call)

    agent = install(monkeypatch, handler)
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    assert result.ok and not result.cached and result.agent_runs == 2
    assert agent.stages() == ["coverage", "supplement"]
    assert [c.tier for c in agent.calls] == ["light", "strong"]

    # S4 inputs: full Markdown with id and anchor, master index, outline; admin blocks excluded
    assert seen["inputs"] == ["master_index.md", "outline.md", "uncovered.md"]
    assert "<!-- S1.b003 paragraph [[S1:s3]] -->" in seen["uncovered"]
    assert "$\\sum_i w_i = 1$" in seen["uncovered"] and "S1.b007" not in seen["uncovered"]
    assert "S1.b001" not in seen["uncovered"]  # covered blocks are not listed
    assert "`## Метод kNN {#sec:s01-02}`" in seen["index"] and "← S1.b004 S1.b005" in seen["index"]
    # S5 inputs: the missing block, the note of S4, the target section; the seed is the section
    assert seen["inputs5"] == ["missing.md", "outline.md"]
    assert "## Раздел `s01-02`" in seen["missing"] and "### Блок P1.b001" in seen["missing"]
    assert "условие сходимости нигде не приведено" in seen["missing"]
    assert "S1.b003" not in seen["missing"] and "P1.b004" not in seen["missing"]
    assert seen["seed"] == before["s01-02"]

    # the section was extended in final/, the others are untouched
    assert "P1.b001" in final_text(ctx, "s01-02")
    assert final_text(ctx, "s01-01") == before["s01-01"] and final_text(ctx, "s02") == before["s02"]

    assert report.total == 10 and report.covered == 10 and report.ratio == 1.0
    assert report.rounds == 1
    assert report.by_source == {"S1": {"total": 6, "covered": 6}, "P1": {"total": 4, "covered": 4}}
    assert report.uncovered == [  # duplicate/admin: covered "with a verdict", in the list
        {"block": "S1.b003", "verdict": "duplicate", "note": DEFAULT_VERDICTS["S1.b003"][2]},
        {"block": "P1.b004", "verdict": "admin", "note": "титульный слайд"},
    ]
    data = json.loads((ctx.synth_dir / "coverage.json").read_text("utf-8"))
    assert {v["block"]: v["verdict"] for v in data["verdicts"]} == {
        "S1.b003": "duplicate",
        "P1.b001": "missing",
        "P1.b004": "admin",
    }
    assert data["covered"] == 10 and data["rounds"] == 1 and data["uncovered"] == report.uncovered
    assert result.warnings == []


def test_block_without_verdict_counts_as_missing(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(call: Call) -> str | None:
        if call.stage == "coverage":
            ids = uncovered_ids(call)
            items = [{"block": "ZZ.b999", "verdict": "admin", "note": "чужой"}]  # ignored
            items.append({"block": ids[0], "verdict": "duplicate", "section": None, "note": "ok"})
            (call.out / "coverage.json").write_text(json.dumps({"verdicts": items}), "utf-8")
        else:
            s5_add(call)
        return None

    agent = install(monkeypatch, handler)
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert result.ok and agent.stages() == ["coverage", "supplement"]
    assert any("не вынес вердикт для 2 блоков" in w for w in result.warnings)
    # the two silent blocks were added by S5 (to the sections the outline assigned them to)
    assert "P1.b001" in final_text(ctx, "s01-02") and "P1.b004" in final_text(ctx, "s02")
    assert [u["block"] for u in report.uncovered] == ["S1.b003"] and report.ratio == 1.0


def test_unusable_s4_result_counts_every_block_as_missing(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(call: Call) -> str | None:
        if call.stage == "coverage":  # violates the JSON Schema of the contract every time
            (call.out / "coverage.json").write_text('{"verdicts": "нет"}', encoding="utf-8")
        else:
            s5_add(call)
        return None

    agent = install(monkeypatch, handler)
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert result.ok and agent.stages() == ["coverage", "supplement"]
    assert any("не вернул годных вердиктов для 3 блоков" in w for w in result.warnings)
    assert "S1.b003" in final_text(ctx, "s01-01") and "P1.b004" in final_text(ctx, "s02")
    assert report.ratio == 1.0 and report.uncovered == []
    data = json.loads((ctx.synth_dir / "coverage.json").read_text("utf-8"))
    assert {v["verdict"] for v in data["verdicts"]} == {"missing"}


def test_two_rounds_and_no_third(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    verdicts: Verdicts = {
        "S1.b003": ("missing", "s01-01", "нет"),
        "P1.b001": ("missing", "s01-02", "нет"),
        "P1.b004": ("missing", "s02", "нет"),
    }
    state = {"round": 0}

    def s5(call: Call) -> None:
        # round 1 adds everything but P1.b004, round 2 adds it
        state["round"] += 1
        s5_add(call, skip={"P1.b004"} if state["round"] == 1 else ())

    agent = install(monkeypatch, handler_for(verdicts, s5))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    assert result.ok and result.agent_runs == 4
    assert agent.stages() == ["coverage", "supplement", "coverage", "supplement"]
    assert uncovered_ids(agent.calls[0]) == ["S1.b003", "P1.b001", "P1.b004"]
    assert uncovered_ids(agent.calls[2]) == ["P1.b004"]  # only what is still missing
    assert sorted(missing_by_section(agent.calls[1])) == ["s01-01", "s01-02", "s02"]
    assert report.rounds == 2 and report.ratio == 1.0 and report.uncovered == []
    assert any("не добавлены блоки: P1.b004" in w or "отклонено" in w for w in result.warnings)


def test_after_two_rounds_whatever_is_left_stays_missing(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdicts: Verdicts = {
        "S1.b003": ("missing", "s01-01", "нет"),
        "P1.b001": ("duplicate", None, "есть"),
        "P1.b004": ("admin", None, "служебное"),
    }
    agent = install(monkeypatch, handler_for(verdicts, lambda call: None))  # S5 adds nothing
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    assert agent.stages() == ["coverage", "supplement", "coverage", "supplement"]  # no 3rd round
    assert result.ok and report.rounds == 2
    assert {u["block"]: u["verdict"] for u in report.uncovered} == {
        "S1.b003": "missing",
        "P1.b001": "duplicate",
        "P1.b004": "admin",
    }
    assert report.covered == 9 and report.ratio == pytest.approx(0.9)
    assert any("по-прежнему нет 1 блоков: S1.b003" in w for w in result.warnings)
    # duplicate/admin verdicts of round 1 are not asked about again
    assert uncovered_ids(agent.calls[2]) == ["S1.b003"]


@pytest.mark.parametrize("damage", ["lose_src", "change_level", "shrink", "add_nothing"])
def test_bad_supplement_is_rolled_back_with_a_warning(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    verdicts: Verdicts = {
        "S1.b003": ("duplicate", None, "есть"),
        "P1.b001": ("missing", "s01-02", "нет"),
        "P1.b004": ("admin", None, "служебное"),
    }
    before = {
        sid: (ctx.synth_dir / "final" / "sections" / f"{sid}.md").read_bytes() for sid in FINAL
    }

    def s5(call: Call) -> None:
        s5_add(call)
        path = call.out / "sections" / "s01-02.md"
        text = path.read_text("utf-8")
        if damage == "lose_src":  # a foreign src id disappears
            text = text.replace("S1.b005", "")
        elif damage == "change_level":
            text = text.replace("## Метод kNN {#sec:s01-02}", "### Метод kNN {#sec:s01-02}")
        elif damage == "shrink":
            text = re.sub(r"Абзац s01-02-0\..*\n", "Абзац.\n", text)
        elif damage == "add_nothing":
            text = text.replace("P1.b001", "")
        path.write_text(text, encoding="utf-8")

    agent = install(monkeypatch, handler_for(verdicts, s5))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    # S5 ran in both rounds and was rejected both times: final/ is exactly as before
    assert agent.stages() == ["coverage", "supplement", "coverage", "supplement"]
    for sid in FINAL:
        assert (ctx.synth_dir / "final" / "sections" / f"{sid}.md").read_bytes() == before[sid]
    assert any(
        "Дополнение раздела s01-02 отклонено, раздел оставлен как был" in w for w in result.warnings
    )
    assert {u["block"]: u["verdict"] for u in report.uncovered}["P1.b001"] == "missing"
    assert result.ok  # a rejected result is a warning, not a failure of the stage


def test_s4_failure_stops_the_stage_but_keeps_a_report(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, lambda call: "лимит подписки")
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert not result.ok and "S4" in result.errors[0] and "лимит подписки" in result.errors[0]
    assert agent.stages() == ["coverage"]
    assert report.covered == 7 and {u["verdict"] for u in report.uncovered} == {"unknown"}
    assert (ctx.synth_dir / "coverage.json").is_file()
    assert json.loads((ctx.synth_dir / "coverage.meta.json").read_text("utf-8"))["ok"] is False


def test_s5_failure_keeps_earlier_progress(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage, "S5_RUN_CHARS", 1)  # one section per run
    verdicts: Verdicts = {
        "S1.b003": ("missing", "s01-01", "нет"),
        "P1.b001": ("missing", "s01-02", "нет"),
        "P1.b004": ("duplicate", None, "есть"),
    }

    def handler(call: Call) -> str | None:
        if call.stage == "coverage":
            s4_handler(verdicts)(call)
            return None
        if call.index == 1:
            s5_add(call)
            return None
        return "не выполнен вход"

    agent = install(monkeypatch, handler)
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert not result.ok and "S5" in result.errors[0]
    assert agent.stages() == ["coverage", "supplement", "supplement"]  # stops at the failure
    assert "S1.b003" in final_text(ctx, "s01-01")  # the first run was taken over
    assert [u["block"] for u in report.uncovered if u["verdict"] == "missing"] == ["P1.b001"]
    # a later run continues from the current final/: nothing is cached after a failure
    again = install(monkeypatch, handler_for(verdicts))
    result2, report2 = run_coverage(ctx, OUTLINE, make_blocks())
    assert result2.ok and not result2.cached
    assert again.stages() == ["coverage", "supplement"]
    assert uncovered_ids(again.calls[0]) == ["P1.b001", "P1.b004"]
    assert report2.ratio == 1.0


# ---------------------------------------------------------------- grouping and targets


def test_missing_blocks_are_grouped_by_target_section_and_packed(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage, "S5_RUN_CHARS", 1)  # every section is a run of its own
    verdicts: Verdicts = {
        "S1.b003": ("missing", "s01", "глава: берём первый раздел главы"),
        "P1.b001": ("missing", None, "раздел не указан: берём из структуры"),
        "P1.b004": ("missing", "нет-такого", "неизвестный раздел"),
    }
    agent = install(monkeypatch, handler_for(verdicts))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())

    supplements = [c for c in agent.calls if c.stage == "supplement"]
    assert len(supplements) == 3 and result.agent_runs == 4
    assert [missing_by_section(c) for c in supplements] == [
        {"s01-01": ["S1.b003"]},
        {"s01-02": ["P1.b001"]},
        {"s02": ["P1.b004"]},
    ]
    for call in supplements:  # each run is seeded with its own section only
        (sid,) = missing_by_section(call)
        assert [f.path for f in call.contract.files] == [f"sections/{sid}.md"]
        assert sorted(p.name for p in (call.bundle.seed_dir / "sections").iterdir()) == [
            f"{sid}.md"
        ]
    assert report.ratio == 1.0


def test_sections_are_packed_into_one_run_when_small(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdicts: Verdicts = {
        "S1.b003": ("missing", "s01-01", "нет"),
        "P1.b001": ("missing", "s01-02", "нет"),
        "P1.b004": ("missing", "s02", "нет"),
    }
    agent = install(monkeypatch, handler_for(verdicts))
    run_coverage(ctx, OUTLINE, make_blocks())
    assert agent.stages() == ["coverage", "supplement"]
    assert sorted(missing_by_section(agent.calls[1])) == ["s01-01", "s01-02", "s02"]
    assert "`out/sections/s02.md`" in agent.calls[1].task


def test_block_outside_the_outline_goes_to_the_section_of_its_neighbour(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = make_blocks()
    blocks["S1.b008"] = make_block("S1.b008")  # not in the outline; neighbours: S1.b007 (admin) …
    verdicts: Verdicts = {
        "S1.b003": ("duplicate", None, "есть"),
        "P1.b001": ("duplicate", None, "есть"),
        "P1.b004": ("duplicate", None, "есть"),
        "S1.b008": ("missing", None, "нет"),
    }
    agent = install(monkeypatch, handler_for(verdicts))
    run_coverage(ctx, OUTLINE, blocks)
    # … S1.b006 is assigned to s01-02 in the outline
    assert missing_by_section(agent.calls[1]) == {"s01-02": ["S1.b008"]}


def test_long_lists_are_split_into_several_s4_runs(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage, "S4_BATCH_CHARS", 1)
    verdicts: Verdicts = {
        "S1.b003": ("duplicate", None, "есть"),
        "P1.b001": ("admin", None, "служебное"),
        "P1.b004": ("duplicate", None, "есть"),
    }
    agent = install(monkeypatch, handler_for(verdicts))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert agent.stages() == ["coverage"] * 3 and result.agent_runs == 3
    assert [uncovered_ids(c) for c in agent.calls] == [["S1.b003"], ["P1.b001"], ["P1.b004"]]
    assert "часть 2 из 3" in agent.calls[1].task
    assert report.ratio == 1.0 and report.rounds == 1


# ---------------------------------------------------------------- cache


def test_cache_survives_the_supplement_that_changed_final(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, handler_for(DEFAULT_VERDICTS))
    first, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert first.agent_runs == 2 and "P1.b001" in final_text(ctx, "s01-02")

    second, report2 = run_coverage(ctx, OUTLINE, make_blocks())
    assert second.ok and second.cached and second.agent_runs == 0 and len(agent.calls) == 2
    assert report2.to_dict() == report.to_dict()

    forced = BuildContext(topic_dir=ctx.topic_dir, settings=ctx.settings, force=True)
    third, _ = run_coverage(forced, OUTLINE, make_blocks())
    assert not third.cached  # S4 asks again for the two remaining judged blocks
    assert len(agent.calls) == 3 and agent.stages()[-1] == "coverage"


def test_changed_final_or_blocks_invalidate_the_cache(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = install(monkeypatch, handler_for(DEFAULT_VERDICTS))
    run_coverage(ctx, OUTLINE, make_blocks())
    calls = len(agent.calls)

    path = ctx.synth_dir / "final" / "sections" / "s02.md"
    path.write_text(path.read_text("utf-8") + "\nправка\n", "utf-8")
    assert not run_coverage(ctx, OUTLINE, make_blocks())[0].cached
    assert len(agent.calls) > calls  # (the judged blocks are asked about again)

    calls = len(agent.calls)
    assert run_coverage(ctx, OUTLINE, make_blocks())[0].cached
    blocks = make_blocks()
    blocks["P1.b002"] = make_block("P1.b002")
    blocks["P1.b002"].md += " изменено"
    assert not run_coverage(ctx, OUTLINE, blocks)[0].cached
    assert len(agent.calls) > calls


def test_outline_input_falls_back_to_a_rendered_copy(
    ctx: BuildContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ctx.synth_dir / "outline.md").unlink()
    seen: dict[str, str] = {}

    def handler(call: Call) -> str | None:
        if call.stage == "coverage":
            seen["outline"] = call.inp("outline.md")
        return handler_for(DEFAULT_VERDICTS)(call)

    install(monkeypatch, handler)
    assert run_coverage(ctx, OUTLINE, make_blocks())[0].ok
    assert "`s01-02` Метод kNN" in seen["outline"]
    assert (ctx.synth_dir / "inputs" / "coverage" / "outline.md").is_file()


def test_missing_final_is_an_error(ctx: BuildContext, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    shutil.rmtree(ctx.synth_dir / "final")
    agent = install(monkeypatch, lambda call: pytest.fail("agent must not run"))
    result, report = run_coverage(ctx, OUTLINE, make_blocks())
    assert not result.ok and "global" in result.errors[0] and agent.calls == []
    assert report.covered == 0 and len(report.uncovered) == 10


# ---------------------------------------------------------------- master index


def test_master_index_lists_headings_snippets_and_src(tmp_path: Path) -> None:
    final = tmp_path / "final"
    (final / "sections").mkdir(parents=True)
    (final / "intro.md").write_text(
        "# Введение {#sec:intro .unnumbered}\n\nКороткое введение.\n", encoding="utf-8"
    )
    long_text = "слово " * 60
    (final / "sections" / "s01.md").write_text(
        "# Глава {#sec:s01}\n\n"
        f"{long_text}[[P1:p3]][[S1:s12]]\n<!-- src: P1.b007 S1.b012 -->\n\n"
        '::: {.definition #def-x title="Метрика"}\n'
        "Функция $\\rho$ называется метрикой.[[P1:p4]]\n:::\n"
        "<!-- src: P1.b008 -->\n\n"
        "Абзац без комментария.\n\n"
        "<!-- src: P1.b009 -->\n",
        encoding="utf-8",
    )
    text = master_index(final, ["s01", "s99"])
    assert text.index("## Файл intro.md") < text.index("## Файл sections/s01.md")
    assert "`# Глава {#sec:s01}`" in text and "`# Введение {#sec:intro .unnumbered}`" in text
    line = next(ln for ln in text.splitlines() if ln.startswith("- слово"))
    snippet = line[2 : line.index(" ← ")]
    assert len(snippet) == coverage.SNIPPET_CHARS and snippet.endswith("…")
    assert line.endswith("← P1.b007 S1.b012")
    assert "[[" not in text  # anchors are not shown
    assert "- [definition] Функция $\\rho$ называется метрикой. ← P1.b008" in text
    # a standalone src comment belongs to the previous paragraph
    assert "- Абзац без комментария. ← P1.b009" in text
    assert "s99" not in text and "glossary" not in text  # missing files are skipped
