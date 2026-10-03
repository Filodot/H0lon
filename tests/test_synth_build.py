"""build orchestrator: review gate, stages, caches, --from/--force, git commit, status, CLI.

The agent stages S1–S5 are replaced by `FakeSynth` (it follows the contract of the real
modules: caches by key, notes, final/, coverage.json); extraction and agents are never run.
`assemble` is real, `render` is real only in the XeLaTeX end-to-end test.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from h0lon import procutil, tools
from h0lon.cli import app
from h0lon.config import Settings
from h0lon.extract.model import Block, ExtractResult
from h0lon.render import check_pdf
from h0lon.sources.ingest import list_sources, update_source
from h0lon.sources.models import SourceRecord
from h0lon.synth import build as build_mod
from h0lon.synth.common import cached, parse_src_ids, stage_key, write_json_atomic, write_meta
from h0lon.synth.model import (
    STAGES,
    BuildContext,
    BuildResult,
    CoverageReport,
    Outline,
    OutlineSection,
    StageResult,
)
from h0lon.workspace import create_topic, load_topic, save_topic

# ---------------------------------------------------------------- data


def blk(sid: str, n: int, type_: str, text: str, md: str | None = None, anchor: str = "") -> dict:
    return {
        "id": f"{sid}.b{n:03d}",
        "source": sid,
        "type": type_,
        "anchor": anchor or f"{sid}:{'s' if sid == 'S1' else 'p'}{n}",
        "text": text,
        "md": md or text,
        "title": None,
    }


BLOCKS: dict[str, list[dict]] = {
    "S1": [
        blk("S1", 1, "paragraph", "Расстояние Минковского задаётся формулой."),
        blk(
            "S1",
            2,
            "formula",
            "rho(a,b)",
            r"$$\rho(a,b)=\left(\sum_i |a_i-b_i|^p\right)^{1/p}$$",
        ),
        blk(
            "S1",
            3,
            "definition",
            "Функция называется метрикой",
            '::: {.definition title="Метрика"}\nФункция $\\rho$ называется метрикой.\n:::',
        ),
        blk("S1", 4, "list", "симметрия неравенство", "- симметрия\n- неравенство треугольника"),
        blk(
            "S1",
            5,
            "table",
            "p метрика",
            "| $p$ | метрика |\n|:---|:---|\n| 1 | манхэттенская |\n| 2 | евклидова |",
        ),
        blk("S1", 6, "admin", "23 сентября 2026 г."),
    ],
    "P1": [
        blk("P1", 1, "paragraph", "Ядро $K(r)$ неотрицательно.", "Ядро $K(r)$ неотрицательно."),
        blk("P1", 2, "heading", "Окно Парзена", "## Окно Парзена"),
        blk(
            "P1",
            3,
            "example",
            "Пример окна",
            '::: {.example title="Окно"}\nПри $h=1$ окно равно единице.\n:::',
        ),
        blk("P1", 4, "paragraph", "Ширина окна $h$ влияет на гладкость.[[P1:p4]]"),
    ],
}
SOURCE_ROWS = (
    ("S1", "slides", "Слайды лекции", "extracted"),
    ("P1", "pdf-text", "Летучка", "extracted"),
    ("V1", "video", "Запись лекции", "skipped"),
)


def make_topic(
    settings: Settings,
    *,
    review_gate: bool | None = None,
    rows: Iterable[tuple[str, str, str, str]] = SOURCE_ROWS,
) -> Path:
    topic = create_topic(settings, title="Метрические методы", course="Машинное обучение")
    meta = load_topic(topic)
    meta.review_gate = review_gate
    records = []
    for sid, kind, title, status in rows:
        records.append(
            SourceRecord(
                id=sid,
                kind=kind,  # type: ignore[arg-type]
                title=title,
                added="2026-10-03T00:00:00Z",
                units={"pages": 10},
                status=status,  # type: ignore[arg-type]
                extracted_key=f"key-{sid}" if status == "extracted" else None,
                error="видео пока не поддерживаются" if status == "skipped" else None,
                quality={"notes": ["проверьте формулы"]} if sid == "S1" else {},
            ).model_dump(mode="json")
        )
        if status == "extracted":
            out = topic / "extracted" / sid
            out.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(b, ensure_ascii=False) for b in BLOCKS[sid]]
            (out / "blocks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (out / "source.md").write_text(f"# {title}\n", encoding="utf-8")
            (out / "summary.md").write_text("## Аннотация\n\nТест.\n", encoding="utf-8")
    meta.sources = records
    save_topic(topic, meta)
    return topic


# ---------------------------------------------------------------- fakes


@dataclass
class FakeExtract:
    """Replacement of extract_topic: everything is cached, `fail` ids fail."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)

    def __call__(self, settings: Settings, topic_dir: Path, **kw: Any) -> list[ExtractResult]:
        self.calls.append(kw)
        results = []
        for rec in list_sources(topic_dir):
            if rec.id in self.fail:
                update_source(topic_dir, rec.model_copy(update={"status": "failed"}))
                results.append(
                    ExtractResult(ok=False, source_id=rec.id, errors=["не удалось извлечь"])
                )
            elif rec.status == "skipped":
                results.append(
                    ExtractResult(ok=True, source_id=rec.id, warnings=[rec.error or "пропущен"])
                )
            else:
                md = topic_dir / "extracted" / rec.id / "source.md"
                results.append(
                    ExtractResult(ok=True, source_id=rec.id, source_md=md, cached=True, blocks=3)
                )
        return results


class FakeSynth:
    """Fake modules outline, sections, globalpass, coverage with the contract of the real ones."""

    def __init__(self, *, leaf_size: int = 3, chapter_size: int = 2) -> None:
        self.leaf_size, self.chapter_size = leaf_size, chapter_size
        self.worked: list[str] = []  # stages that really did their work
        self.calls: list[str] = []  # every call, cached or not
        self.fail_stage: str | None = None
        self.raise_stage: str | None = None
        self.global_fails = False  # A5: global pass failed, sections go to final/ as they are
        self.skip_blocks: set[str] = set()  # blocks left out of the sections

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSynth:
        modules = {
            "outline": {"run_outline": self.run_outline, "load_outline": self.load_outline},
            "sections": {"run_sections": self.run_sections},
            "globalpass": {"run_global": self.run_global},
            "coverage": {
                "run_coverage": self.run_coverage,
                "compute_coverage": self.compute_coverage,
            },
        }
        for name, attrs in modules.items():
            mod = types.ModuleType(f"h0lon.synth.{name}")
            for key, value in attrs.items():
                setattr(mod, key, value)
            monkeypatch.setitem(sys.modules, f"h0lon.synth.{name}", mod)
        return self

    # -- shared
    def _stage(
        self,
        ctx: BuildContext,
        name: str,
        key: str,
        outputs: list[Path],
        work: Callable[[], list[str]],
    ) -> StageResult:
        self.calls.append(name)
        if self.raise_stage == name:
            raise RuntimeError("взрыв в стадии")
        if cached(ctx, name, key, outputs):
            return StageResult(stage=name, ok=True, cached=True)
        if self.fail_stage == name:
            return StageResult(
                stage=name, ok=False, errors=[f"фейковая ошибка стадии {name}"], agent_runs=1
            )
        warnings = work()
        self.worked.append(name)
        write_meta(
            ctx,
            name,
            {
                "key": key,
                "prompt": f"{name}@1.0",
                "model": "fake",
                "duration_s": 0.5,
                "agent_runs": 1,
                "ok": True,
            },
        )
        return StageResult(stage=name, ok=True, agent_runs=1, duration_s=0.5, warnings=warnings)

    # -- S1
    def run_outline(
        self, ctx: BuildContext, blocks: dict[str, Block], sources: list[SourceRecord]
    ) -> StageResult:
        synth = ctx.synth_dir
        key = stage_key(sorted(blocks), [s.id for s in sources], self.leaf_size, self.chapter_size)

        def work() -> list[str]:
            items = [b for b in blocks.values() if b.type != "admin"]
            leaves = [items[i : i + self.leaf_size] for i in range(0, len(items), self.leaf_size)]
            sections: list[OutlineSection] = []
            for ci, start in enumerate(range(0, len(leaves), self.chapter_size), start=1):
                cid = f"s{ci:02d}"
                sections.append(
                    OutlineSection(id=cid, title=f"Глава {ci}", level=1, summary="Глава")
                )
                for li, leaf in enumerate(leaves[start : start + self.chapter_size], start=1):
                    sections.append(
                        OutlineSection(
                            id=f"{cid}-{li:02d}",
                            title=f"Раздел {ci}.{li}",
                            level=2,
                            blocks=[b.id for b in leaf],
                            parent=cid,
                        )
                    )
            outline = Outline(
                title="Метрические методы классификации",
                sections=sections,
                unassigned=[
                    {"block": b.id, "reason": "служебный"}
                    for b in blocks.values()
                    if b.type == "admin"
                ],
            )
            write_json_atomic(synth / "outline.json", outline.to_dict())
            (synth / "outline.md").write_text(
                "\n".join(f"- {s.id} {s.title}" for s in sections) + "\n", encoding="utf-8"
            )
            return []

        return self._stage(ctx, "outline", key, [synth / "outline.json"], work)

    def load_outline(self, ctx: BuildContext) -> Outline:
        data = json.loads((ctx.synth_dir / "outline.json").read_text(encoding="utf-8"))
        return Outline(
            title=data["title"],
            sections=[OutlineSection(**s) for s in data["sections"]],
            unassigned=data["unassigned"],
            conflict_hints=data["conflict_hints"],
        )

    # -- S2
    def run_sections(
        self, ctx: BuildContext, outline: Outline, blocks: dict[str, Block]
    ) -> StageResult:
        synth = ctx.synth_dir
        key = stage_key(synth / "outline.json", [b.md for b in blocks.values()], self.skip_blocks)
        leaves = [s for s in outline.sections if s.blocks]

        def work() -> list[str]:
            for n, sec in enumerate(leaves):
                lines = [f"## {sec.title} {{#sec:{sec.id}}}", ""]
                for bid in sec.blocks:
                    if bid in self.skip_blocks:
                        continue
                    b = blocks[bid]
                    body = f"**{b.text}**" if b.type == "heading" else b.md
                    lines += [body, f"<!-- src: {bid} -->", ""]
                path = synth / "sections" / f"{sec.id}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(lines), encoding="utf-8")
                notes = {"corrections": [], "conflicts": [], "editorial": []}
                if n == 0:
                    notes = {
                        "corrections": [
                            {
                                "anchor": "S1:s1",
                                "block": "S1.b001",
                                "as_written": "метрика | расстояние",
                                "corrected": "расстояние",
                                "reason": "описка",
                            }
                        ],
                        "conflicts": [
                            {
                                "topic": "Обозначение расстояния",
                                "variants": [
                                    {"anchor": "S1:s1", "text": "$\\rho$"},
                                    {"anchor": "P1:p1", "text": "$d$"},
                                ],
                                "resolution": "$\\rho$",
                                "reason": "так в слайдах",
                            }
                        ],
                        "editorial": [],
                    }
                (synth / "sections" / f"{sec.id}.notes.json").write_text(
                    json.dumps(notes, ensure_ascii=False), encoding="utf-8"
                )
            return []

        outputs = [synth / "sections" / f"{s.id}.md" for s in leaves]
        return self._stage(ctx, "sections", key, outputs, work)

    # -- S3
    def run_global(self, ctx: BuildContext, outline: Outline) -> StageResult:
        synth = ctx.synth_dir
        sections = sorted((synth / "sections").glob("*.md"))
        key = stage_key([p.read_text(encoding="utf-8") for p in sections], self.global_fails)
        final = synth / "final"

        def work() -> list[str]:
            (final / "sections").mkdir(parents=True, exist_ok=True)
            (synth / "global" / "sections").mkdir(parents=True, exist_ok=True)
            for p in sections:
                (final / "sections" / p.name).write_text(
                    p.read_text(encoding="utf-8"), encoding="utf-8"
                )
            if self.global_fails:
                return ["Глобальная правка не удалась: разделы S2 остаются без правки"]
            (final / "intro.md").write_text(
                "# Введение {#sec:intro .unnumbered}\n\nТема охватывает метрические методы.\n",
                encoding="utf-8",
            )
            (final / "glossary.md").write_text(
                "# Глоссарий терминов и обозначений {#sec:glossary .unnumbered}\n\n"
                "- **Метрика** (metric) — расстояние, [раздел](#sec:s01-01)\n",
                encoding="utf-8",
            )
            (synth / "global" / "global.notes.json").write_text(
                json.dumps(
                    {
                        "corrections": [],
                        "conflicts": [],
                        "editorial": [
                            {
                                "section": "s01-01",
                                "kind": "clarification",
                                "refers_to": "S1.b001",
                                "text": "Пояснение к обозначению.",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return []

        return self._stage(ctx, "global", key, [final / "sections"], work)

    # -- S4/S5
    def compute_coverage(self, blocks: dict[str, Block], sections_dir: Path) -> CoverageReport:
        found: set[str] = set()
        for p in sorted(Path(sections_dir).glob("*.md")):
            found |= parse_src_ids(p.read_text(encoding="utf-8"))
        by_source: dict[str, dict[str, int]] = {}
        uncovered = []
        total = covered = 0
        for b in blocks.values():
            if b.type == "admin":
                continue
            stat = by_source.setdefault(b.source, {"total": 0, "covered": 0})
            stat["total"] += 1
            total += 1
            if b.id in found:
                stat["covered"] += 1
                covered += 1
            else:
                uncovered.append({"block": b.id, "verdict": "missing", "note": "нет в тексте"})
        return CoverageReport(
            total=total, covered=covered, by_source=by_source, uncovered=uncovered, rounds=0
        )

    def run_coverage(
        self, ctx: BuildContext, outline: Outline, blocks: dict[str, Block]
    ) -> tuple[StageResult, CoverageReport]:
        synth = ctx.synth_dir
        final = synth / "final" / "sections"
        key = stage_key(
            [p.read_text(encoding="utf-8") for p in sorted(final.glob("*.md"))], sorted(blocks)
        )
        holder: list[CoverageReport] = []

        def work() -> list[str]:
            report = self.compute_coverage(blocks, final)
            holder.append(report)
            write_json_atomic(synth / "coverage.json", {**report.to_dict(), "verdicts": []})
            return []

        result = self._stage(ctx, "coverage", key, [synth / "coverage.json"], work)
        report = holder[0] if holder else self.compute_coverage(blocks, final)
        return result, report


@dataclass
class Env:
    settings: Settings
    topic: Path
    synth: FakeSynth
    extract: FakeExtract
    renders: list[Path] = field(default_factory=list)


def fake_render_master(env: Env) -> Callable[[BuildContext, Path], StageResult]:
    def render_master(ctx: BuildContext, master_md: Path) -> StageResult:
        pdf = ctx.topic_dir / "master.pdf"
        key = stage_key(master_md.read_text(encoding="utf-8"), "fake-render")
        if cached(ctx, "render", key, [pdf]):
            return StageResult(stage="render", ok=True, cached=True)
        env.renders.append(master_md)
        pdf.write_bytes(b"%PDF-1.4 stub")
        write_meta(
            ctx,
            "render",
            {
                "key": key,
                "prompt": "fixlatex@1.0",
                "model": "fake",
                "duration_s": 1.0,
                "agent_runs": 0,
                "ok": True,
            },
        )
        return StageResult(stage="render", ok=True, duration_s=1.0)

    return render_master


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in (
        ("GIT_AUTHOR_NAME", "Test"),
        ("GIT_AUTHOR_EMAIL", "test@example.com"),
        ("GIT_COMMITTER_NAME", "Test"),
        ("GIT_COMMITTER_EMAIL", "test@example.com"),
    ):
        monkeypatch.setenv(var, value)


@pytest.fixture
def env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Env:
    """A topic with two extracted sources and a skipped one; all stages faked, render faked."""
    topic = make_topic(settings)
    synth = FakeSynth().install(monkeypatch)
    extract = FakeExtract()
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", extract)
    environment = Env(settings, topic, synth, extract)
    monkeypatch.setattr("h0lon.synth.render.render_master", fake_render_master(environment))
    return environment


def run_build(env: Env, **kw: Any) -> BuildResult:
    kw.setdefault("review", False)
    return build_mod.build_topic(env.settings, env.topic, **kw)


def git(topic: Path, *args: str) -> str:
    git_exe = tools.find_simple("git")
    if git_exe is None:
        pytest.skip("git not found")
    res = procutil.run([str(git_exe), *args], cwd=topic)
    assert res.ok, res.stderr
    return res.stdout


def stage_names(result: BuildResult) -> list[str]:
    return [s.stage for s in result.stages]


# ---------------------------------------------------------------- review gate


def test_review_gate_stops_and_approve_continues(env: Env) -> None:
    result = build_mod.build_topic(env.settings, env.topic)  # gate is on by default
    assert not result.ok and result.stopped_at == "review"
    assert "h0lon approve" in result.message and "--no-review" in result.message
    assert "Проверьте Source Docs в extracted/" in result.message
    assert "S1 (1)" in result.message  # sources with quality notes are named
    assert stage_names(result) == ["extract", "review"]
    assert env.synth.calls == []  # nothing of the synthesis ran
    assert not (env.topic / "master.md").exists()
    assert result.master_md is None and result.coverage is None

    info = build_mod.approve_topic(env.settings, env.topic)
    assert info["sources"] == ["S1", "P1"]  # the skipped source is not part of the approval
    assert info["extracted_keys"] == {"S1": "key-S1", "P1": "key-P1"}
    again = build_mod.build_topic(env.settings, env.topic)
    assert again.ok and again.stopped_at is None
    review_stage = next(s for s in again.stages if s.stage == "review")
    assert review_stage.ok and review_stage.details["valid"] is True


def test_review_is_invalidated_by_changed_extraction(env: Env) -> None:
    build_mod.approve_topic(env.settings, env.topic)
    assert build_mod.build_topic(env.settings, env.topic).ok
    rec = next(r for r in list_sources(env.topic) if r.id == "P1")
    update_source(env.topic, rec.model_copy(update={"extracted_key": "other-key"}))
    result = build_mod.build_topic(env.settings, env.topic)
    assert result.stopped_at == "review" and not result.ok
    assert "изменилось после одобрения (P1)" in result.message
    # approving again lets the build go on
    build_mod.approve_topic(env.settings, env.topic)
    assert build_mod.build_topic(env.settings, env.topic).ok


def test_new_source_invalidates_review(env: Env) -> None:
    build_mod.approve_topic(env.settings, env.topic)
    meta = load_topic(env.topic)
    meta.sources.append(
        SourceRecord(
            id="D1",
            kind="md",
            title="Заметки",
            added="2026-10-04T00:00:00Z",
            status="extracted",
            extracted_key="key-D1",
        ).model_dump(mode="json")
    )
    save_topic(env.topic, meta)
    d1 = env.topic / "extracted" / "D1"
    d1.mkdir(parents=True)
    (d1 / "blocks.jsonl").write_text(
        json.dumps(blk("D1", 1, "paragraph", "Заметка."), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = build_mod.build_topic(env.settings, env.topic)
    assert result.stopped_at == "review" and "D1" in result.message


def test_no_review_flag_and_gate_settings(
    env: Env, make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = build_mod.build_topic(env.settings, env.topic, review=False)
    assert result.ok
    review_stage = next(s for s in result.stages if s.stage == "review")
    assert "--no-review" in review_stage.details["note"] and review_stage.warnings

    # general.review_gate = false: no stop
    off = make_settings(general={"review_gate": False})
    assert build_mod.build_topic(off, env.topic).ok
    # topic.yaml review_gate overrides the settings in both directions
    meta = load_topic(env.topic)
    meta.review_gate = True
    save_topic(env.topic, meta)
    assert build_mod.build_topic(off, env.topic).stopped_at == "review"
    meta.review_gate = False
    save_topic(env.topic, meta)
    assert build_mod.build_topic(env.settings, env.topic).ok


def test_approve_topic(env: Env) -> None:
    info = build_mod.approve_topic(env.settings, env.topic)
    meta = load_topic(env.topic)
    review = (meta.model_extra or {})["review"]
    assert review["approved_at"] == info["approved_at"]
    assert review["extracted_keys"] == {"S1": "key-S1", "P1": "key-P1"}
    assert meta.title == "Метрические методы" and len(meta.sources) == 3  # the rest is kept
    state = build_mod.review_state(env.settings, env.topic, list_sources(env.topic))
    assert state["gate"] and state["approved"] and state["valid"] and not state["changed"]


def test_approve_requires_extraction(env: Env) -> None:
    rec = next(r for r in list_sources(env.topic) if r.id == "P1")
    update_source(env.topic, rec.model_copy(update={"status": "added", "extracted_key": None}))
    with pytest.raises(ValueError, match=r"P1.*h0lon extract"):
        build_mod.approve_topic(env.settings, env.topic)
    review = (load_topic(env.topic).model_extra or {}).get("review")
    assert review is None  # nothing was written


def test_approve_requires_sources(settings: Settings) -> None:
    topic = create_topic(settings, title="Пустая", course="Курс")
    with pytest.raises(ValueError, match="нет источников"):
        build_mod.approve_topic(settings, topic)
    only_video = make_topic(settings, rows=[("V1", "video", "Видео", "skipped")])
    with pytest.raises(ValueError, match="Ни один источник не извлечён"):
        build_mod.approve_topic(settings, only_video)


# ---------------------------------------------------------------- build flow


def test_full_build(env: Env) -> None:
    events: list[str] = []
    result = run_build(env, on_event=events.append)
    assert result.ok and result.stopped_at is None
    assert stage_names(result) == ["extract", "review", *STAGES[1:], "git"]
    assert all(s.ok for s in result.stages)
    assert result.master_md == env.topic / "master.md" and result.master_md.is_file()
    assert result.master_pdf == env.topic / "master.pdf" and result.master_pdf.is_file()
    assert result.coverage is not None and result.coverage.total == 9
    assert result.coverage.covered == 9 and result.coverage.ratio == 1.0
    assert "покрытие" in result.message.lower() and "100 %" in result.message
    assert any("Структура темы" in e for e in events) and any(
        "из кэша" in e or "готово" in e for e in events
    )
    assert any("V1" in w for w in result.stages[0].warnings)  # the skipped source is mentioned
    assert result.stages[0].details["skipped"] == ["V1"]

    master = result.master_md.read_text(encoding="utf-8")
    assert "{#sec:s01-01}" in master and "# Глава 1 {#sec:s01}" in master
    assert "Расхождения между источниками" in master and "Обозначение расстояния" in master
    assert "Пояснение к обозначению." in master  # global notes landed in the appendix
    assert "V1" not in master  # the skipped source is not in the master
    assert parse_src_ids(master) == {
        b["id"] for sid in ("S1", "P1") for b in BLOCKS[sid] if b["type"] != "admin"
    }
    json.dumps(result.to_dict(), ensure_ascii=False)  # serialisable

    # git: one commit with the contract message, master and synthesis in it, runs/ not
    (env.topic / "runs").mkdir(exist_ok=True)
    log = git(env.topic, "log", "--format=%s")
    assert log.splitlines()[0].startswith("Сборка мастера: ")
    assert log.splitlines()[0].endswith(", покрытие 100 %")
    files = set(git(env.topic, "ls-files").split())
    assert {"master.md", "master.pdf", "synthesis/outline.json", "topic.yaml"} <= files
    assert not any(f.startswith("runs/") or ".build" in f for f in files)
    git_stage = result.stages[-1]
    assert git_stage.details["committed"] is True and not git_stage.warnings


def test_second_build_is_cached(env: Env) -> None:
    first = run_build(env)
    assert first.ok
    commits = git(env.topic, "rev-list", "--count", "HEAD").strip()
    worked = list(env.synth.worked)
    second = run_build(env)
    assert second.ok
    by_name = {s.stage: s for s in second.stages}
    for name in ("outline", "sections", "global", "coverage", "assemble", "render"):
        assert by_name[name].cached, name
    assert env.synth.worked == worked  # nothing recomputed
    assert len(env.renders) == 1
    assert git(env.topic, "rev-list", "--count", "HEAD").strip() == commits  # no empty commit
    assert by_name["git"].details["committed"] is False


def test_from_stage_forces_the_rest(env: Env) -> None:
    run_build(env)
    env.extract.calls.clear()
    result = run_build(env, from_stage="global")
    flags = {s.stage: s.cached for s in result.stages}
    assert flags["outline"] and flags["sections"]
    assert not flags["global"] and not flags["coverage"]
    assert not flags["assemble"] and not flags["render"]
    assert env.extract.calls[0]["force"] is False

    result = run_build(env, from_stage="assemble")
    flags = {s.stage: s.cached for s in result.stages}
    assert flags["global"] and flags["coverage"]
    assert not flags["assemble"] and not flags["render"]  # master.md and PDF are rebuilt

    result = run_build(env, from_stage="extract")
    assert env.extract.calls[-1]["force"] is True
    assert not any(s.cached for s in result.stages if s.stage in STAGES[1:])


def test_force_covers_synthesis_not_extraction(env: Env) -> None:
    run_build(env)
    env.extract.calls.clear()
    result = run_build(env, force=True)
    assert env.extract.calls[0]["force"] is False  # re-extraction is --from extract
    assert not any(s.cached for s in result.stages if s.stage in STAGES[1:])
    assert env.extract.calls[0].get("backend") is None


def test_unknown_stage_and_backend(env: Env) -> None:
    with pytest.raises(ValueError, match="Неизвестная стадия «bogus»"):
        run_build(env, from_stage="bogus")
    with pytest.raises(ValueError, match="claude \\| codex"):
        run_build(env, backend="gpt")
    assert env.synth.calls == []


def test_backend_reaches_the_stages(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []
    orig = env.synth.run_outline

    def spy(ctx: BuildContext, blocks: Any, sources: Any) -> StageResult:
        seen.append(ctx.backend)
        return orig(ctx, blocks, sources)

    sys.modules["h0lon.synth.outline"].run_outline = spy  # type: ignore[attr-defined]
    assert run_build(env, backend="codex").ok
    assert seen == ["codex"]


def test_stage_failure_stops_the_build(env: Env) -> None:
    env.synth.fail_stage = "sections"
    result = run_build(env)
    assert not result.ok and result.stopped_at == "sections"
    assert "Разделы (S2)" in result.message and "фейковая ошибка стадии sections" in result.message
    assert stage_names(result) == ["extract", "review", "outline", "sections"]
    assert "global" not in env.synth.calls
    assert result.master_md is None and not (env.topic / "master.md").exists()
    assert "Сборка мастера" not in git(env.topic, "log", "--format=%s")  # no commit
    # the build is resumable: once the stage works again the rest follows
    env.synth.fail_stage = None
    assert run_build(env).ok


def test_stage_exception_becomes_an_error(env: Env) -> None:
    env.synth.raise_stage = "global"
    result = run_build(env)
    assert not result.ok and result.stopped_at == "global"
    assert "Внутренняя ошибка: RuntimeError: взрыв в стадии" in result.message
    log = env.topic / "synthesis" / "global.error.log"
    assert log.is_file() and "RuntimeError" in log.read_text(encoding="utf-8")
    assert any("global.error.log" in e for e in result.stages[-1].errors)


def test_render_failure_stops(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(ctx: BuildContext, master_md: Path) -> StageResult:
        return StageResult(stage="render", ok=False, errors=["XeLaTeX: ошибка"])

    monkeypatch.setattr("h0lon.synth.render.render_master", failing)
    result = run_build(env)
    assert not result.ok and result.stopped_at == "render"
    assert (env.topic / "master.md").is_file()  # the master exists, only the PDF does not
    assert result.master_md == env.topic / "master.md" and result.master_pdf is None
    assert "XeLaTeX: ошибка" in result.message


def test_uncovered_blocks_land_in_the_coverage_appendix(env: Env) -> None:
    env.synth.skip_blocks = {"P1.b004"}
    result = run_build(env)
    assert result.ok and result.coverage is not None
    assert result.coverage.covered == 8 and "88,8 %" in result.message
    master = (env.topic / "master.md").read_text(encoding="utf-8")
    assert "- `P1.b004` [[P1:p4]] — пропущен: нет в тексте" in master
    assert "| P1 — Летучка | 4 | 3 | 75 % |" in master
    assert "P1.b004" not in parse_src_ids(master)


def test_global_pass_failure_still_builds(env: Env) -> None:
    env.synth.global_fails = True
    result = run_build(env)
    assert result.ok
    master = (env.topic / "master.md").read_text(encoding="utf-8")
    assert "# Введение {#sec:intro .unnumbered}" in master  # minimal intro written by the code
    assert "**S1**" in master and "{#sec:glossary" not in master
    assert any("Глобальная правка не удалась" in w for s in result.stages for w in s.warnings)


def test_extract_stage_failures(env: Env) -> None:
    env.extract.fail = {"P1"}
    result = run_build(env)
    assert not result.ok and result.stopped_at == "extract"
    assert "P1: не удалось извлечь" in result.message
    assert "h0lon extract" in result.stages[0].errors[-1]
    assert env.synth.calls == []


def test_no_sources_and_nothing_extracted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSynth().install(monkeypatch)
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", FakeExtract())
    empty = create_topic(settings, title="Пустая", course="Курс")
    result = build_mod.build_topic(settings, empty, review=False)
    assert not result.ok and result.stopped_at == "extract"
    assert "нет источников" in result.message and "h0lon add" in result.message

    only_video = make_topic(settings, rows=[("V1", "video", "Видео", "skipped")])
    result = build_mod.build_topic(settings, only_video, review=False)
    assert not result.ok and result.stopped_at == "extract"
    assert "Ни один источник не извлечён" in result.message


# ---------------------------------------------------------------- git


def test_git_commit_cases(env: Env, tmp_path: Path) -> None:
    # settings: git_per_topic off — no commit, no warning
    off = Settings(general={**env.settings.general.model_dump(), "git_per_topic": False})
    result = build_mod.build_topic(off, env.topic, review=False)
    git_stage = result.stages[-1]
    assert result.ok and git_stage.details["committed"] is False and not git_stage.warnings

    # not a repository
    no_git = Settings(
        general={
            **env.settings.general.model_dump(),
            "workspaces": tmp_path / "ws-no-git",
            "git_per_topic": False,
        }
    )
    plain = make_topic(no_git)
    assert not (plain / ".git").exists()
    result = build_mod.build_topic(env.settings, plain, review=False)
    assert result.ok and result.stages[-1].details["note"] == "тема не является git-репозиторием"

    # runs/ and build directories stay out of the commit even when they hold files
    (env.topic / "runs" / "x").mkdir(parents=True)
    (env.topic / "runs" / "x" / "t.jsonl").write_text("{}", encoding="utf-8")
    (env.topic / "master.build").mkdir()
    (env.topic / "master.build" / "master.log").write_text("log", encoding="utf-8")
    (env.topic / "notes.txt").write_text("заметка", encoding="utf-8")
    result = build_mod.build_topic(env.settings, env.topic, review=False)
    assert result.stages[-1].details["committed"] is True
    files = git(env.topic, "ls-files").split()
    assert "notes.txt" in files and not any("runs/" in f or ".build" in f for f in files)


def test_git_failure_is_a_warning(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = procutil.run

    def fake_run(argv: Any, **kw: Any) -> procutil.ProcResult:
        if "commit" in [str(a) for a in argv]:
            return procutil.ProcResult(
                [str(a) for a in argv],
                128,
                "",
                "Author identity unknown\n*** Please tell me who you are.",
                0.0,
            )
        return real_run(argv, **kw)

    monkeypatch.setattr(build_mod.procutil, "run", fake_run)
    result = run_build(env)
    assert result.ok  # the build itself succeeded
    git_stage = result.stages[-1]
    assert git_stage.ok and git_stage.details["committed"] is False
    assert "git-идентичность" in git_stage.warnings[0]
    assert "git" in result.message.lower() and "идентичность" in result.message


# ---------------------------------------------------------------- status and printing


def states(status: dict[str, Any]) -> dict[str, str]:
    return {k: v["state"] for k, v in status["stages"].items()}


def test_status_before_and_after_build(env: Env) -> None:
    status = build_mod.topic_status(env.settings, env.topic)
    assert status["title"] == "Метрические методы" and status["course"] == "Машинное обучение"
    assert [s["id"] for s in status["sources"]] == ["S1", "P1", "V1"]
    assert status["sources"][0]["notes"] == 1 and status["sources"][2]["status"] == "skipped"
    assert status["review"] == {
        "gate": True,
        "approved": False,
        "approved_at": None,
        "valid": False,
        "changed": [],
    }
    assert states(status) == {
        "extract": "готово",
        **{s: "не выполнялась" for s in STAGES[1:]},
    }
    assert status["coverage"] is None and status["master_md"] is None

    run_build(env)
    status = build_mod.topic_status(env.settings, env.topic)
    assert set(states(status).values()) == {"готово"}
    assert status["coverage"]["total"] == 9 and status["coverage"]["covered"] == 9
    assert status["master_md"].endswith("master.md") and status["master_pdf"].endswith("master.pdf")
    assert status["stages"]["outline"]["agent_runs"] == 1
    json.dumps(status, ensure_ascii=False)

    run_build(env)  # second run: everything from the cache
    status = build_mod.topic_status(env.settings, env.topic)
    assert {k: v for k, v in states(status).items() if k != "extract"} == {
        s: "из кэша" for s in STAGES[1:]
    }


def test_status_detects_stale_stages(env: Env) -> None:
    run_build(env)
    rec = next(r for r in list_sources(env.topic) if r.id == "S1")
    update_source(env.topic, rec.model_copy(update={"extracted_key": "key-new"}))
    status = build_mod.topic_status(env.settings, env.topic)
    assert {k: v for k, v in states(status).items() if k != "extract"} == {
        s: "устарело" for s in STAGES[1:]
    }
    # a rebuild brings them back
    run_build(env)
    assert set(states(build_mod.topic_status(env.settings, env.topic)).values()) <= {
        "готово",
        "из кэша",
    }
    # a deleted PDF makes only the render stage stale
    (env.topic / "master.pdf").unlink()
    status = build_mod.topic_status(env.settings, env.topic)
    assert states(status)["render"] == "устарело" and states(status)["assemble"] != "устарело"


def test_status_shows_failed_stage(env: Env) -> None:
    env.synth.fail_stage = "outline"
    run_build(env)
    env.synth.fail_stage = None
    # the failed stage wrote no meta: not run
    assert states(build_mod.topic_status(env.settings, env.topic))["outline"] == "не выполнялась"
    meta = env.topic / "synthesis" / "outline.meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"key": "k", "ok": False}), encoding="utf-8")
    assert states(build_mod.topic_status(env.settings, env.topic))["outline"] == "ошибка"


def render_text(fn: Callable[..., None], *args: Any) -> str:
    console = Console(record=True, width=140, force_terminal=False)
    fn(*args, console=console)
    return console.export_text()


def test_print_status_and_build(env: Env) -> None:
    text = render_text(build_mod.print_status, build_mod.topic_status(env.settings, env.topic))
    assert "Метрические методы" in text and "Слайды лекции" in text
    assert "извлечение не одобрено" in text and "не выполнялась" in text
    assert "Покрытие: ещё не считалось" in text

    stopped = build_mod.build_topic(env.settings, env.topic)
    text = render_text(build_mod.print_build, stopped)
    assert "Остановка на review gate" in text and "h0lon approve" in text

    build_mod.approve_topic(env.settings, env.topic)
    done = build_mod.build_topic(env.settings, env.topic)
    text = render_text(build_mod.print_build, done)
    assert "Структура темы (S1)" in text and "готово" in text
    assert "master.pdf" in text and "100 %" in text
    text = render_text(build_mod.print_status, build_mod.topic_status(env.settings, env.topic))
    assert "извлечение одобрено" in text and "Покрытие: 100 %" in text

    env.synth.fail_stage = "global"
    env.synth.worked.clear()
    failed = build_mod.build_topic(env.settings, env.topic, review=False, force=True)
    text = render_text(build_mod.print_build, failed)
    assert "остановлена" in text and "фейковая ошибка стадии global" in text


# ---------------------------------------------------------------- CLI


@pytest.fixture
def cli(env: Env, tmp_path: Path) -> Callable[..., Any]:
    config = tmp_path / "h0lon.toml"
    config.write_text(
        "[general]\n"
        f"workspaces = '{env.settings.general.workspaces_dir}'\n"
        f"state_dir = '{tmp_path / 'state'}'\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    def invoke(*args: str) -> Any:
        return runner.invoke(app, ["-c", str(config), *args])

    return invoke


def test_cli_build_exit_codes(env: Env, cli: Callable[..., Any]) -> None:
    topic = str(env.topic)
    res = cli("build", topic, "--json")
    assert res.exit_code == 3, res.output
    data = json.loads(res.stdout)
    assert data["ok"] is False and data["stopped_at"] == "review"

    res = cli("approve", topic)
    assert res.exit_code == 0, res.output
    assert "одобрено" in res.output and "S1" in res.output

    res = cli("build", topic, "--json")
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["ok"] is True and data["master_pdf"].endswith("master.pdf")
    assert data["coverage"]["total"] == 9

    res = cli("build", topic)  # human output
    assert res.exit_code == 0 and "master.pdf" in res.output

    env.synth.fail_stage = "coverage"
    res = cli("build", topic, "--force", "--json")
    assert res.exit_code == 1
    assert json.loads(res.stdout)["stopped_at"] == "coverage"

    res = cli("build", topic, "--no-review", "--from", "bogus")
    assert res.exit_code == 2 and "Ошибка:" in res.output and "bogus" in res.output
    res = cli("build", topic, "--backend", "gpt")
    assert res.exit_code == 2
    res = cli("build", "нет/такой-темы")
    assert res.exit_code == 2 and "Тема не найдена" in res.output


def test_cli_build_no_review_and_from(env: Env, cli: Callable[..., Any]) -> None:
    topic = str(env.topic)
    res = cli("build", topic, "--no-review", "--json")
    assert res.exit_code == 0, res.output
    res = cli("build", topic, "--no-review", "--from", "render", "--json")
    assert res.exit_code == 0
    stages = {s["stage"]: s["cached"] for s in json.loads(res.stdout)["stages"]}
    assert stages["assemble"] is True and stages["render"] is False


def test_cli_approve_errors(env: Env, cli: Callable[..., Any]) -> None:
    rec = next(r for r in list_sources(env.topic) if r.id == "S1")
    update_source(env.topic, rec.model_copy(update={"status": "failed"}))
    res = cli("approve", str(env.topic))
    assert res.exit_code == 2
    assert "Ошибка:" in res.output and "S1" in res.output and "h0lon extract" in res.output


def test_cli_status(env: Env, cli: Callable[..., Any]) -> None:
    topic = str(env.topic)
    res = cli("status", topic)
    assert res.exit_code == 0, res.output
    assert "Метрические методы" in res.output and "Review gate" in res.output
    cli("build", topic, "--no-review")
    res = cli("status", topic, "--json")
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["stages"]["render"]["state"] in ("готово", "из кэша")
    assert data["coverage"]["covered"] == 9
    res = cli("status", "нет-темы")
    assert res.exit_code == 2


# ---------------------------------------------------------------- end to end with XeLaTeX


@pytest.mark.needs_xelatex
def test_end_to_end_with_real_assemble_and_render(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")
    topic = make_topic(settings)
    FakeSynth().install(monkeypatch)
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", FakeExtract())

    def no_agents(*a: Any, **k: Any) -> Any:
        raise AssertionError("no repair expected")

    monkeypatch.setattr("h0lon.synth.render.run_agent", no_agents)
    result = build_mod.build_topic(settings, topic, review=False)
    assert result.ok, result.message
    pdf = topic / "master.pdf"
    report = check_pdf(pdf, expected_title=["Метрические методы классификации"])
    assert report.ok and report.pages >= 3 and not report.not_embedded
    render_stage = next(s for s in result.stages if s.stage == "render")
    assert render_stage.details["engine"] == "xelatex" and not render_stage.details["fallback"]
    assert not (topic / "master.build").exists()
    assert git(topic, "log", "-1", "--format=%s").startswith("Сборка мастера: ")
    # the second build changes nothing
    again = build_mod.build_topic(settings, topic, review=False)
    assert again.ok and next(s for s in again.stages if s.stage == "render").cached


def test_print_build_caps_long_warning_lists() -> None:
    warnings = [f"предупреждение {i}" for i in range(10)]
    result = BuildResult(
        ok=True,
        topic_dir=Path("."),
        stages=[StageResult(stage="sections", ok=True, warnings=warnings)],
        message="Готово: master.pdf.",
    )
    text = render_text(build_mod.print_build, result)
    assert "предупреждение 5" in text and "предупреждение 6" not in text
    assert "и ещё 4" in text and "--json" in text


REAL_TOPIC = os.environ.get("H0LON_M2_REAL_TOPIC")


@pytest.mark.needs_xelatex
@pytest.mark.skipif(not REAL_TOPIC, reason="H0LON_M2_REAL_TOPIC не задан")
def test_end_to_end_on_a_copy_of_the_real_topic(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fake S1–S5 on real Source Docs, real assemble and render: master.pdf passes pdfcheck."""
    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")
    topic = tmp_path / "real"
    shutil.copytree(str(REAL_TOPIC), topic, ignore=shutil.ignore_patterns("runs", "synthesis"))
    sources = [r for r in list_sources(topic) if r.status == "extracted"]
    assert len(sources) >= 2
    FakeSynth(leaf_size=18, chapter_size=4).install(monkeypatch)
    monkeypatch.setattr("h0lon.extract.pipeline.extract_topic", FakeExtract())

    def no_agents(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no agent run expected: the fakes replace every agent stage")

    monkeypatch.setattr("h0lon.synth.render.run_agent", no_agents)
    result = build_mod.build_topic(settings, topic, review=False)
    assert result.ok, result.message
    assert result.coverage is not None and result.coverage.ratio == 1.0
    report = check_pdf(topic / "master.pdf")
    assert report.ok and report.pages >= 10 and not report.not_embedded
    master = (topic / "master.md").read_text(encoding="utf-8")
    assert all(f"id: {s.id}" in master for s in sources)
    assert not (topic / "master.build").exists()
