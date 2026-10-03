"""S6 render: success, XeLaTeX failure repaired by a (fake) agent, rejected repairs, HTML fallback.

The repair loop is tested against a fake `render_document` (fast, deterministic) and, for the
real thing, against XeLaTeX (`needs_xelatex`). Agents are never run: `run_agent` is replaced
by a fake that edits the seed copy of master.md in a real task bundle.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from h0lon import tools
from h0lon.agents import Usage, create_bundle, validate_outputs
from h0lon.agents.runner import RunResult, restore_seed
from h0lon.config import Settings
from h0lon.render import RenderReport
from h0lon.synth import render as render_mod
from h0lon.synth.common import prompt_body, read_meta
from h0lon.synth.model import BuildContext

GOOD = """---
title: "Проверка сборки"
course: "Тест"
---

# Раздел {#sec:s01}

Пусть $x\\in\\mathbb{R}$ и $f(x)=x^2$.[[S1:s2]]
<!-- src: S1.b001 -->

Второй абзац с достаточным количеством слов, чтобы объём текста был заметным: каждая строка
добавляет символы, а допуск по объёму составляет три процента от всего текста мастера.[[S1:s3]]
<!-- src: S1.b002 S1.b003 -->

## Подраздел {#sec:s01-01}

Формула $\\rho(a,b)\\ge 0$ верна для любой метрики.[[P1:p4]]
<!-- src: P1.b001 -->
"""
BROKEN = GOOD.replace("\\mathbb{R}", "\\R")


def fix_r(text: str) -> str:
    return text.replace("\\R", "\\mathbb{R}")


# ---------------------------------------------------------------- fakes


@dataclass
class FakeAgent:
    """Replacement of render.run_agent: a real bundle, a scripted edit, real validation."""

    behaviors: list[Callable[[str], str | None]]
    calls: list[SimpleNamespace] = field(default_factory=list)

    def __call__(self, ctx: BuildContext, **kw: Any) -> RunResult:
        bundle = create_bundle(
            ctx.topic_dir / "runs",
            stage=kw["stage"],
            task=kw["task"],
            contract=kw["contract"],
            inputs=kw.get("inputs", ()),
            seed=kw.get("seed"),
        )
        restore_seed(bundle)
        out = bundle.out_dir / "master.md"
        text = out.read_text(encoding="utf-8")
        behavior = self.behaviors[min(len(self.calls), len(self.behaviors) - 1)]
        self.calls.append(
            SimpleNamespace(
                stage=kw["stage"],
                tier=kw["tier"],
                task=kw["task"],
                text=text,
                errors=(bundle.inputs_dir / "errors.md").read_text(encoding="utf-8"),
                contract=kw["contract"],
            )
        )
        new = behavior(text)
        if new is not None:
            out.write_text(new, encoding="utf-8")
        problems = validate_outputs(bundle.out_dir, bundle.contract)
        return RunResult(
            ok=not problems,
            bundle=bundle,
            backend_used="fake",
            attempts=[],
            usage_total=Usage(),
            final_text="",
            problems=problems,
        )


@dataclass
class FakeRenderer:
    """Replacement of render.render_document: fails on `\\R`, writes a stub PDF otherwise."""

    html_ok: bool = True
    calls: list[tuple[str, bool]] = field(default_factory=list)

    def __call__(
        self,
        src: Path,
        *,
        settings: Settings,
        out_dir: Path | None = None,
        engine: str = "auto",
        keep_build: bool = False,
        **_: Any,
    ) -> RenderReport:
        self.calls.append((engine, keep_build))
        out = Path(out_dir or src.parent)
        text = src.read_text(encoding="utf-8")
        build = out / f"{src.stem}.build"
        build.mkdir(exist_ok=True)
        (build / ".h0lon-build").write_text("x", encoding="utf-8")
        tex = build / f"{src.stem}.tex"
        tex.write_text(
            "\n".join(
                ["% preamble"] * 11 + [r"Пусть $x\in\R$ и $f(x)=x^2$." if "\\R" in text else "ok"]
            )
            + "\n% tail\n",
            encoding="utf-8",
        )
        (build / f"{src.stem}.log").write_text("log", encoding="utf-8")
        report = RenderReport(
            ok=False,
            pdf=None,
            engine_used=None,
            fallback_used=False,
            tex=tex if keep_build else None,
            build_dir=build if keep_build else None,
            passes=1,
        )
        if not keep_build:
            shutil.rmtree(build, ignore_errors=True)
        if engine == "xelatex":
            if re.search(r"!\[[^\]]*\]\(https?://", text):
                report.errors = ["XeLaTeX: Не найден файл LaTeX: https://example.org/a.png"]
                return report
            if "\\R" in text:
                report.errors = [
                    "XeLaTeX: master.tex:12: Undefined control sequence. (l.12 Пусть $x\\in\\R)"
                ]
                return report
            ok, used = True, "xelatex"
        else:
            ok, used = self.html_ok, "html"
            if not ok:
                report.errors = ["HTML → браузер: не найден браузер"]
                return report
        pdf = out / f"{src.stem}.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        report.ok, report.pdf, report.engine_used = True, pdf, used
        return report


@pytest.fixture
def ctx(tmp_path: Path, settings: Settings) -> BuildContext:
    topic = tmp_path / "topic"
    (topic / "synthesis").mkdir(parents=True)
    return BuildContext(topic_dir=topic, settings=settings)


@pytest.fixture
def master(ctx: BuildContext) -> Path:
    path = ctx.topic_dir / "master.md"
    path.write_text(BROKEN, encoding="utf-8")
    return path


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[FakeAgent, FakeRenderer]]:
    """Install fakes: `fakes([behaviors], html_ok=True)` returns (agent, renderer).

    `agent` replaces the default FakeAgent; `wrap` is called with the renderer and returns the
    function that is installed as `render_document` (to tweak its reports).
    """

    def install(
        behaviors: list[Callable[[str], str | None]],
        *,
        html_ok: bool = True,
        agent: FakeAgent | None = None,
        wrap: Callable[[FakeRenderer], Callable[..., RenderReport]] | None = None,
    ) -> tuple[FakeAgent, FakeRenderer]:
        agent = agent or FakeAgent(behaviors)
        renderer = FakeRenderer(html_ok=html_ok)
        monkeypatch.setattr(render_mod, "run_agent", agent)
        monkeypatch.setattr(render_mod, "render_document", wrap(renderer) if wrap else renderer)
        monkeypatch.setattr(render_mod.tools, "find_xelatex", lambda override="": Path("xelatex"))
        return agent, renderer

    return install


# ---------------------------------------------------------------- pure helpers


def test_check_fix_accepts_and_rejects() -> None:
    check = render_mod.check_fix
    assert check(GOOD, BROKEN, fix_r(BROKEN)) is None
    assert check(GOOD, BROKEN, BROKEN) == "агент не изменил файл"
    lost = fix_r(BROKEN).replace("<!-- src: S1.b002 S1.b003 -->", "<!-- src: S1.b002 -->")
    assert "S1.b003" in (check(GOOD, BROKEN, lost) or "")
    extra = fix_r(BROKEN).replace("P1.b001", "P1.b001 P1.b777")
    assert "P1.b777" in (check(GOOD, BROKEN, extra) or "")
    anchors = fix_r(BROKEN).replace("[[S1:s3]]", "")
    assert "якоря" in (check(GOOD, BROKEN, anchors) or "")
    front = fix_r(BROKEN).replace('title: "Проверка сборки"', 'title: "Другое"')
    assert "front matter" in (check(GOOD, BROKEN, front) or "")
    shrunk = fix_r(BROKEN).replace("добавляет символы, а допуск по объёму составляет", "")
    reason = check(GOOD, BROKEN, shrunk) or ""
    assert "объём" in reason and "%" in reason
    # a change within ±3 % is fine
    near = fix_r(BROKEN).replace("Второй абзац", "Второй  абзац!")
    assert check(GOOD, BROKEN, near) is None


def test_fixable_errors() -> None:
    errors = [
        "XeLaTeX: master.tex:12: Undefined control sequence.",
        "XeLaTeX: Не найден файл LaTeX: mdframed.sty",
        "XeLaTeX: XeLaTeX не найден: установите MiKTeX",
        "Шрифты не встроены в PDF: X",
    ]
    assert render_mod.fixable_errors(errors) == [errors[0]]


def test_build_errors_md_finds_places(tmp_path: Path) -> None:
    tex = tmp_path / "master.tex"
    lines = [f"line {i}" for i in range(1, 40)]
    lines[19] = r"Формула $\rho(a,b)\R$ верна для любой метрики Минковского"
    tex.write_text("\n".join(lines), encoding="utf-8")
    master = (
        "# Заголовок\n\nНичего общего.\n\n"
        "Формула $\\rho(a,b)\\R$ верна для любой метрики Минковского.[[P1:p4]]\n"
        "<!-- src: P1.b001 -->\n\nХвост.\n"
    )
    md = render_mod.build_errors_md(
        ["XeLaTeX: master.tex:20: Undefined control sequence. (l.20 \\R)", "Pandoc: без номера"],
        tex,
        master,
        attempt=2,
    )
    assert "Попытка исправления 2 из 3" in md
    assert "1. master.tex:20: Undefined control sequence." in md  # engine prefix stripped
    assert ">>   20 Формула" in md  # the error line is marked
    assert "строки 15–25" in md
    assert "Похожее место в `master.md`, строки 5–6:" in md  # found via \R and via words
    assert "    5 Формула" in md
    assert "Номер строки `.tex` в сообщении не указан" in md  # the second error
    assert md.endswith("\n")
    # no tex file: the errors are still listed
    plain = render_mod.build_errors_md(["boom"], None, master, attempt=1)
    assert "1. boom" in plain and "```tex" not in plain


# ---------------------------------------------------------------- stage logic (fake renderer)


def test_success_without_agent(ctx: BuildContext, master: Path, fakes: Any) -> None:
    master.write_text(GOOD, encoding="utf-8")
    agent, renderer = fakes([fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and not result.cached and result.agent_runs == 0
    assert (ctx.topic_dir / "master.pdf").is_file()
    assert renderer.calls == [("xelatex", True)] and agent.calls == []
    assert not (ctx.topic_dir / "master.build").exists()  # build directory removed
    meta = read_meta(ctx, "render")
    assert meta and meta["ok"] and meta["agent_runs"] == 0
    assert meta["prompt"] == "fixlatex@1.0" and meta["key"] == meta["key_after"]
    assert result.details["engine"] == "xelatex" and result.details["fixes"] == 0


def test_xelatex_error_is_repaired(ctx: BuildContext, master: Path, fakes: Any) -> None:
    agent, renderer = fakes([fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 1
    assert result.details["fixes"] == 1 and result.details["fallback"] is False
    assert [c[0] for c in renderer.calls] == ["xelatex", "xelatex"]
    assert master.read_text(encoding="utf-8") == fix_r(BROKEN)  # master.md overwritten
    call = agent.calls[0]
    assert call.stage == "fixlatex" and call.tier == "light"
    assert call.task == prompt_body("fixlatex")
    assert call.text == BROKEN  # the agent got the master as the seed of out/
    assert "Undefined control sequence" in call.errors
    assert "Пусть $x\\in\\R$" in call.errors  # .tex fragment and master place
    assert call.contract.files[0].path == "master.md"
    # the runs/ directory has the bundle with errors.md in inputs/
    bundles = list((ctx.topic_dir / "runs").iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "inputs" / "errors.md").is_file()
    meta = read_meta(ctx, "render")
    assert meta and meta["key"] != meta["key_after"]  # the master changed under the stage


def test_cached_by_master_before_or_after_fix(ctx: BuildContext, master: Path, fakes: Any) -> None:
    agent, renderer = fakes([fix_r])
    first = render_mod.render_master(ctx, master)
    assert first.ok and not first.cached
    n = len(renderer.calls)
    # the next build: assemble is cached, master.md is the repaired one
    second = render_mod.render_master(ctx, master)
    assert second.ok and second.cached and len(renderer.calls) == n and len(agent.calls) == 1
    assert second.details["fixes"] == 1
    # the master as assembled again (before the repair) is the same stage input
    master.write_text(BROKEN, encoding="utf-8")
    third = render_mod.render_master(ctx, master)
    assert third.cached and len(renderer.calls) == n
    # another text, or --force: rebuilt
    master.write_text(GOOD + "\nЕщё.\n", encoding="utf-8")
    assert not render_mod.render_master(ctx, master).cached
    ctx.force = True
    assert not render_mod.render_master(ctx, master).cached
    # a missing PDF is rebuilt even with a matching key
    ctx.force = False
    (ctx.topic_dir / "master.pdf").unlink()
    assert not render_mod.render_master(ctx, master).cached


def test_fix_that_loses_src_is_rolled_back(ctx: BuildContext, master: Path, fakes: Any) -> None:
    def spoil(text: str) -> str:
        return fix_r(text).replace("<!-- src: P1.b001 -->", "")

    agent, renderer = fakes([spoil, fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 2 and result.details["fixes"] == 1
    assert any("отклонено" in w and "P1.b001" in w for w in result.warnings)
    final = master.read_text(encoding="utf-8")
    assert "<!-- src: P1.b001 -->" in final and "\\mathbb{R}" in final
    # the second attempt started from the original master, not from the spoiled copy
    assert agent.calls[1].text == BROKEN
    # the rejected text never reached the renderer
    assert [c[0] for c in renderer.calls] == ["xelatex", "xelatex"]


def test_fix_that_changes_anchors_or_size_is_rejected(
    ctx: BuildContext, master: Path, fakes: Any
) -> None:
    _agent, _ = fakes(
        [
            lambda t: fix_r(t).replace("[[S1:s2]]", "[[S1:s9]]"),
            lambda t: fix_r(t).replace("Второй абзац", "Второй абзац " + "лишний текст " * 20),
            lambda t: t,  # no change at all
        ]
    )
    ctx.settings.render.fallback_html = False
    result = render_mod.render_master(ctx, master)
    assert not result.ok and result.agent_runs == 3
    kinds = " | ".join(result.warnings)
    assert "якоря" in kinds and "объём" in kinds and "не изменил" in kinds
    assert master.read_text(encoding="utf-8") == BROKEN  # nothing was applied
    assert any("Undefined control sequence" in e for e in result.errors)
    assert "fallback_html" in " ".join(result.errors)
    meta = read_meta(ctx, "render")
    assert meta and meta["ok"] is False


def test_three_failures_fall_back_to_html(ctx: BuildContext, master: Path, fakes: Any) -> None:
    _agent, renderer = fakes([lambda t: t])
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 3
    assert [c[0] for c in renderer.calls] == ["xelatex", "html"]
    assert renderer.calls[1] == ("html", False)
    assert result.details["engine"] == "html" and result.details["fallback"] is True
    assert any("запасным движком" in w for w in result.warnings)
    assert any("Undefined control sequence" in w for w in result.warnings)
    assert (ctx.topic_dir / "master.pdf").is_file()
    assert (ctx.synth_dir / "render.xelatex.log").read_text(encoding="utf-8") == "log"
    assert not (ctx.topic_dir / "master.build").exists()
    meta = read_meta(ctx, "render")
    assert meta and meta["ok"] and meta["agent_runs"] == 3


def test_html_fallback_failure(ctx: BuildContext, master: Path, fakes: Any) -> None:
    fakes([lambda t: t], html_ok=False)
    result = render_mod.render_master(ctx, master)
    assert not result.ok and result.agent_runs == 3
    assert any("браузер" in e for e in result.errors)
    assert not (ctx.topic_dir / "master.pdf").exists()


def test_agent_failure_counts_as_attempt(ctx: BuildContext, master: Path, fakes: Any) -> None:
    class Failing(FakeAgent):
        def __call__(self, ctx: BuildContext, **kw: Any) -> RunResult:
            result = super().__call__(ctx, **kw)
            result.ok = False
            result.problems = ["таймаут агента"]
            return result

    agent = Failing([fix_r])
    _, renderer = fakes([], agent=agent)
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.details["fallback"] is True  # fell back to HTML
    assert result.agent_runs == 3 and len(agent.calls) == 3
    assert sum("таймаут агента" in w for w in result.warnings) == 3
    assert [c[0] for c in renderer.calls] == ["xelatex", "html"]  # failed repairs are not rendered
    assert master.read_text(encoding="utf-8") == BROKEN


def test_unfixable_error_skips_the_agent(ctx: BuildContext, master: Path, fakes: Any) -> None:
    def no_tex(renderer: FakeRenderer) -> Callable[..., RenderReport]:
        def render(src: Path, **kw: Any) -> RenderReport:
            report = renderer(src, **kw)
            if kw["engine"] == "xelatex":
                report.errors = ["XeLaTeX: Не найден файл LaTeX: mdframed.sty"]
            return report

        return render

    agent, _ = fakes([fix_r], wrap=no_tex)
    result = render_mod.render_master(ctx, master)
    assert agent.calls == [] and result.agent_runs == 0
    assert result.ok and result.details["fallback"] is True
    assert any("не связаны с разметкой" in w for w in result.warnings)


def test_html_engine_from_settings(ctx: BuildContext, master: Path, fakes: Any) -> None:
    ctx.settings.render.engine = "html"
    agent, renderer = fakes([fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and renderer.calls == [("html", False)] and agent.calls == []
    assert result.details["engine"] == "html"


def test_missing_master(ctx: BuildContext) -> None:
    result = render_mod.render_master(ctx, ctx.topic_dir / "master.md")
    assert not result.ok and "master.md" in result.errors[0]


IMAGE_MASTER = GOOD.replace(
    "Второй абзац",
    "Рисунок ниже.\n\n"
    "![Схема kNN](https://example.org/a_(1).png){#fig:knn width=50%}\n\n"
    "Второй абзац",
)


def test_replace_remote_images() -> None:
    text = (
        "До ![Схема kNN](https://example.org/a_(1).png){#fig:knn width=50%} после\n"
        '![](https://example.org/b.png "Заголовок")\n'
        "![локальный](img/c.png) и [ссылка](https://example.org)\n"
        "```md\n![в коде](https://example.org/d.png)\n```\n"
        "Конец ![x](http://example.org/e.png)"
    )
    new, count = render_mod.replace_remote_images(text)
    assert count == 3
    assert "До [Рисунок: Схема kNN](https://example.org/a_(1).png){#fig:knn} после" in new
    assert "\n[Рисунок](https://example.org/b.png)\n" in new
    assert "![локальный](img/c.png)" in new  # local images are not touched
    assert "![в коде](https://example.org/d.png)" in new  # neither is code
    assert "Конец [Рисунок: x](http://example.org/e.png)" in new
    assert render_mod.replace_remote_images("Нет рисунков.") == ("Нет рисунков.", 0)
    assert render_mod.has_remote_image_error(["XeLaTeX: Не найден файл LaTeX: https://a/b.png"])
    assert not render_mod.has_remote_image_error(["Не найден файл LaTeX: img/a.png"])


def test_remote_images_are_replaced_without_agent(
    ctx: BuildContext, master: Path, fakes: Any
) -> None:
    master.write_text(IMAGE_MASTER, encoding="utf-8")
    agent, renderer = fakes([fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 0 and agent.calls == []
    assert [c[0] for c in renderer.calls] == ["xelatex", "xelatex"]
    assert result.details["image_links"] == 1 and result.details["fallback"] is False
    assert any("заменены ссылками" in w for w in result.warnings)
    text = master.read_text(encoding="utf-8")
    assert "[Рисунок: Схема kNN](https://example.org/a_(1).png){#fig:knn}" in text
    assert "![" not in text


def test_remote_images_then_markup_error(ctx: BuildContext, master: Path, fakes: Any) -> None:
    broken = IMAGE_MASTER.replace("\\mathbb{R}", "\\R")
    master.write_text(broken, encoding="utf-8")
    agent, renderer = fakes([fix_r])
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 1
    assert [c[0] for c in renderer.calls] == ["xelatex", "xelatex", "xelatex"]
    # the repair is compared with the text after the image links, not with the original
    assert "[Рисунок: Схема kNN]" in agent.calls[0].text
    assert "\\mathbb{R}" in master.read_text(encoding="utf-8")


# ---------------------------------------------------------------- real XeLaTeX


def _real_tools_or_skip() -> None:
    if tools.find_xelatex() is None:
        pytest.skip("XeLaTeX not found")
    if tools.find_pandoc() is None:
        pytest.skip("pandoc not found")


@pytest.mark.needs_xelatex
def test_real_success_and_cache(ctx: BuildContext, master: Path, monkeypatch: Any) -> None:
    _real_tools_or_skip()
    master.write_text(GOOD, encoding="utf-8")

    def no_agents(*a: Any, **k: Any) -> Any:
        raise AssertionError("the agent must not run for a valid master")

    monkeypatch.setattr(render_mod, "run_agent", no_agents)
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.agent_runs == 0, result.errors
    pdf = ctx.topic_dir / "master.pdf"
    assert pdf.is_file() and result.details["pages"] >= 1
    assert result.details["engine"] == "xelatex"
    assert not (ctx.topic_dir / "master.build").exists()
    assert render_mod.render_master(ctx, master).cached


@pytest.mark.needs_xelatex
def test_real_xelatex_failure_is_repaired(
    ctx: BuildContext, master: Path, monkeypatch: Any
) -> None:
    _real_tools_or_skip()
    agent = FakeAgent([fix_r])
    monkeypatch.setattr(render_mod, "run_agent", agent)
    result = render_mod.render_master(ctx, master)
    assert result.ok, result.errors
    assert result.agent_runs == 1 and result.details["fixes"] == 1
    assert result.details["engine"] == "xelatex" and not result.details["fallback"]
    # what the agent received: the real XeLaTeX error, the .tex fragment, the master place
    errors = agent.calls[0].errors
    assert "Undefined control sequence" in errors
    assert "```tex" in errors and "\\R" in errors
    assert "Похожее место в `master.md`" in errors
    assert master.read_text(encoding="utf-8") == fix_r(BROKEN)
    assert (ctx.topic_dir / "master.pdf").stat().st_size > 1000
    assert not (ctx.topic_dir / "master.build").exists()
    assert json.loads((ctx.synth_dir / "render.meta.json").read_text(encoding="utf-8"))["ok"]


@pytest.mark.needs_xelatex
@pytest.mark.needs_browser
def test_real_three_failures_fall_back_to_html(
    ctx: BuildContext, master: Path, monkeypatch: Any
) -> None:
    _real_tools_or_skip()
    if tools.find_browser("auto") is None:
        pytest.skip("Edge/Chrome/Chromium not found")
    agent = FakeAgent([lambda t: t.replace("Второй абзац", "Второй  абзац")])  # fixes nothing
    monkeypatch.setattr(render_mod, "run_agent", agent)
    result = render_mod.render_master(ctx, master)
    assert result.ok, result.errors
    assert result.agent_runs == 3 and result.details["fallback"] is True
    assert result.details["engine"] == "html"
    assert any("запасным движком" in w for w in result.warnings)
    assert (ctx.topic_dir / "master.pdf").stat().st_size > 1000


@pytest.mark.needs_xelatex
def test_real_remote_image_becomes_a_link(
    ctx: BuildContext, master: Path, monkeypatch: Any
) -> None:
    _real_tools_or_skip()
    master.write_text(IMAGE_MASTER, encoding="utf-8")

    def no_agents(*a: Any, **k: Any) -> Any:
        raise AssertionError("an image URL is not a job for the agent")

    monkeypatch.setattr(render_mod, "run_agent", no_agents)
    result = render_mod.render_master(ctx, master)
    assert result.ok, result.errors
    # the render filter already turns the remote picture into a link: no repair needed
    assert result.details.get("image_links", 0) == 0 and result.details["engine"] == "xelatex"
    assert not result.details["fallback"]
    assert (ctx.topic_dir / "master.pdf").stat().st_size > 1000


def test_agent_launch_error_is_a_warning(
    ctx: BuildContext, master: Path, fakes: Any, monkeypatch: Any
) -> None:
    fakes([fix_r])

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("Входной файл не найден")

    monkeypatch.setattr(render_mod, "run_agent", boom)
    result = render_mod.render_master(ctx, master)
    assert result.ok and result.details["fallback"] is True and result.agent_runs == 3
    assert sum("агент не запущен (FileNotFoundError" in w for w in result.warnings) == 3
