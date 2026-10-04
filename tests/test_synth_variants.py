"""Variations of the master: presets, cleaning of agent output, caches, staleness, failures, CLI.

Agents never run: `run_agent` is replaced by a fake that works in a real task bundle (inputs are
copied, the contract is validated); the render is a stub that writes a PDF. Real XeLaTeX runs only
in tests/test_render_compact_template.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from h0lon.agents import Usage, create_bundle, validate_outputs
from h0lon.agents.runner import RunResult
from h0lon.cli import app
from h0lon.config import Settings
from h0lon.render import RenderReport
from h0lon.synth import variants as V
from h0lon.workspace import create_topic

MASTER = r"""---
title: "Метрические методы"
subtitle: "Мастер-конспект темы"
course: "Машинное обучение"
date: "2026-10-04"
author: "H0lon"
sources:
  - id: S1
    kind: slides
    title: "Слайды"
    units: {slides: 12}
---

# Метрики {#sec:s01}

Расстояние Минковского задаётся формулой $\rho_p$.[[S1:s3]]
<!-- src: S1.b001 -->

::: {.definition #def:metric title="Метрика"}
Функция $\rho$ называется метрикой, см. [ядра](#sec:s02).
:::

![Схема](extracted/S1/figures/knn.png)

# Ядра {#sec:s02}

Ядро — функция $K$.[[S1:s5]]
"""

# What a (fake) agent returns: it keeps what it was told not to keep.
AGENT_TEXT = r"""<!-- заметка агента -->
# Кратко {#sec:brief}

Главное: метрика $\rho$ задаёт расстояние.[[S1:s3]] <!-- src: S1.b001 -->

См. [ядра](#sec:s02) и [метрику](#def:metric) и [раздел](#sec:brief).

![Схема](extracted/S1/figures/knn.png)

```
[[S1:s9]] <!-- stays in code -->
```
"""


# ---------------------------------------------------------------- fakes


@dataclass
class FakeAgent:
    """Replacement of variants.run_agent: a real bundle, a scripted answer, real validation."""

    answer: Callable[[str, str | None], str | None] = lambda master, request: AGENT_TEXT
    calls: list[SimpleNamespace] = field(default_factory=list)
    crash: bool = False

    def __call__(self, ctx: Any, **kw: Any) -> RunResult:
        if self.crash:
            raise RuntimeError("агент недоступен")
        bundle = create_bundle(
            ctx.topic_dir / "runs",
            stage=kw["stage"],
            task=kw["task"],
            contract=kw["contract"],
            inputs=kw.get("inputs", ()),
            seed=kw.get("seed"),
        )
        master = (bundle.inputs_dir / "master.md").read_text(encoding="utf-8")
        req_path = bundle.inputs_dir / "request.md"
        request = req_path.read_text(encoding="utf-8") if req_path.is_file() else None
        self.calls.append(
            SimpleNamespace(
                stage=kw["stage"],
                tier=kw["tier"],
                task=kw["task"],
                master=master,
                request=request,
                backend=ctx.backend,
                inputs=sorted(p.name for p in bundle.inputs_dir.iterdir()),
            )
        )
        text = self.answer(master, request)
        if text is not None:
            (bundle.out_dir / "variant.md").write_text(text, encoding="utf-8")
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
class FakeRender:
    """Replacement of variants.render_document: a stub PDF next to the source."""

    ok: bool = True
    fallback: bool = False
    warnings: list[str] = field(default_factory=lambda: ["Строки выходят за поле (Overfull hbox)"])
    calls: list[SimpleNamespace] = field(default_factory=list)

    def __call__(
        self, src: Path, *, settings: Settings, out_dir: Path | None = None, **kw: Any
    ) -> RenderReport:
        self.calls.append(
            SimpleNamespace(
                src=Path(src),
                text=Path(src).read_text(encoding="utf-8"),
                template=kw.get("template"),
                out_dir=out_dir,
            )
        )
        if not self.ok:
            return RenderReport(
                ok=False,
                pdf=None,
                engine_used=None,
                fallback_used=False,
                tex=None,
                build_dir=None,
                passes=0,
                errors=["XeLaTeX: ошибка разметки"],
            )
        pdf = Path(out_dir or Path(src).parent) / f"{Path(src).stem}.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub " + str(len(self.calls)).encode())
        return RenderReport(
            ok=True,
            pdf=pdf,
            engine_used="html" if self.fallback else "xelatex",
            fallback_used=self.fallback,
            tex=None,
            build_dir=None,
            passes=2,
            warnings=list(self.warnings),
            checks=SimpleNamespace(pages=3),  # type: ignore[arg-type]
        )


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(general={"git_per_topic": False})


@pytest.fixture
def topic(settings: Settings) -> Path:
    path = create_topic(settings, title="Метрические методы", course="Курс", slug="metricheskie")
    (path / "master.md").write_text(MASTER, encoding="utf-8")
    return path


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> FakeAgent:
    fake = FakeAgent()
    monkeypatch.setattr(V, "run_agent", fake)
    return fake


@pytest.fixture
def render(monkeypatch: pytest.MonkeyPatch) -> FakeRender:
    fake = FakeRender()
    monkeypatch.setattr(V, "render_document", fake)
    return fake


def read_meta(topic: Path, slug: str) -> dict[str, Any]:
    return json.loads((topic / "variants" / slug / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- presets and requests


def test_presets_follow_decision_a12() -> None:
    assert V.PRESETS == {
        "brief": "light",
        "cheatsheet": "light",
        "study": "strong",
        "custom": "strong",
    }
    assert V.PRESET_TEMPLATES == {"cheatsheet": "a4-compact"}
    assert set(V.PRESET_TITLES) == {*V.PRESETS, "template"}


def test_prompts_exist_for_every_agent_preset() -> None:
    from h0lon.synth.common import prompt_id

    for preset in V.PRESETS:
        assert prompt_id(f"variant_{preset}").startswith(f"variant_{preset}@")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("brief", None, None), ("brief", None, None)),
        (("Study", "  ", None), ("study", None, None)),
        (("custom", " сделай  вопросы ", None), ("custom", "сделай  вопросы", None)),
        ((None, "вопросы к экзамену", None), ("custom", "вопросы к экзамену", None)),
        ((None, None, "a4-compact"), ("template", None, "a4-compact")),
        (("template:a4-compact", None, None), ("template", None, "a4-compact")),
        (("cheatsheet", None, "a4-notes"), ("cheatsheet", None, "a4-notes")),
    ],
)
def test_normalise_request(args: tuple, expected: tuple) -> None:
    assert V.normalise_request(*args) == expected


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ((None, None, None), "Укажите пресет"),
        (("bogus", None, None), "Неизвестный пресет «bogus»"),
        (("custom", None, None), "нужен запрос"),
        (("custom", "я" * 5000, None), "слишком длинный"),
        (("brief", "запрос", None), "только для пресета custom"),
        (("template", None, None), "укажите шаблон"),
    ],
)
def test_normalise_request_errors(args: tuple, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        V.normalise_request(*args)


def test_slugs(topic: Path) -> None:
    assert V.variant_slug(topic, "brief", None, None) == "brief"
    assert V.variant_slug(topic, "template", None, "a4-compact") == "template-a4-compact"
    slug = V.variant_slug(topic, "custom", "Сделай вопросы к экзамену по теме, пожалуйста", None)
    assert slug == "custom-sdelay-voprosy-k-ekzamenu"
    assert V.variant_slug(topic, "custom", "!!!", None) == "custom"
    assert V.variant_slug(topic, "custom", "你好", None) == "custom"


# ---------------------------------------------------------------- cleaning of the agent text


def test_clean_variant_text() -> None:
    text = "\ufeff---\ntitle: Чужой\n---\n" + AGENT_TEXT
    out = V.clean_variant_text(text)
    assert out.startswith("# Кратко")
    assert "[[S1:s3]]" not in out and "заметка агента" not in out and "src:" not in out
    assert "title: Чужой" not in out and "задаёт расстояние." in out
    # code is left as it is
    assert "[[S1:s9]] <!-- stays in code -->" in out
    # a link to an id that the text does not define loses the link, the others stay
    assert "См. ядра и метрику и [раздел](#sec:brief)." in out
    assert out.endswith("\n") and "\n\n\n" not in out


def test_clean_keeps_a_horizontal_rule_that_is_not_front_matter() -> None:
    text = "---\n\nТекст до линии.\n\n---\n\n# Заголовок\n"
    assert V.clean_variant_text(text).startswith("---\n\nТекст до линии.")


def test_dangling_xrefs_keep_pandoc_ids() -> None:
    text = "# Раздел\n\nСм. [Раздел](#razdel), [другое](#sec:nope), [ок](#sec:ok).\n\n"
    text += "## Ок {#sec:ok}\n"
    out = V.drop_dangling_xrefs(text)
    assert "[Раздел](#razdel)" in out and "другое," in out and "[ок](#sec:ok)" in out


def test_unclosed_comment_is_removed() -> None:
    out = V.clean_variant_text("# З\n\n<!-- скрыто\n\nтекст\n")
    assert "<!--" not in out and "скрыто" in out and "текст" in out


def test_relocate_images() -> None:
    text = (
        "![a](extracted/S1/figures/a.png) ![u](https://x.org/y.png) ![l](../../keep.png)\n"
        '<img src="extracted/P1/b.png"> ![w](C:/abs/c.png) ![h](#anchor)\n'
        "```\n![c](extracted/in-code.png)\n```\n"
    )
    out = V.relocate_images(text)
    assert "![a](../../extracted/S1/figures/a.png)" in out
    assert "(https://x.org/y.png)" in out and "(../../keep.png)" in out
    assert '<img src="../../extracted/P1/b.png">' in out
    assert "(C:/abs/c.png)" in out and "(#anchor)" in out
    assert "![c](extracted/in-code.png)" in out


# ---------------------------------------------------------------- the agent presets


def test_brief_variant_end_to_end(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    events: list[str] = []
    res = V.run_variant(settings, topic, preset="brief", on_event=events.append)

    assert res.ok and not res.cached and not res.errors, res.errors
    assert res.slug == "brief" and res.preset == "brief" and res.agent_runs == 1
    assert res.variant_md == topic / "variants" / "brief" / "variant.md"
    assert res.variant_pdf == topic / "variants" / "brief" / "variant.pdf"
    assert res.variant_pdf.is_file() and res.engine == "xelatex" and res.pages == 3
    assert any("Overfull" in w for w in res.warnings)
    assert V.AGENT_START in events and V.RENDER_START in events
    assert events.index(V.AGENT_START) < events.index(V.RENDER_START)

    # what the agent got: a copy of the master, the light tier, the prompt of the preset
    (call,) = agent.calls
    assert call.stage == "variants" and call.tier == "light" and call.request is None
    assert call.master == MASTER and call.inputs == ["master.md"]
    assert call.task.lstrip().startswith("# Задача: краткий конспект")
    assert not (topic / "variants" / "brief" / "request.md").exists()

    # the text: front matter of the module, cleaned body, images moved to variants/brief/
    text = res.variant_md.read_text(encoding="utf-8")
    head, _, body = text[4:].partition("\n---\n")
    assert "title: Метрические методы — Кратко" in head and "subtitle: Кратко" in head
    assert "toc: false" in head  # a short document has no page of contents
    assert "course: Машинное обучение" in head and "author: H0lon" in head
    assert "id: S1" in head  # the sources of the master stay for the title page
    assert body.lstrip().startswith("# Кратко")
    assert "[[" not in body.replace("[[S1:s9]]", "") and "заметка" not in body
    assert "![Схема](../../extracted/S1/figures/knn.png)" in body

    assert render.calls[0].src == res.variant_md
    assert render.calls[0].template == settings.render.template  # a4-notes by default
    assert render.calls[0].out_dir == res.dir

    meta = read_meta(topic, "brief")
    assert meta["preset"] == "brief" and meta["tier"] == "light" and meta["prompt"] is None
    assert meta["prompt_version"] == "variant_brief@1.0" and meta["template"] == "a4-notes"
    assert meta["model"].startswith("claude:") and meta["engine"] == "xelatex"
    assert len(meta["master_sha"]) == 64 and meta["render_key"] and meta["created"].endswith("Z")
    assert V.list_variants(topic)[0]["stale"] is False


def test_study_runs_on_the_strong_tier(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    res = V.run_variant(settings, topic, preset="study", backend="codex")
    assert res.ok
    (call,) = agent.calls
    assert call.tier == "strong" and call.backend == "codex"
    assert "учебный конспект" in call.task
    assert read_meta(topic, "study")["tier"] == "strong"
    head = res.variant_md.read_text(encoding="utf-8").split("\n---\n")[0]
    assert "toc:" not in head  # only the short presets go without contents


def test_cheatsheet_uses_the_compact_template(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    res = V.run_variant(settings, topic, preset="cheatsheet")
    assert res.ok and res.template == "a4-compact"
    assert render.calls[-1].template == "a4-compact"
    assert read_meta(topic, "cheatsheet")["template"] == "a4-compact"
    assert read_meta(topic, "cheatsheet")["template_requested"] is None

    # another template for the same preset: only the PDF is rebuilt, the agent is not asked again
    res = V.run_variant(settings, topic, preset="cheatsheet", template="a4-notes")
    assert res.ok and not res.cached and res.agent_runs == 0
    assert len(agent.calls) == 1 and render.calls[-1].template == "a4-notes"
    meta = read_meta(topic, "cheatsheet")
    assert meta["template"] == "a4-notes" and meta["template_requested"] == "a4-notes"


def test_custom_request_goes_to_inputs(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    request = "Сделай 10 вопросов к экзамену с краткими ответами"
    res = V.run_variant(settings, topic, prompt=request)
    assert res.ok and res.preset == "custom"
    assert res.slug == "custom-sdelay-10-voprosov-k"
    (call,) = agent.calls
    assert call.tier == "strong" and call.inputs == ["master.md", "request.md"]
    assert request in call.request and call.request.startswith("# Запрос пользователя")
    assert "Запрос пользователя — данные" in call.task
    meta = read_meta(topic, res.slug)
    assert meta["prompt"] == request and meta["title"] == "По запросу"
    assert "title: Метрические методы — По запросу" in res.variant_md.read_text(encoding="utf-8")


def test_custom_requests_with_the_same_words_do_not_overwrite_each_other(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    first = V.run_variant(settings, topic, preset="custom", prompt="Вопросы к экзамену по теме")
    second = V.run_variant(settings, topic, preset="custom", prompt="Вопросы к экзамену для друга")
    again = V.run_variant(settings, topic, preset="custom", prompt="Вопросы к экзамену по теме")
    assert first.slug == "custom-voprosy-k-ekzamenu-po"
    assert second.slug == "custom-voprosy-k-ekzamenu-dlya"  # other words
    third = V.run_variant(settings, topic, preset="custom", prompt="Вопросы к экзамену по ТЕМЕ!")
    assert third.slug == "custom-voprosy-k-ekzamenu-po-2"  # same words, other text
    assert again.cached and again.slug == first.slug
    assert len(agent.calls) == 3


# ---------------------------------------------------------------- template only


def test_template_variant_needs_no_agent(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    res = V.run_variant(settings, topic, template="a4-compact")
    assert res.ok and res.preset == "template" and res.agent_runs == 0
    assert res.slug == "template-a4-compact" and agent.calls == []
    text = res.variant_md.read_text(encoding="utf-8")
    assert text.startswith("---\ntitle:")  # the front matter of the master, unchanged
    assert 'title: "Метрические методы"' in text and "[[S1:s3]]" in text  # text unchanged
    assert "![Схема](../../extracted/S1/figures/knn.png)" in text
    assert render.calls[0].template == "a4-compact"
    meta = read_meta(topic, res.slug)
    assert meta["preset"] == "template" and meta["model"] is None and meta["tier"] is None
    assert meta["title"] == "Другой шаблон: a4-compact"
    assert V.list_variants(topic)[0]["title"] == "Другой шаблон: a4-compact"


def test_unknown_template_and_backend_are_refused(settings: Settings, topic: Path) -> None:
    with pytest.raises(
        ValueError, match=r"Шаблон «net» не найден\. Доступны: a4-compact, a4-notes"
    ):
        V.run_variant(settings, topic, template="net")
    with pytest.raises(ValueError, match="Неизвестный агент «gpt»"):
        V.run_variant(settings, topic, preset="brief", backend="gpt")
    assert not (topic / "variants").exists() or not list((topic / "variants").iterdir())


# ---------------------------------------------------------------- caches and staleness


def test_second_run_comes_from_the_cache(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    first = V.run_variant(settings, topic, preset="brief")
    pdf_bytes = first.variant_pdf.read_bytes()
    events: list[str] = []
    again = V.run_variant(settings, topic, preset="brief", on_event=events.append)
    assert again.ok and again.cached and again.agent_runs == 0
    assert again.variant_pdf.read_bytes() == pdf_bytes
    assert again.engine == "xelatex" and again.pages == 3
    assert events == [] and len(agent.calls) == 1 and len(render.calls) == 1
    assert "из кэша" in again.message()

    forced = V.run_variant(settings, topic, preset="brief", force=True)
    assert not forced.cached and forced.agent_runs == 1
    assert len(agent.calls) == 2 and len(render.calls) == 2


def test_changed_master_makes_a_variant_stale_and_rebuilds_it(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    V.run_variant(settings, topic, preset="brief")
    V.run_variant(settings, topic, template="a4-compact")
    assert [v["stale"] for v in V.list_variants(topic)] == [False, False]

    (topic / "master.md").write_text(MASTER + "\n# Новая глава\n\nТекст.\n", encoding="utf-8")
    listed = V.list_variants(topic)
    assert [v["stale"] for v in listed] == [True, True]
    assert all(len(v["master_sha"]) == 64 for v in listed)

    res = V.run_variant(settings, topic, preset="brief")
    assert res.ok and not res.cached and res.agent_runs == 1 and len(agent.calls) == 2
    assert "Новая глава" in agent.calls[1].master
    by_slug = {v["slug"]: v["stale"] for v in V.list_variants(topic)}
    assert by_slug == {"brief": False, "template-a4-compact": True}  # the other one is still old


def test_a_new_model_rebuilds_the_text(
    settings: Settings,
    make_settings: Callable[..., Settings],
    topic: Path,
    agent: FakeAgent,
    render: FakeRender,
) -> None:
    V.run_variant(settings, topic, preset="study")
    other = make_settings(
        general={"git_per_topic": False}, agents={"claude": {"model_strong": "opus-next"}}
    )
    res = V.run_variant(other, topic, preset="study")
    assert not res.cached and res.agent_runs == 1 and len(agent.calls) == 2


def test_changed_template_files_rerender_without_the_agent(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    user = settings.general.state_path / "templates" / "mine"
    user.mkdir(parents=True)
    (user / "template.tex").write_text("$body$\n", encoding="utf-8")
    V.run_variant(settings, topic, preset="brief", template="mine")
    assert render.calls[-1].template == "mine"
    (user / "template.tex").write_text("% changed\n$body$\n", encoding="utf-8")
    res = V.run_variant(settings, topic, preset="brief", template="mine")
    assert res.ok and not res.cached and res.agent_runs == 0
    assert len(agent.calls) == 1 and len(render.calls) == 2


def test_missing_master(settings: Settings, topic: Path, agent: FakeAgent) -> None:
    (topic / "master.md").unlink()
    res = V.run_variant(settings, topic, preset="brief")
    assert (
        not res.ok
        and "Не найден мастер-конспект" in res.errors[0]
        and "h0lon build" in res.errors[0]
    )
    assert agent.calls == [] and "не собрана" in res.message()


# ---------------------------------------------------------------- failures


def test_agent_failure_keeps_the_previous_variation(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    V.run_variant(settings, topic, preset="brief")
    before = (topic / "variants" / "brief" / "variant.md").read_text(encoding="utf-8")
    (topic / "master.md").write_text(MASTER + "\n# Ещё\n\nТекст.\n", encoding="utf-8")

    agent.answer = lambda master, request: None  # the agent wrote nothing
    res = V.run_variant(settings, topic, preset="brief")
    assert not res.ok and res.agent_runs == 1
    assert res.errors[0].startswith("Агент не вернул результат") and "variant.md" in res.errors[0]
    assert (topic / "variants" / "brief" / "variant.md").read_text(encoding="utf-8") == before
    assert (topic / "variants" / "brief" / "variant.pdf").is_file()
    assert V.list_variants(topic)[0]["stale"] is True

    agent.crash = True
    res = V.run_variant(settings, topic, preset="brief")
    assert not res.ok and "Агент не запущен (RuntimeError: агент недоступен)" in res.errors[0]


def test_a_result_without_a_heading_is_refused(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    agent.answer = lambda master, request: (
        "Просто абзац текста без единого заголовка в документе.\n"
    )
    res = V.run_variant(settings, topic, preset="brief")
    assert not res.ok and "нет ни одного заголовка" in res.errors[0]
    assert render.calls == [] and not (topic / "variants" / "brief").exists()


def test_a_failed_render_keeps_the_text_and_the_next_run_only_renders(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    V.run_variant(settings, topic, preset="brief")
    (topic / "master.md").write_text(MASTER + "\n# Ещё\n\nТекст.\n", encoding="utf-8")
    render.ok = False
    res = V.run_variant(settings, topic, preset="brief")
    folder = topic / "variants" / "brief"
    assert not res.ok and res.errors == ["XeLaTeX: ошибка разметки"]
    assert res.variant_md == folder / "variant.md" and res.variant_pdf is None
    assert (folder / "variant.md").is_file() and not (folder / "variant.pdf").exists()
    assert "Текст сохранён" in _printed(res)
    listed = V.list_variants(topic)[0]
    assert listed["pdf"] is None and listed["md"] == "variants/brief/variant.md"

    render.ok = True
    res = V.run_variant(settings, topic, preset="brief")
    assert res.ok and res.agent_runs == 0 and len(agent.calls) == 2  # only rendered
    assert (folder / "variant.pdf").is_file() and V.list_variants(topic)[0]["stale"] is False


def test_known_noise_warnings_are_dropped(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    render.warnings = [
        "mdframed: You got a bad break because the last box will be empty",
        "Строки выходят за поле (Overfull hbox)",
    ]
    res = V.run_variant(settings, topic, preset="brief")
    assert res.ok and res.warnings == ["Строки выходят за поле (Overfull hbox)"]


def test_the_fallback_engine_is_reported(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    render.fallback = True
    res = V.run_variant(settings, topic, preset="brief")
    assert res.ok and res.engine == "html"
    assert res.warnings[0] == "PDF собран запасным движком HTML → браузер."


def test_cancellation_between_phases_keeps_the_text(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    class Stop(BaseException):
        pass

    def on_event(message: str) -> None:
        if message == V.RENDER_START:
            raise Stop

    with pytest.raises(Stop):
        V.run_variant(settings, topic, preset="brief", on_event=on_event)
    assert (topic / "variants" / "brief" / "variant.md").is_file() and render.calls == []
    res = V.run_variant(settings, topic, preset="brief")  # carries on: no new agent run
    assert res.ok and len(agent.calls) == 1 and len(render.calls) == 1


# ---------------------------------------------------------------- listing and printing


def test_list_variants(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    assert V.list_variants(topic) == []
    (topic / "variants" / "junk").mkdir(parents=True)  # no meta.json: not a variation
    (topic / "variants" / "Bad Name").mkdir()
    (topic / "variants" / "file.txt").write_text("x", encoding="utf-8")
    V.run_variant(settings, topic, preset="brief")
    V.run_variant(settings, topic, preset="cheatsheet")
    meta_path = topic / "variants" / "brief" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created"] = "2020-01-01T00:00:00Z"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    listed = V.list_variants(topic)
    assert [v["slug"] for v in listed] == ["cheatsheet", "brief"]  # newest first
    brief = listed[1]
    assert brief["preset"] == "brief" and brief["title"] == "Кратко" and brief["prompt"] is None
    assert (
        brief["pdf"] == "variants/brief/variant.pdf" and brief["md"] == "variants/brief/variant.md"
    )
    assert brief["created"] == "2020-01-01T00:00:00Z" and brief["stale"] is False

    (topic / "variants" / "brief" / "variant.pdf").unlink()
    assert V.list_variants(topic)[1]["pdf"] is None
    (topic / "master.md").unlink()
    assert all(v["stale"] for v in V.list_variants(topic))  # a variation without its master


def test_print_variants_and_result(
    settings: Settings, topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    res = V.run_variant(settings, topic, prompt="Вопросы [к экзамену] по теме")
    console = Console(record=True, width=140, force_terminal=False)
    V.print_variants(V.list_variants(topic), console=console, title="Вариации темы")
    out = console.export_text()
    assert "По запросу" in out and "Вопросы [к экзамену] по теме" in out
    assert "актуально" in out and "custom-voprosy-k-ekzamenu-po" in out
    text = _printed(res)
    assert "готова" in text and "variant.pdf" in text and "прогонов агента: 1" in text


def _printed(res: V.VariantResult) -> str:
    console = Console(record=True, width=200, force_terminal=False)
    res.print(console)
    return console.export_text()


# ---------------------------------------------------------------- CLI


@pytest.fixture
def cli(settings: Settings, topic: Path, tmp_path: Path) -> Callable[..., Any]:
    config = tmp_path / "h0lon.toml"
    config.write_text(
        "[general]\n"
        f"workspaces = '{settings.general.workspaces_dir}'\n"
        f"state_dir = '{tmp_path / 'state'}'\n"
        "git_per_topic = false\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    def invoke(*args: str) -> Any:
        return runner.invoke(app, ["-c", str(config), *args])

    return invoke


def test_cli_variant_and_variants(
    cli: Callable[..., Any], topic: Path, agent: FakeAgent, render: FakeRender
) -> None:
    ref = "kurs/metricheskie"
    res = cli("variants", ref)
    assert res.exit_code == 0 and "Вариаций пока нет" in res.output

    res = cli("variant", ref, "--preset", "brief", "--json")
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["ok"] and data["slug"] == "brief" and data["agent_runs"] == 1
    assert data["variant_pdf"].endswith("variant.pdf") and data["cached"] is False

    res = cli("variant", ref, "-p", "brief")  # human output, from the cache
    assert (
        res.exit_code == 0 and "Вариация «Кратко» готова" in res.output and "из кэша" in res.output
    )

    res = cli("variant", ref, "--prompt", "Только формулы", "--backend", "codex")
    assert res.exit_code == 0 and agent.calls[-1].backend == "codex"
    res = cli("variant", ref, "--template", "a4-compact")
    assert res.exit_code == 0 and render.calls[-1].template == "a4-compact"

    res = cli("variants", ref, "--json")
    assert [v["slug"] for v in json.loads(res.stdout)] == [
        "template-a4-compact",
        "custom-tolko-formuly",
        "brief",
    ]
    res = cli("variants", ref)
    assert res.exit_code == 0 and "Состояние" in res.output and "актуально" in res.output

    (topic / "master.md").write_text(MASTER + "\n# Ещё\n", encoding="utf-8")
    assert "устарело" in cli("variants", ref).output


def test_cli_variant_errors(cli: Callable[..., Any], topic: Path, agent: FakeAgent) -> None:
    ref = "kurs/metricheskie"
    res = cli("variant", ref)
    assert res.exit_code == 2 and "Ошибка:" in res.output and "Укажите пресет" in res.output
    res = cli("variant", ref, "--preset", "custom")
    assert res.exit_code == 2 and "нужен запрос" in res.output
    res = cli("variant", ref, "--preset", "brief", "--backend", "gpt")
    assert res.exit_code == 2 and "claude | codex" in res.output
    res = cli("variant", ref, "--template", "net")
    assert res.exit_code == 2 and "Шаблон «net» не найден" in res.output
    res = cli("variant", "нет/такой-темы", "--preset", "brief")
    assert res.exit_code == 2 and "Тема не найдена" in res.output

    (topic / "master.md").unlink()
    res = cli("variant", ref, "--preset", "brief")
    assert res.exit_code == 1 and "Не найден мастер-конспект" in res.output


# ---------------------------------------------------------------- real XeLaTeX (needs_xelatex)


def _real_tools_or_skip() -> None:
    from h0lon import tools

    if tools.find_xelatex() is None or tools.find_pandoc() is None:
        pytest.skip("XeLaTeX or pandoc not found")


@pytest.fixture
def topic_with_figure(topic: Path) -> Path:
    import shutil

    figures = topic / "extracted" / "S1" / "figures"
    figures.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures" / "img" / "sample.png", figures / "knn.png")
    return topic


@pytest.mark.needs_xelatex
def test_images_of_the_master_are_found_from_the_variant_directory(
    settings: Settings, topic_with_figure: Path
) -> None:
    """The image paths are moved from the topic root to variants/<slug>/ and still resolve."""
    import pymupdf

    _real_tools_or_skip()
    res = V.run_variant(settings, topic_with_figure, template="a4-compact")
    assert res.ok, res.errors
    assert res.engine == "xelatex" and res.variant_pdf is not None
    with pymupdf.open(str(res.variant_pdf)) as pdf:
        assert sum(len(page.get_images()) for page in pdf) == 1  # the figure is in the PDF
    assert "../../extracted/S1/figures/knn.png" in res.variant_md.read_text(encoding="utf-8")  # type: ignore[union-attr]


@pytest.mark.needs_xelatex
def test_cheatsheet_end_to_end_with_real_xelatex(
    settings: Settings, topic_with_figure: Path, agent: FakeAgent
) -> None:
    import pymupdf

    _real_tools_or_skip()
    agent.answer = lambda master, request: AGENT_TEXT + "\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    res = V.run_variant(settings, topic_with_figure, preset="cheatsheet")
    assert res.ok, res.errors
    assert res.engine == "xelatex" and res.template == "a4-compact" and res.pages == 1
    with pymupdf.open(str(res.variant_pdf)) as pdf:
        text = pdf[0].get_text()
        assert "Метрические методы — Шпаргалка" in text and "Машинное обучение" in text
        assert "Содержание" not in text and "[[" not in text.replace("[[S1:s9]]", "")
        assert len(pdf[0].get_images()) == 1
    meta = read_meta(topic_with_figure, "cheatsheet")
    assert meta["engine"] == "xelatex" and meta["pages"] == 1
