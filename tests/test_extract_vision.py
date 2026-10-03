"""Page transcription (h0lon/extract/vision.py): prompt, batches, cache, failures, normalization.

The agent is replaced by a fake `run_task` (full control over the files it writes) or, in
one integration test, by the fake Claude CLI from tests/fakes.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fakes.agentkit import FAKE_CLAUDE, fake_modes

from h0lon.agents import Usage
from h0lon.agents.runner import RunResult
from h0lon.extract import vision
from h0lon.extract.model import ExtractContext, PageImage
from h0lon.sources.models import SourceRecord

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

BS = "\\"


def _png(path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.clear_with(200 + seed % 50)
    pix.save(str(path))
    return path


def _ctx(tmp_path: Path, settings: Any, **kw: Any) -> ExtractContext:
    topic = tmp_path / "topic"
    (topic / "runs").mkdir(parents=True, exist_ok=True)
    rec = SourceRecord(
        id="P1",
        kind="pdf-text",
        title="t",
        file="sources/P1.pdf",
        added=datetime.now(UTC).isoformat(),
    )
    return ExtractContext(
        topic_dir=topic, source=rec, settings=settings, out_dir=topic / "extracted" / "P1", **kw
    )


def _pages(ctx: ExtractContext, numbers: list[int], hint: str = "текст") -> list[PageImage]:
    return [
        PageImage(
            number=n,
            image=_png(ctx.out_dir / "pages" / f"p{n:04d}.png", n),
            text_hint=f"{hint} {n}",
            reason="math",
        )
        for n in numbers
    ]


class FakeAgent:
    """Stands in for `run_task`: writes out/p<NNNN>.md for the requested pages."""

    def __init__(self, write=None, ok: bool = True, problems: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.write = write or (lambda n: f"# Заголовок {n}\n\nТекст страницы {n}.\n")
        self.ok = ok
        self.problems = problems or []

    def __call__(
        self, bundle, *, settings, tier="strong", backend=None, fallback=True, on_event=None
    ):
        names = [f.path for f in bundle.contract.files]
        self.calls.append(
            {
                "bundle": bundle,
                "files": names,
                "tier": tier,
                "backend": backend,
                "inputs": sorted(p.name for p in bundle.inputs_dir.iterdir()),
                "hints": {
                    p.name: p.read_text(encoding="utf-8")
                    for p in bundle.inputs_dir.iterdir()
                    if p.suffix == ".txt"
                },
                "task": bundle.task_path.read_text(encoding="utf-8"),
            }
        )
        for name in names:
            n = int(name[1:5])
            text = self.write(n)
            if text is not None:
                (bundle.out_dir / name).write_text(text, encoding="utf-8")
        return RunResult(
            ok=self.ok,
            bundle=bundle,
            backend_used="claude",
            attempts=[],
            usage_total=Usage(input_tokens=100, output_tokens=10, cost_usd=0.01),
            final_text="готово",
            problems=list(self.problems),
        )


# ---------------------------------------------------------------- prompt


@pytest.mark.parametrize(
    ("flavor", "marker"),
    [
        ("document", "Страницы документа"),
        ("slides", "Страницы — слайды презентации"),
        ("scan", "Страницы — сканы или фотографии"),
    ],
)
def test_prompt_has_common_part_and_one_variant(flavor: str, marker: str) -> None:
    text = vision.prompt_text(flavor)
    assert text.startswith("# Задача: точная транскрипция страниц")
    assert marker in text
    others = {"Страницы документа", "Страницы — слайды презентации", "Страницы — сканы"} - {marker}
    assert not any(o in text for o in others if not marker.startswith(o))
    assert "## Вариант" not in text
    assert "Промпт распознавания страниц" not in text  # the header comment is dropped
    assert "`<!-- пустая страница -->`" in text  # the inline marker is part of the rules
    assert vision.PROMPT_REF == "vision_pages@1.0"


def test_unknown_flavor_is_rejected() -> None:
    with pytest.raises(ValueError):
        vision.prompt_text("video")


def test_batch_task_and_contract_list_every_page() -> None:
    task = vision.batch_task("slides", [3, 12])
    assert "Слайд 3: изображение `inputs/p0003.png`" in task
    assert "`inputs/p0012.txt` → `out/p0012.md`" in task
    contract = vision.batch_contract("document", [3, 12])
    assert [f.path for f in contract.files] == ["p0003.md", "p0012.md"]
    assert all(f.kind == "markdown" and f.min_chars == 1 for f in contract.files)


# ---------------------------------------------------------------- normalization


def test_normalize_unwraps_fence_and_drops_page_heading() -> None:
    raw = "```markdown\n## Страница 3\n\n# Введение\n\nТекст.\n```\n"
    assert vision.normalize_page_markdown(raw) == "### Введение\n\nТекст."


@pytest.mark.parametrize(
    "first",
    ["# Страница 3", "**Слайд 12. Итоги**", "## [[P1:p3]] Страница 3", "Страница 3", "### Page 3"],
)
def test_normalize_drops_page_titles(first: str) -> None:
    assert vision.normalize_page_markdown(f"{first}\n\nСодержание.") == "Содержание."


def test_normalize_keeps_content_that_mentions_a_page() -> None:
    text = "Страница 3 учебника содержит доказательство."
    assert vision.normalize_page_markdown(text) == text


def test_normalize_heading_levels_outside_code_and_math() -> None:
    raw = (
        "# Раздел\n\n## Подраздел\n\n### Уже третий\n\n#### Четвёртый\n\n"
        "```python\n# комментарий\n```\n\n"
        f"$$\n{BS}begin{{aligned}}\n# x\n{BS}end{{aligned}}$$\n\n## После формулы"
    )
    out = vision.normalize_page_markdown(raw)
    assert out.splitlines()[0] == "### Раздел"
    assert "### Подраздел" in out and "### Уже третий" in out and "#### Четвёртый" in out
    assert "# комментарий" in out and "\n# x\n" in out
    assert out.endswith("### После формулы")


def test_normalize_place_headings_inside_are_removed() -> None:
    raw = "Начало.\n\n## [[P1:p4]] Страница 4\n\nКонец."
    assert vision.normalize_page_markdown(raw) == "Начало.\n\nКонец."


def test_normalize_empty_and_brackets_and_raw_tex() -> None:
    assert vision.normalize_page_markdown("  \n") == vision.EMPTY_PAGE
    assert vision.normalize_page_markdown(vision.EMPTY_PAGE) == vision.EMPTY_PAGE
    out = vision.normalize_page_markdown(f"Пусть {BS}(x_1{BS}) и {BS}[a^2{BS}]")
    assert out == "Пусть $x_1$ и $$a^2$$"
    out = vision.normalize_page_markdown(f"сноска: {BS}mu A_1 и ${BS}mu$; `{BS}code`")
    assert out == f"сноска: {BS}{BS}mu A_1 и ${BS}mu$; `{BS}code`"
    # idempotent: cached pages are normalized again when read
    assert vision.normalize_page_markdown(out) == out


def test_unclosed_div_is_closed_within_the_page() -> None:
    """An open `::: {.theorem}` would swallow the following pages and their anchors."""
    text, issues = vision.normalize_page('::: {.theorem title="Т"}\nУтверждение $x$ без закрытия')
    assert text == '::: {.theorem title="Т"}\nУтверждение $x$ без закрытия\n:::'
    assert issues and "не закрыл" in issues[0]
    nested = "::: {.proof}\n::: {.remark}\nЗамечание\n:::\nДоказательство\n"
    assert vision.normalize_page_markdown(nested).endswith("Доказательство\n:::")
    again, issues = vision.normalize_page(text)
    assert again == text and issues == []  # idempotent (cached pages are normalized again)


def test_stray_div_fences_are_dropped_and_code_is_ignored() -> None:
    text, issues = vision.normalize_page("Текст\n:::\n\nЕщё текст\n::::\n")
    assert text == "Текст\n\nЕщё текст" and "лишние" in issues[0]
    code = "```\n::: {.x}\n```\n\n$$\n:::\n$$"
    assert vision.normalize_page(code) == (code, [])
    # an unclosed ``` is text for Pandoc: the div after it still gets closed, headings demoted
    text, issues = vision.normalize_page(
        "```python\nprint(1)\n\n# Раздел\n\n::: {.definition}\nМера"
    )
    assert text.endswith("### Раздел\n\n::: {.definition}\nМера\n:::") and issues


def test_bare_fence_around_code_is_kept() -> None:
    code = "```\nfor i in range(3):\n    print(i)\n# comment\n```"
    assert vision.normalize_page_markdown(code) == code
    wrapped = "```\n# Теорема\n\n::: {.theorem}\nТекст\n:::\n```"
    assert vision.normalize_page_markdown(wrapped) == "### Теорема\n\n::: {.theorem}\nТекст\n:::"
    table = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
    assert vision.normalize_page_markdown(table) == "| a | b |\n|---|---|\n| 1 | 2 |"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # inline math wrapped onto the next line stays math
        (
            f"Пусть $f(x) = {BS}alpha x +\n{BS}beta$ тогда",
            f"Пусть $f(x) = {BS}alpha x +\n{BS}beta$ тогда",
        ),
        # «5$ за штуку» is not an opening dollar (a space follows it)
        (
            f"Цена 5$ за штуку, формула $x{BS}alpha$ далее",
            f"Цена 5$ за штуку, формула $x{BS}alpha$ далее",
        ),
        # \( \) and \[ \] become math even when a $ is elsewhere on the page
        (
            f"Стоит $5. Тогда {BS}({BS}alpha + {BS}beta{BS}) верно.",
            f"Стоит $5. Тогда ${BS}alpha + {BS}beta$ верно.",
        ),
        (f"Цена $5.\n{BS}[\n{BS}alpha = 1\n{BS}]", f"Цена $5.\n$$\n{BS}alpha = 1\n$$"),
        # a blank line ends inline math: the raw command after it is literal text
        (f"$x +\n\n{BS}beta$", f"$x +\n\n{BS}{BS}beta$"),
        (f"Цена {BS}$5 и $x$ и {BS}alpha", f"Цена {BS}$5 и $x$ и {BS}{BS}alpha"),
        (
            f"Команда `{BS}begin{{document}}` и ${BS}alpha$",
            f"Команда `{BS}begin{{document}}` и ${BS}alpha$",
        ),
    ],
)
def test_math_detection_follows_pandoc(raw: str, expected: str) -> None:
    assert vision.normalize_page_markdown(raw) == expected
    assert vision.normalize_page_markdown(expected) == expected


# ---------------------------------------------------------------- transcription


def test_repairs_are_reported_as_warnings(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent(write=lambda n: "::: {.theorem}\nБез закрытия" if n == 2 else "Текст")
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings)
    res = vision.transcribe_pages(ctx, _pages(ctx, [1, 2, 3]), flavor="document")
    assert res.pages[2] == "::: {.theorem}\nБез закрытия\n:::"
    assert len(res.warnings) == 1 and res.warnings[0].startswith("P1: страница 2 — ")
    cached = vision.transcribe_pages(ctx, _pages(ctx, [1, 2, 3]), flavor="document")
    assert cached.cached_pages == 3 and cached.pages[2] == res.pages[2]


def test_batches_inputs_and_runs(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings, backend="codex")
    events: list[str] = []
    ctx.on_event = events.append
    res = vision.transcribe_pages(
        ctx, _pages(ctx, [1, 2, 3, 5, 6, 7, 9]), flavor="document", batch_size=3
    )
    agent.calls.sort(key=lambda c: c["files"][0])  # batches run two at a time
    assert [c["files"] for c in agent.calls] == [
        ["p0001.md", "p0002.md", "p0003.md"],
        ["p0005.md", "p0006.md", "p0007.md"],
        ["p0009.md"],
    ]
    first = agent.calls[0]
    assert first["tier"] == "light" and first["backend"] == "codex"
    assert first["inputs"] == [
        "p0001.png",
        "p0001.txt",
        "p0002.png",
        "p0002.txt",
        "p0003.png",
        "p0003.txt",
    ]
    assert first["hints"]["p0002.txt"] == "текст 2"
    bundle = first["bundle"]
    assert bundle.stage == "extract"
    assert bundle.root.parent == (ctx.topic_dir / "runs").resolve()
    assert [p.name for p in bundle.images] == ["p0001.png", "p0002.png", "p0003.png"]
    assert "Страницы документа" in first["task"]
    assert res.agent_runs == 3 and res.cached_pages == 0 and not res.failed
    assert res.pages[5] == "### Заголовок 5\n\nТекст страницы 5."
    assert res.usage.input_tokens == 300 and res.usage.cost_usd == pytest.approx(0.03)
    assert len(res.bundles) == 3
    assert any("страниц 5–7" in e for e in events)
    # no staging leftovers next to the cache
    assert not list(ctx.out_dir.glob(".vision-*"))


def test_cache_hits_and_invalidation(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings)
    pages = _pages(ctx, [1, 2, 3])
    vision.transcribe_pages(ctx, pages, flavor="document")
    assert len(agent.calls) == 1
    assert (ctx.out_dir / "pages" / "p0002.md").is_file()
    assert (ctx.out_dir / "pages" / "p0002.key").is_file()

    again = vision.transcribe_pages(ctx, pages, flavor="document")
    assert len(agent.calls) == 1 and again.cached_pages == 3 and again.agent_runs == 0
    assert again.pages[1] == "### Заголовок 1\n\nТекст страницы 1."

    pages[1].text_hint = "другая подсказка"
    third = vision.transcribe_pages(ctx, pages, flavor="document")
    assert agent.calls[-1]["files"] == ["p0002.md"] and third.cached_pages == 2

    other = vision.transcribe_pages(ctx, pages, flavor="scan")  # flavor is part of the key
    assert agent.calls[-1]["files"] == ["p0001.md", "p0002.md", "p0003.md"]
    assert other.cached_pages == 0

    ctx.force = True
    forced = vision.transcribe_pages(ctx, pages, flavor="scan")
    assert len(agent.calls) == 4 and forced.cached_pages == 0


def test_failed_pages_keep_the_rest(tmp_path: Path, settings: Any, monkeypatch) -> None:
    agent = FakeAgent(
        write=lambda n: None if n == 2 else f"Страница {n} текст",
        ok=False,
        problems=["out/p0002.md: обязательный файл не создан"],
    )
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings)
    res = vision.transcribe_pages(ctx, _pages(ctx, [1, 2, 3]), flavor="document")
    assert sorted(res.pages) == [1, 3]
    assert res.failed == {2: "out/p0002.md: обязательный файл не создан"}
    assert res.pages[1] == "Страница 1 текст"
    assert not (ctx.out_dir / "pages" / "p0002.md").exists()  # failures are not cached


def test_empty_answer_is_a_failure(tmp_path: Path, settings: Any, monkeypatch) -> None:
    monkeypatch.setattr(vision, "run_task", FakeAgent(write=lambda n: "  \n"))
    ctx = _ctx(tmp_path, settings)
    res = vision.transcribe_pages(ctx, _pages(ctx, [4]), flavor="scan")
    assert res.failed == {4: "агент не создал файл страницы"} and not res.pages


def test_missing_image_fails_the_batch_without_agent(
    tmp_path: Path, settings: Any, monkeypatch
) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings)
    page = PageImage(number=1, image=tmp_path / "nope.png", text_hint="")
    res = vision.transcribe_pages(ctx, [page], flavor="document")
    assert not agent.calls and res.agent_runs == 0
    assert "не удалось подготовить задание" in res.failed[1]


def test_duplicate_numbers_are_rejected(tmp_path: Path, settings: Any) -> None:
    ctx = _ctx(tmp_path, settings)
    pages = _pages(ctx, [1])
    with pytest.raises(ValueError):
        vision.transcribe_pages(ctx, pages + pages, flavor="document")


def test_parallel_batches_keep_all_pages(tmp_path: Path, make_settings, monkeypatch) -> None:
    settings = make_settings(agents={"parallel_runs": 2})
    agent = FakeAgent()
    monkeypatch.setattr(vision, "run_task", agent)
    ctx = _ctx(tmp_path, settings)
    res = vision.transcribe_pages(
        ctx, _pages(ctx, list(range(1, 12))), flavor="slides", batch_size=4
    )
    assert len(agent.calls) == 3 and sorted(res.pages) == list(range(1, 12))


def test_with_fake_claude_cli(tmp_path: Path, monkeypatch) -> None:
    """create_bundle + run_task for real, the CLI is the fake from tests/fakes."""
    from h0lon.config import Settings

    fake_modes(monkeypatch, tmp_path)
    settings = Settings(
        general={"state_dir": str(tmp_path / "state"), "workspaces": str(tmp_path / "ws")},
        agents={
            "fallback": "",
            "claude": {"bin": [sys.executable, str(FAKE_CLAUDE)], "timeout_s": 60},
        },
    )
    ctx = _ctx(tmp_path, settings)
    res = vision.transcribe_pages(ctx, _pages(ctx, [1, 2]), flavor="document")
    assert not res.failed and res.agent_runs == 1 and res.attempts == 1
    # the fake writes "# Ответ …": the level is pulled down to ###
    assert res.pages[2].startswith("### Ответ")
    assert res.usage.output_tokens > 0
    run_json = Path(res.bundles[0]) / "run.json"
    assert run_json.is_file()
