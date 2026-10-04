"""Handwritten notes (h0lon/extract/handwritten.py): page preparation, plan, extraction.

Pictures are drawn with Pillow inside the tests; PDFs with PyMuPDF. The agent is a fake
`run_task` (writes the page files itself) or, in one test, the fake Claude CLI of tests/fakes.
No network, no real agents.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from PIL import Image, ImageChops, ImageDraw

from h0lon import tools
from h0lon.agents import Usage, reset_cooling
from h0lon.agents.runner import RunResult
from h0lon.extract import blocks as bl
from h0lon.extract import handwritten as hw
from h0lon.extract import pipeline, registry, vision
from h0lon.extract import summary as sm
from h0lon.extract.model import ExtractContext, PageImage
from h0lon.extract.registry import ExtractError
from h0lon.sources.ingest import add_sources
from h0lon.sources.models import SourceRecord
from h0lon.workspace import create_topic, load_topic
from tests.fakes.agentkit import fake_modes, make_settings, read_calls

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

BG = (246, 244, 238)
RED = (200, 0, 0)

# One page as the strong agent returns it: headings from ###, an abbreviation kept as
# written, signs as in the manuscript, marks of doubt, the author's question, admin, doubt.
PAGE_MD = """### Равномерная сходимость

::: {.definition title="Опр."}
Посл-ть $\\{f_n(x)\\}$ сх-ся равномерно на $E$ [восстановлено по контексту]
:::

$$\\forall \\varepsilon > 0\\ \\exists N: \\forall n \\ge N\\ |f_n(x) - f(x)| < \\varepsilon$$

Сл-но, предел[?] равен [неразборчиво] при $n \\to \\infty$.

::: author-question
Пометка «??» у обозначения $N(\\varepsilon, x)$: «Что за N?»
:::

::: admin
ДЗ: №3 до пятницы
:::

::: uncertain
Не видно, что стоит после знака равенства внизу страницы.
:::
"""


# ---------------------------------------------------------------- drawing helpers


def sheet(
    size: tuple[int, int] = (1000, 1400), margin: float = 0.18, bg: tuple[int, int, int] = BG
) -> Image.Image:
    """A page with writing inside uniform margins; a red block marks the top-left corner."""
    w, h = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    mx, my = int(w * margin), int(h * margin)
    for i in range(8):
        y = my + 90 + i * (h - 2 * my - 100) // 8
        d.line([(mx, y), (w - mx - i * 25, y + 8)], fill=(25, 25, 45), width=5)
    d.rectangle([mx, my, mx + 140, my + 70], fill=RED)
    return img


def full_bleed(size: tuple[int, int]) -> Image.Image:
    """A page whose writing reaches all four edges (nothing to crop)."""
    w, h = size
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    d.line([(0, 4), (w, 4)], fill=(20, 20, 20), width=6)
    d.line([(0, h - 5), (w, h - 5)], fill=(20, 20, 20), width=6)
    d.line([(4, 0), (4, h)], fill=(20, 20, 20), width=6)
    d.line([(w - 5, 0), (w - 5, h)], fill=(20, 20, 20), width=6)
    d.line([(0, 0), (w, h)], fill=(20, 20, 20), width=6)
    return img


def red_center(img: Image.Image) -> tuple[float, float]:
    """Centre of the red block, in shares of the picture's width and height."""
    r, g, _b = img.convert("RGB").split()
    mask = ImageChops.multiply(
        r.point(lambda v: 255 if v > 150 else 0), g.point(lambda v: 255 if v < 60 else 0)
    )
    box = mask.getbbox()
    assert box is not None, "no red block in the picture"
    return (box[0] + box[2]) / 2 / img.width, (box[1] + box[3]) / 2 / img.height


def save(img: Image.Image, path: Path, **kw: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make_pdf(path: Path, pages: list[tuple[Image.Image, str]]) -> Path:
    """A PDF scan: a full-page picture per page, plus an optional text layer."""
    doc = pymupdf.open()
    for img, text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png_bytes(img))
        if text:
            page.insert_text((72, 100), text, fontname="helv", fontsize=11, render_mode=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------- fakes and contexts


class FakeAgent:
    """Stands in for `run_task`: writes out/p<NNNN>.md (and summary.md for the summary stage)."""

    def __init__(
        self,
        page: Callable[[int], str | None] | None = None,
        *,
        ok: bool = True,
        problems: list[str] | None = None,
    ) -> None:
        self.page = page or (lambda n: PAGE_MD)
        self.ok = ok
        self.problems = problems or []
        self.calls: list[dict[str, Any]] = []

    @property
    def vision_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["stage"] != "summary"]

    def __call__(
        self, bundle, *, settings, tier="strong", backend=None, fallback=True, on_event=None
    ):
        self.calls.append(
            {
                "stage": bundle.stage,
                "tier": tier,
                "backend": backend,
                "files": [f.path for f in bundle.contract.files],
                "inputs": sorted(p.name for p in bundle.inputs_dir.iterdir()),
                "images": [p.name for p in bundle.images],
                "task": bundle.task_path.read_text(encoding="utf-8"),
                "bundle": bundle,
            }
        )
        if bundle.stage == "summary":
            (bundle.out_dir / sm.SUMMARY_FILE).write_text(
                "## Аннотация\n\nКонспект по функциональным последовательностям: поточечная и "
                "равномерная сходимость, определения и примеры, вопросы автора на полях.\n\n"
                "## Оглавление\n\n- Равномерная сходимость [[H1:p1]]\n\n"
                "## Термины и обозначения\n\n- **Равномерная сходимость** — с общим N.\n",
                encoding="utf-8",
            )
        else:
            for f in bundle.contract.files:
                text = self.page(int(f.path[1:5]))
                if text is not None:
                    (bundle.out_dir / f.path).write_text(text, encoding="utf-8")
        return RunResult(
            ok=self.ok,
            bundle=bundle,
            backend_used="claude",
            attempts=[],
            usage_total=Usage(input_tokens=50, output_tokens=5, cost_usd=0.02),
            final_text="готово",
            problems=list(self.problems),
        )


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> FakeAgent:
    """One fake for page transcription and for the summary run of the pipeline."""
    fake = FakeAgent()
    monkeypatch.setattr(vision, "run_task", fake)
    monkeypatch.setattr("h0lon.agents.run_task", fake)
    return fake


def make_ctx(tmp_path: Path, settings: Any, src: Path, **kw: Any) -> ExtractContext:
    """Context of source H1 whose file is a copy of `src` in the topic's sources/."""
    topic = tmp_path / "topic"
    (topic / "sources").mkdir(parents=True, exist_ok=True)
    (topic / "runs").mkdir(exist_ok=True)
    dest = topic / "sources" / f"H1_{src.name}"
    shutil.copy(src, dest)
    rec = SourceRecord(
        id="H1",
        kind="handwritten",
        title="Конспект",
        file=f"sources/{dest.name}",
        original_name=src.name,
        added=datetime.now(UTC).isoformat(),
    )
    return ExtractContext(
        topic_dir=topic, source=rec, settings=settings, out_dir=topic / "extracted" / "H1", **kw
    )


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    return save(sheet(), tmp_path / "in" / "page.jpg", quality=95)


@pytest.fixture
def needs_pandoc() -> None:
    if tools.find_pandoc() is None:
        pytest.skip("Pandoc не найден")


@pytest.fixture(autouse=True)
def _cooling() -> Any:
    reset_cooling()
    yield
    reset_cooling()


# ---------------------------------------------------------------- preparation of pictures


@pytest.mark.parametrize(
    ("orientation", "transpose"),
    [
        (3, Image.Transpose.ROTATE_180),
        (6, Image.Transpose.ROTATE_90),
        (8, Image.Transpose.ROTATE_270),
    ],
)
def test_exif_orientation_is_applied(tmp_path: Path, orientation: int, transpose: Any) -> None:
    upright = sheet((1000, 1400), margin=0.1)
    assert red_center(upright)[0] < 0.5 and red_center(upright)[1] < 0.5
    exif = Image.Exif()
    exif[0x0112] = orientation
    path = save(upright.transpose(transpose), tmp_path / "phone.jpg", quality=95, exif=exif)
    picture = hw.prepare_picture(hw.open_picture(path))
    assert picture.rotated
    assert picture.image.height > picture.image.width  # portrait again
    cx, cy = red_center(picture.image)
    assert cx < 0.5 and cy < 0.5  # the marked corner is at the top left again


def test_picture_without_exif_is_not_rotated(photo: Path) -> None:
    picture = hw.prepare_picture(hw.open_picture(photo))
    assert not picture.rotated and picture.image.height > picture.image.width


def test_uniform_margins_are_cropped() -> None:
    img = sheet((1000, 1400), margin=0.2)
    picture = hw.prepare_picture(img)
    assert picture.cropped and picture.source_size == (1000, 1400)
    w, h = picture.image.size
    assert 0.55 * 1000 < w < 0.75 * 1000 and 0.55 * 1400 < h < 0.75 * 1400
    # nothing of the writing is lost: the red block is still there and lies near the corner
    cx, cy = red_center(picture.image)
    assert cx < 0.25 and cy < 0.15


def test_writing_to_the_edge_is_not_cropped() -> None:
    picture = hw.prepare_picture(full_bleed((1200, 1600)))
    assert not picture.cropped and picture.image.size == (1200, 1600)


def test_blank_page_and_tiny_gain_are_left_whole() -> None:
    blank = Image.new("RGB", (900, 1200), BG)
    assert hw.find_content_box(blank) is None
    assert not hw.prepare_picture(blank).cropped
    # a thin white frame is not worth a crop (< 4 % of the area)
    framed = Image.new("RGB", (1000, 1400), BG)
    ImageDraw.Draw(framed).rectangle([8, 8, 991, 1391], outline=(10, 10, 10), width=4)
    assert not hw.prepare_picture(framed).cropped


def test_nearly_empty_page_is_not_cropped_to_a_speck() -> None:
    img = Image.new("RGB", (1000, 1400), BG)
    ImageDraw.Draw(img).rectangle([480, 700, 520, 720], fill=(20, 20, 20))  # a single word
    picture = hw.prepare_picture(img)
    assert not picture.cropped and picture.image.size == (1000, 1400)


def test_single_specks_do_not_hold_the_margins_back() -> None:
    img = sheet((1000, 1400), margin=0.2)
    for xy in ((2, 2), (997, 3), (3, 1396), (996, 1395)):
        img.putpixel(xy, (0, 0, 0))
    assert hw.prepare_picture(img).cropped


def test_dark_background_is_a_margin_too() -> None:
    img = Image.new("RGB", (1000, 1400), (20, 22, 24))  # a blackboard
    ImageDraw.Draw(img).line([(250, 400), (750, 420)], fill=(240, 240, 240), width=8)
    ImageDraw.Draw(img).line([(250, 900), (700, 1000)], fill=(240, 240, 240), width=8)
    assert hw.prepare_picture(img).cropped


def test_scaled_down_to_2000_never_up() -> None:
    big = hw.prepare_picture(full_bleed((3000, 4000)))
    assert max(big.image.size) == 2000 and big.image.size == (1500, 2000)
    small = hw.prepare_picture(full_bleed((800, 600)))
    assert small.image.size == (800, 600)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "L", "CMYK"])
def test_other_modes_become_rgb(mode: str) -> None:
    assert hw.prepare_picture(sheet((600, 800)).convert(mode)).image.mode == "RGB"


def test_transparency_goes_onto_white_paper() -> None:
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))  # fully transparent
    out = hw.to_rgb(img)
    assert out.mode == "RGB" and out.getpixel((10, 10)) == (255, 255, 255)


def test_enhanced_copy_is_gray_contrasty_and_a_new_picture() -> None:
    img = Image.new("RGB", (300, 300), (140, 140, 140))
    ImageDraw.Draw(img).rectangle([100, 100, 200, 200], fill=(100, 100, 100))
    before = img.tobytes()
    enhanced = hw.enhance(img)
    assert enhanced is not img and enhanced.mode == "L"
    assert img.tobytes() == before and img.mode == "RGB"
    low, high = enhanced.getextrema()
    assert low < 30 and high > 225  # the narrow range was stretched


# ---------------------------------------------------------------- pages of a source


def test_photo_gives_two_pictures_and_keeps_the_source(tmp_path: Path, photo: Path) -> None:
    before = sha(photo)
    pages = hw.prepare_pages(photo, tmp_path / "out" / "pages")
    assert len(pages) == 1 and sha(photo) == before
    page = pages[0]
    assert page.image.name == "p0001.png" and page.enhanced.name == "p0001.enh.png"
    with Image.open(page.image) as color, Image.open(page.enhanced) as gray:
        assert color.mode == "RGB" and gray.mode == "L" and color.size == gray.size
        assert (page.width, page.height) == color.size
    assert page.image.read_bytes() != page.enhanced.read_bytes()
    assert page.dpi == round(1400 / 11.69)  # the estimate as if A4 filled the frame
    assert page.cropped and not page.rotated and page.text_hint == ""


def test_targets_are_never_the_source(tmp_path: Path, photo: Path) -> None:
    picture = hw.prepare_picture(hw.open_picture(photo))
    with pytest.raises(ExtractError, match="не перезаписывается"):
        hw._save_pair(picture, photo, tmp_path / "other.png", photo)
    with pytest.raises(ExtractError, match="не перезаписывается"):
        hw._save_pair(picture, tmp_path / "other.png", photo, photo)
    assert photo.stat().st_size > 0 and Image.open(photo).format == "JPEG"


def test_heic_without_pillow_heif_gives_a_hint(
    tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pillow_heif", None)  # import raises ImportError
    heic = tmp_path / "in" / "page.heic"
    heic.parent.mkdir()
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    ctx = make_ctx(tmp_path, settings, heic)
    for call in (hw.HandwrittenExtractor().plan, hw.HandwrittenExtractor().extract):
        with pytest.raises(ExtractError) as err:
            call(ctx)
        assert "uv add pillow-heif" in str(err.value) and "JPG" in str(err.value)


def test_damaged_picture_is_a_readable_error(
    tmp_path: Path, settings: Any, agent: FakeAgent
) -> None:
    bad = tmp_path / "in" / "page.jpg"
    bad.parent.mkdir()
    bad.write_bytes(b"not a picture at all")
    ctx = make_ctx(tmp_path, settings, bad)
    with pytest.raises(ExtractError, match="Не удалось прочитать изображение"):
        hw.HandwrittenExtractor().extract(ctx)
    assert not agent.calls


def test_pdf_scan_pages_hints_and_dpi(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "in" / "scan.pdf",
        [(sheet((1200, 1700)), ""), (sheet((1200, 1700)), "Layer text of the page")],
    )
    pages = hw.prepare_pages(pdf, tmp_path / "pages")
    assert [p.number for p in pages] == [1, 2]
    assert pages[0].text_hint == "" and "Layer text of the page" in pages[1].text_hint
    for p in pages:
        assert max(p.width, p.height) <= 2000 and p.image.is_file() and p.enhanced.is_file()
        assert p.dpi == pytest.approx(1200 / (595 / 72), abs=2)  # the picture's own resolution


def test_vector_handwriting_pdf_has_no_resolution(tmp_path: Path) -> None:
    """Strokes drawn as paths over a paper picture (an iPad export) are not a low-res scan."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=png_bytes(Image.new("RGB", (300, 400), BG)))
    shape = page.new_shape()
    for i in range(400):
        shape.draw_line((60 + i, 100 + (i % 7) * 40), (80 + i, 120 + (i % 5) * 60))
    shape.finish(color=(0, 0, 0.5), width=1.2)
    shape.commit()
    pdf = tmp_path / "ipad.pdf"
    doc.save(str(pdf))
    pages = hw.prepare_pages(pdf, tmp_path / "pages")
    assert len(pages) == 1 and pages[0].dpi is None  # the 300 px picture alone says ≈35 dpi
    assert max(pages[0].width, pages[0].height) <= 2000
    plain = make_pdf(tmp_path / "plain.pdf", [(Image.new("RGB", (300, 400), BG), "")])
    plain_dpi = hw.prepare_pages(plain, tmp_path / "pages2")[0].dpi
    assert plain_dpi == pytest.approx(300 / (595 / 72), abs=2)


def test_prepared_pages_are_reused_until_the_file_changes(tmp_path: Path, photo: Path) -> None:
    pages_dir = tmp_path / "pages"
    pages = hw.prepare_pages(photo, pages_dir)
    hw._write_prep(pages_dir, hw.file_sha256(photo), pages)
    again = hw.load_prepared(pages_dir, hw.file_sha256(photo))
    assert again is not None and [p.image for p in again] == [pages[0].image]
    assert again[0].dpi == pages[0].dpi and again[0].cropped == pages[0].cropped
    assert hw.load_prepared(pages_dir, "0" * 64) is None  # another file
    pages[0].enhanced.unlink()
    assert hw.load_prepared(pages_dir, hw.file_sha256(photo)) is None  # a picture is missing


# ---------------------------------------------------------------- statistics and notes


def test_page_stats_count_marks_and_questions() -> None:
    text = (
        "слово[?] и [неразборчиво], ещё [?] [Неразборчиво] [восстановлено по контексту]\n\n"
        "::: author-question\nЧто за X?\n:::\n\n"
        '::: {.author-question title="к формуле"}\nПочему так?\n:::\n\n'
        "::: admin\nДЗ\n:::\n"
    )
    stats = hw.page_stats(text)
    assert (stats.unreadable, stats.unsure, stats.restored, stats.questions) == (2, 2, 1, 2)
    assert stats.uncertain == 4
    assert hw.page_stats(PAGE_MD).uncertain == 2 and hw.page_stats("текст").uncertain == 0


def test_notes_name_the_pages_to_check_first() -> None:
    stats = {
        1: hw.PageStats(unreadable=1),
        2: hw.PageStats(unsure=4, questions=1),
        3: hw.PageStats(),
        4: hw.PageStats(unreadable=2, unsure=1, restored=2),
    }
    notes = hw.build_notes(
        stats,
        use_vision=True,
        vision_done=4,
        failed=[5],
        empty=[3],
        dpis=[110.0],
        photo=True,
    )
    text = "\n".join(notes)
    assert "Не распознаны агентом: стр. 5" in text
    assert "Неуверенных мест: 8" in text and "стр. 2 (4), стр. 4 (3), стр. 1 (1)" in text
    assert "восстановленных по контексту, а не прочитанных целиком: 2" in text
    assert "Вопросов автора" in text and "author-question" in text
    assert "Пустые страницы: 3" in text
    assert "Фото низкого разрешения (≈110 dpi в пересчёте на лист A4)" in text
    off = hw.build_notes(
        {}, use_vision=False, vision_done=0, failed=[], empty=[], dpis=[], photo=False
    )
    assert len(off) == 1 and "Агент отключён" in off[0]


# ---------------------------------------------------------------- prompt and vision parameters


def test_handwritten_prompt_is_whole_and_versioned() -> None:
    text = vision.prompt_text("scan", "handwritten_pages")
    assert text.startswith("# Задача: точная оцифровка рукописного конспекта")
    assert "Главные правила достоверности" in text and "## Вариант" not in text
    assert "Распознавание рукописных страниц" not in text  # the header comment is dropped
    assert "`<!-- пустая страница -->`" in text
    assert vision.prompt_ref("handwritten_pages") == "handwritten_pages@1.1" == hw.PROMPT_REF
    assert vision.prompt_ref() == vision.PROMPT_REF == "vision_pages@1.0"
    with pytest.raises(FileNotFoundError):
        vision.prompt_ref("no_such_prompt")
    with pytest.raises(ValueError):
        vision.prompt_ref("../etc")
    with pytest.raises(ValueError):
        vision.prompt_text("video", "handwritten_pages")


def test_batch_task_lists_extra_pictures() -> None:
    task = vision.batch_task(
        "scan",
        [3, 4],
        prompt_name="handwritten_pages",
        extras={3: ["p0003.enh.png"], 4: ["x.png"]},
    )
    assert task.startswith("# Задача: точная оцифровка рукописного конспекта")
    assert (
        "- Страница 3: изображение `inputs/p0003.png`, усиленная копия `inputs/p0003.enh.png`,"
        in task
    )
    assert "`inputs/p0003.txt` → `out/p0003.md`" in task
    assert "дополнительное изображение `inputs/x.png`" in task
    assert vision.batch_task("document", [1]) == vision.batch_task("document", [1], extras={})


def test_page_keys_depend_on_prompt_and_extra_pictures(tmp_path: Path) -> None:
    png = save(sheet((200, 300)), tmp_path / "p0001.png")
    extra = save(sheet((200, 300), margin=0.1), tmp_path / "p0001.enh.png")
    plain = PageImage(number=1, image=png, text_hint="т")
    with_extra = PageImage(number=1, image=png, text_hint="т", extra_images=(extra,))
    # a page without extras keeps the key it always had
    legacy = hashlib.sha256(
        json.dumps(
            {
                "png": sha(png),
                "hint": hashlib.sha256("т".encode()).hexdigest(),
                "prompt": "vision_pages@1.0",
                "flavor": "document",
                "task": vision.TASK_FORMAT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert vision.page_key(plain, "document") == legacy
    keys = {
        vision.page_key(plain, "scan"),
        vision.page_key(with_extra, "scan"),
        vision.page_key(with_extra, "scan", "handwritten_pages"),
    }
    assert len(keys) == 3
    extra.write_bytes(extra.read_bytes() + b"\x00")
    assert vision.page_key(with_extra, "scan", "handwritten_pages") not in keys


def test_transcribe_pages_with_prompt_stage_and_extras(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent
) -> None:
    ctx = make_ctx(tmp_path, settings, photo)
    (page,) = hw.prepare_pages(ctx.topic_dir / str(ctx.source.file), ctx.out_dir / "pages")
    res = vision.transcribe_pages(
        ctx,
        [page.page_image()],
        flavor="scan",
        tier="strong",
        batch_size=4,
        prompt_name="handwritten_pages",
        stage="handwriting",
    )
    (call,) = agent.calls
    assert call["stage"] == "handwriting" and call["tier"] == "strong"
    assert call["images"] == ["p0001.png", "p0001.enh.png"]
    assert call["inputs"] == ["p0001.enh.png", "p0001.png", "p0001.txt"]
    assert "оцифровка рукописного конспекта" in call["task"] and "усиленная копия" in call["task"]
    assert 1 in res.pages and res.agent_runs == 1
    # cached with the same prompt and extras; another prompt or a changed extra is a miss
    again = vision.transcribe_pages(
        ctx, [page.page_image()], flavor="scan", prompt_name="handwritten_pages"
    )
    assert again.cached_pages == 1 and len(agent.calls) == 1
    vision.transcribe_pages(ctx, [page.page_image()], flavor="scan")  # vision_pages
    assert len(agent.calls) == 2


def test_agent_model_follows_stage_backend_and_tier(make_settings: Any) -> None:
    settings = make_settings(
        stages={"handwriting": "codex"}, agents={"codex": {"model_strong": "gpt-x"}}
    )
    assert vision.agent_model(settings, None, tier="strong", stage="handwriting") == (
        "codex",
        "gpt-x",
    )
    assert vision.agent_model(settings, "claude", tier="strong", stage="handwriting") == (
        "claude",
        settings.agents.claude.model_strong,
    )
    assert vision.agent_model(settings, None, tier="light") == ("claude", "sonnet")
    assert vision.agent_model(settings, "codex", tier="light") == ("codex", "default")


# ---------------------------------------------------------------- plan


def test_plan_counts_batches_of_four(tmp_path: Path, settings: Any) -> None:
    pdf = make_pdf(tmp_path / "in" / "scan.pdf", [(sheet((600, 800)), "")] * 9)
    ctx = make_ctx(tmp_path, settings, pdf)
    plan = hw.HandwrittenExtractor().plan(ctx)
    assert plan.pages_total == 9 and plan.pages_vision == 9
    assert plan.agent_runs == 3  # 4 + 4 + 1
    text = " ".join(plan.notes)
    assert "сильный уровень" in text and "батчи по 4" in text and "opus" in text
    assert not (ctx.out_dir / "pages").exists()  # a plan touches nothing

    off = hw.HandwrittenExtractor().plan(make_ctx(tmp_path, settings, pdf, use_vision=False))
    assert off.pages_vision == 0 and off.agent_runs == 0 and "Агент отключён" in off.notes[-1]


def test_plan_of_a_photo_and_after_extraction(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent
) -> None:
    ctx = make_ctx(tmp_path, settings, photo)
    extractor = hw.HandwrittenExtractor()
    first = extractor.plan(ctx)
    assert (first.pages_total, first.pages_vision, first.agent_runs) == (1, 1, 1)
    extractor.extract(ctx)
    again = extractor.plan(ctx)
    assert (again.pages_vision, again.agent_runs) == (0, 0)
    assert any("Из кэша распознавания: 1" in n for n in again.notes)
    ctx.force = True
    forced = extractor.plan(ctx)
    assert forced.pages_vision == 1 and forced.agent_runs == 1


# ---------------------------------------------------------------- extraction with a fake agent


def test_extract_photo_body_quality_and_bundle(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent
) -> None:
    events: list[str] = []
    ctx = make_ctx(tmp_path, settings, photo, on_event=events.append)
    out = hw.HandwrittenExtractor().extract(ctx)
    body = out.body_md.read_text(encoding="utf-8")
    assert body.startswith("## [[H1:p1]] Страница 1\n\n### Равномерная сходимость")
    assert "::: author-question" in body and "::: admin" in body and "::: uncertain" in body
    assert "сх-ся" in body and "Сл-но" in body  # the author's abbreviations stay
    assert "\\ge" in body and "n \\to \\infty" in body
    assert out.pages_total == 1 and out.pages_vision == 1 and out.agent_runs == 1
    q = out.quality
    assert (q["pages"], q["pages_vision"], q["pages_failed"]) == (1, 1, 0)
    assert q["uncertain_marks"] == 2 and q["author_questions"] == 1
    assert q["scan_dpi"] == round(1400 / 11.69)
    notes = "\n".join(q["notes"])
    assert "Неуверенных мест: 2" in notes and "Вопросов автора" in notes
    assert "формулы, знаки и сокращения" in notes
    assert not out.warnings

    (call,) = agent.calls
    assert call["stage"] == "handwriting" and call["tier"] == "strong"
    assert call["files"] == ["p0001.md"] and call["images"] == ["p0001.png", "p0001.enh.png"]
    assert (ctx.out_dir / "pages" / "p0001.enh.png").is_file()
    assert (ctx.out_dir / "pages" / "p0001.md").is_file()  # the transcription is cached
    assert any("распознавание агентом" in e for e in events)


def test_extract_pdf_batches_and_page_order(
    tmp_path: Path, settings: Any, agent: FakeAgent
) -> None:
    pdf = make_pdf(tmp_path / "in" / "scan.pdf", [(sheet((600, 800)), "")] * 9)
    ctx = make_ctx(tmp_path, settings, pdf)
    out = hw.HandwrittenExtractor().extract(ctx)
    assert sorted(c["files"][0] for c in agent.calls) == ["p0001.md", "p0005.md", "p0009.md"]
    assert sorted(len(c["files"]) for c in agent.calls) == [1, 4, 4]
    assert out.agent_runs == 3 and out.pages_total == 9 and out.pages_vision == 9
    heads = [ln for ln in out.body_md.read_text("utf-8").splitlines() if ln.startswith("## ")]
    assert heads == [f"## [[H1:p{n}]] Страница {n}" for n in range(1, 10)]


def test_empty_page_marker_and_counts(
    tmp_path: Path, settings: Any, photo: Path, monkeypatch
) -> None:
    fake = FakeAgent(page=lambda n: "<!-- пустая страница -->")
    monkeypatch.setattr(vision, "run_task", fake)
    out = hw.HandwrittenExtractor().extract(make_ctx(tmp_path, settings, photo))
    assert out.quality["pages_vision"] == 1 and out.quality["uncertain_marks"] == 0
    assert any("Пустые страницы: 1" in n for n in out.quality["notes"])
    assert "<!-- пустая страница -->" in out.body_md.read_text("utf-8")


def test_refusal_gives_failed_pages_with_text_layer_or_placeholder(
    tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAgent(
        page=lambda n: None if n == 2 else PAGE_MD,
        ok=False,
        problems=["out/p0002.md: агент отказался выполнять задание"],
    )
    monkeypatch.setattr(vision, "run_task", fake)
    pdf = make_pdf(
        tmp_path / "in" / "scan.pdf",
        [
            (sheet((600, 800)), ""),
            (sheet((600, 800)), "Layer text of page two"),
            (sheet((600, 800)), ""),
        ],
    )
    ctx = make_ctx(tmp_path, settings, pdf)
    out = hw.HandwrittenExtractor().extract(ctx)
    body = out.body_md.read_text("utf-8")
    page2 = body.split("## [[H1:p2]] Страница 2")[1].split("## [[H1:p3]]")[0]
    assert '::: {.uncertain title="Страница не распознана агентом"}' in page2
    assert "Layer text of page two" in page2  # the text layer stands in
    assert "### Равномерная сходимость" in body.split("## [[H1:p3]]")[1]  # the rest is intact
    assert out.quality["pages_failed"] == 1 and out.quality["pages_vision"] == 2
    assert any("Не распознаны агентом: стр. 2" in n for n in out.quality["notes"])
    assert any("не распознаны агентом стр. 2" in w and "отказался" in w for w in out.warnings)
    # no text layer: a placeholder that points at the picture
    one = make_pdf(tmp_path / "in" / "one.pdf", [(sheet((600, 800)), "")])
    ctx2 = make_ctx(tmp_path / "second", settings, one)
    fake.page = lambda n: None
    text = hw.HandwrittenExtractor().extract(ctx2).body_md.read_text("utf-8")
    assert "Рукописная страница не перенесена в текст" in text and "pages/p0001.png" in text


def test_agent_off_marks_every_page(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent
) -> None:
    ctx = make_ctx(tmp_path, settings, photo, use_vision=False)
    out = hw.HandwrittenExtractor().extract(ctx)
    assert not agent.calls and out.agent_runs == 0 and out.pages_vision == 0
    assert "не распознано: агент отключён" in out.body_md.read_text("utf-8")
    assert out.quality["pages_failed"] == 0 and out.quality["uncertain_marks"] == 0
    assert any("Агент отключён" in n for n in out.quality["notes"])
    assert (ctx.out_dir / "pages" / "p0001.png").is_file()  # pictures help the review anyway


def test_second_extract_sends_nothing_and_force_does(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent
) -> None:
    ctx = make_ctx(tmp_path, settings, photo)
    extractor = hw.HandwrittenExtractor()
    first = extractor.extract(ctx)
    mtimes = {p.name: p.stat().st_mtime_ns for p in (ctx.out_dir / "pages").glob("p0001*.png")}
    second = extractor.extract(ctx)
    assert len(agent.vision_calls) == 1 and second.agent_runs == 0 and second.pages_vision == 1
    assert second.body_md.read_text("utf-8") == first.body_md.read_text("utf-8")
    assert mtimes == {
        p.name: p.stat().st_mtime_ns for p in (ctx.out_dir / "pages").glob("p0001*.png")
    }
    ctx.force = True
    extractor.extract(ctx)
    assert len(agent.vision_calls) == 2


def test_replaced_source_drops_pictures_of_missing_pages(
    tmp_path: Path, settings: Any, agent: FakeAgent
) -> None:
    ctx = make_ctx(
        tmp_path, settings, make_pdf(tmp_path / "in" / "a.pdf", [(sheet((500, 700)), "")] * 3)
    )
    extractor = hw.HandwrittenExtractor()
    extractor.extract(ctx)
    pages = ctx.out_dir / "pages"
    assert (pages / "p0003.png").is_file()
    shorter = make_pdf(tmp_path / "in" / "b.pdf", [(sheet((500, 700)), "")] * 2)
    shutil.copy(shorter, ctx.topic_dir / ctx.source.file)
    extractor.extract(ctx)
    assert (pages / "p0002.png").is_file()
    assert not list(pages.glob("p0003*"))


def test_with_fake_claude_cli(tmp_path: Path, photo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundle, runner and CLI arguments for real; the CLI is the fake from tests/fakes."""
    state = fake_modes(monkeypatch, tmp_path, claude="ok")
    settings = make_settings(tmp_path)
    ctx = make_ctx(tmp_path, settings, photo)
    out = hw.HandwrittenExtractor().extract(ctx)
    assert out.agent_runs == 1 and not out.quality["pages_failed"]
    (call,) = read_calls(state, "claude")
    assert "--model" in call["argv"] and settings.agents.claude.model_strong in call["argv"]
    assert "оцифровка рукописного конспекта" in call["stdin"]
    assert "p0001.png" in call["stdin"] and "p0001.enh.png" in call["stdin"]
    (bundle_dir,) = list((ctx.topic_dir / "runs").iterdir())
    meta = json.loads((bundle_dir / "bundle.json").read_text("utf-8"))
    assert meta["stage"] == "handwriting"
    assert meta["images"] == ["inputs/p0001.png", "inputs/p0001.enh.png"]
    assert out.body_md.read_text("utf-8").count("### Ответ") == 1


# ---------------------------------------------------------------- registry and pipeline


def test_registry_supports_handwritten_and_skips_only_video_audio() -> None:
    res = registry.resolve_extractor("handwritten")
    assert not res.skipped and res.extractor is not None and res.reason is None
    assert res.extractor.kinds == ("handwritten",) and res.extractor.version == hw.VERSION
    assert pipeline.get_extractor("handwritten") is not None
    assert set(registry.LATER_STAGES) == {"video", "audio"}
    for kind in ("video", "audio"):
        later = registry.resolve_extractor(kind)
        assert later.skipped and "этап M5" in (later.reason or "")


def add_photo(settings: Any, tmp_path: Path, photo: Path) -> Path:
    topic = create_topic(settings, title="Матан", course="Анализ")
    report = add_sources(settings, topic, [str(photo)], kind="handwritten", title="Конспект")
    assert [r.id for r in report.added] == ["H1"]
    return topic


def test_pipeline_extracts_handwritten_into_blocks(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent, needs_pandoc: None
) -> None:
    topic = add_photo(settings, tmp_path, photo)
    (result,) = pipeline.extract_topic(settings, topic)
    assert result.ok and result.source_md is not None and not result.cached
    assert result.agent_runs == 1 and result.pages_vision == 1  # the fake reports no attempts
    assert [c["stage"] for c in agent.calls] == ["handwriting", "summary"]
    rec = next(s for s in load_topic(topic).sources if s["id"] == "H1")
    assert rec["status"] == "extracted" and rec["error"] is None
    q = rec["quality"]
    assert (q["pages"], q["pages_vision"], q["pages_failed"]) == (1, 1, 0)
    assert q["uncertain_marks"] == 2 and q["author_questions"] == 1
    assert q["cyrillic_ratio"] > 0.5 and any("Неуверенных мест" in n for n in q["notes"])

    out = topic / "extracted" / "H1"
    blocks = bl.read_blocks_jsonl(out / "blocks.jsonl")
    types = [b.type for b in blocks]
    for expected in ("heading", "definition", "formula", "author-question", "admin", "uncertain"):
        assert expected in types, (expected, types)
    by_type = {b.type: b for b in blocks}
    assert all(b.anchor == "H1:p1" for b in blocks)  # the place heading is the anchor
    assert by_type["definition"].title == "Опр."
    assert "Что за N?" in by_type["author-question"].text
    assert "ДЗ" in by_type["admin"].text
    assert "Страница 1" not in " ".join(b.text for b in blocks)  # the place heading is no block

    source = (out / "source.md").read_text("utf-8")
    front, _ = bl.split_front_matter(source)
    assert front["kind"] == "handwritten" and front["extracted_by"]["prompts"] == [hw.PROMPT_REF]
    assert front["extracted_by"]["model"] == settings.agents.claude.model_strong  # the strong tier
    assert front["quality"]["author_questions"] == 1
    assert "<!-- H1.b002 definition -->" in source
    assert (out / "summary.md").is_file()
    meta = json.loads((out / "meta.json").read_text("utf-8"))
    assert meta["summary"]["mode"] == "agent" and meta["extract_agent_runs"] == 1


def test_pipeline_no_longer_skips_handwritten_but_skips_video(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent, needs_pandoc: None
) -> None:
    topic = add_photo(settings, tmp_path, photo)
    plans = pipeline.extract_topic(settings, topic, dry_run=True)
    assert plans[0].pages_total == 1 and plans[0].pages_vision == 1
    assert plans[0].agent_runs == 2  # the page batch and the summary
    assert not any("M4" in n for n in plans[0].notes)
    add_sources(settings, topic, ["https://www.youtube.com/watch?v=abcdefghijk"])
    results = pipeline.extract_topic(settings, topic, use_vision=False)
    by_id = {r.source_id: r for r in results}
    assert by_id["H1"].ok and by_id["H1"].source_md is not None
    assert by_id["V1"].ok and by_id["V1"].source_md is None and "этап M5" in by_id["V1"].warnings[0]
    statuses = {s["id"]: s["status"] for s in load_topic(topic).sources}
    assert statuses == {"H1": "extracted", "V1": "skipped"}


def test_pipeline_failed_page_is_reported_not_fatal(
    tmp_path: Path, settings: Any, photo: Path, monkeypatch: pytest.MonkeyPatch, needs_pandoc: None
) -> None:
    fake = FakeAgent(page=lambda n: None, ok=False, problems=["агент недоступен"])
    monkeypatch.setattr(vision, "run_task", fake)
    monkeypatch.setattr("h0lon.agents.run_task", fake)
    topic = add_photo(settings, tmp_path, photo)
    (result,) = pipeline.extract_topic(settings, topic)
    assert result.ok and result.quality["pages_failed"] == 1
    assert any("не распознаны агентом" in w for w in result.warnings)
    blocks = bl.read_blocks_jsonl(topic / "extracted" / "H1" / "blocks.jsonl")
    assert [b.type for b in blocks] == ["uncertain"]


def test_pipeline_cache_key_holds_the_strong_model_and_the_prompt(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent, needs_pandoc: None
) -> None:
    topic = add_photo(settings, tmp_path, photo)
    rec = SourceRecord.model_validate(next(s for s in load_topic(topic).sources if s["id"] == "H1"))
    ext = registry.get_extractor("handwritten")
    assert ext is not None

    def key(cfg: Any, *, vision_on: bool = True) -> tuple[str, dict[str, Any]]:
        return pipeline.cache_key(
            cfg, rec, ext, file_sha256="abc", use_vision=vision_on, backend=None
        )

    key1, parts = key(settings)
    assert parts["prompts"] == [hw.PROMPT_REF, sm.prompt_id("summary")]
    assert parts["vision_model"] == settings.agents.claude.model_strong
    assert (
        parts["vision_backend"] == "claude" and parts["model"] == settings.agents.claude.model_light
    )
    strong = settings.model_copy(deep=True)
    strong.agents.claude.model_strong = "opus-next"
    other_stage = settings.model_copy(deep=True)
    other_stage.stages.handwriting = "codex"
    key2, key3 = key(strong)[0], key(other_stage)[0]
    key4, parts4 = key(settings, vision_on=False)
    assert len({key1, key2, key3, key4}) == 4 and "vision_model" not in parts4
    # light-tier extractors keep keys without the vision fields
    md_rec = rec.model_copy(update={"kind": "md"})
    md_ext = registry.get_extractor("md")
    assert md_ext is not None
    _, md_parts = pipeline.cache_key(
        settings, md_rec, md_ext, file_sha256="abc", use_vision=True, backend=None
    )
    assert "vision_model" not in md_parts

    pipeline.extract_topic(settings, topic)
    calls = len(agent.calls)
    again = pipeline.extract_topic(settings, topic)
    assert again[0].cached and len(agent.calls) == calls
    changed = settings.model_copy(deep=True)
    changed.agents.claude.model_strong = "opus-next"
    redo = pipeline.extract_topic(changed, topic)
    assert not redo[0].cached and len(agent.calls) > calls


def test_page_pictures_and_cache_files_live_in_the_pages_dir(
    tmp_path: Path, settings: Any, photo: Path, agent: FakeAgent, needs_pandoc: None
) -> None:
    """The page pictures stay where the review screen looks for them (extracted/<ID>/pages)."""
    topic = add_photo(settings, tmp_path, photo)
    pipeline.extract_topic(settings, topic)
    pages = sorted(p.name for p in (topic / "extracted" / "H1" / "pages").iterdir())
    assert {"p0001.png", "p0001.enh.png", "p0001.md", "p0001.key", "prep.json"} <= set(pages)
    prep = json.loads((topic / "extracted" / "H1" / "pages" / "prep.json").read_text("utf-8"))
    assert prep["version"] == hw.VERSION and len(prep["source_sha256"]) == 64
