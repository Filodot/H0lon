"""Video and audio (h0lon/extract/{video,frames,asr}.py): key frames, recognition, correction.

The lecture is synthetic: slides, a board that fills up and is erased, a moving «lecturer»,
drawn with Pillow and encoded by ffmpeg (tests skip without ffmpeg or numpy). Recognition is a
fake `asr.transcribe` (or a fake `faster_whisper` module) and the agent is a fake `run_task`:
no network, no models, no real agents.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import sys
import time
import types
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image, ImageDraw

from h0lon import procutil, tools
from h0lon.agents import Usage, reset_cooling
from h0lon.agents.runner import RunResult
from h0lon.config import Settings
from h0lon.extract import asr, pipeline, registry, vision
from h0lon.extract import summary as sm
from h0lon.extract import video as vd
from h0lon.extract.registry import ExtractError
from h0lon.sources.ingest import add_sources, list_sources, update_source
from h0lon.sources.models import SourceRecord
from h0lon.workspace import create_topic, load_topic, save_topic

pytestmark = pytest.mark.usefixtures("clean_h0lon_env")

np = pytest.importorskip("numpy")
from h0lon.extract import frames as fr  # noqa: E402  (needs numpy)

FFMPEG = tools.find_simple("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не найден")

W, H, FPS = 640, 360, 5
BOARD = (26, 58, 26)
CHALK = (240, 240, 240)

# Timeline of the synthetic lecture, seconds.
SLIDE_A = (0, 12)
SLIDE_B = (12, 24)
BOARD_1 = (24, 54)
ERASED = (54, 60)
BOARD_2 = (60, 80)
SLIDE_A_AGAIN = (80, 92)
DURATION = 92
LINES_1 = (27, 33, 39, 45, 51)  # when a line appears on the first board
LINES_2 = (62, 68, 74)


# ---------------------------------------------------------------- the synthetic lecture


def slide_a() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 30, 420, 78], fill=(0, 0, 0))
    for i in range(4):
        d.rectangle([60, 120 + i * 50, 300 + i * 40, 140 + i * 50], fill=(20, 20, 20))
    return img


def slide_b() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([200, 20, 600, 60], fill=(0, 0, 160))
    for r in range(3):
        for c in range(5):
            d.rectangle(
                [60 + c * 100, 110 + r * 80, 120 + c * 100, 150 + r * 80], fill=(30, 30, 30)
            )
    return img


def chalk_line(d: ImageDraw.ImageDraw, y: int, count: int = 16) -> None:
    for k in range(count):
        x = 30 + k * 36
        d.rectangle([x, y, x + 20, y + 5], fill=CHALK)
        d.line([x, y, x + 20, y + 26], fill=CHALK, width=3)


def lecture_frame(t: float, *, lecturer: bool = True) -> Image.Image:
    if t < SLIDE_A[1] or t >= SLIDE_A_AGAIN[0]:
        img = slide_a()
    elif t < SLIDE_B[1]:
        img = slide_b()
    else:
        img = Image.new("RGB", (W, H), BOARD)
        d = ImageDraw.Draw(img)
        lines = (
            sum(1 for s in LINES_1 if t >= s)
            if BOARD_1[0] <= t < BOARD_1[1]
            else sum(1 for s in LINES_2 if t >= s)
            if BOARD_2[0] <= t < BOARD_2[1]
            else 0
        )
        for i in range(lines):
            chalk_line(d, 30 + i * 60)
    if lecturer:
        x = int(t * 40) % 520
        ImageDraw.Draw(img).rectangle([x, 120, x + 80, 340], fill=(0, 0, 0))
    return img


def build_video(path: Path, *, seconds: int = DURATION, audio: bool = True) -> Path:
    """The lecture as an mp4 (5 frames per second, a sine as the sound)."""
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(seconds * FPS):
        lecture_frame(i / FPS).save(frames_dir / f"{i:05d}.png")
    argv = [
        str(FFMPEG),
        "-y",
        "-v",
        "error",
        "-framerate",
        str(FPS),
        "-i",
        str(frames_dir / "%05d.png"),
    ]
    if audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}"]
    argv += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "18"]
    if audio:
        argv += ["-c:a", "aac", "-shortest"]
    argv.append(str(path))
    res = procutil.run(argv, timeout=300)
    assert res.ok, res.stderr
    shutil.rmtree(frames_dir, ignore_errors=True)
    return path


def build_slides_pdf(path: Path) -> Path:
    """The two slides of the lecture as a PDF (a picture per page, 16:9)."""
    import io

    import pymupdf

    doc = pymupdf.open()
    for make in (slide_a, slide_b):
        buf = io.BytesIO()
        make().save(buf, "PNG")
        page = doc.new_page(width=W, height=H)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="session")
def lecture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg не найден")
    root = tmp_path_factory.mktemp("lecture")
    return {
        "video": build_video(root / "lecture.mp4"),
        "slides": build_slides_pdf(root / "slides.pdf"),
    }


@pytest.fixture(scope="session")
def lecture_signals(lecture: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Any:
    th, _ = fr.Thresholds.from_mapping({})
    work = tmp_path_factory.mktemp("signals")
    return fr.compute_signals(
        FFMPEG,
        lecture["video"],
        video_size=(W, H),
        start=0.0,
        duration=DURATION,
        th=th,
        work_dir=work,
    )


# ---------------------------------------------------------------- thresholds


def test_thresholds_defaults_and_overrides() -> None:
    th, problems = fr.Thresholds.from_mapping(None)
    assert problems == [] and th.fps == 1.0 and th.hold == 3 and th.roi == "auto"
    th, problems = fr.Thresholds.from_mapping({"fps": 0.5, "hold": 4, "roi": [0.1, 0.1, 0.9, 0.9]})
    assert problems == [] and th.fps == 0.5 and th.hold == 4 and th.roi == [0.1, 0.1, 0.9, 0.9]
    for text in ("none", "off", "", None, False):
        assert fr.Thresholds.from_mapping({"roi": text})[0].roi is None
    assert fr.Thresholds.from_mapping({"roi": " Auto "})[0].roi == "auto"


def test_thresholds_report_wrong_keys_and_values() -> None:
    th, problems = fr.Thresholds.from_mapping(
        {"nope": 1, "hold": "x", "roi": [0.9, 0, 0.1, 1], "fps": 99, "jump_removed": "0.6"}
    )
    text = "\n".join(problems)
    assert "video.nope: неизвестный параметр" in text
    assert "video.hold: неверное значение" in text and "video.roi" in text
    assert "video.fps" in text and th.fps == 1.0
    assert th.jump_removed == 0.6 and th.hold == 3  # the good value is kept, the bad is not
    assert th.roi == "auto"  # the wrong area is ignored, the default stays
    _th, problems = fr.Thresholds.from_mapping({"roi": "left"})
    assert "допустимо auto, none или четыре числа" in problems[0]


# ---------------------------------------------------------------- frames of the synthetic lecture


@needs_ffmpeg
def test_signals_of_the_lecture(lecture_signals: Any) -> None:
    sig = lecture_signals
    assert abs(sig.n - DURATION) <= 1 and (sig.width, sig.height) == (640, 360)
    assert sig.small.shape[1:] == (fr.SMALL, fr.SMALL) and sig.fine.shape[1:] == (44, 79)
    assert float(sig.times[0]) == 0.0 and abs(float(sig.times[-1]) - (sig.n - 1)) < 1e-6
    assert float(sig.fine.min()) >= 0.0 and float(sig.fine.max()) <= 1.0


@needs_ffmpeg
def test_epochs_follow_the_changes_of_the_lecture(lecture_signals: Any) -> None:
    th, _ = fr.Thresholds.from_mapping({})
    sel = fr.select_frames(lecture_signals, th)
    assert sel.analysis.roi is None and sel.analysis.roi_source == "none"  # the whole frame
    bounds = [e.start for e in sel.epochs[1:]]
    expected = [SLIDE_B[0], BOARD_1[0], ERASED[0], SLIDE_A_AGAIN[0]]
    assert len(bounds) == len(expected), bounds
    for found, want in zip(bounds, expected, strict=True):
        assert abs(found - want) <= 2, (bounds, expected)
    assert [e.index for e in sel.epochs] == [1, 2, 3, 4, 5]
    assert sel.epochs[0].start == 0 and sel.epochs[-1].end == lecture_signals.n
    assert [e.reason for e in sel.epochs][:2] == ["start", "switch"]


@needs_ffmpeg
def test_key_frame_is_the_final_state_without_the_lecturer(lecture_signals: Any) -> None:
    th, _ = fr.Thresholds.from_mapping({})
    sel = fr.select_frames(lecture_signals, th)
    board1 = sel.epochs[2]
    (final,) = board1.frames
    # the board has all five lines near the end of the epoch, not the first one or two
    assert final.t >= LINES_1[-1] and final.t <= BOARD_1[1]
    static = sel.analysis.static
    assert final.ink >= 0.9 * static[board1.start : board1.end].max()
    assert final.occlusion < 0.15  # the lecturer is there, but he is a small part of it
    board2 = sel.epochs[3]
    assert board2.frames[-1].t >= LINES_2[-1]


@needs_ffmpeg
def test_slides_of_the_topic_replace_matching_frames(
    lecture: dict[str, Path], lecture_signals: Any
) -> None:
    th, _ = fr.Thresholds.from_mapping({})
    rec = SourceRecord(
        id="S1", kind="slides", title="Слайды", file="slides.pdf", added="2026-10-04T00:00:00Z"
    )
    images, notes = fr.load_slide_images(lecture["slides"].parent, [rec])
    assert [(r.source, r.number) for r in images] == [("S1", 1), ("S1", 2)] and notes == []
    assert images[0].gray.shape == (360, 640) and images[0].gray.dtype == np.uint8
    sel = fr.select_frames(lecture_signals, th, images)
    assert [e.kind for e in sel.epochs] == ["slide", "slide", "board", "board", "slide"]
    slides = [(f.slide["source"], f.slide["number"]) for f in sel.frames if f.kind == "slide"]
    assert slides == [("S1", 1), ("S1", 2), ("S1", 1)]
    assert [f.ordinal for f in sel.frames] == [1, 2, 3, 4, 5]
    assert all(f.duplicate_of is None for f in sel.frames)
    # without the presentation every epoch is a board; the last one repeats the first
    plain = fr.select_frames(lecture_signals, th).frames
    assert [f.kind for f in plain] == ["board"] * 5
    assert plain[4].duplicate_of == 1 and plain[0].duplicate_of is None


@needs_ffmpeg
def test_long_epoch_gets_intermediate_frames(lecture_signals: Any) -> None:
    th, _ = fr.Thresholds.from_mapping({"long_epoch": 20})
    epochs = fr.select_frames(lecture_signals, th).epochs
    board1 = epochs[2]
    levels = [f.level for f in board1.frames]
    assert levels[-1] == 100 and len(levels) >= 3 and levels == sorted(levels)
    times = [f.t for f in board1.frames]
    assert times == sorted(times) and times[0] < times[-1]
    # a short epoch stays with one frame
    assert [len(epochs[i].frames) for i in (0, 1, 4)] == [1, 1, 1]


@needs_ffmpeg
def test_signals_cache_roundtrip(lecture_signals: Any, tmp_path: Path) -> None:
    th, _ = fr.Thresholds.from_mapping({})
    key = fr.signals_key("abc", th, 0.0, 92.0)
    assert key != fr.signals_key("abc", th, 0.0, 60.0)
    other, _ = fr.Thresholds.from_mapping({"edge": 55})
    assert key != fr.signals_key("abc", other, 0.0, 92.0)
    same, _ = fr.Thresholds.from_mapping({"jump_removed": 0.6, "roi": "none"})
    assert key == fr.signals_key("abc", same, 0.0, 92.0)  # not properties of the signals
    path = tmp_path / "frames.cache.npz"
    fr.save_signals(path, lecture_signals, key)
    loaded = fr.load_signals(path, key)
    assert loaded is not None and loaded.n == lecture_signals.n
    assert np.array_equal(loaded.small, lecture_signals.small)
    assert np.allclose(loaded.fine, lecture_signals.fine, atol=1 / 255)
    assert (loaded.width, loaded.height) == (640, 360)
    assert fr.load_signals(path, "other") is None
    assert fr.load_signals(tmp_path / "missing.npz", key) is None


@needs_ffmpeg
def test_extract_png_and_names(lecture: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "frames" / fr.frame_name(3, 3725.4)
    assert target.name == "key_003_010205.png"
    fr.extract_png(FFMPEG, lecture["video"], 50.0, target, max_side=320)
    with Image.open(target) as im:
        assert im.size == (320, 180) and im.mode == "RGB"
    cropped = tmp_path / "frames" / "crop.png"
    fr.extract_png(FFMPEG, lecture["video"], 50.0, cropped, crop=[0.0, 0.0, 0.5, 1.0])
    with Image.open(cropped) as im:
        assert im.size == (320, 360)  # the left half, at its own size
    fr.clean_frames_dir(target.parent, ["other.png"])
    assert not target.exists() and cropped.exists()  # only key frames are cleaned
    with pytest.raises(ExtractError, match="не смог сохранить кадр"):
        fr.extract_png(FFMPEG, tmp_path / "no.mp4", 1.0, tmp_path / "x.png")


# ---------------------------------------------------------------- epochs from synthetic maps

ROWS, COLS = 44, 79  # blocks of the ink map of a 640×360 frame


def ink(*boxes: tuple[int, int, int, int], density: float = 0.4) -> Any:
    """An ink map with the given boxes (row from, row to, column from, column to) filled."""
    blocks = np.zeros((ROWS, COLS), dtype=np.float32)
    for r0, r1, c0, c1 in boxes:
        blocks[r0:r1, c0:c1] = density
    return blocks


def synthetic(states: list[tuple[Any, int]], noise: Any = None, template: Any = None) -> Any:
    """Signals out of (ink map, seconds) states; `noise(t)` adds the lecturer's map, `template`
    a map that is always there."""
    maps, smalls, t = [], [], 0
    for state, seconds in states:
        for _ in range(seconds):
            m = state.copy()
            if template is not None:
                m = np.maximum(m, template)
            if noise is not None:
                m = np.maximum(m, noise(t))
            maps.append(m)
            smalls.append(np.full((fr.SMALL, fr.SMALL), 120, dtype=np.uint8))
            t += 1
    return fr.Signals(
        small=np.stack(smalls),
        fine=np.stack(maps),
        times=np.arange(len(maps), dtype=np.float64),
        width=640,
        height=360,
    )


def lecturer_noise(t: int) -> Any:
    """A moving silhouette: ink in a changing place."""
    return ink((20, 34, (t * 3) % 70, (t * 3) % 70 + 6), density=0.5)


def select(sig: Any, **overrides: Any) -> fr.Selection:
    th, problems = fr.Thresholds.from_mapping(overrides)
    assert problems == []
    return fr.select_frames(sig, th)


def test_writing_is_not_a_boundary_but_a_switch_is() -> None:
    growing = [(ink((4, 8, 4, 50)), 8), (ink((4, 8, 4, 50), (12, 16, 4, 50)), 8)]
    other = (ink((24, 38, 55, 75)), 10)
    sig = synthetic([*growing, other], noise=lecturer_noise)
    sel = select(sig, roi="none")
    assert [e.start for e in sel.epochs] == [0, 16]  # the new content, not the added line
    assert sel.epochs[0].end == 16


def test_erased_board_and_partial_erase_are_boundaries() -> None:
    full = ink((4, 8, 4, 70), (14, 18, 4, 70), (24, 28, 4, 70))
    sig = synthetic([(full, 12), (ink(), 8)], noise=lecturer_noise)
    assert [e.start for e in select(sig, roi="none").epochs] == [0, 12]
    half = ink((4, 8, 4, 70))  # two thirds of the ink wiped away
    sel = select(synthetic([(full, 12), (half, 8)]), roi="none")
    assert [e.start for e in sel.epochs] == [0, 12] and sel.epochs[1].reason == "switch"


def test_covering_part_of_the_board_is_not_a_boundary() -> None:
    full = ink((4, 8, 4, 70), (14, 18, 4, 70), (24, 28, 4, 70), (34, 38, 4, 70))
    covered = full.copy()
    covered[:, 4:26] = 0  # the lecturer stands in front of a third of it
    sig = synthetic([(full, 8), (covered, 8), (full, 8)])
    assert [e.start for e in select(sig, roi="none").epochs] == [0]


def test_repeated_board_is_a_duplicate_and_blank_is_other() -> None:
    x = ink((4, 8, 4, 70), (14, 18, 4, 50))
    y = ink((20, 30, 20, 75))
    sig = synthetic([(x, 12), (y, 12), (x, 12), (ink(), 10)])
    sel = select(sig, roi="none")
    assert len(sel.epochs) == 4  # the board was erased at the end: a blank state
    assert [f.kind for f in sel.frames] == ["board", "board", "board", "other"]
    assert [f.duplicate_of for f in sel.frames] == [None, None, 1, None]
    quiet = synthetic([(ink(), 12)])
    (only,) = select(quiet).frames
    assert only.kind == "other"


def test_template_is_left_out_so_that_a_change_under_it_is_seen() -> None:
    # slides with the same header and footer, 60 % of the ink is the template: the new
    # content replaces only the rest, which is still a switch of the slide
    template = ink((0, 5, 0, 79), (40, 44, 0, 79), density=0.5)
    a = ink((10, 20, 5, 40))
    b = ink((10, 20, 40, 75))
    sig = synthetic([(a, 14), (b, 14), (a, 14), (b, 14)], template=template)
    sel = select(sig, roi="none")
    assert [e.start for e in sel.epochs] == [0, 14, 28, 42]
    analysis = sel.analysis
    assert analysis.template_blocks == 9 * 79 + 0  # header and footer rows, every column
    assert analysis.content_blocks == int(((a > 0) | (b > 0)).sum())
    # without enough frames to tell what is always there nothing is left out
    short = synthetic([(a, 6), (b, 6)], template=template)
    assert select(short, roi="none").analysis.template_blocks == 0


def test_one_state_for_the_whole_video_is_not_all_template() -> None:
    sig = synthetic([(ink((10, 20, 5, 40)), 40)])
    sel = select(sig)
    assert sel.analysis.template_blocks == 0 and sel.analysis.content_blocks > 0
    (only,) = sel.frames
    assert only.kind == "board" and only.ink > 0


def test_content_area_is_found_next_to_the_camera() -> None:
    # slides on the left 45 % of the frame, the lecturer walks on the right
    slides = [
        ink((6, 14, 2, 36), (30, 38, 2, 20)),
        ink((16, 28, 4, 38)),
        ink((6, 14, 20, 40), (30, 38, 22, 40)),
    ]
    states = [(slides[i % 3], 12) for i in range(9)]

    def walker(t: int) -> Any:
        return ink((10, 40, 48 + (t * 7) % 24, 54 + (t * 7) % 24), density=0.5)

    sig = synthetic(states, noise=walker)
    sel = select(sig)
    an = sel.analysis
    assert an.roi_source == "auto" and an.roi is not None
    x0, y0, x1, y1 = an.roi
    assert x0 == 0.0 and 0.4 < x1 < 0.6 and y0 == 0.0 and y1 == 1.0  # the full height, left part
    assert [e.start for e in sel.epochs] == list(range(0, 108, 12))
    # by hand, or off
    hand = select(sig, roi=[0.0, 0.0, 0.5, 1.0]).analysis
    assert hand.roi == [0.0, 0.0, 0.5, 1.0] and hand.roi_source == "topic"
    assert select(sig, roi="none").analysis.roi is None
    # the whole frame is the content: no area
    full = synthetic([(ink((4, 40, 4, 75)), 20), (ink((8, 36, 10, 70)), 20)])
    assert select(full).analysis.roi is None


def test_roi_limits_the_ink_to_the_content_area() -> None:
    left = ink((4, 20, 4, 30))
    both = (ink((4, 20, 4, 30), (4, 20, 45, 70)), 10)
    sig = synthetic([(left, 10), both])
    assert [e.start for e in select(sig, roi=[0.0, 0.0, 0.5, 1.0]).epochs] == [0]
    # the right half is looked at when it is the content area: ink appeared, nothing vanished
    assert fr.roi_mask((4, 10), None).all()
    assert fr.roi_mask((4, 10), [0.0, 0.0, 0.5, 1.0]).sum() == 4 * 5
    assert fr.crop_blocks(ink(), [0.0, 0.0, 0.5, 0.5]).shape == (22, 40)
    assert fr.crop_blocks(ink(), None).shape == (ROWS, COLS)


def test_hash_and_distances() -> None:
    a, b = fr.to_small(slide_a().convert("L")), fr.to_small(slide_b().convert("L"))
    ha, hb = fr.perceptual_hash(a), fr.perceptual_hash(b)
    assert int(fr.popcount(np.array([ha ^ ha]))[0]) == 0
    assert int(fr.popcount(np.array([ha ^ hb]))[0]) > 12  # other slides
    noisy = np.clip(a.astype(int) + np.random.default_rng(1).integers(-8, 9, a.shape), 0, 255)
    near = int(fr.popcount(np.array([ha ^ fr.perceptual_hash(noisy.astype(np.uint8))]))[0])
    assert near <= 6  # the same slide through a video codec
    many = fr.perceptual_hash(np.stack([a, b]))
    assert many.shape == (2,) and int(many[0]) == int(ha)
    x, y = ink((2, 4, 2, 10)), ink((10, 12, 20, 28))
    assert float(fr.ink_distance(x, x, axes=(-2, -1))) == 0.0
    assert float(fr.ink_distance(x, y, axes=(-2, -1))) == 1.0
    assert float(fr.ink_distance(x[x > 0], y[y > 0])) == 0.0  # vectors of the same ink
    coarse = fr.coarse_of(x)
    assert coarse.shape == (18, 32) and abs(float(coarse.mean()) - float(x.mean())) < 1e-3


# ---------------------------------------------------------------- slides next to a camera


@needs_ffmpeg
def test_slides_next_to_a_camera_are_matched_and_cut(tmp_path: Path) -> None:
    """Slides on the left 45 % of a 16:9 frame, a moving lecturer and a board on the right:
    the area is found, the slide change is seen, the slide is matched, the picture is cut."""

    def frame(t: float) -> Image.Image:
        img = Image.new("RGB", (W, H), (40, 60, 50))
        slide = slide_a() if int(t // 12) % 2 == 0 else slide_b()
        img.paste(slide.resize((int(W * 0.45), H)), (0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([int(W * 0.5), 20, W - 20, 120], fill=(220, 220, 220))  # a blank board
        x = int(W * 0.5) + int(t * 70) % 150
        d.rectangle([x, 150, x + 60, H - 10], fill=(0, 0, 0))  # the lecturer
        return img

    frames_dir = tmp_path / "f"
    frames_dir.mkdir()
    seconds = 72
    for i in range(seconds * FPS):
        frame(i / FPS).save(frames_dir / f"{i:05d}.png")
    video = tmp_path / "composite.mp4"
    res = procutil.run(
        [
            *(str(FFMPEG), "-y", "-v", "error", "-framerate", str(FPS)),
            *("-i", str(frames_dir / "%05d.png")),
            *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "18"),
            str(video),
        ],
        timeout=300,
    )
    assert res.ok, res.stderr
    th, _ = fr.Thresholds.from_mapping({})
    sig = fr.compute_signals(
        FFMPEG,
        video,
        video_size=(W, H),
        start=0.0,
        duration=seconds,
        th=th,
        work_dir=tmp_path / "w",
    )
    pdf = build_slides_pdf(tmp_path / "slides.pdf")
    rec = SourceRecord(
        id="S1", kind="slides", title="С", file="slides.pdf", added="2026-10-04T00:00:00Z"
    )
    images, _notes = fr.load_slide_images(tmp_path, [rec])
    sel = fr.select_frames(sig, th, images)
    assert sel.analysis.roi is not None and sel.analysis.roi[2] < 0.6 and sel.analysis.roi[0] == 0.0
    assert [e.start for e in sel.epochs][1:] == pytest.approx([12, 24, 36, 48, 60], abs=2)
    kinds = [(f.kind, f.slide["number"] if f.slide else None) for f in sel.frames]
    assert kinds == [("slide", 1), ("slide", 2)] * 3
    assert pdf.is_file()
    cropped = tmp_path / "cut.png"
    fr.extract_png(FFMPEG, video, 5.0, cropped, crop=sel.analysis.roi)
    with Image.open(cropped) as im:
        assert im.width < 0.6 * W and im.height == H


# ---------------------------------------------------------------- recognition (asr.py)


def seg(start: float, end: float, text: str, words: bool = True) -> asr.Segment:
    parts = text.split()
    step = (end - start) / max(len(parts), 1)
    ws = (
        [asr.Word(start + i * step, start + (i + 1) * step, w, 0.9) for i, w in enumerate(parts)]
        if words
        else []
    )
    return asr.Segment(start, end, text, ws)


def test_srt_and_json_roundtrip(tmp_path: Path) -> None:
    tr = asr.Transcript(
        segments=[seg(0.0, 2.5, "Привет, мир."), seg(3725.5, 3730.0, "Конец")],
        language="ru",
        duration=3731.0,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        seconds=42.0,
        key="k1",
        prompt="Лекция.",
        dropped=2,
    )
    asr.write_transcript(tmp_path, tr)
    srt = (tmp_path / "transcript.srt").read_text("utf-8")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nПривет, мир.\n")
    assert "2\n01:02:05,500 --> 01:02:10,000\nКонец" in srt
    back = asr.read_transcript(tmp_path)
    assert back is not None and back.key == "k1" and back.dropped == 2 and back.device == "cuda"
    assert [s.text for s in back.segments] == ["Привет, мир.", "Конец"]
    assert back.segments[0].words[1].text == "мир." and back.segments[0].words[0].prob == 0.9
    assert asr.read_transcript(tmp_path / "nowhere") is None
    (tmp_path / "transcript.json").write_text("{broken", encoding="utf-8")
    assert asr.read_transcript(tmp_path) is None


def test_clean_segments_drops_junk_and_loops() -> None:
    segs = [
        seg(0, 3, "Продолжаем лекцию."),
        seg(3, 4, "Субтитры сделал DimaTorzok"),
        seg(4, 5, "   "),
        seg(5, 6, "Да да да"),
        seg(6, 7, "Да да да"),
        seg(7, 8, "Да да да"),
        seg(8, 9, "Да да да!"),
        seg(9, 12, "Редактор субтитров А.Семкин Корректор А.Егорова"),
        seg(12, 15, "Метод ближайших соседей."),
    ]
    clean, dropped = asr.clean_segments(segs)
    assert [s.text for s in clean] == [
        "Продолжаем лекцию.",
        "Да да да",
        "Да да да",
        "Метод ближайших соседей.",
    ]
    assert dropped == 5


class FakeTokenizer:
    """One token per three characters."""

    def encode(self, text: str) -> Any:
        return types.SimpleNamespace(ids=list(range(len(text) // 3)))


def test_prompt_is_cut_by_whole_terms_from_the_end() -> None:
    terms = [f"термин номер {n}" for n in range(60)] + ["термин номер 3", "  "]
    text = asr.build_prompt("Метрические методы", terms)
    assert text.startswith("Лекция: Метрические методы. Термины: термин номер 0, ")
    assert len(text) <= asr.PROMPT_MAX_CHARS and text.endswith(".")
    assert text.count("термин номер 3,") == 1  # a repeated term only once
    assert "термин номер 59" not in text  # the tail was cut, the beginning is kept
    assert asr.build_prompt("", ["a"]) == "Лекция. Термины: a."
    tight = asr.build_prompt("Тема", [f"слово{n}" for n in range(200)], FakeTokenizer())
    assert len(FakeTokenizer().encode(" " + tight).ids) <= asr.PROMPT_MAX_TOKENS
    assert asr.build_prompt("Тема", []) == "Лекция: Тема."


def write_wav(path: Path, seconds: float, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(seconds * rate) * channels)
    return path


def test_read_wav_gives_float_samples_and_rejects_other_formats(tmp_path: Path) -> None:
    samples = asr.read_wav(write_wav(tmp_path / "a.wav", 0.5))
    assert samples.dtype == np.float32 and len(samples) == 8000
    assert abs(float(samples[0]) - 0.125) < 1e-3
    with pytest.raises(ExtractError, match="ожидается WAV 16 кГц"):
        asr.read_wav(write_wav(tmp_path / "b.wav", 0.1, rate=44100))
    (tmp_path / "c.wav").write_bytes(b"not a wav")
    with pytest.raises(ExtractError, match="Не удалось прочитать аудио"):
        asr.read_wav(tmp_path / "c.wav")


def make_probe(**kw: Any) -> asr.AsrProbe:
    base: dict[str, Any] = {"installed": True, "version": "1.2.1", "cuda_devices": 1}
    return asr.AsrProbe(**{**base, **kw})


def test_device_choice(make_settings: Callable[..., Settings]) -> None:
    gpu = asr.choose_device(make_settings(), make_probe())
    assert (gpu.device, gpu.compute_type, gpu.model, gpu.warnings) == (
        "cuda",
        "float16",
        "large-v3",
        [],
    )
    assert "CUDA" in gpu.label
    # no GPU: the CPU and the small model, with a warning about the quality
    cpu = asr.choose_device(make_settings(), make_probe(cuda_devices=0))
    assert (cpu.device, cpu.compute_type, cpu.model) == ("cpu", "int8", "small")
    assert "искажаться" in cpu.warnings[0]
    # a model named by the user stays, with a note about the time
    named = asr.choose_device(
        make_settings(compute={"asr_model": "large-v3"}), make_probe(cuda_devices=0)
    )
    assert named.model == "large-v3" and "дольше" in named.warnings[0]
    other = asr.choose_device(
        make_settings(compute={"asr_model": "medium"}), make_probe(cuda_devices=0)
    )
    assert other.model == "medium" and other.warnings == []
    # the GPU is there, the libraries are not: the CPU, and the reason is said
    nolib = asr.choose_device(
        make_settings(), make_probe(cuda_missing=["cublas64_12.dll", "cudnn_ops64_9.dll"])
    )
    assert nolib.device == "cpu" and "cublas64_12.dll" in nolib.warnings[0]
    assert "video-gpu" in nolib.warnings[0]
    # local-cpu never takes the GPU; local-gpu without CUDA is an error
    forced = asr.choose_device(make_settings(compute={"asr": "local-cpu"}), make_probe())
    assert forced.device == "cpu"
    with pytest.raises(ExtractError, match=r"local-gpu.*не найдена"):
        asr.choose_device(make_settings(compute={"asr": "local-gpu"}), make_probe(cuda_devices=0))
    for mode in ("colab", "api"):
        with pytest.raises(ExtractError, match="пока не поддерживается"):
            asr.choose_device(make_settings(compute={"asr": mode}), make_probe())


def test_cuda_libraries_go_to_the_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cublas64_12.dll").write_bytes(b"")
    monkeypatch.setattr(asr, "nvidia_lib_dirs", lambda: [bin_dir])
    monkeypatch.setenv("PATH", "C:\\somewhere")
    assert asr.prepare_cuda_path() == [bin_dir]
    assert str(bin_dir) in os.environ["PATH"].split(os.pathsep)
    asr.prepare_cuda_path()  # twice: not added twice
    assert os.environ["PATH"].count(str(bin_dir)) == 1
    monkeypatch.setattr(asr, "IS_WINDOWS", True)
    assert asr.missing_cuda_libs() == ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"]
    monkeypatch.setattr(asr, "IS_WINDOWS", False)
    assert asr.missing_cuda_libs() == []


def test_calibration_is_remembered_and_averaged(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()
    spm, measured = asr.estimate_seconds_per_minute(settings, "cuda", "large-v3")
    assert spm == 6.0 and not measured  # the PRD estimate
    first = asr.record_calibration(
        settings, device="cuda", model="large-v3", audio_seconds=600, wall_seconds=50
    )
    assert first is not None and first["seconds_per_minute"] == 5.0
    spm, measured = asr.estimate_seconds_per_minute(settings, "cuda", "large-v3")
    assert spm == 5.0 and measured
    entry = asr.record_calibration(
        settings, device="cuda", model="large-v3", audio_seconds=600, wall_seconds=70
    )
    assert entry is not None and entry["seconds_per_minute"] == 6.0
    assert entry["runs"] == 2 and entry["audio_minutes"] == 20.0
    tiny = asr.record_calibration(
        settings, device="cpu", model="small", audio_seconds=5, wall_seconds=1
    )
    assert tiny is None  # too short to say anything
    data = json.loads(asr.calibration_path(settings).read_text("utf-8"))
    assert set(data["asr"]) == {"cuda|large-v3"} and data["format"] == 1
    assert asr.estimate_seconds_per_minute(settings, "cpu", "unknown-model")[0] == 30.0


class FakeWhisperModel:
    """Stands in for `faster_whisper.WhisperModel`."""

    instances: ClassVar[list[FakeWhisperModel]] = []
    fail_cuda = False
    segments: ClassVar[list[tuple[float, float, str]]] = [
        (0.0, 2.0, " Привет"),
        (2.0, 4.0, " мир "),
    ]

    def __init__(self, model: str, device: str = "cpu", compute_type: str = "default") -> None:
        self.args = (model, device, compute_type)
        self.calls: list[dict[str, Any]] = []
        self.hf_tokenizer = FakeTokenizer()
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        self.calls.append({"samples": len(audio), **kwargs})
        if self.args[1] == "cuda" and FakeWhisperModel.fail_cuda:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

        def gen() -> Any:
            for a, b, text in FakeWhisperModel.segments:
                word = types.SimpleNamespace(start=a, end=b, word=text, probability=0.8)
                yield types.SimpleNamespace(start=a, end=b, text=text, words=[word])

        return gen(), types.SimpleNamespace(language="ru")


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    FakeWhisperModel.instances = []
    FakeWhisperModel.fail_cuda = False
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(asr, "require_faster_whisper", lambda: None)
    monkeypatch.setattr(asr, "model_cached", lambda model: True)
    return FakeWhisperModel


def test_transcribe_passes_the_glossary_and_cleans(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
    fake_whisper: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: make_probe())
    events: list[str] = []
    tr = asr.transcribe(
        write_wav(tmp_path / "a.wav", 4.0),
        settings=make_settings(),
        language="ru",
        title="Метрические методы",
        terms=["метод Парзена", "ядро"],
        on_event=events.append,
        label="V1",
    )
    (model,) = fake_whisper.instances
    assert model.args == ("large-v3", "cuda", "float16")
    call = model.calls[0]
    assert call["samples"] == 64000 and call["language"] == "ru" and call["word_timestamps"] is True
    assert call["vad_filter"] is True and call["condition_on_previous_text"] is False
    assert call["initial_prompt"] == "Лекция: Метрические методы. Термины: метод Парзена, ядро."
    assert [s.text for s in tr.segments] == ["Привет", "мир"]
    assert tr.segments[0].words[0].text == "Привет" and tr.segments[0].words[0].prob == 0.8
    assert (tr.device, tr.model, tr.language, tr.duration) == ("cuda", "large-v3", "ru", 4.0)
    assert tr.prompt.startswith("Лекция:") and tr.seconds >= 0
    assert any("модель large-v3 на CUDA (float16) загружена" in e for e in events)


def test_cuda_failure_falls_back_to_int8_then_cpu(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
    fake_whisper: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: make_probe())
    fake_whisper.fail_cuda = True
    events: list[str] = []
    tr = asr.transcribe(
        write_wav(tmp_path / "a.wav", 2.0), settings=make_settings(), on_event=events.append
    )
    assert [m.args for m in fake_whisper.instances] == [
        ("large-v3", "cuda", "float16"),
        ("large-v3", "cuda", "int8_float16"),
        ("small", "cpu", "int8"),
    ]
    assert tr.device == "cpu" and tr.model == "small" and len(tr.segments) == 2
    assert sum("сбой CUDA" in e for e in events) == 2
    assert any("пробую small на CPU" in e for e in events)


def test_a_failure_that_is_not_cuda_is_an_error(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
    fake_whisper: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "probe", lambda: make_probe(cuda_devices=0))

    def boom(self: Any, audio: Any, **kw: Any) -> Any:
        raise ValueError("битые данные")

    monkeypatch.setattr(FakeWhisperModel, "transcribe", boom)
    with pytest.raises(ExtractError, match="Ошибка распознавания речи: битые данные"):
        asr.transcribe(write_wav(tmp_path / "a.wav", 1.0), settings=make_settings())


def test_missing_group_gives_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "package_version", lambda name: None)
    with pytest.raises(ExtractError, match=r"uv sync --extra video"):
        asr.require_faster_whisper()
    assert asr.probe().installed is False


@pytest.mark.skipif(not asr.have_faster_whisper(), reason="faster-whisper не установлен")
def test_probe_sees_the_installed_package() -> None:
    found = asr.probe()
    assert found.installed and found.version and found.error is None
    assert found.ctranslate2 and found.cuda_devices >= 0
    assert found.cuda_ready == (found.cuda_devices > 0 and not found.cuda_missing)


# ---------------------------------------------------------------- helpers of video.py


def test_timestamps() -> None:
    assert vd.fmt_ts(0) == "0:00" and vd.fmt_ts(754.9) == "12:34" and vd.fmt_ts(3725) == "1:02:05"
    assert vd.parse_ts("12", "34", None) == 754 and vd.parse_ts("1", "02", "05") == 3725


def test_parse_fixed_keeps_timestamps_and_joins_orphans() -> None:
    text = (
        "[0:07] Первый абзац\nс переносом.\n\n"
        "[1:02:05] Второй, час спустя.\n\n"
        "Без метки — к предыдущему.\n\n"
        "[12:34] Формула:\n\n$$\nx^2\n$$\n"
    )
    paras = vd.parse_fixed(text, default_start=5.0)
    assert [(p.start, p.text.split("\n")[0]) for p in paras] == [
        (7, "Первый абзац с переносом."),
        (3725, "Второй, час спустя."),
        (754, "Формула:"),
    ]
    assert paras[1].text.endswith("Без метки — к предыдущему.")
    assert "$$" in paras[2].text
    assert vd.parse_fixed("Текст без меток", 30.0)[0].start == 30.0
    assert vd.parse_fixed("", 0.0) == []


def test_group_segments_breaks_on_pauses_and_length() -> None:
    segs = [seg(0, 2, "Раз."), seg(2.5, 4, "Два."), seg(8, 9, "После паузы."), seg(9.2, 10, "Ещё.")]
    paras = vd.group_segments(segs)
    assert [(p.start, p.text) for p in paras] == [(0, "Раз. Два."), (8, "После паузы. Ещё.")]
    long = [seg(i * 3.0, i * 3.0 + 2.9, "Слово " * 30 + "конец.") for i in range(5)]
    assert len(vd.group_segments(long)) >= 2


def test_split_at_boundaries_uses_the_words() -> None:
    s = seg(10.0, 20.0, "один два три четыре пять шесть семь восемь девять десять")
    out = vd.split_at_boundaries([s], [15.0])
    assert [x.text for x in out] == ["один два три четыре пять", "шесть семь восемь девять десять"]
    assert out[0].end <= 15.0 <= out[1].start
    assert vd.split_at_boundaries([s], [10.2, 19.9]) == [s]  # at the very edge: no split
    no_words = seg(10.0, 20.0, "а б в", words=False)
    assert vd.split_at_boundaries([no_words], [15.0]) == [no_words]


def test_chunks_end_at_section_starts() -> None:
    segs = [seg(i * 10.0, i * 10.0 + 9.0, f"фраза {i}") for i in range(180)]  # 30 minutes
    sections = [vd.Section(s, s + 150.0) for s in range(0, 1800, 150)]
    chunks = vd.plan_chunks(segs, sections, 1800.0)
    assert len(chunks) >= 3
    for chunk in chunks[:-1]:
        assert vd.FIX_CHUNK_MIN_S <= chunk.end - chunk.start <= vd.FIX_CHUNK_MAX_S
        assert chunk.start in {sec.start for sec in sections}
    assert sum(len(c.segments) for c in chunks) == 180  # nothing lost, nothing twice
    assert chunks[0].start == 0.0 and chunks[-1].end == 1800.0
    # one long section: cut at the longest pause near nine minutes
    gap_segs = [seg(i * 10.0, i * 10.0 + 9.0, f"x{i}") for i in range(54)] + [
        seg(i * 10.0 + 12.0, i * 10.0 + 21.0, f"y{i}") for i in range(54, 108)
    ]
    one = vd.plan_chunks(gap_segs, [vd.Section(0.0, 1100.0)], 1100.0)
    assert len(one) == 2 and abs(one[1].start - 552.0) < 1.0
    # a short lecture is one piece
    assert len(vd.plan_chunks(segs[:20], [vd.Section(0.0, 200.0)], 200.0)) == 1
    assert vd.plan_chunks([], [vd.Section(0.0, 10.0)], 10.0) == []


def kf(ordinal: int, kind: str, t: float, **kw: Any) -> fr.KeyFrame:
    frame = fr.KeyFrame(index=int(t), t=t, epoch=ordinal, level=100, ink=0.02, occlusion=0.0)
    frame.kind, frame.ordinal = kind, ordinal
    frame.file = f"frames/key_{ordinal:03d}_{fr.hhmmss(t)}.png" if kind == "board" else None
    for key, value in kw.items():
        setattr(frame, key, value)
    return frame


def test_sections_are_rendered_by_the_kind_of_their_frame() -> None:
    slide = kf(1, "slide", 10.0, slide={"source": "S1", "number": 7})
    board = kf(2, "board", 100.0)
    again = kf(3, "board", 200.0, duplicate_of=2)
    lost = kf(4, "board", 300.0)
    empty = kf(5, "board", 400.0)
    blank = kf(6, "other", 500.0)
    ordinals = {f.ordinal: f for f in (slide, board, again, lost, empty, blank)}
    result = vision.VisionResult(
        pages={2: "### Формула\n\n$$x$$", 5: vision.EMPTY_PAGE}, failed={4: "лимит"}
    )
    texts = {("S1", 7): vd.SlideText("Ядро Парзена", "текст")}

    def render(frame: fr.KeyFrame | None, **kw: Any) -> vd.Rendered:
        sec = vd.Section(0.0, 10.0, frame=frame, **kw)
        return vd.render_visual(
            "V1", sec, vis=result, use_vision=True, slide_texts=texts, ordinals=ordinals
        )

    assert render(slide) == vd.Rendered("Слайд 7. Ядро Парзена", "Слайд [[S1:s7]]")
    shown = render(board)
    assert shown.title == "Формула" and shown.visual.startswith("![Доска, 1:40](frames/key_002_")
    assert shown.visual.endswith("### Формула\n\n$$x$$")
    assert render(board, part=2, parts=3).title == "Формула, часть 2 из 3"
    repeat = render(again)
    assert repeat.title == "Формула (как в 1:40)" and "[[V1:1:40]]" in repeat.visual
    assert repeat.duplicate and not shown.duplicate
    assert "![" not in repeat.visual
    broken = render(lost)
    assert "Кадр не распознан агентом" in broken.visual and "(лимит)" in broken.visual
    assert "![Доска, 5:00]" in broken.visual
    for blanked in (render(empty), render(blank)):
        assert blanked == vd.Rendered("Без записей на кадре", "", blank=True)
    assert render(None) == vd.Rendered("", "")
    off = vd.render_visual(
        "V1",
        vd.Section(0.0, 1.0, frame=board),
        vis=None,
        use_vision=False,
        slide_texts={},
        ordinals=ordinals,
    )
    assert "не распознано: агент отключён" in off.visual and "![Доска," in off.visual


def test_body_skips_empty_sections_and_anchors_every_paragraph() -> None:
    first = vd.Section(
        0.0, 60.0, paragraphs=[vd.Paragraph(3.0, "Один."), vd.Paragraph(3725.0, "Два.")]
    )
    silent = vd.Section(60.0, 90.0)
    last = vd.Section(90.0, 3700.0)
    views = [
        vd.Rendered("Слайд 1", "Слайд [[S1:s1]]"),
        vd.Rendered("Доска", ""),
        vd.Rendered("", ""),
    ]
    text = vd.render_body("V1", [first, silent, last], views, show_titles=True)
    assert text == (
        "## [[V1:0:00]] 0:00–1:00 · Слайд 1\n\nСлайд [[S1:s1]]\n\n[[V1:0:03]] Один.\n\n"
        "[[V1:1:02:05]] Два."
    )
    plain = vd.render_body("A1", [first], [vd.Rendered("x", "")], show_titles=False)
    assert plain.startswith("## [[A1:0:00]] 0:00–1:00\n\n[[A1:0:03]] Один.")


def test_the_title_of_a_board_section_is_the_heading_of_its_transcription() -> None:
    assert vd.board_title("### Метод $k$ ближайших соседей ($k$ nearest neighbors)\n\nтекст") == (
        "Метод k ближайших соседей (k nearest neighbors)"
    )
    assert vd.board_title("Текст без заголовка") == "Доска"
    assert vd.board_title(None) == "Доска"
    assert vd.board_title("### $$\n\nтекст") == "Доска"  # a heading that is only markup
    long = vd.board_title("### " + "слово " * 40)
    assert len(long) == 80 and long.endswith("…")
    assert vd.board_title("## Не тот уровень\n### Нужный \\alpha **жирный**") == "Нужный жирный"


def test_the_same_board_again_without_speech_joins_the_section_before() -> None:
    first = vd.Section(0.0, 60.0, paragraphs=[vd.Paragraph(3.0, "Один.")])
    again = vd.Section(60.0, 90.0)
    talk = vd.Section(90.0, 150.0, paragraphs=[vd.Paragraph(95.0, "Два.")])
    views = [
        vd.Rendered("Тема", "![Доска, 0:50](frames/a.png)"),
        vd.Rendered("Тема (как в 0:50)", "Запись на доске та же.", duplicate=True),
        vd.Rendered("Тема (как в 0:50)", "Запись на доске та же.", duplicate=True),
    ]
    text = vd.render_body("V1", [first, again, talk], views, show_titles=True)
    heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert heads == [
        "## [[V1:0:00]] 0:00–1:30 · Тема",  # the silent repeat made the section longer
        "## [[V1:1:30]] 1:30–2:30 · Тема (как в 0:50)",  # a repeat with speech stays
    ]
    assert text.count("Запись на доске та же.") == 1
    # a duplicate first of all has nothing to join: it stays
    alone = vd.render_body("V1", [again], [views[1]], show_titles=True)
    assert alone.startswith("## [[V1:1:00]] 1:00–1:30 · Тема (как в 0:50)")


def test_glossary_and_slide_texts_come_from_other_sources(tmp_path: Path) -> None:
    topic = tmp_path / "t"
    (topic / "extracted" / "S1").mkdir(parents=True)
    (topic / "extracted" / "P1").mkdir(parents=True)
    (topic / "extracted" / "S1" / "summary.md").write_text(
        "## Аннотация\n\nТекст.\n\n## Оглавление\n\n- Метод [[S1:s1]]\n\n"
        "## Термины и обозначения\n\n"
        "- **Метод Парзена** (Parzen window) — оценка плотности [[S1:s4]]\n"
        "- $F_X(t)$ — функция распределения\n"
        "- **Ядро** — весовая функция [[S1:s5]]\n"
        "- Надарая–Ватсона — регрессия\n\n## Другое\n\n- **не термин** — после раздела\n",
        encoding="utf-8",
    )
    (topic / "extracted" / "P1" / "summary.md").write_text(
        "## Термины и обозначения\n\n- **Метод Парзена** — повтор\n"
        "- **Парзеновское окно** — окно\n",
        encoding="utf-8",
    )
    recs = [
        SourceRecord(id=i, kind=k, title=i, added="2026-10-04T00:00:00Z", status=st)  # type: ignore[arg-type]
        for i, k, st in (
            ("S1", "slides", "extracted"),
            ("P1", "pdf-text", "extracted"),
            ("V1", "video", "added"),
            ("W1", "web", "failed"),
        )
    ]
    terms, lines = vd.collect_glossary(topic, recs, exclude="V1")
    assert terms == [
        "Метод Парзена",
        "Ядро",
        "Надарая–Ватсона",
        "Метод Парзена",
        "Парзеновское окно",
    ]
    assert lines[0] == "- **Метод Парзена** (Parzen window) — оценка плотности"
    assert vd.collect_glossary(topic, recs, exclude="S1")[0] == [
        "Метод Парзена",
        "Парзеновское окно",
    ]
    assert vd.parse_terms("нет раздела") == []
    (topic / "extracted" / "S1" / "body.md").write_text(
        "## [[S1:s1]] Слайд 1. Метод ближайших соседей\n\nТекст первого.\n\n"
        "## [[S1:s2]] Слайд 2\n\nВторой $x$.\n\n## [[P1:p3]] Страница 3\n\nчужое\n",
        encoding="utf-8",
    )
    slides = vd.read_slide_texts(topic, "S1")
    assert slides[1].title == "Метод ближайших соседей" and slides[1].text == "Текст первого."
    assert slides[2].title == "" and "Второй" in slides[2].text and 3 not in slides
    assert vd.read_slide_texts(topic, "S9") == {}


# ---------------------------------------------------------------- the extractor with fakes

SPEECH = [
    "Сегодня метод парсен очень важен",
    "Ядро выбирается по расстоянию",
    "Запишем формулу на доске",
    "Теперь сотрём и начнём заново",
    "Переходим к следующему слайду",
]


def fake_segments(duration: float = DURATION) -> list[asr.Segment]:
    out, t, k = [], 1.0, 0
    while t + 5.0 <= duration:
        out.append(seg(t, t + 5.0, f"{SPEECH[k % len(SPEECH)]} номер {k}"))
        t, k = t + 6.0, k + 1
    return out


class FakeAsr:
    """Stands in for `asr.transcribe`."""

    def __init__(self, segments: list[asr.Segment] | None = None) -> None:
        self.segments = fake_segments() if segments is None else segments
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        audio: Path,
        *,
        settings: Settings,
        language: str | None = None,
        title: str = "",
        terms: Any = (),
        on_event: Callable[[str], None] | None = None,
        label: str = "ASR",
    ) -> asr.Transcript:
        with wave.open(str(audio), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        self.calls.append(
            {"audio": audio, "language": language, "title": title, "terms": list(terms)}
        )
        segs = [s for s in self.segments if s.end <= duration + 0.5]
        if on_event:
            on_event(f"{label}: распознано")
        return asr.Transcript(
            segments=segs,
            language="ru",
            duration=duration,
            model="large-v3",
            device="cuda",
            compute_type="float16",
            seconds=3.0,
            prompt=asr.build_prompt(title, terms),
        )


BOARD_TEXT = "### Запись на доске\n\n$$y^2 = 4ax$$\n\nОписание: линия"


class FakeAgent:
    """Stands in for `run_task`: the summary, the board frames and the corrector."""

    def __init__(
        self,
        *,
        fix: str = "ok",
        boards: str = "ok",
        ok: bool = True,
        problems: list[str] | None = None,
    ) -> None:
        self.fix, self.boards, self.ok = fix, boards, ok
        self.problems = problems or []
        self.calls: list[dict[str, Any]] = []

    def by_stage(self, stage: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["stage"] == stage]

    def __call__(
        self,
        bundle: Any,
        *,
        settings: Any,
        tier: str = "strong",
        backend: Any = None,
        fallback: bool = True,
        on_event: Any = None,
    ) -> RunResult:
        call = {
            "stage": bundle.stage,
            "tier": tier,
            "backend": backend,
            "files": [f.path for f in bundle.contract.files],
            "min_chars": [f.min_chars for f in bundle.contract.files],
            "inputs": sorted(p.name for p in bundle.inputs_dir.iterdir()),
            "images": [p.name for p in bundle.images],
            "task": bundle.task_path.read_text(encoding="utf-8"),
            "bundle": bundle,
        }
        self.calls.append(call)
        ok = self.ok
        if bundle.stage == "summary":
            (bundle.out_dir / sm.SUMMARY_FILE).write_text(
                "## Аннотация\n\nЛекция о метрических методах классификации: ближайшие соседи и "
                "ядро Парзена, доска и слайды.\n\n## Оглавление\n\n- Введение [[V1:0:00]]\n\n"
                "## Термины и обозначения\n\n- **Метод Парзена** — оценка плотности.\n",
                encoding="utf-8",
            )
        elif bundle.stage == "transcript_fix":
            ok = self._write_fix(bundle)
        else:
            if self.boards == "ok":
                for f in bundle.contract.files:
                    n = int(f.path[1:5])
                    (bundle.out_dir / f.path).write_text(f"{BOARD_TEXT} {n}", encoding="utf-8")
            elif self.boards == "empty":
                for f in bundle.contract.files:
                    (bundle.out_dir / f.path).write_text(vision.EMPTY_PAGE, encoding="utf-8")
            else:
                ok = False
        return RunResult(
            ok=ok,
            bundle=bundle,
            backend_used="claude",
            attempts=[],
            usage_total=Usage(input_tokens=10, output_tokens=5, cost_usd=0.01),
            final_text="готово",
            problems=list(self.problems) if not ok else [],
        )

    def _write_fix(self, bundle: Any) -> bool:
        if self.fix == "none":
            return False
        raw = (bundle.inputs_dir / "segments.md").read_text(encoding="utf-8").splitlines()
        if self.fix == "short":
            (bundle.out_dir / "fixed.md").write_text(raw[0].split("]")[0] + "] Коротко.\n", "utf-8")
            return True
        paragraphs = []
        for i in range(0, len(raw), 2):
            pair = raw[i : i + 2]
            head = pair[0].split("]")[0] + "]"
            body = " ".join(line.split("] ", 1)[1] for line in pair)
            paragraphs.append(f"{head} {body.replace('парсен', 'Парзена')}")
        (bundle.out_dir / "fixed.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        return True


@dataclass
class Env:
    settings: Settings
    asr: FakeAsr
    agent: FakeAgent
    lecture: dict[str, Path]
    tmp: Path
    events: list[str]

    def topic(self, **kw: Any) -> Path:
        return make_topic(self.settings, self.lecture, **kw)

    def run(self, topic: Path, **kw: Any) -> list[Any]:
        kw.setdefault("source_ids", ["V1"])
        return pipeline.extract_topic(self.settings, topic, on_event=self.events.append, **kw)


S1_BODY = (
    "## [[S1:s1]] Слайд 1. Метод ближайших соседей\n\nБлижайший сосед: $a(x)$.\n\n"
    "## [[S1:s2]] Слайд 2. Ядро Парзена\n\nОкно Парзена шириной $h$.\n"
)
S1_SUMMARY = (
    "## Аннотация\n\nСлайды.\n\n## Оглавление\n\n- Метод [[S1:s1]]\n\n"
    "## Термины и обозначения\n\n- **Метод Парзена** — оценка плотности [[S1:s2]]\n"
    "- **Метод ближайших соседей** — классификатор [[S1:s1]]\n"
)


def make_topic(
    settings: Settings, lecture: dict[str, Path], *, slides: bool = True, media: str = "video"
) -> Path:
    topic = create_topic(settings, title="Метрические методы", course="Машинное обучение")
    if slides:
        report = add_sources(settings, topic, [str(lecture["slides"])], kind="slides")
        assert [r.id for r in report.added] == ["S1"]
        out = topic / "extracted" / "S1"
        out.mkdir(parents=True)
        (out / "body.md").write_text(S1_BODY, encoding="utf-8")
        (out / sm.SUMMARY_FILE).write_text(S1_SUMMARY, encoding="utf-8")
        rec = next(r for r in list_sources(topic) if r.id == "S1")
        update_source(topic, rec.model_copy(update={"status": "extracted"}))
    report = add_sources(settings, topic, [str(lecture[media])], kind=media)
    assert [r.id for r in report.added] == [("V1" if media == "video" else "A1")]
    return topic


@pytest.fixture
def env(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
    lecture: dict[str, Path],
) -> Env:
    if tools.find_pandoc() is None:
        pytest.skip("Pandoc не найден")
    reset_cooling()
    settings = make_settings(general={"git_per_topic": False, "keep_video": True})
    fake_asr, agent = FakeAsr(), FakeAgent()
    monkeypatch.setattr(asr, "transcribe", fake_asr)
    monkeypatch.setattr(asr, "require_faster_whisper", lambda: None)
    monkeypatch.setattr(asr, "probe", lambda: make_probe())
    monkeypatch.setattr(vision, "run_task", agent)
    monkeypatch.setattr("h0lon.agents.run_task", agent)
    yield Env(settings, fake_asr, agent, lecture, tmp_path, [])
    reset_cooling()


def body_of(topic: Path, sid: str = "V1") -> str:
    return (topic / "extracted" / sid / "body.md").read_text(encoding="utf-8")


HEAD_RE = re.compile(r"^## \[\[V1:([\d:]+)\]\] ([\d:]+)–([\d:]+)(?: · (.+))?$", re.MULTILINE)
REAL_REQUIRE = asr.require_faster_whisper


def seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    return sum(p * 60**i for i, p in enumerate(reversed(parts)))


def headings(body: str) -> list[tuple[int, int, str | None]]:
    return [(seconds(m[2]), seconds(m[3]), m[4]) for m in HEAD_RE.finditer(body)]


def stored(topic: Path, sid: str = "V1") -> dict[str, Any]:
    return next(s for s in load_topic(topic).sources if s["id"] == sid)


def test_registry_extracts_video_and_audio() -> None:
    res = registry.resolve_extractor("video")
    assert not res.skipped and res.reason is None and res.extractor is not None
    assert res.extractor.kinds == ("video", "audio") and res.extractor.version == vd.VERSION
    assert registry.resolve_extractor("audio").extractor is not None
    assert registry.LATER_STAGES == {} and pipeline.get_extractor("video") is not None
    assert set(res.extractor.prompts) == {vision.PROMPT_REF, sm.prompt_id("transcript_fix")}


@needs_ffmpeg
def test_lecture_becomes_a_source_doc(env: Env) -> None:
    topic = env.topic()
    (result,) = env.run(topic)
    assert result.ok and not result.cached and result.warnings == [] and result.errors == []
    out = topic / "extracted" / "V1"

    # body.md: a section per epoch with a time range and the slide or the board
    body = body_of(topic)
    sections = headings(body)
    titles = [t for _a, _b, t in sections]
    assert titles == [
        "Слайд 1. Метод ближайших соседей",
        "Слайд 2. Ядро Парзена",
        "Запись на доске",  # the heading of the transcription of the frame
        "Запись на доске",
        "Слайд 1. Метод ближайших соседей",
    ]
    for (_s, end, _t), (start, _e, _t2) in itertools.pairwise(sections):
        assert end == start  # the sections follow each other
    for (start, _e, _t), want in zip(sections, (0, 12, 24, 54, 80), strict=True):
        assert abs(start - want) <= 2
    assert sections[-1][1] == 92 and body.startswith("## [[V1:0:00]] 0:00–0:12 · Слайд 1.")
    assert body.count("Слайд [[S1:s1]]") == 2 and body.count("Слайд [[S1:s2]]") == 1
    assert re.findall(r"!\[Доска, [\d:]+\]\(frames/key_\d{3}_\d{6}\.png\)", body) != []
    assert "$$y^2 = 4ax$$" in body  # the transcription of the board
    # the speech: corrected, every paragraph with its anchor
    assert "Парзена" in body and "парсен" not in body
    paragraphs = [ln for ln in body.splitlines() if re.match(r"^\[\[V1:[\d:]+\]\] \S", ln)]
    assert len(paragraphs) >= 6 and all(len(p) > 30 for p in paragraphs)
    first = paragraphs[0]
    assert re.match(r"^\[\[V1:0:0[0-3]\]\]", first)

    # files: key frames only for boards; slides are links
    pngs = sorted(p.name for p in (out / "frames").glob("*.png"))
    assert len(pngs) == 2 and all(re.fullmatch(r"key_00[34]_\d{6}\.png", n) for n in pngs)
    data = json.loads((out / "frames.json").read_text("utf-8"))
    assert [f["type"] for f in data["frames"]] == ["slide", "slide", "board", "board", "slide"]
    assert [f["matched_slide"]["number"] for f in data["frames"] if f["matched_slide"]] == [1, 2, 1]
    assert [bool(f["file"]) for f in data["frames"]] == [False, False, True, True, False]
    assert len(data["epochs"]) == 5 and data["thresholds"]["hold"] == 3
    for name in ("audio.wav", "transcript.json", "transcript.srt", "frames.cache.npz", "source.md"):
        assert (out / name).is_file(), name
    assert [p.name for p in (out / "fix").glob("*.md")] != []
    srt = (out / "transcript.srt").read_text("utf-8")
    assert srt.startswith("1\n00:00:01,000 --> 00:00:06,000")

    # quality and the record
    q = result.quality
    assert (q["minutes"], q["epochs"], q["frames_board"], q["frames_slide"]) == (1.53, 5, 2, 3)
    assert (q["asr_device"], q["asr_model"], q["asr_seconds"]) == ("cuda", "large-v3", 3.0)
    assert any(
        "faster-whisper large-v3, CUDA" in n and "исправлена агентом" in n for n in q["notes"]
    )
    assert any("Кадров, совпавших со слайдами" in n for n in q["notes"])
    rec = stored(topic)
    assert rec["status"] == "extracted" and rec["units"] == {"minutes": 1.53}
    assert rec["quality"]["frames_board"] == 2 and rec["error"] is None
    assert result.pages_total == 5 and result.pages_vision == 2 and result.agent_runs == 2
    blocks = [ln for ln in (out / "blocks.jsonl").read_text("utf-8").splitlines() if ln]
    assert len(blocks) == result.blocks >= 15
    front = (out / "source.md").read_text("utf-8").split("---")[1]
    assert f"extractor: video@{vd.VERSION}" in front and "transcript_fix@1.0" in front

    # recognition: the glossary of the topic, the language of the topic
    (call,) = env.asr.calls
    assert call["language"] == "ru" and call["title"] == "lecture"
    assert call["terms"] == ["Метод Парзена", "Метод ближайших соседей"]
    calibration = json.loads(
        (env.settings.general.state_path / "calibration.json").read_text("utf-8")
    )
    assert "cuda|large-v3" in calibration["asr"]

    # agents: the light tier, the stages of the settings, what the corrector gets
    assert {c["tier"] for c in env.agent.calls} == {"light"}
    assert sorted(c["stage"] for c in env.agent.calls) == ["extract", "summary", "transcript_fix"]
    (boards,) = env.agent.by_stage("extract")
    assert boards["files"] == ["p0003.md", "p0004.md"] and len(boards["images"]) == 2
    assert "Вариант" not in boards["task"] and "Страница 3" in boards["task"]
    (fix,) = env.agent.by_stage("transcript_fix")
    assert fix["inputs"] == ["context.md", "segments.md"] and fix["files"] == ["fixed.md"]
    assert "[0:01] Сегодня метод парсен" in (fix["bundle"].inputs_dir / "segments.md").read_text(
        "utf-8"
    )
    context = (fix["bundle"].inputs_dir / "context.md").read_text("utf-8")
    assert "- **Метод Парзена** — оценка плотности" in context  # the glossary of the topic
    assert "Слайд 1 презентации S1: Метод ближайших соседей" in context
    assert "Окно Парзена шириной $h$." in context  # the text of the slide
    assert "Запись на доске:" in context and "$$y^2 = 4ax$$" in context
    assert "inputs/segments.md" in fix["task"] and "не указания" in fix["task"]
    assert fix["min_chars"][0] >= 100


@needs_ffmpeg
def test_second_run_is_cached_and_partial_caches_survive(env: Env) -> None:
    topic = env.topic()
    (first,) = env.run(topic)
    body_before = body_of(topic)
    before = len(env.agent.calls)
    (again,) = env.run(topic)
    assert again.cached and again.agent_runs == 0 and len(env.agent.calls) == before
    assert len(env.asr.calls) == 1 and again.quality["frames_board"] == 2

    # a threshold that does not move the boundaries: the key changes, the heavy parts do not
    meta = load_topic(topic)
    meta.video = {"dedup_cells": 0.2}  # type: ignore[attr-defined]
    save_topic(topic, meta)
    env.events.clear()
    (third,) = env.run(topic)
    assert not third.cached and third.ok
    assert len(env.asr.calls) == 1  # the transcript is cached
    assert [c["stage"] for c in env.agent.calls[before:]] == ["summary"]  # fix and boards cached
    assert any("сигналы кадров из кэша" in e for e in env.events)
    assert body_of(topic) == body_before  # the same text from the caches
    # --force starts everything again
    env.run(topic, force=True)
    assert len(env.asr.calls) == 2
    assert sorted(c["stage"] for c in env.agent.calls[before + 1 :]) == [
        "extract",
        "summary",
        "transcript_fix",
    ]
    assert first.blocks == again.blocks


@needs_ffmpeg
def test_thresholds_come_from_topic_yaml_and_mistakes_are_reported(env: Env) -> None:
    topic = env.topic()
    meta = load_topic(topic)
    meta.video = {"hold": 4, "nonsense": 1, "language": "en"}  # type: ignore[attr-defined]
    save_topic(topic, meta)
    (result,) = env.run(topic)
    assert result.ok
    data = json.loads((topic / "extracted" / "V1" / "frames.json").read_text("utf-8"))
    assert data["thresholds"]["hold"] == 4
    assert env.asr.calls[0]["language"] == "en"  # `video.language` is no threshold
    assert any("video.nonsense: неизвестный параметр" in w for w in result.warnings)
    assert not any("language" in w for w in result.warnings)


@needs_ffmpeg
def test_without_the_agent_the_speech_stays_raw_and_boards_are_placeholders(env: Env) -> None:
    topic = env.topic()
    (result,) = env.run(topic, use_vision=False)
    assert result.ok and result.agent_runs == 0 and env.agent.calls == []
    body = body_of(topic)
    assert "метод парсен" in body and "метод Парзена" not in body  # as recognized
    assert body.count("не распознано: агент отключён") == 2 and "![Доска," in body
    assert any("Агент отключён" in n for n in result.quality["notes"])
    assert any("правка агентом не выполнена" in n for n in result.quality["notes"])
    assert len(env.asr.calls) == 1


@needs_ffmpeg
def test_a_correction_that_cuts_the_speech_is_rejected(env: Env) -> None:
    env.agent.fix = "short"
    topic = env.topic()
    (result,) = env.run(topic)
    assert result.ok
    assert any(
        "правка отвергнута" in w and "ничего не должен сокращать" in w for w in result.warnings
    )
    body = body_of(topic)
    assert "парсен" in body and "Коротко." not in body  # the raw text is used
    assert result.quality["fix_chunks_raw"] == 1
    assert any("Фрагментов речи без правки агентом: 1" in n for n in result.quality["notes"])


@needs_ffmpeg
def test_a_failed_correction_falls_back_to_the_raw_text(env: Env) -> None:
    env.agent.fix = "none"
    env.agent.problems = ["лимит агента"]
    topic = env.topic()
    (result,) = env.run(topic)
    assert result.ok and any(
        "правка не получена" in w and "лимит агента" in w for w in result.warnings
    )
    assert "парсен" in body_of(topic)


@needs_ffmpeg
def test_boards_the_agent_found_empty_are_dropped_and_failures_are_marked(env: Env) -> None:
    env.agent.boards = "empty"
    topic = env.topic()
    (result,) = env.run(topic)
    body = body_of(topic)
    assert [t for _a, _b, t in headings(body)].count("Без записей на кадре") == 2
    assert "![Доска," not in body and result.ok

    env.agent.boards = "fail"
    env.agent.problems = ["не сегодня"]
    topic2 = make_topic_again(env)
    (failed,) = env.run(topic2)
    body2 = (topic2 / "extracted" / "V1" / "body.md").read_text("utf-8")
    assert body2.count("Кадр не распознан агентом") == 2 and "![Доска," in body2
    assert any("не распознан агентом" in w for w in failed.warnings)


def make_topic_again(env: Env) -> Path:
    other = Settings(
        general={
            "workspaces": env.tmp / "ws2",
            "state_dir": env.tmp / "state2",
            "git_per_topic": False,
            "keep_video": True,
        }
    )
    env.settings = other
    return env.topic()


@needs_ffmpeg
def test_video_max_minutes_keeps_only_the_start(
    env: Env, make_settings: Callable[..., Settings]
) -> None:
    env.settings = make_settings(
        general={"git_per_topic": False, "keep_video": True}, compute={"video_max_minutes": 1}
    )
    topic = env.topic()
    (result,) = env.run(topic)
    assert result.ok and result.quality["minutes"] == 1.0
    assert stored(topic)["units"] == {"minutes": 1.0}
    assert any("Обработаны только первые 1 мин записи из 2" in n for n in result.quality["notes"])
    assert max(end for _s, end, _t in headings(body_of(topic))) <= 60
    frames = json.loads((topic / "extracted" / "V1" / "frames.json").read_text("utf-8"))
    assert frames["duration"] == 60.0 and len(frames["epochs"]) == 4
    with wave.open(str(topic / "extracted" / "V1" / "audio.wav"), "rb") as w:
        assert abs(w.getnframes() / w.getframerate() - 60.0) < 0.1
    assert max(s.end for s in asr.read_transcript(topic / "extracted" / "V1").segments) <= 60.5  # type: ignore[union-attr]
    # the limit is part of the cache key
    env.settings = make_settings(general={"git_per_topic": False, "keep_video": True})
    (full,) = env.run(topic)
    assert not full.cached and full.quality["minutes"] == 1.53


@needs_ffmpeg
def test_the_video_is_deleted_unless_kept(env: Env, make_settings: Callable[..., Settings]) -> None:
    env.settings = make_settings(general={"git_per_topic": False, "keep_video": False})
    topic = env.topic()
    (result,) = env.run(topic)
    assert result.ok and not any((topic / "sources").glob("V1_*"))
    assert any("видео удалено" in e for e in env.events)
    assert stored(topic)["status"] == "extracted"  # the record stays
    (again,) = env.run(topic)  # the cache needs no file
    assert again.cached
    (forced,) = env.run(topic, force=True)
    assert not forced.ok and "нет ссылки для повторной загрузки" in forced.errors[0]
    assert stored(topic)["status"] == "failed"
    # an audio file is small and stays
    assert (topic / "extracted" / "V1" / "audio.wav").is_file()


def audio_topic(env: Env, tmp: Path, minutes: float) -> Path:
    path = write_wav(tmp / "talk.wav", minutes * 60)
    topic = create_topic(env.settings, title="Лекция", course="Курс")
    report = add_sources(env.settings, topic, [str(path)], kind="audio")
    assert [r.id for r in report.added] == ["A1"]
    return topic


@needs_ffmpeg
def test_audio_has_sections_of_five_minutes_and_no_frames(env: Env) -> None:
    env.asr.segments = [
        seg(i * 30.0, i * 30.0 + 20.0, f"Фраза номер {i} про метод парсен") for i in range(23)
    ]
    topic = audio_topic(env, env.tmp, 11.5)
    (result,) = pipeline.extract_topic(env.settings, topic, on_event=env.events.append)
    assert result.ok and result.quality["minutes"] == 11.5
    body = body_of(topic, "A1")
    heads = re.findall(r"^## \[\[A1:([\d:]+)\]\] ([\d:]+)–([\d:]+)$", body, re.MULTILINE)
    assert [(a, b, c) for a, b, c in heads] == [
        ("0:00", "0:00", "5:00"),
        ("5:00", "5:00", "10:00"),
        ("10:00", "10:00", "11:30"),
    ]
    assert "![" not in body and " · " not in body and "Парзена" in body
    out = topic / "extracted" / "A1"
    assert not (out / "frames").exists() and not (out / "frames.json").exists()
    assert result.quality["frames_board"] == 0 and result.pages_vision == 0
    assert {c["stage"] for c in env.agent.calls} == {"transcript_fix", "summary"}
    assert len(env.agent.by_stage("transcript_fix")) >= 2  # 11.5 minutes: two pieces
    assert stored(topic, "A1")["units"] == {"minutes": 11.5}


@needs_ffmpeg
def test_a_video_without_sound_is_an_error(env: Env, tmp_path: Path) -> None:
    silent = build_video(tmp_path / "silent.mp4", seconds=4, audio=False)
    topic = create_topic(env.settings, title="Тема", course="Курс")
    add_sources(env.settings, topic, [str(silent)], kind="video")
    (result,) = env.run(topic)
    assert not result.ok and "нет звуковой дорожки" in result.errors[0]
    assert stored(topic)["status"] == "failed"


@needs_ffmpeg
def test_missing_group_and_missing_ffmpeg_are_clear_errors(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    topic = env.topic()
    monkeypatch.setattr(asr, "require_faster_whisper", REAL_REQUIRE)
    monkeypatch.setattr(asr, "package_version", lambda name: None)
    (result,) = env.run(topic)
    assert not result.ok and "uv sync --extra video" in result.errors[0]
    assert stored(topic)["status"] == "failed" and env.asr.calls == []
    monkeypatch.setattr(asr, "package_version", lambda name: "1.2.1")
    monkeypatch.setattr(tools, "find_simple", lambda name: None)
    (result,) = env.run(topic)
    assert not result.ok and "ffprobe не найден" in result.errors[0]
    assert "winget install Gyan.FFmpeg" in result.errors[0]


# ---------------------------------------------------------------- download of a link

URL = "https://www.youtube.com/watch?v=abcdefghijk"
REAL_RUN = procutil.run


class FakeYtDlp:
    """Stands in for the yt-dlp process: puts the lecture into the folder of `-o`."""

    def __init__(self, video: Path, title: str = "Машинное обучение. Метрические методы") -> None:
        self.video, self.title = video, title
        self.calls: list[list[str]] = []
        self.fail = False

    def __call__(self, argv: Any, **kw: Any) -> procutil.ProcResult:
        args = [str(a) for a in argv]
        if args[1:3] != ["-m", "yt_dlp"]:
            return REAL_RUN(argv, **kw)
        self.calls.append(args)
        if self.fail:
            return procutil.ProcResult(args, 1, "", "WARNING: x\nERROR: Video unavailable\n", 0.1)
        template = Path(args[args.index("-o") + 1])
        target = template.parent / "abcdefghijk.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.video, target)
        lines = ["H0LONP  0.0% Unknown", "H0LONP 55.0% 00:03", "H0LONP100.0% 00:00"]
        for line in lines:
            if kw.get("on_stdout_line"):
                kw["on_stdout_line"](line)
        meta = {"id": "abcdefghijk", "title": self.title, "duration": 92, "filepath": str(target)}
        lines.append("H0LONMETA " + json.dumps(meta))
        return procutil.ProcResult(args, 0, "\n".join(lines), "", 0.2, stdout_lines=lines)


@pytest.fixture
def ytdlp(env: Env, monkeypatch: pytest.MonkeyPatch) -> FakeYtDlp:
    fake = FakeYtDlp(env.lecture["video"])
    monkeypatch.setattr(procutil, "run", fake)
    return fake


@needs_ffmpeg
def test_a_link_is_downloaded_and_the_record_is_updated(env: Env, ytdlp: FakeYtDlp) -> None:
    topic = create_topic(env.settings, title="Метрические методы", course="Курс")
    add_sources(env.settings, topic, [URL])
    assert stored(topic)["file"] is None and stored(topic)["sha256"] is None
    (result,) = env.run(topic)
    assert result.ok, result.errors
    (call,) = ytdlp.calls
    assert call[-1] == URL and "--no-playlist" in call and "--ffmpeg-location" in call
    assert call[call.index("-S") + 1] == "res:720,vcodec:h264,acodec:m4a"
    assert "--download-sections" not in call and "--merge-output-format" in call
    rec = stored(topic)
    assert rec["file"] == "sources/V1_mashinnoe-obuchenie-metricheskie-metody.mp4"
    assert (topic / rec["file"]).is_file() and rec["sha256"] and rec["size"] > 1000
    assert rec["title"] == "Машинное обучение. Метрические методы"  # from the video, not the URL
    assert rec["url"] == URL and rec["units"] == {"minutes": 1.53}
    assert rec["download"] == {"limit_minutes": 0, "full_minutes": 1.53}
    assert not list((topic / "sources").glob(".incoming-dl-*"))  # no leftovers
    assert any("загрузка 50%" in e or "загрузка 5" in e for e in env.events)
    # the second run needs neither the network nor the download
    (again,) = env.run(topic)
    assert again.cached and len(ytdlp.calls) == 1
    # the title the user gave stays
    topic2 = create_topic(env.settings, title="Другая", course="Курс")
    add_sources(env.settings, topic2, [URL], title="Моё название")
    env.run(topic2)
    assert stored(topic2)["title"] == "Моё название"


@needs_ffmpeg
def test_a_limited_download_is_repeated_when_the_limit_is_lifted(
    env: Env, ytdlp: FakeYtDlp, make_settings: Callable[..., Settings]
) -> None:
    env.settings = make_settings(
        general={"git_per_topic": False, "keep_video": True}, compute={"video_max_minutes": 1}
    )
    topic = create_topic(env.settings, title="Тема", course="Курс")
    add_sources(env.settings, topic, [URL])
    (limited,) = env.run(topic)
    assert limited.ok and limited.quality["minutes"] == 1.0
    assert ytdlp.calls[0][ytdlp.calls[0].index("--download-sections") + 1] == "*0-60"
    assert stored(topic)["download"]["limit_minutes"] == 1
    # a smaller limit needs no new download, a bigger one does
    env.settings = make_settings(
        general={"git_per_topic": False, "keep_video": True}, compute={"video_max_minutes": 90}
    )
    (more,) = env.run(topic)
    assert more.ok and len(ytdlp.calls) == 2 and "--download-sections" in ytdlp.calls[1]
    env.settings = make_settings(general={"git_per_topic": False, "keep_video": True})
    (full,) = env.run(topic)
    assert full.ok and len(ytdlp.calls) == 3 and "--download-sections" not in ytdlp.calls[2]
    assert stored(topic)["download"]["limit_minutes"] == 0 and full.quality["minutes"] == 1.53
    assert len(list((topic / "sources").glob("V1_*"))) == 1  # the old file is replaced


def test_download_watch_reports_the_growth_of_the_file(tmp_path: Path) -> None:
    events: list[str] = []
    with vd._size_watch(tmp_path, events.append, "V1", every=0.05):
        (tmp_path / "x.mp4.part").write_bytes(b"x" * 2_000_000)
        time.sleep(0.4)
    assert any(e == "V1: загружено 2 МБ" for e in events)
    assert len(events) == 1  # the size did not change after that: nothing new to say


@needs_ffmpeg
def test_a_failed_download_is_an_error_with_the_reason(env: Env, ytdlp: FakeYtDlp) -> None:
    ytdlp.fail = True
    topic = create_topic(env.settings, title="Тема", course="Курс")
    add_sources(env.settings, topic, [URL])
    (result,) = env.run(topic)
    assert not result.ok and "Не удалось скачать" in result.errors[0]
    assert "Video unavailable" in result.errors[0]
    rec = stored(topic)
    assert rec["status"] == "failed" and rec["file"] is None
    assert not list((topic / "sources").glob(".incoming-dl-*"))


# ---------------------------------------------------------------- plan


@needs_ffmpeg
def test_dry_run_plan(env: Env) -> None:
    topic = env.topic()
    plans = pipeline.extract_topic(env.settings, topic, dry_run=True, source_ids=["V1"])
    (plan,) = plans
    notes = "; ".join(plan.notes)
    assert (
        "Длительность: 1.5 мин" in notes
        and "Правка речи агентом (лёгкий уровень): 1 прогон" in notes
    )
    assert "Распознавание речи (large-v3 на GPU)" in notes and "оценка" in notes
    assert plan.agent_runs == 3  # the corrector, the boards (at least one run) and the summary
    assert env.agent.calls == [] and env.asr.calls == []
    assert not (topic / "extracted" / "V1").exists()
    off = pipeline.extract_topic(
        env.settings, topic, dry_run=True, source_ids=["V1"], use_vision=False
    )
    assert off[0].agent_runs == 0 and "Агент отключён" in "; ".join(off[0].notes)
    # a link that is not downloaded yet
    topic2 = create_topic(env.settings, title="Тема", course="Курс")
    add_sources(env.settings, topic2, [URL])
    (link,) = pipeline.extract_topic(env.settings, topic2, dry_run=True)
    assert "будет скачано yt-dlp" in "; ".join(link.notes)


def test_print_results_show_video(env: Env) -> None:
    from rich.console import Console

    topic = env.topic()
    console = Console(record=True, width=140)
    pipeline.print_results(env.run(topic), console=console)
    text = console.export_text()
    assert "Извлечение источников" in text and "готово" in text and "2/5" in text
