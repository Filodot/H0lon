"""Key frames of a lecture video (PRD 6.2; docs/ARCHITECTURE.md, «Видео и аудио (M5)»).

The goal is the smallest set of frames that carries everything written on the board or shown
on the screen, without the lecturer in front of it, in the right order, without duplicates of
the slides the topic already has.

1. Signals. ffmpeg decodes the video at 1 frame per second, grayscale, 640 px wide, in chunks
   of ten minutes; per frame the code keeps a 64×64 thumbnail (hash, correlation, occlusion)
   and a map of «ink» (edge pixels) per block of 8×8 pixels. They are cached in
   `frames.cache.npz`, so selecting again with other thresholds takes seconds.
2. What counts. Blocks that hold ink almost all the time (the template of the slides, the
   frame of the board, a logo, the furniture of the room) are the *template*: they say nothing
   about the content and are left out. The *content area* is where the rest of the ink lives;
   when it is a part of the frame (slides next to the camera on the lecturer) it is found
   automatically (`video.roi: auto`), can be set by hand (`video.roi: [x0, y0, x1, y1]`) or
   switched off (`video.roi: none`); the key frames are cut to it.
3. Epochs. The static ink of a state is the minimum of the ink maps over `hold` frames: what
   the lecturer covers or what moves does not survive it. A boundary is a place where 35 % of
   the static ink of the last `hold` seconds is gone in every one of the next `hold` frames (a
   slide was switched, the shot changed, the board was erased) or where the ink falls by 40 %
   below the peak of the epoch. Inside an epoch content only accumulates.
4. The frame of an epoch is a real frame of its final plateau (the static ink within 10 % of
   the peak of the last 15 frames) with the least occlusion — the difference from the temporal
   median, which removes the moving lecturer — and the most ink. Epochs longer than 3 minutes
   get intermediate frames where the ink reached 25/50/75 % of the final level, to keep the
   order of the writing.
5. Classification: `slide` (the frame matches a slide of a presentation of the same topic —
   perceptual hash, correlation and ink map — and a link to the slide replaces the picture),
   `board`, `other` (blank). Duplicates of earlier frames (the same ink) are marked.

Everything is numpy and Pillow; no OpenCV. Thresholds are in `Thresholds` (`topic.yaml`,
section `video:`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from h0lon import procutil
from h0lon.extract.registry import ExtractError

SIGNALS_VERSION = "2"
ANALYSIS_WIDTH = 640
SMALL = 64  # thumbnail side: 64 = 2 × 32, so the hash takes exact 2×2 means
FINE_BLOCK = 8  # pixels of the analysis frame per block of the ink map
COARSE_ROWS, COARSE_COLS = 18, 32  # grid of the map a frame is compared with a slide by
HASH_BITS = 63
CHUNK_SECONDS = 600
EPS = 1e-6
BG_LEVEL = 0.05  # a block with at least this share of ink pixels holds ink
TEMPLATE_MIN_FRAMES = 30  # shorter videos have no «always there» to speak of
ROI_MARGIN = 0.03  # the key frame is cut a little wider than the content area
ROI_MAX_AREA = 0.65  # a content area larger than this share of the frame is no area


# ---------------------------------------------------------------- thresholds


@dataclass
class Thresholds:
    """Parameters of the selection; `topic.yaml` → `video:` overrides any of them."""

    fps: float = 1.0  # frames per second of the analysis (0.5 for pure slide lectures)
    edge: int = 40  # |gx|+|gy| above this is an «ink» pixel (0..510)
    hold: int = 3  # frames a state must hold on each side of a boundary (≥ 2 s at 1 fps)
    jump_removed: float = 0.35  # share of the static ink that vanished: a new state begins
    erase_drop: float = 0.40  # fall of the ink below the epoch's peak: the board was erased
    ink_min: float = 0.002  # static ink density below this is a blank frame
    min_epoch: int = 2  # frames; closer boundaries are merged
    tail: int = 15  # frames at the end of an epoch for the temporal median
    long_epoch: int = 180  # seconds: longer epochs get intermediate frames
    occlusion_delta: int = 40  # gray levels from the median that count as «covered»
    dedup_cells: float = 0.12  # the same board: ink maps closer than this
    match_cells: float = 0.35  # slide match: ink map closer than this …
    match_corr: float = 0.80  # … thumbnail correlation at least this …
    match_hash: int = 16  # … hash closer than this many bits
    # content area: "auto", "none" or [x0, y0, x1, y1] in shares of the frame
    roi: list[float] | str | None = "auto"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> tuple[Thresholds, list[str]]:
        """(thresholds, problems): wrong keys or values are reported and ignored."""
        th = cls()
        problems: list[str] = []
        known = {f.name: f for f in fields(cls)}
        for key, value in (data or {}).items():
            f = known.get(str(key))
            if f is None:
                problems.append(
                    f"video.{key}: неизвестный параметр (допустимы: {', '.join(known)})"
                )
                continue
            try:
                if key == "roi":
                    th.roi = parse_roi(value)
                elif f.type in ("int", int):
                    setattr(th, key, int(value))
                else:
                    setattr(th, key, float(value))
            except (TypeError, ValueError) as exc:
                problems.append(f"video.{key}: неверное значение {value!r} ({exc})")
        if th.fps <= 0 or th.fps > 5:
            problems.append(f"video.fps: {th.fps} вне диапазона (0, 5] — взято 1")
            th.fps = 1.0
        th.hold = max(2, th.hold)
        th.min_epoch = max(1, th.min_epoch)
        th.tail = max(3, th.tail)
        return th, problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signal_key(self) -> str:
        """Part of the thresholds the cached signals depend on."""
        return f"fps={self.fps:g};edge={self.edge}"


def parse_roi(value: Any) -> list[float] | str | None:
    """`video.roi`: «auto», «none» (also null, off, false) or four shares of the frame."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        if text in ("", "none", "off", "no", "false"):
            return None
        raise ValueError("допустимо auto, none или четыре числа x0, y0, x1, y1")
    roi = [float(v) for v in value]
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("нужны четыре числа x0, y0, x1, y1 в долях кадра, x0 < x1, y0 < y1")
    return roi


# ---------------------------------------------------------------- signals


@dataclass
class Signals:
    small: np.ndarray  # (n, SMALL, SMALL) uint8 thumbnails
    fine: np.ndarray  # (n, rows, cols) float32 share of ink pixels per block of FINE_BLOCK²
    times: np.ndarray  # (n,) float64 seconds in the video
    width: int  # size of the analysis frames
    height: int

    @property
    def n(self) -> int:
        return int(self.small.shape[0])


def analysis_size(video_width: int, video_height: int) -> tuple[int, int]:
    w = ANALYSIS_WIDTH
    h = max(2, round(w * max(video_height, 1) / max(video_width, 1) / 2) * 2)
    return w, h


def to_small(gray: Image.Image | np.ndarray) -> np.ndarray:
    """(SMALL, SMALL) uint8 thumbnail (area mean) of a grayscale picture of any size."""
    img = gray if isinstance(gray, Image.Image) else Image.fromarray(gray)
    return np.asarray(img.convert("L").resize((SMALL, SMALL), Image.Resampling.BOX), dtype=np.uint8)


def ink_blocks(gray: np.ndarray, edge_threshold: int) -> np.ndarray:
    """Share of «ink» (strong-gradient) pixels per block of FINE_BLOCK × FINE_BLOCK pixels."""
    a = gray.astype(np.int16)
    gx = np.abs(a[:-1, 1:] - a[:-1, :-1])
    gy = np.abs(a[1:, :-1] - a[:-1, :-1])
    edge = (gx + gy) > edge_threshold
    rows, cols = edge.shape[0] // FINE_BLOCK, edge.shape[1] // FINE_BLOCK
    blocks = edge[: rows * FINE_BLOCK, : cols * FINE_BLOCK].reshape(
        rows, FINE_BLOCK, cols, FINE_BLOCK
    )
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def frame_features(gray: np.ndarray, edge_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    return to_small(gray), ink_blocks(gray, edge_threshold)


def _ffmpeg_failure(res: procutil.ProcResult) -> str:
    text = (res.stderr or res.error or "").strip().splitlines()
    return (text[-1] if text else f"код {res.exit_code}")[:300]


def compute_signals(
    ffmpeg: Path | str,
    video: Path,
    *,
    video_size: tuple[int, int],
    start: float,
    duration: float,
    th: Thresholds,
    work_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> Signals:
    """Decode `video` from `start` for `duration` seconds and measure every sampled frame."""
    w, h = analysis_size(*video_size)
    work_dir.mkdir(parents=True, exist_ok=True)
    smalls: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    chunks = max(1, math.ceil(duration / CHUNK_SECONDS))
    for k in range(chunks):
        offset = k * CHUNK_SECONDS
        span = min(CHUNK_SECONDS, duration - offset)
        if span <= 0:
            break
        raw = work_dir / f"signals_{k}.gray"
        argv = [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start + offset:.3f}",
            "-t",
            f"{span:.3f}",
            "-i",
            str(video),
            "-an",
            "-sn",
            "-vf",
            f"fps={th.fps:g},scale={w}:{h}:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(raw),
        ]
        res = procutil.run(argv, timeout=900 + span * 3)
        if not res.ok:
            raise ExtractError(f"ffmpeg не смог декодировать кадры видео: {_ffmpeg_failure(res)}")
        try:
            data = np.fromfile(raw, dtype=np.uint8)
        finally:
            with contextlib.suppress(OSError):
                raw.unlink()
        count = data.size // (w * h)
        if count == 0:
            continue
        frames = data[: count * w * h].reshape(count, h, w)
        for i in range(count):
            s, b = frame_features(frames[i], th.edge)
            smalls.append(s)
            blocks.append(b)
        times.append(start + offset + np.arange(count, dtype=np.float64) / th.fps)
        if on_progress:
            done = min(offset + span, duration) / 60
            on_progress(f"кадры: проанализировано {done:.1f} из {duration / 60:.1f} мин")
    if not smalls:
        raise ExtractError("В видео не найдено ни одного кадра")
    return Signals(
        small=np.stack(smalls),
        fine=np.stack(blocks).astype(np.float32),
        times=np.concatenate(times),
        width=w,
        height=h,
    )


def signals_key(video_id: str, th: Thresholds, start: float, duration: float) -> str:
    payload = f"{SIGNALS_VERSION}|{video_id}|{th.signal_key()}|{start:.3f}|{duration:.3f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def save_signals(path: Path, sig: Signals, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            key=np.array(key),
            small=sig.small,
            fine=np.rint(sig.fine * 255.0).astype(np.uint8),
            times=sig.times,
            size=np.array([sig.width, sig.height]),
        )
    tmp.replace(path)


def load_signals(path: Path, key: str) -> Signals | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            size = data["size"]
            return Signals(
                small=data["small"],
                fine=data["fine"].astype(np.float32) / 255.0,
                times=data["times"],
                width=int(size[0]),
                height=int(size[1]),
            )
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------- distances and filters

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POPCOUNT[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1)


def _dct_matrix(n: int = 32) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


_DCT = _dct_matrix(32)
_BIT_WEIGHTS = (np.uint64(1) << np.arange(HASH_BITS, dtype=np.uint64)).astype(np.uint64)


def _slice(start: float, stop: float, size: int) -> tuple[int, int]:
    """Index range of the share [start, stop) of an axis of `size` cells (at least one)."""
    lo = min(int(start * size), size - 1)
    return lo, min(size, max(lo + 1, math.ceil(stop * size)))


def crop_small(small: np.ndarray, roi: Sequence[float] | None) -> np.ndarray:
    """Thumbnails of the content area only (resized back to SMALL × SMALL)."""
    if roi is None:
        return small
    x0, y0, x1, y1 = roi
    ys, ye = _slice(y0, y1, SMALL)
    xs, xe = _slice(x0, x1, SMALL)
    out = np.empty_like(small)
    for i in range(small.shape[0]):
        crop = Image.fromarray(small[i, ys:ye, xs:xe])
        out[i] = np.asarray(crop.resize((SMALL, SMALL), Image.Resampling.BILINEAR), dtype=np.uint8)
    return out


def perceptual_hash(small: np.ndarray) -> np.ndarray:
    """63-bit pHash (uint64) of thumbnails (n, SMALL, SMALL) or one thumbnail (SMALL, SMALL)."""
    single = small.ndim == 2
    s = (small[None] if single else small).astype(np.float32)
    n = s.shape[0]
    s = s.reshape(n, 32, 2, 32, 2).mean(axis=(2, 4))
    coef = np.einsum("ij,njk,lk->nil", _DCT, s, _DCT)
    block = coef[:, :8, :8].reshape(n, 64)[:, 1:]  # without the DC term: brightness is no signal
    bits = block > np.median(block, axis=1, keepdims=True)
    out = (bits.astype(np.uint64) * _BIT_WEIGHTS).sum(axis=1, dtype=np.uint64)
    return out[0] if single else out


def ink_distance(a: np.ndarray, b: np.ndarray, axes: tuple[int, ...] = (-1,)) -> np.ndarray | float:
    """Normalized L1 distance of ink maps in [0, 1] (0: equal, 1: nothing in common); `axes`
    are the axes of one map (the last one for vectors, the last two for grids)."""
    num = np.abs(a - b).sum(axis=axes)
    den = a.sum(axis=axes) + b.sum(axis=axes)
    return num / np.maximum(den, EPS)


def coarse_of(fine_map: np.ndarray) -> np.ndarray:
    """The ink map of one picture as a COARSE_ROWS × COARSE_COLS grid (area mean)."""
    img = Image.fromarray(np.ascontiguousarray(fine_map, dtype=np.float32), mode="F")
    out = img.resize((COARSE_COLS, COARSE_ROWS), Image.Resampling.BOX)
    return np.asarray(out, dtype=np.float32)


def soften(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """Box blur of a 2-D map (edge cells repeated): what is compared with a slide must not
    break on a shift of a cell or two (the content area is found only roughly)."""
    k = 2 * radius + 1
    padded = np.pad(a.astype(np.float32), radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return windows.mean(axis=(-2, -1))


def crop_blocks(blocks: np.ndarray, roi: Sequence[float] | None) -> np.ndarray:
    """The part of an ink map (rows × cols) inside the content area."""
    if roi is None:
        return blocks
    x0, y0, x1, y1 = roi
    r0, r1 = _slice(y0, y1, blocks.shape[0])
    c0, c1 = _slice(x0, x1, blocks.shape[1])
    return blocks[r0:r1, c0:c1]


def roi_mask(shape: tuple[int, int], roi: Sequence[float] | None) -> np.ndarray:
    """Boolean mask of a rows × cols grid: the blocks inside the content area."""
    mask = np.zeros(shape, dtype=bool)
    if roi is None:
        mask[:] = True
        return mask
    x0, y0, x1, y1 = roi
    r0, r1 = _slice(y0, y1, shape[0])
    c0, c1 = _slice(x0, x1, shape[1])
    mask[r0:r1, c0:c1] = True
    return mask


def _padded_windows(arr: np.ndarray, size: int, *, before: int) -> np.ndarray:
    """Sliding windows of `size` frames along the time axis, (n, …, size): the window of frame
    i covers the frames i-before … i-before+size-1; the edges are extended by their frame."""
    pad = [(before, size - 1 - before)] + [(0, 0)] * (arr.ndim - 1)
    padded = np.pad(arr, pad, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, size, axis=0)


def trail_min(blocks: np.ndarray, size: int) -> np.ndarray:
    """Static ink: the minimum of the ink maps of the frames i-size+1 … i."""
    return _padded_windows(blocks, size, before=size - 1).min(axis=-1)


def centered_min(blocks: np.ndarray, size: int) -> np.ndarray:
    """The minimum of the ink maps around each frame (no lag): what stays put near it."""
    return _padded_windows(blocks, size, before=(size - 1) // 2).min(axis=-1)


def lead_max(blocks: np.ndarray, size: int) -> np.ndarray:
    """Ink seen at all: the maximum of the ink maps of the frames i … i+size-1."""
    return _padded_windows(blocks, size, before=0).max(axis=-1)


# ---------------------------------------------------------------- what counts


@dataclass
class Analysis:
    """Which blocks of the frame carry the content, and how much ink they hold in time."""

    mask: np.ndarray  # (rows, cols) bool: content blocks inside the content area
    roi: list[float] | None  # the content area in shares of the frame, None — the whole frame
    roi_source: str  # "auto" | "topic" | "none"
    static: np.ndarray  # (n,) density of the ink that stays put around each frame
    raw: np.ndarray  # (n,) density of all the ink, the moving one too
    template_blocks: int  # blocks that hold ink almost all the time (left out)
    content_blocks: int


def _best_run(
    profile: np.ndarray, empty: float = 0.08, gap: int | None = 3
) -> tuple[int, int] | None:
    """The part of a mass profile (per column or per row) between gaps: a gap is `gap` or more
    cells in a row whose mass is below `empty` of the maximum. Of the parts the one with the
    most mass; (first, last) inclusive. A profile without a gap is one part; with `gap=None`
    the profile is never split."""
    if profile.size == 0 or float(profile.max()) <= 0:
        return None
    hollow = profile < empty * float(profile.max())
    if gap is None:  # no splitting: from the first to the last cell with mass
        solid = np.flatnonzero(~hollow)
        return int(solid[0]), int(solid[-1])
    best: tuple[float, int, int] | None = None
    start: int | None = None
    quiet = 0  # hollow cells since the last cell with mass
    last = -1
    for i, is_hollow in enumerate([*hollow, *([True] * gap)]):  # the tail closes a part
        if not is_hollow:
            if start is None:
                start = i
            last, quiet = i, 0
            continue
        quiet += 1
        if start is not None and quiet >= gap:
            mass = float(profile[start : last + 1].sum())
            if best is None or mass > best[0]:
                best = (mass, start, last)
            start = None
    return None if best is None else (best[1], best[2])


def auto_roi(
    blocks: np.ndarray, content: np.ndarray, hold: int, *, margin: float = ROI_MARGIN
) -> list[float] | None:
    """The part of the frame where the content lives (slides next to a camera on the lecturer),
    or None when it is the whole frame or cannot be told.

    The mass of a block is the share of the time it holds static ink; the area is the part with
    the most mass between gaps (empty columns, then empty rows); a profile without a gap gives
    the whole axis. It must be a real part of the frame (at most
    ROI_MAX_AREA of it) that holds at least 75 % of the mass.
    """
    n, rows, cols = blocks.shape
    if n < TEMPLATE_MIN_FRAMES or not content.any():
        return None
    presence = (centered_min(blocks, hold) >= BG_LEVEL).mean(axis=0) * content
    total = float(presence.sum())
    if total <= 0:
        return None
    x_run = _best_run(presence.sum(axis=0))
    if x_run is None:
        return None
    # the camera stands beside the slides, not above them: rows are not split
    y_run = _best_run(presence[:, x_run[0] : x_run[1] + 1].sum(axis=1), gap=None)
    if y_run is None:
        return None
    inside = float(presence[y_run[0] : y_run[1] + 1, x_run[0] : x_run[1] + 1].sum())
    x0 = max(0.0, x_run[0] / cols - margin)
    x1 = min(1.0, (x_run[1] + 1) / cols + margin)
    y0 = max(0.0, y_run[0] / rows - margin)
    y1 = min(1.0, (y_run[1] + 1) / rows + margin)
    # an axis the content mostly fills is filled to the edge: the slide's header and footer
    # (template, no mass of their own) belong to the picture
    if x1 - x0 >= 0.6:
        x0, x1 = 0.0, 1.0
    if y1 - y0 >= 0.6:
        y0, y1 = 0.0, 1.0
    if (x1 - x0) * (y1 - y0) > ROI_MAX_AREA or inside < 0.75 * total:
        return None
    if x1 - x0 < 0.15 or y1 - y0 < 0.15:
        return None
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def analyze(sig: Signals, th: Thresholds) -> Analysis:
    """Template, content area and the ink series of a video (see the module docstring)."""
    blocks = sig.fine
    n = sig.n
    ever = blocks.max(axis=0) >= BG_LEVEL
    template = np.zeros_like(ever)
    if n >= TEMPLATE_MIN_FRAMES:
        template = ever & (np.percentile(blocks, 20, axis=0) >= BG_LEVEL)
    content = ever & ~template
    if ever.any() and content.sum() < 0.1 * ever.sum():
        template[:] = False  # one state all the time: there is nothing to subtract
        content = ever.copy()
    roi: list[float] | None
    if isinstance(th.roi, str):  # "auto"
        roi, source = auto_roi(blocks, content, th.hold), "auto"
    elif th.roi:
        roi, source = list(th.roi), "topic"
    else:
        roi, source = None, "none"
    if roi is None and source == "auto":
        source = "none"
    mask = content & roi_mask(blocks.shape[1:], roi)
    if not mask.any():  # a wrong area: fall back to everything that ever held ink
        mask = content if content.any() else np.ones_like(content)
    static = centered_min(blocks, th.hold)[:, mask].mean(axis=1)
    raw = blocks[:, mask].mean(axis=1)
    return Analysis(
        mask=mask,
        roi=roi,
        roi_source=source,
        static=static,
        raw=raw,
        template_blocks=int(template.sum()),
        content_blocks=int(mask.sum()),
    )


# ---------------------------------------------------------------- epochs


def find_boundaries(sig: Signals, th: Thresholds, an: Analysis) -> list[tuple[int, str, float]]:
    """Rows where a new state begins: [(row, reason, strength)].

    «switch»: the static ink of the `hold` frames before (present in every one of them) has
    gone — `jump_removed` of it or more is absent in every one of the `hold` frames after.
    Ink that only moved (the lecturer) or is covered for a moment is not removed. «erase»: the
    ink seen in the `hold` frames after is `erase_drop` below the peak of static ink since the
    last boundary (a board wiped away slowly, which no single step shows).
    """
    n, h = sig.n, th.hold
    if n < 2 * h:
        return []
    mask = an.mask
    tmin = trail_min(sig.fine, h)  # frames i-h+1 … i
    lmax = lead_max(sig.fine, h)  # frames i … i+h-1
    before = tmin[h - 1 : n - h][:, mask]  # for t = h … n-h: the frames t-h … t-1
    after = lmax[h : n - h + 1][:, mask]
    base = before.sum(axis=1)
    removed = np.maximum(before - after, 0.0).sum(axis=1) / np.maximum(base, EPS)
    removed[base / mask.sum() < th.ink_min] = 0.0  # nothing static to lose
    candidates: dict[int, float] = {}
    count = len(removed)
    for j in range(count):
        lo, hi = max(0, j - h), min(count, j + h + 1)
        if removed[j] >= th.jump_removed and removed[j] >= removed[lo:hi].max():
            candidates[h + j] = float(removed[j] / th.jump_removed)
    ink_static = tmin[:, mask].mean(axis=1)
    ink_seen = lmax[:, mask].mean(axis=1)
    found: list[tuple[int, str, float]] = []
    peak = 0.0
    last = -h  # the row where the current state began
    for t in range(h, n - h + 1):
        after_static = float(ink_static[min(t + h - 1, n - 1)])  # what stays in the new state
        # the frames before t must all belong to the state that ends here
        if t in candidates and t - h >= last:
            found.append((t, "switch", candidates[t]))
            peak, last = after_static, t
            continue
        peak = max(peak, float(ink_static[t - 1]))
        if peak >= th.ink_min * 4 and ink_seen[t] <= (1.0 - th.erase_drop) * peak:
            found.append((t, "erase", 2.0))
            peak, last = after_static, t
    merged: list[tuple[int, str, float]] = []
    for item in found:
        if merged and item[0] - merged[-1][0] < th.min_epoch:
            if item[2] > merged[-1][2]:
                merged[-1] = item
            continue
        merged.append(item)
    return [b for b in merged if th.min_epoch <= b[0] <= n - th.min_epoch]


@dataclass
class KeyFrame:
    index: int  # row in the signals
    t: float  # seconds in the video
    epoch: int  # 1-based
    level: int  # 100 — the final state of the epoch, 25/50/75 — intermediate
    ink: float  # static ink density at the frame
    occlusion: float
    kind: str = "board"  # slide | board | other
    ordinal: int = 0  # 1-based number among the frames in time order
    slide: dict[str, Any] | None = None  # {"source", "number", "cells", "corr", "hash"}
    duplicate_of: int | None = None  # ordinal of the earlier frame with the same content
    file: str | None = None  # path relative to the source folder: frames/key_…png


@dataclass
class Epoch:
    index: int  # 1-based
    start: int  # first signal row
    end: int  # one past the last signal row
    start_t: float
    end_t: float
    reason: str  # how it began: start | switch | erase
    kind: str = "board"  # slide | board | other (by its final frame)
    frames: list[KeyFrame] = field(default_factory=list)


def occlusion_of(small: np.ndarray, median_img: np.ndarray, delta: int) -> np.ndarray:
    """Share of pixels that differ from the median picture by more than `delta` levels."""
    diff = np.abs(small.astype(np.int16) - median_img.astype(np.int16)) > delta
    return diff.mean(axis=(-2, -1))


def pick_frame(
    ink: np.ndarray, static: np.ndarray, lo: int, hi: int, th: Thresholds, small: np.ndarray
) -> tuple[int, float]:
    """The key frame among the rows [lo, hi): the final state of the board, no lecturer.

    The final state is the plateau of the window — the rows whose static ink is within 10 %
    of the peak (writing still going on at the end of the window does not drag the choice
    back). The temporal median of the plateau is the picture without the moving lecturer;
    the real frame nearest to it (the least covered, then the one with the most ink, then the
    latest) is the key frame. Returns (row, occlusion).
    """
    peak = float(static[lo:hi].max())
    rows = np.flatnonzero(static[lo:hi] >= 0.9 * peak) + lo
    if len(rows) < 3:  # the final state is brand new: no median of it, take its latest frame
        pick = int(rows[-1])
        around = np.arange(max(lo, pick - 2), min(hi, pick + 3))
        median_img = np.median(small[around], axis=0).astype(np.uint8)
        return pick, float(occlusion_of(small[pick], median_img, th.occlusion_delta))
    median_img = np.median(small[rows], axis=0).astype(np.uint8)
    occ = occlusion_of(small[rows], median_img, th.occlusion_delta)
    near = rows[occ <= float(occ.min()) + 0.01]
    pick = int(near[np.lexsort((near, ink[near]))[-1]])  # max ink, ties → the latest
    return pick, float(occ[int(np.flatnonzero(rows == pick)[0])])


def segment_epochs(sig: Signals, th: Thresholds, an: Analysis) -> list[Epoch]:
    """Epochs with their key frames (kinds and slide links are set by `classify`)."""
    n = sig.n
    starts = [(0, "start")] + [(b[0], b[1]) for b in find_boundaries(sig, th, an)]
    epochs: list[Epoch] = []
    for k, (s, reason) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else n
        epochs.append(
            Epoch(
                index=k + 1,
                start=s,
                end=e,
                start_t=float(sig.times[s]),
                end_t=float(sig.times[e - 1]) + 1.0 / th.fps,
                reason=reason,
            )
        )
    small = crop_small(sig.small, an.roi)
    for ep in epochs:
        length = ep.end - ep.start
        trim = 1 if length >= 4 else 0  # the last frame may already be the next state
        last = max(ep.start + 1, ep.end - trim)
        row, occ = pick_frame(an.raw, an.static, max(ep.start, last - th.tail), last, th, small)
        final = KeyFrame(
            index=row,
            t=float(sig.times[row]),
            epoch=ep.index,
            level=100,
            ink=float(an.static[row]),
            occlusion=occ,
        )
        ep.frames.append(final)
        if final.ink < th.ink_min * 4 or length <= th.long_epoch * th.fps:
            continue
        for level in (25, 50, 75):  # the order of the writing in a long epoch
            target = final.ink * level / 100.0
            idx = next(
                (i for i in range(ep.start, row) if an.static[i : i + 3].min() >= target),
                None,
            )
            if idx is None or min(idx + 10, row) <= idx:
                continue
            r, o = pick_frame(an.raw, an.static, idx, min(idx + 10, row), th, small)
            if any(
                float(ink_distance(sig.fine[r][an.mask], sig.fine[f.index][an.mask]))
                < th.dedup_cells
                for f in ep.frames
            ):
                continue  # looks like a frame already taken
            ep.frames.append(
                KeyFrame(
                    index=r,
                    t=float(sig.times[r]),
                    epoch=ep.index,
                    level=level,
                    ink=float(an.static[r]),
                    occlusion=o,
                )
            )
        ep.frames.sort(key=lambda f: f.index)
    return epochs


# ---------------------------------------------------------------- slides


@dataclass
class SlideImage:
    """A slide of a presentation of the topic as a gray picture of the analysis width."""

    source: str  # S1
    number: int
    gray: np.ndarray  # (height, width) uint8


@dataclass
class SlideRef:
    """A slide as it would look in the content area of the frame: what a frame is compared to."""

    source: str
    number: int
    small: np.ndarray  # (SMALL, SMALL) float32 thumbnail, softened
    coarse: np.ndarray  # (COARSE_ROWS, COARSE_COLS) float32 ink map, softened
    hash: int  # of the sharp thumbnail


def _gray_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    g = img.convert("L")
    if g.size != size:
        g = g.resize(
            size, Image.Resampling.BOX if g.size[0] > size[0] else Image.Resampling.BICUBIC
        )
    return np.asarray(g, dtype=np.uint8)


def load_slide_images(topic_dir: Path, slides: Sequence[Any]) -> tuple[list[SlideImage], list[str]]:
    """The slides of the topic as pictures: PDF pages rendered here, PPTX pages as far as
    `extracted/<ID>/pages/` has them. `slides` are the source records of kind slides."""
    images: list[SlideImage] = []
    notes: list[str] = []

    def keep(source: str, number: int, img: Image.Image) -> None:
        w = ANALYSIS_WIDTH
        h = max(2, round(w * img.height / max(img.width, 1)))
        images.append(SlideImage(source, number, _gray_array(img, (w, h))))

    for rec in slides:
        path = topic_dir / rec.file if getattr(rec, "file", None) else None
        if path is None or not path.is_file():
            notes.append(f"{rec.id}: файла презентации нет — слайды не сравниваются с кадрами")
            continue
        before = len(images)
        if path.suffix.lower() == ".pdf":
            try:
                import pymupdf

                with pymupdf.open(str(path)) as doc:
                    for number, page in enumerate(doc, start=1):
                        zoom = ANALYSIS_WIDTH / max(page.rect.width, 1.0)
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(zoom, zoom),
                            colorspace=pymupdf.csGRAY,
                            alpha=False,
                        )
                        keep(
                            rec.id,
                            number,
                            Image.frombytes("L", (pix.width, pix.height), pix.samples),
                        )
            except Exception as exc:
                notes.append(f"{rec.id}: PDF слайдов не прочитан ({exc})")
                del images[before:]
        else:
            pages = topic_dir / "extracted" / rec.id / "pages"
            for png in sorted(pages.glob("p[0-9][0-9][0-9][0-9].png")):
                try:
                    with Image.open(png) as im:
                        keep(rec.id, int(png.stem[1:5]), im.copy())
                except (OSError, ValueError):
                    continue
            if len(images) > before:
                notes.append(
                    f"{rec.id}: презентация не в PDF — с кадрами сравниваются только "
                    f"отрисованные страницы ({len(images) - before})"
                )
    return images, notes


def make_refs(
    images: Sequence[SlideImage], size: tuple[int, int], th: Thresholds
) -> list[SlideRef]:
    """References of the slides at `size` (pixels of the analysis frame): the content area of
    the frame, so that the ink of a slide is measured at the scale the frame has it."""
    refs: list[SlideRef] = []
    for img in images:
        gray = _gray_array(Image.fromarray(img.gray), size)
        small, blocks = frame_features(gray, th.edge)
        refs.append(
            SlideRef(
                img.source,
                img.number,
                soften(small),
                soften(coarse_of(blocks)),
                int(perceptual_hash(small)),
            )
        )
    return refs


def match_slide(
    small: np.ndarray, coarse: np.ndarray, refs: Sequence[SlideRef], th: Thresholds
) -> dict[str, Any] | None:
    """The slide this picture is (thumbnail `small` and ink map `coarse` of the content area of
    the frame), None if no slide fits.

    A slide fits when its ink map is within `match_cells`, the thumbnails correlate at least
    `match_corr` and the hashes differ by at most `match_hash` bits; of several the closest
    wins.
    """
    if not refs:
        return None
    q = soften(small).ravel()
    q = (q - q.mean()) / (q.std() + EPS)
    ref_small = np.stack([r.small.ravel() for r in refs])
    ref_small = (ref_small - ref_small.mean(axis=1, keepdims=True)) / (
        ref_small.std(axis=1, keepdims=True) + EPS
    )
    corr = ref_small @ q / q.size
    cd = np.asarray(
        ink_distance(np.stack([r.coarse for r in refs]), soften(coarse)[None], axes=(-2, -1))
    )
    h = perceptual_hash(small)
    hd = popcount(np.array([r.hash for r in refs], dtype=np.uint64) ^ h)
    ok = (cd <= th.match_cells) & (corr >= th.match_corr) & (hd <= th.match_hash)
    if not ok.any():
        return None
    score = np.where(ok, cd + (1.0 - corr), np.inf)
    i = int(np.argmin(score))
    return {
        "source": refs[i].source,
        "number": refs[i].number,
        "cells": round(float(cd[i]), 3),
        "corr": round(float(corr[i]), 3),
        "hash": int(hd[i]),
    }


def classify(
    sig: Signals, epochs: list[Epoch], refs: Sequence[SlideRef], th: Thresholds, an: Analysis
) -> list[KeyFrame]:
    """Kinds, slide links and duplicates of the frames; returns them in time order with
    1-based `ordinal`s."""
    small = crop_small(sig.small, an.roi)
    frames: list[KeyFrame] = []
    maps: dict[int, np.ndarray] = {}  # row → the content ink of the picture (lecturer removed)
    for ep in epochs:
        for kf in ep.frames:
            lo = max(ep.start, kf.index - th.tail + 1)
            pic = np.median(small[lo : kf.index + 1], axis=0).astype(np.uint8)
            ink_map = np.median(sig.fine[lo : kf.index + 1], axis=0).astype(np.float32)
            maps[kf.index] = ink_map[an.mask]
            if an.static[kf.index] < th.ink_min:
                kf.kind = "other"
                continue
            match = match_slide(pic, coarse_of(crop_blocks(ink_map, an.roi)), refs, th)
            if match is not None:
                kf.kind, kf.slide = "slide", match
            else:
                kf.kind = "board"
        final = next((f for f in ep.frames if f.level == 100), ep.frames[-1])
        ep.kind = final.kind
        if ep.kind == "slide":  # a slide does not accumulate: only its final frame is of use
            ep.frames = [final]
        frames += ep.frames
    frames.sort(key=lambda f: f.index)
    kept: list[KeyFrame] = []
    for kf in frames:
        if kf.kind == "board":
            for earlier in kept:
                if earlier.kind != "board" or earlier.epoch == kf.epoch:
                    continue
                if float(ink_distance(maps[kf.index], maps[earlier.index])) <= th.dedup_cells:
                    kf.duplicate_of = earlier.duplicate_of or earlier.ordinal
                    break
        kf.ordinal = len(kept) + 1
        kept.append(kf)
    return kept


@dataclass
class Selection:
    """The result of the whole selection: what `video.py` works with."""

    analysis: Analysis
    epochs: list[Epoch]
    frames: list[KeyFrame]


def select_frames(sig: Signals, th: Thresholds, slides: Sequence[SlideImage] = ()) -> Selection:
    """Epochs and key frames of a video from its signals; `slides` are the pictures of the
    slides of the topic (a frame that is one of them becomes a link to it)."""
    an = analyze(sig, th)
    epochs = segment_epochs(sig, th, an)
    refs: list[SlideRef] = []
    if slides:
        x0, y0, x1, y1 = an.roi or (0.0, 0.0, 1.0, 1.0)
        size = (max(8, round((x1 - x0) * sig.width)), max(8, round((y1 - y0) * sig.height)))
        refs = make_refs(slides, size, th)
    frames = classify(sig, epochs, refs, th, an)
    return Selection(an, epochs, frames)


# ---------------------------------------------------------------- pictures


def extract_png(
    ffmpeg: Path | str,
    video: Path,
    t: float,
    target: Path,
    *,
    max_side: int = 1600,
    crop: Sequence[float] | None = None,
) -> None:
    """The video frame at second `t` as a full-colour PNG, long side ≤ `max_side`; `crop` is a
    part of the frame [x0, y0, x1, y1] in shares (the content area)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".part.png")
    filters = []
    if crop is not None:
        x0, y0, x1, y1 = crop
        # even sizes: some codecs and filters dislike odd ones
        filters.append(
            f"crop=trunc(iw*{x1 - x0:.4f}/2)*2:trunc(ih*{y1 - y0:.4f}/2)*2:"
            f"trunc(iw*{x0:.4f}/2)*2:trunc(ih*{y0:.4f}/2)*2"
        )
    filters.append(f"scale='min({max_side},iw)':-2")
    argv = [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{max(t, 0.0):.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        ",".join(filters),
        str(tmp),
    ]
    res = procutil.run(argv, timeout=300)
    if not res.ok or not tmp.is_file():
        raise ExtractError(f"ffmpeg не смог сохранить кадр на {t:.1f} с: {_ffmpeg_failure(res)}")
    tmp.replace(target)


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}"


def frame_name(ordinal: int, t: float) -> str:
    return f"key_{ordinal:03d}_{hhmmss(t)}.png"


_FRAME_FILE_RE = re.compile(r"^key_\d{3}_\d{6}\.png$")


def clean_frames_dir(frames_dir: Path, keep: Sequence[str]) -> None:
    """Remove key frames of an earlier selection that are not in `keep`."""
    wanted = set(keep)
    if not frames_dir.is_dir():
        return
    for path in frames_dir.iterdir():
        if _FRAME_FILE_RE.match(path.name) and path.name not in wanted:
            with contextlib.suppress(OSError):
                path.unlink()


def make_work_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".frames-", dir=parent))


def remove_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- frames.json


def frames_json(
    selection: Selection, th: Thresholds, *, duration: float, size: tuple[int, int]
) -> dict[str, Any]:
    an = selection.analysis
    return {
        "version": SIGNALS_VERSION,
        "duration": round(duration, 2),
        "analysis": {
            "width": size[0],
            "height": size[1],
            "fps": th.fps,
            "roi": an.roi,
            "roi_source": an.roi_source,
            "template_blocks": an.template_blocks,
            "content_blocks": an.content_blocks,
        },
        "thresholds": th.to_dict(),
        "epochs": [
            {
                "index": e.index,
                "start": round(e.start_t, 2),
                "end": round(e.end_t, 2),
                "reason": e.reason,
                "type": e.kind,
                "frames": [f.ordinal for f in e.frames],
            }
            for e in selection.epochs
        ],
        "frames": [
            {
                "ordinal": f.ordinal,
                "t": round(f.t, 2),
                "epoch": f.epoch,
                "level": f.level,
                "type": f.kind,
                "file": f.file,
                "ink": round(f.ink, 4),
                "occlusion": round(f.occlusion, 3),
                "matched_slide": f.slide,
                "duplicate_of": f.duplicate_of,
            }
            for f in selection.frames
        ],
    }


def write_frames_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    tmp.replace(path)
