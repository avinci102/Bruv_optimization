"""Still objects: boxes that look like the scene's background.

A BRUV's bait arm, canister and float line sit in the same place for the whole drop, and
the detector boxes them as fish again and again. Each box is compared with the background,
the median of frames from the minute either side of it. A box is still when almost none
of its pixels differ from that background. A fish differs from it even while it swims in
front of the arm, so it stays. Still boxes are kept in the project but not tracked, so they
make no cards and never join a fish's card. The review page shows them for checking: a fish
that sits motionless for over a minute would look still too.

Measured on the first 185 answers of the 44-minute NP339 drop (docs/SPEC.md §13). Boxes at
least 6% of the frame wide with under 10% of their pixels changed made up 91% of the
sightings answered "not a fish", and none of the 1,689 sightings answered as fish. A
typical fish box had 47% of its pixels changed, the arm 1%. Narrower boxes are never called
still: at a quarter of the saved frame's size there are too few pixels to tell.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

SCALE = 4  # frames are compared at a quarter of the saved size (240 x 135 for 960 x 540)
PIXEL_CHANGE = 25  # a pixel differs when a colour channel moves by more than this (0-255)
BLOCK_S = 20.0  # one background for every 20 s of video...
EVERY_S = 2.0  # ...from frames 2 s apart


def small_frame(path, size: tuple[int, int]) -> np.ndarray:
    """A saved frame at `size`, lightly blurred so a pixel's noise isn't counted as change.
    JPEG draft mode decodes straight to a quarter size, which is fast."""
    with Image.open(path) as im:
        im.draft("RGB", size)
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size)
        return np.asarray(im.filter(ImageFilter.BoxBlur(1)), dtype=np.uint8)


def changed_share(diff: np.ndarray, box, size: tuple[int, int]) -> float:
    """The share of a box's pixels that differ from the background (diff: per-pixel change)."""
    w, h = size
    x1, x2 = int(box[0] * w), int(np.ceil(box[2] * w))
    y1, y2 = int(box[1] * h), int(np.ceil(box[3] * h))
    patch = diff[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
    return float((patch > PIXEL_CHANGE).mean()) if patch.size else 1.0


def background_change(
    frames: list[tuple[int, float, str]],
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    size: tuple[int, int],
    window_s: float = 60.0,
    load: Callable[[str, tuple[int, int]], np.ndarray] = small_frame,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, float]:
    """Detection id -> the share of its box that differs from the background.

    frames: (frame id, time, image path) in time order. boxes: frame id -> [(detection id,
    (x1, y1, x2, y2) as fractions of the frame)]. The background for each 20 s block is the
    median of frames 2 s apart from `window_s` before the block's middle to `window_s`
    after. Only those frames are held in memory, so a long video needs no more than a short
    one.
    """
    times = [t for _, t, _ in frames]
    picks: list[int] = []  # every frame about EVERY_S apart, for backgrounds
    for i, t in enumerate(times):
        if not picks or t - times[picks[-1]] >= EVERY_S - 1e-6:
            picks.append(i)
    held: deque[tuple[float, np.ndarray]] = deque()
    nxt = 0  # next background frame to load
    out: dict[int, float] = {}
    block, background = None, None
    for n, (frame_id, t, path) in enumerate(frames):
        if frame_id not in boxes:
            continue
        b = int(t // BLOCK_S)
        if b != block:
            block = b
            middle = (b + 0.5) * BLOCK_S
            while nxt < len(picks) and times[picks[nxt]] <= middle + window_s:
                i = picks[nxt]
                held.append((times[i], load(frames[i][2], size)))
                nxt += 1
            while held and held[0][0] < middle - window_s:
                held.popleft()
            background = np.median(np.stack([im for _, im in held]), axis=0).astype(np.int16) if held else None
        image = load(path, size).astype(np.int16)
        diff = np.abs(image - background).max(axis=2) if background is not None else None
        for det_id, box in boxes[frame_id]:
            out[det_id] = 1.0 if diff is None else changed_share(diff, box, size)
        if progress and n % 200 == 0:
            progress(n, len(frames))
    if progress:
        progress(len(frames), len(frames))
    return out


def is_still(box, change: float | None, min_width: float, max_change: float) -> bool:
    """Still: wide enough to judge, and hardly any of it differs from the background."""
    return change is not None and (box[2] - box[0]) >= min_width and change < max_change
