"""Small image helpers shared by the pipeline, the server and the export."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= width:
        return image
    height = max(1, round(h * width / w))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def save_jpeg(image: np.ndarray, path: Path, quality: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, "JPEG", quality=quality)


@lru_cache(maxsize=64)
def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def draw_boxes(image: np.ndarray, boxes: list[tuple[float, float, float, float]], colour=(255, 59, 48)) -> np.ndarray:
    """A copy of the image with normalised (0-1) boxes drawn on it."""
    from PIL import ImageDraw  # noqa: PLC0415

    out = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    width = max(2, round(w / 320))
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=width)
    return np.asarray(out)


def crop_box(image: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad: float = 0.1) -> np.ndarray:
    """Crop a normalised (0-1) box, padded by `pad` of its size, clamped to the image."""
    h, w = image.shape[:2]
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    left = int(max(0, np.floor(x1 * w - pad * bw)))
    top = int(max(0, np.floor(y1 * h - pad * bh)))
    right = int(min(w, np.ceil(x2 * w + pad * bw)))
    bottom = int(min(h, np.ceil(y2 * h + pad * bh)))
    if right <= left or bottom <= top:
        return image[0:1, 0:1]
    return image[top:bottom, left:right]
