"""Fish detection behind a small interface, so the backend can be swapped (e.g. ONNX/OpenVINO).

Boxes are normalised to 0-1 of frame width/height, so detections don't depend on the
decode or storage resolution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    # Cheap colour-histogram vector used by the tracker to avoid identity swaps.
    app: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


class Detector(Protocol):
    name: str

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """RGB uint8 frames in, one list of detections per frame out."""
        ...


def resolution_from_name(filename: str) -> int | None:
    """CFD release names carry their training resolution, e.g. cfd-rf-detr-nano-640-...pth."""
    m = re.search(r"rf-detr-[a-z]+-(\d{3,4})(?=[-_.])", filename.lower())
    return int(m.group(1)) if m else None


class RFDETRDetector:
    """Community Fish Detector, RF-DETR release (Apache-2.0 inference code; the weights'
    own license isn't stated by the CFD repo — see docs/SOURCES.md).

    Weights: github.com/filippovarini/community-fish-detector/releases — Nano (640) for CPU,
    Small/Medium (1024) when a GPU is present.
    """

    name = "cfd-rfdetr"

    def __init__(self, weights: Path, threshold: float = 0.3, device: str = "cpu"):
        try:
            import rfdetr  # noqa: PLC0415 — optional heavy dependency
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "RF-DETR is not installed. Run: pip install -r requirements-ml.txt"
            ) from exc
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {weights}")
        self.threshold = threshold
        try:  # pragma: no cover - needs weights
            self.model = rfdetr.from_checkpoint(str(weights), device=device)
            # Older checkpoints without a saved model_config silently fall back to the
            # class default (Nano 384, Small 512); force the resolution CFD trained at.
            expected = resolution_from_name(weights.name)
            if expected and self.model.model_config.resolution != expected:
                log.warning(
                    "%s loaded at %s px; reloading at %s px",
                    weights.name, self.model.model_config.resolution, expected,
                )
                self.model = rfdetr.from_checkpoint(str(weights), device=device, resolution=expected)
        except RuntimeError as exc:  # pragma: no cover - needs weights
            if "trust_checkpoint" in str(exc):
                raise RuntimeError(
                    f"{weights.name} can't be loaded safely. Only use checkpoints downloaded "
                    "from the Community Fish Detector releases page."
                ) from exc
            raise
        self.name = f"cfd-rfdetr:{weights.name}@{self.model.model_config.resolution}"

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:  # pragma: no cover
        # Equivalent to CFD's rf_detr_batch_inference.py (CFD passes PIL images; uint8 RGB
        # arrays take the same preprocessing path). Boxes come back in input-image pixels.
        preds = self.model.predict(list(images), threshold=self.threshold, include_source_image=False)
        if not isinstance(preds, list):
            preds = [preds]
        out: list[list[Detection]] = []
        for image, pred in zip(images, preds):
            h, w = image.shape[:2]
            dets = [
                Detection(
                    float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h, float(score)
                )
                for (x1, y1, x2, y2), score in zip(pred.xyxy, pred.confidence)
            ]
            out.append(dets)
        return out


class ColorBlobDetector:
    """Fake detector for the synthetic test video: anything far from the water colour is a fish."""

    name = "color-blob"

    def __init__(self, min_area_frac: float = 0.0005, threshold: float = 60.0):
        self.min_area_frac = min_area_frac
        self.threshold = threshold

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        from scipy import ndimage  # noqa: PLC0415

        out = []
        for image in images:
            h, w = image.shape[:2]
            img = image.astype(np.int16)
            water = np.median(img.reshape(-1, 3), axis=0)
            mask = np.abs(img - water).sum(axis=2) > self.threshold
            labels, n = ndimage.label(mask)
            dets = []
            for i, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is None:
                    continue
                area = int((labels[sl] == i).sum())
                if area < self.min_area_frac * h * w:
                    continue
                ys, xs = sl
                dets.append(Detection(xs.start / w, ys.start / h, xs.stop / w, ys.stop / h, 0.9))
            out.append(dets)
        return out


class ScriptedDetector:
    """Fake detector driven by a function of the frame index (for unit tests)."""

    name = "scripted"

    def __init__(self, script: Callable[[int], list[Detection]]):
        self.script = script
        self.calls = 0

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        out = []
        for _ in images:
            out.append(self.script(self.calls))
            self.calls += 1
        return out


def make_detector(settings, device: str) -> Detector:
    if settings.detector == "color":
        return ColorBlobDetector()
    if settings.detector == "rfdetr":
        if not settings.detector_weights:
            raise RuntimeError(
                "No detector weights set. Download a Community Fish Detector RF-DETR .pth "
                "(Nano for CPU) and pass --weights."
            )
        return RFDETRDetector(Path(settings.detector_weights), settings.det_threshold, device)
    raise ValueError(f"Unknown detector: {settings.detector}")
