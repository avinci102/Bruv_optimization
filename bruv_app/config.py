"""Settings and defaults. Saved as JSON inside each project folder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_FPS = 4.0  # sampled frames per second; trials run at 3-4 fps


@dataclass
class Settings:
    # Frame sampling. None means DEFAULT_FPS. At 3-4 fps fish move little between frames,
    # which keeps tracking reliable, and MaxN comes closer to a frame-by-frame count.
    sample_fps: float | None = None
    # Width frames are decoded at (detection runs on these).
    decode_width: int = 1280
    # Width of the JPEGs stored for the review UI (~150 MB per 30 min at 960 px).
    frame_width: int = 960
    jpeg_quality: int = 85
    # Ask ffmpeg for hardware decode; it falls back to software on its own.
    hwaccel: bool = True

    # Analysis window on the combined timeline, in seconds (e.g. skip the descent).
    start_s: float = 0.0
    duration_s: float | None = None
    # clip_in/clip_out: "file" uses the GoPro file name, "index" uses clip_1, clip_2, ...
    # "auto" uses index when the template's clip_in dropdown lists clip_1, clip_2, ...
    clip_names: str = "auto"

    # Detection. "rfdetr" = Community Fish Detector (RF-DETR), "color" = synthetic-video fake.
    detector: str = "rfdetr"
    detector_weights: str | None = None
    det_threshold: float = 0.3
    batch_size: int = 4

    # Tracking (lightweight IoU + centre distance, Hungarian matching).
    track_min_iou: float = 0.05
    track_max_dist: float = 1.5  # centre distance / larger box diagonal
    track_max_size_ratio: float = 3.0
    track_max_missed_s: float = 1.0  # how long a fish may go unseen and keep its track
    track_min_appearance: float = 0.5  # colour-histogram cosine; blocks obvious swaps
    track_direction: bool = True  # expect a fish to keep its course (keeps crossing fish apart)

    # Continuity merge (review dedupe only; never used for counts).
    gap_seconds: float = 10.0
    # BioCLIP similarity of the two pieces. Most fish score 0.8 or more against each other,
    # so the bar is above that; with the colour check below it blocked 65% of joins between
    # different fish and kept 91% between parts of the same fish (docs/SPEC.md §12).
    merge_min_similarity: float = 0.85
    merge_max_size_ratio: float = 3.0
    # The two pieces' average colour (cosine of their colour histograms; 1 = identical).
    merge_min_colour: float = 0.7
    # Tiny fish: smaller than this at their largest (the box's longer side, in pixels of the
    # saved review frames). Too small to identify, they join no other piece and the review
    # page hides them until asked.
    tiny_fish_px: float = 40.0

    # Classification. "bioclip" = BioCLIP 2 zero-shot, "histogram" = no-guess fallback.
    classifier: str = "bioclip"
    crops_per_tracklet: int = 3
    crops_per_fish: int = 5
    top_k: int = 3
    # Crops are padded by this fraction of the box size before embedding.
    crop_pad: float = 0.1

    # Device: "auto", "cpu" or "cuda".
    device: str = "auto"

    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Settings":
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if gpu_available() else "cpu"

    def resolved_fps(self) -> float:
        return float(self.sample_fps) if self.sample_fps else DEFAULT_FPS


def gpu_available() -> bool:
    try:
        import torch  # noqa: PLC0415 — optional heavy dependency
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
