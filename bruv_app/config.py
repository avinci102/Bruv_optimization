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
    # Ask ffmpeg for hardware decode; a chapter it fails on is decoded again in software.
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
    # The operating point: only boxes scoring this much are tracked and become cards (0.3
    # comes from the CFD README's example, not a BRUV study; see docs/SOURCES.md).
    det_threshold: float = 0.3
    # Boxes down to this score are kept too (not tracked), so other thresholds can be tried
    # on the same detections without running the detector again.
    det_floor: float = 0.1
    batch_size: int = 4

    # Still objects (still.py): boxes that hardly differ from the scene's background (the
    # bait arm, canister, float line, rocks) are kept but not tracked, so they make no cards
    # and never join a fish's. Only boxes at least still_min_width of the frame wide are
    # judged; still means under still_max_change of the box's pixels differ from the median
    # of the frames within still_window_s either side. Measured on the NP339 answers.
    still_filter: bool = True
    still_min_width: float = 0.06
    still_max_change: float = 0.10
    still_window_s: float = 60.0

    # Tracking (lightweight IoU + centre distance, Hungarian matching).
    track_min_iou: float = 0.05
    track_max_dist: float = 1.5  # centre distance / larger box diagonal
    track_max_size_ratio: float = 3.0
    # How long a fish may go unseen and keep its track: 0.25 s at 4 fps lets it miss one
    # frame; 0 links consecutive frames only. Short tracks rarely jump to another fish; the
    # merge and the review page's look-alikes bring the pieces of one fish back together.
    track_max_missed_s: float = 0.25
    track_min_appearance: float = 0.5  # colour-histogram cosine; blocks obvious swaps
    track_direction: bool = True  # expect a fish to keep its course (keeps crossing fish apart)
    # Don't link a box to a track when another option was better (another track for the
    # box, or another box for the track): where two fish are that close, a track jumps from
    # one to the other. On the NP339 fixes this avoided 33 of 49 such jumps, against 14 for
    # the shorter gap alone; both pieces start again and are judged by the merge and you.
    track_refuse_contested: bool = True
    # Options to measure on followed fish before they become defaults (docs/SPEC.md §15):
    # boxes between det_floor and det_threshold may carry on a track that found no box in a
    # frame (never start one), so a fish that fades for a moment keeps its card...
    track_faint: bool = False
    # ...and each link keeps this share of the track's earlier velocity (0: the last step
    # alone), so one wobble doesn't turn the course the tracker expects.
    track_velocity_smoothing: float = 0.0

    # Continuity merge (review dedupe only; never used for counts).
    gap_seconds: float = 10.0
    # Record rows in the export. "card": each answered card is its own row, one fish while it
    # stays on screen (docs/SPEC.md §15; Vinci, 26 Oct 2026). "grouped": same-species fish
    # within sighting_gap_s share a row, counted at its busiest frame (the rows before then).
    record_rows: str = "card"
    # For "grouped" rows. Kept apart from gap_seconds, so tuning the merge never changes how
    # rows are grouped.
    sighting_gap_s: float = 10.0
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
    # Run the species model on tiny fish too. Off: it's the slow step, and their guesses
    # were blobs named after whatever species looked closest.
    guess_tiny_fish: bool = False

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
        # A settings file from before a setting existed keeps the behaviour it was made with,
        # so an older project re-run reproduces its cards; new projects get the new default.
        return cls(**{**SETTING_ADDED_OFF, **{k: v for k, v in data.items() if k in known}})

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if gpu_available() else "cpu"

    def resolved_fps(self) -> float:
        return float(self.sample_fps) if self.sample_fps else DEFAULT_FPS


# Settings added later, as older settings files had them: off.
SETTING_ADDED_OFF = {"still_filter": False, "track_refuse_contested": False}


def gpu_available() -> bool:
    try:
        import torch  # noqa: PLC0415 — optional heavy dependency
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
