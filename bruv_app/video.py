"""GoPro chapter ordering, the combined timeline, and the frame sampler."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# imageio-ffmpeg warns every time we ask ffmpeg to scale; that's intended here.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

# HERO6+ : GX010123.MP4 / GH010123.MP4 (2-char prefix, 2-digit chapter, 4-digit file number)
_GOPRO_NEW = re.compile(r"^G[HXL](\d{2})(\d{4})$", re.IGNORECASE)
# HERO5 and older: GOPR0123.MP4 is chapter 1, GP010123.MP4 is chapter 2, ...
_GOPRO_OLD_FIRST = re.compile(r"^GOPR(\d{4})$", re.IGNORECASE)
_GOPRO_OLD_NEXT = re.compile(r"^GP(\d{2})(\d{4})$", re.IGNORECASE)


def gopro_key(path: Path) -> tuple:
    """Sort key: (file number, chapter). Non-GoPro names sort after, by name."""
    stem = path.stem
    if m := _GOPRO_NEW.match(stem):
        return (0, int(m.group(2)), int(m.group(1)), stem)
    if m := _GOPRO_OLD_FIRST.match(stem):
        return (0, int(m.group(1)), 0, stem)
    if m := _GOPRO_OLD_NEXT.match(stem):
        return (0, int(m.group(2)), int(m.group(1)), stem)
    return (1, 0, 0, stem.lower())


def order_chapters(paths: list[Path]) -> list[Path]:
    return sorted((Path(p) for p in paths), key=gopro_key)


@dataclass
class Chapter:
    index: int
    path: Path
    name: str  # what goes into clip_in / clip_out
    start_s: float  # offset on the combined timeline
    duration_s: float
    fps: float


@dataclass
class Timeline:
    chapters: list[Chapter]

    @property
    def duration_s(self) -> float:
        if not self.chapters:
            return 0.0
        last = self.chapters[-1]
        return last.start_s + last.duration_s

    def locate(self, t: float) -> tuple[str, float]:
        """Global time -> (clip name, time within that clip)."""
        for ch in reversed(self.chapters):
            if t >= ch.start_s - 1e-6:
                return ch.name, max(0.0, t - ch.start_s)
        first = self.chapters[0]
        return first.name, 0.0


def probe(path: Path) -> tuple[float, float]:
    """Return (duration seconds, fps) from the ffmpeg header."""
    import imageio_ffmpeg  # noqa: PLC0415

    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(gen)
    finally:
        gen.close()
    duration = float(meta.get("duration") or 0.0)
    fps = float(meta.get("fps") or 0.0)
    if duration <= 0:
        _, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    return duration, fps


def shrink_video(
    src: Path,
    dst: Path,
    max_mb: float = 38.0,
    width: int = 1280,
    fps: float = 10.0,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> Path:
    """Re-encode a clip small enough to share: H.264 at a two-pass bitrate that fits `max_mb`.

    The pipeline decodes at 1280 px and samples 1-2 fps, so 1280 px at 10 fps keeps
    everything it uses. Audio is dropped. Never upscales.
    """
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import imageio_ffmpeg  # noqa: PLC0415

    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("The shrunk copy must not overwrite the original")
    total, _ = probe(src)
    length = max(0.0, total - start_s)
    if duration_s is not None:
        length = min(length, duration_s)
    if length <= 0:
        raise ValueError("Nothing to encode: start is past the end of the clip")
    # 8% headroom for the MP4 container and rate-control overshoot.
    kbps = int(max_mb * 1024 * 1024 * 8 * 0.92 / length / 1000)
    if kbps < 200:
        raise ValueError(f"{max_mb} MB is too small for {length:.0f} s; use --duration to cut it shorter")
    kbps = min(kbps, 8000)

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    common = [
        "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{length:.3f}",
        "-vf", f"fps={fps},scale='min({width},iw)':-2",
        "-an", "-c:v", "libx264", "-preset", "medium", "-b:v", f"{kbps}k",
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        log = str(Path(tmp) / "pass")
        for n, out in ((1, ["-f", "null", "-"]), (2, ["-movflags", "+faststart", str(dst)])):
            cmd = [exe, "-y", "-loglevel", "error", *common, "-pass", str(n), "-passlogfile", log, *out]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return dst


def build_timeline(paths: list[Path], clip_names: str = "file", probe_fn=probe) -> Timeline:
    chapters: list[Chapter] = []
    offset = 0.0
    for i, path in enumerate(order_chapters(paths)):
        duration, fps = probe_fn(path)
        name = path.stem if clip_names == "file" else f"clip_{i + 1}"
        chapters.append(Chapter(i, path, name, offset, duration, fps))
        offset += duration
    return Timeline(chapters)


@dataclass
class SampledFrame:
    t: float  # global time
    chapter: int
    local_t: float
    image: np.ndarray  # RGB uint8, H x W x 3


def sample_frames(
    timeline: Timeline,
    fps: float,
    width: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    hwaccel: bool = True,
) -> Iterator[SampledFrame]:
    """Decode frames at `fps` across all chapters with an ffmpeg pipe."""
    import imageio_ffmpeg  # noqa: PLC0415

    end_s = timeline.duration_s if end_s is None else min(end_s, timeline.duration_s)
    step = 1.0 / fps
    for ch in timeline.chapters:
        ch_end = ch.start_s + ch.duration_s
        if ch_end <= start_s or ch.start_s >= end_s:
            continue
        local_start = max(0.0, start_s - ch.start_s)
        local_len = min(ch_end, end_s) - (ch.start_s + local_start)
        if local_len <= 0:
            continue
        input_params = ["-ss", f"{local_start:.3f}"]
        if hwaccel:
            input_params = ["-hwaccel", "auto", *input_params]
        # round=up makes output frame k the source frame at exactly local_start + k/fps;
        # the default (round=near) hands back the frame ~half a step later.
        output_params = ["-vf", f"fps={fps}:round=up,scale={width}:-2", "-t", f"{local_len:.3f}"]
        gen = imageio_ffmpeg.read_frames(
            str(ch.path), input_params=input_params, output_params=output_params
        )
        try:
            meta = next(gen)
            w, h = meta["size"]
            for k, raw in enumerate(gen):
                local_t = local_start + k * step
                if local_t >= local_start + local_len - 1e-6:
                    break
                image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                yield SampledFrame(ch.start_s + local_t, ch.index, local_t, image)
        finally:
            gen.close()
