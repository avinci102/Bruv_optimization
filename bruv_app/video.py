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
log = logging.getLogger(__name__)

# HERO6+ : GX010123.MP4 (HEVC) / GH010123.MP4 (H.264): 2-digit chapter, 4-digit file number.
# GL010123.LRV is the low-res proxy of the same recording, so it is deliberately not matched.
_GOPRO_NEW = re.compile(r"^G[HX](\d{2})(\d{4})$", re.IGNORECASE)
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
        "-vf", f"fps={fps}:round=up,scale='min({width},iw)':-2",
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


def exact_rate(fps: float) -> float:
    """The exact frame rate behind a rounded one. GoPros record at NTSC rates: "29.97" is
    30000/1001, and counting frames at 29.97 would put some frame numbers off by one in a
    long video."""
    for base in (24, 30, 48, 60, 120, 240):
        ntsc = base * 1000 / 1001
        if abs(fps - ntsc) < 0.006:
            return ntsc
    return fps


def source_index(local_t: float, chapter_fps: float) -> int:
    """The number of the source frame sample_frames hands back for a moment in a chapter:
    the latest frame shown at or before it (frame 0 is at 0 s)."""
    return int(np.floor(local_t * exact_rate(chapter_fps) + 1e-6))


def grab_frames(
    path: Path, fps: float, indices: list[int], width: int | None = None, span: int = 300, threads: int = 4
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode source frames by number (source_index), in order: at full size, or scaled to
    `width` the way sample_frames scales. Numbers within `span` frames of each other are
    decoded in one pass. A seek half a frame before frame i lands on frame i (the first frame
    at or after the seek point), and the select filter counts from there. Few decoder threads:
    each holds several full-size frames, and with the species model loaded too, a 16 GB PC
    ran out of memory decoding 2.7K HEVC on all of them."""
    import imageio_ffmpeg  # noqa: PLC0415

    rate = exact_rate(fps)
    groups: list[list[int]] = []
    for i in sorted({i for i in indices if i >= 0}):
        if groups and i - groups[-1][0] <= span:
            groups[-1].append(i)
        else:
            groups.append([i])
    for group in groups:
        first = group[0]
        picks = "+".join(f"eq(n\\,{i - first})" for i in group)
        vf = f"select={picks}" + (f",scale={width}:-2" if width else "")
        # -frames:v: ffmpeg holds each frame back until the next one arrives, and after the
        # last wanted frame none does until the end of the video; the count releases it.
        gen = imageio_ffmpeg.read_frames(
            str(path),
            input_params=["-threads", str(threads), "-ss", f"{max(0.0, (first - 0.5) / rate):.6f}"],
            output_params=["-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(len(group))],
        )
        try:
            meta = next(gen)
            w, h = meta["size"]
            for i, raw in zip(group, gen):
                yield i, np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        finally:
            gen.close()


def probe_capture(path: Path) -> dict:
    """What the file says about the camera: encoder tag, size, codec (a DJI Osmo Action 4
    writes "DJI OsmoAction4"). Lens or field-of-view mode isn't in these tags: record it by
    hand (cli capture), or it stays unknown."""
    import re  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import imageio_ffmpeg  # noqa: PLC0415

    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, errors="replace",
    ).stderr
    video = re.search(r"Stream #\S+: Video: (\w+).*?, (\d{2,5})x(\d{2,5})", out)
    tags = dict(re.findall(r"^\s{4,}(encoder|firmware|make|model)\s*:\s(.*)$", out, re.M))
    return {
        "camera": tags.get("model") or tags.get("encoder") or "unknown",
        "firmware": tags.get("firmware"),
        "codec": video.group(1) if video else None,
        "size": [int(video.group(2)), int(video.group(3))] if video else None,
    }


@dataclass
class SampledFrame:
    t: float  # global time
    chapter: int
    local_t: float
    image: np.ndarray  # RGB uint8, H x W x 3
    src_index: int | None = None  # the source frame in its chapter (see source_index)


# How far before the start a run seeks; at least one source frame (see sample_frames).
SEEK_EARLY_S = 1.0
# Recorded with each detection run. 2: frames carry their source frame number, and a run
# that starts mid-video (a resume, or skipping the descent) is no longer one step late.
SAMPLER_VERSION = 2


def sample_frames(
    timeline: Timeline,
    fps: float,
    width: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    hwaccel: bool = True,
) -> Iterator[SampledFrame]:
    """Decode frames at `fps` across all chapters with an ffmpeg pipe.

    Frame k of a chapter is the latest source frame shown at or before local_start + k/fps.
    After a seek, ffmpeg's fps filter numbers its output from the first frame it decodes,
    which lands one step late whenever the start isn't exactly on a source frame (a resumed
    run, or one that skips the descent). So the run seeks a second early and tells the filter
    where it really starts (start_time): frames before it are dropped, and the numbering is
    the same wherever the run starts.
    """
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
        # Hardware decoding can fail rather than fall back (out of graphics memory, say),
        # before the first frame (OSError) or part way (RuntimeError): the rest of the
        # chapter is then decoded in software, from the next frame. A run that starts at a
        # frame numbers its frames as one that started earlier would, so nothing changes.
        done = 0  # frames of this chapter already yielded
        for hardware in (True, False) if hwaccel else (False,):
            begin = local_start + done * step
            seek = max(0.0, begin - SEEK_EARLY_S)
            # round=up: a source frame counts for the first output moment at or after it, so
            # each output moment gets the latest frame at or before it (round=near would hand
            # back the frame up to half a step later).
            vf = f"fps=fps={fps}:start_time={begin - seek:.6f}:round=up,scale={width}:-2"
            # -t counts from the seek point and may cut a frame early; the loop below stops exactly.
            output_params = ["-vf", vf, "-t", f"{begin - seek + local_len - done * step + step:.6f}"]
            input_params = ["-ss", f"{seek:.6f}"]
            if hardware:
                input_params = ["-hwaccel", "auto", *input_params]
            gen = imageio_ffmpeg.read_frames(
                str(ch.path), input_params=input_params, output_params=output_params
            )
            first = done
            try:
                meta = next(gen)
                w, h = meta["size"]
                for k, raw in enumerate(gen):
                    local_t = local_start + (first + k) * step
                    if local_t >= local_start + local_len - 1e-6:
                        break
                    image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                    done += 1
                    yield SampledFrame(
                        ch.start_s + local_t, ch.index, local_t, image, source_index(local_t, ch.fps)
                    )
                break
            except (OSError, RuntimeError) as exc:
                if not hardware:
                    raise
                log.warning("Hardware decoding failed on %s after %d frames (%s); decoding the rest in software",
                            ch.path.name, done, str(exc).splitlines()[0])
            finally:
                gen.close()
