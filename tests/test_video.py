from pathlib import Path

import numpy as np
import pytest

from bruv_app.video import build_timeline, order_chapters, sample_frames


def test_order_chapters_new_gopro_naming():
    names = ["GX020042.MP4", "GX010043.MP4", "GX010042.MP4", "GX030042.MP4"]
    ordered = [p.name for p in order_chapters([Path(n) for n in names])]
    assert ordered == ["GX010042.MP4", "GX020042.MP4", "GX030042.MP4", "GX010043.MP4"]


def test_order_chapters_old_gopro_naming_and_other_files():
    names = ["GP020012.MP4", "other.mp4", "GOPR0012.MP4", "GP010012.MP4"]
    ordered = [p.name for p in order_chapters([Path(n) for n in names])]
    assert ordered == ["GOPR0012.MP4", "GP010012.MP4", "GP020012.MP4", "other.mp4"]


def test_timeline_locate_maps_global_time_to_clip():
    durations = {"GX010001": 100.0, "GX020001": 50.0}
    timeline = build_timeline(
        [Path("GX020001.MP4"), Path("GX010001.MP4")],
        probe_fn=lambda p: (durations[p.stem], 30.0),
    )
    assert [c.name for c in timeline.chapters] == ["GX010001", "GX020001"]
    assert timeline.duration_s == 150.0
    assert timeline.locate(10.0) == ("GX010001", 10.0)
    assert timeline.locate(120.0) == ("GX020001", 20.0)


def test_timeline_index_clip_names():
    timeline = build_timeline(
        [Path("GX010001.MP4"), Path("GX020001.MP4")], "index", probe_fn=lambda p: (10.0, 30.0)
    )
    assert [c.name for c in timeline.chapters] == ["clip_1", "clip_2"]


def test_sample_frames_across_chapters(synthetic_videos):
    timeline = build_timeline(synthetic_videos)
    assert timeline.duration_s == pytest.approx(24.0, abs=0.2)
    frames = list(sample_frames(timeline, fps=1.0, width=320))
    assert len(frames) == 24
    assert [round(f.t) for f in frames] == list(range(24))
    assert frames[12].chapter == 1 and frames[12].local_t == pytest.approx(0.0)
    assert frames[0].image.shape == (180, 320, 3)


def test_sample_frames_analysis_window(synthetic_videos):
    timeline = build_timeline(synthetic_videos)
    frames = list(sample_frames(timeline, fps=2.0, width=320, start_s=10.0, end_s=14.0))
    assert len(frames) == 8
    assert frames[0].t == pytest.approx(10.0)
    assert frames[-1].t == pytest.approx(13.5)


def test_sampled_frame_times_are_exact(tmp_path):
    """Each sampled frame is the source frame at its timestamp (brightness encodes the index)."""
    import imageio_ffmpeg

    path = tmp_path / "GX010001.MP4"
    writer = imageio_ffmpeg.write_frames(str(path), (64, 64), fps=10, codec="libx264", macro_block_size=8, quality=10)
    writer.send(None)
    for i in range(60):
        writer.send(np.full((64, 64, 3), i * 4, np.uint8))
    writer.close()
    timeline = build_timeline([path])
    for start in (0.0, 2.3):
        frames = list(sample_frames(timeline, fps=1.0, width=64, start_s=start))
        for f in frames:
            index = round(f.image.mean() / 4)
            assert index == pytest.approx(f.t * 10, abs=0.6), (start, f.t, index)


def test_expand_videos_accepts_folders_and_wildcards(tmp_path):
    from bruv_app.cli import expand_videos

    for name in ("GX020001.MP4", "GX010001.MP4", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert {p.name for p in expand_videos([str(tmp_path)])} == {"GX010001.MP4", "GX020001.MP4"}
    assert len(expand_videos([str(tmp_path / "GX*.MP4")])) == 2
    with pytest.raises(FileNotFoundError):
        expand_videos([str(tmp_path / "missing.MP4")])
