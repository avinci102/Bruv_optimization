from pathlib import Path

import numpy as np
import pytest

from bruv_app.video import build_timeline, order_chapters, sample_frames


def test_order_chapters_new_gopro_naming():
    names = ["GX020042.MP4", "GX010043.MP4", "GX010042.MP4", "GX030042.MP4"]
    ordered = [p.name for p in order_chapters([Path(n) for n in names])]
    assert ordered == ["GX010042.MP4", "GX020042.MP4", "GX030042.MP4", "GX010043.MP4"]


def test_order_chapters_old_gopro_naming_and_other_files():
    names = ["GP020012.MP4", "other.mp4", "GOPR0012.MP4", "GP010012.MP4", "GL010012.LRV"]
    ordered = [p.name for p in order_chapters([Path(n) for n in names])]
    # GL… is GoPro's low-res proxy: it is not a chapter, so it sorts with the non-GoPro files.
    assert ordered == ["GOPR0012.MP4", "GP010012.MP4", "GP020012.MP4", "GL010012.LRV", "other.mp4"]


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

    for name in ("GX020001.MP4", "GX010001.MP4", "GL010001.LRV", "GX010001.THM", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert {p.name for p in expand_videos([str(tmp_path)])} == {"GX010001.MP4", "GX020001.MP4"}
    assert len(expand_videos([str(tmp_path / "GX*.MP4")])) == 2
    with pytest.raises(FileNotFoundError):
        expand_videos([str(tmp_path / "missing.MP4")])


def test_shrink_fits_the_size_limit_and_keeps_the_name(synthetic_videos, tmp_path):
    from bruv_app.cli import main
    from bruv_app.video import probe

    src = next(p for p in synthetic_videos if p.name == "GX010042.MP4")
    main(["shrink", str(src), "--max-mb", "0.5", "--out", str(tmp_path / "small" / src.name)])
    out = tmp_path / "small" / src.name
    assert out.stat().st_size <= 0.5 * 1024 * 1024
    duration, fps = probe(out)
    assert duration == pytest.approx(12.0, abs=0.3) and fps == pytest.approx(10.0)
    # Still a usable input: same GoPro name, frames sample at the right times.
    frames = list(sample_frames(build_timeline([out]), fps=1.0, width=640))
    assert out.stem == "GX010042" and len(frames) == 12


def test_shrink_refuses_impossible_targets(synthetic_videos, tmp_path):
    from bruv_app.video import shrink_video

    with pytest.raises(ValueError):
        shrink_video(synthetic_videos[0], tmp_path / "x.mp4", max_mb=0.001)
    with pytest.raises(ValueError):
        shrink_video(synthetic_videos[0], synthetic_videos[0])


def test_quickstart_from_a_folder(synthetic_videos, tmp_path):
    import shutil

    from bruv_app import demo
    from bruv_app.cli import main
    from bruv_app.store import Project

    folder = tmp_path / "BRUV_optimization"
    folder.mkdir()
    for p in synthetic_videos:
        shutil.copy(p, folder / p.name)
    demo.make_mock_template(folder / "viewing template v.2024.xlsx")
    (folder / "~$viewing template v.2024.xlsx").write_bytes(b"lock file")  # Excel's lock file
    main(["quickstart", str(folder), "--detector", "color", "--classifier", "histogram", "--no-review"])
    project = Project(folder / "bruv_project")
    assert project.frame_count() == 96 and project.all_fish()  # 24 s at the default 4 fps
    # Running it again resumes instead of starting over.
    main(["quickstart", str(folder), "--detector", "color", "--classifier", "histogram", "--no-review"])
    assert project.frame_count() == 96


def test_quickstart_needs_exactly_one_template(synthetic_videos, tmp_path):
    import shutil

    from bruv_app import demo
    from bruv_app.cli import find_inputs

    shutil.copy(synthetic_videos[0], tmp_path / synthetic_videos[0].name)
    with pytest.raises(FileNotFoundError):
        find_inputs(tmp_path)
    demo.make_mock_template(tmp_path / "a.xlsx")
    demo.make_mock_template(tmp_path / "b.xlsx")
    with pytest.raises(ValueError):
        find_inputs(tmp_path)


def test_download_rejects_web_pages(tmp_path):
    from bruv_app.cli import download

    good, page = tmp_path / "good.bin", tmp_path / "page.html"
    good.write_bytes(b"PK\x03\x04 weights")
    page.write_bytes(b"<html>Not found</html>")
    assert download(good.as_uri(), tmp_path / "out" / "w.pth").read_bytes().startswith(b"PK")
    with pytest.raises(RuntimeError):
        download(page.as_uri(), tmp_path / "out" / "bad.pth")
    assert not (tmp_path / "out" / "bad.pth").exists()
