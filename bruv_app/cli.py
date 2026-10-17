"""Command line: process a video, open the review UI, export, or run the synthetic demo."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .config import Settings


def _progress():
    last = {"stage": None, "t": 0.0}

    def report(stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        if stage == last["stage"] and done < total and now - last["t"] < 0.5:
            return
        last.update(stage=stage, t=now)
        pct = 100 * done / total if total else 100
        end = "\n" if done >= total else ""
        print(f"\r  {stage:<9} {done:>6}/{total:<6} {pct:5.1f}%", end=end, flush=True)

    return report


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}


def expand_videos(items: list[str]) -> list[Path]:
    """Accept files, folders and wildcards (Windows' shell doesn't expand *.MP4 for us)."""
    out: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            out.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES)
        elif any(ch in item for ch in "*?["):
            # Skip GoPro's .LRV proxies and .THM thumbnails that a G* pattern also matches.
            matches = (Path(p) for p in sorted(glob.glob(item)))
            out.extend(p for p in matches if p.suffix.lower() in VIDEO_SUFFIXES)
        else:
            out.append(path)
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Video not found: {', '.join(missing)}")
    if items and not out:
        raise FileNotFoundError(f"No videos matched: {' '.join(items)}")
    return out


def cmd_process(args) -> None:
    from .pipeline import Pipeline, create_project  # noqa: PLC0415

    root = Path(args.project)
    settings_path = root / "settings.json"
    settings = Settings.load(settings_path) if settings_path.exists() else Settings()
    for key in ("detector", "classifier", "start_s", "duration_s", "clip_names", "sample_fps", "device"):
        value = getattr(args, key)
        if value is not None:
            setattr(settings, key, value)
    if args.weights:
        settings.detector_weights = str(Path(args.weights).resolve())
    if settings.detector == "rfdetr" and not settings.detector_weights:
        raise ValueError(
            "No detector weights. Download a Community Fish Detector RF-DETR .pth "
            "(Nano for CPU) and pass --weights."
        )
    if args.videos:
        from .store import Project  # noqa: PLC0415

        if not args.template and not (root / "project.db").exists():
            raise ValueError("--template is required for a new project (it holds the species list)")
        if not args.template:
            args.template = Project(root).get_meta("template")
        project = create_project(root, expand_videos(args.videos), args.template, settings)
    else:
        from .store import Project  # noqa: PLC0415

        project = Project(root)
        project.save_settings(settings)
    print(f"Processing on {settings.resolved_device()} at {settings.resolved_fps():g} fps…")
    Pipeline(project, progress=_progress()).run(force=args.force)
    print(f"Done: {len(project.all_fish())} fish cards. Review with:")
    print(f"  python -m bruv_app review --project \"{root}\"")


def serve(root: Path, port: int, open_browser: bool) -> None:
    import uvicorn  # noqa: PLC0415

    from .server import create_app  # noqa: PLC0415

    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Review UI at {url}  (Ctrl+C to stop)")
    uvicorn.run(create_app(root), host="127.0.0.1", port=port, log_level="warning")


def cmd_review(args) -> None:
    serve(Path(args.project), args.port, not args.no_browser)


def cmd_export(args) -> None:
    from .export import export_project  # noqa: PLC0415
    from .store import Project  # noqa: PLC0415

    path = export_project(Project(Path(args.project)), Path(args.out) if args.out else None)
    print(f"Saved {path}")


def cmd_shrink(args) -> None:
    from .video import shrink_video  # noqa: PLC0415

    src = Path(args.video)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    # Same file name in a "small" folder, so GoPro chapter naming still works.
    dst = Path(args.out) if args.out else src.parent / "small" / src.name
    print(f"Shrinking {src.name} to under {args.max_mb:g} MB…")
    shrink_video(src, dst, args.max_mb, args.width, args.fps, args.start, args.duration)
    print(f"Saved {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


def cmd_demo(args) -> None:
    from . import demo  # noqa: PLC0415
    from .pipeline import Pipeline, create_project  # noqa: PLC0415

    root = Path(args.project)
    if (root / "project.db").exists():
        print(f"Re-opening existing demo project at {root}")
    else:
        print("Making a synthetic video and a mock template…")
        videos = demo.make_synthetic_videos(root / "videos")
        template = demo.make_mock_template(root / "mock_template.xlsx")
        settings = Settings(detector="color", classifier="histogram")
        project = create_project(root, videos, template, settings)
        Pipeline(project, classifier=demo.DemoClassifier(), progress=_progress()).run()
    if not args.no_review:
        serve(root, args.port, not args.no_browser)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="bruv_app", description="BRUV fish ID assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="ingest videos and build review cards (resumable)")
    p.add_argument("videos", nargs="*", help="GoPro chapter files or a folder (any order); omit to resume")
    p.add_argument("--project", required=True, help="project folder (created if missing)")
    p.add_argument("--template", help="the Excel viewing template (.xlsx)")
    p.add_argument("--weights", help="Community Fish Detector RF-DETR .pth")
    p.add_argument("--detector", choices=["rfdetr", "color"])
    p.add_argument("--classifier", choices=["bioclip", "histogram"])
    p.add_argument("--fps", dest="sample_fps", type=float, help="sampled frames per second")
    p.add_argument("--start", dest="start_s", type=float, help="skip this many seconds (descent)")
    p.add_argument("--duration", dest="duration_s", type=float, help="analyse this many seconds")
    p.add_argument("--clip-names", choices=["file", "index"], help="GoPro file name or clip_1, clip_2…")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--force", action="store_true", help="rebuild cards even if some are reviewed")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("review", help="open the review UI")
    p.add_argument("--project", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("export", help="write the reviewed results into a copy of the template")
    p.add_argument("--project", required=True)
    p.add_argument("--out", help="output .xlsx (default: <project>/exports/…)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("shrink", help="make a small copy of a clip to share (e.g. under an upload limit)")
    p.add_argument("video")
    p.add_argument("--max-mb", type=float, default=38.0, help="target size in MB (default 38)")
    p.add_argument("--width", type=int, default=1280, help="max width in pixels (default 1280)")
    p.add_argument("--fps", type=float, default=10.0, help="frames per second to keep (default 10)")
    p.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    p.add_argument("--duration", type=float, help="keep only this many seconds")
    p.add_argument("--out", help="output file (default: <video folder>/small/<same name>)")
    p.set_defaults(func=cmd_shrink)

    p = sub.add_parser("demo", help="run the full workflow on a synthetic video")
    p.add_argument("--project", default="projects/demo")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-review", action="store_true", help="process only, don't start the UI")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
