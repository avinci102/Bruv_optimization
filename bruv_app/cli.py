"""Command line: process a video, open the review UI, export, or run the synthetic demo."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
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


@contextmanager
def keep_awake():
    """Ask Windows not to sleep while processing runs (closing a laptop's lid still does)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes  # noqa: PLC0415

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def cmd_process(args) -> None:
    from .pipeline import Pipeline, create_project, load_extra_species, update_extra_species  # noqa: PLC0415

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
        extra = load_extra_species(args.template)
        project = create_project(root, expand_videos(args.videos), args.template, settings, extra)
    else:
        from .store import Project  # noqa: PLC0415

        project = Project(root)
        project.save_settings(settings)
        update_extra_species(project, load_extra_species(project.get_meta("template")))
    print(f"Processing on {settings.resolved_device()} at {settings.resolved_fps():g} fps…")
    with keep_awake():
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


def cmd_score(args) -> None:
    from .score import score_against  # noqa: PLC0415
    from .store import Project  # noqa: PLC0415

    for folder in (args.project, args.answers):
        if not (Path(folder) / "project.db").exists():
            raise ValueError(f"No project in {folder}")
    score = score_against(Project(Path(args.project)), Project(Path(args.answers)))
    print(f"{score.cards} cards: {score.cards - score.tiny} to review, {score.tiny} tiny (hidden)")
    print(f"{score.checked} hold sightings answered in {args.answers}; {len(score.mixed)} of them mix answers:")
    for card, answers in score.mixed:
        print(f"  card #{card}: {' / '.join(answers)}")
    print(f"{score.rejoined} of {score.pulled_apart} groups of sightings you pulled apart are on one card again")


CFD_NANO_URL = (
    "https://github.com/filippovarini/community-fish-detector/releases/download/"
    "2026.07.06-release/cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth"
)


def model_cache() -> Path:
    """Per-user folder for downloaded model files (reused across projects)."""
    import os  # noqa: PLC0415

    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "bruv_app" / "models"


def download(url: str, dest: Path) -> Path:
    """Download once to `dest` (via a .part file), printing progress."""
    import urllib.request  # noqa: PLC0415

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name}…")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    with open(part, "rb") as f:
        is_web_page = f.read(1) == b"<"  # an HTML error page, not a checkpoint
    if is_web_page:  # (Windows can't delete a file that is still open)
        part.unlink()
        raise RuntimeError(f"The download from {url} returned a web page, not a model file")
    part.replace(dest)
    return dest


def find_inputs(folder: Path) -> tuple[list[Path], Path]:
    """The video file(s) and the one Excel template in a folder."""
    from .template import read_template  # noqa: PLC0415

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    videos = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise FileNotFoundError(f"No .MP4/.MOV video in {folder}")
    if any(p.suffix.lower() == ".xls" for p in folder.iterdir()):
        print("Note: .xls files can't be read; save the template as .xlsx in Excel.")
    candidates = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in (".xlsx", ".xlsm") and not p.name.startswith("~$")
    ]
    templates = []
    for p in candidates:
        try:
            read_template(p)
            templates.append(p)
        except (ValueError, KeyError, OSError):
            continue
    if not templates:
        raise FileNotFoundError(
            f"No Excel template with a 'record' sheet (species, time_in, …) in {folder}"
        )
    if len(templates) > 1:
        names = ", ".join(p.name for p in templates)
        raise ValueError(f"More than one template in the folder ({names}); keep just one")
    return videos, templates[0]


def cmd_quickstart(args) -> None:
    """Everything in one go: find inputs, fetch the detector, process, open the review page."""
    from .pipeline import Pipeline, create_project, load_extra_species, update_extra_species  # noqa: PLC0415
    from .store import Project  # noqa: PLC0415

    folder = Path(args.folder).expanduser().resolve()
    videos, template = find_inputs(folder)
    print(f"Video: {', '.join(p.name for p in videos)}")
    print(f"Template: {template.name}")
    root = folder / "bruv_project"
    settings_path = root / "settings.json"
    settings = Settings.load(settings_path) if settings_path.exists() else Settings()
    for key in ("detector", "classifier", "sample_fps"):
        if getattr(args, key) is not None:
            setattr(settings, key, getattr(args, key))
    if settings.detector == "rfdetr":
        weights = Path(args.weights) if args.weights else download(
            CFD_NANO_URL, model_cache() / CFD_NANO_URL.rsplit("/", 1)[1]
        )
        settings.detector_weights = str(weights.resolve())
    extra = load_extra_species(template)
    if (root / "project.db").exists():
        print(f"Resuming the project in {root}")
        project = Project(root)
        project.save_settings(settings)
        update_extra_species(project, extra)
    else:
        project = create_project(root, videos, template, settings, extra)
    print(f"Processing on {settings.resolved_device()} at {settings.resolved_fps():g} fps…")
    with keep_awake():
        Pipeline(project, progress=_progress()).run()
    print(f"Done: {len(project.all_fish())} fish cards. Exports will be saved in {root / 'exports'}")
    if not args.no_review:
        print("Keep this window open while you review; close it when you're finished.")
        serve(root, args.port, not args.no_browser)


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
    p.add_argument(
        "--clip-names", choices=["auto", "file", "index"],
        help="GoPro file name or clip_1, clip_2… (auto: clip_N if the template lists them)",
    )
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

    p = sub.add_parser("score", help="check a re-run's cards against the answers in another copy")
    p.add_argument("--project", required=True, help="the project to score (e.g. re-run with new rules)")
    p.add_argument("--answers", required=True, help="a copy of the same project with reviewed cards")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("quickstart", help="one step: a folder with your video(s) + template → review page")
    p.add_argument("folder", help="folder holding the GoPro .MP4 file(s) and the Excel template")
    p.add_argument("--weights", help="use this detector .pth instead of downloading CFD Nano")
    p.add_argument("--detector", choices=["rfdetr", "color"])
    p.add_argument("--classifier", choices=["bioclip", "histogram"])
    p.add_argument("--fps", dest="sample_fps", type=float)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-review", action="store_true")
    p.set_defaults(func=cmd_quickstart)

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
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
