"""Process every new BRUV video in a folder and get it onto the review site: what the GPU PC
runs (docs/GPU_PC.md).

    python -m bruv_app.batch --videos C:\\BRUV\\videos --site C:\\BRUV\\site [--deploy]

- Each subfolder of --videos is one deployment: its video file(s) (the camera's chapters,
  any order) and the Excel template. Its name is the video's name on the site (letters,
  digits, - and _; anything else becomes -).
- A folder without a template gets a copy of --template (default: the one .xlsx in
  --videos itself).
- Each is processed as quickstart does, into <folder>\\bruv_project. Processing resumes
  where it stopped, and is retried twice after a crash (running out of memory, say).
- The site folder is brought in step with the live site (sitesync mirror). Then each
  processed video not yet published from its project is published under its name (with
  --follow unless --no-follow).
- Last, sitesync's check says what a deploy would do; with --deploy it deploys (asking
  first unless --yes).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import sitesync

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}  # as cli.py reads
TEMPLATE_SUFFIXES = {".xlsx", ".xlsm"}


def site_name(folder_name: str) -> str:
    """A folder's name as a video name on the site (it becomes part of the address)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", folder_name).strip("-")


def deployments(videos: Path) -> list[Path]:
    """The subfolders holding video files, by name."""
    return [p for p in sorted(videos.iterdir())
            if p.is_dir() and any(f.suffix.lower() in VIDEO_SUFFIXES for f in p.iterdir())]


def _templates(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in TEMPLATE_SUFFIXES and not p.name.startswith("~$")]


def ensure_template(folder: Path, template: Path | None) -> None:
    """Copy the template into a deployment folder that has none."""
    if _templates(folder):
        return
    if template is None:
        raise ValueError(f"{folder.name} has no Excel template: put one in it, or one in the videos folder")
    shutil.copy2(template, folder / template.name)


def published_as(project: Path) -> str | None:
    """The video name a project was last published under (publish records it)."""
    db = project / "project.db"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM meta WHERE key='published'").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    return json.loads(row[0]).get("video") if row else None


def run_app(*args: str, retries: int = 0, wait_s: float = 30, run=subprocess.run) -> int:
    """`python -m bruv_app ...` in its own process, retried after a crash. Exit code 1 is an
    error the app explained (no template, say): trying again won't help."""
    for attempt in range(retries + 1):
        code = run([sys.executable, "-m", "bruv_app", *args]).returncode
        if code in (0, 1) or attempt == retries:
            return code
        print(f"\nStopped unexpectedly (exit code {code}). Picking up where it left off in {wait_s:g} seconds…")
        time.sleep(wait_s)
    return code


def main(argv: list[str] | None = None, run=subprocess.run) -> int:
    parser = argparse.ArgumentParser(prog="python -m bruv_app.batch", description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", required=True, help="the folder holding one subfolder per deployment")
    parser.add_argument("--site", required=True, help="the site folder (sitesync init made it)")
    parser.add_argument("--template", help="the Excel template for folders without one")
    parser.add_argument("--only", nargs="+", metavar="FOLDER", help="just these deployment folders")
    parser.add_argument("--fps", help="sampled frames per second (default: the app's, 4)")
    parser.add_argument("--no-follow", action="store_true", help="publish without following a fish (smaller)")
    parser.add_argument("--republish", nargs="+", default=[], metavar="VIDEO",
                        help="publish these again even if they were published from their projects")
    parser.add_argument("--no-mirror", action="store_true", help="don't fetch the live videos first")
    parser.add_argument("--deploy", action="store_true", help="deploy at the end (it checks, then asks)")
    parser.add_argument("--yes", action="store_true", help="with --deploy: don't ask")
    args = parser.parse_args(argv)

    videos, site = Path(args.videos), Path(args.site)
    if not videos.is_dir():
        print(f"No folder {videos}: put each deployment's videos in a subfolder of it (docs/GPU_PC.md)")
        return 1
    folders = deployments(videos)
    if args.only:
        folders = [f for f in folders if f.name in args.only]
    if not folders:
        print(f"No video folders in {videos}: put each deployment's videos in a subfolder of it")
        return 1
    template = Path(args.template) if args.template else None
    if template is None:
        found = _templates(videos)
        if len(found) > 1:
            print(f"More than one template in {videos} ({', '.join(p.name for p in found)}): keep one, or give "
                  "--template")
            return 1
        template = found[0] if found else None
    names = {f: site_name(f.name) for f in folders}
    clash = {n for n in names.values() if list(names.values()).count(n) > 1 or not n}
    if clash:
        print(f"Two folders would have the same name on the site, or none: {', '.join(sorted(clash))}. Rename them.")
        return 1

    failed: list[str] = []
    processed: list[Path] = []
    for n, folder in enumerate(folders, 1):
        print(f"\n=== {n}/{len(folders)}: {folder.name} (on the site as {names[folder]}) ===")
        try:
            ensure_template(folder, template)
        except ValueError as exc:
            print(exc)
            failed.append(folder.name)
            continue
        extra = ["--fps", args.fps] if args.fps else []
        if run_app("quickstart", str(folder), "--no-review", *extra, retries=2, run=run) == 0:
            processed.append(folder)
        else:
            failed.append(folder.name)

    if not args.no_mirror:
        print("\n=== The live site's videos ===")
        try:
            sitesync.mirror(site)
        except (sitesync.SyncError, OSError, ValueError) as exc:
            print(f"Couldn't bring the site folder in step with the live site: {exc}")

    print("\n=== Publishing ===")
    for folder in processed:
        name, project = names[folder], folder / "bruv_project"
        here = site / "public" / "videos" / name
        if published_as(project) == name and here.is_dir() and name not in args.republish:
            print(f"{name}: already published from {folder.name}")
            continue
        if here.is_dir() and published_as(project) != name:
            print(f"{name}: the site already has a video by that name, from another project. Rename the folder "
                  f"{folder.name}, or delete public/videos/{name} if it's an old copy.")
            failed.append(folder.name)
            continue
        follow = [] if args.no_follow else ["--follow"]
        if run_app("publish", "--project", str(project), "--video", name, "--site", str(site), *follow,
                   run=run) != 0:
            failed.append(folder.name)

    print("\n=== Ready to go online? ===")
    try:
        if args.deploy:
            ok = sitesync.deploy(site, yes=args.yes)
        else:
            result = sitesync.check(site, sitesync.local_videos(site), sitesync.live_index(sitesync.site_url(site)),
                                    sitesync.load_state(site), sitesync.code_version())
            for note in result.notes:
                print(f"  {note}")
            for problem in result.problems:
                print(f"  - {problem}")
            ok = not result.problems
            if ok:
                print(f"To put it online: python -m bruv_app.sitesync deploy --site {site}")
    except (sitesync.SyncError, OSError, ValueError) as exc:
        print(f"Couldn't check against the live site: {exc}")
        ok = False
    if failed:
        print(f"\nNot done: {', '.join(failed)} (see above)")
    return 0 if ok and not failed else 2


if __name__ == "__main__":
    sys.exit(main())
