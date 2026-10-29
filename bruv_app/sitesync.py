"""Keep a site folder in step with the live site, so any PC can add videos and deploy.

A Pages deploy replaces the whole live site with the folder it uploads: a PC that deploys
without every live video takes the ones it lacks off the site. Site folders keep in step
through the live site itself (plain downloads, free), with no copying between PCs:

- every deploy puts an index of the videos' files (size and SHA-256) and the code's git
  commit at <site>/videos/index.json;
- `mirror` downloads the videos this folder lacks, and those changed online since it last
  saw them;
- `deploy` refreshes the site's own files from this checkout, writes the index, checks
  what the deploy would do, asks, runs `wrangler pages deploy` and records what is live.

The check refuses a deploy that would take a live video off the site, put back an older
copy of one, or run code older than the live site's, unless told to. Each folder
remembers the videos as it last saw them live (.sync.json, beside public/, never
uploaded): that is how it tells a video re-published here (goes up) from one changed
online (comes down). Standard library only, apart from write_site. See docs/GPU_PC.md.

    python -m bruv_app.sitesync init   --site C:\\BRUV\\site
    python -m bruv_app.sitesync mirror --site C:\\BRUV\\site
    python -m bruv_app.sitesync check  --site C:\\BRUV\\site
    python -m bruv_app.sitesync deploy --site C:\\BRUV\\site
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

INDEX = "videos/index.json"  # under public/, so served at <site>/videos/index.json
STATE = ".sync.json"  # beside public/: not uploaded
FILE_LIMIT = 20_000  # files in one Pages deployment, free plan
SIZE_LIMIT = 25 * 1024 * 1024  # bytes in one file
REPO = Path(__file__).resolve().parents[1]
NODE_DIR = Path("C:/Program Files/nodejs")  # where Node's installer puts npx


class SyncError(RuntimeError):
    """Something the person has to sort out; the message says what."""


# --- What a folder holds ---------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_files(folder: Path) -> dict[str, list]:
    """Every file of one published video: {path inside its folder: [size, sha256]}."""
    return {
        p.relative_to(folder).as_posix(): [p.stat().st_size, sha256(p)]
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.endswith(".part")
    }


def digest(files: dict[str, list]) -> str:
    """A video's fingerprint: it changes when any of its files does."""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def local_videos(site: Path) -> dict[str, dict]:
    """{video: {"digest", "files"}} for every video folder under public/videos."""
    root = site / "public" / "videos"
    out = {}
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            files = video_files(folder)
            out[folder.name] = {"digest": digest(files), "files": files}
    return out


def load_state(site: Path) -> dict:
    path = site / STATE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(site: Path, state: dict) -> None:
    site.mkdir(parents=True, exist_ok=True)
    (site / STATE).write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")


# --- The code this checkout runs ------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None


def site_code() -> str:
    """A fingerprint of the code the site runs (the pages, scripts and functions publish
    copies), the same on any PC with the same code whatever its line endings."""
    h = hashlib.sha256()
    package = REPO / "bruv_app"
    for folder in (package / "static", package / "hosted"):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.relative_to(package).as_posix().encode() + b"\0")
                h.update(hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()


def code_version() -> dict:
    """This checkout's commit (None outside a git checkout), whether the app's code has
    uncommitted changes, and the site code's fingerprint."""
    head = _git("rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return {"commit": None, "dirty": False, "site_code": site_code()}
    status = _git("status", "--porcelain", "--", "bruv_app")
    return {"commit": head.stdout.strip(), "dirty": bool(status and status.stdout.strip()), "site_code": site_code()}


def has_commit(commit: str) -> bool:
    """Whether this checkout's code includes `commit` (False when git hasn't fetched it)."""
    run = _git("merge-base", "--is-ancestor", commit, "HEAD")
    return run is not None and run.returncode == 0


# --- The live site ---------------------------------------------------------------------


def site_url(site: Path, url: str | None = None) -> str:
    """The live site's address: as given, as last used, or from wrangler.toml's project name."""
    if url:
        return url.rstrip("/")
    known = load_state(site).get("url")
    if known:
        return known
    toml = site / "wrangler.toml"
    if toml.exists():
        for line in toml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("["):
                break  # past the top-level settings
            key, _, value = line.partition("=")
            if key.strip() == "name":
                return "https://" + value.strip().strip('"') + ".pages.dev"
    raise SyncError("Give the live site's address with --url (e.g. https://bruv-review.pages.dev)")


def _get(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "bruv-sitesync"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def live_index(url: str) -> dict | None:
    """The live site's index, or None when it has none yet (deployed before this existed)."""
    try:
        data = _get(f"{url}/{INDEX}?t={int(time.time())}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    index = json.loads(data)
    if index.get("format") != 1:
        raise SyncError(f"{url}/{INDEX} is in a format this program doesn't know: update it (git pull)")
    return index


def _download_video(site: Path, url: str, name: str, files: dict[str, list], workers: int, log) -> None:
    """Make public/videos/<name> hold exactly the live copy's files, fetching only those that
    differ."""
    folder = site / "public" / "videos" / name
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.relative_to(folder).as_posix() not in files:
                p.unlink()
    todo = []
    for rel, (size, sha) in files.items():
        p = folder / rel
        if not (p.is_file() and p.stat().st_size == size and sha256(p) == sha):
            todo.append((rel, size, sha))
    if not todo:
        return
    log(f"  {name}: downloading {len(todo)} of its {len(files)} files ({sum(t[1] for t in todo) / 1e6:.0f} MB)")

    def fetch(item: tuple[str, int, str]) -> int:
        rel, size, sha = item
        address = url + "/" + "/".join(urllib.parse.quote(part) for part in f"videos/{name}/{rel}".split("/"))
        for attempt in range(3):
            try:
                data = _get(address)
                if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                    raise SyncError(f"{address} isn't the file the live index lists (deployed again meanwhile?)")
                target = folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                part.write_bytes(data)
                part.replace(target)
                return size
            except (OSError, SyncError):  # URLError is an OSError
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        return 0

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(fetch, todo), 1):
            done = n
            if n % 200 == 0:
                log(f"    {n} / {len(todo)}")
    log(f"    {done} files done")


# --- Comparing a folder with the live site ---------------------------------------------


@dataclass
class Plan:
    """Each video, against the live site and what this folder last saw live."""

    same: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)  # here only, never seen live: goes up
    changed_here: list[str] = field(default_factory=list)  # re-published here: goes up
    missing: list[str] = field(default_factory=list)  # live, not here: mirror fetches it
    changed_online: list[str] = field(default_factory=list)  # another PC re-published it: mirror fetches it
    removed_here: list[str] = field(default_factory=list)  # live, deleted here since: would drop off
    removed_online: list[str] = field(default_factory=list)  # taken off the site since: would go back up
    conflict: list[str] = field(default_factory=list)  # changed both here and online


def compare(here: dict[str, str], live: dict[str, str], seen: dict[str, str]) -> Plan:
    """Sort the videos by their digests: here, live, and as this folder last saw them live."""
    plan = Plan()
    for name in sorted(here.keys() | live.keys()):
        h, l, s = here.get(name), live.get(name), seen.get(name)
        if h == l:
            plan.same.append(name)
        elif h is None:
            (plan.removed_here if s == l else plan.missing).append(name)
        elif l is None:
            (plan.new if s is None else plan.removed_online).append(name)
        elif s == h:
            plan.changed_online.append(name)
        elif s == l:
            plan.changed_here.append(name)
        else:
            plan.conflict.append(name)
    return plan


def _digests(videos: dict[str, dict]) -> dict[str, str]:
    return {name: v["digest"] for name, v in videos.items()}


def mirror(site: Path, url: str | None = None, replace: tuple[str, ...] = (), workers: int = 6, log=print) -> Plan:
    """Download the live videos this folder lacks or holds an older copy of. With `replace`,
    those videos' live copies replace this folder's whatever changed where."""
    url = site_url(site, url)
    live = live_index(url)
    if live is None:
        raise SyncError(
            f"{url} has no index of its videos yet, so they can't be fetched. Deploy once from the PC "
            "that has every live video, with `sitesync deploy --first` (docs/GPU_PC.md)."
        )
    here = local_videos(site)
    state = load_state(site)
    seen = state.setdefault("videos", {})
    unfinished = state.setdefault("unfinished", [])  # a download stopped part way: not a change made here
    plan = compare(_digests(here), _digests(live["videos"]), seen)
    fetch = [n for n in sorted(live["videos"]) if n in set(plan.missing + plan.changed_online + unfinished)
             or (n in replace and n not in plan.same)]
    state["url"] = url
    for name in fetch:
        if name not in unfinished:
            unfinished.append(name)
        save_state(site, state)
        _download_video(site, url, name, live["videos"][name]["files"], workers, log)
        seen[name] = live["videos"][name]["digest"]
        unfinished.remove(name)
    for name in plan.same:
        seen[name] = live["videos"][name]["digest"]
    save_state(site, state)
    log(f"Live at {url}: {', '.join(sorted(live['videos'])) or 'no videos'}")
    log(f"  fetched: {', '.join(fetch) or 'nothing (all up to date)'}")
    waiting = [n for n in plan.new + plan.changed_here if n not in fetch]
    if waiting:
        log(f"  here, not live yet (they go up with the next deploy): {', '.join(waiting)}")
    for name in plan.removed_here:
        log(f"  {name}: live, but deleted here. The next deploy takes it off the site only with --drop {name}.")
    for name in plan.removed_online:
        log(f"  {name}: taken off the live site since this folder last saw it. Delete public/videos/{name} here, "
            f"or deploy with --overwrite {name} to put it back.")
    for name in plan.conflict:
        if name not in fetch:
            log(f"  {name}: changed both here and online. `mirror --replace {name}` takes the live copy; "
                f"`deploy --overwrite {name}` puts this folder's copy online instead.")
    return plan


@dataclass
class Check:
    plan: Plan
    problems: list[str]
    notes: list[str]
    files: int
    bytes: int


def check(site: Path, here: dict[str, dict], live: dict | None, state: dict, code: dict, *,
          first: bool = False, drop: tuple[str, ...] = (), overwrite: tuple[str, ...] = (),
          force_code: bool = False, file_limit: int = FILE_LIMIT, size_limit: int = SIZE_LIMIT) -> Check:
    """What deploying this folder would do, and what stands in the way. `state` is the
    folder's .sync.json."""
    problems: list[str] = []
    notes: list[str] = []
    if live is None:
        if not first:
            problems.append(
                "The live site has no index of its videos yet, so this can't tell which ones it holds. Deploy "
                "once with --first from the PC that has every live video (docs/GPU_PC.md)."
            )
        live = {"videos": {}}
    for name in state.get("unfinished", []):
        problems.append(f"{name}'s download didn't finish: run mirror again.")
    plan = compare(_digests(here), _digests(live["videos"]), state.get("videos", {}))
    for name in plan.missing:
        if name not in drop:
            problems.append(f"{name} is live but not in this folder: run mirror first (or --drop {name} to take it "
                            "off the site).")
    for name in plan.removed_here:
        if name not in drop:
            problems.append(f"{name} is live but was deleted here: deploying takes it off the site. Pass --drop "
                            f"{name} if that's meant.")
    for name in plan.changed_online:
        if name not in overwrite:
            problems.append(f"{name} was re-published online since this folder saw it: run mirror first (or "
                            f"--overwrite {name} to put this folder's older copy back).")
    for name in plan.conflict:
        if name not in overwrite:
            problems.append(f"{name} changed both here and online: `mirror --replace {name}` takes the live copy; "
                            f"--overwrite {name} puts this one online.")
    for name in plan.removed_online:
        if name not in overwrite:
            problems.append(f"{name} was taken off the live site since this folder saw it: delete "
                            f"public/videos/{name} here, or --overwrite {name} to put it back.")
    live_code = live.get("commit")
    if live_code and code.get("commit") and not force_code:
        if not has_commit(live_code):
            problems.append(f"The live site runs code (commit {live_code[:7]}) this checkout doesn't have: update "
                            "it (git pull) first, or --force-code to put this older code online.")
        elif live.get("dirty") and live.get("site_code") != code.get("site_code"):
            problems.append(f"The live site was deployed with uncommitted changes on top of {live_code[:7]}; "
                            "deploying from here replaces them with this checkout's code. Ask whoever deployed "
                            "it to commit and push them, or pass --force-code.")
    if code.get("commit") is None:
        notes.append("This copy of the program isn't a git checkout, so its code can't be checked against the "
                     "live site's.")
    elif code.get("dirty"):
        notes.append("This checkout has uncommitted changes to bruv_app: they go online too.")
    public = site / "public"
    files = [p for p in public.rglob("*") if p.is_file()] if public.is_dir() else []
    count = len(files) + (0 if (public / INDEX).exists() else 1)
    size = sum(p.stat().st_size for p in files)
    if count > file_limit:
        problems.append(f"{count:,} files is more than one free Pages site takes ({file_limit:,}): publish some "
                        "videos to a second site (docs/HOSTED.md), or drop old ones.")
    for p in files:
        if p.stat().st_size > size_limit:
            problems.append(f"{p.relative_to(public).as_posix()} is {p.stat().st_size / 2**20:.1f} MiB, over the "
                            f"{size_limit / 2**20:.0f} MiB a Pages file may be: the site can't take this video yet.")
    gone = [n for n in plan.missing + plan.removed_here if n in drop]
    up = plan.new + plan.changed_here + [n for n in plan.changed_online + plan.conflict + plan.removed_online
                                         if n in overwrite]
    notes.append(f"Going up: {', '.join(sorted(up)) or 'no new or changed videos'}")
    notes.append(f"Unchanged: {', '.join(plan.same) or '-'}")
    if gone:
        notes.append(f"Taken off the site: {', '.join(gone)}")
    notes.append(f"{count:,} files ({count / file_limit:.0%} of the free plan's {file_limit:,}), {size / 1e9:.2f} GB")
    return Check(plan, problems, notes, count, size)


# --- Deploying -------------------------------------------------------------------------


def _npx() -> str:
    found = shutil.which("npx.cmd") or shutil.which("npx")
    if found:
        return found
    if (NODE_DIR / "npx.cmd").exists():
        return str(NODE_DIR / "npx.cmd")
    raise SyncError("Node.js isn't installed (it brings npx, which runs Cloudflare's wrangler): docs/GPU_PC.md")


def wrangler(site: Path, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Cloudflare's wrangler, run from the site folder (where wrangler.toml is)."""
    return subprocess.run([_npx(), "--yes", "wrangler@4", *args], cwd=site, text=True,
                          capture_output=capture)


def write_index(site: Path, here: dict[str, dict], code: dict) -> None:
    path = site / "public" / INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 1, **code, "videos": here}, separators=(",", ":")), encoding="utf-8")


def deploy(site: Path, url: str | None = None, *, yes: bool = False, first: bool = False,
           drop: tuple[str, ...] = (), overwrite: tuple[str, ...] = (), force_code: bool = False,
           run=None, ask=input, log=print) -> bool:
    """Check, ask, deploy, and record what is now live. True when it deployed."""
    from .publish import write_site  # noqa: PLC0415 - the site's own files, from this checkout

    url = site_url(site, url)
    write_site(site)
    code = code_version()
    here = local_videos(site)
    live = live_index(url)
    state = load_state(site)
    result = check(site, here, live, state, code, first=first, drop=drop, overwrite=overwrite,
                   force_code=force_code)
    log(f"Deploying {site / 'public'} to {url}:")
    for note in result.notes:
        log(f"  {note}")
    if result.problems:
        log("Not deployed:")
        for problem in result.problems:
            log(f"  - {problem}")
        return False
    if not yes and ask(f"Put this online at {url} now? [y/N] ").strip().lower() not in ("y", "yes"):
        log("Not deployed.")
        return False
    write_index(site, here, code)
    done = (run or (lambda: wrangler(site, "pages", "deploy", "--branch", "main", "--commit-dirty=true")))()
    if done.returncode != 0:
        log("wrangler didn't finish the deploy (above). Nothing was recorded; run deploy again.")
        return False
    state.update(url=url, videos=_digests(here))
    save_state(site, state)
    try:
        after = live_index(url)
    except (OSError, ValueError):
        after = None
    if after is not None and _digests(after["videos"]) == state["videos"]:
        log(f"Live at {url}: {', '.join(sorted(here)) or 'no videos'}")
    else:
        log(f"Deployed. The live index doesn't show it yet; check {url}/{INDEX} in a minute.")
    return True


def init(site: Path, url: str | None = None, log=print) -> None:
    """A site folder for a PC that hasn't deployed before: the site's own files, wrangler.toml
    with the database's id (wrangler must be signed in), then the live videos."""
    from .publish import write_site  # noqa: PLC0415

    site.mkdir(parents=True, exist_ok=True)
    write_site(site)
    toml = site / "wrangler.toml"
    if not toml.exists():
        example = (site / "wrangler.toml.example").read_text(encoding="utf-8")
        name = next(line.split("=", 1)[1].strip().strip('"') for line in example.splitlines()
                    if line.strip().startswith("database_name"))
        listed = wrangler(site, "d1", "list", "--json", capture=True)
        try:
            databases = json.loads(listed.stdout)
        except json.JSONDecodeError:
            raise SyncError("wrangler couldn't list the databases: sign it in first with `npx wrangler login`\n"
                            + (listed.stderr or listed.stdout)[-800:]) from None
        ids = [d["uuid"] for d in databases if d.get("name") == name]
        if not ids:
            raise SyncError(f"This Cloudflare account has no database called {name}: sign wrangler in to the "
                            "account the site runs on (`npx wrangler logout`, then `npx wrangler login`)")
        toml.write_text(example.replace("PASTE-THE-ID-HERE", ids[0]), encoding="utf-8")
        log(f"wrangler.toml written (database {name})")
    mirror(site, url, log=log)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bruv_app.sitesync", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("init", "set up a site folder on a PC that hasn't deployed before, then mirror"),
                       ("mirror", "download the live videos this folder lacks or has older copies of"),
                       ("check", "say what a deploy would do, without deploying"),
                       ("deploy", "check, ask, put the folder online, record what is live")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--site", required=True, help="the site folder (holds public/ and wrangler.toml)")
        p.add_argument("--url", help="the live site's address (default: from wrangler.toml)")
        if name == "mirror":
            p.add_argument("--replace", nargs="+", default=[], metavar="VIDEO",
                           help="take these videos' live copies even where this folder's changed")
        if name in ("check", "deploy"):
            p.add_argument("--first", action="store_true",
                           help="the live site has no index yet: this folder holds every live video")
            p.add_argument("--drop", nargs="+", default=[], metavar="VIDEO", help="take these videos off the site")
            p.add_argument("--overwrite", nargs="+", default=[], metavar="VIDEO",
                           help="put this folder's copies of these videos online over the live ones")
            p.add_argument("--force-code", action="store_true",
                           help="deploy this checkout's code even if the live site's is newer")
        if name == "deploy":
            p.add_argument("--yes", action="store_true", help="don't ask before deploying")
    args = parser.parse_args(argv)
    site = Path(args.site)
    try:
        if args.command == "init":
            init(site, args.url)
        elif args.command == "mirror":
            mirror(site, args.url, tuple(args.replace))
        elif args.command == "check":
            url = site_url(site, args.url)
            result = check(site, local_videos(site), live_index(url), load_state(site),
                           code_version(), first=args.first, drop=tuple(args.drop),
                           overwrite=tuple(args.overwrite), force_code=args.force_code)
            print(f"A deploy of {site / 'public'} to {url} would:")
            for note in result.notes:
                print(f"  {note}")
            for problem in result.problems:
                print(f"  - {problem}")
            print("Ready to deploy." if not result.problems else "Not ready (above).")
            sys.exit(0 if not result.problems else 2)
        elif not deploy(site, args.url, yes=args.yes, first=args.first, drop=tuple(args.drop),
                        overwrite=tuple(args.overwrite), force_code=args.force_code):
            sys.exit(2)
    except (SyncError, OSError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
