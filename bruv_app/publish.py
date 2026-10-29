"""Publish a processed project as static files for the hosted review site (the volunteer kit).

The heavy work (finding, tracking and describing fish) is done once, on whatever PC
processed the video. Publishing writes what the review page needs to a folder that any
static host can serve, so volunteers review in a browser with nothing installed and the
PC can be switched off (docs/SPEC.md §16):

- the review page's own answers, made by the review server's own code: every card's
  entry in the card list and its detail (/api/fish, /api/fish/{id}), the summary and the
  still-object sample. The hosted site answers the same requests from these (hosted/runtime.js),
  so the page itself is unchanged: one page, one API;
- one picture file per card ("sheet"): the close-ups the page shows (the sightings strip,
  the best view, each piece's best view, the tracker's closest calls) and the full scene
  for up to FRAMES_PER_CARD moments, packed together. The free plan allows 20,000 files
  per site; one file per card keeps a 44-minute video to about 1,100 files;
- the numbers the browser needs to redo the light work when a volunteer splits a card:
  each crop's and piece's description (BioCLIP embedding, float16) and the names'
  descriptions, so a new card's guesses, warnings and look-alikes come out as the server
  would make them.

Tiny fish (never `tiny_fish_px` across) and cards outside the review window are left out.

With `follow`, it also writes what following a fish needs (follow.py): every sampled frame,
FOLLOW_STRIP to a picture file, and every box in them. A follow can step anywhere in the
video, because the fish wanders from card to card. It costs about 350 MB for a 44-minute
video, so it's for the videos people will follow fish in.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import imaging
from .families import display_name, family_entry, family_of
from .lookalike import LOOKALIKE, card_looks, neighbours
from .pipeline import species_labels, text_embedding_path
from .store import Project, from_blob
from .views import view_score

FORMAT = 1
SHEET_WIDTH = 1600  # px; tiles are packed in rows up to this width
CROP_MAX = 400  # px; a close-up's longer side is capped at this (the page shows them smaller)
FRAME_WIDTH = 640  # px; full scenes are shrunk to this width (the page shows them at about 46% of the screen height)
FRAMES_PER_CARD = 8  # full scenes per card: best view, pieces, closest calls, then moments along the strip
QUALITY = 80  # WebP quality of the sheets
FOLLOW_STRIP = 20  # frames per picture file for following a fish (5 s at 4 fps)
FOLLOW_COLS = 4  # those frames in rows of 4: 2,560 px wide, well inside WebP's 16,383

STATIC_DIR = Path(__file__).parent / "static"
HOSTED_DIR = Path(__file__).parent / "hosted"


@dataclass
class Sheet:
    """Tiles packed in rows; each tile's place is (x, y, w, h) in the sheet."""

    width: int = SHEET_WIDTH
    tiles: list[tuple[str, int, Image.Image]] = field(default_factory=list)  # (kind, id, image)

    def pack(self) -> tuple[Image.Image, dict[str, dict[int, list[int]]]]:
        order = sorted(range(len(self.tiles)), key=lambda i: -self.tiles[i][2].height)
        places: dict[str, dict[int, list[int]]] = {"crops": {}, "frames": {}}
        x = y = row_h = 0
        spots = []
        for i in order:
            kind, key, im = self.tiles[i]
            if x + im.width > self.width and x > 0:
                x, y, row_h = 0, y + row_h, 0
            spots.append((i, x, y))
            places[kind][key] = [x, y, im.width, im.height]
            x += im.width
            row_h = max(row_h, im.height)
        sheet = Image.new("RGB", (self.width, max(1, y + row_h)), (0, 0, 0))
        for i, sx, sy in spots:
            sheet.paste(self.tiles[i][2], (sx, sy))
        return sheet, places


def _endpoints(app) -> dict[tuple[str, str], object]:
    """The review server's endpoint functions by (path, method): publish calls the same code
    the local server runs, so the hosted answers have exactly its shape."""
    out = {}
    for route in app.routes:
        for method in getattr(route, "methods", None) or ():
            out[(route.path, method)] = route.endpoint
    return out


def _shrink(im: Image.Image, longest: int) -> Image.Image:
    if max(im.size) <= longest:
        return im
    scale = longest / max(im.size)
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)


class _Frames:
    """Saved frames, loaded once each (cards share frames)."""

    def __init__(self, project: Project, keep: int = 64):
        self.project = project
        self.keep = keep
        self.cache: dict[int, np.ndarray] = {}

    def get(self, frame_id: int) -> np.ndarray:
        if frame_id not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            row = self.project.frame(frame_id)
            self.cache[frame_id] = imaging.load_rgb(str(self.project.root / row.path))
        return self.cache[frame_id]


def published_cards(project: Project) -> list[int]:
    """Cards the hosted review shows: not tiny, inside the review window, with sightings."""
    settings = project.load_settings()
    sizes = project.fish_sizes()
    outside = project.outside_window()
    return [
        f["id"] for f in project.all_fish()
        if f["id"] not in outside and sizes.get(f["id"], 0.0) >= settings.tiny_fish_px
    ]


def _frames_for(detail: dict, by_det: dict[int, dict]) -> list[int]:
    """Up to FRAMES_PER_CARD frames to publish whole: the best view, each piece's, the
    closest calls, then moments spread along the sightings strip."""
    wanted: list[int] = []

    def add(frame_id):
        if frame_id is not None and frame_id not in wanted and len(wanted) < FRAMES_PER_CARD:
            wanted.append(frame_id)

    if detail.get("best"):
        add(detail["best"]["frame_id"])
    for p in detail.get("pieces", []):
        d = by_det.get(p["best_det"])
        add(d["frame_id"] if d else None)
    for c in detail.get("closest_calls", []):
        add(c["frame_id"])
    strip = detail.get("filmstrip", [])
    free = FRAMES_PER_CARD - len(wanted)
    if strip and free > 0:
        for k in range(free):
            add(strip[min(len(strip) - 1, round(k * (len(strip) - 1) / max(free - 1, 1)))]["frame_id"])
    return wanted


def _crops_for(detail: dict) -> list[int]:
    dets = [s["det_id"] for s in detail.get("filmstrip", [])]
    if detail.get("best"):
        dets.append(detail["best"]["det_id"])
    dets += [p["best_det"] for p in detail.get("pieces", [])]
    dets += [c["det_id"] for c in detail.get("closest_calls", [])]
    return list(dict.fromkeys(d for d in dets if d is not None))


def _publish_follow(project: Project, folder: Path, progress=None) -> dict:
    """Every sampled frame, packed FOLLOW_STRIP to a WebP strip, and every box in each frame
    as follow.frame_boxes lists it (highest score first): what the hosted follow page steps
    through (hosted/core.js). Returns the manifest's "follow" entry."""
    from .follow import frame_boxes  # noqa: PLC0415

    out = folder / "follow"
    out.mkdir()
    frames = project.db.execute("SELECT id, t, path, width, height FROM frames ORDER BY seq").fetchall()
    if not frames:
        raise ValueError("No frames to publish for following a fish")
    w = FRAME_WIDTH
    h = round(frames[0]["height"] * FRAME_WIDTH / frames[0]["width"])
    strips = 0
    for k in range(0, len(frames), FOLLOW_STRIP):
        chunk = frames[k:k + FOLLOW_STRIP]
        strip = Image.new("RGB", (FOLLOW_COLS * w, -(-len(chunk) // FOLLOW_COLS) * h), (0, 0, 0))
        for i, fr in enumerate(chunk):
            im = Image.fromarray(imaging.load_rgb(str(project.root / fr["path"]))).resize((w, h), Image.LANCZOS)
            strip.paste(im, ((i % FOLLOW_COLS) * w, (i // FOLLOW_COLS) * h))
        strip.save(out / f"{strips}.webp", "WEBP", quality=QUALITY, method=4)
        strips += 1
        if progress and strips % 25 == 0:
            progress("frames", k, len(frames))
    # Times exactly as stored: next_step compares them with gone_s to the microsecond.
    (out / "frames.json").write_text(json.dumps({"ids": [f["id"] for f in frames], "t": [f["t"] for f in frames]},
                                                separators=(",", ":")), encoding="utf-8")
    # Each frame's boxes: [sighting, x1, y1, x2, y2, score, card, faint]. The card is the one
    # it's on now (none for faint ones); the browser keeps it up to date after fixes.
    boxes = [
        [[b["det"], round(b["x1"], 6), round(b["y1"], 6), round(b["x2"], 6), round(b["y2"], 6), b["score"], b["card"],
          int(b["faint"])] for b in frame_boxes(project, f["id"], 0)]
        for f in frames
    ]
    (out / "boxes.json").write_text(json.dumps({"boxes": boxes}, separators=(",", ":")), encoding="utf-8")
    return {"frames": "follow/frames.json", "boxes": "follow/boxes.json", "strip": FOLLOW_STRIP, "cols": FOLLOW_COLS,
            "w": w, "h": h, "files": strips}


def publish(project_root: Path, out: Path, key: str, progress=None, trial: str | None = None,
            follow: bool = False) -> dict:
    """Write one video's data under <site>/public/videos/<key>/ and return a short report.
    With `trial`, the cards of that rule trial (trials.py) instead of the project's own; with
    `follow`, what following a fish needs too (_publish_follow)."""
    from .server import create_app  # noqa: PLC0415 - the server's own code makes the answers
    from .trials import open_trial  # noqa: PLC0415

    if not key or not all(c.isalnum() or c in "-_" for c in key):
        raise ValueError("A video key is letters, digits, - and _ only (it becomes part of the address)")
    if not (Path(project_root) / "project.db").exists():
        raise ValueError(f"{project_root} is not a project folder (it has no project.db)")
    project = Project(Path(project_root))
    if trial:
        project = open_trial(project, trial)
    ids = published_cards(project)
    if not ids:
        raise ValueError(f"{project_root} has no cards to publish: process the video first")
    settings = project.load_settings()
    app = create_app(project.root, db=project.db_path.relative_to(project.root).as_posix())
    ep = _endpoints(app)
    folder = out / "public" / "videos" / key
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "sheets").mkdir(parents=True)

    keep = set(ids)
    listing = {c["id"]: c for c in ep[("/api/fish", "GET")]() if c["id"] in keep}
    summary = ep[("/api/summary", "GET")]()
    # The site is public: file names only, not this PC's folders (they hold the user name).
    for k in ("template", "last_export"):
        if summary.get(k):
            summary[k] = Path(str(summary[k])).name
    still = ep[("/api/still", "GET")](limit=60)

    labels = project.get_meta("guess_labels") or []
    text = np.load(text_embedding_path(project)) if labels and text_embedding_path(project).exists() else None
    looks = card_looks(project, settings.crops_per_fish)
    rows = {int(f): i for i, f in enumerate(looks.ids)}
    near = neighbours(looks)

    crop_rows: list[np.ndarray] = []
    piece_rows: list[np.ndarray] = []
    frames = _Frames(project)
    cards = []
    total = len(ids)
    for n, fid in enumerate(ids):
        detail = ep[("/api/fish/{fish_id}", "GET")](fid)
        dets = project.fish_detections(fid)
        by_det = {d["id"]: d for d in dets}
        det_rows = []
        for d in dets:
            emb = from_blob(d["embedding"])
            row = -1
            if emb is not None:
                row = len(crop_rows)
                crop_rows.append(emb)
            det_rows.append([
                d["id"], d["frame_id"], round(d["t"], 3), round(d["x1"], 5), round(d["y1"], 5), round(d["x2"], 5),
                round(d["y2"], 5), round(d["score"], 4), round(d["quality"] or 0.0, 5), d["tracklet_id"],
                round(view_score(d), 6), row,
                None if d["link_margin"] is None else round(d["link_margin"], 4),
                None if d["history_sim"] is None else round(d["history_sim"], 4),
                # Its size in pixels, exactly as store.fish_sizes works it out (not rounded):
                # look-alike groups are led by the biggest fish, and the server's order rests
                # on differences as small as 139.50000000000006 against 139.5.
                max((d["x2"] - d["x1"]) * d["width"], (d["y2"] - d["y1"]) * d["height"]),
            ])
        pieces = []
        for t in project.db.execute("SELECT id, embedding FROM tracklets WHERE fish_id=? ORDER BY start_t, id", (fid,)):
            emb = from_blob(t["embedding"])
            prow = -1
            if emb is not None:
                prow = len(piece_rows)
                piece_rows.append(emb)
            pieces.append([t["id"], prow])

        sheet = Sheet()
        for det_id in _crops_for(detail):
            d = by_det.get(det_id)
            if d is None:
                continue
            part = imaging.crop_box(frames.get(d["frame_id"]), d["x1"], d["y1"], d["x2"], d["y2"], settings.crop_pad)
            sheet.tiles.append(("crops", det_id, _shrink(Image.fromarray(part), CROP_MAX)))
        for frame_id in _frames_for(detail, by_det):
            im = Image.fromarray(frames.get(frame_id))
            sheet.tiles.append(("frames", frame_id, im.resize((FRAME_WIDTH, round(im.height * FRAME_WIDTH / im.width)), Image.LANCZOS)))
        image, places = sheet.pack()
        image.save(folder / "sheets" / f"{fid}.webp", "WEBP", quality=QUALITY, method=4)

        cards.append({
            "id": fid,
            "list": listing.get(fid),
            "detail": detail,
            "dets": det_rows,
            "pieces": pieces,
            "look": rows.get(fid, -1),
            "neighbours": [[c, round(s, 4)] for c, s in near.get(fid, []) if c in keep],
            "sheet": f"sheets/{fid}.webp",
            "crops": places["crops"],
            "frames": places["frames"],
        })
        if progress and n % 25 == 0:
            progress("publish", n, total)

    # Still objects: their close-ups in one more sheet.
    still_sheet = Sheet()
    for s in still["sample"]:
        d = project.db.execute("SELECT x1, y1, x2, y2, frame_id FROM detections WHERE id=?", (s["det_id"],)).fetchone()
        part = imaging.crop_box(frames.get(d["frame_id"]), d["x1"], d["y1"], d["x2"], d["y2"], settings.crop_pad)
        still_sheet.tiles.append(("crops", s["det_id"], _shrink(Image.fromarray(part), CROP_MAX)))
    still_image, still_places = still_sheet.pack()
    still_image.save(folder / "still.webp", "WEBP", quality=QUALITY, method=4)

    # The descriptions, float16, one file: names, then crops, then pieces, then card looks.
    dim = text.shape[1] if text is not None else (looks.emb.shape[1] if looks.emb.size else 0)
    blocks = [
        text if text is not None else np.zeros((0, dim)),
        np.array(crop_rows).reshape(len(crop_rows), dim) if crop_rows else np.zeros((0, dim)),
        np.array(piece_rows).reshape(len(piece_rows), dim) if piece_rows else np.zeros((0, dim)),
        looks.emb.reshape(len(looks.ids), dim) if looks.emb.size else np.zeros((0, dim)),
    ]
    np.concatenate(blocks).astype(np.float16).tofile(folder / "emb.bin")
    followable = _publish_follow(project, folder, progress) if follow else None

    names = species_labels(project)
    name_family = family_of(project, names)
    label_family = family_of(project, labels)
    members: dict[str, list[str]] = {}
    for label, fam in label_family.items():
        members.setdefault(fam, []).append(label)
    name_members: dict[str, list[str]] = {}
    for name, fam in name_family.items():
        name_members.setdefault(fam, []).append(name)

    manifest = {
        "format": FORMAT,
        "video": key,
        "project": project.root.name,
        "deployment": project.get_meta("deployment"),
        "summary": summary,
        "still": {**still, "sheet": "still.webp", "crops": still_places["crops"]},
        "chapters": [{"name": c["name"], "start_s": c["start_s"], "duration_s": c["duration_s"]} for c in project.chapters()],
        "frame_size": list(project.db.execute("SELECT width, height FROM frames LIMIT 1").fetchone() or (960, 540)),
        "settings": {
            "crops_per_fish": settings.crops_per_fish,
            "top_k": settings.top_k,
            "merge_min_similarity": settings.merge_min_similarity,
            "lookalike": LOOKALIKE,
            "crop_pad": settings.crop_pad,
            "piece_sure": 0.5,
            "filmstrip_max": 60,
        },
        "labels": labels,
        "logit_scale": float(project.get_meta("logit_scale", 100.0)),
        "library_weight": float(project.get_meta("library_weight") or 0.0),
        "label_family": label_family,
        "families": {k: {"name": display_name(k, family_entry(v, k)), "entry": family_entry(v, k)} for k, v in members.items()},
        "names": names,
        "name_family": name_family,
        "name_entries": {k: family_entry(v, k) for k, v in name_members.items()},
        "emb": {"file": "emb.bin", "dtype": "float16", "dim": int(dim), "labels": len(blocks[0]),
                "crops": len(blocks[1]), "pieces": len(blocks[2]), "looks": len(blocks[3])},
        "look_ids": [int(x) for x in looks.ids],
        "cards": "cards.json",
        "follow": followable,
    }
    (folder / "cards.json").write_text(json.dumps({"cards": cards}, separators=(",", ":")), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    # So `pull` can tell this project from another video's (copies made from now on carry it).
    project.set_meta("published", {"video": key, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "cards": len(ids)})
    if progress:
        progress("publish", total, total)
    files = sum(1 for p in folder.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    return {"video": key, "cards": len(ids), "files": files, "bytes": size}


def _hosted_page(name: str, assets: tuple[str, ...]) -> str:
    """A page of the review app as the site serves it: unchanged, with the same cache-busting
    stamps the local server adds, and the in-page server loaded first (hosted/runtime.js)."""
    page = (STATIC_DIR / name).read_text(encoding="utf-8")
    for asset in assets:
        page = page.replace(f"/static/{asset}", f"/static/{asset}?v={(STATIC_DIR / asset).stat().st_mtime_ns}")
    stamp = {js: (HOSTED_DIR / "public" / "hosted" / js).stat().st_mtime_ns for js in ("core.js", "runtime.js")}
    return page.replace("<head>", '<head>\n'
                        f'  <script src="/hosted/core.js?v={stamp["core.js"]}"></script>\n'
                        f'  <script src="/hosted/runtime.js?v={stamp["runtime.js"]}"></script>', 1)


def write_site(out: Path) -> None:
    """The site's own files: the review page and the follow page (Pages serves follow.html at
    /follow, as the local server does), the hosted runtime, and the Cloudflare functions and
    schema."""
    public = out / "public"
    public.mkdir(parents=True, exist_ok=True)
    (public / "index.html").write_text(_hosted_page("index.html", ("app.js", "styles.css")), encoding="utf-8")
    (public / "follow.html").write_text(_hosted_page("follow.html", ("follow.js", "styles.css")), encoding="utf-8")
    shutil.copytree(STATIC_DIR, public / "static", dirs_exist_ok=True)
    for stale in ("sw.js", "hosted/guard.js"):  # an earlier design's files
        (public / stale).unlink(missing_ok=True)
    for item in (HOSTED_DIR / "public").iterdir():
        target = public / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    shutil.copytree(HOSTED_DIR / "functions", out / "functions", dirs_exist_ok=True)
    for name in ("schema.sql", "wrangler.toml.example"):
        shutil.copy2(HOSTED_DIR / name, out / name)
    (public / "videos.json").unlink(missing_ok=True)  # an earlier version's listing, never read


def site_report(out: Path) -> dict:
    """Files and bytes the site holds, against the free plan's 20,000-file limit."""
    public = out / "public"
    files = [p for p in public.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files), "limit": 20_000}
