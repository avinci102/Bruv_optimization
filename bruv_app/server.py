"""Local review server: FastAPI + plain HTML/JS, opened in the default browser."""

from __future__ import annotations

import io
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import imaging
from .diagnostics import card_flags, closest_calls, pieces
from .equipment import best_match, mark_equipment, rig_candidates
from .export import export_project
from .families import family_of, rank_families
from .lookalike import answered_like, card_looks, groups, unanswered_like
from .pipeline import EARLIER_KEY, card_scores, extra_species, load_timeline, refresh_fish_guesses, species_labels
from .store import REVIEWED, STATUSES, Project

STATIC_DIR = Path(__file__).parent / "static"
FILMSTRIP_MAX = 60


class LabelIn(BaseModel):
    status: str
    species: str | None = None
    type: str | None = None
    sex: str | None = None
    behavior: str | None = None
    comment: str | None = None
    # The reviewer has checked the card's warnings (diagnostics.py) and it is one fish.
    confirm: bool = False


class EquipmentIn(BaseModel):
    confirm: bool = False


class BatchIn(BaseModel):
    ids: list[int]


class UndoIn(BaseModel):
    batch: str


class SplitIn(BaseModel):
    from_t: float


class MergeIn(BaseModel):
    other_id: int


class MoveIn(BaseModel):
    det_ids: list[int]


class LookalikeIn(BaseModel):
    source: int  # the card whose look-alikes these are
    ids: list[int]
    status: str
    species: str | None = None


def _box(d) -> dict:
    return {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"], "score": round(d["score"], 3)}


def create_app(project_root: Path) -> FastAPI:
    project = Project(Path(project_root))
    settings = project.load_settings()
    lock = threading.Lock()  # one SQLite connection, so serialise access
    app = FastAPI(title="BRUV Fish ID")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def always_current_page(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # after an update, a reload gets it
        return response

    def get_fish(fish_id: int):
        row = project.fish(fish_id)
        if row is None:
            raise HTTPException(404, "No such fish")
        return row

    cache: dict = {}

    def looks():
        """Every card's look (lookalike.py), remade when the cards change; answers are read
        fresh each time."""
        stamp = tuple(project.db.execute(
            "SELECT COUNT(*), MAX(id), (SELECT MAX(id) FROM corrections) FROM fish"
        ).fetchone())
        if cache.get("stamp") != stamp:
            cache.update(stamp=stamp, looks=card_looks(project, settings.crops_per_fish), groups=None)
        current = cache["looks"]
        for f in project.db.execute("SELECT id, status, species FROM fish"):
            current.status[f["id"]], current.answer[f["id"]] = f["status"], f["species"]
        return current

    def earlier_answer(dets) -> dict | None:
        """What a card's sightings were answered before the cards were last rebuilt, when
        some were: the answer most of them had, how many had it, and how many answers."""
        if "earlier" not in cache:
            cache["earlier"] = project.get_meta(EARLIER_KEY) or {}
        held = Counter()
        for d in dets:
            a = cache["earlier"].get(str(d["id"]))
            if a:
                held[(a["status"], a["species"])] += 1
        if not held:
            return None
        (status, species), n = held.most_common(1)[0]
        return {"status": status, "species": species, "sightings": n, "of": len(dets), "answers": len(held)}

    def look_groups() -> dict[int, int]:
        """Look-alike groups, biggest fish leading (for Sort: look-alikes together)."""
        current = looks()
        if cache.get("groups") is None:
            sizes = project.fish_sizes()
            order = sorted(sizes, key=lambda f: -sizes[f])
            cache["groups"] = groups(current, order)
        return cache["groups"]

    def family_view(fish_id: int, top: int = 3) -> list[dict]:
        """The model's top families for a card (families.py), each with its names by score
        within the family, and the name that records just the family, if any."""
        scored = card_scores(project, fish_id, settings)
        if scored is None:
            return []
        labels, probs = scored
        ranked = rank_families(labels, probs, family_of(project, labels))
        return [
            {"key": f.key, "name": f.name, "p": round(f.p, 4), "best": round(f.best, 4), "entry": f.entry,
             "members": [[name, round(p, 4)] for name, p in f.members]}
            for f in ranked[:top]
        ]

    @app.get("/")
    def index():
        # The script and styles get a new address whenever they change, so a browser never
        # runs an old copy from its cache after an update.
        page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "styles.css"):
            page = page.replace(f"/static/{name}", f"/static/{name}?v={(STATIC_DIR / name).stat().st_mtime_ns}")
        return HTMLResponse(page)

    @app.get("/api/summary")
    def summary():
        with lock:
            fish = project.all_fish()
            counts = {s: 0 for s in STATUSES}
            for f in fish:
                counts[f["status"]] += 1
            return {
                "project": project.root.name,
                "total": len(fish),
                "reviewed": sum(counts[s] for s in REVIEWED),
                "counts": counts,
                "dropdowns": project.get_meta("dropdowns") or {},
                "dependent": project.get_meta("dependent") or {},
                "extra_species": extra_species(project),
                "classifier": project.get_meta("classifier"),
                "tiny_fish_px": settings.tiny_fish_px,
                "template": project.get_meta("template"),
                "last_export": project.get_meta("last_export"),
                "frames": project.frame_count(),
                # Boxes hidden as still objects (still.py) that would otherwise be tracked.
                "still_hidden": len(project.still_detections(settings, settings.det_threshold)),
            }

    @app.get("/api/fish")
    def list_fish():
        with lock:
            timeline = load_timeline(project)
            sizes = project.fish_sizes()
            group = look_groups()
            out = []
            for f in project.all_fish():
                clip, local = timeline.locate(f["start_t"])
                guesses = json.loads(f["guesses"] or "[]")
                out.append(
                    {
                        "id": f["id"],
                        "start_t": f["start_t"],
                        "end_t": f["end_t"],
                        "clip": clip,
                        "local_t": local,
                        "status": f["status"],
                        "species": f["species"],
                        "top": guesses[0][0] if guesses else None,
                        "guesses": [g[0] for g in guesses[:3]],
                        "max_px": round(sizes.get(f["id"], 0.0), 1),
                        "group": group.get(f["id"]),  # its look-alike group (lookalike.py)
                    }
                )
            return out

    @app.get("/api/fish/{fish_id}")
    def fish_detail(fish_id: int):
        with lock:
            f = get_fish(fish_id)
            timeline = load_timeline(project)
            dets = project.fish_detections(fish_id)
            best = next((d for d in dets if d["id"] == f["best_det"]), dets[0] if dets else None)
            step = max(1, -(-len(dets) // FILMSTRIP_MAX))
            strip = []
            for i in range(0, len(dets), step):
                d = dets[i]
                clip, local = timeline.locate(d["t"])
                strip.append(
                    {
                        "det_id": d["id"],
                        "frame_id": d["frame_id"],
                        "t": d["t"],
                        "clip": clip,
                        "local_t": local,
                        "tracklet": d["tracklet_id"],
                        "box": _box(d),
                        # A long card shows every n-th sighting: this one stands for those
                        # up to the next, so ticking it moves them all.
                        "covers": [x["id"] for x in dets[i : i + step]],
                    }
                )
            clip_in, t_in = timeline.locate(f["start_t"])
            clip_out, t_out = timeline.locate(f["end_t"])
            parts = pieces(project, fish_id)
            review = project.last_review(fish_id)
            rig_match = (
                best_match(dets, project.rig_refs(), settings.merge_min_colour)
                if f["status"] in ("pending", "skipped") else None
            )
            piece_view = []
            for p in parts:
                clip, local = timeline.locate(p.start_t)
                piece_view.append({
                    "tracklet": p.tracklet, "clip": clip, "local_t": local, "seconds": p.end_t - p.start_t,
                    "detections": p.detections, "best_det": p.best_det, "guesses": p.guesses,
                })
            calls = []
            for c in closest_calls(project, fish_id):
                clip, local = timeline.locate(c["t"])
                calls.append({"det_id": c["det_id"], "frame_id": c["frame_id"], "t": c["t"], "clip": clip,
                              "local_t": local, "box": _box(c), "margin": round(c["link_margin"], 3),
                              "history": None if c["history_sim"] is None else round(c["history_sim"], 3)})
            current = looks()
            best_view = None
            if best is not None:
                best_clip, best_local = timeline.locate(best["t"])
                best_view = {
                    "det_id": best["id"],
                    "frame_id": best["frame_id"],
                    "t": best["t"],
                    "clip": best_clip,
                    "local_t": best_local,
                    "box": _box(best),
                }
            return {
                "id": f["id"],
                "status": f["status"],
                "species": f["species"],
                "type": f["type"],
                "sex": f["sex"],
                "behavior": f["behavior"],
                "comment": f["comment"],
                "guesses": json.loads(f["guesses"] or "[]"),
                "clip_in": clip_in,
                "time_in": t_in,
                "clip_out": clip_out,
                "time_out": t_out,
                "n_detections": len(dets),
                "tracklets": project.fish_tracklets(fish_id),
                "best": best_view,
                "filmstrip": strip,
                "pieces": piece_view,
                "flags": [{"kind": x.kind, "text": x.text} for x in card_flags(project, fish_id, settings.merge_min_similarity, parts)],
                "closest_calls": calls,
                "families": family_view(fish_id),
                "equipment": bool(review and review["details"] and json.loads(review["details"]).get("equipment")),
                # Your answer on the card that looks most like this one, and how many unanswered
                # cards look like it (lookalike.py).
                "answered_like": answered_like(current, fish_id),
                "lookalikes": len(unanswered_like(current, fish_id)),
                # Some of its sightings were answered before the cards were rebuilt.
                "earlier": earlier_answer(dets) if f["status"] in ("pending", "skipped") else None,
                "rig": None if rig_match is None else {
                    "card": rig_match.ref_card, "overlap": round(rig_match.overlap, 2),
                    "colour": None if rig_match.colour is None else round(rig_match.colour, 2),
                    "odd": rig_match.odd,
                },
            }

    @app.post("/api/fish/{fish_id}/label")
    def label(fish_id: int, body: LabelIn):
        with lock:
            get_fish(fish_id)
            species_list = species_labels(project)  # the template's list plus the extra names
            if body.status == "species" and species_list and body.species not in species_list:
                raise HTTPException(400, "Pick a species from the list")
            # One answer for every piece of a flagged card only once the reviewer confirms
            # they're one fish; the confirmation is kept with the answer.
            flags = card_flags(project, fish_id, settings.merge_min_similarity) if body.status in REVIEWED else []
            if flags and not body.confirm:
                raise HTTPException(
                    409, "Check this card first: " + " ".join(x.text for x in flags)
                    + " Answer again to confirm it's one fish, or use Multiple fish.",
                )
            details = {"flags": [x.kind for x in flags], "confirmed": True} if flags else None
            try:
                project.label_fish(fish_id, body.status, details, **body.model_dump(exclude={"status", "confirm"}))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"ok": True}

    # --- equipment (equipment.py): marked by the reviewer, look-alikes queued, never auto-removed

    @app.post("/api/fish/{fish_id}/equipment")
    def equipment(fish_id: int, body: EquipmentIn):
        with lock:
            get_fish(fish_id)
            flags = card_flags(project, fish_id, settings.merge_min_similarity)
            if flags and not body.confirm:
                raise HTTPException(409, "Check this card first: " + " ".join(x.text for x in flags))
            return {"ref": mark_equipment(project, fish_id)}

    @app.get("/api/rig")
    def rig():
        with lock:
            found = rig_candidates(project, settings.merge_min_colour)
            return {
                "refs": [{"id": r["id"], "card": r["fish_id"]} for r in project.rig_refs()],
                "candidates": {
                    fid: {"ref": m.ref, "card": m.ref_card, "overlap": round(m.overlap, 2),
                          "colour": None if m.colour is None else round(m.colour, 2), "odd": m.odd}
                    for fid, m in found.items()
                },
            }

    @app.post("/api/rig/mark")
    def rig_mark(body: BatchIn):
        """Answer queued cards as equipment together. A card with sightings unlike the rig (a
        fish joined on?) or with warnings is left to check one by one."""
        with lock:
            found = rig_candidates(project, settings.merge_min_colour)
            batch = datetime.now(timezone.utc).strftime("rig-%Y%m%dT%H%M%S%f")
            marked, left = [], []
            for fid in body.ids:
                if fid not in found:
                    continue
                if found[fid].odd or card_flags(project, fid, settings.merge_min_similarity):
                    left.append(fid)
                    continue
                mark_equipment(project, fid, batch, found[fid])
                marked.append(fid)
            return {"batch": batch, "marked": marked, "left": left}

    @app.post("/api/rig/undo")
    def rig_undo(body: UndoIn):
        with lock:
            ids = project.reviews_in_batch(body.batch)
            for fid in ids:
                project.label_fish(fid, "pending")
            return {"undone": ids}

    @app.post("/api/rig/{ref_id}/forget")
    def rig_forget(ref_id: int):
        with lock:
            project.forget_rig_ref(ref_id)
            return {"ok": True}

    # --- look-alikes (lookalike.py): answered together only from their pictures, with undo

    @app.get("/api/fish/{fish_id}/lookalikes")
    def lookalikes(fish_id: int, limit: int = 120):
        with lock:
            get_fish(fish_id)
            timeline = load_timeline(project)
            out = []
            for fid, sim in unanswered_like(looks(), fish_id)[:limit]:
                f = project.fish(fid)
                clip, local = timeline.locate(f["start_t"])
                out.append({"id": fid, "similarity": round(sim, 3), "best_det": f["best_det"], "clip": clip,
                            "local_t": local, "seconds": f["end_t"] - f["start_t"],
                            "flagged": bool(card_flags(project, fid, settings.merge_min_similarity))})
            return out

    @app.post("/api/lookalikes/answer")
    def lookalikes_answer(body: LookalikeIn):
        """Give the cards you left ticked the same answer. Cards with warnings, or answered
        meanwhile, are left for you; the batch can be undone."""
        with lock:
            get_fish(body.source)
            if body.status not in REVIEWED:
                raise HTTPException(400, "Pick an answer for them")
            species_list = species_labels(project)
            if body.status == "species" and species_list and body.species not in species_list:
                raise HTTPException(400, "Pick a species from the list")
            alike = dict(unanswered_like(looks(), body.source))
            batch = datetime.now(timezone.utc).strftime("like-%Y%m%dT%H%M%S%f")
            equipment = body.status == "not_fish" and bool(project.answer_details(body.source).get("equipment"))
            answered, left = [], []
            for fid in body.ids:
                if fid not in alike or card_flags(project, fid, settings.merge_min_similarity):
                    left.append(fid)
                    continue
                details = {"batch": batch, "like": body.source, "similarity": round(alike[fid], 3)}
                if equipment:
                    details["equipment"] = True
                project.label_fish(fid, body.status, details, species=body.species)
                answered.append(fid)
            return {"batch": batch, "answered": answered, "left": left}

    @app.post("/api/batch/undo")
    def batch_undo(body: UndoIn):
        """Put a batch's cards (look-alikes or equipment) back to pending."""
        return rig_undo(body)

    @app.get("/api/still")
    def still(limit: int = 60):
        """Boxes hidden as still objects, a sample spread over the video, for checking."""
        with lock:
            rows = project.still_detections(settings, settings.det_threshold)
            timeline = load_timeline(project)
            step = max(1, -(-len(rows) // limit))
            sample = []
            for d in rows[::step]:
                clip, local = timeline.locate(d["t"])
                sample.append({"det_id": d["id"], "frame_id": d["frame_id"], "clip": clip, "local_t": local,
                               "change": round(d["change"], 3), "box": _box(d)})
            return {"count": len(rows), "sample": sample}

    @app.post("/api/fish/{fish_id}/split")
    def split(fish_id: int, body: SplitIn):
        with lock:
            get_fish(fish_id)
            try:
                new_id = project.split_fish(fish_id, body.from_t)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            refresh_fish_guesses(project, fish_id, settings)
            refresh_fish_guesses(project, new_id, settings)
            project.commit()
            return {"ok": True, "new_id": new_id}

    @app.post("/api/fish/{fish_id}/move")
    def move(fish_id: int, body: MoveIn):
        """Ticked sightings go to a new card (e.g. the other fish, after two fish crossed)."""
        with lock:
            get_fish(fish_id)
            try:
                new_id = project.move_to_new_fish(fish_id, body.det_ids)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            refresh_fish_guesses(project, fish_id, settings)
            refresh_fish_guesses(project, new_id, settings)
            project.commit()
            return {"ok": True, "new_id": new_id}

    @app.post("/api/fish/{fish_id}/unmerge")
    def unmerge(fish_id: int):
        """The card mixes different fish: each tracked piece becomes its own card."""
        with lock:
            get_fish(fish_id)
            try:
                new_ids = project.unmerge_fish(fish_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            for fid in (fish_id, *new_ids):
                refresh_fish_guesses(project, fid, settings)
            project.commit()
            return {"ok": True, "new_ids": new_ids}

    @app.post("/api/fish/{fish_id}/merge")
    def merge(fish_id: int, body: MergeIn):
        with lock:
            get_fish(fish_id)
            get_fish(body.other_id)
            try:
                project.merge_fish(fish_id, body.other_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            refresh_fish_guesses(project, fish_id, settings)
            project.commit()
            return {"ok": True}

    @app.get("/frame/{frame_id}")
    def frame(frame_id: int):
        with lock:
            row = project.frame(frame_id)
        if row is None:
            raise HTTPException(404, "No such frame")
        return FileResponse(project.root / row.path, media_type="image/jpeg")

    @app.get("/crop/{det_id}")
    def crop(det_id: int):
        with lock:
            d = project.db.execute(
                "SELECT d.*, f.path FROM detections d JOIN frames f ON f.id=d.frame_id WHERE d.id=?",
                (det_id,),
            ).fetchone()
        if d is None:
            raise HTTPException(404, "No such detection")
        image = imaging.load_rgb(str(project.root / d["path"]))
        part = imaging.crop_box(image, d["x1"], d["y1"], d["x2"], d["y2"], settings.crop_pad)
        buf = io.BytesIO()
        Image.fromarray(part).save(buf, "JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.post("/api/export")
    def export():
        with lock:
            try:
                path = export_project(project)
            except RuntimeError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "path": str(path)}

    return app
