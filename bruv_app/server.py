"""Local review server: FastAPI + plain HTML/JS, opened in the default browser."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import imaging
from .export import export_project
from .pipeline import load_timeline, refresh_fish_guesses
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


class SplitIn(BaseModel):
    from_t: float


class MergeIn(BaseModel):
    other_id: int


def _box(d) -> dict:
    return {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"], "score": round(d["score"], 3)}


def create_app(project_root: Path) -> FastAPI:
    project = Project(Path(project_root))
    settings = project.load_settings()
    lock = threading.Lock()  # one SQLite connection, so serialise access
    app = FastAPI(title="BRUV Fish ID")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_fish(fish_id: int):
        row = project.fish(fish_id)
        if row is None:
            raise HTTPException(404, "No such fish")
        return row

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

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
                "classifier": project.get_meta("classifier"),
                "template": project.get_meta("template"),
                "last_export": project.get_meta("last_export"),
                "frames": project.frame_count(),
            }

    @app.get("/api/fish")
    def list_fish():
        with lock:
            timeline = load_timeline(project)
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
            for d in dets[::step]:
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
                    }
                )
            clip_in, t_in = timeline.locate(f["start_t"])
            clip_out, t_out = timeline.locate(f["end_t"])
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
            }

    @app.post("/api/fish/{fish_id}/label")
    def label(fish_id: int, body: LabelIn):
        with lock:
            get_fish(fish_id)
            species_list = (project.get_meta("dropdowns") or {}).get("species", [])
            if body.status == "species" and species_list and body.species not in species_list:
                raise HTTPException(400, "Pick a species from the template's list")
            try:
                project.label_fish(fish_id, body.status, **body.model_dump(exclude={"status"}))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"ok": True}

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
