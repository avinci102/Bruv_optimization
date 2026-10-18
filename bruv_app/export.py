"""Turn reviewed fish into record rows, a MaxN tab and an app_details tab in a template copy."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from . import imaging
from .metrics import UNKNOWN, FishLabel, compute_maxn, compute_sightings, per_frame_counts
from .pipeline import extra_species, load_timeline
from .store import Project
from .template import export_workbook, read_template, seconds_to_time

MAXN_HEADER = [
    "species",
    "maxn",
    "clip_maxn",
    "time_maxn",
    "clip_first_seen",
    "time_first_seen",
    "first_arrival_s",
]

DETAILS_HEADER = [
    "fish_id",
    "record_row",
    "status",
    "user_species",
    "guess_1",
    "p_1",
    "guess_2",
    "p_2",
    "guess_3",
    "p_3",
    "agrees_top1",
    "in_top3",
    "clip_in",
    "time_in",
    "clip_out",
    "time_out",
    "n_detections",
    "rel_size_pct",
    "tracklets",
    "crop_path",
    "classifier",
    "reviewed_at",
]


CORRECTIONS_HEADER = ["at", "action", "card", "details"]


def _most_common(values) -> str | None:
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else None


def export_project(project: Project, out_path: Path | None = None) -> Path:
    settings = project.load_settings()
    template = project.get_meta("template")
    if not template:
        raise RuntimeError("This project has no template; create it with --template.")
    info = read_template(Path(template))
    timeline = load_timeline(project)

    fish_rows = project.all_fish()
    labels: dict[int, str] = {}
    fish_labels: list[FishLabel] = []
    for f in fish_rows:
        if f["status"] == "species":
            key = f["species"]
        elif f["status"] == "unknown":
            key = UNKNOWN
        else:
            continue
        labels[f["id"]] = key
        fish_labels.append(FishLabel(f["id"], key, f["start_t"], f["end_t"]))

    fish_dets = {f["id"]: project.fish_detections(f["id"]) for f in fish_rows}
    detections = [(d["t"], fid) for fid, dets in fish_dets.items() if fid in labels for d in dets]
    counts = per_frame_counts(detections, labels)
    maxn = compute_maxn(counts)
    sightings = compute_sightings(fish_labels, detections, settings.gap_seconds)

    # Names from the app's extra list go into the template as its catch-all ("other ..."),
    # with the name in the comment, so every value stays inside the template's dropdown.
    # So does any other name outside the template's list (e.g. one renamed in the list).
    template_species = set(info.options("species"))
    extras = set(extra_species(project))
    other = next((o for o in info.options("species") if o.strip().lower().startswith("other")), None)

    by_id = {f["id"]: f for f in fish_rows}
    record_rows = []
    for s in sightings:
        members = [by_id[i] for i in s.fish_ids]
        clip_in, t_in = timeline.locate(s.t_in)
        clip_out, t_out = timeline.locate(s.t_out)
        comments = list(dict.fromkeys(m["comment"] for m in members if m["comment"]))
        species = s.species
        if s.species == UNKNOWN:
            comments.insert(0, "Unidentified fish")
            species = None
        elif s.species in extras or (template_species and s.species not in template_species):
            comments.insert(0, s.species)
            species = other
        record_rows.append(
            {
                "clip_in": clip_in,
                "time_in": seconds_to_time(t_in),
                "clip_out": clip_out,
                "time_out": seconds_to_time(t_out),
                "type": _most_common(m["type"] for m in members),
                "species": species,
                "count": s.count,
                "sex": _most_common(m["sex"] for m in members),
                "behavior": _most_common(m["behavior"] for m in members),
                "comment": "; ".join(comments) or None,
            }
        )

    maxn_rows = []
    for m in maxn:
        clip_m, t_m = timeline.locate(m.t_maxn)
        clip_f, t_f = timeline.locate(m.first_t)
        maxn_rows.append(
            [m.species, m.maxn, clip_m, seconds_to_time(t_m), clip_f, seconds_to_time(t_f),
             round(m.first_t - settings.start_s, 1)]
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if out_path is None:
        out_path = project.root / "exports" / f"{Path(template).stem}_{project.root.name}_{stamp}.xlsx"
    crops_dir = project.root / "crops"
    classifier = project.get_meta("classifier", "")

    sighting_row = {fid: info.next_row + i for i, s in enumerate(sightings) for fid in s.fish_ids}
    details_rows = []
    for f in fish_rows:
        dets = fish_dets[f["id"]]
        guesses = json.loads(f["guesses"] or "[]")[:3]
        guesses += [[None, None]] * (3 - len(guesses))
        top = [g[0] for g in guesses if g[0]]
        picked = f["species"] if f["status"] == "species" else None
        # Relative size: the box's longer side as a % of frame width (no true length yet).
        sizes = [
            max((d["x2"] - d["x1"]) * d["width"], (d["y2"] - d["y1"]) * d["height"]) / d["width"] * 100
            for d in dets
        ]
        crop_path = None
        best = next((d for d in dets if d["id"] == f["best_det"]), dets[0] if dets else None)
        if best is not None:
            image = imaging.load_rgb(str(project.root / best["path"]))
            crop = imaging.crop_box(image, best["x1"], best["y1"], best["x2"], best["y2"], settings.crop_pad)
            crop_file = crops_dir / f"fish_{f['id']:05d}.jpg"
            imaging.save_jpeg(crop, crop_file)
            crop_path = str(crop_file)
        clip_in, t_in = timeline.locate(f["start_t"])
        clip_out, t_out = timeline.locate(f["end_t"])
        details_rows.append(
            [
                f["id"],
                sighting_row.get(f["id"]),
                f["status"],
                picked,
                guesses[0][0], guesses[0][1],
                guesses[1][0], guesses[1][1],
                guesses[2][0], guesses[2][1],
                None if picked is None or not top else picked == top[0],
                None if picked is None or not top else picked in top,
                clip_in, seconds_to_time(t_in), clip_out, seconds_to_time(t_out),
                len(dets),
                round(float(np.median(sizes)), 1) if sizes else None,
                ",".join(map(str, project.fish_tracklets(f["id"]))),
                crop_path,
                classifier,
                f["reviewed_at"],
            ]
        )

    # The reviewer's fixes to the cards (splits, merges, "multiple fish"): where tracking and
    # the continuity merge go wrong, for tuning them and as training data.
    corrections = [[c["at"], c["action"], c["fish_id"], c["details"]] for c in project.corrections()]
    export_workbook(
        info, out_path, record_rows, MAXN_HEADER, maxn_rows, DETAILS_HEADER, details_rows,
        {"corrections": (CORRECTIONS_HEADER, corrections)},
    )
    project.set_meta("last_export", str(out_path))
    return out_path
