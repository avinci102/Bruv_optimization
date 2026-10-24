"""Turn reviewed fish into record rows, a MaxN tab and an app_details tab in a template copy.

Alongside them: how complete the review is (review_status, review_by_minute; until every
card is answered, tiny fish included, the counts are marked provisional), what could count
one fish twice (checks), the reviewer's corrections, and where the numbers came from
(provenance).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import imaging
from .diagnostics import all_flags
from .metrics import (
    UNKNOWN,
    FishLabel,
    Observation,
    compute_maxn,
    compute_sightings,
    double_count_checks,
    per_frame_counts,
)
from .pipeline import extra_species, load_timeline
from .store import REVIEWED, Project
from .template import export_workbook, read_template, seconds_to_time

MAXN_HEADER = [
    "species",
    "maxn",
    "clip_maxn",
    "time_maxn",
    "clip_first_seen",
    "time_first_seen",
    "first_arrival_s",
    "review",  # "complete", or "provisional" while cards are still unanswered
]
FAMILY_MAXN_HEADER = ["family", "maxn", "clip_maxn", "time_maxn", "clip_first_seen", "time_first_seen", "names", "review"]
STATUS_HEADER = ["item", "value"]
BY_MINUTE_HEADER = ["minute", "clip", "from", "detections", "counted", "not_fish", "unreviewed", "unreviewed_tiny"]
CHECKS_HEADER = ["check", "label", "clip", "time", "detections", "cards", "on_maxn_frame"]

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
    "warnings",  # diagnostics.py; "(confirmed one fish)" once the reviewer confirmed them
    "equipment",  # answered "not a fish" as equipment (equipment.py), by hand or from the queue
]


CORRECTIONS_HEADER = ["at", "action", "card", "details"]
PROVENANCE_HEADER = ["item", "value"]


def provenance_rows(project: Project) -> list[list]:
    """Where the numbers came from: the code, the footage by fingerprint, the settings, and
    every stage run with the models it used (store.py runs), so they can be traced and redone."""
    from .provenance import code_revision, digest  # noqa: PLC0415

    settings = asdict(project.load_settings())
    rows = [
        ["exported by", f"bruv_app {code_revision()}"],
        ["project", str(project.root)],
        ["deployment", project.get_meta("deployment") or "footage not fingerprinted"],
    ]
    rows += [
        ["video", f"{v['path']} ({v['size']:,} bytes) {v['fingerprint']}"]
        for v in project.get_meta("video_fingerprints") or []
    ]
    rows.append(["capture", json.dumps(project.get_meta("capture") or "not recorded", sort_keys=True)])
    rows.append(["settings", f"{digest(settings)} {json.dumps(settings, sort_keys=True)}"])
    if project.get_meta("src_index_inferred"):
        rows.append(["source frames", "numbered from frame times (frames saved by an older version)"])
    for r in project.runs():
        rows.append([
            f"run {r['id']}: {r['stage']}",
            f"{r['started_at']} to {r['finished_at'] or 'stopped'}; code {r['code']}; "
            f"settings {digest(json.loads(r['settings']))}; {r['info']}",
        ])
    rows += [["snapshot", str(p)] for p in sorted((project.root / "snapshots").glob("*.db"))]
    return rows


def _most_common(values) -> str | None:
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else None


def _confirmed(project: Project, fish_id: int) -> bool:
    """The card's current answer was given after confirming its warnings."""
    return bool(project.answer_details(fish_id).get("confirmed"))


def export_project(project: Project, out_path: Path | None = None) -> Path:
    settings = project.load_settings()
    template = project.get_meta("template")
    if not template:
        raise RuntimeError("This project has no template; create it with --template.")
    info = read_template(Path(template))
    timeline = load_timeline(project)

    fish_rows = project.all_fish()
    fish_dets = {f["id"]: project.fish_detections(f["id"]) for f in fish_rows}
    # An answer counts for the detections it was given for (its scope), never for ones that
    # joined the card afterwards. Cards answered before scopes were kept count whole.
    scopes = project.label_scopes()
    labels: dict[int, str] = {}
    fish_labels: list[FishLabel] = []
    counted: dict[int, list] = {}
    for f in fish_rows:
        if f["status"] == "species":
            key = f["species"]
        elif f["status"] == "unknown":
            key = UNKNOWN
        else:
            continue
        scope = scopes.get(f["id"])
        dets = [d for d in fish_dets[f["id"]] if scope is None or d["id"] in scope]
        if not dets:
            continue
        labels[f["id"]] = key
        counted[f["id"]] = dets
        fish_labels.append(FishLabel(f["id"], key, dets[0]["t"], dets[-1]["t"]))

    detections = [(d["t"], fid) for fid, dets in counted.items() for d in dets]
    counts = per_frame_counts(detections, labels)
    maxn = compute_maxn(counts)
    sightings = compute_sightings(fish_labels, detections, settings.sighting_gap_s)
    status_rows, minute_rows, complete = review_status(project, settings, fish_rows, fish_dets, counted, scopes)
    observations = [
        Observation(d["t"], d["id"], fid, labels[fid], (d["x1"], d["y1"], d["x2"], d["y2"]))
        for fid, dets in counted.items() for d in dets
    ]
    peaks = {(m.species, m.t_maxn) for m in maxn}
    checks = []
    for c in double_count_checks(observations):
        clip, local = timeline.locate(c.t)
        checks.append([
            c.kind, "unknown fish" if c.key == UNKNOWN else c.key, clip, seconds_to_time(local),
            ", ".join(map(str, c.det_ids)), ", ".join(map(str, sorted(set(c.fish_ids)))), (c.key, c.t) in peaks,
        ])
    status_rows.append(["double-count checks", f"{len(checks)} (see the checks sheet)"])
    warnings = all_flags(project, settings.merge_min_similarity)
    status_of = {f["id"]: f["status"] for f in fish_rows}
    confirmed = {fid for fid in warnings if status_of[fid] in REVIEWED and _confirmed(project, fid)}
    open_warnings = [fid for fid in warnings if fid not in confirmed and status_of[fid] in REVIEWED]
    status_rows.append([
        "cards with warnings (may hold more than one fish)",
        f"{len(warnings)}: {len(confirmed)} answered after confirming one fish, "
        f"{len(open_warnings)} answered without confirming (before warnings existed), "
        f"{len(warnings) - len(confirmed) - len(open_warnings)} not answered",
    ])

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
             round(m.first_t - settings.start_s, 1), "complete" if complete else "provisional"]
        )
    family_rows = family_maxn(project, labels, detections, timeline, complete)

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
                "; ".join(w.kind for w in warnings.get(f["id"], []))
                + (" (confirmed one fish)" if f["id"] in confirmed else ""),
                f["status"] == "not_fish" and bool(project.answer_details(f["id"]).get("equipment")),
            ]
        )

    # The reviewer's fixes to the cards (splits, merges, "multiple fish"): where tracking and
    # the continuity merge go wrong, for tuning them and as training data.
    corrections = [[c["at"], c["action"], c["fish_id"], c["details"]] for c in project.corrections()]
    export_workbook(
        info, out_path, record_rows, MAXN_HEADER, maxn_rows, DETAILS_HEADER, details_rows,
        {
            "MaxN_by_family": (FAMILY_MAXN_HEADER, family_rows),
            "review_status": (STATUS_HEADER, status_rows),
            "review_by_minute": (BY_MINUTE_HEADER, minute_rows),
            "checks": (CHECKS_HEADER, checks),
            "corrections": (CORRECTIONS_HEADER, corrections),
            "provenance": (PROVENANCE_HEADER, provenance_rows(project)),
        },
    )
    project.set_meta("last_export", str(out_path))
    return out_path


def family_maxn(project: Project, labels: dict[int, str], detections, timeline, complete: bool) -> list[list]:
    """MaxN per family (families.py): each frame's fish counted by their family first, then
    the most at once. Answers at species and at family level ("grouper family") count
    together, and never by adding up the species' own MaxN, which can come from different
    moments."""
    from .families import display_name, family_entry, family_of  # noqa: PLC0415

    names = sorted({key for key in labels.values() if key != UNKNOWN})
    family = family_of(project, names)
    members: dict[str, list[str]] = defaultdict(list)
    for name in names:
        members[family[name]].append(name)
    by_card = {fid: family[key] for fid, key in labels.items() if key != UNKNOWN}
    rows = []
    for m in compute_maxn(per_frame_counts(detections, by_card)):
        clip_m, t_m = timeline.locate(m.t_maxn)
        clip_f, t_f = timeline.locate(m.first_t)
        rows.append([
            display_name(m.species, family_entry(members[m.species], m.species)), m.maxn, clip_m, seconds_to_time(t_m),
            clip_f, seconds_to_time(t_f), "; ".join(members[m.species]), "complete" if complete else "provisional",
        ])
    return rows


def review_status(project, settings, fish_rows, fish_dets, counted, scopes) -> tuple[list, list, bool]:
    """How complete the review is: cards and detections by status, tiny fish apart but never
    left out, and the same by minute of video. Complete means every card is answered (tiny
    ones too) and every counted label covers its whole card; until then counts are provisional."""
    timeline = load_timeline(project)
    sizes = project.fish_sizes()
    tiny_px = settings.tiny_fish_px

    def tally(cards):
        status = Counter(f["status"] for f in cards)
        return (
            len(cards), sum(status[s] for s in REVIEWED), status["skipped"], status["pending"],
            sum(len(fish_dets[f["id"]]) for f in cards),
        )

    tiny = [f for f in fish_rows if sizes.get(f["id"], 0.0) < tiny_px]
    big = [f for f in fish_rows if sizes.get(f["id"], 0.0) >= tiny_px]
    unanswered = [f for f in fish_rows if f["status"] in ("pending", "skipped")]
    outside = sum(
        len(fish_dets[f["id"]]) - len(counted.get(f["id"], []))
        for f in fish_rows if f["status"] in ("species", "unknown")
    )
    complete = not unanswered and not outside
    status_of = {f["id"]: f["status"] for f in fish_rows}
    unknown = sum(len(dets) for fid, dets in counted.items() if status_of[fid] == "unknown")
    legacy = sum(1 for f in fish_rows if f["status"] in REVIEWED and f["id"] not in scopes)
    equipment = [f for f in fish_rows if f["status"] == "not_fish" and project.answer_details(f["id"]).get("equipment")]
    rows = [
        ["review", "complete" if complete else
         f"provisional: {len(unanswered)} cards not answered ({sum(f in tiny for f in unanswered)} of them tiny)"],
        ["sampled rate", f"{project.frame_rate():g} frames per second"],
        ["cards (all / answered / skipped / pending / detections)", "{} / {} / {} / {} / {}".format(*tally(fish_rows))],
        [f"  not tiny (â‰¥ {tiny_px:g} px)", "{} / {} / {} / {} / {}".format(*tally(big))],
        [f"  tiny (< {tiny_px:g} px, hidden until asked for)", "{} / {} / {} / {} / {}".format(*tally(tiny))],
        ["detections counted", sum(len(d) for d in counted.values())],
        ["  of them unknown fish (in no species' MaxN)", unknown],
        ["detections answered as not a fish", sum(len(fish_dets[f["id"]]) for f in fish_rows if f["status"] == "not_fish")],
        ["answered as equipment (not a fish)", "{} cards, {} detections".format(
            len(equipment), sum(len(fish_dets[f["id"]]) for f in equipment))],
        ["detections on answered cards outside the answer's scope (not counted)", outside],
        ["answered cards from before scopes were recorded (counted whole)", legacy],
    ]

    by_minute: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for f in fish_rows:
        is_tiny = sizes.get(f["id"], 0.0) < tiny_px
        ok = {d["id"] for d in counted.get(f["id"], [])}
        for d in fish_dets[f["id"]]:
            row = by_minute[int(d["t"] // 60)]
            row[0] += 1
            if d["id"] in ok:
                row[1] += 1
            elif f["status"] == "not_fish":
                row[2] += 1
            else:
                row[3] += 1
                row[4] += is_tiny
    minutes = []
    for minute, (n, ok, not_fish, open_, open_tiny) in sorted(by_minute.items()):
        clip, local = timeline.locate(minute * 60.0)
        minutes.append([minute, clip, seconds_to_time(local), n, ok, not_fish, open_, open_tiny])
    return rows, minutes, complete
