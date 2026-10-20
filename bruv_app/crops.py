"""E1: the same crops from four sources, scored by the species model (review V2, task 2).

A   the saved review frame (960 px JPEG, quality 85): what the pipeline embeds today;
A'  the same 960 px frame before JPEG (from the 1280 decode): tells JPEG loss from size;
B   the 1280 px decode the detector saw;
C   the source frame at full size (2688 px on the 44-minute test video).

Every arm crops the same sightings: the ones the embed stage chose for each piece, frozen
(tiny pieces too with tiny=True, reported apart, since the pipeline skips them), with the
same normalised boxes on each frame's size, the same padding and the model's own 224 px
squash. Source frames are decoded by number (video.grab_frames) and each crop is checked
against the saved review frame before use, in its own box, where the fish moves: the
decoded frame must match the saved crop better than the two frames either side of it, else
the crop is left out ("misaligned"). A crop that matches all five about as well ("static":
nothing moved in it) is kept, since its pixels are the same either way. Guesses are the
model's own, with no library blend, so the arms differ only in their pixels.

Results go to experiments/e1-<time>/: a row per crop (crops.csv) and a summary. Nothing in
the pipeline or the cards changes: which source to use is decided only once this has been
run on labelled footage (accuracy needs answered cards; without them only agreement is
reported). Image position (centre, middle, edge) and the camera's recorded lens mode are
kept with each row, for E11.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import imaging
from .classify import l2_normalize, softmax
from .diagnostics import all_flags
from .provenance import code_revision, describe_model
from .store import Project
from .video import grab_frames

ARMS = ("A", "A'", "B", "C")
NATIVE_BANDS = (60, 120, 240)  # the box's longer side in source-frame pixels
TOLERANCE = 0.1  # grey levels: closer than this, the frame and its neighbours can't be told apart
UNLIKE = 10.0  # a frame this far from the saved one isn't the same picture at all


def position_band(x1: float, y1: float, x2: float, y2: float) -> str:
    """Where the box's centre sits: distance from the frame's centre, 0 there and 1 in a
    corner, in thirds."""
    r = float(np.hypot((x1 + x2) / 2 - 0.5, (y1 + y2) / 2 - 0.5) / np.hypot(0.5, 0.5))
    return "centre" if r < 1 / 3 else "middle" if r < 2 / 3 else "edge"


def native_band(px: float) -> str:
    lower = 0
    for edge in NATIVE_BANDS:
        if px < edge:
            return f"{lower}-{edge}"
        lower = edge
    return f"{NATIVE_BANDS[-1]}+"


WINDOW = 2  # frames either side compared with the saved crop


def alignment(review: np.ndarray, small: dict[int, np.ndarray], i: int, box) -> tuple[str, float]:
    """Is the saved review crop made from source frame i? Compares the box region (padded)
    of the saved frame with the same region of frames i-2..i+2, each already at the saved
    frame's size (mean absolute difference, grey levels)."""
    x1, y1, x2, y2 = box
    h, w = review.shape[:2]
    pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    c0, c1 = max(0, int((x1 - pw) * w)), min(w, int(np.ceil((x2 + pw) * w)))
    r0, r1 = max(0, int((y1 - ph) * h)), min(h, int(np.ceil((y2 + ph) * h)))
    saved = review[r0:r1, c0:c1].astype(np.float32)
    mads = {j: float(np.abs(img[r0:r1, c0:c1].astype(np.float32) - saved).mean()) for j, img in small.items()}
    here = mads[i]
    others = [m for j, m in mads.items() if j != i]
    if here > UNLIKE:
        return "different", here
    if others and min(others) < here - TOLERANCE:
        return "misaligned", here
    if others and here < min(others) - TOLERANCE:
        return "aligned", here
    return "static", here


def choose_crops(project: Project, crops_per_tracklet: int, tiny_px: float, tiny: bool, limit: int | None) -> list[dict]:
    """The sightings to crop: those the embed stage chose for each piece (the ones with an
    embedding), or for pieces it skipped (tiny), the same choice by quality. `limit` keeps
    that many pieces, spread evenly over the video."""
    pieces = []
    for tid in project.tracklet_ids():
        dets = project.tracklet_detections(tid)
        is_tiny = max(max((d["x2"] - d["x1"]) * d["width"], (d["y2"] - d["y1"]) * d["height"]) for d in dets) < tiny_px
        chosen = [d for d in dets if d["embedding"] is not None]
        if not chosen:
            if not tiny:
                continue
            chosen = sorted(dets, key=lambda d: d["quality"] or 0.0, reverse=True)[:crops_per_tracklet]
        pieces.append((tid, is_tiny, chosen))
    if limit and len(pieces) > limit:
        pieces = [pieces[int(k)] for k in np.linspace(0, len(pieces) - 1, limit)]
    return [{"det": d, "tracklet": tid, "tiny": is_tiny} for tid, is_tiny, chosen in pieces for d in chosen]


def _labels(project: Project, min_similarity: float) -> dict[int, str]:
    """Detection -> the species its card was answered as, for cards without warnings (a card
    that may mix fish isn't a clean answer for its pieces), within the answer's scope."""
    flagged = set(all_flags(project, min_similarity))
    scopes = project.label_scopes()
    out = {}
    for f in project.all_fish():
        if f["status"] != "species" or f["id"] in flagged:
            continue
        scope = scopes.get(f["id"])
        out.update((d["id"], f["species"]) for d in project.fish_detections(f["id"]) if scope is None or d["id"] in scope)
    return out


def run(project: Project, classifier, tiny: bool = False, limit: int | None = None,
        chunk: int = 12, progress=None) -> tuple[dict, Path]:
    from .pipeline import record_videos  # noqa: PLC0415

    settings = project.load_settings()
    record_videos(project)  # the footage's fingerprint and what it says about the camera
    project.fill_source_indices()
    text_path = project.root / "text_embeddings.npy"
    names = project.get_meta("guess_labels") or []
    if not (text_path.exists() and names):
        raise RuntimeError("No species names scored yet: process the project first")
    text = np.load(text_path)
    scale = float(project.get_meta("logit_scale", 100.0))
    chapters = {c["idx"]: c for c in project.chapters()}
    missing = [c["path"] for c in chapters.values() if not Path(c["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"The source video isn't where the project says: {', '.join(missing)}")

    crops = choose_crops(project, settings.crops_per_tracklet, settings.tiny_fish_px, tiny, limit)
    frames = {f["id"]: f for f in project.db.execute("SELECT * FROM frames_src")}
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for c in crops:
        by_frame[c["det"]["frame_id"]].append(c)
    order = sorted(by_frame, key=lambda fid: frames[fid]["seq"])
    labels = _labels(project, settings.merge_min_similarity)

    def cut(image, d):
        return imaging.crop_box(image, d["x1"], d["y1"], d["x2"], d["y2"], settings.crop_pad)

    rows, embs = [], {arm: [] for arm in ARMS}
    status = Counter()
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        pending = {arm: [] for arm in ARMS}
        for ch in {frames[fid]["chapter"] for fid in part}:
            mine = [fid for fid in part if frames[fid]["chapter"] == ch]
            path, fps = Path(chapters[ch]["path"]), chapters[ch]["fps"]
            at = {frames[fid]["src_index"]: fid for fid in mine}
            width = frames[mine[0]]["width"]
            # Frames are cropped as they're decoded and dropped: only 960 px copies of the
            # nearby frames (for the alignment check) are kept for a chunk.
            small, crop_b, crop_c, native = {}, {}, {}, {}
            near = sorted({j for i in at for j in range(i - WINDOW, i + WINDOW + 1)})
            for j, image in grab_frames(path, fps, near, width=settings.decode_width):
                small[j] = imaging.resize_to_width(image, width)  # A': as the pipeline made it before JPEG
                for c in by_frame.get(at.get(j), []):
                    crop_b[c["det"]["id"]] = cut(image, c["det"])
            for i, image in grab_frames(path, fps, sorted(at)):
                native[i] = image.shape
                for c in by_frame[at[i]]:
                    crop_c[c["det"]["id"]] = cut(image, c["det"])
            for i, fid in sorted(at.items()):
                f = frames[fid]
                review = imaging.load_rgb(str(project.root / f["path"]))
                near_i = {j: small[j] for j in range(i - WINDOW, i + WINDOW + 1) if j in small}
                for c in by_frame[fid]:
                    d = c["det"]
                    state, mad = alignment(review, near_i, i, (d["x1"], d["y1"], d["x2"], d["y2"]))
                    status[state] += 1
                    rows.append({
                        "det_id": d["id"], "tracklet": c["tracklet"], "tiny": c["tiny"], "t": round(f["t"], 2),
                        "src_index": i, "alignment": state, "mad": round(mad, 2),
                        "native_px": round(max((d["x2"] - d["x1"]) * native[i][1], (d["y2"] - d["y1"]) * native[i][0]), 1),
                        "position": position_band(d["x1"], d["y1"], d["x2"], d["y2"]),
                        "label": labels.get(d["id"], ""),
                    })
                    pending["A"].append(cut(review, d))
                    pending["A'"].append(cut(small[i], d))
                    pending["B"].append(crop_b[d["id"]])
                    pending["C"].append(crop_c[d["id"]])
        for arm in ARMS:
            if pending[arm]:
                embs[arm].extend(classifier.embed(pending[arm]))
        if progress:
            progress("crops", min(start + chunk, len(order)), len(order))

    scores = {arm: softmax_rows(scale * (np.asarray(embs[arm]) @ text.T)) for arm in ARMS if embs[arm]}
    for k, row in enumerate(rows):
        for arm in scores:
            top = np.argsort(-scores[arm][k])[:3]
            row[f"{arm} top1"] = names[top[0]]
            row[f"{arm} p1"] = round(float(scores[arm][k][top[0]]), 3)
            row[f"{arm} top3"] = "; ".join(names[t] for t in top)

    summary = summarise(rows, embs, text, scale, names)
    summary["provenance"] = {
        "code": code_revision(), "classifier": describe_model(classifier), "deployment": project.get_meta("deployment"),
        "capture": project.get_meta("capture"), "crop_pad": settings.crop_pad, "decode_width": settings.decode_width,
        "review_frames": f"{settings.frame_width} px JPEG quality {settings.jpeg_quality}", "tiny_included": tiny,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary["alignment"] = dict(status)
    out = project.root / "experiments" / f"e1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["det_id"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.txt").write_text("\n".join(summary_lines(summary)), encoding="utf-8")
    return summary, out


def softmax_rows(x: np.ndarray) -> np.ndarray:
    return np.stack([softmax(r) for r in x]) if len(x) else x


def summarise(rows: list[dict], embs: dict, text: np.ndarray, scale: float, names: list[str]) -> dict:
    """Agreement with today's crops (A), and accuracy where cards were answered: per crop,
    and per piece (its crops' embeddings averaged, as the pipeline does), with the paired
    A-versus-C counts and the same by native size, image position and tiny or not."""
    usable = [k for k, r in enumerate(rows) if r["alignment"] in ("aligned", "static")]
    out: dict = {"crops": len(rows), "usable": len(usable), "pieces": len({rows[k]["tracklet"] for k in usable})}

    def tally(keys):
        res = {}
        for arm in ARMS:
            if f"{arm} top1" not in rows[0]:
                continue
            same = sum(rows[k][f"{arm} top1"] == rows[k]["A top1"] for k in keys)
            labelled = [k for k in keys if rows[k]["label"]]
            right = sum(rows[k][f"{arm} top1"] == rows[k]["label"] for k in labelled)
            in3 = sum(rows[k]["label"] in rows[k][f"{arm} top3"].split("; ") for k in labelled)
            res[arm] = {"n": len(keys), "same_top1_as_A": same, "labelled": len(labelled), "top1_right": right, "top3_right": in3,
                        "mean_p1": round(float(np.mean([rows[k][f"{arm} p1"] for k in keys])), 3) if keys else None}
        return res

    if not rows:
        return out
    out["per_crop"] = tally(usable)
    out["by_native_size"] = {b: tally([k for k in usable if native_band(rows[k]["native_px"]) == b])
                             for b in sorted({native_band(rows[k]["native_px"]) for k in usable})}
    out["by_position"] = {p: tally([k for k in usable if rows[k]["position"] == p])
                          for p in ("centre", "middle", "edge") if any(rows[k]["position"] == p for k in usable)}
    out["tiny"] = {t: tally([k for k in usable if rows[k]["tiny"] == t]) for t in (False, True)
                   if any(rows[k]["tiny"] == t for k in usable)}

    # Per piece: average its usable crops' embeddings in each arm, then guess (as the pipeline does).
    pieces: dict[int, list[int]] = defaultdict(list)
    for k in usable:
        pieces[rows[k]["tracklet"]].append(k)
    paired = Counter()
    piece_tally = {arm: Counter() for arm in ARMS if embs.get(arm)}
    for keys in pieces.values():
        label = Counter(rows[k]["label"] for k in keys if rows[k]["label"]).most_common(1)
        tops = {}
        for arm in piece_tally:
            emb = l2_normalize(np.mean([embs[arm][k] for k in keys], axis=0))
            p = softmax(scale * (text @ emb))
            tops[arm] = [names[t] for t in np.argsort(-p)[:3]]
            piece_tally[arm]["n"] += 1
            piece_tally[arm]["same_top1_as_A"] += tops[arm][0] == (tops.get("A") or tops[arm])[0]
            if label:
                piece_tally[arm]["labelled"] += 1
                piece_tally[arm]["top1_right"] += tops[arm][0] == label[0][0]
                piece_tally[arm]["top3_right"] += label[0][0] in tops[arm]
        if label and "A" in tops and "C" in tops:
            a, c = tops["A"][0] == label[0][0], tops["C"][0] == label[0][0]
            paired["both right" if a and c else "A only" if a else "C only" if c else "both wrong"] += 1
    out["per_piece"] = {arm: dict(t) for arm, t in piece_tally.items()}
    out["paired_A_vs_C_pieces"] = dict(paired)
    return out


def summary_lines(s: dict) -> list[str]:
    lines = [
        f"E1 crop sources: {s['crops']} crops from {s.get('pieces', 0)} pieces; {s['usable']} usable "
        f"(alignment: {s.get('alignment', {})})",
        "A = review JPEG 960 px (today) · A' = 960 px without JPEG · B = 1280 px decode · C = source frame",
    ]

    def show(title, tallies):
        lines.append(title)
        for arm, t in tallies.items():
            acc = (f"; answered: top-1 {t['top1_right']}/{t['labelled']}, top-3 {t['top3_right']}/{t['labelled']}"
                   if t.get("labelled") else "")
            p1 = f", mean top score {t['mean_p1']}" if t.get("mean_p1") is not None else ""
            lines.append(f"  {arm:2}: top guess same as A {t['same_top1_as_A']}/{t['n']}{p1}{acc}")

    if "per_crop" in s:
        show("Per crop:", s["per_crop"])
        show("Per piece (crops averaged, as the pipeline does):", s["per_piece"])
        if s.get("paired_A_vs_C_pieces"):
            lines.append(f"Answered pieces, A against C: {s['paired_A_vs_C_pieces']}")
        for name, groups in (("native size (px)", s["by_native_size"]), ("image position", s["by_position"]), ("tiny", s["tiny"])):
            for key, tallies in groups.items():
                show(f"By {name}: {key}", tallies)
    lines.append(f"Capture: {s.get('provenance', {}).get('capture')}")
    return lines
