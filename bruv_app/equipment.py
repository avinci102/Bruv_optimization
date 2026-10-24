"""The BRUV rig (bait arm, canister, rope) without losing fish.

A reviewer marks a card as equipment, which makes it a reference: the box it usually sits
in, and its colours. Cards that sit in the same place and look the same are queued so they
can be checked together, instead of one by one across the video. Nothing is removed
automatically: the review asked for automatic exclusion only once it has been shown to keep
the fish on annotated footage (docs/SPEC.md §13). Answers given from the queue are ordinary
"not a fish" answers marked as equipment, and can be undone.

A card is queued when at least half its boxes overlap the reference's box (IoU 0.5 or more,
the test evaluate.py uses for "the same box") and its colours are as alike as the merge asks
of two pieces of one fish (Settings.merge_min_colour). A fish crossing the rig overlaps it
only briefly, and one hovering in front of it has other colours, so neither is queued. If
the rig moves, its new place matches no reference and nothing is queued until a card there
is marked too.

The queue can be answered in one go, but only for cards whose every sighting sits in the
rig's box with the rig's colours. On the 44-minute test video, cards that were mostly the
bait canister but had a passing fish joined on by the tracker were queued too; their odd
sightings are counted (Match.odd), and such cards are left to be checked one by one.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .classify import l2_normalize
from .detect import Detection
from .store import Project, from_blob
from .track import iou

MIN_OVERLAP = 0.5


@dataclass
class Match:
    ref: int  # the rig reference it matches
    ref_card: int  # the card that reference was marked on
    overlap: float  # median IoU of the card's boxes with the reference box
    colour: float | None  # cosine of the card's and the reference's average colours
    odd: int = 0  # sightings off the rig's box, or with other colours: maybe a fish joined on


def card_footprint(dets) -> tuple[Detection, np.ndarray | None]:
    """A card's usual box (the median of each edge) and its average colour histogram."""
    box = Detection(*(float(np.median([d[k] for d in dets])) for k in ("x1", "y1", "x2", "y2")), 1.0)
    apps = [from_blob(d["app"]) for d in dets if d["app"] is not None]
    return box, (l2_normalize(np.mean(apps, axis=0)) if apps else None)


def match_card(dets, ref, min_colour: float) -> Match | None:
    box = Detection(ref["x1"], ref["y1"], ref["x2"], ref["y2"], 1.0)
    overlaps = [iou(Detection(d["x1"], d["y1"], d["x2"], d["y2"], 1.0), box) for d in dets]
    if not overlaps or float(np.median(overlaps)) < MIN_OVERLAP:
        return None
    _, colour = card_footprint(dets)
    ref_colour = from_blob(ref["colour"])
    likeness = float(np.dot(colour, ref_colour)) if colour is not None and ref_colour is not None else None
    if likeness is not None and likeness < min_colour:
        return None

    def like_rig(d, overlap) -> bool:
        if overlap < MIN_OVERLAP:
            return False
        app = from_blob(d["app"])
        return app is None or ref_colour is None or float(np.dot(app, ref_colour)) >= min_colour

    odd = sum(not like_rig(d, o) for d, o in zip(dets, overlaps))
    return Match(ref["id"], ref["fish_id"], float(np.median(overlaps)), likeness, odd)


def best_match(dets, refs, min_colour: float) -> Match | None:
    matches = [m for m in (match_card(dets, ref, min_colour) for ref in refs) if m is not None]
    return max(matches, key=lambda m: (-m.odd, m.overlap), default=None)


def rig_candidates(project: Project, min_colour: float) -> dict[int, Match]:
    """Unanswered cards (pending or skipped) that look like a marked piece of equipment."""
    refs = project.rig_refs()
    if not refs:
        return {}
    by_card: dict[int, list] = defaultdict(list)
    rows = project.db.execute(
        """SELECT d.x1, d.y1, d.x2, d.y2, d.app, t.fish_id FROM detections d
           JOIN tracklets t ON t.id=d.tracklet_id JOIN fish f ON f.id=t.fish_id
           WHERE f.status IN ('pending', 'skipped')"""
    )
    for d in rows:
        by_card[d["fish_id"]].append(d)
    out = {}
    for fish_id, dets in by_card.items():
        m = best_match(dets, refs, min_colour)
        if m is not None:
            out[fish_id] = m
    return out


def mark_equipment(project: Project, fish_id: int, batch: str | None = None, match: Match | None = None) -> int | None:
    """Answer a card "not a fish: equipment". Marked by hand (no match), it also becomes a
    reference, unless it already matches one. Returns the reference it was matched to or made."""
    ref = match.ref if match is not None else None
    if ref is None:
        dets = project.fish_detections(fish_id)
        existing = best_match(dets, project.rig_refs(), 0.0)
        ref = existing.ref if existing else project.add_rig_ref(fish_id, *card_footprint(dets))
    details = {"equipment": True, "rig": ref}
    if batch:
        details["batch"] = batch
    if match is not None:
        details.update(overlap=round(match.overlap, 3), colour=None if match.colour is None else round(match.colour, 3))
    project.label_fish(fish_id, "not_fish", details)
    return ref
