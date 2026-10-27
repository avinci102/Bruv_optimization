"""Which sighting to show a person as a card's or a piece's best view.

The species model gets the card's sharpest, biggest crops (pipeline.card_scores); a person
identifying a fish wants the fish side-on and whole. So the picture shown is chosen
separately, and only for display: the crop quality (detection confidence x size x
sharpness, classify.crop_quality), lowered for a box that is short for its height (a fish
seen head-on or tail-on is about as tall as long; side-on it is longer) and for a box that
touches the frame's edge (the fish is probably cut off). Nothing here changes guesses,
cards or counts.
"""

from __future__ import annotations

SIDE_ON = 2.0  # a fish seen side-on is about this many times as long as tall, fins included
EDGE = 0.01  # a box closer than this (fraction of the frame) to an edge touches it
EDGE_PENALTY = 0.6


def view_score(d) -> float:
    """How good a detection is to show a person (rows with x1..y2, quality, width, height)."""
    w = (d["x2"] - d["x1"]) * d["width"]
    h = (d["y2"] - d["y1"]) * d["height"]
    if w <= 0 or h <= 0:
        return 0.0
    side = min(1.0, max(0.4, (w / h) / SIDE_ON))
    edge = EDGE_PENALTY if min(d["x1"], d["y1"], 1 - d["x2"], 1 - d["y2"]) < EDGE else 1.0
    return (d["quality"] or 0.0) * side * edge


def best_view(dets):
    """The detection to show for a card or a piece (None for none)."""
    return max(dets, key=view_score, default=None)
