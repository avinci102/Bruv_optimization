"""Stitch fragmented tracklets into one review card per fish.

This only cuts review work. Abundance (MaxN) is computed from per-frame detections in
metrics.py and never from these merges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackletSummary:
    id: int
    start_t: float
    end_t: float
    embedding: np.ndarray  # L2-normalised
    start_size: float  # sqrt(box area) near the start, as a fraction of the frame
    end_size: float
    max_px: float = float("inf")  # the fish at its largest: the box's longer side, in pixels
    colour: np.ndarray | None = None  # average colour histogram of its boxes, L2-normalised


def merge_tracklets(
    tracklets: list[TrackletSummary],
    gap_seconds: float = 10.0,
    min_similarity: float = 0.8,
    max_size_ratio: float = 3.0,
    tiny_px: float = 0.0,
    min_colour: float = 0.0,
) -> list[list[int]]:
    """Greedy best-similarity merge into time-ordered chains of tracklet ids.

    B may follow A when B starts after A ends (never overlapping), within `gap_seconds`,
    with similar appearance and colour and a plausible size change. Each tracklet gets at
    most one predecessor and one successor, so a chain can never contain two fish on screen
    at once. Tiny fish (under `tiny_px` at their largest) join nothing: blurry blobs look
    alike, so they chained unrelated fish together.
    """
    by_id = {t.id: t for t in tracklets}
    candidates: list[tuple[float, int, int]] = []
    for a in tracklets:
        for b in tracklets:
            gap = b.start_t - a.end_t
            if a.id == b.id or gap <= 0 or gap > gap_seconds:
                continue
            if min(a.max_px, b.max_px) < tiny_px:
                continue
            if a.colour is not None and b.colour is not None and float(np.dot(a.colour, b.colour)) < min_colour:
                continue
            ratio = max(a.end_size, b.start_size) / max(min(a.end_size, b.start_size), 1e-9)
            if ratio > max_size_ratio:
                continue
            sim = float(np.dot(a.embedding, b.embedding))
            if sim >= min_similarity:
                candidates.append((sim, a.id, b.id))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for _, a, b in candidates:
        if a in succ or b in pred:
            continue
        succ[a], pred[b] = b, a

    chains = []
    for t in sorted(tracklets, key=lambda t: (t.start_t, t.id)):
        if t.id in pred:
            continue
        chain = [t.id]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        chains.append(chain)
    chains.sort(key=lambda c: (by_id[c[0]].start_t, c[0]))
    return chains
