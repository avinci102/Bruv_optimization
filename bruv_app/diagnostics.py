"""Warnings for the reviewer, from evidence the pipeline already has.

They never change the cards or the counts: the review asked for diagnostics now and new
rules only once they're tested on annotated footage (docs/SPEC.md §13). A card is flagged
when:

- its pieces look different: two of its pieces are less alike (BioCLIP 2) than the merge
  asks of a join. The merge only compares pieces it joins directly, so A~B and B~C can still
  put an unlike A and C on one card;
- its pieces disagree on species: two pieces each favour a species (at PIECE_SURE or more)
  that isn't even among the other's three best guesses. Ordinary doubt between look-alikes
  doesn't count, only a clear conflict (on the full test video: 29 of 437 cards with several
  pieces, against 79 for any difference in top guess);
- it holds two boxes at once: two of its boxes are in the same frame (two fish, or one fish
  boxed twice), which only a manual merge can do.

A flagged card gets one answer for all its pieces only once the reviewer confirms they are
one fish (server.py); that confirmation is kept with the answer in the review log. The
tracker's closest calls (track.py: links whose next-best option was nearly as good) are
listed for the reviewer to look at, without a flag: how close is too close isn't known yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import guess_scores, load_timeline, top_guesses
from .store import Project, from_blob

PIECE_SURE = 0.5  # a piece "favours" a species at this guess score or more (a display cue)


@dataclass
class Piece:
    """One tracked stretch (tracklet) of a card."""

    tracklet: int
    start_t: float
    end_t: float
    detections: int
    best_det: int | None
    guesses: list
    embedding: np.ndarray | None


@dataclass
class Flag:
    kind: str  # "pieces look different" | "pieces disagree on species" | "two boxes at once"
    text: str


def _when(timeline, t: float) -> str:
    clip, local = timeline.locate(t)
    s = int(round(local))
    return f"{clip} {s // 60:02d}:{s % 60:02d}"


def pieces(project: Project, fish_id: int, top_k: int = 3) -> list[Piece]:
    out = []
    for t in project.db.execute("SELECT * FROM tracklets WHERE fish_id=? ORDER BY start_t, id", (fish_id,)).fetchall():
        dets = project.tracklet_detections(t["id"])
        if not dets:
            continue
        emb = from_blob(t["embedding"])
        best = max(dets, key=lambda d: d["quality"] or 0.0)
        guesses = top_guesses(guess_scores(project, emb), top_k) if emb is not None else []
        out.append(Piece(t["id"], t["start_t"], t["end_t"], len(dets), best["id"], guesses, emb))
    return out


def card_flags(project: Project, fish_id: int, min_similarity: float, parts: list[Piece] | None = None) -> list[Flag]:
    parts = pieces(project, fish_id) if parts is None else parts
    timeline = load_timeline(project)
    flags = []
    embedded = [p for p in parts if p.embedding is not None]
    worst = min(
        ((float(np.dot(a.embedding, b.embedding)), a, b) for i, a in enumerate(embedded) for b in embedded[i + 1 :]),
        key=lambda x: x[0], default=None,
    )
    if worst is not None and worst[0] < min_similarity:
        sim, a, b = worst
        flags.append(Flag(
            "pieces look different",
            f"The pieces at {_when(timeline, a.start_t)} and {_when(timeline, b.start_t)} are less alike "
            f"({sim:.3f}) than the merge asks of a join ({min_similarity:g}).",
        ))
    sure = [p for p in parts if p.guesses and p.guesses[0][1] >= PIECE_SURE]
    top3 = {p.tracklet: {g[0] for g in p.guesses[:3]} for p in sure}
    clash = next(
        ((a, b) for i, a in enumerate(sure) for b in sure[i + 1 :]
         if a.guesses[0][0] not in top3[b.tracklet] and b.guesses[0][0] not in top3[a.tracklet]),
        None,
    )
    if clash:
        a, b = clash
        flags.append(Flag(
            "pieces disagree on species",
            f"The piece at {_when(timeline, a.start_t)} favours {a.guesses[0][0]} ({a.guesses[0][1]:.0%}), the one at "
            f"{_when(timeline, b.start_t)} {b.guesses[0][0]} ({b.guesses[0][1]:.0%}); neither has the other's "
            "species among its three best guesses.",
        ))
    twice = project.db.execute(
        """SELECT MIN(f.t) FROM detections d JOIN tracklets t ON t.id=d.tracklet_id
           JOIN frames f ON f.id=d.frame_id WHERE t.fish_id=? GROUP BY d.frame_id HAVING COUNT(*) > 1
           ORDER BY MIN(f.t)""",
        (fish_id,),
    ).fetchall()
    if twice:
        flags.append(Flag(
            "two boxes at once",
            f"{len(twice)} frame{'s' if len(twice) > 1 else ''} hold two of its boxes, the first at "
            f"{_when(timeline, twice[0][0])}: two fish, or one boxed twice.",
        ))
    return flags


def closest_calls(project: Project, fish_id: int, n: int = 3) -> list:
    """The card's links where the tracker's next-best option was nearly as good."""
    return project.db.execute(
        """SELECT d.id AS det_id, d.frame_id, f.t, d.link_margin, d.history_sim,
                  d.x1, d.y1, d.x2, d.y2, d.score FROM detections d
           JOIN tracklets t ON t.id=d.tracklet_id JOIN frames f ON f.id=d.frame_id
           WHERE t.fish_id=? AND d.link_margin IS NOT NULL ORDER BY d.link_margin LIMIT ?""",
        (fish_id, n),
    ).fetchall()


def all_flags(project: Project, min_similarity: float) -> dict[int, list[Flag]]:
    """Card id -> its warnings, for cards that have any."""
    out = {}
    for f in project.all_fish():
        flags = card_flags(project, f["id"], min_similarity)
        if flags:
            out[f["id"]] = flags
    return out
