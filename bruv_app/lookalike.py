"""Look-alike cards: the same fish, or the same kind of fish, answered once instead of over
and over.

Short tracks keep each card to one fish, so a fish that circles the bait for twenty
minutes comes back as many cards. Each card's look is the mean embedding of its best crops,
the same one its guesses use, and two cards look alike when those are at least 0.85 alike.
That's the merge's bar for joining two pieces of one fish (Settings.merge_min_similarity).

Two rules use it, and neither answers anything by itself:
- a card that looks like one you've answered shows that answer, for one key (R);
- after you answer a card, the unanswered cards that look like it can be answered together
  from a grid of their pictures, where you untick any that differ. Cards with warnings are
  left out of that, as in the equipment queue.

Replaying the first 185 answers on the NP339 drop in the order they were given, the closest
earlier-answered card named the fish right on 40 of 45 cards, and was never wrong; for the
other 5 nothing earlier was alike enough. The model alone got 24 of those 45. This learns
the fish of one video as you go; it doesn't make the model better on the next video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classify import l2_normalize
from .store import REVIEWED, Project

LOOKALIKE = 0.85


@dataclass
class Looks:
    """Every card's look (cards whose crops were embedded; tiny fish have none)."""

    ids: np.ndarray  # card ids
    emb: np.ndarray  # one row per card, unit length
    status: dict[int, str]
    answer: dict[int, str | None]

    def row(self, fish_id: int) -> int | None:
        hits = np.flatnonzero(self.ids == fish_id)
        return int(hits[0]) if hits.size else None


def card_looks(project: Project, crops_per_fish: int) -> Looks:
    """Each card's mean embedding over its best crops (by crop quality), in one query."""
    by_card: dict[int, list] = {}
    rows = project.db.execute(
        """SELECT t.fish_id, d.quality, d.embedding FROM detections d
           JOIN tracklets t ON t.id=d.tracklet_id WHERE d.embedding IS NOT NULL"""
    )
    for r in rows:
        by_card.setdefault(r["fish_id"], []).append((r["quality"] or 0.0, r["embedding"]))
    ids, embs = [], []
    for fish_id, crops in by_card.items():
        best = sorted(crops, key=lambda c: c[0], reverse=True)[:crops_per_fish]
        ids.append(fish_id)
        embs.append(l2_normalize(np.mean([np.frombuffer(e, np.float32) for _, e in best], axis=0)))
    status, answer = {}, {}
    for f in project.db.execute("SELECT id, status, species FROM fish"):
        status[f["id"]], answer[f["id"]] = f["status"], f["species"]
    dim = len(embs[0]) if embs else 0
    return Looks(np.array(ids, dtype=np.int64), np.array(embs, dtype=np.float32).reshape(len(ids), dim), status, answer)


def answered_like(looks: Looks, fish_id: int, min_similarity: float = LOOKALIKE) -> dict | None:
    """The answered card that looks most like this one, if it's alike enough."""
    i = looks.row(fish_id)
    if i is None:
        return None
    done = np.array([looks.status.get(int(f)) in REVIEWED and int(f) != fish_id for f in looks.ids])
    if not done.any():
        return None
    sims = looks.emb[done] @ looks.emb[i]
    j = int(np.argmax(sims))
    if sims[j] < min_similarity:
        return None
    card = int(looks.ids[done][j])
    return {"card": card, "status": looks.status[card], "species": looks.answer[card], "similarity": float(sims[j])}


def unanswered_like(looks: Looks, fish_id: int, min_similarity: float = LOOKALIKE) -> list[tuple[int, float]]:
    """Unanswered cards (pending or skipped) that look like this one, most alike first."""
    i = looks.row(fish_id)
    if i is None:
        return []
    sims = looks.emb @ looks.emb[i]
    out = [
        (int(f), float(s)) for f, s in zip(looks.ids, sims)
        if int(f) != fish_id and s >= min_similarity and looks.status.get(int(f)) in ("pending", "skipped")
    ]
    return sorted(out, key=lambda x: -x[1])


def groups(looks: Looks, order: list[int], min_similarity: float = LOOKALIKE) -> dict[int, int]:
    """Card id -> its look-alike group: each card joins the first earlier card in `order`
    that leads a group and looks like it, or leads a new one. Cards without a look get none."""
    rows = {int(f): i for i, f in enumerate(looks.ids)}
    leaders: list[int] = []  # rows of the cards leading a group
    out: dict[int, int] = {}
    for fish_id in order:
        i = rows.get(fish_id)
        if i is None:
            continue
        if leaders:
            sims = looks.emb[leaders] @ looks.emb[i]
            j = int(np.argmax(sims))
            if sims[j] >= min_similarity:
                out[fish_id] = int(looks.ids[leaders[j]])
                continue
        leaders.append(i)
        out[fish_id] = fish_id
    return out
