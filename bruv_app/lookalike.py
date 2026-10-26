"""Look-alike cards: the same fish, or the same kind of fish, answered once instead of over
and over.

Short tracks keep each card to one fish, so a fish that circles the bait for twenty
minutes comes back as many cards. Each card's look is the mean embedding of its best crops,
the same one its guesses use, and two cards look alike when those are at least 0.85 alike.
That's the merge's bar for joining two pieces of one fish (Settings.merge_min_similarity).

Three rules use it, and none answers anything by itself:
- a card that looks like one you've answered shows that answer, for one key (R);
- cards at least 0.90 alike to an answered card make up the quick-confirm queue (Show →
  quick confirm), where each is shown beside the answered card and Y gives it that answer;
- after you answer a card, the unanswered cards that look like it can be answered together
  from a scrolling list of their pictures, one per piece, where you untick any that
  differ. A card joins that list only if every one of its pieces looks like the card you
  answered, not just its average (a fish joined onto another would pass on the average),
  and cards with warnings are left out, as in the equipment queue.

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


def neighbours(looks: Looks, min_similarity: float = LOOKALIKE) -> dict[int, list[tuple[int, float]]]:
    """Card id -> the other cards that look like it, most alike first. It doesn't depend
    on answers, so it can be worked out once when a project is published (a static review
    site then finds a card's answered look-alike from it and the answers so far)."""
    out: dict[int, list[tuple[int, float]]] = {}
    for start in range(0, len(looks.ids), 512):
        sims = looks.emb[start:start + 512] @ looks.emb.T
        for row, card in enumerate(looks.ids[start:start + 512]):
            hits = np.flatnonzero(sims[row] >= min_similarity)
            pairs = [(int(looks.ids[j]), float(sims[row, j])) for j in hits if int(looks.ids[j]) != int(card)]
            out[int(card)] = sorted(pairs, key=lambda p: -p[1])
    return out


def card_pieces(project: Project, fish_id: int, ref: np.ndarray | None) -> list[dict]:
    """A card's tracked pieces, each with the sighting to show for it (views.py) and how
    alike the piece is to `ref` (a card's look): None when the piece or `ref` has none."""
    from .store import from_blob  # noqa: PLC0415
    from .views import best_view  # noqa: PLC0415

    out = []
    for tid in project.fish_tracklets(fish_id):
        row = project.db.execute("SELECT embedding, start_t, end_t FROM tracklets WHERE id=?", (tid,)).fetchone()
        dets = project.tracklet_detections(tid)
        if not dets:
            continue
        emb = from_blob(row["embedding"])
        sim = float(emb @ ref) if emb is not None and ref is not None else None
        out.append({"tracklet": tid, "det": best_view(dets)["id"], "similarity": sim,
                    "start_t": row["start_t"], "seconds": row["end_t"] - row["start_t"]})
    return out


def every_piece_alike(parts: list[dict], min_similarity: float = LOOKALIKE) -> bool:
    """Every piece of a card looks like the reference, not just the card's average: a card
    that joined another fish onto this one fails. A piece that can't be compared fails."""
    return bool(parts) and all(p["similarity"] is not None and p["similarity"] >= min_similarity for p in parts)


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


def best_answered(looks: Looks, min_similarity: float = LOOKALIKE) -> dict[int, tuple[int, float]]:
    """Unanswered card id -> (the answered card it looks most like, how alike), for every
    unanswered card with an answered look-alike at least `min_similarity` alike: what the
    quick-confirm queue and the R key offer."""
    done = np.array([looks.status.get(int(f)) in REVIEWED for f in looks.ids], dtype=bool)
    open_ = np.array([looks.status.get(int(f)) in ("pending", "skipped") for f in looks.ids], dtype=bool)
    if not done.any() or not open_.any():
        return {}
    sims = looks.emb[open_] @ looks.emb[done].T
    pick = sims.argmax(axis=1)
    best = sims[np.arange(len(pick)), pick]
    open_ids, done_ids = looks.ids[open_], looks.ids[done]
    return {int(c): (int(done_ids[k]), float(s)) for c, k, s in zip(open_ids, pick, best) if s >= min_similarity}
