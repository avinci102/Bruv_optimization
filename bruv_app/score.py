"""Score a project's cards against the answers and corrections in another copy of it.

A reviewer's answers mark the detections of the cards they reviewed, and their corrections
(split, move, "Multiple fish") say which sightings belong to different fish. A re-run with
new rules can be scored against both:

- does any card now hold sightings that got different answers?
- does any card join sightings the reviewer pulled apart?

Detections are matched by frame time and box, so a re-run at another frame rate (new
detections) can still be scored: the frames both runs sampled hold the same boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .detect import Detection
from .store import Project
from .track import iou

ANSWERS = {"unknown": "unknown fish", "not_fish": "not a fish"}


def detection_answers(project: Project) -> dict[int, str]:
    """Detection id -> the answer its card got: a species, "unknown fish" or "not a fish"."""
    out: dict[int, str] = {}
    for f in project.all_fish():
        answer = f["species"] if f["status"] == "species" else ANSWERS.get(f["status"])
        if answer:
            out.update((d["id"], answer) for d in project.fish_detections(f["id"]))
    return out


def pulled_apart(project: Project) -> list[tuple[set[int], set[int]]]:
    """Pairs of sighting groups the reviewer separated: neighbouring pieces of each card they
    broke up ("Multiple fish"), and the two sides of each split or move."""
    pairs = []
    for c in project.corrections():
        details = json.loads(c["details"])
        if c["action"] == "multiple_fish":
            sides = [{r["id"] for r in project.tracklet_detections(t)} for t in details["tracklets"]]
            pairs += [(a, b) for a, b in zip(sides, sides[1:]) if a and b]
        elif c["action"] in ("split", "move_out"):
            a = {r["id"] for r in project.fish_detections(c["fish_id"])}
            b = {r["id"] for r in project.fish_detections(details["new_card"])}
            if a and b:
                pairs.append((a, b))
    return pairs


def _boxes(project: Project) -> list:
    return project.db.execute(
        "SELECT d.id, f.t, d.x1, d.y1, d.x2, d.y2 FROM detections d JOIN frames f ON f.id=d.frame_id"
    ).fetchall()


def match_detections(project: Project, other: Project) -> dict[int, int]:
    """Map `other`'s detection ids onto `project`'s: same frame time, boxes overlapping."""
    box = lambda r: Detection(r["x1"], r["y1"], r["x2"], r["y2"], 1.0)  # noqa: E731
    mine: dict[float, list] = {}
    for r in _boxes(project):
        mine.setdefault(round(r["t"], 2), []).append(r)
    out = {}
    for r in _boxes(other):
        best = max(mine.get(round(r["t"], 2), []), key=lambda m: iou(box(m), box(r)), default=None)
        if best is not None and iou(box(best), box(r)) >= 0.5:
            out[r["id"]] = best["id"]
    return out


@dataclass
class CardScore:
    cards: int = 0
    tiny: int = 0  # hidden on the review page until asked for
    checked: int = 0  # cards holding answered sightings
    mixed: list[tuple[int, list[str]]] = field(default_factory=list)  # (card id, its answers)
    pulled_apart: int = 0  # groups of sightings the reviewer separated (that this run has)
    rejoined: int = 0  # of those, the ones this run puts on one card again


def score_cards(project: Project, answers: dict[int, str], tiny_px: float) -> CardScore:
    score = CardScore()
    sizes = project.fish_sizes()
    for f in project.all_fish():
        score.cards += 1
        score.tiny += sizes.get(f["id"], 0.0) < tiny_px
        got = sorted({answers[d["id"]] for d in project.fish_detections(f["id"]) if d["id"] in answers})
        score.checked += bool(got)
        if len(got) > 1:
            score.mixed.append((f["id"], got))
    return score


def score_against(project: Project, reference: Project) -> CardScore:
    """Score `project`'s cards with the answers and corrections given in `reference`."""
    mapping = match_detections(project, reference)
    answers = {mapping[d]: a for d, a in detection_answers(reference).items() if d in mapping}
    score = score_cards(project, answers, project.load_settings().tiny_fish_px)
    card_of = dict(project.db.execute(
        "SELECT d.id, t.fish_id FROM detections d JOIN tracklets t ON t.id=d.tracklet_id"
    ).fetchall())
    for a, b in pulled_apart(reference):
        cards_a = {card_of.get(mapping.get(d)) for d in a} - {None}
        cards_b = {card_of.get(mapping.get(d)) for d in b} - {None}
        if cards_a and cards_b:
            score.pulled_apart += 1
            score.rejoined += bool(cards_a & cards_b)
    return score
