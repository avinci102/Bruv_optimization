"""Sightings, MaxN and time of first arrival, all from per-frame detections.

Only human-confirmed labels count. A fish's label applies to the detections it was given
for, then each sampled frame is counted on its own; merges never change the counts, and
abundance is never deduplicated by card. double_count_checks lists what could still count
one fish twice: two boxes on one fish, or one card holding two boxes at once.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

UNKNOWN = "__unknown__"


@dataclass
class FishLabel:
    fish_id: int
    key: str  # species name, or UNKNOWN for "Unknown fish"
    start_t: float
    end_t: float


@dataclass
class MaxNRow:
    species: str
    maxn: int
    t_maxn: float
    first_t: float


@dataclass
class Sighting:
    species: str  # may be UNKNOWN
    t_in: float
    t_out: float
    count: int
    fish_ids: list[int] = field(default_factory=list)


def per_frame_counts(
    detections: list[tuple[float, int]], labels: dict[int, str]
) -> dict[float, Counter]:
    """detections: (frame time, fish id). Returns frame time -> Counter(species)."""
    counts: dict[float, Counter] = defaultdict(Counter)
    for t, fish_id in detections:
        key = labels.get(fish_id)
        if key is not None:
            counts[t][key] += 1
    return counts


def compute_maxn(counts: dict[float, Counter]) -> list[MaxNRow]:
    best: dict[str, MaxNRow] = {}
    for t in sorted(counts):
        for species, n in counts[t].items():
            if species == UNKNOWN:
                continue
            row = best.get(species)
            if row is None:
                best[species] = MaxNRow(species, n, t, t)
            elif n > row.maxn:
                row.maxn, row.t_maxn = n, t
    return sorted(best.values(), key=lambda r: (r.first_t, r.species))


@dataclass
class Observation:
    """A counted detection: when, on which card, counted as what, and where in the frame."""

    t: float
    det_id: int
    fish_id: int
    key: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (0-1)


@dataclass
class CountCheck:
    t: float
    kind: str  # "overlapping boxes" or "one card, two boxes"
    key: str
    det_ids: tuple[int, int]
    fish_ids: tuple[int, int]


def _overlap(a, b) -> tuple[float, float]:
    """(IoU, share of the smaller box inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0), (inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0)


def double_count_checks(observations: list[Observation], min_iou: float = 0.5, inside: float = 0.8) -> list[CountCheck]:
    """Frames where one fish may be counted twice under the same label: two boxes that
    overlap heavily (the detector boxed one fish twice, or a fish and part of it), or two
    boxes of one card at the same moment (two fish on one card, or one fish boxed twice).
    Both count as two; a reviewer decides which they are."""
    by_frame: dict[float, list[Observation]] = defaultdict(list)
    for o in observations:
        by_frame[o.t].append(o)
    out = []
    for t, obs in sorted(by_frame.items()):
        for i, a in enumerate(obs):
            for b in obs[i + 1 :]:
                if a.key != b.key:
                    continue
                share_iou, share_inside = _overlap(a.box, b.box)
                if share_iou >= min_iou or share_inside >= inside:
                    out.append(CountCheck(t, "overlapping boxes", a.key, (a.det_id, b.det_id), (a.fish_id, b.fish_id)))
                elif a.fish_id == b.fish_id:
                    out.append(CountCheck(t, "one card, two boxes", a.key, (a.det_id, b.det_id), (a.fish_id, b.fish_id)))
    return out


def card_sightings(fish: list[FishLabel]) -> list[Sighting]:
    """One sighting per answered card: a card is one fish while it stays on screen (docs/SPEC.md
    §15), so each is a row of one. Cards a reviewer joined are one card, so one row."""
    return sorted((Sighting(f.key, f.start_t, f.end_t, 1, [f.fish_id]) for f in fish), key=lambda s: (s.t_in, s.species))


def compute_sightings(
    fish: list[FishLabel],
    detections: list[tuple[float, int]],
    gap_seconds: float = 10.0,
) -> list[Sighting]:
    """Group same-species fish whose spans overlap or sit within the gap window."""
    by_key: dict[str, list[FishLabel]] = defaultdict(list)
    for f in fish:
        by_key[f.key].append(f)

    sightings: list[Sighting] = []
    for key, items in by_key.items():
        items.sort(key=lambda f: f.start_t)
        current: Sighting | None = None
        for f in items:
            if current is not None and f.start_t - current.t_out <= gap_seconds:
                current.t_out = max(current.t_out, f.end_t)
                current.fish_ids.append(f.fish_id)
            else:
                current = Sighting(key, f.start_t, f.end_t, 0, [f.fish_id])
                sightings.append(current)

    fish_to_sighting = {fid: s for s in sightings for fid in s.fish_ids}
    per_frame: dict[int, Counter] = defaultdict(Counter)
    for t, fish_id in detections:
        s = fish_to_sighting.get(fish_id)
        if s is not None:
            per_frame[id(s)][t] += 1
    for s in sightings:
        frame_counts = per_frame.get(id(s))
        s.count = max(frame_counts.values()) if frame_counts else 1
    return sorted(sightings, key=lambda s: (s.t_in, s.species))
