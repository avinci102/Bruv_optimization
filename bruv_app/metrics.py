"""Sightings, MaxN and time of first arrival, all from per-frame detections.

Only human-confirmed labels count. A fish's label applies to every detection of that fish,
then each sampled frame is counted on its own; merges never change the counts.
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
