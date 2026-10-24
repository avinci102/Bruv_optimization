"""Lightweight tracker: IoU + centre distance + a colour check, solved as a linear assignment
(SciPy's linear_sum_assignment, a modified Jonker-Volgenant solver; same result as Hungarian).

Each track expects its fish to keep swimming the way it was going, so two fish that cross
keep their own tracks. At 1 fps fish move a long way between frames, so tracks fragment;
continuity.py stitches the fragments back together for review.

Each link is recorded with its evidence (Tracklet.links), without changing how links are
chosen: its cost; its margin, how much worse the next-best option was (another track for
this box, or another box for this track; small or negative means a close call); and how
alike the box's colours are to everything the track held before it (a drop there can mark
the point where a track jumped to another fish).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection

_INF = 1e6


def iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def centre_distance(a: Detection, b: Detection) -> float:
    """Centre distance divided by the larger box diagonal."""
    ca = ((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2)
    cb = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
    diag = max(np.hypot(a.w, a.h), np.hypot(b.w, b.h), 1e-6)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / diag)


def size_ratio(a: Detection, b: Detection) -> float:
    sa, sb = max(a.w * a.h, 1e-9), max(b.w * b.h, 1e-9)
    return float(np.sqrt(max(sa, sb) / min(sa, sb)))


@dataclass
class Tracklet:
    id: int
    # (frame index, detection key) pairs in time order
    members: list[tuple[int, int]] = field(default_factory=list)
    last_box: Detection | None = None
    last_frame: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)  # box shift per sampled frame, from the last match
    # detection key -> (cost, margin, colour likeness to the track so far), for each box
    # that continued the track (see the module docstring); margin None: no other option.
    links: dict[int, tuple[float, float | None, float | None]] = field(default_factory=dict)
    colour_sum: np.ndarray | None = field(default=None, repr=False)

    def predicted(self, frame: int) -> Detection:
        """Where the fish should be by `frame` if it keeps its direction and speed."""
        b, dt = self.last_box, frame - self.last_frame
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        return Detection(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, b.score, b.app)

    def add_colour(self, app: np.ndarray | None) -> None:
        if app is not None:
            self.colour_sum = app.copy() if self.colour_sum is None else self.colour_sum + app

    def history_likeness(self, app: np.ndarray | None) -> float | None:
        if app is None or self.colour_sum is None:
            return None
        return float(np.dot(app, self.colour_sum) / max(float(np.linalg.norm(self.colour_sum)), 1e-12))


def _centre(d: Detection) -> tuple[float, float]:
    return (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2


def missed_frames(seconds: float, fps: float) -> int:
    """Settings.track_max_missed_s as sampled frames: how many frames in a row a track may
    go without its fish. As built (not yet changed; see docs/SPEC.md §13): the longest gap
    between two sightings a track bridges is (n + 1) / fps, so 1.0 s at 4 fps bridges 1.25 s;
    round() takes halves to even (1.0 s at 2.5 fps is 2 frames); one frame is the minimum."""
    return max(1, round(seconds * fps))


def _margin(cost: np.ndarray, r: int, c: int) -> float | None:
    """How much worse the next-best allowed option was: another track taking this box, or
    this track taking another box. None when there was no other option."""
    others = np.concatenate([np.delete(cost[:, c], r), np.delete(cost[r, :], c)])
    others = others[others < _INF]
    return float(others.min() - cost[r, c]) if others.size else None


class Tracker:
    def __init__(
        self,
        min_iou: float = 0.05,
        max_dist: float = 1.5,
        max_size_ratio: float = 3.0,
        max_missed: int = 1,
        min_appearance: float = 0.5,
        direction: bool = True,
    ):
        self.min_iou = min_iou
        self.max_dist = max_dist
        self.max_size_ratio = max_size_ratio
        self.max_missed = max_missed
        self.min_appearance = min_appearance
        self.direction = direction
        self.active: list[Tracklet] = []
        self.finished: list[Tracklet] = []
        self._next_id = 1

    def _cost(self, track: Tracklet, det: Detection, frame: int) -> float:
        box = track.last_box
        if size_ratio(box, det) > self.max_size_ratio:
            return _INF
        # Scored against where the fish was or where it was heading, whichever fits better:
        # a fish that keeps its course matches cheaply (two crossing fish keep their own
        # tracks), and one that turns matches as it did before.
        cost = _INF
        for ref in (box, track.predicted(frame)) if self.direction else (box,):
            o, d = iou(ref, det), centre_distance(ref, det)
            if o >= self.min_iou or d <= self.max_dist:
                cost = min(cost, (1.0 - o) + 0.5 * d)
        if cost >= _INF:
            return _INF
        if box.app is not None and det.app is not None:
            sim = float(np.dot(box.app, det.app))
            if sim < self.min_appearance:
                return _INF
            cost += 0.5 * (1.0 - sim)
        return cost

    def update(self, frame: int, detections: list[tuple[int, Detection]]) -> None:
        """Feed one sampled frame: a list of (detection key, Detection)."""
        # Retire tracks that have been missing too long.
        still = []
        for t in self.active:
            (still if frame - t.last_frame <= self.max_missed + 1 else self.finished).append(t)
        self.active = still

        matched_dets: set[int] = set()
        if self.active and detections:
            cost = np.array([[self._cost(t, d, frame) for _, d in detections] for t in self.active])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= _INF:
                    continue
                key, det = detections[c]
                track = self.active[r]
                track.links[key] = (float(cost[r, c]), _margin(cost, r, c), track.history_likeness(det.app))
                track.members.append((frame, key))
                track.add_colour(det.app)
                (x0, y0), (x1, y1) = _centre(track.last_box), _centre(det)
                dt = frame - track.last_frame
                track.velocity = ((x1 - x0) / dt, (y1 - y0) / dt)
                track.last_box, track.last_frame = det, frame
                matched_dets.add(c)

        for c, (key, det) in enumerate(detections):
            if c in matched_dets:
                continue
            track = Tracklet(self._next_id, [(frame, key)], det, frame)
            track.add_colour(det.app)
            self._next_id += 1
            self.active.append(track)

    def finish(self) -> list[Tracklet]:
        self.finished.extend(self.active)
        self.active = []
        return sorted(self.finished, key=lambda t: (t.members[0][0], t.id))


def track_all(
    frames: list[list[tuple[int, Detection]]], **kwargs
) -> list[Tracklet]:
    """Convenience: run the tracker over every frame (index = position in `frames`)."""
    tracker = Tracker(**kwargs)
    for i, dets in enumerate(frames):
        tracker.update(i, dets)
    return tracker.finish()
