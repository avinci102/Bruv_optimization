"""Accuracy against independent annotations: the evaluation sidecar and stage-by-stage scores.

The sidecar is a CSV of annotated boxes keyed by deployment, chapter and source frame
number (video.source_index). It never refers to a run's frames, detections or groupings,
so one answer key scores every re-run of the same footage, whatever the rules. Each row is
one box: a fish (its local individual id, and its taxon at the highest rank the annotator
can defend), equipment, or another object that isn't a fish. A frame that was checked and
holds nothing gets one row of kind "none", so "nothing here" differs from "not looked at".
Fish the detector never found are boxes no detection matches. An individual whose identity
the annotator can't be sure of (after a crossing, or a disappearance) is marked
"ambiguous", and left out of the identity counts. A sheet pre-filled from a run's
detections (annotation_sheet) marks its rows "unchecked": frames still holding one are
left out, so a detector box never passes for an answer just because nobody changed it.

The scores keep the stages apart (review V2, section E), so a tracker error isn't blamed on
the merge, nor a merge error on the species model:

- detection: boxes matched one to one at IoU 0.5; precision, recall, missed fish, by size;
- tracking (A): identity changes inside the tracklets as the tracker built them;
- merging (B): merge edges joining two pure tracklets of different fish; cards hit by
  both A and B are counted apart;
- cards: multi-fish and multi-taxon cards, impurity, duplicate cards per fish;
- science: MaxN per taxon over the annotated frames, from the reviewed labels.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .detect import Detection
from .metrics import UNKNOWN
from .store import Project
from .track import iou

KINDS = ("fish", "equipment", "other", "none", "unchecked")
MIN_IOU = 0.5
# Size bands for recall, as the box's longer side in pixels of the review frames (the
# same measure as Settings.tiny_fish_px).
SIZE_BANDS = (20, 30, 40, 60)
# Bands of the tracker's link margin (how much worse its next-best option was), for seeing
# whether identity changes gather at close calls. For analysis only: no rule uses them.
MARGIN_BANDS = (0.0, 0.1, 0.3)


def margin_band(margin: float | None) -> str:
    if margin is None:
        return "no other option"
    if margin <= MARGIN_BANDS[0]:
        return f"≤{MARGIN_BANDS[0]:g}"
    for lo, hi in zip(MARGIN_BANDS, MARGIN_BANDS[1:]):
        if margin < hi:
            return f"{lo:g}-{hi:g}"
    return f"≥{MARGIN_BANDS[-1]:g}"


@dataclass
class Box:
    """One row of the evaluation sidecar."""

    deployment: str
    chapter: int
    src_index: int
    t: float = 0.0  # global seconds, for people; matching uses the source frame
    x1: float = 0.0  # 0-1 of the frame's width / height
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    kind: str = "fish"  # fish | equipment | other (not a fish) | none (frame checked, empty) | unchecked
    individual: str = ""  # local identity within the deployment
    identity: str = "certain"  # certain | ambiguous
    taxon: str = ""  # species, or the highest rank the annotator can defend
    rank: str = ""  # species | genus | family | ... | unknown
    visibility: str = ""
    det_id: str = ""  # the detection a pre-filled row came from (a convenience, not a key)
    annotator: str = ""
    note: str = ""

    @property
    def frame(self) -> tuple[str, int, int]:
        return (self.deployment, self.chapter, self.src_index)

    @property
    def detection(self) -> Detection:
        return Detection(self.x1, self.y1, self.x2, self.y2, 1.0)

    @property
    def certain(self) -> bool:
        return self.kind == "fish" and bool(self.individual) and self.identity != "ambiguous"


COLUMNS = [f.name for f in fields(Box)]
_NUMBERS = {"chapter": int, "src_index": int, "t": float, "x1": float, "y1": float, "x2": float, "y2": float}


def read_sidecar(path: Path) -> list[Box]:
    boxes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            values = {k: (row.get(k) or "").strip() for k in COLUMNS}
            try:
                for key, kind in _NUMBERS.items():
                    values[key] = kind(float(values[key])) if values[key] else kind(0)
            except ValueError as exc:
                raise ValueError(f"{path.name} line {n}: {exc}") from exc
            values["kind"] = values["kind"].lower() or "fish"
            values["identity"] = values["identity"].lower() or "certain"
            if values["kind"] not in KINDS:
                raise ValueError(f"{path.name} line {n}: kind must be one of {', '.join(KINDS)}")
            boxes.append(Box(**values))
    return boxes


def write_sidecar(path: Path, boxes: list[Box]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for b in boxes:
            writer.writerow({k: getattr(b, k) for k in COLUMNS})
    return path


def match_boxes(truth: list[Detection], found: list[Detection], min_iou: float = MIN_IOU) -> list[tuple[int, int]]:
    """One-to-one matching: the most pairs overlapping at least `min_iou`, then the most
    overlap (Hungarian). Each annotated box and each detection is used at most once, so two
    detections on one fish can't both count as found. Returns (truth index, found index)."""
    if not truth or not found:
        return []
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    overlap = np.array([[iou(t, f) for f in found] for t in truth])
    cost = np.where(overlap >= min_iou, 1.0 - overlap, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < 1e6]


# --- the run being scored ---------------------------------------------------------------


@dataclass
class Found:
    """A detection, how the machine first grouped it, and the label it carries now."""

    det_id: int
    t: float
    box: Detection
    tracklet: int | None  # as the tracker built it
    card: int | None  # as the merge built it
    label: str | None  # its current card's reviewed answer: a species, UNKNOWN, or None
    margin: float | None = None  # the tracker's link into this box: how close a call (track.py)


@dataclass
class RunView:
    frames: dict[tuple[str, int, int], list[Found]]
    chains: dict[int, list[int]]  # original card -> its tracklets, in time order
    frame_size: tuple[int, int]  # the review frames' width and height
    grouping: str  # "frozen" (as built) or "current" (may include review edits)


def load_run(project: Project) -> RunView:
    """The project's detections keyed like the sidecar, with the original grouping."""
    project.fill_source_indices()
    deployment = project.get_meta("deployment") or ""
    groups = project.original_groups()
    grouping = "frozen"
    if not groups:
        grouping = "current"
        groups = {
            r[0]: (r[1], r[2])
            for r in project.db.execute(
                "SELECT d.id, d.tracklet_id, t.fish_id FROM detections d LEFT JOIN tracklets t ON t.id=d.tracklet_id"
            )
        }
    labels: dict[int, str] = {}
    for f in project.all_fish():
        answer = f["species"] if f["status"] == "species" else UNKNOWN if f["status"] == "unknown" else None
        if answer:
            labels.update((d["id"], answer) for d in project.fish_detections(f["id"]))
    frames: dict[tuple[str, int, int], list[Found]] = {}
    sizes = []
    for f in project.db.execute("SELECT chapter, src_index, width, height FROM frames_src ORDER BY seq"):
        frames[(deployment, f["chapter"], f["src_index"])] = []
        sizes.append((f["width"], f["height"]))
    starts: dict[int, float] = {}
    rows = project.db.execute(
        """SELECT d.id, d.x1, d.y1, d.x2, d.y2, d.link_margin, f.chapter, f.src_index, f.t
           FROM detections d JOIN frames_src f ON f.id=d.frame_id ORDER BY f.seq, d.id"""
    )
    for r in rows:
        tracklet, card = groups.get(r["id"], (None, None))
        if tracklet is None:
            continue  # kept below the operating threshold (Settings.det_floor): not part of this run's output
        box = Detection(r["x1"], r["y1"], r["x2"], r["y2"], 1.0)
        frames[(deployment, r["chapter"], r["src_index"])].append(
            Found(r["id"], r["t"], box, tracklet, card, labels.get(r["id"]), r["link_margin"])
        )
        if tracklet is not None:
            starts.setdefault(tracklet, r["t"])
    chains: dict[int, list[int]] = defaultdict(list)
    for tracklet, card in {(t, c) for t, c in groups.values() if t is not None and c is not None}:
        chains[card].append(tracklet)
    for card in chains:
        chains[card].sort(key=lambda t: (starts.get(t, 0.0), t))
    size = Counter(sizes).most_common(1)[0][0] if sizes else (960, 540)
    return RunView(frames, dict(chains), size, grouping)


# --- scores ---------------------------------------------------------------------------------


@dataclass
class Report:
    grouping: str = "frozen"
    frames: int = 0  # annotated frames this run sampled
    frames_unsampled: int = 0  # annotated frames this run has no frame for
    frames_unchecked: int = 0  # frames still holding a pre-filled row nobody checked
    # detection
    fish: int = 0
    found: int = 0
    matched: int = 0
    missed: int = 0
    false: int = 0
    false_on_equipment: int = 0
    recall_by_size: dict = field(default_factory=dict)  # band -> (matched, fish)
    # tracking (A)
    transitions: int = 0
    switches: int = 0
    tracklets_eligible: int = 0
    tracklets_mixed: int = 0
    fragments_per_fish: float = 0.0
    # (identity changes, steps) by how close a call the tracker's link was (MARGIN_BANDS)
    switches_by_margin: dict = field(default_factory=dict)
    # merging (B) and cards
    edges_eligible: int = 0
    edges_wrong: int = 0
    edges_inherited: int = 0  # touch a mixed tracklet: the tracker's error, not the merge's
    cards_eligible: int = 0
    cards_multi_fish: int = 0
    cards_multi_taxon: int = 0
    impure_detections: int = 0
    pure_eligible_detections: int = 0
    duplicate_cards_per_fish: float = 0.0
    cards_tracker_only: int = 0  # A
    cards_merge_only: int = 0  # B
    cards_both: int = 0  # C
    # science
    maxn: dict = field(default_factory=dict)  # taxon -> (annotated, from the run)

    @property
    def precision(self) -> float:
        return self.matched / (self.matched + self.false) if self.matched + self.false else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.fish if self.fish else 0.0

    def lines(self) -> list[str]:
        def rate(n, d):
            return f"{n}/{d} ({n / d:.0%})" if d else f"{n}/0"

        skipped = [
            f"{self.frames_unsampled} not sampled by this run" if self.frames_unsampled else "",
            f"{self.frames_unchecked} left out: rows still unchecked" if self.frames_unchecked else "",
        ]
        skipped = "; ".join(s for s in skipped if s)
        out = [
            f"{self.frames} annotated frames scored" + (f" ({skipped})" if skipped else ""),
            f"Detection: {rate(self.matched, self.fish)} fish found, {self.missed} missed; "
            f"{self.false} false detections ({self.false_on_equipment} on equipment); precision {self.precision:.0%}",
            "  recall by size (review px): " + ", ".join(f"{band} {rate(m, n)}" for band, (m, n) in self.recall_by_size.items()),
            f"Tracking (A): {self.switches} identity changes in {self.transitions} steps"
            + (f" ({1000 * self.switches / self.transitions:.1f} per 1,000)" if self.transitions else "")
            + f"; {rate(self.tracklets_mixed, self.tracklets_eligible)} tracklets mix fish; "
            f"{self.fragments_per_fish:.2f} tracklets per fish",
            "  identity changes by the tracker's margin: "
            + ", ".join(f"{band} {rate(s, n)}" for band, (s, n) in sorted(self.switches_by_margin.items())),
            f"Merging (B): {rate(self.edges_wrong, self.edges_eligible)} joins of pure tracklets put different fish together; "
            f"{self.edges_inherited} joins touch a mixed tracklet (counted under A)",
            f"Cards ({self.grouping} grouping): {rate(self.cards_multi_fish, self.cards_eligible)} hold more than one fish, "
            f"{rate(self.cards_multi_taxon, self.cards_eligible)} more than one taxon; "
            f"{rate(self.impure_detections, self.pure_eligible_detections)} detections sit on a card with another fish's majority; "
            f"{self.duplicate_cards_per_fish:.2f} extra cards per fish",
            f"  mixed cards by cause: tracker {self.cards_tracker_only}, merge {self.cards_merge_only}, both {self.cards_both}",
        ]
        if self.maxn:
            exact = sum(a == b for a, b in self.maxn.values())
            out.append(f"MaxN over the annotated frames: {exact} of {len(self.maxn)} taxa exact")
            for taxon, (truth, run) in sorted(self.maxn.items()):
                note = "" if truth == run else f"  ({run - truth:+d})"
                out.append(f"  {taxon}: annotated {truth}, from the run {run}{note}")
        return out


def _band(px: float) -> str:
    lower = 0
    for edge in SIZE_BANDS:
        if px < edge:
            return f"{lower}-{edge}"
        lower = edge
    return f"{SIZE_BANDS[-1]}+"


def evaluate(run: RunView, boxes: list[Box], min_iou: float = MIN_IOU) -> Report:
    report = Report(grouping=run.grouping)
    by_frame: dict[tuple, list[Box]] = defaultdict(list)
    for b in boxes:
        by_frame[b.frame].append(b)

    who: dict[int, Box] = {}  # detection id -> the fish box it matched
    sizes: dict[str, list[int]] = {f"{lo}-{hi}": [0, 0] for lo, hi in zip((0, *SIZE_BANDS), SIZE_BANDS)}
    sizes[f"{SIZE_BANDS[-1]}+"] = [0, 0]
    width, height = run.frame_size
    truth_counts: dict[tuple, Counter] = {}
    run_counts: dict[tuple, Counter] = {}
    for key, annotated in by_frame.items():
        if any(b.kind == "unchecked" for b in annotated):
            report.frames_unchecked += 1
            continue
        if key not in run.frames:
            report.frames_unsampled += 1
            continue
        report.frames += 1
        objects = [b for b in annotated if b.kind != "none"]
        found = run.frames[key]
        pairs = match_boxes([b.detection for b in objects], [f.box for f in found], min_iou)
        hit = {i: j for i, j in pairs}
        used = set(hit.values())
        for i, b in enumerate(objects):
            if b.kind != "fish":
                continue
            report.fish += 1
            band = _band(max((b.x2 - b.x1) * width, (b.y2 - b.y1) * height))
            sizes[band][1] += 1
            if i in hit:
                report.matched += 1
                sizes[band][0] += 1
                who[found[hit[i]].det_id] = b
            else:
                report.missed += 1
        for i, j in pairs:
            if objects[i].kind != "fish":
                report.false += 1
                report.false_on_equipment += objects[i].kind == "equipment"
        report.false += len(found) - len(used)
        report.found += len(found)
        truth_counts[key] = Counter(b.taxon for b in objects if b.kind == "fish" and b.taxon)
        run_counts[key] = Counter(f.label for f in found if f.label and f.label != UNKNOWN)
    report.recall_by_size = {band: tuple(v) for band, v in sizes.items() if v[1]}

    # Tracking (A): the matched fish along each tracklet, in time order.
    along: dict[int, list[tuple]] = defaultdict(list)
    found_by_id = {f.det_id: f for frame in run.frames.values() for f in frame}
    for det_id, b in who.items():
        f = found_by_id[det_id]
        if b.certain and f.tracklet is not None:
            along[f.tracklet].append((f.t, det_id, b.individual))
    fish_of: dict[int, set[str]] = {}
    by_margin: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for tracklet, seen in along.items():
        seen = sorted(seen)
        seq = [ind for _, _, ind in seen]
        report.transitions += len(seq) - 1
        report.switches += sum(a != b for a, b in zip(seq, seq[1:]))
        for (_, _, a), (_, det_id, b) in zip(seen, seen[1:]):
            band = by_margin[margin_band(found_by_id[det_id].margin)]
            band[0] += a != b
            band[1] += 1
        fish_of[tracklet] = set(seq)
    report.switches_by_margin = {k: tuple(v) for k, v in by_margin.items()}
    report.tracklets_eligible = len(fish_of)
    report.tracklets_mixed = sum(len(s) > 1 for s in fish_of.values())
    pieces: dict[str, set[int]] = defaultdict(set)
    for tracklet, inds in fish_of.items():
        for ind in inds:
            pieces[ind].add(tracklet)
    report.fragments_per_fish = float(np.mean([len(p) for p in pieces.values()])) if pieces else 0.0

    # Merging (B), and cards.
    card_dets: dict[int, list[Box]] = defaultdict(list)
    for det_id, b in who.items():
        card = found_by_id[det_id].card
        if card is not None and b.certain:
            card_dets[card].append(b)
    cards_of: dict[str, set[int]] = defaultdict(set)
    for card, members in card_dets.items():
        report.cards_eligible += 1
        inds = Counter(b.individual for b in members)
        taxa = {b.taxon for b in members if b.taxon}
        report.cards_multi_fish += len(inds) > 1
        report.cards_multi_taxon += len(taxa) > 1
        report.pure_eligible_detections += len(members)
        report.impure_detections += len(members) - inds.most_common(1)[0][1]
        for ind in inds:
            cards_of[ind].add(card)
        chain = run.chains.get(card, [])
        mixed = any(len(fish_of.get(t, ())) > 1 for t in chain)
        wrong = False
        for a, b in zip(chain, chain[1:]):
            fa, fb = fish_of.get(a), fish_of.get(b)
            if not fa or not fb:
                continue  # a piece with no annotated fish: the join can't be judged
            if len(fa) > 1 or len(fb) > 1:
                report.edges_inherited += 1
                continue
            report.edges_eligible += 1
            if fa != fb:
                report.edges_wrong += 1
                wrong = True
        if len(inds) > 1:
            report.cards_tracker_only += mixed and not wrong
            report.cards_merge_only += wrong and not mixed
            report.cards_both += wrong and mixed
    report.duplicate_cards_per_fish = float(np.mean([len(c) - 1 for c in cards_of.values()])) if cards_of else 0.0

    # Science: MaxN per taxon over the annotated frames.
    taxa = {t for c in truth_counts.values() for t in c} | {t for c in run_counts.values() for t in c}
    report.maxn = {
        taxon: (max((c[taxon] for c in truth_counts.values()), default=0), max((c[taxon] for c in run_counts.values()), default=0))
        for taxon in taxa
    }
    return report


@dataclass
class CurvePoint:
    threshold: float
    kept: int  # detections at or above the threshold, in all sampled frames
    matched: int  # annotated fish found (annotated frames only)
    missed: int
    false: int

    @property
    def precision(self) -> float:
        return self.matched / (self.matched + self.false) if self.matched + self.false else 0.0

    @property
    def recall(self) -> float:
        return self.matched / (self.matched + self.missed) if self.matched + self.missed else 0.0


def threshold_curve(project: Project, boxes: list[Box], thresholds: list[float], min_iou: float = MIN_IOU) -> list[CurvePoint]:
    """Detection at other score thresholds, from the boxes the run kept (Settings.det_floor),
    without running the detector again (review V2, E6). Only detection is scored here: how
    tracks and cards change needs the later stages re-run at each threshold. A threshold under
    the lowest score the run kept can't be scored and is left out."""
    project.fill_source_indices()
    deployment = project.get_meta("deployment") or ""
    floor = project.db.execute("SELECT MIN(score) FROM detections").fetchone()[0] or 0.0
    # Every detection run must have kept boxes down to a threshold for it to be scored.
    kept_from = max([json.loads(r["info"]).get("kept_from", floor) for r in project.runs("detect")] or [floor])
    rows = project.db.execute(
        """SELECT d.score, d.x1, d.y1, d.x2, d.y2, f.chapter, f.src_index
           FROM detections d JOIN frames_src f ON f.id=d.frame_id"""
    ).fetchall()
    found: dict[tuple, list] = defaultdict(list)
    for r in rows:
        found[(deployment, r["chapter"], r["src_index"])].append((r["score"], Detection(r["x1"], r["y1"], r["x2"], r["y2"], 1.0)))
    annotated: dict[tuple, list[Box]] = defaultdict(list)
    for b in boxes:
        if b.deployment == deployment:
            annotated[b.frame].append(b)
    checked = {k: v for k, v in annotated.items() if not any(b.kind == "unchecked" for b in v)}
    out = []
    for t in sorted(thresholds):
        if t < kept_from - 1e-9:
            continue
        point = CurvePoint(t, sum(s >= t for s, _ in (x for v in found.values() for x in v)), 0, 0, 0)
        for key, objects in checked.items():
            objects = [b for b in objects if b.kind != "none"]
            dets = [d for s, d in found.get(key, []) if s >= t]
            pairs = match_boxes([b.detection for b in objects], dets, min_iou)
            fish_hit = sum(objects[i].kind == "fish" for i, _ in pairs)
            point.matched += fish_hit
            point.missed += sum(b.kind == "fish" for b in objects) - fish_hit
            point.false += len(dets) - fish_hit
        out.append(point)
    return out


def evaluate_project(project: Project, boxes: list[Box], min_iou: float = MIN_IOU) -> Report:
    deployment = project.get_meta("deployment")
    others = {b.deployment for b in boxes} - {deployment}
    if others and not any(b.deployment == deployment for b in boxes):
        raise ValueError(
            f"The annotations are for {', '.join(sorted(others))}, but this project's footage is {deployment or 'not fingerprinted'}"
        )
    return evaluate(load_run(project), [b for b in boxes if b.deployment == deployment], min_iou)


def annotation_sheet(project: Project, windows: list[tuple[float, float]]) -> list[Box]:
    """Rows to start annotating from, for the frames in each (start, end) window of global
    seconds: one "unchecked" row per detection with its box only (no species guess), and one
    for each frame without detections. Annotators set every row's kind (fish, equipment,
    other; "none" for an empty frame), fix the boxes, add the fish the detector missed, and
    give each fish its individual and taxon."""
    project.fill_source_indices()
    deployment = project.get_meta("deployment") or ""
    rows = []
    for start, end in windows:
        frames = project.db.execute(
            "SELECT * FROM frames_src WHERE t >= ? AND t < ? ORDER BY seq", (start, end)
        ).fetchall()
        for f in frames:
            dets = project.db.execute(
                "SELECT * FROM detections WHERE frame_id=? AND tracklet_id IS NOT NULL ORDER BY id", (f["id"],)
            ).fetchall()
            base = dict(deployment=deployment, chapter=f["chapter"], src_index=f["src_index"], t=round(f["t"], 3))
            if not dets:
                rows.append(Box(**base, kind="unchecked", identity=""))
            for d in dets:
                rows.append(Box(**base, x1=round(d["x1"], 5), y1=round(d["y1"], 5), x2=round(d["x2"], 5),
                                y2=round(d["y2"], 5), kind="unchecked", identity="", det_id=str(d["id"])))
    return rows
