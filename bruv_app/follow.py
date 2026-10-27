"""Follow one fish: clicked ground truth for continuity (docs/SPEC.md §15).

A person follows one fish frame by frame through the sampled frames. In each frame they
click its box (any box, faint ones and still ones too), click the fish itself where it has
no box, press a key for "here but no box" without placing it, or say it's gone. A click
moves on to the next frame. The follow starts from the card's best view (the clearest
picture, so the person knows which fish it is), goes forward until the fish has been gone
for `gone_s` seconds (in case it comes back), then backward from the start the same way.

Each follow is one individual's visit. Its marks go to the evaluation sidecar as rows of
scope "individual" (evaluate.py): only that fish was annotated in those frames, so they
score identity (tracker, merge, cards) and not the detector's false boxes.

The next frame to show is worked out from the marks each time, so a follow can be undone
step by step, left and resumed.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .evaluate import Box
from .store import Project
from .views import best_view

GONE_S = 5.0  # keep stepping this long after the fish has gone, in case it comes back
STATES = ("box", "nobox", "gone")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def copy_tag(project: Project) -> str:
    """A short tag for this copy of the project, so follows made in volunteers' copies of one
    deployment get different names."""
    tag = project.get_meta("copy_tag")
    if not tag:
        tag = secrets.token_hex(2)
        project.set_meta("copy_tag", tag)
    return tag


def name(project: Project, follow_id: int) -> str:
    """F<id>-<this copy's tag>, or the name a follow pulled from the hosted site had there."""
    row = project.db.execute("SELECT name FROM follows WHERE id=?", (follow_id,)).fetchone()
    if row is not None and row["name"]:
        return row["name"]
    return f"F{follow_id}-{copy_tag(project)}"


def start(project: Project, card: int, gone_s: float = GONE_S, det_id: int | None = None,
          name: str | None = None, at: str | None = None) -> int:
    """A new follow of the fish on `card`, from its best view. A follow pulled from the hosted
    site (pull.py) brings the sighting it started from (`det_id`), its name and its time: the
    card may hold other sightings by then, so its best view isn't worked out again."""
    if det_id is None:
        det = best_view(project.fish_detections(card))
        if det is None:
            raise ValueError(f"Card #{card} has no sightings")
    else:
        det = project.db.execute("SELECT id, frame_id FROM detections WHERE id=?", (det_id,)).fetchone()
        if det is None:
            raise ValueError(f"No sighting {det_id} to start from")
    copy_tag(project)  # named now, so a read-only copy can name its follows later
    cur = project.db.execute(
        "INSERT INTO follows(at, card, start_det, gone_s, name) VALUES(?, ?, ?, ?, ?)",
        (at or _now(), card, det["id"], gone_s, name),
    )
    follow_id = cur.lastrowid
    mark(project, follow_id, det["frame_id"], "box", det_id=det["id"], at=at)
    return follow_id


def mark(project: Project, follow_id: int, frame_id: int, state: str, det_id: int | None = None,
         x: float | None = None, y: float | None = None, at: str | None = None) -> None:
    """What the person said about one frame: the fish's box (`det_id`), no box (where the
    fish was, `x` `y`, if they clicked it), or gone. `at`: when, for marks pulled from the site."""
    if state not in STATES:
        raise ValueError(f"A mark is one of {', '.join(STATES)}")
    if state == "box":
        d = project.db.execute("SELECT frame_id FROM detections WHERE id=?", (det_id,)).fetchone()
        if d is None or d["frame_id"] != frame_id:
            raise ValueError("That box isn't in this frame")
        x = y = None
    else:
        det_id = None
        if state == "gone":
            x = y = None
    # INSERT OR REPLACE gives the row a new rowid, so the latest mark is always the last.
    project.db.execute(
        "INSERT OR REPLACE INTO follow_marks(follow_id, frame_id, state, det_id, x, y, at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        (follow_id, frame_id, state, det_id, x, y, at or _now()),
    )
    project.db.execute("UPDATE follows SET done_at=NULL WHERE id=?", (follow_id,))
    project.commit()


def undo(project: Project, follow_id: int) -> int | None:
    """Take back the latest mark (never the starting one). Returns its frame, or None."""
    row = project.db.execute(
        """SELECT m.rowid, m.frame_id FROM follow_marks m JOIN follows f ON f.id=m.follow_id
           LEFT JOIN detections d ON d.id=f.start_det
           WHERE m.follow_id=? AND m.frame_id IS NOT d.frame_id ORDER BY m.rowid DESC LIMIT 1""",
        (follow_id,),
    ).fetchone()
    if row is None:
        return None
    project.db.execute("DELETE FROM follow_marks WHERE rowid=?", (row["rowid"],))
    project.db.execute("UPDATE follows SET done_at=NULL WHERE id=?", (follow_id,))
    project.commit()
    return row["frame_id"]


def discard(project: Project, follow_id: int) -> None:
    project.db.execute("DELETE FROM follow_marks WHERE follow_id=?", (follow_id,))
    project.db.execute("DELETE FROM follows WHERE id=?", (follow_id,))
    project.commit()


@dataclass
class Step:
    phase: str  # forward | backward | done
    frame_id: int | None
    last: tuple[float, float] | None  # where the fish was last seen, going this way


def _frames(project: Project) -> list:
    return project.db.execute("SELECT id, seq, t FROM frames ORDER BY seq").fetchall()


def _spot(project: Project, m) -> tuple[float, float] | None:
    """The middle of a mark's box, or the spot clicked."""
    if m["state"] == "box":
        d = project.db.execute("SELECT x1, y1, x2, y2 FROM detections WHERE id=?", (m["det_id"],)).fetchone()
        return ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2) if d else None
    if m["state"] == "nobox" and m["x"] is not None:
        return (m["x"], m["y"])
    return None


def next_step(project: Project, follow_id: int) -> Step:
    """The next frame to look at: forward from the start until the fish has been gone for
    the follow's gone_s, then backward the same way."""
    f = project.db.execute(
        "SELECT f.*, d.frame_id AS start_frame FROM follows f JOIN detections d ON d.id=f.start_det WHERE f.id=?",
        (follow_id,),
    ).fetchone()
    if f is None:
        raise ValueError(f"No follow {follow_id}")
    frames = _frames(project)
    index = {r["id"]: i for i, r in enumerate(frames)}
    marks = {m["frame_id"]: m for m in project.db.execute("SELECT * FROM follow_marks WHERE follow_id=?", (follow_id,))}
    s = index[f["start_frame"]]
    for phase, order in (("forward", range(s + 1, len(frames))), ("backward", range(s - 1, -1, -1))):
        seen_t = frames[s]["t"]
        last = _spot(project, marks[f["start_frame"]])
        for i in order:
            fr = frames[i]
            if abs(fr["t"] - seen_t) > f["gone_s"] + 1e-6:
                break  # gone long enough: this way is finished
            m = marks.get(fr["id"])
            if m is None:
                return Step(phase, fr["id"], last)
            if m["state"] != "gone":
                seen_t = fr["t"]
                last = _spot(project, m) or last
    if f["done_at"] is None:
        project.db.execute("UPDATE follows SET done_at=? WHERE id=?", (_now(), follow_id))
        project.commit()
    return Step("done", None, None)


def frame_boxes(project: Project, frame_id: int, follow_id: int) -> list[dict]:
    """Every box in a frame: tracked ones, and those below the operating threshold or hidden
    as still (marked faint), with the card each is on and any other follow that took it."""
    taken = {
        r["det_id"]: r["follow_id"]
        for r in project.db.execute(
            "SELECT det_id, follow_id FROM follow_marks WHERE frame_id=? AND state='box' AND follow_id<>?",
            (frame_id, follow_id),
        )
    }
    out = []
    for d in project.db.execute(
        """SELECT d.id, d.x1, d.y1, d.x2, d.y2, d.score, d.tracklet_id, t.fish_id FROM detections d
           LEFT JOIN tracklets t ON t.id=d.tracklet_id WHERE d.frame_id=? ORDER BY d.score DESC""",
        (frame_id,),
    ):
        out.append({
            "det": d["id"], "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
            "score": round(d["score"], 2), "card": d["fish_id"], "faint": d["tracklet_id"] is None,
            "taken": None if d["id"] not in taken else name(project, taken[d["id"]]),
        })
    return out


@dataclass
class Summary:
    """What one follow found, against the cards as they are now."""

    id: int
    name: str
    card: int
    done: bool
    gone_s: float = GONE_S
    boxes: int = 0  # frames where the fish's box was clicked
    nobox: int = 0  # frames where the fish was there without a box
    gone: int = 0
    first_t: float | None = None
    last_t: float | None = None
    cards: dict = field(default_factory=dict)  # card -> frames of the fish's boxes on it
    faint: int = 0  # boxes clicked that aren't on any card (faint, or hidden as still)
    breaks: list = field(default_factory=list)  # (t, from card, to card) where the card changes
    shared: dict = field(default_factory=dict)  # card -> other follows with boxes on it
    # card -> frames where the card has a box that isn't this fish (the person clicked
    # another box, said it had no box, or said it was gone): the card holds something else
    others: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def summary(project: Project, follow_id: int) -> Summary:
    f = project.db.execute("SELECT * FROM follows WHERE id=?", (follow_id,)).fetchone()
    s = Summary(follow_id, name(project, follow_id), f["card"], f["done_at"] is not None, f["gone_s"])
    rows = project.db.execute(
        """SELECT m.state, fr.t, t.fish_id AS card FROM follow_marks m JOIN frames fr ON fr.id=m.frame_id
           LEFT JOIN detections d ON d.id=m.det_id LEFT JOIN tracklets t ON t.id=d.tracklet_id
           WHERE m.follow_id=? ORDER BY fr.seq""",
        (follow_id,),
    ).fetchall()
    previous = None
    for r in rows:
        if r["state"] == "gone":
            s.gone += 1
            continue
        s.first_t = r["t"] if s.first_t is None else s.first_t
        s.last_t = r["t"]
        if r["state"] == "nobox":
            s.nobox += 1
            continue
        s.boxes += 1
        if r["card"] is None:
            s.faint += 1
            continue
        s.cards[r["card"]] = s.cards.get(r["card"], 0) + 1
        if previous is not None and r["card"] != previous:
            s.breaks.append((r["t"], previous, r["card"]))
        previous = r["card"]
    if s.cards:
        marks = ",".join("?" * len(s.cards))
        for r in project.db.execute(
            f"""SELECT DISTINCT t.fish_id AS card, m.follow_id FROM follow_marks m JOIN detections d ON d.id=m.det_id
                JOIN tracklets t ON t.id=d.tracklet_id WHERE m.state='box' AND m.follow_id<>? AND t.fish_id IN ({marks})""",
            (follow_id, *s.cards),
        ):
            s.shared.setdefault(r["card"], []).append(name(project, r["follow_id"]))
        for r in project.db.execute(
            f"""SELECT t.fish_id AS card, COUNT(DISTINCT m.frame_id) AS n FROM follow_marks m
                JOIN detections d ON d.frame_id=m.frame_id JOIN tracklets t ON t.id=d.tracklet_id
                WHERE m.follow_id=? AND t.fish_id IN ({marks}) AND (m.det_id IS NULL OR d.id<>m.det_id)
                GROUP BY t.fish_id""",
            (follow_id, *s.cards),
        ):
            s.others[r["card"]] = r["n"]
    return s


def all_follows(project: Project) -> list[Summary]:
    return [summary(project, r["id"]) for r in project.db.execute("SELECT id FROM follows ORDER BY id")]


def sidecar_rows(project: Project) -> list[Box]:
    """The follows as evaluation sidecar rows of scope "individual": the clicked box, the
    spot where the fish was without a box (a point; nothing matches it, so it counts as not
    found), or kind "none" where it was gone. The taxon is the starting card's answer."""
    if project.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='follows'").fetchone() is None:
        return []  # a copy made before follows existed
    project.fill_source_indices()
    deployment = project.get_meta("deployment") or ""
    rows = []
    for f in project.db.execute("SELECT * FROM follows ORDER BY id"):
        card = project.fish(f["card"])
        taxon = card["species"] if card is not None and card["status"] == "species" else ""
        who = name(project, f["id"])
        for m in project.db.execute(
            """SELECT m.*, fr.chapter, fr.src_index, fr.t, d.x1, d.y1, d.x2, d.y2 FROM follow_marks m
               JOIN frames_src fr ON fr.id=m.frame_id LEFT JOIN detections d ON d.id=m.det_id
               WHERE m.follow_id=? ORDER BY fr.seq""",
            (f["id"],),
        ):
            base = dict(deployment=deployment, chapter=m["chapter"], src_index=m["src_index"], t=round(m["t"], 3),
                        individual=who, scope="individual", annotator="follow")
            if m["state"] == "box":
                rows.append(Box(**base, x1=round(m["x1"], 5), y1=round(m["y1"], 5), x2=round(m["x2"], 5),
                                y2=round(m["y2"], 5), taxon=taxon, det_id=str(m["det_id"])))
            elif m["state"] == "nobox":
                x, y = (m["x"], m["y"]) if m["x"] is not None else (0.0, 0.0)
                rows.append(Box(**base, x1=x, y1=y, x2=x, y2=y, taxon=taxon,
                                note="no box" if m["x"] is not None else "no box, not placed"))
            else:
                rows.append(Box(**base, kind="none"))
    return rows
