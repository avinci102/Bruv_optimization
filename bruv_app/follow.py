"""Follow one fish: clicked ground truth for continuity (docs/SPEC.md §15).

A person follows one fish frame by frame through the sampled frames. In each frame they
click its box (any box, faint ones and still ones too), click the fish itself where it has
no box (or drag a box around it), press a key for "here but no box" without placing it, or
say it's gone. A click moves on to the next frame. The follow starts from a card's best view
(the clearest picture, so the person knows which fish it is), or from any box, or a box drawn
around a fish, picked while looking through the video; it goes forward until the fish has
been gone for `gone_s` seconds (in case it comes back), then backward from the start the
same way. "Gone for good" says it's gone in every frame left that way at once.

Each follow is one individual's visit. Its marks go to the evaluation sidecar as rows of
scope "individual" (evaluate.py): only that fish was annotated in those frames, so they
score identity (tracker, merge, cards) and not the detector's false boxes.

While following, a person can also say what's wrong with a box (box_fixes): part of a fish
that has its own box, not a fish, or a second box on one fish. These are kept for measuring
the detector; they change nothing.

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
FIXES = ("part", "not_fish", "twice")  # what can be wrong with a box (box_fixes)
RECENT = 5  # close-ups of the fish shown while following: its boxes nearest the frame


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


def card_of(project: Project, det_id: int) -> int | None:
    """The card a box is on now (none for faint boxes and those hidden as still)."""
    row = project.db.execute(
        "SELECT t.fish_id FROM detections d LEFT JOIN tracklets t ON t.id=d.tracklet_id WHERE d.id=?", (det_id,)
    ).fetchone()
    return None if row is None else row["fish_id"]


def drawn(x, y, w, h) -> bool:
    """A box drawn around a fish: its middle and size, all inside the frame."""
    return all(isinstance(v, (int, float)) for v in (x, y, w, h)) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def start(project: Project, card: int | None, gone_s: float = GONE_S, det_id: int | None = None,
          name: str | None = None, at: str | None = None, frame_id: int | None = None,
          box: tuple[float, float, float, float] | None = None) -> int:
    """A new follow of the fish on `card`, from its best view; or from the box `det_id`
    (`card` is then the card it's on); or from `box` (x, y, w, h: a box drawn around a fish
    with none) in `frame_id`. A follow pulled from the hosted site (pull.py) brings the
    sighting it started from, its name and its time: the card may hold other sightings by
    then, so its best view isn't worked out again."""
    if box is not None:
        if not drawn(*box) or project.frame(frame_id) is None:
            raise ValueError("A drawn box needs a frame and its middle and size, inside the frame")
        det = {"id": None, "frame_id": frame_id}
    elif det_id is None:
        det = best_view(project.fish_detections(card)) if card is not None else None
        if det is None:
            raise ValueError(f"Card #{card} has no sightings")
    else:
        det = project.db.execute("SELECT id, frame_id FROM detections WHERE id=?", (det_id,)).fetchone()
        if det is None:
            raise ValueError(f"No sighting {det_id} to start from")
    copy_tag(project)  # named now, so a read-only copy can name its follows later
    cur = project.db.execute(
        "INSERT INTO follows(at, card, start_det, gone_s, name, start_frame) VALUES(?, ?, ?, ?, ?, ?)",
        (at or _now(), card, det["id"], gone_s, name, det["frame_id"]),
    )
    follow_id = cur.lastrowid
    if box is None:
        mark(project, follow_id, det["frame_id"], "box", det_id=det["id"], at=at)
    else:
        mark(project, follow_id, frame_id, "nobox", x=box[0], y=box[1], w=box[2], h=box[3], at=at)
    return follow_id


def mark(project: Project, follow_id: int, frame_id: int, state: str, det_id: int | None = None,
         x: float | None = None, y: float | None = None, at: str | None = None, w: float | None = None,
         h: float | None = None, via: str | None = None, batch: int | None = None) -> None:
    """What the person said about one frame: the fish's box (`det_id`), no box (where the
    fish was, `x` `y`, if they clicked it, and `w` `h` if they drew a box around it), or gone.
    `via`: "suggested" when they took the box the page offered. `at`: when, for marks pulled
    from the site."""
    if state not in STATES:
        raise ValueError(f"A mark is one of {', '.join(STATES)}")
    if state == "box":
        d = project.db.execute("SELECT frame_id FROM detections WHERE id=?", (det_id,)).fetchone()
        if d is None or d["frame_id"] != frame_id:
            raise ValueError("That box isn't in this frame")
        x = y = w = h = None
    else:
        det_id = None
        if state == "gone":
            x = y = None
        if not drawn(x, y, w, h):
            w = h = None
    if via not in ("suggested", "rest") or (via == "suggested" and state != "box"):
        via = None
    # INSERT OR REPLACE gives the row a new rowid, so the latest mark is always the last.
    project.db.execute(
        "INSERT OR REPLACE INTO follow_marks(follow_id, frame_id, state, det_id, x, y, at, w, h, via, batch) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (follow_id, frame_id, state, det_id, x, y, at or _now(), w, h, via, batch),
    )
    project.db.execute("UPDATE follows SET done_at=NULL WHERE id=?", (follow_id,))
    project.commit()


def _follow(project: Project, follow_id: int):
    """The follow, with the frame it started from (`first_frame`)."""
    return project.db.execute(
        "SELECT f.*, COALESCE(f.start_frame, d.frame_id) AS first_frame FROM follows f "
        "LEFT JOIN detections d ON d.id=f.start_det WHERE f.id=?",
        (follow_id,),
    ).fetchone()


def undo(project: Project, follow_id: int) -> int | None:
    """Take back the latest mark (never the starting one), or all the marks of the "gone for
    good" it was one of. Returns its frame, or None."""
    f = _follow(project, follow_id)
    row = project.db.execute(
        "SELECT rowid, frame_id, batch FROM follow_marks WHERE follow_id=? AND frame_id IS NOT ? ORDER BY rowid DESC LIMIT 1",
        (follow_id, f["first_frame"] if f else None),
    ).fetchone()
    if row is None:
        return None
    if row["batch"] is not None:
        project.db.execute("DELETE FROM follow_marks WHERE follow_id=? AND batch=?", (follow_id, row["batch"]))
    else:
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


def frames(project: Project) -> list:
    """Every sampled frame, in order."""
    return project.db.execute("SELECT id, seq, t FROM frames ORDER BY seq").fetchall()


def _spot(project: Project, m) -> tuple[float, float] | None:
    """The middle of a mark's box, or the spot clicked."""
    if m is None:
        return None
    if m["state"] == "box":
        d = project.db.execute("SELECT x1, y1, x2, y2 FROM detections WHERE id=?", (m["det_id"],)).fetchone()
        return ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2) if d else None
    if m["state"] == "nobox" and m["x"] is not None:
        return (m["x"], m["y"])
    return None


def _walk(project: Project, follow_id: int):
    """Where a follow has got to: (phase, the next frame's place in `frames`, the time the fish
    was last seen going this way, where it was), the frames, and the marks by frame."""
    f = _follow(project, follow_id)
    if f is None:
        raise ValueError(f"No follow {follow_id}")
    every = frames(project)
    index = {r["id"]: i for i, r in enumerate(every)}
    marks = {m["frame_id"]: m for m in project.db.execute("SELECT * FROM follow_marks WHERE follow_id=?", (follow_id,))}
    s = index[f["first_frame"]]
    for phase, order in (("forward", range(s + 1, len(every))), ("backward", range(s - 1, -1, -1))):
        seen_t = every[s]["t"]
        last = _spot(project, marks.get(f["first_frame"]))
        for i in order:
            fr = every[i]
            if abs(fr["t"] - seen_t) > f["gone_s"] + 1e-6:
                break  # gone long enough: this way is finished
            m = marks.get(fr["id"])
            if m is None:
                return (phase, i, seen_t, last), f, every, marks
            if m["state"] != "gone":
                seen_t = fr["t"]
                last = _spot(project, m) or last
    return ("done", None, None, None), f, every, marks


def next_step(project: Project, follow_id: int) -> Step:
    """The next frame to look at: forward from the start until the fish has been gone for
    the follow's gone_s, then backward the same way."""
    (phase, i, _, last), f, every, _ = _walk(project, follow_id)
    if phase != "done":
        return Step(phase, every[i]["id"], last)
    if f["done_at"] is None:
        project.db.execute("UPDATE follows SET done_at=? WHERE id=?", (_now(), follow_id))
        project.commit()
    return Step("done", None, None)


def rest_frames(project: Project, follow_id: int) -> list[int]:
    """"Gone for good": the next frame and every frame after it, this way, that would be
    looked at before the follow turns (or ends) if the fish stays gone."""
    (phase, i, seen_t, _), f, every, marks = _walk(project, follow_id)
    if phase == "done":
        return []
    step = 1 if phase == "forward" else -1
    out = []
    while 0 <= i < len(every) and abs(every[i]["t"] - seen_t) <= f["gone_s"] + 1e-6:
        if every[i]["id"] not in marks:
            out.append(every[i]["id"])
        i += step
    return out


def mark_rest(project: Project, follow_id: int, frame_id: int, frame_ids: list[int] | None = None,
              at: str | None = None) -> list[int]:
    """The fish is gone for good, from `frame_id` (the next frame to look at) on, this way:
    marks it gone there and in the rest of the frames rest_frames lists (or `frame_ids`, as
    the hosted site worked them out), as one batch that undo takes back whole."""
    if frame_ids is None:
        frame_ids = rest_frames(project, follow_id)
        if not frame_ids or frame_ids[0] != frame_id:
            raise ValueError("That isn't the frame to look at now")
    for fid in frame_ids:
        mark(project, follow_id, fid, "gone", at=at, via="rest", batch=frame_id)
    return frame_ids


def frame_boxes(project: Project, frame_id: int, follow_id: int) -> list[dict]:
    """Every box in a frame: tracked ones, and those below the operating threshold or hidden
    as still (marked faint), with the card each is on, any other follow that took it, and
    what a person said is wrong with it (box_fixes)."""
    taken = {
        r["det_id"]: r["follow_id"]
        for r in project.db.execute(
            "SELECT det_id, follow_id FROM follow_marks WHERE frame_id=? AND state='box' AND follow_id<>?",
            (frame_id, follow_id),
        )
    }
    fixes = box_fixes(project, frame_id)
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
            "fix": fixes.get(d["id"]),
        })
    return out


def box_fixes(project: Project, frame_id: int) -> dict[int, str]:
    """What holds for each box in a frame that a person said something about."""
    latest = {}
    for r in project.db.execute(
        "SELECT x.det_id, x.kind FROM box_fixes x JOIN detections d ON d.id=x.det_id WHERE d.frame_id=? ORDER BY x.id",
        (frame_id,),
    ):
        latest[r["det_id"]] = r["kind"]
    return {det: kind for det, kind in latest.items() if kind}


def fix_box(project: Project, det_id: int, kind: str | None, follow_id: int | None = None,
            reviewer: str | None = None, at: str | None = None) -> None:
    """Say what's wrong with a box (FIXES), or take it back (`kind` None)."""
    if kind is not None and kind not in FIXES:
        raise ValueError(f"A box is fixed as one of {', '.join(FIXES)}")
    if project.db.execute("SELECT 1 FROM detections WHERE id=?", (det_id,)).fetchone() is None:
        raise ValueError(f"No box {det_id}")
    project.db.execute(
        "INSERT INTO box_fixes(at, det_id, kind, follow_id, reviewer) VALUES(?, ?, ?, ?, ?)",
        (at or _now(), det_id, kind, follow_id, reviewer),
    )
    project.commit()


def mark_box(project: Project, m) -> tuple[float, float, float, float] | None:
    """A mark's box: the fish's box, or the one drawn around it."""
    if m["state"] == "box":
        d = project.db.execute("SELECT x1, y1, x2, y2 FROM detections WHERE id=?", (m["det_id"],)).fetchone()
        return (d["x1"], d["y1"], d["x2"], d["y2"]) if d else None
    if m["state"] == "nobox" and m["w"] is not None:
        return (m["x"] - m["w"] / 2, m["y"] - m["h"] / 2, m["x"] + m["w"] / 2, m["y"] + m["h"] / 2)
    return None


def recent(project: Project, follow_id: int, frame_id: int, n: int = RECENT) -> list[tuple[int, tuple]]:
    """The fish's boxes (clicked or drawn) in the n frames nearest `frame_id`, nearest first:
    close-ups to remember it by."""
    every = frames(project)
    index = {r["id"]: i for i, r in enumerate(every)}
    here = index.get(frame_id, 0)
    found = []
    for m in project.db.execute("SELECT * FROM follow_marks WHERE follow_id=? AND state IN ('box', 'nobox')", (follow_id,)):
        box = mark_box(project, m)
        if box is not None and m["frame_id"] in index:
            found.append((abs(index[m["frame_id"]] - here), index[m["frame_id"]], m["frame_id"], box))
    return [(fid, box) for _, _, fid, box in sorted(found)[:n]]


@dataclass
class Summary:
    """What one follow found, against the cards as they are now."""

    id: int
    name: str
    card: int | None  # the card it started from (none when it started from a faint or drawn box)
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
    """The follows as evaluation sidecar rows of scope "individual": the clicked box, the box
    drawn where the fish had none, the spot where it was without a box (a point; nothing
    matches it, so it counts as not found), or kind "none" where it was gone. The taxon is
    the starting card's answer."""
    if project.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='follows'").fetchone() is None:
        return []  # a copy made before follows existed
    project.fill_source_indices()
    deployment = project.get_meta("deployment") or ""
    rows = []
    for f in project.db.execute("SELECT * FROM follows ORDER BY id"):
        card = project.fish(f["card"]) if f["card"] is not None else None
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
            w = m["w"] if "w" in m.keys() else None  # a copy made before boxes could be drawn
            if m["state"] == "box":
                rows.append(Box(**base, x1=round(m["x1"], 5), y1=round(m["y1"], 5), x2=round(m["x2"], 5),
                                y2=round(m["y2"], 5), taxon=taxon, det_id=str(m["det_id"])))
            elif m["state"] == "nobox" and w is not None:
                x, y, h = m["x"], m["y"], m["h"]
                rows.append(Box(**base, x1=round(x - w / 2, 5), y1=round(y - h / 2, 5), x2=round(x + w / 2, 5),
                                y2=round(y + h / 2, 5), taxon=taxon, note="no box, drawn"))
            elif m["state"] == "nobox":
                x, y = (m["x"], m["y"]) if m["x"] is not None else (0.0, 0.0)
                rows.append(Box(**base, x1=x, y1=y, x2=x, y2=y, taxon=taxon,
                                note="no box" if m["x"] is not None else "no box, not placed"))
            else:
                rows.append(Box(**base, kind="none"))
    return rows
