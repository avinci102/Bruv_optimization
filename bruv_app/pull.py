"""Bring a volunteer's work from the hosted review site into a project (docs/SPEC.md §16).

The site keeps each volunteer's answers and fixes as an event log (hosted/core.js). Pulling
replays one volunteer's log onto a project folder of the same video, made by the same
processing that was published: their answers land per detection (reviews.scope) with their
name (reviews.reviewer) and the time they gave them, and their splits, moves and "multiple
fish" become that copy's cards. A volunteer's copy of their own video is the review of
record; fixes stay traceable in the corrections table and the snapshots taken before and
after.

New cards made on the site carry the site's numbers; here they get this project's next
numbers (card numbers are never reused), and the mapping is kept in the run record.

Follows made on the site (following a fish, follow.py) are replayed in order through
follow.py: each keeps the sighting (or drawn box) it started from, its name on the site and
its times. What was said about boxes while following (box_fixes) comes with the
volunteer's name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import follow
from .pipeline import refresh_fish_guesses
from .store import Project


@dataclass
class PullReport:
    volunteer: str
    events: int = 0
    answers: int = 0
    fixes: int = 0
    undone: int = 0
    follows: int = 0  # follows kept (made on the site and not discarded)
    box_fixes: int = 0  # what was said about boxes while following (follow.fix_box)
    scope_mismatch: list[int] = field(default_factory=list)  # cards whose sightings differ from the site's
    new_cards: dict[int, int] = field(default_factory=dict)  # site card number -> this project's


def find_volunteer(export: dict, who: str) -> dict:
    for v in export["volunteers"]:
        if who in (str(v["id"]), v["name"], v.get("email") or ""):
            return v
    names = ", ".join(v["name"] for v in export["volunteers"])
    raise ValueError(f"No volunteer {who!r} in this export (it has: {names})")


def pull(project_root: Path, export_path: Path, who: str, trial: str | None = None) -> PullReport:
    """One volunteer's answers and fixes onto the project, or onto its rule trial `trial` when
    that's what was published (publish --trial)."""
    from .trials import open_trial  # noqa: PLC0415

    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    volunteer = find_volunteer(export, who)
    events = [e["event"] for e in export["events"] if e["volunteer_id"] == volunteer["id"]]
    project = Project(Path(project_root))
    if trial:
        project = open_trial(project, trial)
    published = project.get_meta("published")
    if published and published["video"] != export["video"]:
        raise ValueError(f"These answers are for video {export['video']}, but this project was published as "
                         f"{published['video']}: pull them into that video's project")
    settings = project.load_settings()
    reviewed_before = project.reviewed_count()
    report = PullReport(volunteer["name"], events=len(events))
    run_id = project.start_run("pull", settings, video=export["video"], volunteer=volunteer["name"],
                               events=len(events), reviewed_before=reviewed_before)
    project.snapshot(f"before-pull-{volunteer['name']}")
    ids: dict[int, int] = {}
    follows: dict[int, int] = {}  # site follow -> this project's

    def real(card: int) -> int:
        return ids.get(card, card)

    for e in events:
        kind = e.get("kind")
        if kind == "label":
            card = real(e["card"])
            if e.get("scope") and sorted(e["scope"]) != sorted(d["id"] for d in project.fish_detections(card)):
                report.scope_mismatch.append(card)
            details = dict(e.get("details") or {})
            details["site"] = {"card": e["card"], "at": e.get("at")}
            project.label_fish(
                card, e["status"], details, species=e.get("species"), family=e.get("family"), type=e.get("type"),
                sex=e.get("sex"), behavior=e.get("behavior"), comment=e.get("comment"), reviewer=volunteer["name"],
                at=e.get("at"),
            )
            report.answers += e["status"] != "pending"
        elif kind == "undo":
            for c in e.get("cards", []):
                project.label_fish(real(c), "pending", {"undo": e.get("batch"), "site": {"at": e.get("at")}},
                                   reviewer=volunteer["name"], at=e.get("at"))
                report.undone += 1
        elif kind in ("split", "move"):
            card = real(e["card"])
            new = (project.split_fish(card, e["from_t"]) if kind == "split"
                   else project.move_to_new_fish(card, e["detections"]))
            ids[e["new_card"]] = new
            for fid in (card, new):
                refresh_fish_guesses(project, fid, settings)
            project.commit()
            report.fixes += 1
        elif kind == "unmerge":
            card = real(e["card"])
            news = project.unmerge_fish(card)
            ids.update(zip(e["new_cards"], news))
            for fid in (card, *news):
                refresh_fish_guesses(project, fid, settings)
            project.commit()
            report.fixes += 1
        elif kind == "suggestions":
            project.log_suggestions([(real(f), real(s), sim, decision, batch) for f, s, sim, decision, batch in e["rows"]],
                                   at=e.get("at"))
        elif kind == "follow_start":
            card = None if e.get("card") is None else real(e["card"])
            box = (e["x"], e["y"], e["w"], e["h"]) if e.get("det") is None else None
            follows[e["follow"]] = follow.start(project, card, e["gone_s"], det_id=e.get("det"), name=e["name"],
                                                at=e.get("at"), frame_id=e["frame"] if box else None, box=box)
        elif kind == "box_fix":
            follow.fix_box(project, e["det"], e.get("fix"), follow_id=follows.get(e.get("follow")),
                           reviewer=volunteer["name"], at=e.get("at"))
            report.box_fixes += 1
        elif kind in ("follow_mark", "follow_rest", "follow_undo", "follow_delete") and e["follow"] in follows:
            fid = follows[e["follow"]]
            if kind == "follow_mark":
                follow.mark(project, fid, e["frame"], e["state"], e.get("det"), e.get("x"), e.get("y"), at=e.get("at"),
                            w=e.get("w"), h=e.get("h"), via=e.get("via"))
            elif kind == "follow_rest":
                follow.mark_rest(project, fid, e["frame"], e["frames"], at=e.get("at"))
            elif kind == "follow_undo":
                follow.undo(project, fid)
            else:
                follow.discard(project, fid)
                del follows[e["follow"]]
    for fid in follows.values():
        follow.next_step(project, fid)  # done or not, as on the site
    report.follows = len(follows)
    report.new_cards = {k: v for k, v in ids.items() if k != v}
    project.finish_run(run_id, answers=report.answers, fixes=report.fixes, undone=report.undone, follows=report.follows,
                       box_fixes=report.box_fixes,
                       new_cards={str(k): v for k, v in report.new_cards.items()},
                       scope_mismatch=report.scope_mismatch)
    project.snapshot(f"pulled-{volunteer['name']}")
    return report
