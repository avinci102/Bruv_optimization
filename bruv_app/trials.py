"""Rule trials: the project rebuilt with other rules, scored against the work in it.

A trial is a copy of the project's database in trials/<name>.db, with its own settings
(trials/<name>.settings.json), sharing the project's frames. try_rules makes one and
rebuilds its cards with the changed rules (Pipeline.rebuild); the project itself isn't
touched. compare scores every trial against what the reviewer did in the project:

- answers: cards that now hold sightings answered differently (two fish on one card);
- fixes: sightings the reviewer pulled apart (split, move, multiple fish) that are on one
  card again;
- followed fish (follow.py): how much of each visit was boxed, cards per fish, why a fish's
  card changed, and cards holding another box where the fish wasn't.

The "baseline" trial is the project rebuilt with its own rules, so every trial is compared
with cards the machine built, not with cards the reviewer has since fixed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .store import Project

FOLDER = "trials"
BASELINE = "baseline"


def trial_names(project: Project) -> list[str]:
    folder = project.root / FOLDER
    names = sorted(p.stem for p in folder.glob("*.db")) if folder.exists() else []
    return sorted(names, key=lambda n: n != BASELINE)


def open_trial(project: Project, name: str) -> Project:
    path = project.root / FOLDER / f"{name}.db"
    if not path.exists():
        raise ValueError(f"No trial called {name} (trials: {', '.join(trial_names(project)) or 'none yet'})")
    return Project(project.root, db=f"{FOLDER}/{name}.db")


def _value(field_type, text: str):
    kind = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    if "bool" in kind:
        if text.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if text.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{text} isn't true or false")
    if "float" in kind:
        return float(text)
    if "int" in kind:
        return int(text)
    return text


def apply_changes(settings: Settings, changes: dict[str, str]) -> Settings:
    """Settings with `changes` ({name: text}) applied, each read as its setting's type."""
    types = {f.name: f.type for f in fields(Settings)}
    for key, text in changes.items():
        if key not in types:
            raise ValueError(f"No setting called {key}")
        setattr(settings, key, _value(types[key], text))
    return settings


def try_rules(project: Project, name: str, changes: dict[str, str], classifier=None, progress=None) -> Project:
    """A trial called `name`: the project's database copied to trials/, its settings with
    `changes`, and its cards rebuilt with them. A trial of the same name is replaced."""
    from .pipeline import Pipeline  # noqa: PLC0415

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("A trial's name is letters, digits, - and _ (up to 40)")
    settings = apply_changes(project.load_settings(), changes)
    folder = project.root / FOLDER
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.db"
    path.unlink(missing_ok=True)
    project.commit()
    with sqlite3.connect(path) as copy:
        project.db.backup(copy)
    copy.close()
    trial = Project(project.root, db=f"{FOLDER}/{name}.db")
    trial.save_settings(settings)
    trial.set_meta("trial", {"name": name, "changes": changes, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    Pipeline(trial, classifier=classifier, progress=progress).rebuild(snapshots=False)
    return trial


@dataclass
class TrialScore:
    name: str
    changes: dict = field(default_factory=dict)
    cards: int = 0
    to_review: int = 0  # not tiny, not outside the review window
    checked: int = 0  # cards holding answered sightings
    mixed: int = 0  # of them, cards holding sightings answered differently
    pulled_apart: int = 0
    rejoined: int = 0
    followed: int = 0
    follow_present: int = 0
    follow_found: int = 0
    cards_per_followed: float = 0.0
    card_changes: int = 0
    cards_followed: int = 0
    cards_with_another: int = 0


def score_trial(trial: Project, project: Project) -> TrialScore:
    from .evaluate import evaluate_project  # noqa: PLC0415
    from .follow import sidecar_rows  # noqa: PLC0415
    from .score import score_against  # noqa: PLC0415

    meta = trial.get_meta("trial") or {}
    s = TrialScore(meta.get("name", "?"), meta.get("changes", {}))
    card = score_against(trial, project)
    outside = trial.outside_window()
    sizes = trial.fish_sizes()
    tiny_px = trial.load_settings().tiny_fish_px
    s.cards = card.cards
    s.to_review = sum(1 for f in trial.all_fish() if sizes.get(f["id"], 0.0) >= tiny_px and f["id"] not in outside)
    s.checked, s.mixed = card.checked, len(card.mixed)
    s.pulled_apart, s.rejoined = card.pulled_apart, card.rejoined
    follows = sidecar_rows(project)
    if follows:
        r = evaluate_project(trial, follows, follows=False)
        s.followed, s.follow_present, s.follow_found = r.followed, r.follow_present, r.follow_found
        s.cards_per_followed, s.card_changes = r.cards_per_followed, sum(r.card_changes.values())
        s.cards_followed, s.cards_with_another = r.cards_followed, r.cards_with_another
    return s


def compare(project: Project) -> list[TrialScore]:
    return [score_trial(open_trial(project, name), project) for name in trial_names(project)]


def table(scores: list[TrialScore]) -> list[str]:
    """The scores as lines of a table, one trial per row."""
    def pct(n, d):
        return f"{n / d:.0%}" if d else "-"

    head = ["trial", "cards", "to review", "mix answers", "fixes undone", "followed boxed",
            "cards/fish", "card changes", "cards w/ another"]
    rows = []
    for s in scores:
        rows.append([
            s.name, str(s.cards), str(s.to_review), f"{s.mixed} of {s.checked}", f"{s.rejoined} of {s.pulled_apart}",
            pct(s.follow_found, s.follow_present) if s.followed else "-",
            f"{s.cards_per_followed:.2f}" if s.followed else "-",
            str(s.card_changes) if s.followed else "-",
            f"{s.cards_with_another} of {s.cards_followed}" if s.followed else "-",
        ])
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    out = ["  ".join(c.ljust(w) for c, w in zip(head, widths))]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    out += [f"  {s.name}: {json.dumps(s.changes) if s.changes else 'the project as it is'}" for s in scores]
    return out
