"""Families: the level above species, which a reviewer (and the species model) gets right
more often than the species itself.

A name's family is its group in the template: every species list hangs off the template's
type dropdown, e.g. "grouper" holds "grouper family" and the groupers. A group counts as a
family when most of its names carry the group's own name ("grouper", "shark") or the
template can record just the family ("grouper family", "unknown sea turtle"). A mixed list,
like a template's "species of interest" (a squid, a whale and a wrasse), is not a family:
adding up its names' scores would make it look like a strong guess, so each stands alone.
The extra names (extra_species.txt) are mostly families already; a line
"unicornfish > surgeonfish" puts one inside another. Any other name is its own family.
Nothing here is written into the code from the template: groups are read from the
project's own template.

Families are ranked by their best name's score, as the species are. Adding up a family's
scores looked better for groupers (on the 2026 answer key, grouper became the top family at
17 of 29 grouper sightings instead of 13), but on cards the model is unsure of, the long
lists win by their length: on the 44-minute video it made sharks or rays the top family for
64 cards, in a drop with none recorded. The sum is still given (Family.p). A family-level
answer uses the family's own entry in the template ("grouper family", "unknown shark").
Counting families is done per frame before MaxN (export.py), so a family's count never adds
up the separate MaxN of its species.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .store import Project

EXTRA_FILE = "extra_species.txt"


def extra_lines(template: Path | None) -> list[tuple[str, str | None]]:
    """(name, family or None) per line of the extra list next to the template, or the app's
    own list. "name > family" gives the family; lines starting with # are notes."""
    candidates = [Path(template).parent / EXTRA_FILE] if template else []
    candidates.append(Path(__file__).parent / EXTRA_FILE)
    for path in candidates:
        if path.is_file():
            out = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name, _, family = line.partition(">")
                out.append((name.strip(), family.strip() or None))
            return list({name: (name, fam) for name, fam in out}.values())
    return []


def template_groups(project: Project) -> dict[str, str]:
    """Species name -> its group in the project's template (kept in the project once read)."""
    cached = project.get_meta("species_groups")
    if cached is not None:
        return cached
    from .template import read_template  # noqa: PLC0415

    path = project.get_meta("template")
    groups: dict[str, str] = {}
    if path and Path(path).exists():
        groups = dict(read_template(Path(path)).dependent.get("species", (None, {}))[1])
    try:
        project.set_meta("species_groups", groups)
    except Exception:  # noqa: BLE001 — a read-only snapshot: just don't keep it
        pass
    return groups


def is_family(group: str, members: list[str]) -> bool:
    """A template group is a family, not a mixed list (see the module docstring)."""
    if family_entry(members, group) not in (None, group):
        return True
    words = [w for w in group.lower().replace("_", " ").replace("/", " ").split() if len(w) > 2]
    named = sum(any(w in m.lower() for w in words) for m in members)
    return bool(words) and 2 * named >= len(members)


def family_of(project: Project, labels: list[str]) -> dict[str, str]:
    """Every name -> its family (see the module docstring)."""
    groups = template_groups(project)
    members: dict[str, list[str]] = {}
    for name, group in groups.items():
        members.setdefault(group, []).append(name)
    families = {g for g, names in members.items() if is_family(g, names)}
    extras = {name: fam for name, fam in extra_lines(project.get_meta("template")) if fam}
    out = {}
    for label in labels:
        group = groups.get(label)
        out[label] = (group if group in families else extras.get(label) or label).strip()
    return out


@dataclass
class Family:
    key: str  # the template's group name, or the extra name
    name: str  # for people: "jack/trevally" rather than "jack_trevally"
    p: float  # the sum of the model's scores for the family's names
    members: list[tuple[str, float]] = field(default_factory=list)  # (name, score), best first
    entry: str | None = None  # the name that records just the family, if the template has one

    @property
    def best(self) -> float:
        """The family's best name's score: what families are ranked by."""
        return self.members[0][1] if self.members else 0.0


def family_entry(members: list[str], key: str) -> str | None:
    """The name that answers "this family, species unknown": "<x> family" or "unknown <x>"."""
    for name in members:
        low = name.strip().lower()
        if low.endswith(" family") or low.startswith("unknown "):
            return name
    return key if key in members else None


def display_name(key: str, entry: str | None) -> str:
    if entry and entry.strip().lower().endswith(" family"):
        return entry.strip()[: -len(" family")]
    return key.strip().replace("_", " ")


def rank_families(labels: list[str], probs: np.ndarray, family: dict[str, str]) -> list[Family]:
    """Families by their best name's score (see the module docstring), each with its names
    best first."""
    by_family: dict[str, list[tuple[str, float]]] = {}
    for label, p in zip(labels, probs):
        by_family.setdefault(family.get(label, label), []).append((label, float(p)))
    out = []
    for key, members in by_family.items():
        members.sort(key=lambda m: -m[1])
        entry = family_entry([m[0] for m in members], key)
        out.append(Family(key, display_name(key, entry), sum(p for _, p in members), members, entry))
    return sorted(out, key=lambda f: -f.best)
