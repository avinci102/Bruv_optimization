"""The training loop: confirmed fish teach the species guesses.

A library on this PC holds examples: the confirmed cards of reviewed projects you choose to
add, and fish photos sorted into one folder per species. For each, it keeps the BioCLIP
embedding and the species. BioCLIP itself is never retrained.

Guesses blend BioCLIP's zero-shot scores with a vote of the nearest library examples, but
only once that blend has beaten zero-shot on footage it didn't learn from. Examples are
grouped by the recording they show (the deployment id from the videos' fingerprints, or a
photo set's own fingerprint), so the same footage copied to two folders is one group, never
two independent ones. In the check each group is guessed from the other groups alone, and
the vote's weight is chosen for each held-out group from the remaining groups only (nested),
so the accuracy that decides is never the one the weight was tuned on (review V2). It takes
at least MIN_GROUPS recordings. Until it passes, the library is ignored, so it can't make
guesses worse.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classify import l2_normalize, softmax
from .store import Project

K = 10  # nearest library examples that vote
VOTE_SCALE = 20.0  # how much more a closer example counts
WEIGHTS = (0.25, 0.5, 0.75)  # share of the vote in a blended guess, picked by the check
DEFAULT_WEIGHT = 0.5  # for a held-out group when too few others are left to choose one
MIN_EXAMPLES = 20  # examples guessed in the check before the blend may switch on
MIN_GAIN = 0.05  # the blend must beat zero-shot top-1 accuracy by this much
MIN_GROUPS = 3  # recordings: one held out, and at least two left to tune the weight on
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def library_path() -> Path:
    if os.environ.get("BRUV_LIBRARY"):
        return Path(os.environ["BRUV_LIBRARY"])
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "bruv_app" / "library.npz"


def library_stamp() -> str:
    """Changes whenever the library does, so projects know to refresh their guesses."""
    path = library_path()
    return f"{path.stat().st_size}-{path.stat().st_mtime_ns}" if path.exists() else ""


def _key(label: str) -> str:
    return label.strip().lower()


@dataclass
class Library:
    model: str  # the embedding model; examples from another model can't be compared
    emb: np.ndarray  # (N, D) L2-normalised, one row per card or photo
    label: np.ndarray  # (N,) species
    source: np.ndarray  # (N,) the project or photo folder it came from
    group: np.ndarray | None = None  # (N,) the recording it shows: copies of footage share it

    def __post_init__(self):
        if self.group is None:  # a library from before groups: each folder is its own
            self.group = self.source.copy()

    @classmethod
    def load(cls) -> Library:
        path = library_path()
        if not path.exists():
            return cls("", np.zeros((0, 0), np.float32), np.array([], str), np.array([], str))
        with np.load(path) as z:
            return cls(str(z["model"]), z["emb"], z["label"], z["source"], z["group"] if "group" in z.files else None)

    def save(self) -> None:
        path = library_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("library.tmp.npz")
        np.savez(tmp, model=np.array(self.model), emb=self.emb, label=self.label, source=self.source, group=self.group)
        tmp.replace(path)

    def replace_source(self, model: str, source: str, group: str, emb: np.ndarray, labels: list[str]) -> None:
        """Swap in one source's examples. Adding it again, or the same footage from another
        folder (the same group), replaces the earlier ones."""
        if self.model and len(self.label) and self.model != model:
            raise ValueError(f"The library holds {self.model} examples, not {model}")
        keep = (self.source != source) & (self.group != group)
        old = self.emb[keep] if len(self.label) else np.zeros((0, emb.shape[1]), np.float32)
        self.model = model
        self.emb = np.concatenate([old, emb]).astype(np.float32)
        self.label = np.concatenate([self.label[keep], np.array(labels, str)])
        self.source = np.concatenate([self.source[keep], np.array([source] * len(labels), str)])
        self.group = np.concatenate([self.group[keep], np.array([group] * len(labels), str)])

    def summary(self) -> dict[str, dict[str, int]]:
        """source -> {species: examples}"""
        out: dict[str, dict[str, int]] = {}
        for label, source in zip(self.label.tolist(), self.source.tolist()):
            counts = out.setdefault(source, {})
            counts[label] = counts.get(label, 0) + 1
        return out


def add_project(project: Project) -> int:
    """Add a reviewed project's confirmed cards: one example per card (the mean embedding of
    its crops the answer was given for). Left out: cards recorded as the template's catch-all
    ("other …"; the real name is only in the comment), and cards with warnings that were never
    confirmed as one fish (diagnostics.py), which may mix species."""
    from .diagnostics import all_flags  # noqa: PLC0415
    from .pipeline import record_videos  # noqa: PLC0415

    record_videos(project)
    group = project.get_meta("deployment")
    if not group:
        raise ValueError(
            f"{project.root}: its videos aren't where the project says, so the library can't tell which "
            "recording these fish come from (and copies of one recording would pass as several)"
        )
    model = project.get_meta("embedder")
    warned = all_flags(project, project.load_settings().merge_min_similarity)
    scopes = project.label_scopes()
    emb, labels = [], []
    for f in project.all_fish():
        if f["status"] != "species" or _key(f["species"]).startswith("other"):
            continue
        if f["id"] in warned and not project.answer_details(f["id"]).get("confirmed"):
            continue
        scope = scopes.get(f["id"])
        crops = [np.frombuffer(d["embedding"], np.float32) for d in project.fish_detections(f["id"])
                 if d["embedding"] is not None and (scope is None or d["id"] in scope)]
        if crops:
            emb.append(l2_normalize(np.mean(crops, axis=0)))
            labels.append(f["species"])
    if not model or not labels:
        return 0
    library = Library.load()
    library.replace_source(model, str(project.root.resolve()), group, np.array(emb), labels)
    library.save()
    return len(labels)


def photo_set_id(photos: list[Path]) -> str:
    """One id for a set of photos, from their contents: the same photos copied elsewhere
    get the same id."""
    from .provenance import fingerprint  # noqa: PLC0415

    h = hashlib.sha256()
    for p in sorted(photos, key=lambda p: (p.parent.name, p.name)):
        h.update(f"{p.parent.name}/{p.name}:{fingerprint(p)}|".encode())
    return "photos-" + h.hexdigest()[:16]


def add_photos(folder: Path, classifier) -> int:
    """Add photos sorted into one subfolder per species, named as in the species list."""
    from . import imaging  # noqa: PLC0415

    emb, labels, files = [], [], []
    for sub in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        photos = sorted(p for p in sub.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
        files += photos
        for i in range(0, len(photos), 16):
            batch = photos[i : i + 16]
            emb.extend(classifier.embed([imaging.load_rgb(str(p)) for p in batch]))
            labels.extend([sub.name] * len(batch))
    if not labels:
        return 0
    library = Library.load()
    library.replace_source(classifier.name, str(Path(folder).resolve()), photo_set_id(files), l2_normalize(np.array(emb)), labels)
    library.save()
    return len(labels)


def remove_source(source: str) -> int:
    library = Library.load()
    keep = library.source != source
    removed = int((~keep).sum())
    if removed:
        library.emb, library.label = library.emb[keep], library.label[keep]
        library.source, library.group = library.source[keep], library.group[keep]
        library.save()
    return removed


def vote(emb: np.ndarray, examples: np.ndarray, examples_label: np.ndarray, labels: list[str]) -> np.ndarray:
    """The K nearest examples' vote, as a probability over `labels` (zeros if none count)."""
    index = {_key(label): i for i, label in enumerate(labels)}
    usable = np.array([_key(label) in index for label in examples_label.tolist()], bool)
    p = np.zeros(len(labels))
    if not usable.any():
        return p
    sims = examples[usable] @ emb
    names = examples_label[usable]
    top = np.argsort(-sims)[:K]
    for j, weight in zip(top, np.exp(VOTE_SCALE * (sims[top] - sims[top].max()))):
        p[index[_key(names[j])]] += weight
    return p / p.sum()


def blend(zero_shot: np.ndarray, votes: np.ndarray, weight: float) -> np.ndarray:
    return (1 - weight) * zero_shot + weight * votes if votes.any() else zero_shot


@dataclass
class Check:
    examples: int = 0  # library examples guessed from other recordings
    zero_shot: float = 0.0  # top-1 accuracy of BioCLIP alone
    blended: float = 0.0  # top-1 accuracy with the library's vote, its weight tuned without them
    weight: float = 0.0  # the vote's share in guesses; 0 = library not used
    groups: int = 0  # recordings (or photo sets) the examples come from


def check(library: Library, model: str, labels: list[str], text_emb: np.ndarray, scale: float) -> Check:
    """Guess every library example of these labels from the other recordings (never its
    own, wherever a copy of it sits): zero-shot alone, against blended with the library's
    vote at a weight chosen on the remaining recordings only (nested leave-one-recording-out)."""
    if library.model != model or not len(library.label):
        return Check()
    index = {_key(label): i for i, label in enumerate(labels)}
    rows = [i for i, label in enumerate(library.label.tolist()) if _key(label) in index]
    groups = sorted({library.group[i] for i in rows})
    truth = {i: index[_key(library.label[i])] for i in rows}
    zero = {i: softmax(scale * (text_emb @ library.emb[i])) for i in rows}

    def right(test: list[int], pool: np.ndarray, w: float) -> int:
        if not pool.any():
            return sum(int(np.argmax(zero[i]) == truth[i]) for i in test)
        return sum(
            int(np.argmax(blend(zero[i], vote(library.emb[i], library.emb[pool], library.label[pool], labels), w)) == truth[i])
            for i in test
        )

    by_group = {g: [i for i in rows if library.group[i] == g] for g in groups}
    n = right_zero = right_blend = 0
    for g in groups:
        pool = library.group != g
        if not pool.any():
            continue
        # The weight for this recording, chosen from the others alone.
        inner = [h for h in groups if h != g]
        if len(inner) >= 2:
            score = {w: sum(right(by_group[h], pool & (library.group != h), w) for h in inner) for w in WEIGHTS}
            w = max(WEIGHTS, key=lambda x: (score[x], -abs(x - DEFAULT_WEIGHT)))
        else:
            w = DEFAULT_WEIGHT
        n += len(by_group[g])
        right_zero += sum(int(np.argmax(zero[i]) == truth[i]) for i in by_group[g])
        right_blend += right(by_group[g], pool, w)
    if not n:
        return Check(groups=len(groups))
    result = Check(n, right_zero / n, right_blend / n, 0.0, len(groups))
    if n >= MIN_EXAMPLES and len(groups) >= MIN_GROUPS and result.blended >= result.zero_shot + MIN_GAIN:
        # In use, the weight is chosen on every recording held out in turn.
        score = {w: sum(right(by_group[g], library.group != g, w) for g in groups) for w in WEIGHTS}
        result.weight = max(WEIGHTS, key=lambda x: (score[x], -abs(x - DEFAULT_WEIGHT)))
    return result
