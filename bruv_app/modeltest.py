"""Species models side by side on your answered cards: today's BioCLIP 2, another model
(BioCLIP 2.5), and the two together.

Every model scores the same crops: the ones today's guesses use (each card's best crops by
crop quality, as the embed stage saved them), against the same names with the same 80
prompts each. A card's guess is its crops' mean embedding scored against the names; "both"
averages the two models' scores. Scored on your answers: the exact name first, the name in
the top three, and the right family first (families.py).

"+ your answers" replays your answers in the order you gave them: a card that looks like one
you'd already answered (embeddings 0.85 alike or more) gets that answer, otherwise the
model's. That is what the review page's look-alike suggestions do.

With the team's answer key, its clip_1 sightings are matched by time to the cards on screen
(docs/SPEC.md §13): a sighting counts as found when a card on screen then has its family as
the top guess. Rough, since other fish share the screen, but the same for every model.

The other model's crop and name embeddings are saved in the output folder and reused, so a
second run is quick. The project is only read.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import imaging
from .classify import l2_normalize, softmax
from .families import family_of
from .store import Project

LOOKALIKE = 0.85  # as the review page: answers are suggested from cards at least this alike


@dataclass
class Card:
    id: int
    status: str
    answer: str | None
    answered_at: str | None
    start_t: float
    end_t: float
    crops: list[int]  # detection ids, best first


@dataclass
class ModelScores:
    name: str
    labels: list[str]
    text: np.ndarray  # one row per label
    logit_scale: float
    crops: dict[int, np.ndarray] = field(default_factory=dict)  # detection id -> embedding
    seconds_per_crop: float | None = None  # how long this PC took per new crop

    def card_embedding(self, card: Card) -> np.ndarray | None:
        embs = [self.crops[d] for d in card.crops if d in self.crops]
        return l2_normalize(np.mean(embs, axis=0)) if embs else None

    def probs(self, card: Card) -> np.ndarray | None:
        emb = self.card_embedding(card)
        return None if emb is None else softmax(self.logit_scale * (self.text @ emb))


def cards_to_test(project: Project, key_times: list[tuple[float, float]] = ()) -> list[Card]:
    """Answered cards, plus (for the key) unanswered ones on screen during its sightings;
    each with the crops today's guesses use."""
    settings = project.load_settings()
    last = {r["fish_id"]: r["at"] for r in project.db.execute("SELECT fish_id, MAX(at) AS at FROM reviews GROUP BY fish_id")}
    sizes = project.fish_sizes()
    out = []
    for f in project.all_fish():
        answered = f["status"] in ("species", "not_fish", "unknown")
        on_key = any(f["start_t"] <= t1 + 1 and f["end_t"] >= t0 - 1 for t0, t1 in key_times)
        if not answered and not (on_key and sizes.get(f["id"], 0) >= settings.tiny_fish_px):
            continue
        dets = sorted(project.fish_detections(f["id"]), key=lambda d: d["quality"] or 0.0, reverse=True)
        crops = [d["id"] for d in dets if d["embedding"] is not None][: settings.crops_per_fish]
        if crops:
            out.append(Card(f["id"], f["status"], f["species"], last.get(f["id"]) if answered else None,
                            f["start_t"], f["end_t"], crops))
    return out


def todays_model(project: Project, cards: list[Card]) -> ModelScores:
    """BioCLIP 2 as the project has it: its saved crop and name embeddings, no model run."""
    labels = project.get_meta("guess_labels") or []
    text = np.load(project.root / "text_embeddings.npy")
    wanted = {d for c in cards for d in c.crops}
    crops = {
        r["id"]: np.frombuffer(r["embedding"], np.float32)
        for r in project.db.execute("SELECT id, embedding FROM detections WHERE embedding IS NOT NULL")
        if r["id"] in wanted
    }
    return ModelScores(project.get_meta("classifier") or "bioclip-2", labels, text,
                       float(project.get_meta("logit_scale", 100.0)), crops)


def other_model(project: Project, classifier, cards: list[Card], out: Path, progress=None) -> ModelScores:
    """Another model's embeddings of the same crops and names, made once and saved."""
    settings = project.load_settings()
    labels = project.get_meta("guess_labels") or []
    out.mkdir(parents=True, exist_ok=True)
    names_path, crops_path = out / f"{classifier.name}-names.npz", out / f"{classifier.name}-crops.npz"
    text = None
    if names_path.exists():
        with np.load(names_path) as saved:
            if saved["labels"].tolist() == labels:
                text = saved["emb"]
    if text is None:
        parts = []
        for i in range(0, len(labels), 10):
            parts.append(classifier.text_embeddings(labels[i:i + 10]))
            if progress:
                progress("names", min(i + 10, len(labels)), len(labels))
        text = np.concatenate(parts)
        np.savez(names_path, labels=np.array(labels), emb=text)
    crops: dict[int, np.ndarray] = {}
    if crops_path.exists():
        with np.load(crops_path) as saved:
            crops = dict(zip(saved["ids"].tolist(), saved["emb"]))
    todo = sorted({d for c in cards for d in c.crops} - set(crops))
    rows = {}
    if todo:
        marks = ",".join("?" * len(todo))
        rows = {r["id"]: r for r in project.db.execute(
            f"SELECT d.id, d.x1, d.y1, d.x2, d.y2, f.path FROM detections d JOIN frames f ON f.id=d.frame_id WHERE d.id IN ({marks})",  # noqa: S608
            todo,
        )}
    started = time.monotonic()
    for i in range(0, len(todo), 16):
        part = todo[i:i + 16]
        images = [
            imaging.crop_box(imaging.load_rgb(str(project.root / rows[d]["path"])), rows[d]["x1"], rows[d]["y1"],
                             rows[d]["x2"], rows[d]["y2"], settings.crop_pad)
            for d in part
        ]
        crops.update(zip(part, classifier.embed(images)))
        np.savez(crops_path, ids=np.array(list(crops)), emb=np.stack(list(crops.values())))
        if progress:
            progress("crops", i + len(part), len(todo))
    seconds = (time.monotonic() - started) / len(todo) if todo else None
    return ModelScores(classifier.name, labels, text, float(classifier.logit_scale), crops, seconds)


def _top(labels: list[str], probs: np.ndarray, k: int = 3) -> list[str]:
    return [labels[i] for i in np.argsort(-probs)[:k]]


def score(cards: list[Card], models: list[ModelScores], family: dict[str, str]) -> dict:
    """Each model, and the models together, against your answers (cards answered with a name)."""
    fam = lambda name: family.get(name, name)  # noqa: E731
    named = sorted([c for c in cards if c.status == "species"], key=lambda c: c.answered_at or "")
    runs: dict[str, dict] = {}
    combos = [(m.name, [m]) for m in models]
    if len(models) > 1:
        combos.append((" + ".join(m.name for m in models), models))
    for name, group in combos:
        tally = Counter()
        by_family: dict[str, Counter] = defaultdict(Counter)
        picks = {}
        for c in named:
            probs = [m.probs(c) for m in group]
            if any(p is None for p in probs):
                continue
            p = np.mean(probs, axis=0)
            top = _top(group[0].labels, p)
            picks[c.id] = top[0]
            hit = {"exact": top[0] == c.answer, "top 3": c.answer in top, "family": fam(top[0]) == fam(c.answer)}
            tally["cards"] += 1
            by_family[fam(c.answer)]["cards"] += 1
            for k, v in hit.items():
                tally[k] += v
                by_family[fam(c.answer)][k] += v
        runs[name] = {"tally": dict(tally), "by_family": {k: dict(v) for k, v in by_family.items()}, "picks": picks}
    # "+ your answers": the closest earlier-answered card in each model's own embedding space.
    for m in models:
        tally = Counter()
        seen: list[tuple[np.ndarray, str]] = []
        for c in named:
            emb, p = m.card_embedding(c), m.probs(c)
            if emb is None:
                continue
            guess = _top(m.labels, p)[0]
            if seen:
                sims = [float(emb @ e) for e, _ in seen]
                j = int(np.argmax(sims))
                if sims[j] >= LOOKALIKE:
                    guess = seen[j][1]
                    tally["from your answers"] += 1
            tally["cards"] += 1
            tally["exact"] += guess == c.answer
            tally["family"] += fam(guess) == fam(c.answer)
            seen.append((emb, c.answer))
        runs[f"{m.name} + your answers"] = {"tally": dict(tally)}
    # What each model calls the cards answered "not a fish".
    others = {m.name: Counter(_top(m.labels, m.probs(c))[0] for c in cards if c.status == "not_fish" and m.probs(c) is not None)
              for m in models}
    return {"runs": runs, "not_fish_called": {k: v.most_common(5) for k, v in others.items()}}


def key_sightings(key_path: Path) -> list[tuple[str, float, float]]:
    """(species, start s, end s) of the key's clip_1 sightings."""
    import openpyxl  # noqa: PLC0415

    from .template import read_template  # noqa: PLC0415

    info = read_template(key_path)
    wb = openpyxl.load_workbook(key_path, read_only=True, data_only=True)
    rows = list(wb[info.record_sheet].iter_rows(values_only=True))
    col = {str(h).strip().lower(): i for i, h in enumerate(rows[0]) if h}

    def secs(v):
        return v.hour * 3600 + v.minute * 60 + v.second if isinstance(v, dt.time) else None

    out = []
    for r in rows[1:]:
        sp, t0 = r[col["species"]], secs(r[col["time_in"]])
        if not sp or r[col["clip_in"]] != "clip_1" or t0 is None:
            continue
        out.append((str(sp), float(t0), float(secs(r[col["time_out"]]) or t0)))
    return out


def score_key(cards: list[Card], models: list[ModelScores], family: dict[str, str], sightings) -> dict:
    fam = lambda name: family.get(name, name)  # noqa: E731
    out = {}
    combos = [(m.name, [m]) for m in models]
    if len(models) > 1:
        combos.append((" + ".join(m.name for m in models), models))
    for name, group in combos:
        tops = []
        for c in cards:
            probs = [m.probs(c) for m in group]
            if all(p is not None for p in probs):
                tops.append((c, _top(group[0].labels, np.mean(probs, axis=0), 1)[0]))
        found = defaultdict(Counter)
        for sp, t0, t1 in sightings:
            here = [top for c, top in tops if c.start_t <= t1 + 1 and c.end_t >= t0 - 1]
            found[fam(sp)]["sightings"] += 1
            found[fam(sp)]["found"] += any(fam(top) == fam(sp) for top in here)
        out[name] = {k: dict(v) for k, v in found.items()}
    return out


def report_lines(result: dict) -> list[str]:
    lines = ["Your answered cards (named): right first guess, exact name / right family; name in the top 3"]
    for name, run in result["runs"].items():
        t = run["tally"]
        if not t.get("cards"):
            continue
        extra = f"; {t['from your answers']} named from your earlier answers" if "from your answers" in t else ""
        top3 = f"; top 3 {t['top 3']}" if "top 3" in t else ""
        lines.append(f"  {name:<34} {t['exact']:>3} / {t['family']:>3} of {t['cards']}{top3}{extra}")
    for name, run in result["runs"].items():
        if "by_family" in run:
            parts = ", ".join(f"{k} {v.get('family', 0)}/{v['cards']}" for k, v in sorted(run["by_family"].items(), key=lambda kv: -kv[1]["cards"]))
            lines.append(f"  by family, {name}: {parts}")
    for name, called in result["not_fish_called"].items():
        lines.append(f"  'not a fish' cards, {name} calls them: " + ", ".join(f"{n} {k}" for n, k in called))
    if result.get("key"):
        lines.append("Answer key, clip_1 sightings with a card on screen whose top guess is the right family:")
        for name, fams in result["key"].items():
            parts = ", ".join(f"{k} {v['found']}/{v['sightings']}" for k, v in sorted(fams.items(), key=lambda kv: -kv[1]["sightings"]))
            lines.append(f"  {name}: {parts}")
    if result.get("seconds_per_crop"):
        lines.append(f"Speed on this PC: {result['seconds_per_crop']:.2f} s per crop for the other model")
    return lines


def run(project: Project, classifier, out: Path, key_path: Path | None = None, progress=None) -> tuple[dict, Path]:
    sightings = key_sightings(key_path) if key_path else []
    cards = cards_to_test(project, [(t0, t1) for _, t0, t1 in sightings])
    today = todays_model(project, cards)
    models = [today]
    if classifier is not None:
        models.append(other_model(project, classifier, cards, out, progress))
    family = family_of(project, today.labels)
    result = score(cards, models, family)
    if sightings:
        result["key"] = score_key(cards, models, family, sightings)
    result["seconds_per_crop"] = models[-1].seconds_per_crop
    result["cards"] = {"answered": sum(c.status != "pending" for c in cards), "tested": len(cards)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model-test.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    return result, path

