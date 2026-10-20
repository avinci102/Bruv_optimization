"""Orchestration: ingest -> sample + detect -> track -> embed -> continuity -> classify.

Every stage persists to the project's SQLite store. Detection, the slow stage, resumes
from the last saved frame; the later stages are quick and re-run as a block. Each stage
run is recorded (settings, code, models; Project.start_run), and a processing run that
changed anything ends with a read-only snapshot of the results (Project.snapshot).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import imaging
from .classify import (
    Classifier,
    HistogramClassifier,
    crop_quality,
    histogram_embed,
    l2_normalize,
    make_classifier,
    softmax,
)
from .config import Settings
from .continuity import TrackletSummary, merge_tracklets
from .detect import Detector, make_detector
from .learn import Check, Library, blend, check, library_stamp, vote
from .provenance import deployment_id, describe_model, fingerprint
from .store import Project, box_px
from .template import read_template
from .track import Tracker
from .video import SAMPLER_VERSION, Chapter, Timeline, build_timeline, sample_frames, source_index

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]
POST_DETECT_STAGES = ("track", "embed", "merge", "classify")


def _quiet(stage: str, done: int, total: int) -> None:
    pass


def create_project(
    root: Path,
    videos: list[Path],
    template: Path | None,
    settings: Settings,
    extra_species: list[str] = (),
) -> Project:
    project = Project(root)
    project.save_settings(settings)
    project.set_meta("videos", [str(Path(v).resolve()) for v in videos])
    if template:
        info = read_template(Path(template))
        project.set_meta("template", str(Path(template).resolve()))
        project.set_meta("dropdowns", info.dropdowns)
        # child header -> parent header, e.g. {"species": "type"} (see TemplateInfo.dependent)
        project.set_meta("dependent", {child: parent for child, (parent, _) in info.dependent.items()})
    update_extra_species(project, list(extra_species))
    return project


EXTRA_SPECIES_FILE = "extra_species.txt"


def load_extra_species(template: Path | None) -> list[str]:
    """Names to guess and pick beyond the template's species list: a file named
    extra_species.txt next to the template if there is one, else the app's default list."""
    candidates = [Path(template).parent / EXTRA_SPECIES_FILE] if template else []
    candidates.append(Path(__file__).parent / EXTRA_SPECIES_FILE)
    for path in candidates:
        if path.is_file():
            lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
            return list(dict.fromkeys(line for line in lines if line and not line.startswith("#")))
    return []


def update_extra_species(project: Project, names: list[str]) -> None:
    """Store the extra names. When they change, only the guesses need recomputing."""
    if project.get_meta("extra_species") != names:
        project.set_meta("extra_species", names)
        project.mark_stage("classify", False)


def record_videos(project: Project) -> None:
    """Fingerprint the footage once (provenance.py): results then name the exact recording
    they came from, and one drop copied to two folders can't pass for two deployments."""
    if project.get_meta("deployment"):
        return
    paths = [Path(c["path"]) for c in project.chapters()]
    if not paths or not all(p.exists() for p in paths):
        log.warning("Video files not found, so this project's footage isn't fingerprinted yet")
        return
    prints = [{"path": str(p), "size": p.stat().st_size, "fingerprint": fingerprint(p)} for p in paths]
    project.set_meta("video_fingerprints", prints)
    project.set_meta("deployment", deployment_id([p["fingerprint"] for p in prints]))


def fill_source_indices(project: Project) -> int:
    """Give frames saved before source frame numbers were kept a number worked out from
    their time. That is exact when detection ran in one go from the start of the video;
    older versions handed back frames one step late after a resume, so the project is
    flagged and anything that relies on the numbers checks them against the saved frames."""
    rows = project.db.execute(
        """SELECT f.id, f.local_t, c.fps FROM frames f JOIN chapters c ON c.idx=f.chapter
           WHERE f.src_index IS NULL"""
    ).fetchall()
    if rows:
        project.db.executemany(
            "UPDATE frames SET src_index=? WHERE id=?",
            [(source_index(r["local_t"], r["fps"]), r["id"]) for r in rows],
        )
        project.set_meta("src_index_inferred", True)
    return len(rows)


def load_timeline(project: Project) -> Timeline:
    return Timeline(
        [
            Chapter(r["idx"], Path(r["path"]), r["name"], r["start_s"], r["duration_s"], r["fps"])
            for r in project.chapters()
        ]
    )


def extra_species(project: Project) -> list[str]:
    """The project's extra names that aren't already in the template's species list."""
    known = {s.strip().lower() for s in (project.get_meta("dropdowns") or {}).get("species", [])}
    return [s for s in project.get_meta("extra_species") or [] if s.strip().lower() not in known]


def species_labels(project: Project) -> list[str]:
    """Everything a card can be recorded as: the template's species list, then the extras."""
    return list((project.get_meta("dropdowns") or {}).get("species", [])) + extra_species(project)


def text_embedding_path(project: Project) -> Path:
    return project.root / "text_embeddings.npy"


def refresh_fish_guesses(project: Project, fish_id: int, settings: Settings | None = None) -> None:
    """Pick the best crop and recompute top-k guesses from cached embeddings (no model needed)."""
    settings = settings or project.load_settings()
    dets = project.fish_detections(fish_id)
    if not dets:
        project.set_fish_guesses(fish_id, [], None)
        return
    ranked = sorted(dets, key=lambda d: (d["quality"] or 0.0), reverse=True)
    embedded = [d for d in ranked if d["embedding"] is not None][: settings.crops_per_fish]
    guesses: list = []
    path = text_embedding_path(project)
    labels = project.get_meta("guess_labels") or []
    if embedded and path.exists() and labels:
        emb = l2_normalize(np.mean([np.frombuffer(d["embedding"], np.float32) for d in embedded], axis=0))
        text_emb = np.load(path)
        scale = float(project.get_meta("logit_scale", 100.0))
        probs = softmax(scale * (text_emb @ emb))
        weight = float(project.get_meta("library_weight") or 0.0)
        if weight:  # the library passed its check for this project (learn.py)
            library = _library()
            probs = blend(probs, vote(emb, library.emb, library.label, labels), weight)
        guesses = [[labels[i], round(float(probs[i]), 4)] for i in np.argsort(-probs)[: settings.top_k]]
    best = (embedded or ranked)[0]["id"]
    project.set_fish_guesses(fish_id, guesses, best)


_LIBRARY: dict = {}


def _library() -> Library:
    """The library, read once per change (the review server refreshes guesses often)."""
    stamp = library_stamp()
    if _LIBRARY.get("stamp") != stamp:
        _LIBRARY.update(stamp=stamp, library=Library.load())
    return _LIBRARY["library"]


class Pipeline:
    # Work saved together; a stop loses at most one chunk of crops or of species names.
    EMBED_CHUNK = 64
    LABEL_CHUNK = 20

    def __init__(
        self,
        project: Project,
        detector: Detector | None = None,
        classifier: Classifier | None = None,
        progress: Progress | None = None,
    ):
        self.project = project
        self.settings = project.load_settings()
        self.device = self.settings.resolved_device()
        self._detector = detector
        self._classifier = classifier
        self.progress = progress or _quiet

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = make_detector(self.settings, self.device)
        return self._detector

    @property
    def classifier(self) -> Classifier:
        if self._classifier is None:
            self._classifier = make_classifier(self.settings, self.device)
        return self._classifier

    def run(self, force: bool = False) -> None:
        runs_before = len(self.project.runs())
        self.ingest()
        record_videos(self.project)
        fill_source_indices(self.project)  # frames from older versions
        self._process(force)
        if len(self.project.runs()) > runs_before:
            self.project.snapshot("processed")

    def _process(self, force: bool) -> None:
        self.detect()
        self._detector = None  # free the detector's memory before BioCLIP 2 loads
        gc.collect()
        reviewed = self.project.reviewed_count()
        cards_done = all(self.project.stage_done(s) for s in ("track", "embed", "merge"))
        # Cards built without BioCLIP 2 (it couldn't load, e.g. short of memory) are rebuilt
        # with it on the next run, as long as nobody has reviewed them yet.
        fell_back = (
            not reviewed
            and self.settings.classifier == "bioclip"
            and self.project.get_meta("classifier") == HistogramClassifier.name
        )
        if self.project.get_meta("library_stamp") != library_stamp():
            self.project.mark_stage("classify", False)  # the library changed: refresh the guesses
        if force or fell_back or not cards_done:
            if reviewed and not force:
                raise RuntimeError("Reviews exist; pass force=True to rebuild cards and lose them.")
            for stage in POST_DETECT_STAGES:
                self.project.mark_stage(stage, False)
            self.track()
            self.embed()
            self.merge()
            self.classify()
        elif not self.project.stage_done("classify"):
            self.classify()  # e.g. new species names: the guesses change, cards and reviews stay

    # --- stages ------------------------------------------------------------------------

    def ingest(self) -> None:
        if self.project.stage_done("ingest"):
            return
        videos = [Path(v) for v in self.project.get_meta("videos", [])]
        if not videos:
            raise RuntimeError("No videos in this project")
        clips = (self.project.get_meta("dropdowns") or {}).get("clip_in", [])
        naming = self.settings.clip_names
        if naming == "auto":
            naming = "index" if "clip_1" in clips else "file"
        timeline = build_timeline(videos, naming)
        outside = [c.name for c in timeline.chapters if clips and c.name not in clips]
        if outside:
            log.warning("Clip name(s) %s aren't in the template's clip_in list", ", ".join(outside))
        self.project.save_chapters(timeline.chapters)
        self.project.mark_stage("ingest")
        self.progress("ingest", 1, 1)

    def detect(self) -> None:
        if self.project.stage_done("detect"):
            return
        s = self.settings
        timeline = load_timeline(self.project)
        # A stopped run resumes at the rate it started with, even if the default has changed.
        fps = self.project.frame_rate() if self.project.frame_count() >= 2 else s.resolved_fps()
        step = 1.0 / fps
        start = s.start_s
        end = timeline.duration_s if s.duration_s is None else min(timeline.duration_s, start + s.duration_s)
        total = max(1, int(round((end - start) * fps)))
        last = self.project.last_frame()
        seq = 0
        if last is not None:
            seq = last.seq + 1
            start = last.t + step
        self.progress("detect", seq, total)
        if start >= end - 1e-6:
            self.project.mark_stage("detect")
            return
        first_seq = seq
        run_id = self.project.start_run("detect", s, **self._detect_info(fps, seq))

        batch = []

        def flush():
            nonlocal seq
            results = self.detector.detect([f.image for f in batch])
            for frame, dets in zip(batch, results):
                small = imaging.resize_to_width(frame.image, s.frame_width)
                rel = f"frames/{seq:06d}.jpg"
                imaging.save_jpeg(small, self.project.root / rel, s.jpeg_quality)
                frame_id = self.project.add_frame(
                    seq, frame.t, frame.chapter, frame.local_t, rel, small.shape[1], small.shape[0],
                    frame.src_index,
                )
                qualities = []
                for d in dets:
                    crop = imaging.crop_box(frame.image, d.x1, d.y1, d.x2, d.y2, pad=0.0)
                    qualities.append(crop_quality(crop, d.score, max(d.w * d.h, 0.0)))
                    d.app = histogram_embed([crop])[0]
                self.project.add_detections(frame_id, dets, qualities)
                seq += 1
            self.project.commit()
            batch.clear()
            self.progress("detect", seq, total)

        for frame in sample_frames(timeline, fps, s.decode_width, start, end, s.hwaccel):
            batch.append(frame)
            if len(batch) >= s.batch_size:
                flush()
        if batch:
            flush()
        self.project.finish_run(run_id, frames=seq - first_seq, last_seq=seq - 1)
        self.project.mark_stage("detect")

    def _detect_info(self, fps: float, first_seq: int) -> dict:
        """What a detection run used: the detector (weights by fingerprint), the footage, and
        the frame sampler, whose ffmpeg build decides which source frames are decoded."""
        import imageio_ffmpeg  # noqa: PLC0415

        return {
            "detector": describe_model(self.detector),
            "device": self.device,
            "fps": fps,
            "first_seq": first_seq,
            "deployment": self.project.get_meta("deployment"),
            "sampler": SAMPLER_VERSION,
            "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
        }

    def track(self) -> None:
        s = self.settings
        run_id = self.project.start_run("track", s, fps=self.project.frame_rate())
        self.project.reset_tracking()
        tracker = Tracker(
            s.track_min_iou,
            s.track_max_dist,
            s.track_max_size_ratio,
            max(1, round(s.track_max_missed_s * self.project.frame_rate())),  # in sampled frames
            s.track_min_appearance,
            s.track_direction,
        )
        n = self.project.frame_count()
        for i, (frame, dets) in enumerate(self.project.frames_with_detections()):
            tracker.update(frame.seq, dets)
            if i % 200 == 0:
                self.progress("track", i, n)
        tracklets = tracker.finish()
        self.project.save_tracklets(tracklets)
        self.project.finish_run(run_id, tracklets=len(tracklets))
        self.project.mark_stage("track")
        self.progress("track", n, n)

    def embed(self) -> None:
        """Embed each tracklet's best crops. Embeddings are saved chunk by chunk and kept
        across runs (a crop's embedding doesn't depend on tracking), so a run that was
        stopped embeds only the crops it hadn't done yet."""
        s = self.settings
        key = f"{self.classifier.name} pad={s.crop_pad}"
        # A stand-in (BioCLIP 2 couldn't load) neither reuses nor overwrites the stored
        # embeddings, so the next run with BioCLIP 2 still finds them.
        keep = not (isinstance(self.classifier, HistogramClassifier) and s.classifier != "histogram")
        run_id = self.project.start_run(
            "embed", s,
            classifier=describe_model(self.classifier),
            stand_in=not keep,  # colour histograms instead of the species model: never pool
            crops_from=f"saved review frames ({s.frame_width} px wide, JPEG quality {s.jpeg_quality})",
        )
        stored = self.project.get_meta("embedding_key")
        if stored is None and self.project.get_meta("embedder"):  # projects from before the key
            stored = f"{self.project.get_meta('embedder')} pad={s.crop_pad}"
        if keep and stored != key:
            self.project.clear_embeddings()  # made by another model or crop setting
        if keep:
            self.project.set_meta("embedding_key", key)

        chosen: dict[int, list] = {}
        todo = []
        skipped = 0
        for tid in self.project.tracklet_ids():
            dets = self.project.tracklet_detections(tid)
            # Tiny fish join no other piece, and their blobs can't be identified, so the
            # species model (the slow step) skips them unless asked to.
            if not s.guess_tiny_fish and max(box_px(d) for d in dets) < s.tiny_fish_px:
                skipped += 1
                continue
            dets = sorted(dets, key=lambda d: d["quality"] or 0.0, reverse=True)
            chosen[tid] = dets[: s.crops_per_tracklet]
            todo += [d for d in chosen[tid] if not keep or d["embedding"] is None]
        total = sum(len(dets) for dets in chosen.values())
        done = total - len(todo)
        self.progress("embed", done, total)

        made: dict[int, np.ndarray] = {}
        for i in range(0, len(todo), self.EMBED_CHUNK):
            part = todo[i : i + self.EMBED_CHUNK]
            crops = [
                imaging.crop_box(imaging.load_rgb(str(self.project.root / d["path"])), d["x1"], d["y1"], d["x2"], d["y2"], s.crop_pad)
                for d in part
            ]
            for d, emb in zip(part, self.classifier.embed(crops)):
                made[d["id"]] = emb
                if keep:
                    self.project.set_detection_embedding(d["id"], emb)
            self.project.commit()
            self.progress("embed", done + i + len(part), total)
        for tid, dets in chosen.items():
            embs = [made[d["id"]] if d["id"] in made else np.frombuffer(d["embedding"], np.float32) for d in dets]
            self.project.set_tracklet_embedding(tid, l2_normalize(np.mean(embs, axis=0)))
        self.project.set_meta("embedder", self.classifier.name)
        self.project.commit()
        self.project.finish_run(
            run_id, tracklets=len(chosen), tiny_skipped=skipped, crops=total, embedded_now=len(todo)
        )
        self.project.mark_stage("embed")

    def merge(self) -> None:
        s = self.settings
        run_id = self.project.start_run("merge", s)
        summaries = []
        alone = []  # pieces the species model skipped (tiny fish): a card each
        start = {}
        for row in self.project.tracklets():
            start[row["id"]] = row["start_t"]
            dets = self.project.tracklet_detections(row["id"])
            sizes = [np.sqrt(max(d["x2"] - d["x1"], 0) * max(d["y2"] - d["y1"], 0)) for d in dets]
            emb = np.frombuffer(row["embedding"], np.float32) if row["embedding"] else None
            if emb is None:
                alone.append([row["id"]])
                continue
            apps = [np.frombuffer(d["app"], np.float32) for d in dets if d["app"] is not None]
            summaries.append(
                TrackletSummary(
                    row["id"],
                    row["start_t"],
                    row["end_t"],
                    emb,
                    float(np.median(sizes[:3])),
                    float(np.median(sizes[-3:])),
                    max(box_px(d) for d in dets),
                    l2_normalize(np.mean(apps, axis=0)) if apps else None,
                )
            )
        chains = merge_tracklets(
            summaries, s.gap_seconds, s.merge_min_similarity, s.merge_max_size_ratio, s.tiny_fish_px,
            s.merge_min_colour,
        )
        chains = sorted(chains + alone, key=lambda chain: (start[chain[0]], chain[0]))
        for chain in chains:
            self.project.create_fish(chain)
        self.project.commit()
        frozen = self.project.freeze_groups(run_id)  # the cards before anyone reviews them
        self.project.finish_run(
            run_id, pieces=len(start), cards=len(chains), not_embedded=len(alone), detections=frozen
        )
        self.project.mark_stage("merge")
        self.progress("merge", len(chains), len(chains))

    def _text_embeddings(self, labels: list[str]):
        """Label embeddings (80 prompts each), saved as they're made. A stopped run picks up
        where it was, and a longer list only encodes its new names."""
        path = self.project.root / "label_embeddings.npz"
        name = self.classifier.name
        cache: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as saved:
                if str(saved["model"]) == name:
                    cache = dict(zip(saved["labels"].tolist(), saved["emb"]))
        elif text_embedding_path(self.project).exists() and self.project.get_meta("classifier") == name:
            # A project from before this cache: reuse its last label embeddings.
            old = np.load(text_embedding_path(self.project))
            cache = dict(zip(self.project.get_meta("guess_labels") or [], old))
        missing = [label for label in labels if label not in cache]
        self.progress("labels", len(labels) - len(missing), len(labels))
        for i in range(0, len(missing), self.LABEL_CHUNK):
            part = missing[i : i + self.LABEL_CHUNK]
            emb = self.classifier.text_embeddings(part)
            if emb is None:
                return None
            cache.update(zip(part, emb))
            tmp = path.with_name("label_embeddings.tmp.npz")
            np.savez(tmp, model=np.array(name), labels=np.array(list(cache)), emb=np.stack(list(cache.values())))
            tmp.replace(path)
            self.progress("labels", len(labels) - len(missing) + i + len(part), len(labels))
        return np.stack([cache[label] for label in labels])

    def classify(self) -> None:
        labels = species_labels(self.project)
        run_id = self.project.start_run(
            "classify", self.settings, classifier=describe_model(self.classifier), labels=len(labels)
        )
        text_emb = self._text_embeddings(labels) if labels else None
        path = text_embedding_path(self.project)
        result = Check()
        if text_emb is not None:
            np.save(path, text_emb)
            self.project.set_meta("guess_labels", labels)
            self.project.set_meta("logit_scale", self.classifier.logit_scale)
            # Your confirmed examples join the guesses only if they beat the model alone.
            result = check(_library(), self.classifier.name, labels, text_emb, self.classifier.logit_scale)
        else:
            path.unlink(missing_ok=True)
            self.project.set_meta("guess_labels", [])
            if not labels:
                log.warning("No species dropdown found in the template; no guesses will be shown.")
        self.project.set_meta("library_check", asdict(result))
        self.project.set_meta("library_weight", result.weight)
        self.project.set_meta("library_stamp", library_stamp())
        self.project.set_meta("classifier", self.classifier.name)
        fish = self.project.all_fish()
        for i, f in enumerate(fish):
            refresh_fish_guesses(self.project, f["id"], self.settings)
            if i % 50 == 0:
                self.progress("classify", i, len(fish))
        self.project.commit()
        self.project.finish_run(
            run_id, cards=len(fish), guesses=text_emb is not None, library_check=asdict(result),
            library_stamp=library_stamp(),
        )
        self.project.mark_stage("classify")
        self.progress("classify", len(fish), len(fish))
