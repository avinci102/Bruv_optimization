"""Orchestration: ingest -> sample + detect -> track -> embed -> continuity -> classify.

Every stage persists to the project's SQLite store. Detection, the slow stage, resumes
from the last saved frame; the later stages are quick and re-run as a block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from . import imaging
from .classify import (
    Classifier,
    crop_quality,
    histogram_embed,
    l2_normalize,
    make_classifier,
    zero_shot_topk,
)
from .config import Settings
from .continuity import TrackletSummary, merge_tracklets
from .detect import Detector, make_detector
from .store import Project
from .template import read_template
from .track import Tracker
from .video import Chapter, Timeline, build_timeline, sample_frames

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]
POST_DETECT_STAGES = ("track", "embed", "merge", "classify")


def _quiet(stage: str, done: int, total: int) -> None:
    pass


def create_project(
    root: Path, videos: list[Path], template: Path | None, settings: Settings
) -> Project:
    project = Project(root)
    project.save_settings(settings)
    project.set_meta("videos", [str(Path(v).resolve()) for v in videos])
    if template:
        info = read_template(Path(template))
        project.set_meta("template", str(Path(template).resolve()))
        project.set_meta("dropdowns", info.dropdowns)
    return project


def load_timeline(project: Project) -> Timeline:
    return Timeline(
        [
            Chapter(r["idx"], Path(r["path"]), r["name"], r["start_s"], r["duration_s"], r["fps"])
            for r in project.chapters()
        ]
    )


def species_labels(project: Project) -> list[str]:
    return list((project.get_meta("dropdowns") or {}).get("species", []))


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
        guesses = [[label, round(p, 4)] for label, p in zero_shot_topk(emb, text_emb, labels, scale, settings.top_k)]
    best = (embedded or ranked)[0]["id"]
    project.set_fish_guesses(fish_id, guesses, best)


class Pipeline:
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
        self.ingest()
        self.detect()
        if self.project.reviewed_count() and not force:
            if not all(self.project.stage_done(s) for s in POST_DETECT_STAGES):
                raise RuntimeError("Reviews exist; pass force=True to rebuild cards and lose them.")
            return
        if force or not all(self.project.stage_done(s) for s in POST_DETECT_STAGES):
            for stage in POST_DETECT_STAGES:
                self.project.mark_stage(stage, False)
            self.track()
            self.embed()
            self.merge()
            self.classify()

    # --- stages ------------------------------------------------------------------------

    def ingest(self) -> None:
        if self.project.stage_done("ingest"):
            return
        videos = [Path(v) for v in self.project.get_meta("videos", [])]
        if not videos:
            raise RuntimeError("No videos in this project")
        timeline = build_timeline(videos, self.settings.clip_names)
        self.project.save_chapters(timeline.chapters)
        self.project.mark_stage("ingest")
        self.progress("ingest", 1, 1)

    def detect(self) -> None:
        if self.project.stage_done("detect"):
            return
        s = self.settings
        timeline = load_timeline(self.project)
        fps = s.resolved_fps()
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

        batch = []

        def flush():
            nonlocal seq
            results = self.detector.detect([f.image for f in batch])
            for frame, dets in zip(batch, results):
                small = imaging.resize_to_width(frame.image, s.frame_width)
                rel = f"frames/{seq:06d}.jpg"
                imaging.save_jpeg(small, self.project.root / rel, s.jpeg_quality)
                frame_id = self.project.add_frame(
                    seq, frame.t, frame.chapter, frame.local_t, rel, small.shape[1], small.shape[0]
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

        if start < end - 1e-6:
            for frame in sample_frames(timeline, fps, s.decode_width, start, end, s.hwaccel):
                batch.append(frame)
                if len(batch) >= s.batch_size:
                    flush()
            if batch:
                flush()
        self.project.mark_stage("detect")

    def track(self) -> None:
        s = self.settings
        self.project.reset_tracking()
        tracker = Tracker(
            s.track_min_iou,
            s.track_max_dist,
            s.track_max_size_ratio,
            s.track_max_missed,
            s.track_min_appearance,
        )
        n = self.project.frame_count()
        for i, (frame, dets) in enumerate(self.project.frames_with_detections()):
            tracker.update(frame.seq, dets)
            if i % 200 == 0:
                self.progress("track", i, n)
        tracklets = tracker.finish()
        self.project.save_tracklets(tracklets)
        self.project.mark_stage("track")
        self.progress("track", n, n)

    def embed(self) -> None:
        s = self.settings
        ids = self.project.tracklet_ids()
        jobs: list[tuple[int, int]] = []  # (tracklet id, detection id)
        crops: list[np.ndarray] = []
        for tid in ids:
            dets = sorted(self.project.tracklet_detections(tid), key=lambda d: d["quality"] or 0.0, reverse=True)
            for d in dets[: s.crops_per_tracklet]:
                image = imaging.load_rgb(str(self.project.root / d["path"]))
                crops.append(imaging.crop_box(image, d["x1"], d["y1"], d["x2"], d["y2"], s.crop_pad))
                jobs.append((tid, d["id"]))

        per_tracklet: dict[int, list[np.ndarray]] = {}
        chunk = 64
        for i in range(0, len(crops), chunk):
            embs = self.classifier.embed(crops[i : i + chunk])
            for (tid, det_id), emb in zip(jobs[i : i + chunk], embs):
                self.project.set_detection_embedding(det_id, emb)
                per_tracklet.setdefault(tid, []).append(emb)
            self.progress("embed", min(i + chunk, len(crops)), len(crops))
        for tid, embs in per_tracklet.items():
            self.project.set_tracklet_embedding(tid, l2_normalize(np.mean(embs, axis=0)))
        self.project.set_meta("embedder", self.classifier.name)
        self.project.commit()
        self.project.mark_stage("embed")

    def merge(self) -> None:
        s = self.settings
        summaries = []
        for row in self.project.tracklets():
            dets = self.project.tracklet_detections(row["id"])
            sizes = [np.sqrt(max(d["x2"] - d["x1"], 0) * max(d["y2"] - d["y1"], 0)) for d in dets]
            emb = np.frombuffer(row["embedding"], np.float32) if row["embedding"] else None
            if emb is None:
                continue
            summaries.append(
                TrackletSummary(
                    row["id"],
                    row["start_t"],
                    row["end_t"],
                    emb,
                    float(np.median(sizes[:3])),
                    float(np.median(sizes[-3:])),
                )
            )
        chains = merge_tracklets(summaries, s.gap_seconds, s.merge_min_similarity, s.merge_max_size_ratio)
        for chain in chains:
            self.project.create_fish(chain)
        self.project.commit()
        self.project.mark_stage("merge")
        self.progress("merge", len(chains), len(chains))

    def _text_embeddings(self, labels: list[str]):
        """Label embeddings in chunks, so a long species list shows progress (80 prompts each)."""
        chunks = []
        for i in range(0, len(labels), 20):
            emb = self.classifier.text_embeddings(labels[i : i + 20])
            if emb is None:
                return None
            chunks.append(emb)
            self.progress("labels", min(i + 20, len(labels)), len(labels))
        return np.concatenate(chunks)

    def classify(self) -> None:
        labels = species_labels(self.project)
        text_emb = self._text_embeddings(labels) if labels else None
        path = text_embedding_path(self.project)
        if text_emb is not None:
            np.save(path, text_emb)
            self.project.set_meta("guess_labels", labels)
            self.project.set_meta("logit_scale", self.classifier.logit_scale)
        else:
            path.unlink(missing_ok=True)
            self.project.set_meta("guess_labels", [])
            if not labels:
                log.warning("No species dropdown found in the template; no guesses will be shown.")
        self.project.set_meta("classifier", self.classifier.name)
        fish = self.project.all_fish()
        for i, f in enumerate(fish):
            refresh_fish_guesses(self.project, f["id"], self.settings)
            if i % 50 == 0:
                self.progress("classify", i, len(fish))
        self.project.commit()
        self.project.mark_stage("classify")
        self.progress("classify", len(fish), len(fish))
