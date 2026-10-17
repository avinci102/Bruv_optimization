from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from bruv_app import demo
from bruv_app.config import Settings
from bruv_app.pipeline import Pipeline, create_project
from bruv_app.track import iou
from bruv_app.detect import Detection


@pytest.fixture(scope="session")
def synthetic_videos(tmp_path_factory) -> list[Path]:
    return demo.make_synthetic_videos(tmp_path_factory.mktemp("videos"))


@pytest.fixture
def mock_template(tmp_path) -> Path:
    return demo.make_mock_template(tmp_path / "template" / "viewing template v.2024.xlsx")


def demo_settings(**overrides) -> Settings:
    settings = Settings(detector="color", classifier="histogram")
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


@pytest.fixture(scope="session")
def processed_project(tmp_path_factory, synthetic_videos):
    """A fully processed synthetic project; tests copy it before changing anything."""
    root = tmp_path_factory.mktemp("base")
    template = demo.make_mock_template(root / "template.xlsx")
    project = create_project(root / "project", synthetic_videos, template, demo_settings())
    Pipeline(project, classifier=demo.DemoClassifier()).run()
    project.db.close()
    return root


@pytest.fixture
def project(processed_project, tmp_path):
    from bruv_app.store import Project

    dst = tmp_path / "copy"
    shutil.copytree(processed_project, dst)
    project = Project(dst / "project")
    # The template path stored in the project points at the session copy; keep it there.
    return project


def ground_truth_species(det) -> str | None:
    """Species of the scene fish best matching a stored detection (the 'perfect human')."""
    box = Detection(det["x1"], det["y1"], det["x2"], det["y2"], 1.0)
    best, best_iou = None, 0.0
    for species, (x1, y1, x2, y2) in demo.scene_at(det["t"]):
        o = iou(box, Detection(x1, y1, x2, y2, 1.0))
        if o > best_iou:
            best, best_iou = species, o
    return best if best_iou > 0.3 else None


def label_like_perfect_human(project) -> None:
    for f in project.all_fish():
        votes = [ground_truth_species(d) for d in project.fish_detections(f["id"])]
        votes = [v for v in votes if v]
        if not votes:
            project.label_fish(f["id"], "not_fish")
            continue
        species = max(set(votes), key=votes.count)
        project.label_fish(f["id"], "species", species=species)


def expected_maxn(fps: float = 1.0) -> dict[str, int]:
    total = demo.CHAPTER_S * 2
    best: dict[str, int] = {}
    for k in range(int(total * fps)):
        visible = [s for s, _ in demo.scene_at(k / fps)]
        for species in set(visible):
            best[species] = max(best.get(species, 0), visible.count(species))
    return best


@pytest.fixture
def rng():
    return np.random.default_rng(0)
