"""E1: crops from four sources on the same frames and boxes (review V2, task 2)."""

import csv
import json

import pytest

from bruv_app import demo
from bruv_app.cli import main
from bruv_app.crops import native_band, position_band, run
from bruv_app.pipeline import Pipeline

from .conftest import label_like_perfect_human


def test_e1_compares_the_same_crops_from_four_sources(project):
    label_like_perfect_human(project)
    summary, out = run(project, demo.DemoClassifier())
    assert summary["crops"] > 0 and summary["usable"] == summary["crops"]
    assert set(summary["alignment"]) <= {"aligned", "static"} and summary["alignment"].get("aligned")
    pieces = summary["per_piece"]
    assert set(pieces) == {"A", "A'", "B", "C"}
    assert pieces["A"]["labelled"] > 0 and pieces["A"]["top1_right"] == pieces["A"]["labelled"]
    with open(out / "crops.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == summary["crops"]
    assert {"A top1", "A' top1", "B top1", "C top1", "native_px", "position", "alignment", "label"} <= set(rows[0])
    # The source frames here are 640 px wide: the native box sizes are in their pixels.
    assert all(float(r["native_px"]) < 640 for r in rows)
    saved = json.loads((out / "summary.json").read_text())
    assert saved["provenance"]["classifier"] == {"name": "demo-colour"} and saved["provenance"]["crop_pad"] == 0.1
    assert (out / "summary.txt").read_text(encoding="utf-8").startswith("E1 crop sources:")


def test_e1_leaves_out_crops_from_the_wrong_frame(project):
    """A frame numbered three frames off (as an older version's resume could leave it) is
    caught by the check against the saved review frame, and its crops are left out."""
    frame = project.db.execute(
        "SELECT DISTINCT frame_id FROM detections WHERE embedding IS NOT NULL ORDER BY frame_id LIMIT 1 OFFSET 5"
    ).fetchone()[0]
    project.db.execute("UPDATE frame_sources SET src_index = src_index + 3 WHERE frame_id=?", (frame,))
    project.commit()
    summary, out = run(project, demo.DemoClassifier())
    with open(out / "crops.csv", encoding="utf-8") as f:
        shifted = [r for r in csv.DictReader(f) if r["src_index"] == str(project.frame(frame).src_index)]
    assert shifted and all(r["alignment"] in ("misaligned", "different") for r in shifted)
    assert summary["usable"] == summary["crops"] - len(shifted)


def test_position_and_size_bands():
    assert position_band(0.45, 0.45, 0.55, 0.55) == "centre"
    assert position_band(0.65, 0.6, 0.85, 0.8) == "middle"
    assert position_band(0.0, 0.0, 0.1, 0.1) == "edge"
    assert [native_band(px) for px in (30, 60, 100, 200, 500)] == ["0-60", "60-120", "60-120", "120-240", "240+"]


def test_capture_is_recorded_and_completed_by_hand(project, capsys):
    capture = project.get_meta("capture")
    assert capture["size"] == list(demo.SIZE) and capture["lens"] == "unknown" and capture["camera"]
    main(["capture", "--project", str(project.root), "--lens", "wide", "--housing", "flat port"])
    assert "lens: wide" in capsys.readouterr().out
    assert {k: project.get_meta("capture")[k] for k in ("lens", "housing", "stabilization")} == {
        "lens": "wide", "housing": "flat port", "stabilization": "unknown"}


def test_crop_test_needs_the_projects_own_model(project):
    with pytest.raises(SystemExit):  # the demo project was scored with the demo model
        main(["crop-test", "--project", str(project.root), "--limit", "2"])


def test_embeddings_are_reused_only_when_made_the_same_way(project, monkeypatch):
    class Counting(demo.DemoClassifier):
        crops = 0

        def embed(self, crops):
            Counting.crops += len(crops)
            return super().embed(crops)

    # A project from before the full key (model and padding only): its embeddings are kept.
    project.set_meta("embedding_key", "demo-colour pad=0.1")
    Pipeline(project, classifier=Counting()).run(force=True)
    assert Counting.crops == 0 and json.loads(project.get_meta("embedding_key"))["model"] == "demo-colour"
    # Crops from differently saved frames are a different source: embedded again.
    settings = project.load_settings()
    settings.jpeg_quality = 70
    project.save_settings(settings)
    Pipeline(project, classifier=Counting()).run(force=True)
    assert Counting.crops > 0
