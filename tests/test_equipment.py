"""Reviewable equipment handling and the raw-detection floor (review V2, task 4)."""

import json

import numpy as np
import openpyxl
from fastapi.testclient import TestClient

from bruv_app import demo
from bruv_app.config import Settings
from bruv_app.detect import Detection, ScriptedDetector, make_detector
from bruv_app.equipment import match_card
from bruv_app.evaluate import Box, threshold_curve
from bruv_app.export import export_project
from bruv_app.pipeline import Pipeline, create_project
from bruv_app.server import create_app
from bruv_app.store import to_blob

from .conftest import demo_settings

RIG = (0.85, 0.85, 0.95, 0.97)  # bottom right corner: no scene fish swims there


def _scene(k: int) -> list[tuple[Detection, str]]:
    """What's in frame k: the rig for frames 0-3 and 15-18, a fish crossing its spot in
    frames 8-11, and a faint fish (score 0.2, under the operating threshold) in frame 5."""
    out = []
    if k <= 3 or 15 <= k <= 18:
        out.append((Detection(*RIG, 0.9), "equipment"))
    if 8 <= k <= 11:
        x = 0.70 + 0.05 * (k - 8)
        out.append((Detection(x, 0.85, x + 0.10, 0.97, 0.9), "fish"))
    if k == 5:
        out.append((Detection(0.1, 0.1, 0.2, 0.2, 0.2), "fish"))
    return out


def _scripted(k: int) -> list[Detection]:
    return [d for d, _ in _scene(k)]


def _project(tmp_path, synthetic_videos):
    template = demo.make_mock_template(tmp_path / "t.xlsx")
    # A short merge window keeps the rig's two visits and the crossing fish on cards of their own.
    project = create_project(tmp_path / "p", synthetic_videos, template, demo_settings(gap_seconds=1.0))
    Pipeline(project, detector=ScriptedDetector(_scripted), classifier=demo.DemoClassifier()).run()
    cards = sorted(project.all_fish(), key=lambda f: f["start_t"])
    assert [(round(f["start_t"]), round(f["end_t"])) for f in cards] == [(0, 3), (8, 11), (15, 18)]
    return project, [f["id"] for f in cards]


def test_marking_equipment_queues_its_look_alikes_for_review(tmp_path, synthetic_videos):
    project, (rig_a, crossing, rig_b) = _project(tmp_path, synthetic_videos)
    client = TestClient(create_app(project.root))
    assert client.get("/api/rig").json() == {"refs": [], "candidates": {}}

    assert client.post(f"/api/fish/{rig_a}/equipment", json={}).status_code == 200
    assert project.fish(rig_a)["status"] == "not_fish"
    rig = client.get("/api/rig").json()
    assert rig["refs"] == [{"id": 1, "card": rig_a}]
    # The rig's other visit is queued; the fish that swam across its spot is not.
    assert set(rig["candidates"]) == {str(rig_b)} and rig["candidates"][str(rig_b)]["overlap"] == 1.0
    detail = client.get(f"/api/fish/{rig_b}").json()
    assert detail["rig"]["card"] == rig_a and detail["equipment"] is False
    assert client.get(f"/api/fish/{rig_a}").json()["equipment"] is True

    # Marked together as a batch, which can be undone.
    done = client.post("/api/rig/mark", json={"ids": [rig_b, crossing]}).json()
    assert done["marked"] == [rig_b] and done["left"] == []  # the crossing fish isn't queued, so it's untouched
    assert project.fish(rig_b)["status"] == "not_fish" and project.fish(crossing)["status"] == "pending"
    details = json.loads(project.last_review(rig_b)["details"])
    assert details["equipment"] and details["batch"] == done["batch"] and details["rig"] == 1
    wb = openpyxl.load_workbook(export_project(project, tmp_path / "out.xlsx"))
    status = dict(list(wb["review_status"].iter_rows(values_only=True))[1:])
    assert status["answered as equipment (not a fish)"] == "2 cards, 8 detections"
    undone = client.post("/api/rig/undo", json={"batch": done["batch"]}).json()
    assert undone["undone"] == [rig_b] and project.fish(rig_b)["status"] == "pending"
    assert set(client.get("/api/rig").json()["candidates"]) == {str(rig_b)}

    # A fish joined onto the rig's track: still queued, but left out of answering in one go.
    d = project.fish_detections(rig_b)[0]
    project.db.execute(
        "INSERT INTO detections(frame_id, x1, y1, x2, y2, score, tracklet_id, app) VALUES(?,?,?,?,?,?,?,?)",
        (d["frame_id"], 0.1, 0.1, 0.2, 0.2, 0.9, d["tracklet_id"], d["app"]),
    )
    project.commit()
    assert client.get("/api/rig").json()["candidates"][str(rig_b)]["odd"] == 1
    held = client.post("/api/rig/mark", json={"ids": [rig_b]}).json()
    assert held["marked"] == [] and held["left"] == [rig_b] and project.fish(rig_b)["status"] == "pending"

    # Forgetting the reference stops the queue; answers already given stay.
    assert client.post("/api/rig/1/forget").status_code == 200
    assert client.get("/api/rig").json()["candidates"] == {}
    assert project.fish(rig_a)["status"] == "not_fish"


def test_rig_matching_keeps_fish_that_cross_or_hover():
    grey, orange = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ref = {"id": 1, "fish_id": 9, "x1": RIG[0], "y1": RIG[1], "x2": RIG[2], "y2": RIG[3], "colour": to_blob(grey)}

    def card(boxes, app):
        return [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "app": None if app is None else to_blob(app)} for b in boxes]

    whole = match_card(card([RIG] * 4, grey), ref, 0.7)
    assert (whole.overlap, whole.odd) == (1.0, 0)
    crossing = [(x, 0.85, x + 0.1, 0.97) for x in (0.6, 0.7, 0.8, 0.85)]
    assert match_card(card(crossing, grey), ref, 0.7) is None  # over the rig only briefly
    assert match_card(card([RIG] * 4, orange), ref, 0.7) is None  # in front of it, other colours
    assert match_card(card([RIG] * 4, None), ref, 0.7) is not None  # no colours to compare: place decides
    # Mostly the rig, with a fish joined on (elsewhere, or in front of it): queued, but its odd
    # sightings keep it out of answering in one go.
    assert match_card(card([RIG] * 3 + [(0.6, 0.85, 0.7, 0.97)], grey), ref, 0.7).odd == 1
    assert match_card(card([RIG] * 3, grey) + card([RIG], orange), ref, 0.7).odd == 1


def test_faint_boxes_are_kept_but_never_tracked(tmp_path, synthetic_videos):
    project, cards = _project(tmp_path, synthetic_videos)
    faint = project.db.execute("SELECT * FROM detections WHERE score < 0.3").fetchall()
    assert len(faint) == 1 and faint[0]["tracklet_id"] is None  # kept for threshold curves
    on_cards = {d["id"] for f in cards for d in project.fish_detections(f)}
    assert faint[0]["id"] not in on_cards
    detect = json.loads(project.runs("detect")[0]["info"])
    assert (detect["kept_from"], detect["tracked_from"]) == (0.1, 0.3)


def test_threshold_curve_from_the_kept_boxes(tmp_path, synthetic_videos):
    """E6 without re-running the detector: the faint box (0.2) is a real fish the 0.3
    operating point misses; the rig's boxes are equipment, so they count as false."""
    project, _ = _project(tmp_path, synthetic_videos)
    dep = project.get_meta("deployment")
    truth = []
    for f in project.db.execute("SELECT chapter, src_index, t FROM frames_src").fetchall():
        key = (dep, f["chapter"], f["src_index"], f["t"])
        scene = _scene(round(f["t"]))
        if not scene:
            truth.append(Box(*key, kind="none"))
        for d, kind in scene:
            truth.append(Box(*key, d.x1, d.y1, d.x2, d.y2, kind=kind))
    curve = {p.threshold: p for p in threshold_curve(project, truth, [0.05, 0.1, 0.15, 0.3, 0.95])}
    assert set(curve) == {0.1, 0.15, 0.3, 0.95}  # 0.05 is under what the run kept
    assert (curve[0.15].matched, curve[0.15].missed, curve[0.15].false) == (5, 0, 8)
    assert (curve[0.3].matched, curve[0.3].missed, curve[0.3].false) == (4, 1, 8)
    assert (curve[0.1].kept, curve[0.3].kept, curve[0.95].kept) == (13, 12, 0)
    assert curve[0.15].recall == 1.0 and curve[0.3].recall == 0.8 and curve[0.95].recall == 0.0


def test_the_detector_keeps_boxes_down_to_the_floor(monkeypatch):
    seen = {}

    class Stub:
        def __init__(self, weights, threshold, device):
            seen["threshold"] = threshold

    monkeypatch.setattr("bruv_app.detect.RFDETRDetector", Stub)
    make_detector(Settings(detector="rfdetr", detector_weights="w.pth"), "cpu")
    assert seen["threshold"] == 0.1
    make_detector(Settings(detector="rfdetr", detector_weights="w.pth", det_threshold=0.05), "cpu")
    assert seen["threshold"] == 0.05  # never above the operating threshold
