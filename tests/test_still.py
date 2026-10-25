"""Still objects hidden before tracking, shorter tracks, and rebuilding reviewed cards
without losing the answers (docs/SPEC.md §13, after the first 185 answers)."""

import json

import numpy as np
from fastapi.testclient import TestClient

from bruv_app import demo
from bruv_app.config import Settings
from bruv_app.detect import ColorBlobDetector, Detection
from bruv_app.export import export_project
from bruv_app.pipeline import CARRY_KEY, EARLIER_KEY, Pipeline, create_project
from bruv_app.server import create_app
from bruv_app.still import background_change, is_still
from bruv_app.store import REVIEWED
from bruv_app.track import track_all

from .conftest import demo_settings, label_like_perfect_human

RIG = (0.85, 0.85, 0.95, 0.97)  # bottom right: no synthetic fish swims there


def test_a_box_that_never_changes_is_still_and_a_moving_fish_is_not():
    size = (40, 20)
    frames = [(i, float(i), f"f{i}") for i in range(30)]

    def load(path, _size):
        k = int(path[1:])
        image = np.full((20, 40, 3), 80, np.uint8)
        image[12:18, 30:38] = 200  # the canister: same place, same colour, every frame
        x = k % 25  # a fish swimming across
        image[2:8, x:x + 6] = (220, 40, 30)
        return image

    boxes = {k: [(100 + k, (30 / 40, 12 / 20, 38 / 40, 18 / 20)), (200 + k, ((k % 25) / 40, 2 / 20, (k % 25 + 6) / 40, 8 / 20))]
             for k in range(30)}
    change = background_change(frames, boxes, size, window_s=60.0, load=load)
    assert all(change[100 + k] == 0.0 for k in range(30))
    assert all(change[200 + k] > 0.5 for k in range(30))
    # Judged only when wide enough, and the cut-off is "under".
    assert is_still((0.1, 0.1, 0.2, 0.2), 0.05, 0.06, 0.10)
    assert not is_still((0.1, 0.1, 0.13, 0.2), 0.0, 0.06, 0.10)
    assert not is_still((0.1, 0.1, 0.2, 0.2), 0.10, 0.06, 0.10)
    assert not is_still((0.1, 0.1, 0.2, 0.2), None, 0.06, 0.10)


class FishAndRig:
    """The synthetic fish as the colour detector finds them, plus the rig in its corner of
    still water in every frame."""

    name = "fish-and-rig"

    def __init__(self):
        self.blobs = ColorBlobDetector()

    def detect(self, images):
        return [found + [Detection(*RIG, 0.9)] for found in self.blobs.detect(images)]


def _rig_project(tmp_path, synthetic_videos, **settings):
    template = demo.make_mock_template(tmp_path / "t.xlsx")
    project = create_project(tmp_path / "p", synthetic_videos, template, demo_settings(**settings))
    Pipeline(project, detector=FishAndRig(), classifier=demo.DemoClassifier()).run()
    return project


def _rig_ids(project):
    rows = project.db.execute("SELECT id, x1, y1 FROM detections").fetchall()
    return {r["id"] for r in rows if (round(r["x1"], 2), round(r["y1"], 2)) == RIG[:2]}


def test_the_rig_is_hidden_before_tracking_and_the_fish_keep_their_cards(tmp_path, synthetic_videos):
    project = _rig_project(tmp_path, synthetic_videos)
    rig = _rig_ids(project)
    assert len(rig) == project.frame_count()
    assert project.still_ids(project.load_settings()) == rig
    on_cards = {d["id"] for f in project.all_fish() for d in project.fish_detections(f["id"])}
    assert not rig & on_cards  # no card holds the rig, and it joins no fish's card
    tracked = {r[0] for r in project.db.execute("SELECT id FROM detections WHERE tracklet_id IS NOT NULL")}
    assert tracked == {r[0] for r in project.db.execute("SELECT id FROM detections")} - rig
    assert project.get_meta("still_hidden") == len(rig)
    # With the filter off, the rig is tracked like any box and makes cards again.
    off = _rig_project(tmp_path / "off", synthetic_videos, still_filter=False)
    assert _rig_ids(off) & {d["id"] for f in off.all_fish() for d in off.fish_detections(f["id"])}
    # Still objects aren't counted: the export's MaxN is the fish's.
    label_like_perfect_human(project)
    export_project(project, tmp_path / "out.xlsx")


def test_settings_files_from_before_keep_their_behaviour(tmp_path):
    assert Settings().still_filter and Settings().track_refuse_contested
    assert Settings().track_max_missed_s == 0.25
    path = tmp_path / "settings.json"
    old = {k: v for k, v in json.loads(json.dumps(Settings().__dict__)).items()
           if k not in ("still_filter", "track_refuse_contested")}
    old["track_max_missed_s"] = 1.0
    path.write_text(json.dumps(old))
    loaded = Settings.load(path)
    assert not loaded.still_filter and not loaded.track_refuse_contested and loaded.track_max_missed_s == 1.0


def _box(x: float) -> Detection:
    return Detection(x, 0.4, x + 0.1, 0.5, 0.9)


def test_a_contested_link_starts_new_pieces_instead_of_jumping():
    """Two fish side by side: the box between them suits fish B better than fish A, yet the
    best overall pairing gives it to A. Refusing such links keeps each piece one fish."""
    frames = [[(1, _box(0.10)), (2, _box(0.24))], [(3, _box(0.20)), (4, _box(0.36))]]
    linked = track_all(frames, max_missed=0)
    assert sorted(t.members for t in linked) == [[(0, 1), (1, 3)], [(0, 2), (1, 4)]]
    assert all(m < 0 for t in linked for _, m, _ in t.links.values())
    refused = track_all(frames, max_missed=0, refuse_contested=True)
    assert len(refused) == 4 and all(len(t.members) == 1 for t in refused)


def test_rebuilding_keeps_the_answers_and_new_cards_get_new_numbers(project):
    label_like_perfect_human(project)
    answered = {f["id"]: (f["status"], f["species"]) for f in project.all_fish() if f["status"] in REVIEWED}
    by_det = {d["id"]: answered[f] for f in answered for d in project.fish_detections(f)}
    # Shorter tracks and a smaller merge window cut the cards into more pieces.
    settings = project.load_settings()
    settings.gap_seconds, settings.track_max_missed_s = 0.5, 0.0
    project.save_settings(settings)
    report = Pipeline(project, classifier=demo.DemoClassifier()).rebuild()
    cards = project.all_fish()
    assert len(cards) >= len(answered)
    assert min(f["id"] for f in cards) > max(answered)  # never an old card's number
    for f in cards:
        held = {by_det[d["id"]] for d in project.fish_detections(f["id"]) if d["id"] in by_det}
        if f["status"] in REVIEWED:
            assert held == {(f["status"], f["species"])}
            assert project.answer_details(f["id"])["carried"] is True
    assert report["carried"] == sum(f["status"] in REVIEWED for f in cards) > 0
    assert report["carried"] + report["partly_answered"] + report["conflicting"] + report["new"] == len(cards)
    # Frozen before and after, and the rebuild is on record with its counts.
    snaps = sorted(p.name for p in (project.root / "snapshots").glob("*.db"))
    assert any(s.endswith("before-rebuild.db") for s in snaps) and any(s.endswith("rebuilt.db") for s in snaps)
    run = json.loads(project.runs("rebuild")[-1]["info"])
    assert run["carried"] == report["carried"] and run["answered_detections"] == len(by_det)


def test_a_partly_answered_card_says_what_its_sightings_were(project):
    """Sightings moved off an answered card and never answered join it again in a rebuild
    (tracking doesn't know about the move): the card waits for you, and the review page
    says what most of its sightings were answered."""
    first = project.all_fish()[0]
    dets = [d["id"] for d in project.fish_detections(first["id"])]
    project.label_fish(first["id"], "species", species=demo.RED)
    project.move_to_new_fish(first["id"], dets[-3:])
    answered = set(dets[:-3])
    report = Pipeline(project, classifier=demo.DemoClassifier()).rebuild()
    assert report["partly_answered"] >= 1 and set(project.get_meta(EARLIER_KEY)) == {str(d) for d in answered}
    client = TestClient(create_app(project.root))
    for f in project.all_fish():
        dets = {d["id"] for d in project.fish_detections(f["id"])}
        earlier = client.get(f"/api/fish/{f['id']}").json()["earlier"]
        if f["status"] == "pending" and dets & answered:
            assert earlier == {"status": "species", "species": demo.RED, "sightings": len(dets & answered),
                               "of": len(dets), "answers": 1}
        else:
            assert earlier is None


def test_a_stopped_rebuild_loses_no_answers(project):
    label_like_perfect_human(project)
    answered = sum(f["status"] in REVIEWED for f in project.all_fish())
    # Stopped after the cards were cleared: the answers were set aside first.
    project.set_meta(CARRY_KEY, {str(k): v for k, v in project.answers_by_detection().items()})
    project.reset_tracking()
    assert not project.all_fish()
    report = Pipeline(project, classifier=demo.DemoClassifier()).rebuild()
    assert report["carried"] == sum(f["status"] in REVIEWED for f in project.all_fish()) >= answered
    assert project.get_meta(CARRY_KEY) is None


def test_a_card_mixing_answers_waits_for_the_reviewer(project):
    a, b = project.all_fish()[:2]
    project.label_fish(a["id"], "species", species=demo.RED)
    project.label_fish(b["id"], "species", species=demo.YELLOW)
    answers = project.answers_by_detection()
    # Put both cards' detections on one card, as a rebuild with a long merge window might.
    project.merge_fish(a["id"], b["id"])
    report = project.carry_answers(answers)
    assert report["conflicting"] == 1 and project.fish(a["id"])["status"] == "pending"


def test_a_merged_away_card_number_is_never_reused(project):
    a, b = sorted(f["id"] for f in project.all_fish())[-2:]
    project.label_fish(b, "species", species=demo.RED)
    project.merge_fish(a, b)  # b, the highest number, is gone
    new = project.create_fish([])
    assert new > b
    # b's answer stays b's: the new card has none.
    assert project.last_review(new) is None
