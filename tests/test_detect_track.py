import numpy as np

from bruv_app import demo
from bruv_app.classify import l2_normalize
from bruv_app.detect import ColorBlobDetector, Detection, ScriptedDetector
from bruv_app.track import track_all


def test_color_blob_detector_finds_every_scene_fish(rng):
    detector = ColorBlobDetector()
    for t in (1.0, 4.0, 17.0, 22.0):
        frame = demo.render_frame(t, rng)
        dets = detector.detect([frame])[0]
        assert len(dets) == len(demo.scene_at(t)), t


def test_scripted_detector_counts_calls():
    det = ScriptedDetector(lambda i: [Detection(0, 0, 0.1, 0.1, 0.5)] * i)
    out = det.detect([None, None, None])
    assert [len(x) for x in out] == [0, 1, 2]


def box(cx, cy, w=0.1, h=0.05, app=None):
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, app)


def test_tracker_links_a_moving_fish():
    frames = [[(i, box(0.1 + 0.05 * i, 0.5))] for i in range(6)]
    tracks = track_all(frames)
    assert len(tracks) == 1
    assert [k for _, k in tracks[0].members] == list(range(6))


def test_tracker_keeps_two_fish_apart():
    frames = [[(2 * i, box(0.1 + 0.03 * i, 0.2)), (2 * i + 1, box(0.9 - 0.03 * i, 0.8))] for i in range(5)]
    tracks = track_all(frames)
    assert len(tracks) == 2
    assert {len(t.members) for t in tracks} == {5}


def test_tracker_survives_one_missed_frame_but_not_three():
    frames = [[(0, box(0.2, 0.5))], [], [(1, box(0.24, 0.5))], [], [], [], [(2, box(0.3, 0.5))]]
    tracks = track_all(frames, max_missed=1)
    assert sorted(len(t.members) for t in tracks) == [1, 2]


def test_appearance_gate_blocks_identity_swap():
    red, green = l2_normalize(np.array([1.0, 0.0])), l2_normalize(np.array([0.0, 1.0]))
    frames = [[(0, box(0.5, 0.5, app=red))], [(1, box(0.52, 0.5, app=green))]]
    assert len(track_all(frames)) == 2
    frames = [[(0, box(0.5, 0.5, app=red))], [(1, box(0.52, 0.5, app=red))]]
    assert len(track_all(frames)) == 1
