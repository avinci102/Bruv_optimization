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


def test_crossing_fish_keep_their_own_tracks():
    # A swims right fast, B left slowly; they pass each other at the third frame. Both look
    # the same, so only their direction of travel tells them apart afterwards.
    a = [0.02, 0.20, 0.38, 0.58, 0.78]
    b = [0.58, 0.50, 0.42, 0.32, 0.22]
    frames = [[(2 * i, box(xa, 0.5, 0.2, 0.1)), (2 * i + 1, box(xb, 0.5, 0.2, 0.1))] for i, (xa, xb) in enumerate(zip(a, b))]
    tracks = track_all(frames)
    assert sorted([k for _, k in t.members] for t in tracks) == [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]


def test_appearance_gate_blocks_identity_swap():
    red, green = l2_normalize(np.array([1.0, 0.0])), l2_normalize(np.array([0.0, 1.0]))
    frames = [[(0, box(0.5, 0.5, app=red))], [(1, box(0.52, 0.5, app=green))]]
    assert len(track_all(frames)) == 2
    frames = [[(0, box(0.5, 0.5, app=red))], [(1, box(0.52, 0.5, app=red))]]
    assert len(track_all(frames)) == 1


def test_resolution_from_cfd_file_names():
    from bruv_app.detect import resolution_from_name

    assert resolution_from_name("cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth") == 640
    assert resolution_from_name("cfd-rf-detr-medium-1024-2026.03.24.cp-011.20260706-release.pth") == 1024
    assert resolution_from_name("checkpoint_best_total.pth") is None


def test_prompt_ensemble_averages_templates():
    from bruv_app.classify import prompt_ensemble

    templates = (lambda c: f"a photo of {c}.", lambda c: f"a {c} underwater.")
    seen = []

    def encode(prompts):
        seen.extend(prompts)
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    out = prompt_ensemble(encode, ["snapper"], templates)
    assert seen == ["a photo of snapper.", "a snapper underwater."]
    assert out.shape == (1, 2)
    assert np.allclose(out[0], [np.sqrt(0.5), np.sqrt(0.5)])
