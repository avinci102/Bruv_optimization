import numpy as np

from bruv_app.classify import l2_normalize
from bruv_app.continuity import TrackletSummary, merge_tracklets
from bruv_app.metrics import UNKNOWN, FishLabel, compute_maxn, compute_sightings, per_frame_counts

A = l2_normalize(np.array([1.0, 0.1, 0.0]))
B = l2_normalize(np.array([0.0, 1.0, 0.1]))


def tl(id, start, end, emb=A, size=0.1, px=100.0):
    return TrackletSummary(id, start, end, emb, size, size, px)


def test_returning_fish_within_gap_is_one_card():
    assert merge_tracklets([tl(1, 0, 5), tl(2, 9, 12)], gap_seconds=10) == [[1, 2]]


def test_gap_too_long_gives_two_cards():
    assert merge_tracklets([tl(1, 0, 5), tl(2, 16, 20)], gap_seconds=10) == [[1], [2]]


def test_gap_window_is_configurable():
    assert merge_tracklets([tl(1, 0, 5), tl(2, 16, 20)], gap_seconds=12) == [[1, 2]]


def test_overlapping_tracklets_never_merge():
    assert merge_tracklets([tl(1, 0, 5), tl(2, 4, 8)]) == [[1], [2]]


def test_different_looking_fish_dont_merge():
    assert merge_tracklets([tl(1, 0, 5), tl(2, 7, 9, emb=B)]) == [[1], [2]]


def test_implausible_size_change_blocks_merge():
    assert merge_tracklets([tl(1, 0, 5, size=0.05), tl(2, 7, 9, size=0.3)]) == [[1], [2]]


def test_tiny_fish_join_nothing():
    # A distant blob between two sightings of a fish would chain them through itself.
    pieces = [tl(1, 0, 5), tl(2, 7, 9, px=20), tl(3, 11, 14)]
    assert merge_tracklets(pieces) == [[1, 2, 3]]
    assert merge_tracklets(pieces, tiny_px=40) == [[1, 3], [2]]


def test_best_similarity_wins_and_chains_stay_linear():
    near = l2_normalize(A + 0.05 * B)
    far = l2_normalize(A + 0.6 * B)
    # Two fish (2 and 3, on screen together) could follow 1; the more similar one wins.
    chains = merge_tracklets([tl(1, 0, 5), tl(2, 7, 9, emb=far), tl(3, 7, 9, emb=near)], min_similarity=0.5)
    assert chains == [[1, 3], [2]]


def test_maxn_is_from_frames_not_from_cards():
    # One card for a fish that leaves and returns: MaxN stays 1.
    # Two cards for fish seen together: MaxN 2. Unknown fish never count.
    dets = [(1.0, 10), (2.0, 10), (8.0, 10), (5.0, 20), (5.0, 21), (5.0, 22), (6.0, 30)]
    labels = {10: "snapper", 20: "snapper", 21: "snapper", 22: "jack", 30: UNKNOWN}
    rows = {r.species: r for r in compute_maxn(per_frame_counts(dets, labels))}
    assert set(rows) == {"snapper", "jack"}
    assert rows["snapper"].maxn == 2 and rows["snapper"].t_maxn == 5.0
    assert rows["snapper"].first_t == 1.0
    assert rows["jack"].maxn == 1 and rows["jack"].first_t == 5.0


def test_unreviewed_fish_dont_count():
    dets = [(1.0, 1), (1.0, 2)]
    assert compute_maxn(per_frame_counts(dets, {1: "snapper"}))[0].maxn == 1


def test_sightings_group_by_species_and_gap_with_peak_count():
    fish = [
        FishLabel(1, "snapper", 0, 10),
        FishLabel(2, "snapper", 5, 12),
        FishLabel(3, "snapper", 40, 45),
        FishLabel(4, "jack", 6, 8),
    ]
    dets = [(t, 1) for t in range(0, 11)] + [(t, 2) for t in range(5, 13)] + [(t, 3) for t in range(40, 46)]
    dets += [(6, 4), (7, 4), (8, 4)]
    sightings = compute_sightings(fish, dets, gap_seconds=10)
    snapper = [s for s in sightings if s.species == "snapper"]
    assert [(s.t_in, s.t_out, s.count) for s in snapper] == [(0, 12, 2), (40, 45, 1)]
    assert [s.fish_ids for s in snapper] == [[1, 2], [3]]
    assert [s.species for s in sightings] == ["snapper", "jack", "snapper"]
