import openpyxl
import pytest

from bruv_app import demo
from bruv_app.cli import main
from bruv_app.detect import Detection
from bruv_app.evaluate import (
    Box,
    Found,
    RunView,
    annotation_sheet,
    evaluate,
    evaluate_project,
    match_boxes,
    read_sidecar,
    write_sidecar,
)
from bruv_app.export import export_project
from bruv_app.metrics import UNKNOWN
from bruv_app.store import Project

from .conftest import expected_maxn, label_like_perfect_human

DEP = "dep-test"
WHERE = {  # fixed boxes: three fish, equipment, and an empty patch of water
    "a": (0.10, 0.10, 0.20, 0.20),
    "b": (0.50, 0.10, 0.60, 0.20),
    "c": (0.80, 0.10, 0.90, 0.20),
    "rig": (0.30, 0.50, 0.40, 0.60),
    "water": (0.60, 0.70, 0.70, 0.80),
}
TAXON = {"a": "snapper", "b": "jack", "c": "jack"}


def _fish(src, who):
    return Box(DEP, 0, src, src / 10, *WHERE[who], kind="fish", individual=who, taxon=TAXON[who])


def _found(det_id, src, where, tracklet, card, label):
    return Found(det_id, src / 10, Detection(*WHERE[where], 1.0), tracklet, card, label)


def test_matching_is_one_to_one():
    fish = Detection(0.1, 0.1, 0.3, 0.3, 1.0)
    twice = [Detection(0.1, 0.1, 0.3, 0.3, 1.0), Detection(0.11, 0.1, 0.31, 0.3, 1.0)]
    assert match_boxes([fish], twice) == [(0, 0)]  # the second box on the same fish is a false detection
    # Two fish close together: each box goes to its own fish, not both to the bigger overlap.
    left, right = Detection(0.1, 0.1, 0.3, 0.3, 1.0), Detection(0.2, 0.1, 0.4, 0.3, 1.0)
    found = [Detection(0.15, 0.1, 0.35, 0.3, 1.0), Detection(0.21, 0.1, 0.41, 0.3, 1.0)]
    assert sorted(match_boxes([left, right], found)) == [(0, 0), (1, 1)]
    assert match_boxes([left], [Detection(0.5, 0.5, 0.6, 0.6, 1.0)]) == []


def test_stage_scores_on_a_known_scene():
    """Three fish (a snapper, two jacks). The tracker swaps a for b inside tracklet 1 (a
    tracking error), the merge joins jack b's tracklet to jack c's (a merge error), and c
    gets a second card. Every number below can be counted by hand."""
    truth = [
        _fish(0, "a"), _fish(0, "b"), Box(DEP, 0, 0, 0.0, *WHERE["rig"], kind="equipment"),
        _fish(10, "a"), _fish(10, "b"),
        _fish(20, "a"), _fish(20, "b"),  # the detector misses a here
        _fish(30, "a"), _fish(30, "c"),
        _fish(40, "b"), _fish(40, "c"),
        Box(DEP, 0, 50, 5.0, kind="none"),  # checked: nothing there
        Box(DEP, 0, 60, 6.0, *WHERE["a"], kind="unchecked"),  # nobody checked this frame yet
        _fish(70, "a"),  # a frame this run didn't sample
    ]
    frames = {
        (DEP, 0, 0): [_found(1, 0, "a", 1, 1, "snapper"), _found(2, 0, "b", 2, 2, "jack"), _found(3, 0, "rig", 9, 9, None)],
        (DEP, 0, 10): [_found(4, 10, "a", 1, 1, "snapper"), _found(5, 10, "b", 2, 2, "jack")],
        (DEP, 0, 20): [_found(6, 20, "b", 1, 1, "snapper")],  # tracklet 1 jumps to fish b
        (DEP, 0, 30): [_found(7, 30, "a", 3, 1, "snapper"), _found(8, 30, "c", 4, 2, "jack")],
        (DEP, 0, 40): [_found(9, 40, "c", 5, 3, "jack"), _found(11, 40, "b", 7, 5, UNKNOWN)],
        (DEP, 0, 50): [_found(10, 50, "water", 6, 4, None)],
        (DEP, 0, 60): [_found(12, 60, "a", 8, 6, "snapper")],
    }
    chains = {1: [1, 3], 2: [2, 4], 3: [5], 4: [6], 5: [7], 6: [8], 9: [9]}
    r = evaluate(RunView(frames, chains, (960, 540), "frozen"), truth)

    assert (r.frames, r.frames_unchecked, r.frames_unsampled) == (6, 1, 1)
    assert (r.fish, r.matched, r.missed, r.found) == (10, 9, 1, 11)
    assert (r.false, r.false_on_equipment) == (2, 1)
    assert r.precision == pytest.approx(9 / 11) and r.recall == pytest.approx(0.9)
    assert r.recall_by_size == {"60+": (9, 10)}  # 0.1 of 960 px wide
    # Tracking (A): tracklet 1 goes a, a, b.
    assert (r.transitions, r.switches) == (3, 1)
    assert (r.tracklets_eligible, r.tracklets_mixed) == (6, 1)
    assert r.fragments_per_fish == pytest.approx(7 / 3)  # a in 2 tracklets, b in 3, c in 2
    # Merging (B): b->c is a wrong join; the join after tracklet 1 is the tracker's fault.
    assert (r.edges_eligible, r.edges_wrong, r.edges_inherited) == (1, 1, 1)
    assert (r.cards_eligible, r.cards_multi_fish, r.cards_multi_taxon) == (4, 2, 1)
    assert (r.impure_detections, r.pure_eligible_detections) == (2, 9)
    assert (r.cards_tracker_only, r.cards_merge_only, r.cards_both) == (1, 1, 0)
    assert r.duplicate_cards_per_fish == pytest.approx(1.0)  # a on 1 card, b on 3, c on 2
    # Science: two jacks at once, but one was reviewed as unknown.
    assert r.maxn == {"snapper": (1, 1), "jack": (2, 1)}
    text = "\n".join(r.lines())
    assert "1 identity changes in 3 steps" in text and "jack: annotated 2, from the run 1  (-1)" in text


def _scene_sidecar(project) -> list[Box]:
    """The synthetic scene as an annotator would record it: every sampled frame, each fish
    with its own id ("red visits twice": scene fish 0 and 1 are the same red)."""
    dep = project.get_meta("deployment")
    rows = []
    for f in project.db.execute("SELECT chapter, src_index, t FROM frames"):
        visible = [(i, fish, fish.box(f["t"])) for i, fish in enumerate(demo.SCENE) if fish.box(f["t"])]
        if not visible:
            rows.append(Box(dep, f["chapter"], f["src_index"], f["t"], kind="none"))
        w, h = demo.SIZE
        for i, fish, (x1, y1, x2, y2) in visible:
            rows.append(Box(dep, f["chapter"], f["src_index"], f["t"], x1 / w, y1 / h, x2 / w, y2 / h,
                            individual=str(0 if i == 1 else i), taxon=fish.species))
    return rows


def test_synthetic_scene_end_to_end(project):
    label_like_perfect_human(project)
    truth = _scene_sidecar(project)
    report = evaluate_project(project, truth)
    assert report.grouping == "frozen" and report.frames == project.frame_count()
    assert report.recall >= 0.9 and report.precision >= 0.9
    assert report.switches == 0 and report.tracklets_mixed == 0
    assert {t: run for t, (_, run) in report.maxn.items()} == expected_maxn()
    assert all(truth_n == run_n for truth_n, run_n in report.maxn.values())

    # Review edits change the cards, not the score of what the machine built.
    card = next(f for f in project.all_fish() if len(project.fish_tracklets(f["id"])) > 1)
    project.unmerge_fish(card["id"])
    again = evaluate_project(project, truth)
    assert (again.edges_wrong, again.cards_multi_fish, again.switches) == (report.edges_wrong, report.cards_multi_fish, 0)


def test_annotations_must_be_for_this_footage(project):
    wrong = [Box("dep-elsewhere", 0, 0, 0.0, kind="none")]
    with pytest.raises(ValueError, match="dep-elsewhere"):
        evaluate_project(project, wrong)


def test_annotation_sheet_starts_unchecked_and_round_trips(project, tmp_path):
    rows = annotation_sheet(project, [(2.0, 6.0)])
    frames = {(r.chapter, r.src_index) for r in rows}
    assert len(frames) == 4 and all(r.kind == "unchecked" and not r.taxon for r in rows)
    assert all(r.deployment == project.get_meta("deployment") for r in rows)
    path = write_sidecar(tmp_path / "key.csv", rows)
    back = read_sidecar(path)
    assert [(b.src_index, b.det_id, b.kind) for b in back] == [(r.src_index, r.det_id, r.kind) for r in rows]
    # Untouched, it scores nothing: a detector box isn't an answer until someone checks it.
    report = evaluate_project(project, back)
    assert report.frames == 0 and report.frames_unchecked == 4


def test_evaluate_and_sheet_commands(project, tmp_path, capsys):
    label_like_perfect_human(project)
    project.commit()
    write_sidecar(tmp_path / "key.csv", _scene_sidecar(project))
    main(["evaluate", "--project", str(project.root), "--annotations", str(tmp_path / "key.csv")])
    out = capsys.readouterr().out
    assert "24 annotated frames scored" in out and "0 identity changes" in out
    # A snapshot scores the same as the project it froze.
    snapshot = next((project.root / "snapshots").glob("*.db"))
    main(["evaluate", "--project", str(snapshot), "--annotations", str(tmp_path / "key.csv")])
    assert "24 annotated frames scored" in capsys.readouterr().out
    main(["annotation-sheet", "--project", str(project.root), "--window", "0-3", "--out", str(tmp_path / "s.csv")])
    assert "over 3 frames" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["annotation-sheet", "--project", str(project.root), "--window", "soon", "--out", str(tmp_path / "x.csv")])


def test_read_sidecar_rejects_unknown_kinds(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("deployment,chapter,src_index,kind\ndep-x,0,5,shark\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_sidecar(path)


def _maxn(project, tmp_path, name):
    wb = openpyxl.load_workbook(export_project(project, tmp_path / name))
    return {r[0]: r[1] for r in wb["MaxN"].iter_rows(min_row=2, values_only=True)}


def test_maxn_is_unchanged_by_pure_regrouping(project, tmp_path):
    """Scientific output: when every detection keeps its label, cutting cards apart or
    joining them must not change MaxN (it counts detections per frame, not cards)."""
    label_like_perfect_human(project)
    before = _maxn(project, tmp_path, "before.xlsx")
    named = [f for f in project.all_fish() if f["status"] == "species"]
    label = {d["id"]: f["species"] for f in named for d in project.fish_detections(f["id"])}
    for f in [f for f in named if len(project.fish_tracklets(f["id"])) > 1]:
        project.unmerge_fish(f["id"])  # cut every card into its tracked pieces...
    by_species: dict[str, list[int]] = {}
    for f in project.all_fish():
        first = project.fish_detections(f["id"])[0]["id"]
        if first in label:
            by_species.setdefault(label[first], []).append(f["id"])
    for cards in by_species.values():  # ...then join all of a species' pieces into one card
        for other in cards[1:]:
            project.merge_fish(cards[0], other)
    for f in project.all_fish():  # every detection keeps the label it had
        labels = {label.get(d["id"]) for d in project.fish_detections(f["id"])}
        if labels != {None}:
            assert len(labels) == 1
            project.label_fish(f["id"], "species", species=labels.pop())
    assert sum(f["status"] == "species" for f in project.all_fish()) == 3
    assert _maxn(project, tmp_path, "after.xlsx") == before == expected_maxn()


def test_score_matching_is_one_to_one(project, tmp_path):
    """A re-run with two boxes on one fish maps only one of them to the reviewed answer."""
    import shutil

    shutil.copytree(project.root, tmp_path / "rerun")
    rerun = Project(tmp_path / "rerun")
    d = rerun.db.execute("SELECT * FROM detections LIMIT 1").fetchone()
    rerun.db.execute(
        "INSERT INTO detections(frame_id, x1, y1, x2, y2, score) VALUES(?,?,?,?,?,?)",
        (d["frame_id"], d["x1"] + 0.002, d["y1"], d["x2"] + 0.002, d["y2"], 0.5),
    )
    rerun.commit()
    from bruv_app.score import match_detections

    mapping = match_detections(project, rerun)  # rerun ids -> project ids
    assert len(mapping) == len(set(mapping.values()))
    assert sum(v == d["id"] for v in mapping.values()) == 1
