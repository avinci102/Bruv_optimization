"""Following one fish frame by frame (follow.py), and how evaluate.py scores it."""

import csv

import pytest
from fastapi.testclient import TestClient

from bruv_app.cli import main
from bruv_app.detect import Detection
from bruv_app.evaluate import COLUMNS, Box, Found, RunView, evaluate, evaluate_project, read_sidecar, write_sidecar
from bruv_app.follow import sidecar_rows
from bruv_app.server import create_app

DEP = "dep-test"
WHERE = {
    "a": (0.10, 0.10, 0.20, 0.20),
    "b": (0.60, 0.60, 0.70, 0.70),
    "beside_a": (0.18, 0.10, 0.28, 0.20),  # overlapping a: another fish close by
}


def _row(src, where=None, **kw):
    box = WHERE[where] if where else (0.15, 0.15, 0.15, 0.15)  # no box: where the fish was
    return Box(DEP, 0, src, src / 10, *box, individual="F1", scope="individual", **kw)


def _found(det_id, src, where, tracklet, card):
    return Found(det_id, src / 10, Detection(*WHERE[where], 1.0), tracklet, card, None)


def test_a_followed_fish_scores_identity_and_why_its_card_changed():
    """Fish a, followed: boxed at 0 and 1 s (card 1), no box at 2 s, on card 3 at 3 s, and on
    card 4 at 4 s with another fish overlapping it; gone at 5 s, when card 4 has carried on
    with fish b (a swap). Fish b was never followed."""
    truth = [_row(0, "a"), _row(10, "a"), _row(20), _row(30, "a"), _row(40, "a"),
             Box(DEP, 0, 50, 5.0, kind="none", individual="F1", identity="", scope="individual")]
    frames = {
        (DEP, 0, 0): [_found(1, 0, "a", 1, 1), _found(2, 0, "b", 2, 2)],
        (DEP, 0, 10): [_found(3, 10, "a", 1, 1)],
        (DEP, 0, 20): [],
        (DEP, 0, 30): [_found(4, 30, "a", 3, 3)],
        (DEP, 0, 40): [_found(5, 40, "a", 4, 4), _found(6, 40, "beside_a", 5, 5)],
        (DEP, 0, 50): [_found(7, 50, "b", 4, 4)],
    }
    times = {key: key[2] / 10 for key in frames}
    r = evaluate(RunView(frames, {1: [1], 2: [2], 3: [3], 4: [4], 5: [5]}, (960, 540), "frozen", times), truth)
    # Only fish a was annotated: b's box is no false detection, and detection isn't scored.
    assert (r.frames, r.fish, r.false) == (0, 0, 0)
    assert (r.followed, r.follow_frames, r.follow_present, r.follow_found) == (1, 6, 5, 4)
    assert r.cards_per_followed == 3 and r.fragments_per_fish == 3
    assert r.card_changes == {"gap in its boxes": 1, "near another fish": 1}
    assert (r.cards_followed, r.cards_with_another) == (3, 1)  # card 4 holds b when a has gone
    text = "\n".join(r.lines())
    assert "1 fish followed over 6 frames" in text and "4/5 (80%)" in text


def test_a_sidecar_without_scope_is_whole_frames(tmp_path):
    path = tmp_path / "old.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c in COLUMNS if c != "scope"])
        writer.writeheader()
        writer.writerow({"deployment": DEP, "chapter": 0, "src_index": 0, "kind": "none"})
    assert read_sidecar(path)[0].scope == "frame"


def _contiguous_card(project):
    """A card whose sightings are in consecutive frames, one box per frame."""
    seqs = {r["id"]: r["seq"] for r in project.db.execute("SELECT id, seq FROM frames")}
    for f in project.all_fish():
        frames = [seqs[d["frame_id"]] for d in project.fish_detections(f["id"])]
        if len(frames) >= 3 and len(set(frames)) == len(frames) and frames[-1] - frames[0] == len(frames) - 1:
            return f["id"]
    raise AssertionError("no contiguous card in the synthetic project")


def test_following_a_fish_forward_then_back(project, tmp_path, capsys):
    client = TestClient(create_app(project.root))
    assert 'id="follow"' in client.get("/").text and "/static/follow.js?v=" in client.get("/follow").text
    card = _contiguous_card(project)
    dets = {d["frame_id"]: d["id"] for d in project.fish_detections(card)}
    view = client.post("/api/follows", json={"card": card, "gone_s": 2}).json()
    follow_id = view["follow"]["id"]
    assert view["phase"] == "forward" and view["start"]["id"] in dets and view["frame"]["id"] != view["start"]["id"]
    assert view["last"] is not None  # the start box: where to look first
    other = next(d for d in project.db.execute("SELECT id, frame_id FROM detections") if d["frame_id"] != view["frame"]["id"])
    bad = client.post(f"/api/follows/{follow_id}/mark", json={"frame_id": view["frame"]["id"], "state": "box", "det_id": other["id"]})
    assert bad.status_code == 400
    assert client.post(f"/api/follows/{follow_id}/mark", json={"frame_id": view["frame"]["id"], "state": "maybe"}).status_code == 400

    phases, steps = [], 0
    while view["phase"] != "done":
        phases.append(view["phase"])
        frame = view["frame"]["id"]
        body = {"frame_id": frame, "state": "box", "det_id": dets[frame]} if frame in dets else {"frame_id": frame, "state": "gone"}
        view = client.post(f"/api/follows/{follow_id}/mark", json=body).json()
        steps += 1
        assert steps < 200
    assert phases[0] == "forward" and phases[-1] == "backward" and phases == sorted(phases, key=lambda p: p != "forward")
    f = view["follow"]
    # Every sighting of the card was clicked; two seconds of "gone" closes each direction (1 fps).
    assert f["done"] and f["boxes"] == len(dets) and f["cards"] == {str(card): len(dets)} and not f["breaks"]
    assert 1 <= f["gone"] <= 4

    # Undo takes back the last frame: the follow isn't done any more and offers it again.
    view = client.post(f"/api/follows/{follow_id}/undo").json()
    assert view["phase"] == "backward" and not view["follow"]["done"]
    view = client.post(f"/api/follows/{follow_id}/mark", json={"frame_id": view["frame"]["id"], "state": "gone"}).json()
    assert view["phase"] == "done"

    # The follow as sidecar rows, and scored: its fish on one card, every box found.
    rows = sidecar_rows(project)
    assert {r.scope for r in rows} == {"individual"} and len({r.individual for r in rows}) == 1
    assert sum(r.kind == "fish" for r in rows) == len(dets)
    report = evaluate_project(project, [])
    assert (report.followed, report.follow_present, report.follow_found) == (1, len(dets), len(dets))
    assert report.cards_per_followed == 1 and report.switches == 0 and report.frames == 0
    # A saved sheet scores the same, and isn't counted twice beside the project's own follows.
    path = write_sidecar(tmp_path / "follows.csv", rows)
    assert read_sidecar(path) == rows
    for again in (evaluate_project(project, read_sidecar(path), follows=False), evaluate_project(project, read_sidecar(path))):
        assert (again.followed, again.follow_present, again.follow_found) == (1, len(dets), len(dets))

    main(["follow-sheet", "--project", str(project.root), "--out", str(tmp_path / "sheet.csv")])
    assert "1 fish followed" in capsys.readouterr().out
    main(["evaluate", "--project", str(project.root)])
    assert "Following: 1 fish followed" in capsys.readouterr().out

    assert [s["id"] for s in client.get("/api/follows").json()] == [follow_id]
    assert client.delete(f"/api/follows/{follow_id}").json() == {"ok": True}
    assert client.get("/api/follows").json() == [] and client.get(f"/api/follows/{follow_id}").status_code == 404


def test_a_box_another_follow_took_is_named(project):
    client = TestClient(create_app(project.root))
    card = _contiguous_card(project)
    first = client.post("/api/follows", json={"card": card}).json()
    frame = first["frame"]["id"]
    det = next(b["det"] for b in first["boxes"] if b["card"] == card)
    client.post(f"/api/follows/{first['follow']['id']}/mark", json={"frame_id": frame, "state": "box", "det_id": det})
    second = client.post("/api/follows", json={"card": card}).json()
    assert second["frame"]["id"] == frame  # both start from the card's best view
    taken = {b["det"]: b["taken"] for b in second["boxes"]}
    assert taken[det] == first["follow"]["name"]
    # Both follows hold boxes of this card: the summary says so.
    client.post(f"/api/follows/{second['follow']['id']}/mark", json={"frame_id": frame, "state": "box", "det_id": det})
    shared = client.get(f"/api/follows/{second['follow']['id']}").json()["follow"]["shared"]
    assert shared == {str(card): [first["follow"]["name"]]}


def test_a_card_with_a_box_where_the_fish_was_not(project):
    client = TestClient(create_app(project.root))
    card = _contiguous_card(project)
    view = client.post("/api/follows", json={"card": card}).json()
    assert any(b["card"] == card for b in view["boxes"])  # the card goes on in the next frame...
    view = client.post(f"/api/follows/{view['follow']['id']}/mark",
                       json={"frame_id": view["frame"]["id"], "state": "gone"}).json()
    assert view["follow"]["others"] == {str(card): 1}  # ...but the person said the fish had gone


@pytest.mark.parametrize("gone", [0.0, 100.0])
def test_how_long_to_keep_looking_is_kept_sensible(project, gone):
    client = TestClient(create_app(project.root))
    view = client.post("/api/follows", json={"card": _contiguous_card(project), "gone_s": gone}).json()
    assert 1.0 <= view["gone_s"] <= 30.0
