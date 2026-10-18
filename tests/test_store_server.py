import json

import pytest
from fastapi.testclient import TestClient

from bruv_app import demo, server
from bruv_app.pipeline import update_extra_species
from bruv_app.server import create_app


def multi_tracklet_fish(project):
    return next(f for f in project.all_fish() if len(project.fish_tracklets(f["id"])) > 1)


def test_label_requires_a_species(project):
    fid = project.all_fish()[0]["id"]
    with pytest.raises(ValueError):
        project.label_fish(fid, "species")
    with pytest.raises(ValueError):
        project.label_fish(fid, "maybe")
    project.label_fish(fid, "not_fish", species="ignored")
    assert project.fish(fid)["species"] is None
    assert project.reviewed_count() == 1


def test_split_inside_a_tracklet(project):
    f = project.all_fish()[0]
    dets = project.fish_detections(f["id"])
    cut = dets[1]["t"]  # second sighting: falls inside the first tracklet
    before = len(dets)
    new_id = project.split_fish(f["id"], cut)
    left, right = project.fish_detections(f["id"]), project.fish_detections(new_id)
    assert len(left) + len(right) == before
    assert max(d["t"] for d in left) < cut <= min(d["t"] for d in right)
    assert project.fish(new_id)["status"] == "pending"


def test_split_needs_detections_on_both_sides(project):
    f = project.all_fish()[0]
    with pytest.raises(ValueError):
        project.split_fish(f["id"], f["start_t"])


def test_multiple_fish_breaks_a_card_into_its_tracked_pieces(project):
    f = multi_tracklet_fish(project)
    pieces = project.fish_tracklets(f["id"])
    project.label_fish(f["id"], "species", species=demo.RED)
    new_ids = project.unmerge_fish(f["id"])
    assert len(new_ids) == len(pieces) - 1
    for fid, tid in zip([f["id"], *new_ids], pieces):
        assert project.fish_tracklets(fid) == [tid]  # one tracked piece per card
        assert project.fish(fid)["status"] == "pending"
    log = project.corrections()
    assert [c["action"] for c in log] == ["multiple_fish"]
    assert json.loads(log[0]["details"]) == {"tracklets": pieces, "new_cards": new_ids}
    single = next(f for f in project.all_fish() if len(project.fish_tracklets(f["id"])) == 1)
    with pytest.raises(ValueError):
        project.unmerge_fish(single["id"])


def test_ticked_sightings_move_to_a_new_card(project):
    f = multi_tracklet_fish(project)
    project.label_fish(f["id"], "species", species=demo.RED)
    dets = project.fish_detections(f["id"])
    pieces = {t: [d["id"] for d in dets if d["tracklet_id"] == t] for t in project.fish_tracklets(f["id"])}
    cut = next(t for t, ids in pieces.items() if len(ids) >= 2)
    whole = next(t for t in pieces if t != cut)
    ticked = pieces[whole] + pieces[cut][-1:]  # one whole tracked piece, and part of another
    new_id = project.move_to_new_fish(f["id"], ticked)
    assert sorted(d["id"] for d in project.fish_detections(new_id)) == sorted(ticked)
    assert len(project.fish_tracklets(new_id)) == 2  # the part became a piece of its own
    assert len(project.fish_detections(f["id"])) == len(dets) - len(ticked)
    assert project.fish(f["id"])["status"] == "species"  # the card keeps its answer
    assert project.fish(new_id)["status"] == "pending"
    log = project.corrections()[-1]
    assert log["action"] == "move_out" and json.loads(log["details"]) == {"detections": sorted(ticked), "new_card": new_id}
    with pytest.raises(ValueError):
        project.move_to_new_fish(f["id"], [])  # nothing ticked
    with pytest.raises(ValueError):
        project.move_to_new_fish(f["id"], [d["id"] for d in project.fish_detections(f["id"])])  # all of it


def test_splits_and_merges_are_logged(project):
    f = multi_tracklet_fish(project)
    new_id = project.split_fish(f["id"], project.fish_detections(f["id"])[-1]["t"])
    project.merge_fish(f["id"], new_id)
    assert [c["action"] for c in project.corrections()] == ["split", "merge"]


def test_merge_puts_card_back_to_pending(project):
    a, b = project.all_fish()[:2]
    project.label_fish(a["id"], "species", species=demo.RED)
    n = len(project.fish_detections(a["id"])) + len(project.fish_detections(b["id"]))
    project.merge_fish(a["id"], b["id"])
    assert project.fish(b["id"]) is None
    assert len(project.fish_detections(a["id"])) == n
    assert project.fish(a["id"])["status"] == "pending"


@pytest.fixture
def client(project):
    return TestClient(create_app(project.root))


def test_api_review_flow(client, project):
    summary = client.get("/api/summary").json()
    assert summary["total"] == len(project.all_fish()) and summary["reviewed"] == 0
    assert summary["dropdowns"]["species"][0] == demo.RED
    assert summary["tiny_fish_px"] == 40  # the review page hides fish smaller than this

    cards = client.get("/api/fish").json()
    detail = client.get(f"/api/fish/{cards[0]['id']}").json()
    assert detail["status"] == "pending" and len(detail["guesses"]) == 3
    assert detail["filmstrip"] and detail["best"]

    assert client.get(f"/crop/{detail['best']['det_id']}").headers["content-type"] == "image/jpeg"
    assert client.get(f"/frame/{detail['best']['frame_id']}").status_code == 200

    bad = client.post(f"/api/fish/{detail['id']}/label", json={"status": "species", "species": "Nemo"})
    assert bad.status_code == 400
    ok = client.post(
        f"/api/fish/{detail['id']}/label",
        json={"status": "species", "species": demo.RED, "behavior": "feeding"},
    )
    assert ok.status_code == 200
    assert client.get("/api/summary").json()["reviewed"] == 1

    res = client.post("/api/export")
    assert res.status_code == 200 and res.json()["path"].endswith(".xlsx")


def test_api_accepts_names_from_the_extra_list(client, project):
    update_extra_species(project, ["batfish"])
    assert client.get("/api/summary").json()["extra_species"] == ["batfish"]
    fid = project.all_fish()[0]["id"]
    assert client.post(f"/api/fish/{fid}/label", json={"status": "species", "species": "batfish"}).status_code == 200
    assert client.post(f"/api/fish/{fid}/label", json={"status": "species", "species": "Nemo"}).status_code == 400


def test_api_card_list_has_guesses_and_multiple_fish_works(client, project):
    cards = client.get("/api/fish").json()
    assert all(len(c["guesses"]) == 3 for c in cards)  # for the template-species filter
    f = multi_tracklet_fish(project)
    res = client.post(f"/api/fish/{f['id']}/unmerge", json={})
    assert res.status_code == 200 and res.json()["new_ids"]
    assert all(json.loads(project.fish(i)["guesses"]) for i in res.json()["new_ids"])  # recomputed
    single = next(c for c in client.get("/api/fish").json() if len(project.fish_tracklets(c["id"])) == 1)
    assert client.post(f"/api/fish/{single['id']}/unmerge", json={}).status_code == 400


def test_api_card_list_has_each_fish_at_its_largest(client, project):
    for card in client.get("/api/fish").json():
        dets = project.fish_detections(card["id"])
        largest = max(max((d["x2"] - d["x1"]) * d["width"], (d["y2"] - d["y1"]) * d["height"]) for d in dets)
        assert card["max_px"] == round(largest, 1) > 0  # for the "fish ≥ 40 px" filter


def test_api_moves_ticked_sightings(client, project, monkeypatch):
    f = multi_tracklet_fish(project)
    all_ids = [d["id"] for d in project.fish_detections(f["id"])]
    monkeypatch.setattr(server, "FILMSTRIP_MAX", 2)  # a long card shows every n-th sighting
    strip = client.get(f"/api/fish/{f['id']}").json()["filmstrip"]
    assert [i for item in strip for i in item["covers"]] == all_ids  # each stands for its stretch
    res = client.post(f"/api/fish/{f['id']}/move", json={"det_ids": strip[-1]["covers"]})
    assert res.status_code == 200
    assert [d["id"] for d in project.fish_detections(res.json()["new_id"])] == strip[-1]["covers"]
    assert json.loads(project.fish(res.json()["new_id"])["guesses"])  # recomputed
    assert client.post(f"/api/fish/{f['id']}/move", json={"det_ids": []}).status_code == 400


def test_api_split_and_merge(client, project):
    f = multi_tracklet_fish(project)
    detail = client.get(f"/api/fish/{f['id']}").json()
    cut = detail["filmstrip"][-1]["t"]
    res = client.post(f"/api/fish/{f['id']}/split", json={"from_t": cut})
    assert res.status_code == 200
    new_id = res.json()["new_id"]
    assert json.loads(project.fish(new_id)["guesses"])  # recomputed from cached embeddings
    assert client.post(f"/api/fish/{f['id']}/merge", json={"other_id": new_id}).status_code == 200
    assert client.get(f"/api/fish/{new_id}").status_code == 404
    assert client.post(f"/api/fish/{f['id']}/split", json={"from_t": -1}).status_code == 400


def test_index_page_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "BRUV Fish ID" in res.text
    assert client.get("/static/app.js").status_code == 200
