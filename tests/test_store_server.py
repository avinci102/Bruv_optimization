import json

import pytest
from fastapi.testclient import TestClient

from bruv_app import demo
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
