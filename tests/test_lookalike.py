"""Look-alike cards answered once instead of over and over (lookalike.py; the review page's
R and L keys)."""

import numpy as np
from fastapi.testclient import TestClient

from bruv_app.lookalike import Looks, answered_like, groups, unanswered_like
from bruv_app.server import create_app

from .conftest import ground_truth_species


def _looks(vectors, status, answer):
    emb = np.array([v / np.linalg.norm(v) for v in vectors], np.float32)
    ids = np.arange(1, len(vectors) + 1)
    return Looks(ids, emb, dict(zip(ids.tolist(), status)), dict(zip(ids.tolist(), answer)))


def test_an_answer_is_suggested_only_from_a_card_alike_enough():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    looks = _looks([a, a + 0.1 * b, b, a + 0.05 * b], ["species", "pending", "pending", "skipped"],
                   ["unicornfish", None, None, None])
    like = answered_like(looks, 2)
    assert like["card"] == 1 and like["species"] == "unicornfish" and like["similarity"] > 0.99
    assert answered_like(looks, 3) is None  # nothing answered looks like it
    assert answered_like(looks, 1) is None  # a card isn't its own look-alike
    assert [f for f, _ in unanswered_like(looks, 1)] == [4, 2]  # most alike first, unanswered only
    # Groups: each card joins the first leading card it looks like, in the order given.
    assert groups(looks, [1, 3, 2, 4]) == {1: 1, 3: 3, 2: 1, 4: 1}


def _species(project, fish_id):
    votes = [ground_truth_species(d) for d in project.fish_detections(fish_id)]
    votes = [v for v in votes if v]
    return max(set(votes), key=votes.count) if votes else None


def test_look_alikes_are_answered_together_and_can_be_undone(project):
    client = TestClient(create_app(project.root))
    cards = client.get("/api/fish").json()
    assert all("group" in c and c["like"] is None for c in cards)  # nothing answered yet
    assert client.get("/api/summary").json()["still_hidden"] == 0  # every synthetic fish moves
    source = next(c["id"] for c in cards if client.get(f"/api/fish/{c['id']}/lookalikes").json()["cards"])
    species = _species(project, source)
    alike = client.get(f"/api/fish/{source}/lookalikes").json()["cards"]
    assert alike and all(_species(project, c["id"]) == species for c in alike)  # same colour, same fish kind
    # One picture per piece, and whether every piece looks like the source.
    assert all(1 <= len(c["pieces"]) <= 3 and c["n_pieces"] >= len(c["pieces"]) and "mixed" in c for c in alike)
    answered = client.post(f"/api/fish/{source}/label", json={"status": "species", "species": species, "seconds": 7.5})
    assert answered.status_code == 200
    details = project.answer_details(source)
    assert details["seconds"] == 7.5 and details["suggestion"]["like"] is None and details["matched"] is None

    # Each look-alike now shows that answer, for R, and joins the quick-confirm queue.
    reference = client.get(f"/api/fish/{source}/lookalikes").json()["source"]
    assert reference["id"] == source and reference["species"] == species and reference["det"]
    other = alike[0]["id"]
    like = client.get(f"/api/fish/{other}").json()["answered_like"]
    assert like["card"] == source and like["species"] == species and like["similarity"] >= 0.85 and like["det"]
    listed = {c["id"]: c for c in client.get("/api/fish").json()}
    assert listed[other]["like"]["card"] == source and listed[other]["like"]["sim"] >= 0.85

    # Answering a look-alike from the queue keeps what was offered and that it was taken.
    assert client.post(f"/api/fish/{other}/label",
                       json={"status": "species", "species": species, "queue": True, "seconds": 1.2}).status_code == 200
    kept = project.answer_details(other)
    assert kept["queue"] == "quick" and kept["matched"] == "same" and kept["suggestion"]["like"]["card"] == source
    client.post(f"/api/fish/{other}/label", json={"status": "pending"})

    # L: the ticked ones get the same answer together; a card answered meanwhile is left alone.
    ids = [c["id"] for c in alike if not c["mixed"]]
    assert ids
    client.post(f"/api/fish/{ids[-1]}/label", json={"status": "skipped"})
    client.post(f"/api/fish/{ids[-1]}/label", json={"status": "unknown"})
    res = client.post("/api/lookalikes/answer", json={"source": source, "ids": ids, "status": "species",
                                                      "species": species, "seconds": 10.0}).json()
    assert set(res["answered"]) | set(res["left"]) == set(ids) and ids[-1] in res["left"]
    for fid in res["answered"]:
        f = project.fish(fid)
        assert (f["status"], f["species"]) == ("species", species)
        assert project.answer_details(fid)["like"] == source
        assert project.answer_family(fid) == species  # the demo's template has no groups: a name is its family
    # Undo puts the batch back to pending; the card answered on its own keeps its answer.
    undone = client.post("/api/batch/undo", json={"batch": res["batch"]}).json()["undone"]
    assert set(undone) == set(res["answered"])
    assert all(project.fish(fid)["status"] == "pending" for fid in undone)
    assert project.fish(ids[-1])["status"] == "unknown"


def test_a_look_alike_answer_must_be_an_answer(project):
    client = TestClient(create_app(project.root))
    source = project.all_fish()[0]["id"]
    assert client.post("/api/lookalikes/answer", json={"source": source, "ids": [], "status": "skipped"}).status_code == 400
    bad = client.post("/api/lookalikes/answer", json={"source": source, "ids": [], "status": "species", "species": "x"})
    assert bad.status_code == 400
