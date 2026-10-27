"""hosted/core.js against the review server (docs/SPEC.md §16). Answers, fixes and follows
made through the browser's code, then pulled into the project, must give the same card
list, details, look-alikes, counts and follows as the server; and opening the page again
must replay to the same state. Needs Node.js (skipped without it)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bruv_app
from bruv_app.detect import Detection
from bruv_app.pipeline import refresh_fish_guesses
from bruv_app.publish import publish, published_cards
from bruv_app.pull import pull
from bruv_app.server import create_app

CORE = Path(bruv_app.__file__).parent / "hosted" / "public" / "hosted" / "core.js"
SCRIPT = Path(__file__).with_name("hosted_parity.js")


def _node() -> str | None:
    found = shutil.which("node")
    default = Path("C:/Program Files/nodejs/node.exe")  # a shell opened before Node was installed
    return found or (str(default) if default.exists() else None)


NODE = _node()
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")


def _differences(py, js, path="", out=None) -> list[str]:
    """Where two answers differ: numbers within rounding, everything else exactly."""
    out = [] if out is None else out
    if len(out) > 40:
        return out
    if isinstance(py, (int, float)) and isinstance(js, (int, float)) and not isinstance(py, bool):
        if abs(py - js) > 0.002 + 0.002 * abs(js):
            out.append(f"{path}: server {py}, browser {js}")
    elif isinstance(py, dict) and isinstance(js, dict):
        for k in py.keys() | js.keys():
            if k not in js:
                out.append(f"{path}.{k}: only on the server")
            elif k not in py:
                out.append(f"{path}.{k}: only in the browser")
            else:
                _differences(py[k], js[k], f"{path}.{k}", out)
    elif isinstance(py, list) and isinstance(js, list):
        if len(py) != len(js):
            out.append(f"{path}: {len(py)} on the server, {len(js)} in the browser")
        for i, (a, b) in enumerate(zip(py, js)):
            _differences(a, b, f"{path}[{i}]", out)
    elif py != js:
        out.append(f"{path}: server {json.dumps(py)[:120]}, browser {json.dumps(js)[:120]}")
    return out


def test_the_browser_gives_the_servers_answers(project, tmp_path):
    # Publish every card (none tiny here), so both sides hold the same cards and numbers.
    settings = project.load_settings()
    settings.tiny_fish_px = 0
    project.save_settings(settings)
    # One card made of two different fish, as a tracker's wrong join leaves it: it has pieces
    # to break up, and warnings to confirm before answering.
    keep, other = [f["id"] for f in project.all_fish()][:2]
    project.merge_fish(keep, other)
    refresh_fish_guesses(project, keep, settings)
    # A faint box (under det_threshold, so on no card) beside a fish in every frame, as
    # footage processed since faint boxes were kept has: a follow clicks one.
    for frame_id, x1, y1, x2, y2 in project.db.execute(
        "SELECT frame_id, x1, y1, x2, y2 FROM detections WHERE tracklet_id IS NOT NULL GROUP BY frame_id"
    ).fetchall():
        dx = 0.12 if x2 + 0.12 <= 1 else -0.12
        project.add_detections(frame_id, [Detection(x1 + dx, y1, x2 + dx, y2, settings.det_threshold / 1.5, None)], [0.0])
    project.commit()
    site = tmp_path / "site"
    publish(project.root, site, "V1", follow=True)
    assert len(published_cards(project)) == len(project.all_fish())

    result = tmp_path / "browser.json"
    run = subprocess.run([NODE, str(SCRIPT), str(CORE), str(site / "public" / "videos" / "V1"), "V1", str(result)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    js = json.loads(result.read_text(encoding="utf-8"))
    steps = js["steps"]
    assert {"label", "label confirmed", "lookalikes", "multiple fish", "split", "split again", "move"} <= set(steps)
    assert {"follow done", "follow undo", "follow faint", "follow again", "follow taken", "follow delete",
            "follow short"} <= set(steps)
    assert steps["label"]["status"] == 409  # the joined card's warnings, then confirmed
    turned_down = {"label": 409, "undo without a batch": 422, "join": 400, "unknown card": 404,
                   "follow bad state": 400, "follow box elsewhere": 400, "follow no frame": 422, "follow unknown": 404,
                   "follow unknown card": 404}
    for name, res in steps.items():
        assert res["status"] == turned_down.get(name, 200), (name, res)
    assert js["replayed_same"], "the page opened again doesn't show what was done"

    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({
        "format": 1, "video": "V1", "volunteers": [{"id": 1, "name": "Ana"}],
        "events": [{"id": i, "volunteer_id": 1, "event": e} for i, e in enumerate(js["events"])],
    }), encoding="utf-8")
    report = pull(project.root, answers, "Ana")
    assert report.new_cards == {} and report.scope_mismatch == [] and report.follows == 3

    client = TestClient(create_app(project.root))
    live = js["live"]
    py = {"list": client.get("/api/fish").json(), "details": {}, "looks": {}}
    for card in py["list"]:
        py["details"][str(card["id"])] = client.get(f"/api/fish/{card['id']}").json()
        py["looks"][str(card["id"])] = client.get(f"/api/fish/{card['id']}/lookalikes").json()
    diffs = _differences(py, {k: live[k] for k in ("list", "details", "looks")})
    assert diffs == [], "\n".join(diffs)
    assert client.get("/api/summary").json()["counts"] == live["summary"]["counts"]

    # The follows, by name: ids here are this project's own. One ended "done", as on the site.
    follows = client.get("/api/follows").json()
    assert [f["name"] for f in follows] == [f["name"] for f in live["follows"]] == ["F1-v1", "F2-v1", "F4-v1"]
    py_views = {f["name"]: client.get(f"/api/follows/{f['id']}").json() for f in follows}
    for summaries in (follows, live["follows"], [v["follow"] for v in py_views.values()],
                      [v["follow"] for v in live["followViews"].values()]):
        for f in summaries:
            f["id"] = None
    diffs = _differences({"list": follows, "views": py_views}, {"list": live["follows"], "views": live["followViews"]})
    assert diffs == [], "\n".join(diffs)
    assert [f["done"] for f in follows] == [True, False, False]
    # The faint box clicked in the first follow: on no card, counted apart, on both sides.
    assert py_views["F1-v1"]["follow"]["faint"] == 1
    assert any(b["faint"] and b["card"] is None for b in py_views["F2-v1"]["boxes"])
