"""The hosted review site (docs/SPEC.md §16): publishing a project as static files, the
volunteer list, the site's /sync API (run here as the local copy of the site), and pulling a
volunteer's answers and fixes back into a project."""

import json
import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bruv_app import follow
from bruv_app.publish import FOLLOW_COLS, FOLLOW_STRIP, FRAMES_PER_CARD, publish, published_cards, site_report, write_site
from bruv_app.pull import pull
from bruv_app.site import PASS_LETTERS, create_dev_app, export, read_volunteers, volunteers_sql, write_volunteers
from bruv_app.store import Project
from bruv_app.trials import open_trial


@pytest.fixture
def site(project, tmp_path):
    out = tmp_path / "site"
    report = publish(project.root, out, "V1")
    write_site(out)
    return out, report


def _cards(out, key="V1"):
    return json.loads((out / "public" / "videos" / key / "cards.json").read_text(encoding="utf-8"))["cards"]


def test_publish_writes_each_card_and_its_pictures(project, site):
    out, report = site
    folder = out / "public" / "videos" / "V1"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    cards = _cards(out)
    ids = published_cards(project)
    assert report["cards"] == len(ids) == len(cards) > 0
    assert sorted(c["id"] for c in cards) == sorted(ids)
    for card in cards:
        sheet = Image.open(folder / card["sheet"])
        rects = list(card["crops"].values()) + list(card["frames"].values())
        assert all(x >= 0 and y >= 0 and x + w <= sheet.width and y + h <= sheet.height for x, y, w, h in rects)
        assert len(card["frames"]) <= FRAMES_PER_CARD
        # Every close-up the page shows when the card opens is on the card's sheet.
        detail = card["detail"]
        shown = [s["det_id"] for s in detail["filmstrip"]] + [p["best_det"] for p in detail["pieces"]]
        if detail["best"]:
            shown.append(detail["best"]["det_id"])
        assert {str(d) for d in shown} <= set(card["crops"])
    # The descriptions: names, crops, pieces and card looks, float16, in that order.
    counts = manifest["emb"]
    assert (folder / "emb.bin").stat().st_size == 2 * manifest["emb"]["dim"] * sum(
        counts[k] for k in ("labels", "crops", "pieces", "looks"))


def test_the_published_site_holds_no_local_paths(project, site):
    out, _ = site
    for path in (out / "public").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert str(project.root) not in text and ":\\\\" not in text, path.name


def test_publish_refuses_a_bad_key_or_an_empty_folder(project, tmp_path):
    with pytest.raises(ValueError, match="letters, digits"):
        publish(project.root, tmp_path / "site", "A 1")
    with pytest.raises(ValueError, match="not a project folder"):
        publish(tmp_path / "nothing", tmp_path / "site", "A1")
    with pytest.raises(ValueError, match="no cards"):
        Project(tmp_path / "empty").db.close()
        publish(tmp_path / "empty", tmp_path / "site", "A1")


def test_the_site_page_is_the_review_page_plus_the_runtime(site):
    out, _ = site
    public = out / "public"
    page = (public / "index.html").read_text(encoding="utf-8")
    head = page[: page.index("<meta")]
    assert "/hosted/core.js?v=" in head and "/hosted/runtime.js?v=" in head  # before the page's own script
    assert "/static/app.js?v=" in page
    assert (public / "hosted" / "core.js").exists() and (public / "start.html").exists()
    assert not (public / "sw.js").exists()
    assert json.loads((public / "_routes.json").read_text())["include"] == ["/sync/*", "/api/*"]
    assert (out / "functions" / "sync" / "[[path]].js").exists() and (out / "schema.sql").exists()
    assert (public / "videos" / "V1" / "manifest.json").exists()
    # The follow page too, served at /follow as the local server serves it.
    follow_page = (public / "follow.html").read_text(encoding="utf-8")
    assert "/hosted/runtime.js?v=" in follow_page[: follow_page.index("<meta")] and "/static/follow.js?v=" in follow_page
    report = site_report(out)
    assert report["files"] < report["limit"]


def test_publish_follow_writes_every_frame_and_box(project, tmp_path, site):
    out, _ = site
    plain = json.loads((out / "public" / "videos" / "V1" / "manifest.json").read_text(encoding="utf-8"))
    assert plain["follow"] is None and not (out / "public" / "videos" / "V1" / "follow").exists()

    publish(project.root, out, "V2", follow=True)
    folder = out / "public" / "videos" / "V2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    f = manifest["follow"]
    frames = json.loads((folder / f["frames"]).read_text(encoding="utf-8"))
    ids = [r["id"] for r in project.db.execute("SELECT id FROM frames ORDER BY seq")]
    assert frames["ids"] == ids and len(frames["t"]) == len(ids)
    strips = -(-len(ids) // FOLLOW_STRIP)
    assert f["files"] == strips and len(list((folder / "follow").glob("*.webp"))) == strips
    with Image.open(folder / "follow" / "0.webp") as im:
        assert im.width == FOLLOW_COLS * f["w"] and im.height % f["h"] == 0
    boxes = json.loads((folder / f["boxes"]).read_text(encoding="utf-8"))["boxes"]
    assert len(boxes) == len(ids)
    assert sum(len(b) for b in boxes) == project.db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    first = next(b for b in boxes if b)
    assert len(first[0]) == 8 and [r[5] for r in first] == sorted((r[5] for r in first), reverse=True)


def test_pull_replays_follows_made_on_the_site(project, tmp_path):
    """The site's follow events, replayed: each follow keeps its name and start, its marks and
    their times, and ends "done" exactly when it did on the site."""
    project.commit()
    copy = Project(shutil.copytree(project.root, tmp_path / "site-side"))  # the site's side
    card = next(c["id"] for c in copy.all_fish() if len(copy.fish_detections(c["id"])) >= 3)
    dets = {d["frame_id"]: d["id"] for d in copy.fish_detections(card)}

    def started(site_id, gone):
        fid = follow.start(copy, card, gone)
        det = copy.db.execute("SELECT start_det FROM follows WHERE id=?", (fid,)).fetchone()[0]
        frame = copy.db.execute("SELECT frame_id FROM detections WHERE id=?", (det,)).fetchone()[0]
        return fid, {"kind": "follow_start", "follow": site_id, "card": card, "det": det, "frame": frame, "gone_s": gone,
                     "name": f"F{site_id}-v2"}

    def marked(fid, site_id, step):
        state, det = ("box", dets[step.frame_id]) if step.frame_id in dets else ("gone", None)
        follow.mark(copy, fid, step.frame_id, state, det_id=det)
        return {"kind": "follow_mark", "follow": site_id, "frame": step.frame_id, "state": state, "det": det, "x": None, "y": None}

    first, e = started(1, 1.0)
    events = [e]
    while (step := follow.next_step(copy, first)).phase != "done":
        events.append(marked(first, 1, step))
    second, e = started(2, 5.0)
    events += [e, marked(second, 2, follow.next_step(copy, second)), {"kind": "follow_undo", "follow": 2}]
    events.append(marked(second, 2, follow.next_step(copy, second)))
    third, e = started(3, 5.0)
    events += [e, {"kind": "follow_delete", "follow": 3}]

    report = pull(project.root, _export(tmp_path, events), "Ana")
    assert report.follows == 2
    master = Project(project.root)
    rows = {r["name"]: r for r in master.db.execute("SELECT * FROM follows")}
    assert set(rows) == {"F1-v2", "F2-v2"}
    done = {name: follow.summary(master, r["id"]) for name, r in rows.items()}
    assert done["F1-v2"].done and not done["F2-v2"].done
    assert done["F1-v2"].boxes == follow.summary(copy, first).boxes and done["F1-v2"].gone == follow.summary(copy, first).gone
    assert done["F2-v2"].boxes + done["F2-v2"].gone == 2  # the start, and the mark made again after the undo
    assert rows["F1-v2"]["at"] == "2026-11-01T10:00:00+00:00" and rows["F1-v2"]["start_det"] == events[0]["det"]
    assert master.runs("pull")[-1] is not None


def _volunteer_csv(tmp_path):
    path = tmp_path / "volunteers.csv"
    path.write_text(
        "name,email,pass,role,videos,calibration\n"
        "Vinci,vinci@example.org,BRUV-ADMN-TEST,admin,V1,\n"
        'Ana,Ana@Example.org,,volunteer,V1 V2,"V3=12,15 V4"\n'  # commas in a quoted cell (as Excel saves it)
        "Bea,,,volunteer,,V3=12;15\n",  # or semicolons, unquoted
        encoding="utf-8",
    )
    return path


def test_volunteers_get_passes_and_assignments(tmp_path):
    people = read_volunteers(_volunteer_csv(tmp_path))
    ana, bea = people[1], people[2]
    assert people[0].passcode == "BRUV-ADMN-TEST" and people[0].role == "admin"
    assert ana.passcode.startswith("BRUV-") and all(c in PASS_LETTERS + "-" for c in ana.passcode[5:])
    assert ana.passcode != bea.passcode
    assert ana.own == ["V1", "V2"] and ana.calibration == {"V3": [12, 15], "V4": None}
    assert bea.own == [] and bea.calibration == {"V3": [12, 15]} and bea.email is None
    write_volunteers(people, tmp_path / "out.csv")
    assert [v.passcode for v in read_volunteers(tmp_path / "out.csv")] == [v.passcode for v in people]


def test_a_volunteer_keeps_their_answers_when_the_list_changes(tmp_path):
    people = read_volunteers(_volunteer_csv(tmp_path))
    db = sqlite3.connect(tmp_path / "site.db")
    from bruv_app.site import HOSTED_DIR

    db.executescript((HOSTED_DIR / "schema.sql").read_text(encoding="utf-8"))
    db.executescript(volunteers_sql(people))
    ana = db.execute("SELECT id FROM volunteers WHERE name='Ana'").fetchone()[0]
    # The list again, reordered, with the coordinator left off: Ana is still Ana.
    db.executescript(volunteers_sql([people[1]]))
    assert db.execute("SELECT id FROM volunteers WHERE name='Ana'").fetchone()[0] == ana
    assert db.execute("SELECT active FROM volunteers WHERE name='Vinci'").fetchone()[0] == 0
    rows = db.execute("SELECT video, role, cards FROM assignments WHERE volunteer_id=? ORDER BY video", (ana,)).fetchall()
    assert rows == [("V1", "own", None), ("V2", "own", None), ("V3", "calibration", "[12, 15]"), ("V4", "calibration", None)]


@pytest.fixture
def client(site, tmp_path):
    out, _ = site
    people = read_volunteers(_volunteer_csv(tmp_path))
    from bruv_app.site import load_volunteers_into

    load_volunteers_into(tmp_path / "site.db", volunteers_sql(people))
    return TestClient(create_dev_app(out, tmp_path / "site.db")), people


def test_sign_in_by_pass_or_email(client):
    c, people = client
    assert c.get("/sync/me").status_code == 401
    assert c.get("/api/fish").status_code == 409  # the page opened without signing in
    assert c.post("/sync/login", json={"who": "BRUV-NOPE-NOPE"}).status_code == 401
    me = c.post("/sync/login", json={"who": " " + people[1].passcode.lower() + " "}).json()
    assert me["name"] == "Ana" and [v["video"] for v in me["videos"]] == ["V1", "V2", "V3", "V4"]
    assert me["videos"][2] == {"video": "V3", "role": "calibration", "cards": [12, 15]}
    c.post("/sync/logout")
    assert c.post("/sync/login", json={"who": "ana@example.ORG"}).json()["name"] == "Ana"
    assert c.get("/sync/me").json()["name"] == "Ana"


def test_events_are_kept_per_volunteer_and_video(client):
    c, people = client
    c.post("/sync/login", json={"who": people[1].passcode})
    assert c.post("/sync/events", json={"video": "V9", "events": [{"kind": "label"}]}).status_code == 403
    assert c.post("/sync/events", json={"video": "V1", "events": []}).status_code == 400
    sent = [{"kind": "label", "card": 1, "status": "unknown"}, {"kind": "split", "card": 2, "from_t": 3.0, "new_card": 9}]
    assert c.post("/sync/events", json={"video": "V1", "events": sent, "app": "test"}).json()["saved"] == 2
    got = c.get("/sync/events", params={"video": "V1"}).json()["events"]
    assert [e["kind"] for e in got] == ["label", "split"] and all(e["at"] for e in got)  # the site's clock
    assert c.post("/sync/feedback", json={"video": "V1", "rating": 4, "text": "fine"}).json() == {"ok": True}
    assert c.get("/sync/admin/progress").status_code == 403
    # The coordinator sees someone else's work only through progress and export.
    c.post("/sync/logout")
    c.post("/sync/login", json={"who": "BRUV-ADMN-TEST"})
    assert c.get("/sync/events", params={"video": "V1"}).json()["events"] == []
    rows = {(r["name"], r["video"]): r for r in c.get("/sync/admin/progress").json()["rows"]}
    assert rows[("Ana", "V1")]["answers"] == 1 and rows[("Ana", "V1")]["fixes"] == 1
    dump = c.get("/sync/admin/export", params={"video": "V1"}).json()
    assert [e["event"]["kind"] for e in dump["events"]] == ["label", "split"]
    assert dump["feedback"][0]["text"] == "fine"


def _export(tmp_path, events, name="Ana"):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({
        "format": 1, "video": "V1", "volunteers": [{"id": 1, "name": "Vinci"}, {"id": 2, "name": name}],
        "events": [{"id": i, "volunteer_id": 2, "event": {"at": f"2026-11-01T10:00:{i:02d}+00:00", **e}}
                   for i, e in enumerate(events)],
    }), encoding="utf-8")
    return path


def _two_piece_card(project):
    for f in project.all_fish():
        if len(project.fish_tracklets(f["id"])) >= 2:
            return f["id"]
    ids = [f["id"] for f in project.all_fish()]
    project.merge_fish(ids[0], ids[1])
    return ids[0]


def test_pull_replays_a_volunteers_answers_and_fixes(project, tmp_path):
    mixed = _two_piece_card(project)
    fish = [f["id"] for f in project.all_fish()]
    times = {f: sorted({d["t"] for d in project.fish_detections(f)}) for f in fish if f != mixed}
    second = next(f for f, ts in times.items() if len(ts) >= 2)  # one that can be split
    first = next(f for f in times if f != second)
    cut = times[second][1]
    top = max(fish)
    events = [
        {"kind": "label", "card": first, "status": "species", "species": "Lutjanus campechanus", "family": "Lutjanidae",
         "type": "individual", "details": {"seconds": 3.5}, "scope": [d["id"] for d in project.fish_detections(first)]},
        # The site numbers new cards after the cards it was given; here they get this project's next numbers.
        {"kind": "split", "card": second, "from_t": cut, "new_card": top + 40},
        {"kind": "label", "card": top + 40, "status": "not_fish"},
        {"kind": "unmerge", "card": mixed, "tracklets": project.fish_tracklets(mixed), "new_cards": [top + 41]},
        {"kind": "label", "card": second, "status": "unknown", "details": {"batch": "like-1"}},
        {"kind": "undo", "batch": "like-1", "cards": [second]},
        {"kind": "suggestions", "rows": [[top + 40, first, 0.91, "skipped", "like-1"]]},
    ]
    report = pull(project.root, _export(tmp_path, events), "Ana")
    project = Project(project.root)
    assert report.volunteer == "Ana" and report.answers == 3 and report.fixes == 2 and report.undone == 1
    split_card = report.new_cards[top + 40]
    assert split_card not in fish and report.new_cards[top + 41] not in fish
    assert all(d["t"] >= cut for d in project.fish_detections(split_card))
    assert project.fish(split_card)["status"] == "not_fish"
    assert project.fish(first)["status"] == "species" and project.fish(second)["status"] == "pending"
    review = project.db.execute("SELECT * FROM reviews WHERE fish_id=? ORDER BY id DESC", (first,)).fetchone()
    assert review["reviewer"] == "Ana" and review["at"] == "2026-11-01T10:00:00+00:00" and review["family"] == "Lutjanidae"
    assert json.loads(review["details"])["site"] == {"card": first, "at": "2026-11-01T10:00:00+00:00"}
    actions = [r["action"] for r in project.corrections()]
    assert actions[-2:] == ["split", "multiple_fish"]
    logged = project.db.execute("SELECT at FROM suggestion_log WHERE batch='like-1'").fetchall()
    assert [r[0] for r in logged] == ["2026-11-01T10:00:06+00:00"]  # when it was seen on the site
    run = project.runs("pull")[-1]
    assert run is not None and report.scope_mismatch == []


def test_pull_notes_answers_whose_sightings_differ(project, tmp_path):
    card = project.all_fish()[0]["id"]
    events = [{"kind": "label", "card": card, "status": "unknown", "scope": [-1]}]
    assert pull(project.root, _export(tmp_path, events), "Ana").scope_mismatch == [card]
    with pytest.raises(ValueError, match="No volunteer"):
        pull(project.root, _export(tmp_path, events), "Cleo")


def test_answers_record_who_gave_them_and_when(project):
    card = project.all_fish()[0]["id"]
    project.label_fish(card, "unknown", reviewer="Ana", at="2026-11-01T09:00:00+00:00")
    row = project.db.execute("SELECT reviewer, at FROM reviews WHERE fish_id=? ORDER BY id DESC", (card,)).fetchone()
    assert tuple(row) == ("Ana", "2026-11-01T09:00:00+00:00")
    assert project.fish(card)["reviewed_at"] == "2026-11-01T09:00:00+00:00"
    project.label_fish(card, "not_fish")  # on this PC: nobody named
    assert project.db.execute("SELECT reviewer FROM reviews WHERE fish_id=? ORDER BY id DESC", (card,)).fetchone()[0] is None


def test_export_reads_the_sites_database(tmp_path):
    from bruv_app.site import HOSTED_DIR

    db = sqlite3.connect(tmp_path / "x.db")
    db.executescript((HOSTED_DIR / "schema.sql").read_text(encoding="utf-8"))
    db.execute("INSERT INTO volunteers (name, pass) VALUES ('Ana', 'P1')")
    db.execute("INSERT INTO events (volunteer_id, video, at, kind, body) VALUES (1, 'V1', 'now', 'label', '{\"kind\": \"label\"}')")
    db.execute("INSERT INTO events (volunteer_id, video, at, kind, body) VALUES (1, 'V2', 'now', 'label', '{}')")
    data = export(db, "V1")
    assert data["format"] == 1 and data["volunteers"][0]["name"] == "Ana"
    assert [e["event"] for e in data["events"]] == [{"kind": "label"}]


def test_pull_refuses_another_videos_answers(project, tmp_path):
    publish(project.root, tmp_path / "site", "V2")
    card = project.all_fish()[0]["id"]
    with pytest.raises(ValueError, match="published as V2"):
        pull(project.root, _export(tmp_path, [{"kind": "label", "card": card, "status": "unknown"}]), "Ana")
    assert Project(project.root).fish(card)["status"] == "pending"


def test_a_rule_trial_is_published_and_pulled_on_its_own(project, tmp_path):
    # A trial is another database beside the project's (trials.py): its cards go online under
    # their own video name, and their answers come back into the trial, not the project.
    project.commit()
    (project.root / "trials").mkdir()
    shutil.copy(project.root / "project.db", project.root / "trials" / "t1.db")
    open_trial(project, "t1").set_meta("trial", {"name": "t1"})
    publish(project.root, tmp_path / "site", "V1", trial="t1")
    manifest = json.loads((tmp_path / "site" / "public" / "videos" / "V1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["project"].endswith("· trial: t1")
    assert project.get_meta("published") is None and open_trial(project, "t1").get_meta("published")["video"] == "V1"

    card = project.all_fish()[0]["id"]
    pull(project.root, _export(tmp_path, [{"kind": "label", "card": card, "status": "unknown"}]), "Ana", trial="t1")
    assert open_trial(project, "t1").fish(card)["status"] == "unknown"
    assert Project(project.root).fish(card)["status"] == "pending"
    with pytest.raises(ValueError, match="No trial called t2"):
        publish(project.root, tmp_path / "site", "V2", trial="t2")


def test_the_commands(project, tmp_path, capsys):
    from bruv_app.cli import main

    site = tmp_path / "site"
    main(["publish", "--project", str(project.root), "--video", "V1", "--site", str(site), "--follow"])
    assert "of the free plan's 20,000" in capsys.readouterr().out
    assert (site / "public" / "videos" / "V1" / "follow" / "frames.json").exists()
    main(["site", "volunteers", "--site", str(site), "--csv", str(_volunteer_csv(tmp_path))])
    assert "BRUV-ADMN-TEST" in (site / "volunteers.sql").read_text(encoding="utf-8")
    assert len(read_volunteers(tmp_path / "volunteers-with-passes.csv")) == 3
    card = project.all_fish()[0]["id"]
    main(["pull", "--project", str(project.root), "--answers",
          str(_export(tmp_path, [{"kind": "label", "card": card, "status": "unknown"}])), "--volunteer", "Ana"])
    assert "Ana: 1 events" in capsys.readouterr().out
