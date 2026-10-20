import hashlib
import json
import os
import sqlite3
from pathlib import Path

import openpyxl

from bruv_app import demo
from bruv_app.cli import main
from bruv_app.export import export_project
from bruv_app.pipeline import Pipeline
from bruv_app.provenance import deployment_id, fingerprint
from bruv_app.store import Project

from .conftest import label_like_perfect_human


def test_fingerprints_tell_files_and_recordings_apart(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 1000 + b"y")
    b.write_bytes(b"x" * 1000 + b"z")
    assert fingerprint(a) == "sha256:" + hashlib.sha256(a.read_bytes()).hexdigest()
    # Big files (videos) are fingerprinted from their size, start and end; the end counts.
    assert fingerprint(a, quick=True).startswith("sha256-quick:")
    assert fingerprint(a, quick=True) != fingerprint(b, quick=True)
    one = deployment_id([fingerprint(a), fingerprint(b)])
    assert one.startswith("dep-") and one == deployment_id([fingerprint(a), fingerprint(b)])
    assert one != deployment_id([fingerprint(b), fingerprint(a)])  # chapter order matters


def test_every_stage_run_is_recorded(project):
    runs = project.runs()
    assert [r["stage"] for r in runs] == ["detect", "track", "embed", "merge", "classify"]
    assert all(r["finished_at"] and r["code"] for r in runs)
    assert all(json.loads(r["settings"])["sample_fps"] == 1.0 for r in runs)
    detect = json.loads(runs[0]["info"])
    assert detect["detector"] == {"name": "color-blob"} and detect["sampler"] == 2
    assert detect["frames"] == project.frame_count() == 24 and detect["ffmpeg"]
    embed = json.loads(runs[2]["info"])
    assert embed["classifier"] == {"name": "demo-colour"} and embed["stand_in"] is False
    assert json.loads(runs[3]["info"])["cards"] == len(project.all_fish())

    # The footage, by fingerprint: the same recording always gets the same deployment id.
    videos = [Path(c["path"]) for c in project.chapters()]
    assert [v["fingerprint"] for v in project.get_meta("video_fingerprints")] == [fingerprint(p) for p in videos]
    assert detect["deployment"] == project.get_meta("deployment") == deployment_id([fingerprint(p) for p in videos])

    # Each frame names the source frame it came from (the synthetic video runs at 10 fps).
    frames = project.db.execute("SELECT local_t, src_index FROM frames ORDER BY seq").fetchall()
    assert [r["src_index"] for r in frames] == [round(r["local_t"] * 10) for r in frames]


def test_processing_ends_with_a_read_only_snapshot(project):
    snaps = sorted((project.root / "snapshots").glob("*.db"))
    assert len(snaps) == 1 and snaps[0].name.endswith("-processed.db")
    assert not os.access(snaps[0], os.W_OK)
    manifest = json.loads(snaps[0].with_suffix(".json").read_text())
    assert manifest["deployment"] == project.get_meta("deployment")
    assert manifest["database"] == fingerprint(snaps[0], quick=False)
    assert manifest["counts"]["fish"] == len(project.all_fish()) and len(manifest["runs"]) == 5
    frozen = sqlite3.connect(f"file:{snaps[0]}?mode=ro", uri=True)
    try:
        assert frozen.execute("SELECT COUNT(*) FROM fish").fetchone()[0] == len(project.all_fish())
    finally:
        frozen.close()
    # A run with nothing to do adds no run record and no snapshot.
    Pipeline(project, classifier=demo.DemoClassifier()).run()
    assert len(project.runs()) == 5 and len(list((project.root / "snapshots").glob("*.db"))) == 1


def test_the_machine_grouping_survives_review_edits(project):
    before = project.original_groups()
    tracked = project.db.execute("SELECT COUNT(*) FROM detections WHERE tracklet_id IS NOT NULL").fetchone()[0]
    assert len(before) == tracked
    card = next(f for f in project.all_fish() if len(project.fish_tracklets(f["id"])) > 1)
    project.unmerge_fish(card["id"])
    a, b = [f["id"] for f in project.all_fish()[:2]]
    project.merge_fish(a, b)
    live = dict(project.db.execute("SELECT d.id, t.fish_id FROM detections d JOIN tracklets t ON t.id=d.tracklet_id"))
    assert any(live[d] != fish for d, (_, fish) in before.items())  # the cards changed...
    assert project.original_groups() == before  # ...the record of how they were built didn't


def test_projects_from_older_versions_get_source_frame_numbers(project):
    project.db.execute("ALTER TABLE frames DROP COLUMN src_index")  # as an older version made it
    project.db.commit()
    project.db.close()
    old = Project(project.root)
    assert all(r[0] is None for r in old.db.execute("SELECT src_index FROM frames"))
    assert old.last_frame().src_index is None
    assert old.fill_source_indices() == old.frame_count()
    assert old.get_meta("src_index_inferred") is True  # worked out, not recorded
    rows = old.db.execute("SELECT local_t, src_index FROM frames").fetchall()
    assert all(r["src_index"] == round(r["local_t"] * 10) for r in rows)
    assert old.fill_source_indices() == 0


def test_export_says_where_the_numbers_came_from(project, tmp_path):
    label_like_perfect_human(project)
    wb = openpyxl.load_workbook(export_project(project, tmp_path / "out.xlsx"))
    rows = list(wb["provenance"].iter_rows(values_only=True))
    assert rows[0] == ("item", "value")
    items = dict(rows[1:])
    assert items["deployment"] == project.get_meta("deployment")
    assert "color-blob" in items["run 1: detect"] and "demo-colour" in items["run 3: embed"]
    assert items["snapshot"].endswith("-processed.db")


def test_snapshot_command(project, capsys):
    main(["snapshot", "--project", str(project.root), "--label", "before review"])
    assert "before-review.db" in capsys.readouterr().out
    assert len(list((project.root / "snapshots").glob("*-before-review.db"))) == 1
