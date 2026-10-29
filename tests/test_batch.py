"""The GPU PC's batch: every deployment folder processed, each new video published once
under its folder's name, never over another project's video of the same name."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from bruv_app import batch, sitesync


class FakeApp:
    """Stands in for `python -m bruv_app ...`: quickstart makes a project, publish writes the
    video's folder and records it in the project, as the real ones do."""

    def __init__(self, codes=()):
        self.calls, self.codes = [], list(codes)

    def __call__(self, cmd):
        args = cmd[3:]
        self.calls.append(args)
        if self.codes:
            return SimpleNamespace(returncode=self.codes.pop(0))
        if args[0] == "quickstart":
            project = Path(args[1]) / "bruv_project"
            project.mkdir(exist_ok=True)
            with sqlite3.connect(project / "project.db") as con:
                con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            con.close()
        elif args[0] == "publish":
            project, name, site = (args[args.index(flag) + 1] for flag in ("--project", "--video", "--site"))
            folder = Path(site) / "public" / "videos" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "manifest.json").write_text("{}", encoding="utf-8")
            with sqlite3.connect(Path(project) / "project.db") as con:
                con.execute("INSERT OR REPLACE INTO meta VALUES ('published', ?)", (json.dumps({"video": name}),))
            con.close()
        return SimpleNamespace(returncode=0)


def setup(tmp_path, monkeypatch):
    videos, site = tmp_path / "videos", tmp_path / "site"
    for folder, clip in (("A1", "GX010042.MP4"), ("B 2", "DJI_0001.mp4")):
        (videos / folder).mkdir(parents=True)
        (videos / folder / clip).write_bytes(b"video")
    (videos / "notes").mkdir()  # no video: not a deployment
    (videos / "viewing template.xlsx").write_bytes(b"template")
    sitesync.save_state(site, {"url": "http://live.invalid"})
    monkeypatch.setattr(sitesync, "live_index", lambda url: {"format": 1, "commit": None, "videos": {}})
    return videos, site


def test_each_folder_is_processed_and_each_new_video_published_once(tmp_path, monkeypatch):
    videos, site = setup(tmp_path, monkeypatch)
    app = FakeApp()
    assert batch.main(["--videos", str(videos), "--site", str(site), "--no-mirror"], run=app) == 0
    assert [c[:2] for c in app.calls if c[0] == "quickstart"] == [
        ["quickstart", str(videos / "A1")], ["quickstart", str(videos / "B 2")]]
    published = [c for c in app.calls if c[0] == "publish"]
    assert [c[c.index("--video") + 1] for c in published] == ["A1", "B-2"]
    assert all("--follow" in c for c in published)
    assert (videos / "A1" / "viewing template.xlsx").exists() and (videos / "B 2" / "viewing template.xlsx").exists()

    # Run again: processing resumes (nothing to do), nothing is published twice.
    app = FakeApp()
    assert batch.main(["--videos", str(videos), "--site", str(site), "--no-mirror", "--no-follow"], run=app) == 0
    assert [c[0] for c in app.calls] == ["quickstart", "quickstart"]
    app = FakeApp()
    batch.main(["--videos", str(videos), "--site", str(site), "--no-mirror", "--only", "A1", "--republish", "A1",
                "--no-follow"], run=app)
    assert [c[0] for c in app.calls] == ["quickstart", "publish"] and "--follow" not in app.calls[1]


def test_a_name_already_on_the_site_from_another_project_is_left_alone(tmp_path, monkeypatch):
    videos, site = setup(tmp_path, monkeypatch)
    taken = site / "public" / "videos" / "A1"
    taken.mkdir(parents=True)
    (taken / "manifest.json").write_text('{"theirs": true}', encoding="utf-8")
    app = FakeApp()
    assert batch.main(["--videos", str(videos), "--site", str(site), "--no-mirror"], run=app) == 2
    assert [c[c.index("--video") + 1] for c in app.calls if c[0] == "publish"] == ["B-2"]
    assert (taken / "manifest.json").read_text(encoding="utf-8") == '{"theirs": true}'


def test_a_crash_is_retried_and_an_explained_error_is_not(monkeypatch):
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    crash_then_ok = FakeApp(codes=[3221225477, 0])
    assert batch.run_app("quickstart", "x", retries=2, run=crash_then_ok) == 0 and len(crash_then_ok.calls) == 2
    explained = FakeApp(codes=[1])
    assert batch.run_app("quickstart", "x", retries=2, run=explained) == 1 and len(explained.calls) == 1


def test_site_names():
    assert batch.site_name("NP339_20260906_BM (1)") == "NP339_20260906_BM-1"
    assert batch.site_name("A1") == "A1"
