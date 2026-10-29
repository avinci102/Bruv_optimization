"""sitesync: two PCs deploy to one site through the live site itself, and a deploy never
takes a live video off the site, puts back an older copy, or runs older code unasked."""

import functools
import json
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from bruv_app import sitesync
from bruv_app.sitesync import Plan, check, compare, deploy, local_videos, mirror


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def live(tmp_path):
    """A stand-in for the live site: a folder served over HTTP, which a deploy replaces."""
    folder = tmp_path / "live"
    folder.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(folder=folder, url=f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()


def pages(live):
    """What `wrangler pages deploy` does: the live site becomes the folder's public/."""
    def run(site):
        shutil.rmtree(live.folder)
        shutil.copytree(site / "public", live.folder)
        return SimpleNamespace(returncode=0)
    return run


def video(site, name, **files):
    folder = site / "public" / "videos" / name
    for rel, text in {"manifest.json": "{}", "sheets/1.webp": name, "follow/0.webp": name * 3, **files}.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text(text, encoding="utf-8")


def ship(site, live, **kw):
    return deploy(site, live.url, yes=True, run=lambda: pages(live)(site), log=lambda *_: None, **kw)


def test_a_second_pc_mirrors_the_live_videos_and_adds_its_own_without_dropping_any(tmp_path, live):
    laptop, gpu = tmp_path / "laptop", tmp_path / "gpu"
    video(laptop, "trial")
    video(laptop, "A1", **{"cards.json": "[1]"})
    # The live site has no index yet: the laptop, which holds every live video, goes first.
    assert not ship(laptop, live)
    assert ship(laptop, live, first=True)
    index = json.loads((live.folder / "videos" / "index.json").read_text(encoding="utf-8"))
    assert sorted(index["videos"]) == ["A1", "trial"] and index["commit"]

    # A new PC: nothing of its own yet. It may not deploy until it has the live videos.
    video(gpu, "B1")
    assert not ship(gpu, live)
    mirror(gpu, live.url, log=lambda *_: None)
    assert local_videos(gpu)["A1"] == local_videos(laptop)["A1"]
    assert local_videos(gpu)["trial"] == local_videos(laptop)["trial"]
    assert ship(gpu, live)
    assert sorted(p.name for p in (live.folder / "videos").iterdir() if p.is_dir()) == ["A1", "B1", "trial"]

    # The laptop now lacks B1: deploying would drop it, until it mirrors.
    result = check(laptop, local_videos(laptop), sitesync.live_index(live.url), sitesync.load_state(laptop),
                   sitesync.code_version())
    assert result.plan.missing == ["B1"] and any("B1 is live but not in this folder" in p for p in result.problems)
    mirror(laptop, live.url, log=lambda *_: None)
    assert ship(laptop, live)


def test_a_video_re_published_on_one_pc_goes_up_and_the_other_pc_takes_it(tmp_path, live):
    laptop, gpu = tmp_path / "laptop", tmp_path / "gpu"
    video(laptop, "A1")
    assert ship(laptop, live, first=True)
    mirror(gpu, live.url, log=lambda *_: None)
    # Re-published on the GPU PC (new cards): a change made here goes up.
    video(gpu, "A1", **{"cards.json": "[1, 2]"})
    (gpu / "public" / "videos" / "A1" / "follow" / "0.webp").unlink()
    assert ship(gpu, live)
    # The laptop's copy is now the older one: it may not put it back, and mirror replaces it.
    result = check(laptop, local_videos(laptop), sitesync.live_index(live.url), sitesync.load_state(laptop),
                   sitesync.code_version())
    assert result.plan.changed_online == ["A1"] and result.problems
    mirror(laptop, live.url, log=lambda *_: None)
    assert local_videos(laptop)["A1"] == local_videos(gpu)["A1"]
    assert not (laptop / "public" / "videos" / "A1" / "follow" / "0.webp").exists()


def test_taking_a_video_off_the_site_is_asked_for_by_name(tmp_path, live):
    site = tmp_path / "site"
    video(site, "A1")
    video(site, "old")
    assert ship(site, live, first=True)
    shutil.rmtree(site / "public" / "videos" / "old")
    assert not ship(site, live)
    assert ship(site, live, drop=("old",))
    assert not (live.folder / "videos" / "old").exists()


def test_an_interrupted_download_is_fetched_again_not_uploaded(tmp_path, live, monkeypatch):
    laptop, gpu = tmp_path / "laptop", tmp_path / "gpu"
    video(laptop, "A1", **{f"sheets/{n}.webp": str(n) for n in range(2, 12)})
    assert ship(laptop, live, first=True)
    real, calls = sitesync._get, []

    def flaky(url, timeout=120):
        calls.append(url)
        if "/videos/A1/" in url and len(calls) > 4:
            raise OSError("connection reset")
        return real(url, timeout)

    monkeypatch.setattr(sitesync, "_get", flaky)
    monkeypatch.setattr(sitesync.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        mirror(gpu, live.url, workers=1, log=lambda *_: None)
    monkeypatch.setattr(sitesync, "_get", real)
    assert sitesync.load_state(gpu)["unfinished"] == ["A1"]
    result = check(gpu, local_videos(gpu), sitesync.live_index(live.url), sitesync.load_state(gpu),
                   sitesync.code_version())
    assert any("didn't finish" in p for p in result.problems)
    mirror(gpu, live.url, log=lambda *_: None)
    assert sitesync.load_state(gpu)["unfinished"] == []
    assert local_videos(gpu)["A1"] == local_videos(laptop)["A1"]


def test_the_sort_of_each_video():
    plan = compare(
        here={"same": "1", "new": "1", "mine": "2", "theirs": "1", "both": "3", "gone_online": "1"},
        live={"same": "1", "mine": "1", "theirs": "2", "both": "2", "missing": "1", "deleted_here": "1"},
        seen={"mine": "1", "theirs": "1", "both": "1", "gone_online": "1", "deleted_here": "1"},
    )
    assert plan == Plan(same=["same"], new=["new"], changed_here=["mine"], missing=["missing"],
                        changed_online=["theirs"], removed_here=["deleted_here"], removed_online=["gone_online"],
                        conflict=["both"])


def test_code_older_than_the_live_sites_and_oversized_files_are_refused(tmp_path):
    site = tmp_path / "site"
    video(site, "A1", **{"emb.bin": "x" * 300})
    here = local_videos(site)
    code = sitesync.code_version()
    live = {"format": 1, "commit": "0" * 40, "dirty": False, "videos": {}}
    result = check(site, here, live, {}, code)
    assert any("this checkout doesn't have" in p for p in result.problems)
    assert not check(site, here, live, {}, code, force_code=True).problems
    if code["commit"]:  # the live site deployed from this commit, plus uncommitted changes
        live.update(commit=code["commit"], dirty=True, site_code="other")
        assert any("uncommitted changes" in p for p in check(site, here, live, {}, code).problems)
        live.update(site_code=code["site_code"])  # the same uncommitted code as here
        assert not check(site, here, live, {}, code).problems
    result = check(site, here, None, {}, code, first=True, size_limit=200, file_limit=3)
    assert any("emb.bin is" in p for p in result.problems) and any("more than one free" in p for p in result.problems)


def test_the_site_address_comes_from_wrangler_toml(tmp_path):
    (tmp_path / "wrangler.toml").write_text(
        'name = "bruv-review"\npages_build_output_dir = "public"\n\n[[d1_databases]]\nbinding = "DB"\n'
        'database_name = "other"\n', encoding="utf-8")
    assert sitesync.site_url(tmp_path) == "https://bruv-review.pages.dev"
    assert sitesync.site_url(tmp_path, "https://x.pages.dev/") == "https://x.pages.dev"
