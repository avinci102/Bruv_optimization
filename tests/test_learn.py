import json

import numpy as np
import pytest
from PIL import Image

from bruv_app import demo, learn
from bruv_app.classify import l2_normalize
from bruv_app.cli import main
from bruv_app.learn import Library, add_photos, add_project, check, remove_source
from bruv_app.pipeline import Pipeline, refresh_fish_guesses

from .conftest import label_like_perfect_human


@pytest.fixture(autouse=True)
def own_library(tmp_path, monkeypatch):
    monkeypatch.setenv("BRUV_LIBRARY", str(tmp_path / "library.npz"))


def test_projects_go_in_replace_themselves_and_come_out(project, capsys):
    label_like_perfect_human(project)
    project.commit()
    confirmed = sum(f["status"] == "species" for f in project.all_fish())
    assert add_project(project) == confirmed
    main(["learn", "--add", str(project.root)])  # adding again replaces, not doubles
    assert f"Added {confirmed} confirmed cards" in capsys.readouterr().out
    library = Library.load()
    assert len(library.label) == confirmed and library.model == demo.DemoClassifier.name
    assert remove_source(str(project.root.resolve())) == confirmed
    assert len(Library.load().label) == 0


def _clusters(rng, sources, per_species=6, dim=8):
    """Three species, each a tight cluster of examples, in every source."""
    centres = l2_normalize(rng.normal(size=(3, dim)))
    emb, label, source = [], [], []
    for s in sources:
        for k, name in enumerate("abc"):
            for _ in range(per_species):
                emb.append(l2_normalize(centres[k] + 0.1 * rng.normal(size=dim)))
                label.append(name)
                source.append(s)
    return centres, Library("m", np.array(emb, np.float32), np.array(label), np.array(source))


def test_the_library_is_used_only_once_it_beats_the_model_alone(rng):
    centres, library = _clusters(rng, ["video 1", "video 2", "video 3"])
    confused = l2_normalize(rng.normal(size=(3, 8)))  # a species model that mixes them up
    result = check(library, "m", list("abc"), confused, scale=10.0)
    assert result.examples == 54 and result.groups == 3
    assert result.blended >= result.zero_shot + learn.MIN_GAIN and result.weight > 0
    assert check(library, "m", list("abc"), centres, scale=10.0).weight == 0  # the model knows them already
    assert check(library, "another model", list("abc"), confused, scale=10.0).weight == 0
    one_video = Library("m", library.emb[:18], library.label[:18], library.source[:18])
    assert check(one_video, "m", list("abc"), confused, scale=10.0).examples == 0  # nothing to test on
    # Two recordings help just as much, but are too few to tune the weight on others.
    two = Library("m", library.emb[:36], library.label[:36], library.source[:36])
    result = check(two, "m", list("abc"), confused, scale=10.0)
    assert result.groups == 2 and result.blended > result.zero_shot and result.weight == 0


def test_copies_of_one_recording_are_one_group(rng):
    """The same footage in two folders can't pass for two independent recordings: grouped by
    footage, each copy is never guessed from the other."""
    _, library = _clusters(rng, ["D:/drop 1", "E:/backup of drop 1", "D:/drop 2", "D:/drop 3"])
    confused = l2_normalize(rng.normal(size=(3, 8)))
    by_folder = check(library, "m", list("abc"), confused, scale=10.0)
    library.group = np.where(library.source == "E:/backup of drop 1", "D:/drop 1", library.source)
    by_footage = check(library, "m", list("abc"), confused, scale=10.0)
    assert (by_folder.groups, by_footage.groups) == (4, 3)


def test_the_same_footage_added_from_another_folder_replaces_it(project, tmp_path):
    import shutil

    from bruv_app.store import Project

    label_like_perfect_human(project)
    project.commit()
    shutil.copytree(project.root, tmp_path / "another folder")
    first = add_project(project)
    assert add_project(Project(tmp_path / "another folder")) == first
    library = Library.load()
    assert len(library.label) == first and set(library.group.tolist()) == {project.get_meta("deployment")}


def test_cards_that_may_mix_fish_stay_out_of_the_library(project):
    label_like_perfect_human(project)
    cards = [f for f in project.all_fish() if f["status"] == "species"]
    a = next(f for f in cards if f["species"] == demo.RED and f["end_t"] > 17)
    b = next(f for f in cards if f["species"] == demo.YELLOW and f["start_t"] < 17 < f["end_t"])
    project.merge_fish(a["id"], b["id"])
    project.label_fish(a["id"], "species", species=demo.RED)  # answered without confirming its warnings
    clean = sum(f["status"] == "species" for f in project.all_fish()) - 1
    assert add_project(project) == clean
    project.label_fish(a["id"], "species", {"flags": ["pieces look different"], "confirmed": True}, species=demo.RED)
    assert add_project(project) == clean + 1  # once confirmed as one fish, it counts


def test_add_project_needs_its_footage(project):
    project.db.execute("UPDATE chapters SET path='D:/gone/GX010042.MP4'")
    project.db.execute("DELETE FROM meta WHERE key IN ('deployment', 'video_fingerprints')")
    project.commit()
    label_like_perfect_human(project)
    with pytest.raises(ValueError, match="which\\s+recording"):
        add_project(project)


def test_guesses_follow_the_library_once_it_passed(project):
    card = project.all_fish()[0]
    before = json.loads(card["guesses"])[0][0]
    other = next(label for label in project.get_meta("guess_labels") if label != before)
    # Another video's examples, all named `other`, right where this card's crops are.
    crops = [np.frombuffer(d["embedding"], np.float32) for d in project.fish_detections(card["id"]) if d["embedding"] is not None]
    library = Library.load()
    library.replace_source(project.get_meta("embedder"), "another video", "dep-another",
                           np.array([l2_normalize(np.mean(crops, axis=0))] * 3), [other] * 3)
    library.save()
    project.set_meta("library_weight", 0.75)
    refresh_fish_guesses(project, card["id"])
    assert json.loads(project.fish(card["id"])["guesses"])[0][0] == other
    project.set_meta("library_weight", 0.0)
    refresh_fish_guesses(project, card["id"])
    assert json.loads(project.fish(card["id"])["guesses"])[0][0] == before


def test_a_changed_library_refreshes_guesses_and_keeps_reviews(project):
    label_like_perfect_human(project)
    reviewed = project.reviewed_count()
    add_project(project)
    Pipeline(project, classifier=demo.DemoClassifier()).run()  # cards stay; only the guesses re-run
    assert project.reviewed_count() == reviewed
    assert project.get_meta("library_stamp") == learn.library_stamp() != ""
    assert project.get_meta("library_check")["examples"] == 0  # one video: nothing to test it on


def test_photos_sorted_into_species_folders(tmp_path):
    for name, colour in demo.COLORS.items():
        folder = tmp_path / "photos" / name
        folder.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (40, 30), colour).save(folder / f"{i}.jpg")
    assert add_photos(tmp_path / "photos", demo.DemoClassifier()) == 6
    assert sorted(set(Library.load().label.tolist())) == sorted(demo.COLORS)
    # The same photos copied elsewhere are the same set: they replace, not double.
    import shutil

    shutil.copytree(tmp_path / "photos", tmp_path / "copy of photos")
    assert add_photos(tmp_path / "copy of photos", demo.DemoClassifier()) == 6
    assert len(Library.load().label) == 6 and len(set(Library.load().group.tolist())) == 1
