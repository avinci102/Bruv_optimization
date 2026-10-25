"""Species models side by side on answered cards (modeltest.py, cli model-test)."""

import numpy as np

from bruv_app import demo
from bruv_app.fishial import family_name, prepare, same_family
from bruv_app.modeltest import Card, ModelScores, closed_families, report_lines, run, score
from bruv_app.store import Project


def test_fishial_species_map_to_a_templates_families():
    assert family_name("Lutjanus bohar") == "snapper" and family_name("Plectropomus laevis") == "grouper"
    assert family_name("Naso unicornis") == "unicornfish" and family_name("Cheilinus undulatus") == "napoleon wrasse"
    assert family_name("Cheilinus trilobatus") == "wrasse" and family_name("Abramis brama") is None
    assert same_family("jack_trevally", "jack trevally") and same_family("batfish / spadefish", "batfish spadefish")
    # A crop is padded to Fishial's wide shape, not stretched, and comes out channels first.
    tall = np.zeros((100, 40, 3), np.uint8)
    assert prepare(tall).shape == (3, 154, 434)


def test_a_closed_list_is_scored_by_family_and_as_a_second_opinion():
    """Today's model names the template's entries; a closed-list model (as Fishial) names
    species of its own, scored only by family, and set against today's model."""
    eye = np.eye(3, dtype=np.float32)
    cards = [Card(i, "species", answer, f"t{i}", 0.0, 1.0, [i]) for i, answer in
             enumerate(["grouper family", "snapper family", "unicornfish"])]
    today = ModelScores("open", ["grouper family", "snapper family", "unicornfish"], eye, 10.0,
                        {0: eye[0], 1: eye[0], 2: eye[2]})  # calls the snapper a grouper
    species = ["Epinephelus coioides", "Lutjanus bohar", "Naso unicornis"]
    closed = ModelScores("closed", species, None, 1.0, {0: eye[0], 1: eye[1], 2: eye[2]}, scorer=lambda e: eye @ e)
    family = {"grouper family": "grouper", "snapper family": "snapper", "unicornfish": "unicornfish"}
    family.update(closed_families([today, closed], family))
    assert family["Lutjanus bohar"] == "snapper"
    result = score(cards, [today, closed], family)
    assert result["runs"]["open"]["tally"]["family"] == 2 and result["runs"]["closed"]["tally"]["family"] == 3
    assert "exact" not in result["runs"]["closed"]["tally"]  # its names aren't the template's
    assert "open + closed" not in result["runs"]  # different lists of names aren't averaged
    assert result["agreement"]["closed"] == {"agree": 2, "agree, open right": 2, "disagree": 1,
                                             "disagree, open right": 0, "disagree, closed right": 1}
    assert any("agree on the family" in line for line in report_lines(result))

from .conftest import label_like_perfect_human


class OtherModel(demo.DemoClassifier):
    """A second model for the test: the demo's colours under another name, counting its work."""

    name = "other-colour"

    def __init__(self):
        self.crops = 0
        self.names = 0

    def embed(self, crops):
        self.crops += len(crops)
        return super().embed(crops)

    def text_embeddings(self, labels):
        self.names += len(labels)
        return super().text_embeddings(labels)


def test_models_are_scored_on_the_same_crops_and_your_answers(project, tmp_path):
    label_like_perfect_human(project)
    reader = Project.read_only(project.root / "project.db")  # the test only reads the project
    other = OtherModel()
    result, saved = run(reader, other, tmp_path / "out")
    runs = result["runs"]
    assert set(runs) == {"demo-colour", "other-colour", "demo-colour + other-colour",
                         "demo-colour + your answers", "other-colour + your answers"}
    named = sum(f["status"] == "species" for f in project.all_fish())
    for name in ("demo-colour", "other-colour", "demo-colour + other-colour"):
        t = runs[name]["tally"]
        assert t["cards"] == named and 0 <= t["exact"] <= t["top 3"] <= t["cards"]
    # The same model under another name scores the same; together they score the same too.
    assert runs["demo-colour"]["picks"] == runs["other-colour"]["picks"] == runs["demo-colour + other-colour"]["picks"]
    assert runs["demo-colour + your answers"]["tally"]["cards"] == named
    assert saved.exists() and any("demo-colour" in line for line in report_lines(result))
    # The other model's embeddings are kept: a second run makes none.
    before = (other.crops, other.names)
    assert before[0] > 0 and before[1] == len(project.get_meta("guess_labels"))
    run(reader, other, tmp_path / "out")
    assert (other.crops, other.names) == before
