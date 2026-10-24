"""Species models side by side on answered cards (modeltest.py, cli model-test)."""

from bruv_app import demo
from bruv_app.modeltest import report_lines, run
from bruv_app.store import Project

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
