import shutil

from bruv_app import demo
from bruv_app.cli import main
from bruv_app.pipeline import Pipeline, create_project
from bruv_app.score import detection_answers, match_detections, score_against, score_cards
from bruv_app.store import Project

from .conftest import demo_settings, label_like_perfect_human


def _two_species(project):
    cards = [f for f in project.all_fish() if f["status"] == "species"]
    a = cards[0]
    return a, next(f for f in cards if f["species"] != a["species"])


def test_scoring_finds_cards_that_mix_answers(project):
    label_like_perfect_human(project)  # the reference answers
    answers = detection_answers(project)
    clean = score_cards(project, answers, tiny_px=0)
    assert clean.cards == clean.checked == len(project.all_fish()) and not clean.mixed

    # A bad joining rule puts two fish that got different answers on one card.
    a, b = _two_species(project)
    project.merge_fish(a["id"], b["id"])
    mixed = score_cards(project, answers, tiny_px=0)
    assert mixed.mixed == [(a["id"], sorted({a["species"], b["species"]}))]
    assert score_cards(project, answers, tiny_px=1e9).tiny == mixed.cards


def test_scoring_checks_what_the_reviewer_pulled_apart(project, tmp_path):
    shutil.copytree(project.root, tmp_path / "reviewed")
    reviewed = Project(tmp_path / "reviewed")
    card = next(f for f in reviewed.all_fish() if len(reviewed.fish_tracklets(f["id"])) > 1)
    pieces = len(reviewed.fish_tracklets(card["id"]))
    reviewed.unmerge_fish(card["id"])  # "Multiple fish"
    again = score_against(project, reviewed)  # the original cards still join those pieces
    assert again.pulled_apart == again.rejoined == pieces - 1
    assert score_against(reviewed, reviewed).rejoined == 0


def test_answers_carry_over_to_a_run_at_another_frame_rate(project, tmp_path, synthetic_videos):
    label_like_perfect_human(project)
    faster = create_project(tmp_path / "2fps", synthetic_videos, None, demo_settings(sample_fps=2.0))
    Pipeline(faster, classifier=demo.DemoClassifier()).run()
    n_detections = project.db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    assert len(match_detections(faster, project)) == n_detections  # every 1 fps frame is in the 2 fps run
    score = score_against(faster, project)
    assert score.checked == score.cards - score.tiny and not score.mixed


def test_score_command(project, tmp_path, capsys):
    label_like_perfect_human(project)
    project.commit()
    shutil.copytree(project.root, tmp_path / "answers")
    main(["score", "--project", str(project.root), "--answers", str(tmp_path / "answers")])
    out = capsys.readouterr().out
    assert f"{len(project.all_fish())} cards" in out and "0 of them mix answers" in out
    assert "0 of 0 groups of sightings you pulled apart" in out
