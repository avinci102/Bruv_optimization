import json

import openpyxl
import pytest

from bruv_app import demo
from bruv_app.export import export_project
from bruv_app.pipeline import Pipeline, create_project
from bruv_app.detect import ColorBlobDetector

from .conftest import demo_settings, expected_maxn, ground_truth_species, label_like_perfect_human


def test_pipeline_builds_one_card_per_fish_visit(project):
    fish = project.all_fish()
    assert project.frame_count() == 24
    # 10 scene fish, but returning same-species fish within 10 s share a card.
    assert 4 <= len(fish) < 10
    for f in fish:
        guesses = json.loads(f["guesses"])
        assert len(guesses) == 3
        assert f["status"] == "pending"  # nothing is pre-selected


def test_cards_are_pure(project):
    """No card mixes species (the tracker's appearance gate stops identity swaps)."""
    for f in project.all_fish():
        species = {ground_truth_species(d) for d in project.fish_detections(f["id"])}
        assert len(species - {None}) == 1, (f["id"], species)


def test_top_guess_matches_truth_on_synthetic_data(project):
    for f in project.all_fish():
        truth = ground_truth_species(project.fish_detections(f["id"])[0])
        assert json.loads(f["guesses"])[0][0] == truth


def test_end_to_end_maxn_matches_ground_truth(project, tmp_path):
    label_like_perfect_human(project)
    out = export_project(project, tmp_path / "export.xlsx")
    wb = openpyxl.load_workbook(out)
    maxn = {r[0]: r[1] for r in wb["MaxN"].iter_rows(min_row=2, values_only=True)}
    assert maxn == expected_maxn() == {demo.RED: 2, demo.YELLOW: 3, demo.GREEN: 1}

    first = {r[0]: r[6] for r in wb["MaxN"].iter_rows(min_row=2, values_only=True)}
    assert first == {demo.RED: 1.0, demo.YELLOW: 3.0, demo.GREEN: 21.0}

    ws = wb[demo.SHEET_RECORD]
    col = {c.value: c.column for c in ws[1]}
    records = [
        {h: ws.cell(r, c).value for h, c in col.items()} for r in range(3, ws.max_row + 1)
    ]
    assert {r["species"] for r in records} == {demo.RED, demo.YELLOW, demo.GREEN}
    assert all(r["clip_in"] in ("GX010042", "GX020042") for r in records)
    assert all(r["length_ind"] is None for r in records)
    # The yellow school: one sighting row, counted 3 at its peak.
    assert max(r["count"] for r in records if r["species"] == demo.YELLOW) == 3

    details = list(wb["app_details"].iter_rows(min_row=2, values_only=True))
    assert len(details) == len(project.all_fish())
    assert all(row[10] is True for row in details)  # user agreed with top-1 everywhere


def test_detection_resumes_after_interruption(tmp_path, synthetic_videos):
    template = demo.make_mock_template(tmp_path / "t.xlsx")
    project = create_project(tmp_path / "p", synthetic_videos, template, demo_settings(batch_size=4))

    class Crashy(ColorBlobDetector):
        calls = 0

        def detect(self, images):
            Crashy.calls += 1
            if Crashy.calls == 3:
                raise KeyboardInterrupt
            return super().detect(images)

    with pytest.raises(KeyboardInterrupt):
        Pipeline(project, detector=Crashy(), classifier=demo.DemoClassifier()).run()
    assert project.frame_count() == 8
    Pipeline(project, detector=ColorBlobDetector(), classifier=demo.DemoClassifier()).run()
    times = [r["t"] for r in project.db.execute("SELECT t FROM frames ORDER BY seq")]
    assert len(times) == 24
    assert [round(t) for t in times] == list(range(24))


def test_rerun_keeps_reviews_and_refuses_to_wipe_them(project):
    first = project.all_fish()[0]["id"]
    project.label_fish(first, "species", species=demo.RED)
    Pipeline(project, classifier=demo.DemoClassifier()).run()  # all stages done: no-op
    assert project.fish(first)["species"] == demo.RED
    for stage in ("merge", "classify"):
        project.mark_stage(stage, False)
    with pytest.raises(RuntimeError):
        Pipeline(project, classifier=demo.DemoClassifier()).run()


def test_analysis_window_limits_frames(tmp_path, synthetic_videos):
    template = demo.make_mock_template(tmp_path / "t.xlsx")
    settings = demo_settings(start_s=5.0, duration_s=10.0, sample_fps=2.0)
    project = create_project(tmp_path / "p", synthetic_videos, template, settings)
    Pipeline(project, classifier=demo.DemoClassifier()).run()
    times = [r["t"] for r in project.db.execute("SELECT t FROM frames ORDER BY seq")]
    assert len(times) == 20 and times[0] == pytest.approx(5.0) and times[-1] < 15.0
