import hashlib
from datetime import time

import openpyxl
import pytest

from bruv_app import demo
from bruv_app.template import export_workbook, read_raw_validations, read_template, seconds_to_time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reads_dropdowns_from_all_validation_kinds(mock_template):
    info = read_template(mock_template)
    assert info.record_sheet == demo.SHEET_RECORD
    assert info.header_row == 1
    assert info.next_row == 3  # the example row 2 is left alone
    assert info.options("species")[:3] == [demo.RED, demo.YELLOW, demo.GREEN]  # x14 cross-sheet
    assert info.options("behavior") == demo.BEHAVIORS  # x14 cross-sheet
    assert info.options("type") == demo.TYPES  # named range
    assert info.options("sex") == demo.SEXES  # literal list
    assert sum(v.from_extension for v in info.validations) == 2


def test_falls_back_to_dropdown_sheet_column_when_no_validation(tmp_path):
    wb = openpyxl.Workbook()
    dd = wb.active
    dd.title = demo.SHEET_DROPDOWN
    dd.append(["species", "sex"])
    dd.append(["Fish one", "M"])
    dd.append(["Fish two", "F"])
    rec = wb.create_sheet(demo.SHEET_RECORD)
    rec.append(["clip_in", "time_in", "species", "sex"])
    path = tmp_path / "plain.xlsx"
    wb.save(path)
    info = read_template(path)
    assert info.options("species") == ["Fish one", "Fish two"]
    assert info.options("sex") == ["M", "F"]


def test_finds_header_row_below_a_title(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "record"
    ws.append(["My BRUV sheet"])
    ws.append(["clip_in", "time_in", "species"])
    path = tmp_path / "titled.xlsx"
    wb.save(path)
    info = read_template(path)
    assert info.header_row == 2 and info.headers["species"] == 3


def test_export_writes_by_header_name_into_a_copy(mock_template, tmp_path):
    before = sha(mock_template)
    info = read_template(mock_template)
    out = tmp_path / "out.xlsx"
    rows = [
        {"clip_in": "GX010042", "time_in": seconds_to_time(75), "clip_out": "GX010042",
         "time_out": seconds_to_time(80), "species": demo.RED, "count": 2, "behavior": "feeding",
         "not_a_header": "x"},
    ]
    result = export_workbook(info, out, rows, ["species", "maxn"], [[demo.RED, 2]], ["fish_id"], [[1]])
    assert sha(mock_template) == before, "the original template must never change"
    assert result["first_row"] == 3 and result["skipped_headers"] == ["not_a_header"]

    wb = openpyxl.load_workbook(out)
    ws = wb[demo.SHEET_RECORD]
    col = {c.value: c.column for c in ws[1]}
    assert ws.cell(3, col["species"]).value == demo.RED
    assert ws.cell(3, col["time_in"]).value == time(0, 1, 15)
    assert ws.cell(3, col["time_in"]).number_format == "hh:mm:ss"
    assert ws.cell(3, col["count"]).value == 2
    assert ws.cell(3, col["length_ind"]).value is None
    assert ws.cell(2, col["clip_in"]).value == "clip_1"  # example row untouched
    for letter in demo.HIDDEN:
        assert ws.column_dimensions[letter].hidden
        assert ws[f"{letter}3"].value is None
    assert wb["MaxN"]["B2"].value == 2
    assert wb["app_details"]["A2"].value == 1


def test_export_keeps_every_dropdown(mock_template, tmp_path):
    info = read_template(mock_template)
    out = tmp_path / "out.xlsx"
    export_workbook(info, out, [], ["species"], [], ["fish_id"], [])
    formulas = {v.formula1 for v in read_raw_validations(out, demo.SHEET_RECORD)}
    assert formulas == {v.formula1 for v in info.validations}
    # The exported copy reads back with the same lists.
    assert read_template(out).dropdowns == info.dropdowns


def test_refuses_to_overwrite_the_template(mock_template):
    info = read_template(mock_template)
    with pytest.raises(ValueError):
        export_workbook(info, mock_template, [], [], [], [], [])


def test_seconds_to_time():
    assert seconds_to_time(3725.4) == time(1, 2, 5)
    assert seconds_to_time(-3) == time(0, 0, 0)


def test_reattached_dropdowns_keep_their_error_settings(mock_template, tmp_path):
    info = read_template(mock_template)
    out = tmp_path / "out.xlsx"
    export_workbook(info, out, [], ["species"], [], ["fish_id"], [])
    species = next(v for v in read_raw_validations(out, demo.SHEET_RECORD) if v.sqref == "N2:N1000")
    # Typed values outside the list must still be rejected in Excel.
    assert species.options.get("showErrorMessage") is True
    assert species.options.get("showInputMessage") is True


def test_export_reports_template_features_it_cannot_keep(mock_template, tmp_path, caplog, monkeypatch):
    import logging
    import warnings

    info = read_template(mock_template)
    real = openpyxl.load_workbook

    def noisy(*args, **kwargs):
        warnings.warn("Conditional Formatting extension is not supported and will be removed")
        return real(*args, **kwargs)

    monkeypatch.setattr(openpyxl, "load_workbook", noisy)
    with caplog.at_level(logging.WARNING):
        export_workbook(info, tmp_path / "out.xlsx", [], ["species"], [], ["fish_id"], [])
    assert any("Conditional Formatting" in r.message for r in caplog.records)
    assert not any("Data Validation" in r.message for r in caplog.records)
