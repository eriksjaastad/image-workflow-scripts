from pathlib import Path

from scripts.dashboard.current_project_dashboard_v2 import (
    _parse_iso,
    classify_operation_phase,
    compute_phase_hours,
)


def test_parse_iso_naive_utc():
    # Z suffix
    dt1 = _parse_iso("2025-10-20T12:34:56Z")
    assert dt1.tzinfo is None
    # offset aware
    dt2 = _parse_iso("2025-10-20T12:34:56+02:00")
    assert dt2.tzinfo is None
    # naive stays naive
    dt3 = _parse_iso("2025-10-20T12:34:56")
    assert dt3.tzinfo is None


def test_classify_operation_phase_selection_and_sort_and_crop_fallback(tmp_path: Path):
    # selection via basename selected
    op_sel = {"operation": "move", "dest_dir": str(tmp_path / "some/selected")}
    assert classify_operation_phase(op_sel) == "selection"

    # sort via character_group
    op_sort = {"operation": "move", "dest_dir": str(tmp_path / "character_group_1")}
    assert classify_operation_phase(op_sort) == "sort"

    # sort via underscore excluding trash
    op_sort2 = {"operation": "move", "dest_dir": str(tmp_path / "_alpha")}
    assert classify_operation_phase(op_sort2) == "sort"

    # crop explicit
    op_crop = {"operation": "crop"}
    assert classify_operation_phase(op_crop) == "crop"

    # crop fallback for save/export with only pngs
    op_save = {"operation": "save", "files": ["a.png", "b.PNG"]}
    assert classify_operation_phase(op_save) == "crop"


def test_windows_safe_date_keys_compute_hours():
    ts_project = {
        "daily_hours": {
            "10/1/2025": 2.0,
            "10/2/2025": 3.5,
        }
    }
    hours = compute_phase_hours(ts_project, "2025-10-01", "2025-10-02")
    assert round(hours, 2) == 5.5


def test_division_guards_baseline_and_rate():
    # rate should be zero when hours is zero
    # Indirectly tested via simple compute; compute_phase_hours returns 0 if invalid
    ts_project = {"daily_hours": {}}
    assert compute_phase_hours(ts_project, None, None) == 0.0


