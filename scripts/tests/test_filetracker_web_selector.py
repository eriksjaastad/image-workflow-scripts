# minimal integration test for web selector FileTracker
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.file_tracker import read_log


def create_images(dirpath: Path, n: int):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dirpath / f"20250101_00000{i}_stage1_generated.png").write_bytes(b"fake")
        (dirpath / f"20250101_00000{i}_stage1_generated.yaml").write_text("meta: 1")


def test_filetracker_metric_marker_and_image_only_counts(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "src"
        create_images(src, 3)

        # simulate FileTracker log entries directly (we won't spin Flask here)
        from scripts.file_tracker import FileTracker
        tracker = FileTracker("test_web_selector")
        tracker.log_metric_mode_update("image_only", details="Counts reflect only .png images; companions excluded")

        # simulate move of one image to selected using companion utils move_with_yaml
        from scripts.utils.companion_file_utils import move_file_with_all_companions
        dest_selected = tmp / "selected"
        dest_selected.mkdir(exist_ok=True)
        moved = move_file_with_all_companions(next(src.glob("*.png")), dest_selected, dry_run=True)
        # Log image-only move (1)
        tracker.log_operation("move", source_dir=str(src.name), dest_dir=str(dest_selected.name), file_count=1, files=[moved[0]], notes="image-only count")

        # simulate delete of one image
        tracker.log_operation("delete", source_dir=str(src.name), dest_dir="trash", file_count=1, files=["x.png"], notes="image-only count")

        entries = read_log()
        assert any(e.get("type") == "metric_mode_update" and e.get("mode") == "image_only" for e in entries)
        # Ensure image-only counts are present (1 move to selected, 1 delete)
        moved_selected = sum(e.get("file_count", 0) for e in entries if e.get("type") == "file_operation" and e.get("operation") == "move" and e.get("dest_dir") == str(dest_selected.name))
        deletes = sum(e.get("file_count", 0) for e in entries if e.get("type") == "file_operation" and e.get("operation") in ("delete", "send_to_trash"))
        assert moved_selected >= 1
        assert deletes >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
