#!/usr/bin/env python3
from pathlib import Path

from scripts.file_tracker import FileTracker


def test_filetracker_sandbox_suppresses_writes(tmp_path: Path):
    # Create a tracker pointing to a temp log file but with sandbox=True
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "file_operations.log"

    tracker = FileTracker("sandbox_test", log_file=str(log_file.name), sandbox=True)

    # Attempt to write entries
    tracker.log_operation("move", source_dir="a", dest_dir="b", file_count=1)
    tracker.log_batch_start("batch")
    tracker.log_batch_end("done")
    tracker.log_user_action("delete", details="x")

    # No log file should have been created
    assert not log_file.exists(), "Sandbox mode must suppress log writes"
