#!/usr/bin/env python3
"""Universal File Tracking System for Image Processing Scripts

This module provides a simple logging system that all scripts can use to track:
- File movements between directories
- File operations (copy, move, delete)
- Batch operations with before/after counts
- Script execution sessions

Usage in any script:
    from file_tracker import FileTracker

    tracker = FileTracker("script_name")
    tracker.log_operation("move", source_dir="1101", dest_dir="face_group_1", file_count=25)
    tracker.log_batch_start("Face grouping batch 1")
    tracker.log_batch_end()
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class FileTracker:
    def __init__(
        self,
        script_name: str,
        log_file: str = "file_operations.log",
        sandbox: bool = False,
    ):
        self.script_name = script_name
        self.sandbox = sandbox
        self.logger = logging.getLogger(f"file_tracker.{script_name}")

        # Create file_operations_logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / "data" / "file_operations_logs"
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / log_file
        self.session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.current_batch = None
        self._write_failures = 0  # Track consecutive failures

        # Check if we need to start fresh for a new day
        self._check_and_clean_log()

        # Log session start (suppressed in sandbox)
        self._log_entry(
            {
                "type": "session_start",
                "script": script_name,
                "session_id": self.session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "working_dir": str(Path.cwd()),
            }
        )

    def _check_and_clean_log(self):
        """Check if log is from a previous day and clean it if so"""
        try:
            if not self.log_file.exists():
                return  # No log file yet, nothing to clean

            # Get the current date
            today = datetime.now(UTC).date()

            # Read the last entry to get its date
            last_entry_date = None
            try:
                with self.log_file.open() as f:
                    lines = f.readlines()
                    if lines:
                        # Get the last line that's not empty
                        for line in reversed(lines):
                            line = line.strip()
                            if line:
                                try:
                                    entry = json.loads(line)
                                    timestamp = datetime.fromisoformat(
                                        entry["timestamp"]
                                    )
                                    last_entry_date = timestamp.date()
                                    break
                                except (json.JSONDecodeError, KeyError):
                                    continue
            except OSError:
                # If we can't read the file, start fresh
                last_entry_date = None

            # If last entry is from a different day, or earlier time today, clean the log
            if last_entry_date is None or last_entry_date < today:
                self.logger.info("Starting fresh log for %s", today)

                # Create backup of old log with date
                if last_entry_date:
                    backup_name = (
                        f"file_operations_{last_entry_date.strftime('%Y%m%d')}.log"
                    )
                    backup_path = self.log_file.parent / backup_name
                    try:
                        self.log_file.rename(backup_path)
                        self.logger.info("Backed up previous log to %s", backup_name)
                    except OSError as e:
                        # If backup fails, log and remove the old log
                        self.logger.warning("Could not backup log file: %s", e)
                        self.log_file.unlink()
                else:
                    # Just remove the corrupted log
                    self.log_file.unlink()

        except OSError as e:
            # File system errors during log rotation are critical
            self.logger.error("Failed to check/rotate log file: %s", e)
            raise  # Don't continue with broken logging

    def _log_entry(self, entry: dict[str, Any]):
        """Write a log entry to the file."""
        if self.sandbox:
            return

        try:
            with self.log_file.open("a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()  # Ensure write completes
            self._write_failures = 0  # Reset failure counter on success

        except OSError as e:
            self._write_failures += 1
            self.logger.error(
                "Failed to write log entry (failure #%d): %s", self._write_failures, e
            )

            # After 3 consecutive failures, this is a critical issue
            if self._write_failures >= 3:
                msg = f"FileTracker failed {self._write_failures} consecutive writes. Logging is broken."
                raise RuntimeError(msg) from e

            # Otherwise re-raise to alert caller
            raise

    def log_operation(
        self,
        operation: str,
        source_dir: str | None = None,
        dest_dir: str | None = None,
        file_count: int | None = None,
        files: list | None = None,
        notes: str | None = None,
    ):
        """Log a file operation.

        Args:
            operation: 'move', 'copy', 'delete', 'create'
            source_dir: Source directory name
            dest_dir: Destination directory name
            file_count: Number of files affected
            files: List of specific filenames (optional)
            notes: Additional notes
        """
        entry = {
            "type": "file_operation",
            "script": self.script_name,
            "session_id": self.session_id,
            "batch_id": self.current_batch,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "source_dir": source_dir,
            "dest_dir": dest_dir,
            "file_count": file_count,
            "notes": notes,
        }

        # Only include files list if it's small (to avoid huge logs)
        if files and len(files) <= 10:
            entry["files"] = files
        elif files:
            entry["files_sample"] = files[:5]  # Just first 5 as sample

        self._log_entry(entry)

    def log_metric_mode_update(self, mode: str, details: str = ""):
        """Record a change in how metrics are counted (e.g., image-only)."""
        self._log_entry(
            {
                "type": "metric_mode_update",
                "script": self.script_name,
                "session_id": self.session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "mode": mode,
                "details": details,
            }
        )

    def log_directory_state(self, directory: str, description: str = ""):
        """Log the current state of a directory"""
        try:
            dir_path = Path(directory)
            if dir_path.exists():
                png_count = len(list(dir_path.glob("*.png")))
                yaml_count = len(list(dir_path.glob("*.yaml")))
                total_files = len(list(dir_path.iterdir()))

                self._log_entry(
                    {
                        "type": "directory_state",
                        "script": self.script_name,
                        "session_id": self.session_id,
                        "batch_id": self.current_batch,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "directory": directory,
                        "png_files": png_count,
                        "yaml_files": yaml_count,
                        "total_files": total_files,
                        "description": description,
                    }
                )
        except OSError as e:
            # Directory access failures should be visible
            self.logger.error("Could not log directory state for %s: %s", directory, e)
            raise

    def log_batch_start(self, batch_name: str, description: str = ""):
        """Start a new batch operation"""
        self.current_batch = f"{self.session_id}_{batch_name.replace(' ', '_')}"
        self._log_entry(
            {
                "type": "batch_start",
                "script": self.script_name,
                "session_id": self.session_id,
                "batch_id": self.current_batch,
                "timestamp": datetime.now(UTC).isoformat(),
                "batch_name": batch_name,
                "description": description,
            }
        )

    def log_batch_end(self, summary: str = ""):
        """End the current batch operation"""
        if self.current_batch:
            self._log_entry(
                {
                    "type": "batch_end",
                    "script": self.script_name,
                    "session_id": self.session_id,
                    "batch_id": self.current_batch,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "summary": summary,
                }
            )
            self.current_batch = None

    def log_user_action(self, action: str, details: str = ""):
        """Log a user action/decision"""
        self._log_entry(
            {
                "type": "user_action",
                "script": self.script_name,
                "session_id": self.session_id,
                "batch_id": self.current_batch,
                "timestamp": datetime.now(UTC).isoformat(),
                "action": action,
                "details": details,
            }
        )

    def close(self):
        """Explicitly close the session and log session end"""
        try:
            self._log_entry(
                {
                    "type": "session_end",
                    "script": self.script_name,
                    "session_id": self.session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except (OSError, RuntimeError) as e:
            # Session end failures are important - don't hide them
            self.logger.error("Failed to log session end: %s", e)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_log(
    log_file: str | None = None,
    script_filter: str | None = None,
    session_filter: str | None = None,
    operation_filter: str | None = None,
):
    """Read and filter log entries.

    Args:
        log_file: Path to log file (default: file_operations.log)
        script_filter: Only show entries from this script
        session_filter: Only show entries from this session
        operation_filter: Only show this type of operation
    """
    logger = logging.getLogger("file_tracker.read_log")

    if log_file is None:
        log_file = (
            Path(__file__).parent.parent
            / "data"
            / "file_operations_logs"
            / "file_operations.log"
        )

    if not Path(log_file).exists():
        logger.warning("No log file found at %s", log_file)
        return []

    entries = []
    try:
        with Path(log_file).open() as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Apply filters
                    if script_filter and entry.get("script") != script_filter:
                        continue
                    if session_filter and entry.get("session_id") != session_filter:
                        continue
                    if operation_filter and entry.get("operation") != operation_filter:
                        continue

                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.error("Error reading log file %s: %s", log_file, e)
        raise

    return entries


def print_recent_activity(hours: int = 24):
    """Print recent file activity in a human-readable format."""
    from datetime import timedelta  # noqa: PLC0415

    entries = read_log()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    print(f"\n=== File Activity (Last {hours} hours) ===")  # noqa: T201

    for entry in entries:
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if timestamp < cutoff:
                continue

            time_str = timestamp.strftime("%H:%M:%S")

            if entry["type"] == "file_operation":
                op = entry["operation"]
                source = entry.get("source_dir", "")
                dest = entry.get("dest_dir", "")
                count = entry.get("file_count", "?")

                if op == "move":
                    print(  # noqa: T201
                        f"{time_str} [{entry['script']}] MOVED {count} files: {source} → {dest}"
                    )
                elif op == "delete":
                    print(  # noqa: T201
                        f"{time_str} [{entry['script']}] DELETED {count} files from {source}"
                    )
                elif op == "copy":
                    print(  # noqa: T201
                        f"{time_str} [{entry['script']}] COPIED {count} files: {source} → {dest}"
                    )

            elif entry["type"] == "batch_start":
                print(  # noqa: T201
                    f"{time_str} [{entry['script']}] BATCH START: {entry['batch_name']}"
                )

            elif entry["type"] == "session_start":
                print(f"{time_str} [{entry['script']}] SESSION START")  # noqa: T201

        except (KeyError, ValueError):  # Specific exceptions for timestamp parsing
            continue

    print("=" * 50)  # noqa: T201


if __name__ == "__main__":
    # If run directly, show recent activity
    print_recent_activity()
