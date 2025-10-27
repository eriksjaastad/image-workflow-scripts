#!/usr/bin/env python3
"""
Watchdog and Heartbeat utilities for preventing silent hangs in long-running scripts.

Features:
- Heartbeat: thread-safe progress counters and timestamps
- Watchdog: background monitor that aborts on no-progress stall or max-runtime breach
- Error reporting: write sandbox-local error reports and optional stack dump

Sandbox-only usage: callers should pass a sandbox_root and run_id to write
artifacts strictly under the sandbox. No global trackers/logs are used here.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class ProgressSnapshot:
    timestamp_utc: float
    files_scanned: int
    groups_built: int
    items_processed: int
    notes: str = ""


class Heartbeat:
    """Thread-safe progress tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_snapshot = ProgressSnapshot(
            timestamp_utc=time.time(),
            files_scanned=0,
            groups_built=0,
            items_processed=0,
            notes="start",
        )

    def update(self, *, files_scanned: Optional[int] = None, groups_built: Optional[int] = None,
               items_processed: Optional[int] = None, notes: str = "") -> ProgressSnapshot:
        with self._lock:
            now = time.time()
            snap = self._last_snapshot
            self._last_snapshot = ProgressSnapshot(
                timestamp_utc=now,
                files_scanned=snap.files_scanned if files_scanned is None else files_scanned,
                groups_built=snap.groups_built if groups_built is None else groups_built,
                items_processed=snap.items_processed if items_processed is None else items_processed,
                notes=notes or snap.notes,
            )
            return self._last_snapshot

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return self._last_snapshot


class Watchdog:
    """Background watchdog monitoring a Heartbeat.

    Aborts the process via provided abort callback when:
    - No progress for stall_threshold_sec
    - Total wall time exceeds max_runtime_sec
    """

    def __init__(self,
                 heartbeat: Heartbeat,
                 *,
                 start_time_utc: float,
                 max_runtime_sec: float = 900.0,
                 stall_threshold_sec: float = 120.0,
                 poll_interval_sec: float = 5.0,
                 on_abort: Optional[Callable[[str], None]] = None,
                 sandbox_root: Optional[Path] = None,
                 run_id: Optional[str] = None,
                 write_stack: bool = True) -> None:
        self.heartbeat = heartbeat
        self.start_time_utc = start_time_utc
        self.max_runtime_sec = max_runtime_sec
        self.stall_threshold_sec = stall_threshold_sec
        self.poll_interval_sec = poll_interval_sec
        self.on_abort = on_abort
        self.sandbox_root = sandbox_root
        self.run_id = run_id
        self.write_stack = write_stack
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.poll_interval_sec * 2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            # Max runtime enforcement
            if now - self.start_time_utc > self.max_runtime_sec:
                self._abort("max_runtime_exceeded")
                return

            snap = self.heartbeat.snapshot()
            if now - snap.timestamp_utc > self.stall_threshold_sec:
                self._abort("no_progress_stall")
                return

            time.sleep(self.poll_interval_sec)

    def _abort(self, reason: str) -> None:
        try:
            self._write_error_report(reason)
        except Exception:
            pass
        if self.on_abort:
            try:
                self.on_abort(reason)
            except Exception:
                pass

    def _write_error_report(self, reason: str) -> None:
        if not self.sandbox_root or not self.run_id:
            return
        logs_dir = Path(self.sandbox_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Error JSON
        error_path = logs_dir / f"error_{self.run_id}.json"
        snap = self.heartbeat.snapshot()
        payload: Dict[str, Any] = {
            "reason": reason,
            "time": time.time(),
            "start_time": self.start_time_utc,
            "max_runtime_sec": self.max_runtime_sec,
            "stall_threshold_sec": self.stall_threshold_sec,
            "snapshot": asdict(snap),
        }
        error_path.write_text(json.dumps(payload, indent=2))
        # Optional stack dump
        if self.write_stack:
            try:
                import faulthandler
                stack_path = logs_dir / f"stack_{self.run_id}.txt"
                with stack_path.open("w") as f:
                    faulthandler.dump_traceback(file=f)
            except Exception:
                # best-effort only
                pass


def print_progress(prefix: str, hb: Heartbeat, *, interval_sec: float = 10.0, stop_event: Optional[threading.Event] = None) -> threading.Thread:
    """Spawn a background printer that emits periodic progress to stdout."""
    def _loop():
        while not (stop_event and stop_event.is_set()):
            snap = hb.snapshot()
            print(f"{prefix} files={snap.files_scanned} groups={snap.groups_built} items={snap.items_processed} note={snap.notes}")
            time.sleep(interval_sec)
    t = threading.Thread(target=_loop, name="progress_printer", daemon=True)
    t.start()
    return t


