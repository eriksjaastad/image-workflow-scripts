#!/usr/bin/env python3
"""
Lightweight profiling/instrumentation utilities for the reducer experiments.

Features (Investigation Mode):
- Stage timers: scan, group, embed, dedupe, select, moves
- Counters: n_files, n_groups_total, n_groups_shard, avg_group_size, pairs_examined, pairs_pruned
- Throughput estimates (files/min, groups/min) at summary time
- Periodic checkpoints: append JSONL every N seconds to runs/<id>/checkpoint.jsonl
- Exit summary: write runs/<id>/prof_summary.json

Safety: sandbox-only output under provided sandbox_root and run_id.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class StageTiming:
    total_sec: float = 0.0
    started_at: float | None = None


class Profiler:
    def __init__(
        self, sandbox_root: Path, run_id: str, checkpoint_interval_sec: float = 300.0
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.run_id = run_id
        self.checkpoint_interval_sec = max(5.0, float(checkpoint_interval_sec))

        self._stages: dict[str, StageTiming] = {
            "scan": StageTiming(),
            "group": StageTiming(),
            "embed": StageTiming(),
            "dedupe": StageTiming(),
            "select": StageTiming(),
            "moves": StageTiming(),
        }
        self._counters: dict[str, float] = {
            "n_files": 0,
            "n_groups_total": 0,
            "n_groups_shard": 0,
            "avg_group_size": 0.0,
            "pairs_examined": 0,
            "pairs_pruned": 0,
        }

        self._stop_event = threading.Event()
        self._checkpoint_thread: threading.Thread | None = None
        (self.sandbox_root / "runs" / self.run_id).mkdir(parents=True, exist_ok=True)

    # ------------------------------ stages ------------------------------
    def start_stage(self, name: str) -> None:
        st = self._stages.get(name)
        if st is None:
            st = StageTiming()
            self._stages[name] = st
        if st.started_at is None:
            st.started_at = time.time()

    def end_stage(self, name: str) -> None:
        st = self._stages.get(name)
        if not st or st.started_at is None:
            return
        st.total_sec += max(0.0, time.time() - st.started_at)
        st.started_at = None

    def add_duration(self, name: str, seconds: float) -> None:
        st = self._stages.get(name)
        if st is None:
            st = StageTiming()
            self._stages[name] = st
        st.total_sec += max(0.0, float(seconds))

    # ------------------------------ counters ------------------------------
    def set_counter(self, name: str, value: float) -> None:
        self._counters[name] = float(value)

    def inc_counter(self, name: str, delta: float = 1.0) -> None:
        self._counters[name] = float(self._counters.get(name, 0.0) + delta)

    # ------------------------------ checkpoints ------------------------------
    def _checkpoint_payload(self, hb_snapshot: Any | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time": time.time(),
            "stages": {k: asdict(v) for k, v in self._stages.items()},
            "counters": dict(self._counters),
        }
        if hb_snapshot is not None:
            # hb_snapshot is a dataclass from watchdog. Convert with asdict-like behavior.
            try:
                payload["heartbeat"] = {
                    "timestamp_utc": getattr(hb_snapshot, "timestamp_utc", None),
                    "files_scanned": getattr(hb_snapshot, "files_scanned", None),
                    "groups_built": getattr(hb_snapshot, "groups_built", None),
                    "items_processed": getattr(hb_snapshot, "items_processed", None),
                    "notes": getattr(hb_snapshot, "notes", None),
                }
            except Exception:
                payload["heartbeat"] = None
        return payload

    def start_checkpoints(self, hb) -> None:
        ckpt_path = self.sandbox_root / "runs" / self.run_id / "checkpoint.jsonl"

        def _loop():
            # First write happens after interval to avoid too-frequent I/O at start
            while not self._stop_event.wait(self.checkpoint_interval_sec):
                try:
                    snap = None
                    if hb is not None:
                        try:
                            snap = hb.snapshot()
                        except Exception:
                            snap = None
                    payload = self._checkpoint_payload(snap)
                    with ckpt_path.open("a") as f:
                        f.write(json.dumps(payload) + "\n")
                except Exception:
                    # best-effort only; continue
                    pass

        self._checkpoint_thread = threading.Thread(
            target=_loop, name="prof_checkpoints", daemon=True
        )
        self._checkpoint_thread.start()

    def stop_checkpoints(self) -> None:
        self._stop_event.set()
        if self._checkpoint_thread is not None:
            try:
                self._checkpoint_thread.join(timeout=self.checkpoint_interval_sec * 2)
            except Exception:
                pass

    # ------------------------------ summary ------------------------------
    def _throughput_estimates(self) -> dict[str, Any]:
        # files/min: based on scan stage if available, fallback to total wall (sum of all stages)
        scan_s = self._stages.get("scan", StageTiming()).total_sec
        group_s = self._stages.get("group", StageTiming()).total_sec
        total_wall = sum(st.total_sec for st in self._stages.values())
        n_files = self._counters.get("n_files", 0.0)
        n_groups_shard = self._counters.get("n_groups_shard", 0.0)

        def rate(count: float, seconds: float) -> float:
            return (count / (seconds / 60.0)) if seconds and seconds > 0 else 0.0

        return {
            "files_per_min_scan": rate(n_files, scan_s),
            "groups_per_min_group": rate(n_groups_shard, group_s),
            "files_per_min_total": rate(n_files, total_wall),
            "groups_per_min_total": rate(n_groups_shard, total_wall),
        }

    def write_summary(self) -> Path:
        out_path = self.sandbox_root / "runs" / self.run_id / "prof_summary.json"
        summary = {
            "stages": {k: asdict(v) for k, v in self._stages.items()},
            "counters": dict(self._counters),
            "throughput": self._throughput_estimates(),
            "written_at": time.time(),
        }
        out_path.write_text(json.dumps(summary, indent=2))
        return out_path

    def finish(self) -> None:
        # End any open stages
        for name, st in self._stages.items():
            if st.started_at is not None:
                self.end_stage(name)
