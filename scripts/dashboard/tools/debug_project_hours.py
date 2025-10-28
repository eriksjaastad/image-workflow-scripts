#!/usr/bin/env python3
"""
Debug Project Hours - Per-day Work Time Breakdown
=================================================
Print a per-day breakdown of work_time_minutes used by the dashboard for a given
project, so you can verify where "actual hours" come from.

Usage:
  python scripts/dashboard/tools/debug_project_hours.py mojo3
  python scripts/dashboard/tools/debug_project_hours.py mojo3 --days 60
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pathlib import Path as _P
from typing import Any, Dict, List

# Ensure project root is on sys.path so 'scripts' package can be imported
_ROOT = _P(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dashboard.project_metrics_aggregator import ProjectMetricsAggregator
from scripts.utils.companion_file_utils import get_file_operation_metrics


def _parse_iso_date(value: str) -> str | None:
    if not value:
        return None
    try:
        v = value
        if isinstance(v, str) and v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        return dt.date().isoformat()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-day work-time breakdown for a project"
    )
    parser.add_argument("project_id", help="Project ID (e.g., mojo3)")
    parser.add_argument("--data-dir", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback days for context (unused but reserved)",
    )
    args = parser.parse_args()

    base = Path(args.data_dir)
    agg = ProjectMetricsAggregator(base)

    # Aggregate once (this populates internal sources/helpers)
    results = agg.aggregate()
    proj = results.get(args.project_id)
    if not proj:
        print(
            f"Project '{args.project_id}' not found in metrics. Available: {', '.join(results.keys())}"
        )
        return

    totals = (proj or {}).get("totals", {})
    started_at = proj.get("startedAt")
    finished_at = proj.get("finishedAt")
    root_hint = (
        (proj.get("paths") or {}).get("root")
        if isinstance(proj.get("paths"), dict)
        else None
    )

    # Rebuild the exact file-ops list the aggregator used
    detailed_ops = list(agg._iter_file_operations())
    summary_ops = list(agg._iter_daily_summaries())

    # Deduplicate days where a summary exists
    summary_days: set[str] = set()
    for rec in summary_ops:
        day = _parse_iso_date(rec.get("timestamp_str") or rec.get("timestamp") or "")
        if day:
            summary_days.add(day)

    filtered_detailed: List[Dict[str, Any]] = []
    for rec in detailed_ops:
        day = _parse_iso_date(rec.get("timestamp") or rec.get("timestamp_str") or "")
        if day and day in summary_days:
            continue
        filtered_detailed.append(rec)

    file_ops = [*summary_ops, *filtered_detailed]

    # Match operations to this project by path and time window (same logic as aggregator)
    ops_by_path = agg._filter_ops_for_project(file_ops, root_hint or "")
    ops_by_time = agg._filter_ops_by_time_window(file_ops, started_at, finished_at)

    if ops_by_path:
        proj_ops = list(ops_by_path)
        for r in ops_by_time:
            if not (r.get("source_dir") or r.get("dest_dir") or r.get("working_dir")):
                proj_ops.append(r)
    else:
        proj_ops = list(ops_by_time)

    # Group by day and compute work_time_minutes per day
    by_day_ops: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in proj_ops:
        day = _parse_iso_date(rec.get("timestamp") or rec.get("timestamp_str") or "")
        if not day:
            continue
        by_day_ops[day].append(rec)

    # Compute metrics
    rows = []
    total_minutes = 0.0
    for day in sorted(by_day_ops.keys()):
        metrics = get_file_operation_metrics(by_day_ops[day])
        minutes = float(metrics.get("work_time_minutes") or 0.0)
        files = int(metrics.get("files_processed") or 0)
        total_minutes += minutes
        rows.append((day, minutes, files))

    # Print report
    print(f"\nProject: {args.project_id}")
    print(f"Manifest window: {started_at} → {finished_at}")
    print(f"Dashboard totals.work_hours: {totals.get('work_hours')}")
    print(f"Recomputed sum (hours): {round(total_minutes / 60.0, 2)}")
    print("\nTop contributing days (minutes, files):")
    for day, minutes, files in sorted(rows, key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {day}: {round(minutes, 1)} min ({files} files)")


if __name__ == "__main__":
    main()
