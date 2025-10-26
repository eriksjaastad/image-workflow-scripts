#!/usr/bin/env python3
"""
File Operations Report by Project Window
=======================================

Purpose:
- Explain seemingly overlapping counts (e.g., stage-delete vs delete) by
  scoping to a project’s active window.
- Summarize FileTracker file operations within a project’s start→finish.

Data sources:
- Project manifests: data/projects/*.project.json (startedAt, finishedAt, projectId)
- FileTracker logs: data/file_operations_logs/*.log (JSON lines)

Notes:
- Overlap is expected: an image may be moved to delete_staging (stage delete),
  then later sent to trash (delete). That’s two distinct operations.
- This tool counts operations, not unique files. De-dup across stages is
  non-trivial because large ops often omit explicit file lists for performance.

Usage:
  python scripts/tools/report_file_ops_by_project.py \
    --project mojo3 \
    --write-report

  # Or list projects to choose from:
  python scripts/tools/report_file_ops_by_project.py --list-projects
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECTS_DIR = Path("data") / "projects"
LOGS_DIR = Path("data") / "file_operations_logs"
SAFE_REPORTS_DIR = Path("data") / "daily_summaries"


@dataclass
class ProjectWindow:
    project_id: str
    started_at: datetime
    finished_at: Optional[datetime]  # None means active


@dataclass
class OpsSummary:
    total_entries: int
    by_operation: Dict[str, int]
    moves_to_delete_staging: int
    moves_to_cropped: int
    deletes: int
    other_moves: int
    window_start: str
    window_end: str


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        v = ts
        if isinstance(v, str) and v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.fromisoformat(v)
    except Exception:
        return None


def load_projects() -> List[ProjectWindow]:
    projects: List[ProjectWindow] = []
    if not PROJECTS_DIR.exists():
        return projects

    for pf in PROJECTS_DIR.glob("*.project.json"):
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = str(data.get("projectId") or "unknown")
        started_at = parse_iso(str(data.get("startedAt") or "")) or parse_iso(str(data.get("createdAt") or ""))
        finished_at = parse_iso(str(data.get("finishedAt") or "")) if data.get("finishedAt") else None
        if not started_at:
            # Skip if no start
            continue
        projects.append(ProjectWindow(project_id=pid, started_at=started_at, finished_at=finished_at))
    return projects


def pick_project(projects: List[ProjectWindow], project_id: str) -> Optional[ProjectWindow]:
    for p in projects:
        if p.project_id == project_id:
            return p
    return None


def summarize_ops(window: ProjectWindow) -> OpsSummary:
    total_entries = 0
    by_operation: Dict[str, int] = {}
    moves_to_delete_staging = 0
    moves_to_cropped = 0
    deletes = 0
    other_moves = 0

    start = window.started_at
    end = window.finished_at or datetime.now(timezone.utc)

    if not LOGS_DIR.exists():
        return OpsSummary(
            total_entries=0,
            by_operation={},
            moves_to_delete_staging=0,
            moves_to_cropped=0,
            deletes=0,
            other_moves=0,
            window_start=start.isoformat(),
            window_end=end.isoformat(),
        )

    for log_path in sorted(LOGS_DIR.glob("*.log")):
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    ts_str = entry.get("timestamp")
                    dt = parse_iso(ts_str) if ts_str else None
                    if not dt or dt < start or dt > end:
                        continue

                    if str(entry.get("type") or "").lower() != "file_operation":
                        continue

                    total_entries += 1
                    op = (entry.get("operation") or "").lower()
                    by_operation[op] = by_operation.get(op, 0) + 1

                    dest_dir = str(entry.get("dest_dir") or "").lower().strip()
                    if op == "move":
                        if "delete" in dest_dir:
                            moves_to_delete_staging += 1
                        elif "cropped" in dest_dir:
                            moves_to_cropped += 1
                        else:
                            other_moves += 1
                    elif op == "delete":
                        deletes += 1
        except Exception:
            continue

    return OpsSummary(
        total_entries=total_entries,
        by_operation=by_operation,
        moves_to_delete_staging=moves_to_delete_staging,
        moves_to_cropped=moves_to_cropped,
        deletes=deletes,
        other_moves=other_moves,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
    )


def print_summary(window: ProjectWindow, summary: OpsSummary) -> None:
    print("\n" + "=" * 80)
    print(f"FILE OPERATIONS FOR PROJECT: {window.project_id}")
    print("=" * 80 + "\n")
    print(f"Window: {summary.window_start} → {summary.window_end}")
    print(f"Total file_operation entries: {summary.total_entries:,}\n")
    print("By operation:")
    for op, cnt in sorted(summary.by_operation.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {op}: {cnt:,}")
    print()
    print("Breakdown (heuristics):")
    print(f"  Move → delete_staging: {summary.moves_to_delete_staging:,}")
    print(f"  Move → cropped:        {summary.moves_to_cropped:,}")
    print(f"  Delete (send to trash): {summary.deletes:,}")
    print(f"  Other moves:            {summary.other_moves:,}")
    print()
    print("Note: counts reflect operations (pipeline stages), not unique files.")
    print("An image moved to delete_staging and later trashed will count twice.")


def write_report(window: ProjectWindow, summary: OpsSummary) -> Path:
    out = {
        "project_id": window.project_id,
        "window": {
            "start": summary.window_start,
            "end": summary.window_end,
        },
        "ops": asdict(summary),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    SAFE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAFE_REPORTS_DIR / f"file_ops_{window.project_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize FileTracker file operations by project window.")
    parser.add_argument("--project", help="Project ID to report (e.g., mojo3)")
    parser.add_argument("--list-projects", action="store_true", help="List known projects and exit")
    parser.add_argument("--write-report", action="store_true", help="Write JSON report to data/daily_summaries/")
    args = parser.parse_args(argv)

    projects = load_projects()
    if args.list_projects or not args.project:
        print("Available projects:")
        for p in sorted(projects, key=lambda x: x.started_at):
            end = p.finished_at.isoformat() if p.finished_at else "(active)"
            print(f"  - {p.project_id}: {p.started_at.isoformat()} → {end}")
        if not args.project:
            return 0

    window = pick_project(projects, args.project)
    if not window:
        print(f"Project not found: {args.project}")
        return 1

    summary = summarize_ops(window)
    print_summary(window, summary)

    if args.write_report:
        out = write_report(window, summary)
        print(f"\nReport written: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


