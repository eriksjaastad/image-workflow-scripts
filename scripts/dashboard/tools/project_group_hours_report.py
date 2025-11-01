#!/usr/bin/env python3
"""
Project Group Hours Report
==========================
Compute per-project image group counts, join with hours (from timesheet.csv), and export a simple timeline.

- Safe by design: read-only for project/image data; writes NEW report files to safe zones only
- Sources:
  - data/projects/*.project.json (project manifests)
  - timesheet.csv (at repo root data/timesheet.csv)
  - character groups live under paths.characterGroups in manifests
- Outputs:
  - data/daily_summaries/project_group_hours_report.json
  - data/daily_summaries/project_group_hours_report.csv

Group Count Definition
----------------------
A "group" is defined by unique base timestamp stem (e.g., 20250725_035504) across images and their stages.
We count unique stems across all character_group directories listed in the manifest for each project.

Usage:
  python scripts/dashboard/tools/project_group_hours_report.py
  python scripts/dashboard/tools/project_group_hours_report.py --format csv
  python scripts/dashboard/tools/project_group_hours_report.py --projects mojo1 mojo2

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Ensure project root import path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.companion_file_utils import extract_timestamp_from_filename

SAFE_JSON_PATH = _ROOT / "data" / "daily_summaries" / "project_group_hours_report.json"
SAFE_CSV_PATH = _ROOT / "data" / "daily_summaries" / "project_group_hours_report.csv"
PROJECTS_DIR = _ROOT / "data" / "projects"
TIMESHEET_CSV = _ROOT / "data" / "timesheet.csv"


@dataclass
class ProjectSummary:
    project_id: str
    title: str
    status: str | None
    started_at: str | None
    finished_at: str | None
    root_path: str | None
    group_count: int
    timesheet_hours: float
    hours_per_group: float | None
    timeline_days: list[str]


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _iter_project_manifests(
    project_filter: set[str] | None = None,
) -> Iterable[dict]:
    if not PROJECTS_DIR.exists():
        return []
    for mf in sorted(PROJECTS_DIR.glob("*.project.json")):
        pj = _read_json(mf)
        if not pj:
            continue
        pid = str(pj.get("projectId") or pj.get("project_id") or "").strip()
        if project_filter and pid and pid not in project_filter:
            continue
        # Normalize shape
        pj.setdefault("paths", {})
        yield pj


def _collect_group_stems(dir_paths: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in dir_paths:
        try:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.rglob("*.png"):
                ts = extract_timestamp_from_filename(p.name)
                if ts:
                    stems.add(ts)
        except Exception:
            # best-effort
            continue
    return stems


def _parse_timesheet_hours() -> dict[str, float]:
    """Return total hours per projectId from data/timesheet.csv.

    Expected columns include a project name/id column (4th index per sample). We do a
    forgiving parse: strip, lower, and normalize underscores. Blank project rows are ignored.
    """
    totals: dict[str, float] = defaultdict(float)
    if not TIMESHEET_CSV.exists():
        return totals
    try:
        with open(TIMESHEET_CSV) as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip aggregate/footer rows heuristically: if duration empty but hours in a single summary cell, ignore
                if not row or len(row) < 5:
                    continue
                # Columns in sample: date, start, end, duration, project, hours, initialImages, finalImages, note
                proj = (row[4] or "").strip()
                hrs_raw = (row[5] or "").strip()
                if not proj or not hrs_raw:
                    continue
                try:
                    hours = float(hrs_raw)
                except ValueError:
                    continue
                # Normalize projectId-ish key: lower, replace spaces and dashes with underscores
                key = proj.strip().lower().replace(" ", "_").replace("-", "_")
                totals[key] += hours
    except Exception:
        return totals
    return totals


def _collect_timeline_days_for_project(project: dict) -> list[str]:
    """Approximate timeline days using startedAt/finishedAt from manifest.
    Falls back to timesheet (if we later wire per-day), but for now just a date span list.
    """
    started = str(project.get("startedAt") or project.get("started_at") or "").strip()
    finished = str(
        project.get("finishedAt") or project.get("finished_at") or ""
    ).strip()
    try:

        def _parse_iso(v: str) -> datetime | None:
            if not v:
                return None
            v2 = v[:-1] + "+00:00" if v.endswith("Z") else v
            try:
                dt = datetime.fromisoformat(v2)
                return dt
            except Exception:
                return None

        sdt = _parse_iso(started)
        fdt = _parse_iso(finished)
        if not sdt or not fdt or fdt < sdt:
            return []
        days = []
        cur = sdt.date()
        end = fdt.date()
        while cur <= end:
            days.append(cur.isoformat())
            cur = cur.fromordinal(cur.toordinal() + 1)
        return days
    except Exception:
        return []


def build_report(project_ids: list[str] | None = None) -> list[ProjectSummary]:
    project_filter = set(project_ids) if project_ids else None
    timesheet_hours = _parse_timesheet_hours()

    results: list[ProjectSummary] = []
    for pj in _iter_project_manifests(project_filter):
        pid = str(pj.get("projectId") or "").strip()
        title = str(pj.get("title") or pid)
        status = pj.get("status")
        started_at = pj.get("startedAt")
        finished_at = pj.get("finishedAt")
        root_path = (pj.get("paths") or {}).get("root")
        character_groups = (pj.get("paths") or {}).get("characterGroups") or []

        # Resolve character group directories relative to manifest location
        manifest_path = PROJECTS_DIR / f"{pid}.project.json"
        group_dirs: list[Path] = []
        for rel in character_groups:
            try:
                group_dirs.append((manifest_path.parent / rel).resolve())
            except Exception:
                continue

        stems = _collect_group_stems(group_dirs)
        group_count = len(stems)

        # Timesheet hours: normalize key like we normalized earlier
        t_key = pid.strip().lower().replace(" ", "_").replace("-", "_")
        hours = float(timesheet_hours.get(t_key, 0.0))
        hours_per_group = (
            round(hours / group_count, 4) if group_count > 0 and hours > 0 else None
        )

        timeline_days = _collect_timeline_days_for_project(pj)

        results.append(
            ProjectSummary(
                project_id=pid,
                title=title,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                root_path=root_path,
                group_count=group_count,
                timesheet_hours=hours,
                hours_per_group=hours_per_group,
                timeline_days=timeline_days,
            )
        )

    return results


def write_json(results: list[ProjectSummary], path: Path = SAFE_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(results: list[ProjectSummary], path: Path = SAFE_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project_id",
        "title",
        "status",
        "started_at",
        "finished_at",
        "root_path",
        "group_count",
        "timesheet_hours",
        "hours_per_group",
        "timeline_days_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "project_id": r.project_id,
                    "title": r.title,
                    "status": r.status or "",
                    "started_at": r.started_at or "",
                    "finished_at": r.finished_at or "",
                    "root_path": r.root_path or "",
                    "group_count": r.group_count,
                    "timesheet_hours": r.timesheet_hours,
                    "hours_per_group": r.hours_per_group
                    if r.hours_per_group is not None
                    else "",
                    "timeline_days_count": len(r.timeline_days),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project group counts and hours per group"
    )
    parser.add_argument("--format", choices=["json", "csv", "both"], default="both")
    parser.add_argument(
        "--projects", nargs="*", help="Optional list of projectIds to include"
    )
    args = parser.parse_args()

    results = build_report(project_ids=args.projects)

    fmt = args.format
    if fmt in ("both", "json"):
        write_json(results)
        print(f"Wrote JSON: {SAFE_JSON_PATH}")
    if fmt in ("both", "csv"):
        write_csv(results)
        print(f"Wrote CSV: {SAFE_CSV_PATH}")

    # Quick stdout summary
    for r in results:
        print(
            f"{r.project_id}: groups={r.group_count}, hours={r.timesheet_hours}, h/group={r.hours_per_group}"
        )


if __name__ == "__main__":
    main()
