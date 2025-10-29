#!/usr/bin/env python3
"""
Current Project Dashboard V2 - Process-Centric View
===================================================
Tracks time spent per PROCESS (selection, crop, sort) and compares to historical averages.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run the dashboard server:
  python scripts/dashboard/current_project_dashboard_v2.py

Optional flags:
  --port 8082              # Specify port (default: 8082)
  --no-browser             # Don't auto-open browser
  --debug                  # Enable Flask debug mode

Access the dashboard:
  http://localhost:8082

Key Features:
-------------
- Integrates timesheet data for accurate billing hours
- Per-process breakdown: selection, crop, sort
- Current project vs historical baseline
- Real-time progress tracking
- Automatically opens in browser on launch
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, render_template_string

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.data_engine import DashboardDataEngine
from scripts.dashboard.timesheet_parser import TimesheetParser
from scripts.utils.companion_file_utils import launch_browser

DATA_DIR = PROJECT_ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"
TIMESHEET_PATH = DATA_DIR / "timesheet.csv"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp and normalize to naive UTC.

    - If tz-aware, convert to UTC and drop tzinfo
    - If naive, assume it's already UTC and return as-is
    """
    if not ts:
        return None
    try:
        v = (
            ts.replace("Z", "+00:00")
            if isinstance(ts, str) and ts.endswith("Z")
            else ts
        )
        dt = datetime.fromisoformat(v) if isinstance(v, str) else v
        if isinstance(dt, datetime) and getattr(dt, "tzinfo", None) is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def get_directory_status(project: Dict[str, Any]) -> Dict[str, Any]:
    """Count images in workflow directories to show real-time progress."""
    status = {
        "selected": 0,
        "__selected": 0,
        "__crop": 0,
        "__crop_auto": 0,
        "__cropped": 0,
        "content_dir": 0,
        "sort_images": 0,
    }

    paths = project.get("paths") or {}
    root = paths.get("root") or ""

    if not root:
        return status

    try:
        # Resolve relative paths from project root, and fallback to manifest directory
        project_root = Path.cwd()
        manifest_path_str = project.get("manifestPath") or ""
        manifest_dir = (
            Path(manifest_path_str).parent.resolve()
            if manifest_path_str
            else project_root
        )

        # Try multiple candidates for the content root
        root_candidates: List[Path] = []
        if root:
            try:
                # Candidate 1: as absolute or already-resolved path
                root_candidates.append(Path(root).resolve())
            except Exception:
                pass
            try:
                # Candidate 2: relative to project root
                if root.startswith("../../"):
                    root_candidates.append(
                        (project_root / root.lstrip("../../")).resolve()
                    )
                else:
                    root_candidates.append((project_root / root).resolve())
            except Exception:
                pass
            try:
                # Candidate 3: relative to manifest directory
                root_candidates.append((manifest_dir / root).resolve())
            except Exception:
                pass

        # First existing candidate wins
        root_path: Optional[Path] = None
        for cand in root_candidates:
            try:
                if cand and cand.exists():
                    root_path = cand
                    break
            except Exception:
                continue

        # Count remaining in content directory
        if root_path and root_path.exists():
            status["content_dir"] = len(list(root_path.glob("*.png")))

        # Standard workflow directories at project root
        for dir_key in ["__selected", "__crop", "__crop_auto", "__cropped"]:
            try:
                dir_path = (project_root / dir_key).resolve()
                if dir_path.exists():
                    count = len(list(dir_path.glob("*.png")))
                    status[dir_key] = count
            except Exception:
                continue

        # Count images in sort directories (character_group* and __character_group_*)
        try:
            sort_images = 0
            for child in project_root.iterdir():
                name = child.name.lower()
                if child.is_dir() and (
                    name.startswith("character_group")
                    or name.startswith("__character_group_")
                ):
                    sort_images += len(list(child.glob("*.png")))
            status["sort_images"] = sort_images
        except Exception:
            pass

        # Also check manifest-specified selectedDir
        selected_dir = paths.get("selectedDir")
        if selected_dir:
            try:
                selected_candidates: List[Path] = []
                try:
                    selected_candidates.append(
                        (project_root / selected_dir.lstrip("../../")).resolve()
                    )
                except Exception:
                    pass
                try:
                    selected_candidates.append((manifest_dir / selected_dir).resolve())
                except Exception:
                    pass
                for sc in selected_candidates:
                    try:
                        if sc.exists():
                            count = len(list(sc.glob("*.png")))
                            status["selected"] = count
                            break
                    except Exception:
                        continue
            except Exception:
                pass

    except Exception as e:
        logging.warning(f"Failed to count directory status: {e}")

    return status


def find_active_project() -> Optional[Dict[str, Any]]:
    """Find the active project (status='active' or no finishedAt)."""
    if not PROJECTS_DIR.exists():
        return None

    candidates: List[Tuple[datetime, Dict[str, Any]]] = []
    scanned = 0
    finished_count = 0
    not_finished_count = 0
    status_counts: Dict[str, int] = {}
    for manifest_file in PROJECTS_DIR.glob("*.project.json"):
        try:
            project_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        status = (project_data.get("status") or "").lower().strip()
        finished_at = project_data.get("finishedAt")
        started_at = project_data.get("startedAt")

        # Parse start date
        try:
            start_dt = _parse_iso(started_at) or datetime.fromtimestamp(
                manifest_file.stat().st_mtime
            )
        except Exception:
            start_dt = datetime.fromtimestamp(manifest_file.stat().st_mtime)

        # Track counts for diagnostics
        scanned += 1
        if finished_at in (None, "", []):
            not_finished_count += 1
        else:
            finished_count += 1
        status_counts[status] = status_counts.get(status, 0) + 1

        # Include any project without finishedAt (regardless of status)
        not_finished = finished_at in (None, "", [])

        if not_finished:
            project_data["manifestPath"] = str(manifest_file)
            candidates.append((start_dt, project_data))

    if not candidates:
        try:
            logging.info(
                f"No active project candidates. scanned={scanned} not_finished={not_finished_count} finished={finished_count} status_counts={status_counts}"
            )
        except Exception:
            pass
        return None

    # Return most recent
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_timesheet_data() -> Dict[str, Any]:
    """Load and parse timesheet CSV."""
    if not TIMESHEET_PATH.exists():
        return {"projects": [], "totals": {"total_hours": 0, "total_projects": 0}}

    parser = TimesheetParser(TIMESHEET_PATH)
    return parser.parse()


def get_project_file_operations(
    engine: DashboardDataEngine,
    project: Dict[str, Any],
    preloaded_ops: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Get all file operations for a specific project."""
    # Load all file operations (use preloaded if provided)
    all_ops = (
        preloaded_ops if preloaded_ops is not None else engine.load_file_operations()
    )

    # Filter to this project
    started_at = _parse_iso(project.get("startedAt"))
    finished_at = _parse_iso(project.get("finishedAt"))
    root_hint = (project.get("paths") or {}).get("root") or ""
    try:
        root_hint_resolved = str(Path(root_hint).resolve()) if root_hint else ""
    except Exception:
        root_hint_resolved = root_hint

    project_ops = []
    for op in all_ops:
        # Match by path
        matches_path = False
        if root_hint_resolved:
            for key in ("source_dir", "dest_dir", "working_dir"):
                v = str(op.get(key) or "")
                if not v:
                    continue
                try:
                    pv = Path(v).resolve()
                    pr = Path(root_hint_resolved)
                    try:
                        is_sub = pv.is_relative_to(pr)  # py3.9+ in pathlib
                    except AttributeError:
                        spr = str(pr)
                        sep = str(Path.sep)
                        spr = spr if spr.endswith(sep) else spr + sep
                        is_sub = str(pv).startswith(spr)
                except Exception:
                    is_sub = str(v).startswith(root_hint_resolved)
                if is_sub:
                    matches_path = True
                    break

        # Match by timestamp
        matches_time = False
        ts = op.get("timestamp") or op.get("timestamp_str")
        if ts:
            try:
                op_dt = _parse_iso(ts)
                if op_dt:
                    start_compare = started_at
                    end_compare = finished_at or datetime.utcnow()
                    if start_compare:
                        if start_compare <= op_dt <= end_compare:
                            matches_time = True
                    else:
                        if op_dt <= end_compare:
                            matches_time = True
            except Exception:
                pass

        if matches_path or matches_time:
            project_ops.append(op)

    return project_ops


def classify_operation_phase(op: Dict[str, Any]) -> str:
    """Classify a file operation into a phase: selection, crop, or sort."""
    operation = str(op.get("operation") or "").lower()
    dest_dir_raw = str(op.get("dest_dir") or "").strip()
    dest_base = Path(dest_dir_raw).name.lower() if dest_dir_raw else ""

    # Crop phase
    if operation == "crop":
        return "crop"

    # Fallback: infer crop from save/export that only contain PNGs
    files = op.get("files") or []
    if operation in {"save", "export"} and files:
        try:
            if all(str(f).lower().endswith(".png") for f in files):
                return "crop"
        except Exception:
            pass

    # Move operations
    if operation == "move":
        # Selection phase: moved to 'selected' or '__selected'
        if dest_base in {"selected", "__selected"}:
            return "selection"

        # Sort phase: moved to character_group_* or __character_group_*
        if dest_base.startswith("character_group") or dest_base.startswith(
            "__character_group_"
        ):
            return "sort"

    return "unknown"


def compute_phase_metrics(ops: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    """Compute metrics for a specific phase."""
    phase_ops = [op for op in ops if classify_operation_phase(op) == phase]

    if not phase_ops:
        return {"images": 0, "start_date": None, "end_date": None, "days_active": 0}

    # Count PNG images only
    total_images = 0
    dates = set()

    for op in phase_ops:
        # Count PNGs
        files = op.get("files") or []
        if files:
            png_count = sum(1 for f in files if str(f).lower().endswith(".png"))
            total_images += png_count
        else:
            # Fallback to file_count
            total_images += int(op.get("file_count") or 0)

        # Track dates
        ts = op.get("timestamp") or op.get("timestamp_str")
        if ts:
            try:
                dt = _parse_iso(ts)
                if dt:
                    dates.add(dt.date().isoformat())
            except Exception:
                pass

    sorted_dates = sorted(dates) if dates else []

    return {
        "images": total_images,
        "start_date": sorted_dates[0] if sorted_dates else None,
        "end_date": sorted_dates[-1] if sorted_dates else None,
        "days_active": len(dates),
    }


def compute_phase_active_days(ops: List[Dict[str, Any]], phase: str) -> Set[str]:
    """Return the set of ISO date strings where the given phase had activity.

    We derive phase-active days from operation timestamps. While timestamps can be
    batchy, at a day granularity they are good enough to detect overlaps. This
    function is used to apportion daily billed hours across overlapping phases.
    """
    active_days: Set[str] = set()
    for op in ops:
        if classify_operation_phase(op) != phase:
            continue
        ts = op.get("timestamp") or op.get("timestamp_str")
        if not ts:
            continue
        try:
            dt = _parse_iso(ts)
            if dt:
                active_days.add(dt.date().isoformat())
        except Exception:
            continue
    return active_days


def compute_phase_hours_by_active_days(
    timesheet_project: Dict[str, Any], active_days_map: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Allocate timesheet hours per day across phases active that day.

    - Convert timesheet daily hours keys (M/D/YYYY) to ISO dates for matching
    - For each day, split that day's hours evenly across all phases with activity
    - Ensures sum(phase_hours) <= total billed hours (no double-counting)
    """
    # Map ISO date -> hours from timesheet
    daily_hours_mdy: Dict[str, float] = timesheet_project.get("daily_hours", {})
    iso_to_hours: Dict[str, float] = {}
    for mdy, hours in daily_hours_mdy.items():
        try:
            # Parse M/D/YYYY
            month, day, year = [int(x) for x in mdy.split("/")]
            dt = datetime(year, month, day)
            iso_to_hours[dt.date().isoformat()] = float(hours or 0)
        except Exception:
            continue

    # Union of all active days
    all_active_days: Set[str] = set()
    for day_set in active_days_map.values():
        all_active_days.update(day_set)

    # Allocate hours per day across active phases
    allocated: Dict[str, float] = {"selection": 0.0, "crop": 0.0, "sort": 0.0}
    for day_iso in sorted(all_active_days):
        hours_for_day = float(iso_to_hours.get(day_iso, 0.0))
        if hours_for_day <= 0:
            continue
        phases_today = [p for p, days in active_days_map.items() if day_iso in days]
        if not phases_today:
            continue
        share = hours_for_day / len(phases_today)
        for p in phases_today:
            allocated[p] = allocated.get(p, 0.0) + share

    # Helpful diagnostics for overlaps
    try:
        overlap_days = [
            d
            for d in all_active_days
            if sum(1 for p in active_days_map if d in active_days_map[p]) > 1
        ]
        if overlap_days:
            logging.info(
                "Overlapping phase days: %s (allocated split per day)",
                ", ".join(overlap_days[:10])
                + ("..." if len(overlap_days) > 10 else ""),
            )
    except Exception:
        pass

    return allocated


def match_timesheet_to_project(
    timesheet_data: Dict[str, Any], project_id: str
) -> Optional[Dict[str, Any]]:
    """Find matching timesheet entry for a project."""
    # Normalize project ID for matching
    proj_normalized = (
        project_id.lower().replace(" ", "").replace("-", "").replace("_", "")
    )

    for ts_project in timesheet_data.get("projects", []):
        ts_name = ts_project["name"]
        ts_normalized = (
            ts_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        )

        # Exact match
        if ts_normalized == proj_normalized:
            return ts_project

        # Partial match (project ID contained in timesheet name)
        if proj_normalized in ts_normalized:
            # Check word boundaries
            idx = ts_normalized.find(proj_normalized)
            if idx != -1:
                before_ok = (idx == 0) or (
                    ts_normalized[idx - 1] in ["/", "_", "-", " "]
                )
                after_idx = idx + len(proj_normalized)
                after_ok = (after_idx >= len(ts_normalized)) or (
                    ts_normalized[after_idx] in ["/", "_", "-", " "]
                )

                if before_ok and after_ok:
                    return ts_project

    return None


def compute_phase_hours(
    timesheet_project: Dict[str, Any],
    phase_start: Optional[str],
    phase_end: Optional[str],
) -> float:
    """Compute billed hours for a specific phase based on date range."""
    if not phase_start or not phase_end:
        return 0.0

    # Get daily billed hours from timesheet
    daily_hours = timesheet_project.get("daily_hours", {})

    # Convert phase dates to timesheet date format (M/D/YYYY)
    try:
        start_dt = datetime.fromisoformat(phase_start)
        end_dt = datetime.fromisoformat(phase_end)
    except Exception:
        return 0.0

    total_hours = 0.0
    current_dt = start_dt

    while current_dt <= end_dt:
        # Cross-platform M/D/YYYY without leading zeros
        date_key = f"{current_dt.month}/{current_dt.day}/{current_dt.year}"

        hours_for_day = daily_hours.get(date_key, 0)
        total_hours += hours_for_day

        current_dt += timedelta(days=1)

    return total_hours


def compute_crop_daily_progression(
    ops: List[Dict[str, Any]],
    timesheet_project: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compute daily crop rate progression to show improvement over time.

    Returns list of daily stats: [{date, images, hours, rate}, ...]
    """
    # Filter to crop operations only
    crop_ops = [op for op in ops if classify_operation_phase(op) == "crop"]

    if not crop_ops:
        return []

    # Group images by date
    images_by_date: Dict[str, int] = {}
    for op in crop_ops:
        ts = op.get("timestamp") or op.get("timestamp_str")
        if not ts:
            continue

        try:
            dt = _parse_iso(ts)
            if not dt:
                continue

            date_key = dt.date().isoformat()

            # Count PNGs
            files = op.get("files") or []
            if files:
                png_count = sum(1 for f in files if str(f).lower().endswith(".png"))
                images_by_date[date_key] = images_by_date.get(date_key, 0) + png_count
            else:
                # Fallback to file_count
                images_by_date[date_key] = images_by_date.get(date_key, 0) + int(op.get("file_count") or 0)
        except Exception:
            continue

    # Get daily hours from timesheet if available
    daily_hours = timesheet_project.get("daily_hours", {}) if timesheet_project else {}

    # Build daily progression
    progression = []
    for date_str in sorted(images_by_date.keys()):
        images = images_by_date[date_str]

        # Convert to timesheet date format (M/D/YYYY)
        try:
            date_dt = datetime.fromisoformat(date_str)
            date_key = f"{date_dt.month}/{date_dt.day}/{date_dt.year}"
            hours = daily_hours.get(date_key, 0)
        except Exception:
            hours = 0

        # Calculate rate (only if we have hours)
        rate = round(images / hours, 1) if hours > 0 else 0

        progression.append({
            "date": date_str,
            "images": images,
            "hours": round(hours, 1),
            "rate": rate
        })

    return progression


def build_historical_baseline(
    timesheet_data: Dict[str, Any],
    engine: DashboardDataEngine,
    preloaded_ops: Optional[List[Dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-phase baseline from completed projects."""
    baselines = {
        "selection": {"total_hours": 0, "total_images": 0, "projects": 0},
        "crop": {"total_hours": 0, "total_images": 0, "projects": 0},
        "sort": {"total_hours": 0, "total_images": 0, "projects": 0},
    }

    # Find all completed projects
    all_manifests = list(PROJECTS_DIR.glob("*.project.json"))
    # Pre-count finished projects for progress reporting
    finished_total = 0
    for manifest_file in all_manifests:
        try:
            project = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Only include finished projects
        if not project.get("finishedAt"):
            continue

        finished_total += 1

    processed_count = 0

    for manifest_file in all_manifests:
        try:
            project = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not project.get("finishedAt"):
            continue

        project_id = project.get("projectId")
        if not project_id:
            continue

        # Match with timesheet
        ts_project = match_timesheet_to_project(timesheet_data, project_id)
        if not ts_project:
            processed_count += 1
            if progress_cb:
                progress_cb(processed_count, finished_total, f"skip:{project_id}")
            continue

        # Get file operations for this project
        ops = get_project_file_operations(engine, project, preloaded_ops=preloaded_ops)

        # Compute per-phase metrics
        for phase in ["selection", "crop", "sort"]:
            metrics = compute_phase_metrics(ops, phase)
            images = metrics["images"]

            if images > 0:
                # Estimate hours for this phase
                hours = compute_phase_hours(
                    ts_project, metrics["start_date"], metrics["end_date"]
                )

                if hours > 0:
                    baselines[phase]["total_hours"] += hours
                    baselines[phase]["total_images"] += images
                    baselines[phase]["projects"] += 1

        processed_count += 1
        if progress_cb:
            progress_cb(processed_count, finished_total, project_id)

    # Compute averages
    result = {}
    for phase, data in baselines.items():
        # Require at least 2 projects and non-zero totals to compute
        if (
            data["projects"] >= 2
            and data["total_hours"] > 0
            and data["total_images"] > 0
        ):
            result[phase] = {
                "avg_hours": round(data["total_hours"] / data["projects"], 1),
                "avg_images": round(data["total_images"] / data["projects"], 0),
                "avg_rate": round(data["total_images"] / data["total_hours"], 1),
                "projects_count": data["projects"],
            }
        else:
            result[phase] = {
                "avg_hours": 0,
                "avg_images": 0,
                "avg_rate": 0,
                "projects_count": data["projects"],
            }

    return result


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    engine = DashboardDataEngine(str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Simple in-process progress tracker
    progress_state: Dict[str, Any] = {
        "phase": "idle",
        "detail": "",
        "current": 0,
        "total": 0,
    }

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_TEMPLATE)

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/api/progress_status")
    def progress_status():
        return jsonify(progress_state)

    # Simple memo cache for progress responses
    _progress_cache: Dict[str, Any] = {"data": None, "ts": None}

    @app.route("/api/progress")
    def progress_api():
        from datetime import datetime

        # Serve cached response if within 5 minutes and not forced
        force = False
        try:
            from flask import request

            force = request.args.get("force") == "1"
        except Exception:
            force = False
        now = datetime.utcnow()
        if (
            (not force)
            and _progress_cache["data"] is not None
            and _progress_cache["ts"] is not None
        ):
            if (now - _progress_cache["ts"]).total_seconds() < 300:
                return jsonify(_progress_cache["data"])  # Cached
        progress_state.update(
            {"phase": "start", "detail": "locating project", "current": 0, "total": 0}
        )
        # Find active project
        active_project = find_active_project()
        if not active_project:
            progress_state.update(
                {"phase": "idle", "detail": "", "current": 0, "total": 0}
            )
            return jsonify({"error": "No active project found"}), 404

        project_id = active_project.get("projectId")
        title = active_project.get("title") or project_id
        counts = active_project.get("counts") or {}
        total_images = int(counts.get("initialImages") or 0)
        total_groups = int(counts.get("groupCount") or 0)

        # Get real-time directory counts (ground truth for current progress)
        directory_status = get_directory_status(active_project)
        logging.info(f"Directory status: {directory_status}")

        # Load timesheet
        timesheet_data = load_timesheet_data()

        # Preload operations ONCE to avoid multiple heavy loads per request
        progress_state.update(
            {
                "phase": "loading_ops",
                "detail": "Loading file operations",
                "current": 0,
                "total": 1,
            }
        )
        all_ops = engine.load_file_operations()
        progress_state.update(
            {
                "phase": "loading_ops",
                "detail": "Loaded file operations",
                "current": 1,
                "total": 1,
            }
        )

        # Build historical baseline
        progress_state.update(
            {
                "phase": "baseline",
                "detail": "Building historical baseline",
                "current": 0,
                "total": 0,
            }
        )

        def baseline_progress(cur: int, tot: int, pid: str):
            progress_state.update(
                {
                    "phase": "baseline",
                    "detail": f"{pid}",
                    "current": cur,
                    "total": max(tot, 1),
                }
            )

        baseline = build_historical_baseline(
            timesheet_data, engine, preloaded_ops=all_ops, progress_cb=baseline_progress
        )
        progress_state.update(
            {
                "phase": "baseline",
                "detail": "Baseline ready",
                "current": progress_state.get("current", 0),
                "total": progress_state.get("total", 1),
            }
        )

        # Get file operations for current project
        progress_state.update(
            {
                "phase": "filter_ops",
                "detail": "Filtering project operations",
                "current": 0,
                "total": 1,
            }
        )
        ops = get_project_file_operations(engine, active_project, preloaded_ops=all_ops)
        progress_state.update(
            {
                "phase": "filter_ops",
                "detail": "Project operations ready",
                "current": 1,
                "total": 1,
            }
        )

        # Fallback initialImages from unique PNGs in ops if missing/zero
        if total_images <= 0:
            from os.path import basename

            seen_pngs = set()
            for op in ops:
                files = op.get("files") or []
                for f in files:
                    try:
                        s = str(f)
                        if s.lower().endswith(".png"):
                            seen_pngs.add(basename(s).lower())
                    except Exception:
                        continue
            total_images = len(seen_pngs)

        # Match with timesheet
        ts_project = match_timesheet_to_project(timesheet_data, project_id)
        total_billed_hours = ts_project.get("total_hours", 0) if ts_project else 0

        # Compute per-phase metrics for current project
        phases_current = {}
        phases = ["selection", "crop", "sort"]
        progress_state.update(
            {
                "phase": "metrics",
                "detail": "Computing phase metrics",
                "current": 0,
                "total": len(phases),
            }
        )
        # Determine phase-active days from operations and apportion daily hours across overlaps
        active_days_map = {p: compute_phase_active_days(ops, p) for p in phases}
        allocated_hours = (
            compute_phase_hours_by_active_days(ts_project, active_days_map)
            if ts_project
            else {p: 0.0 for p in phases}
        )
        for idx, phase in enumerate(phases, start=1):
            metrics = compute_phase_metrics(ops, phase)
            # Override images for current project with directory counts (ground truth)
            if phase == "selection":
                images = int(directory_status.get("__selected", 0))
            elif phase == "crop":
                images = int(directory_status.get("__cropped", 0))
            elif phase == "sort":
                images = int(directory_status.get("sort_images", 0))
            else:
                images = metrics.get("images", 0)
            # Allocate billed hours for this phase based on phase-active days
            hours = allocated_hours.get(phase, 0.0)

            rate = round(images / hours, 1) if hours > 0 else 0

            # Compare to baseline
            baseline_rate = baseline.get(phase, {}).get("avg_rate", 0)
            vs_baseline = (
                round((rate / baseline_rate - 1) * 100, 1)
                if baseline_rate > 0 and rate >= 0
                else None
            )

            phases_current[phase] = {
                "images": images,
                "hours": round(hours, 1),
                "rate": rate,
                "start_date": metrics["start_date"],
                "end_date": metrics["end_date"],
                "days_active": metrics["days_active"],
                "baseline_rate": baseline_rate,
                "vs_baseline_pct": vs_baseline,
            }
            progress_state.update(
                {
                    "phase": "metrics",
                    "detail": f"Computed {phase}",
                    "current": idx,
                    "total": len(phases),
                }
            )

        # Overall progress based on directory counts (ground truth)
        ds = directory_status
        selected_only = int(ds.get("__selected", 0)) + int(ds.get("selected", 0))
        cropped_count = int(ds.get("__cropped", 0)) + int(ds.get("cropped", 0))
        remaining_to_crop = int(ds.get("__crop", 0)) + int(ds.get("__crop_auto", 0))
        total_completed = selected_only + cropped_count
        total_that_needs_processing = total_completed + remaining_to_crop
        total_processed = total_completed
        percent_complete = (
            round((total_completed / total_that_needs_processing) * 100, 1)
            if total_that_needs_processing > 0
            else 0
        )

        # Build historical timeline from timesheet
        historical_timeline: List[Dict[str, Any]] = []
        for ts_proj in timesheet_data.get("projects", []):
            try:
                historical_timeline.append(
                    {
                        "name": ts_proj.get("name"),
                        "rate": ts_proj.get("images_per_hour", 0) or 0,
                        "hours": ts_proj.get("total_hours", 0) or 0,
                    }
                )
            except Exception:
                continue

        # Compute daily crop progression to show rate improvement
        crop_daily_progression = compute_crop_daily_progression(ops, ts_project)

        # Final log line to confirm response isn't hanging
        logging.info(
            "/api/progress computed. images=%s processed=%s",
            total_images,
            total_processed,
        )
        progress_state.update({"phase": "idle", "detail": "", "current": 0, "total": 0})

        # Helpful logging
        root_hint = (active_project.get("paths") or {}).get("root")
        logging.info(f"Active project: id={project_id} title={title} root={root_hint}")
        logging.info(
            f"Initial images: {counts.get('initialImages')} (effective: {total_images})"
        )
        logging.info(f"Loaded file ops: {len(ops)}")
        try:
            total_alloc = sum(allocated_hours.get(p, 0.0) for p in phases)
            logging.info(
                f"Billed hours: total={total_billed_hours} allocated={round(total_alloc, 1)} by active-day split"
            )
        except Exception:
            pass
        for ph, data in phases_current.items():
            logging.info(
                f"Phase {ph}: images={data['images']} hours={data['hours']} rate={data['rate']} baseline_rate={data['baseline_rate']} vs={data['vs_baseline_pct']}%"
            )

        resp_data = {
            "project": {
                "projectId": project_id,
                "title": title,
                # Show current-phase total based on directory counts (more accurate for active work)
                "totalImages": total_that_needs_processing
                if total_that_needs_processing > 0
                else total_images,
                "totalGroups": total_groups,
                "processedImages": total_processed,
                "percentComplete": percent_complete,
                "totalBilledHours": total_billed_hours,
                "directoryStatus": directory_status,
            },
            "phases": phases_current,
            "baseline": baseline,
            "historical_timeline": historical_timeline,
            "crop_daily_progression": crop_daily_progression,
        }
        _progress_cache["data"] = resp_data
        _progress_cache["ts"] = now
        return jsonify(resp_data)

    return app


# Minimal dashboard template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Current Project Dashboard</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #101014;
            --surface: #181821;
            --accent: #4f9dff;
            --success: #51cf66;
            --warning: #ffd43b;
            --danger: #ff6b6b;
            --muted: #a0a3b1;
        }

        * { box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
            color: white;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2rem;
            margin: 0 0 10px 0;
            color: var(--accent);
        }

        .subtitle {
            color: var(--muted);
            font-size: 1.1rem;
        }

        .phases-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .phase-card {
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
            border: 2px solid rgba(255,255,255,0.1);
        }

        .phase-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0 0 20px 0;
            color: var(--accent);
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .metric-label {
            color: var(--muted);
        }

        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .vs-baseline {
            margin-top: 20px;
            padding: 15px;
            background: rgba(79, 157, 255, 0.1);
            border-radius: 8px;
        }

        .vs-baseline-label {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 5px;
        }

        .vs-baseline-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .ahead { color: var(--success); }
        .behind { color: var(--danger); }
        .ontrack { color: var(--warning); }

        .loading {
            text-align: center;
            color: var(--muted);
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 id="project-title">Loading...</h1>
            <div class="subtitle" id="project-subtitle"></div>
        </header>

        <div id="dashboard-content" class="loading">
            <div id="loading-status">Loading dashboard...</div>
            <div id="loading-detail" style="margin-top:6px;color:var(--muted);"></div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        let statusTimer = null;
        async function loadDashboard() {
            try {
                const response = await fetch('/api/progress');
                if (!response.ok) {
                    const msg = response.status === 404 ? 'No active project found' : `HTTP ${response.status}`;
                    document.getElementById('dashboard-content').innerHTML = `
                        <div class="error">
                            ${msg}
                            <div style="margin-top:10px;color:var(--muted);font-size:0.95rem;">
                                Checklist:
                                <ul>
                                    <li>Ensure there is an active project in data/projects/*.project.json</li>
                                    <li>Confirm data/timesheet.csv exists (optional)</li>
                                    <li>Verify file operation logs under data/file_operations_logs/</li>
                                </ul>
                            </div>
                        </div>`;
                    return;
                }
                const data = await response.json();

                if (data.error) {
                    document.getElementById('dashboard-content').innerHTML =
                        `<div class="error">${data.error}</div>`;
                    return;
                }

                // Update header
                document.getElementById('project-title').textContent = data.project.title;
                let subtitle = `${data.project.processedImages.toLocaleString()} / ${data.project.totalImages.toLocaleString()} images (${data.project.percentComplete}%)`;
                if (data.project.totalGroups > 0) {
                    subtitle += ` • ${data.project.totalGroups.toLocaleString()} groups`;
                }
                subtitle += ` • ${data.project.totalBilledHours}h billed`;
                document.getElementById('project-subtitle').textContent = subtitle;

                // Build phases grid
                const phasesHTML = ['selection', 'crop', 'sort'].map(phase => {
                    const phaseData = data.phases[phase];
                    const baseline = data.baseline && data.baseline[phase] ? data.baseline[phase] : { avg_rate: 0 };

                    const vsBaseline = (baseline.avg_rate && baseline.avg_rate > 0 && phaseData.rate > 0) ? phaseData.vs_baseline_pct : null;
                    const vsClass = vsBaseline === null ? '' : (vsBaseline > 10 ? 'ahead' : (vsBaseline < -10 ? 'behind' : 'ontrack'));
                    const vsLabel = vsBaseline === null ? '—' : (vsBaseline > 0 ? `+${vsBaseline}%` : `${vsBaseline}%`);
                    const baselineRateLabel = (baseline.avg_rate && baseline.avg_rate > 0) ? baseline.avg_rate : 'N/A';

                    return `
                        <div class="phase-card">
                            <h2 class="phase-title">${phase.charAt(0).toUpperCase() + phase.slice(1)}</h2>

                            <div class="metric-row">
                                <span class="metric-label">Images Processed</span>
                                <span class="metric-value">${phaseData.images.toLocaleString()}</span>
                            </div>

                            <div class="metric-row">
                                <span class="metric-label">Hours Spent</span>
                                <span class="metric-value">${phaseData.hours}h</span>
                            </div>

                            <div class="metric-row">
                                <span class="metric-label">Rate</span>
                                <span class="metric-value">${phaseData.rate} img/h</span>
                            </div>

                            <div class="metric-row">
                                <span class="metric-label">Days Active</span>
                                <span class="metric-value">${phaseData.days_active}</span>
                            </div>

                            <div class="vs-baseline">
                                <div class="vs-baseline-label">vs Historical Avg (${baselineRateLabel} img/h)</div>
                                <div class="vs-baseline-value ${vsClass}">${vsLabel}</div>
                            </div>
                        </div>
                    `;
                }).join('');

                // Build directory status card
                console.log('Directory status data:', data.project.directoryStatus);
                let dirStatusHTML = '';
                if (data.project.directoryStatus) {
                    const ds = data.project.directoryStatus;
                    console.log('Building directory status card with:', ds);
                    const contentRemaining = ds.content_dir || 0;
                    const selectedOnly = (ds.__selected || 0);  // No crop needed (double-underscore only)
                    const croppedCount = (ds.__cropped || 0);
                    
                    // Determine current phase based on actual directory contents
                    let currentPhase = 'Unknown';
                    if (contentRemaining > 3) {  // More than a few stray files
                        currentPhase = 'Phase 1: Selection (AI Reviewer)';
                    } else if (ds.__crop > 0 || ds.__crop_auto > 0) {
                        currentPhase = 'Phase 2: Cropping';
                    } else if (croppedCount > 0 || selectedOnly > 0) {
                        currentPhase = 'Phase 3: Sorting (or done)';
                    }
                    
                    // Calculate crop-only progress during Phase 2
                    const totalCompleted = croppedCount + selectedOnly; // overall completed (info only)
                    const remainingToCrop = (ds.__crop || 0) + (ds.__crop_auto || 0);
                    const imagesPerBatch = 3;
                    const remainingBatches = Math.ceil(remainingToCrop / imagesPerBatch);
                    const cropTotal = croppedCount + remainingToCrop;
                    const cropProgress = cropTotal > 0 ? 
                        (100 * croppedCount / cropTotal).toFixed(1) : 0;
                    
                    console.log('Calculated values:', {totalCompleted, remainingToCrop, cropProgress});
                    
                    dirStatusHTML = `
                        <div class="phase-card" style="margin-bottom: 30px;">
                            <h2 class="phase-title">Current Status</h2>
                            <div style="font-size: 1.2rem; color: var(--accent); margin-bottom: 20px;">
                                ${currentPhase}
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Content Dir (remaining)</span>
                                <span class="metric-value">${contentRemaining}</span>
                            </div>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <div style="color: var(--success); font-weight: 600; margin-bottom: 10px;">✓ Completed:</div>
                                <div class="metric-row">
                                    <span class="metric-label">__selected/ (no crop needed)</span>
                                    <span class="metric-value">${((ds.__selected || 0) + (ds.selected || 0)).toLocaleString()}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">__cropped/</span>
                                    <span class="metric-value">${croppedCount.toLocaleString()}</span>
                                </div>
                                <div class="metric-row" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                                    <span class="metric-label" style="font-weight: 600;">Total Completed:</span>
                                    <span class="metric-value" style="color: var(--success);">${totalCompleted.toLocaleString()}</span>
                                </div>
                            </div>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <div style="color: var(--accent); font-weight: 600; margin-bottom: 10px;">Remaining:</div>
                                <div class="metric-row">
                                    <span class="metric-label">__crop/ + __crop_auto/</span>
                                    <span class="metric-value">${remainingToCrop.toLocaleString()} (${remainingBatches.toLocaleString()} batches)</span>
                                </div>
                                ${cropTotal > 0 ? `
                                <div class="metric-row" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                                    <span class="metric-label" style="font-weight: 600;">Crop Progress:</span>
                                    <span class="metric-value">${cropProgress}%</span>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }
                
                let extra = '';

                // Crop Daily Progression Chart
                if (data.crop_daily_progression && data.crop_daily_progression.length > 0) {
                    extra += `
                    <h2 style="margin: 40px 0 20px 0; color: var(--accent);">Crop Rate Progression</h2>
                    <div class="chart-container" style="background: var(--surface); padding: 20px; border-radius: 12px;">
                        <canvas id="progressionChart"></canvas>
                    </div>`;
                }

                // Historical Productivity Chart
                if (data.historical_timeline) {
                    extra += `
                    <h2 style="margin: 40px 0 20px 0; color: var(--accent);">Historical Productivity</h2>
                    <div class="chart-container" style="background: var(--surface); padding: 20px; border-radius: 12px;">
                        <canvas id="historyChart"></canvas>
                    </div>`;
                }

                console.log('Directory status HTML length:', dirStatusHTML.length);
                console.log('Full HTML:', dirStatusHTML);
                
                const finalHTML = `${dirStatusHTML}<div class="phases-grid">${phasesHTML}</div>${extra}`;
                console.log('Setting dashboard content, length:', finalHTML.length);
                
                document.getElementById('dashboard-content').innerHTML = finalHTML;
                // Stop loader polling now that content is rendered
                const dc = document.getElementById('dashboard-content');
                if (dc && dc.classList.contains('loading')) {
                    dc.classList.remove('loading');
                }
                if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }

                // Render crop daily progression chart
                if (data.crop_daily_progression && data.crop_daily_progression.length > 0 && window.Chart) {
                    const progression = data.crop_daily_progression;
                    const progLabels = progression.map(d => {
                        // Format date as MM/DD
                        const date = new Date(d.date);
                        return `${date.getMonth() + 1}/${date.getDate()}`;
                    });
                    const progRates = progression.map(d => d.rate);
                    const progImages = progression.map(d => d.images);

                    // Get crop baseline for comparison line
                    const cropBaseline = (data.baseline && data.baseline.crop && data.baseline.crop.avg_rate > 0)
                        ? data.baseline.crop.avg_rate
                        : 0;

                    // Color-code points based on baseline
                    const progColors = progRates.map(r =>
                        r > cropBaseline * 1.1 ? '#51cf66' :  // Green: above baseline
                        r < cropBaseline * 0.9 ? '#ff6b6b' :  // Red: below baseline
                        '#4f9dff'  // Blue: on target
                    );

                    new Chart(document.getElementById('progressionChart'), {
                        type: 'line',
                        data: {
                            labels: progLabels,
                            datasets: [{
                                label: 'Your Rate (img/h)',
                                data: progRates,
                                borderColor: '#4f9dff',
                                backgroundColor: 'rgba(79, 157, 255, 0.1)',
                                pointBackgroundColor: progColors,
                                pointBorderColor: progColors,
                                pointRadius: 6,
                                pointHoverRadius: 8,
                                tension: 0.3,
                                fill: true
                            }, {
                                label: 'Baseline (126.2 img/h)',
                                data: Array(progLabels.length).fill(cropBaseline),
                                borderColor: '#ffd43b',
                                borderDash: [5, 5],
                                pointRadius: 0,
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    ticks: { color: '#a0a3b1' },
                                    title: { display: true, text: 'Images per Hour', color: 'white' },
                                    beginAtZero: false
                                },
                                x: {
                                    ticks: { color: '#a0a3b1' },
                                    title: { display: true, text: 'Date', color: 'white' }
                                }
                            },
                            plugins: {
                                legend: { labels: { color: 'white' } },
                                tooltip: {
                                    callbacks: {
                                        afterLabel: function(context) {
                                            const idx = context.dataIndex;
                                            const images = progImages[idx];
                                            const hours = progression[idx].hours;
                                            return `${images} images in ${hours}h`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }

                // Render historical chart
                if (data.historical_timeline) {
                    const labels = data.historical_timeline.map(p => p.name);
                    const rates = data.historical_timeline.map(p => p.rate);
                    // Use crop baseline (most relevant for productivity comparison)
                    // Fall back to selection if crop unavailable
                    const baseline = (data.baseline && data.baseline.crop && data.baseline.crop.avg_rate > 0)
                        ? data.baseline.crop.avg_rate
                        : (data.baseline && data.baseline.selection) ? data.baseline.selection.avg_rate : 0;
                    const colors = rates.map(r =>
                        r > baseline * 1.1 ? '#51cf66' :
                        r < baseline * 0.9 ? '#ff6b6b' : '#4f9dff'
                    );
                    if (window.Chart) {
                        new Chart(document.getElementById('historyChart'), {
                            type: 'bar',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Images per Hour',
                                    data: rates,
                                    backgroundColor: colors
                                }, {
                                    label: 'Baseline (Crop)',
                                    data: Array(labels.length).fill(baseline),
                                    type: 'line',
                                    borderColor: '#ffd43b',
                                    borderDash: [5, 5],
                                    pointRadius: 0
                                }]
                            },
                            options: {
                                responsive: true,
                                scales: {
                                    y: {
                                        ticks: { color: '#a0a3b1' },
                                        title: { display: true, text: 'img/h', color: 'white' }
                                    },
                                    x: { ticks: { color: '#a0a3b1', maxRotation: 45 } }
                                },
                                plugins: {
                                    legend: { labels: { color: 'white' } }
                                }
                            }
                        });
                    }
                }

            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('dashboard-content').innerHTML =
                    `<div class="error">Failed to load dashboard: ${error.message}</div>`;
            }
        }

        // Simple loader that polls /api/progress_status while dashboard loads
        async function pollStatus() {
            try {
                const r = await fetch('/api/progress_status');
                if (!r.ok) return;
                const s = await r.json();
                const map = {
                    start: 'Starting...',
                    loading_ops: 'Loading operations...',
                    baseline: 'Building baseline...',
                    filter_ops: 'Filtering project operations...',
                    metrics: 'Computing metrics...',
                    idle: 'Ready.'
                };
                document.getElementById('loading-status').textContent = map[s.phase] || 'Loading...';
                const detail = s.detail ? `${s.detail}` : '';
                const frac = (s.total && s.total > 0) ? ` (${s.current}/${s.total})` : '';
                document.getElementById('loading-detail').textContent = detail + frac;
            } catch (e) { /* noop */ }
        }

        // Load on page load (manual refresh only; no periodic auto-refresh)
        loadDashboard();
        // Poll status more frequently while loading
        statusTimer = setInterval(pollStatus, 400);
        // Stop status polling once content is rendered
        const observer = new MutationObserver(() => {
            const content = document.getElementById('dashboard-content');
            if (content && !content.classList.contains('loading')) {
                clearInterval(statusTimer);
                observer.disconnect();
            }
        });
        observer.observe(document.getElementById('dashboard-content'), { childList: true, subtree: true });
        // Removed periodic auto-refresh to prevent unnecessary load
    </script>
</body>
</html>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    host = args.host
    port = args.port

    # Auto-launch browser
    if not args.no_browser:
        threading.Thread(
            target=launch_browser, args=(host, port, 1.2), daemon=True
        ).start()

    print(f"🚀 Current Project Dashboard starting at http://{host}:{port}")
    print("📊 Process-centric view: Selection • Crop • Sort")
    print("📈 Current vs Historical Baseline")
    print("\nPress Ctrl+C to stop")

    # Avoid caching through proxies/CDNs just in case
    @app.after_request
    def add_no_store(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
