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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        v = ts.replace("Z", "+00:00") if isinstance(ts, str) and ts.endswith("Z") else ts
        dt = datetime.fromisoformat(v) if isinstance(v, str) else v
        if isinstance(dt, datetime) and getattr(dt, "tzinfo", None) is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


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
            start_dt = _parse_iso(started_at) or datetime.fromtimestamp(manifest_file.stat().st_mtime)
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
    preloaded_ops: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Get all file operations for a specific project."""
    # Load all file operations (use preloaded if provided)
    all_ops = preloaded_ops if preloaded_ops is not None else engine.load_file_operations()

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
        if dest_base.startswith("character_group") or dest_base.startswith("__character_group_"):
            return "sort"

    return "unknown"


def compute_phase_metrics(
    ops: List[Dict[str, Any]],
    phase: str
) -> Dict[str, Any]:
    """Compute metrics for a specific phase."""
    phase_ops = [op for op in ops if classify_operation_phase(op) == phase]

    if not phase_ops:
        return {
            "images": 0,
            "start_date": None,
            "end_date": None,
            "days_active": 0
        }

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
        "days_active": len(dates)
    }


def match_timesheet_to_project(
    timesheet_data: Dict[str, Any],
    project_id: str
) -> Optional[Dict[str, Any]]:
    """Find matching timesheet entry for a project."""
    # Normalize project ID for matching
    proj_normalized = project_id.lower().replace(" ", "").replace("-", "").replace("_", "")

    for ts_project in timesheet_data.get("projects", []):
        ts_name = ts_project["name"]
        ts_normalized = ts_name.lower().replace(" ", "").replace("-", "").replace("_", "")

        # Exact match
        if ts_normalized == proj_normalized:
            return ts_project

        # Partial match (project ID contained in timesheet name)
        if proj_normalized in ts_normalized:
            # Check word boundaries
            idx = ts_normalized.find(proj_normalized)
            if idx != -1:
                before_ok = (idx == 0) or (ts_normalized[idx-1] in ['/', '_', '-', ' '])
                after_idx = idx + len(proj_normalized)
                after_ok = (after_idx >= len(ts_normalized)) or (ts_normalized[after_idx] in ['/', '_', '-', ' '])

                if before_ok and after_ok:
                    return ts_project

    return None


def compute_phase_hours(
    timesheet_project: Dict[str, Any],
    phase_start: Optional[str],
    phase_end: Optional[str]
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
        "sort": {"total_hours": 0, "total_images": 0, "projects": 0}
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
                    ts_project,
                    metrics["start_date"],
                    metrics["end_date"]
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
        if data["projects"] >= 2 and data["total_hours"] > 0 and data["total_images"] > 0:
            result[phase] = {
                "avg_hours": round(data["total_hours"] / data["projects"], 1),
                "avg_images": round(data["total_images"] / data["projects"], 0),
                "avg_rate": round(data["total_images"] / data["total_hours"], 1),
                "projects_count": data["projects"]
            }
        else:
            result[phase] = {
                "avg_hours": 0,
                "avg_images": 0,
                "avg_rate": 0,
                "projects_count": data["projects"]
            }

    return result


def create_app() -> Flask:
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    engine = DashboardDataEngine(str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Simple in-process progress tracker
    progress_state: Dict[str, Any] = {
        "phase": "idle",
        "detail": "",
        "current": 0,
        "total": 0
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
            force = request.args.get('force') == '1'
        except Exception:
            force = False
        now = datetime.utcnow()
        if (not force) and _progress_cache["data"] is not None and _progress_cache["ts"] is not None:
            if (now - _progress_cache["ts"]).total_seconds() < 300:
                return jsonify(_progress_cache["data"])  # Cached
        progress_state.update({"phase": "start", "detail": "locating project", "current": 0, "total": 0})
        # Find active project
        active_project = find_active_project()
        if not active_project:
            progress_state.update({"phase": "idle", "detail": "", "current": 0, "total": 0})
            return jsonify({"error": "No active project found"}), 404

        project_id = active_project.get("projectId")
        title = active_project.get("title") or project_id
        counts = active_project.get("counts") or {}
        total_images = int(counts.get("initialImages") or 0)

        # Load timesheet
        timesheet_data = load_timesheet_data()

        # Preload operations ONCE to avoid multiple heavy loads per request
        progress_state.update({"phase": "loading_ops", "detail": "Loading file operations", "current": 0, "total": 1})
        all_ops = engine.load_file_operations()
        progress_state.update({"phase": "loading_ops", "detail": "Loaded file operations", "current": 1, "total": 1})

        # Build historical baseline
        progress_state.update({"phase": "baseline", "detail": "Building historical baseline", "current": 0, "total": 0})
        def baseline_progress(cur: int, tot: int, pid: str):
            progress_state.update({"phase": "baseline", "detail": f"{pid}", "current": cur, "total": max(tot, 1)})
        baseline = build_historical_baseline(timesheet_data, engine, preloaded_ops=all_ops, progress_cb=baseline_progress)
        progress_state.update({"phase": "baseline", "detail": "Baseline ready", "current": progress_state.get("current", 0), "total": progress_state.get("total", 1)})

        # Get file operations for current project
        progress_state.update({"phase": "filter_ops", "detail": "Filtering project operations", "current": 0, "total": 1})
        ops = get_project_file_operations(engine, active_project, preloaded_ops=all_ops)
        progress_state.update({"phase": "filter_ops", "detail": "Project operations ready", "current": 1, "total": 1})

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
        progress_state.update({"phase": "metrics", "detail": "Computing phase metrics", "current": 0, "total": len(phases)})
        for idx, phase in enumerate(phases, start=1):
            metrics = compute_phase_metrics(ops, phase)
            images = metrics["images"]

            # Estimate hours for this phase
            hours = 0
            if ts_project and metrics["start_date"] and metrics["end_date"]:
                hours = compute_phase_hours(
                    ts_project,
                    metrics["start_date"],
                    metrics["end_date"]
                )

            rate = round(images / hours, 1) if hours > 0 else 0

            # Compare to baseline
            baseline_rate = baseline.get(phase, {}).get("avg_rate", 0)
            vs_baseline = round((rate / baseline_rate - 1) * 100, 1) if baseline_rate > 0 and rate >= 0 else None

            phases_current[phase] = {
                "images": images,
                "hours": round(hours, 1),
                "rate": rate,
                "start_date": metrics["start_date"],
                "end_date": metrics["end_date"],
                "days_active": metrics["days_active"],
                "baseline_rate": baseline_rate,
                "vs_baseline_pct": vs_baseline
            }
            progress_state.update({"phase": "metrics", "detail": f"Computed {phase}", "current": idx, "total": len(phases)})

        # Overall progress
        total_processed = sum(p["images"] for p in phases_current.values())
        percent_complete = round((total_processed / total_images) * 100, 1) if total_images > 0 else 0

        # Build historical timeline from timesheet
        historical_timeline: List[Dict[str, Any]] = []
        for ts_proj in timesheet_data.get("projects", []):
            try:
                historical_timeline.append({
                    "name": ts_proj.get("name"),
                    "rate": ts_proj.get("images_per_hour", 0) or 0,
                    "hours": ts_proj.get("total_hours", 0) or 0
                })
            except Exception:
                continue

        # Final log line to confirm response isn't hanging
        logging.info("/api/progress computed. images=%s processed=%s", total_images, total_processed)
        progress_state.update({"phase": "idle", "detail": "", "current": 0, "total": 0})

        # Helpful logging
        root_hint = (active_project.get("paths") or {}).get("root")
        logging.info(f"Active project: id={project_id} title={title} root={root_hint}")
        logging.info(f"Initial images: {counts.get('initialImages')} (effective: {total_images})")
        logging.info(f"Loaded file ops: {len(ops)}")
        for ph, data in phases_current.items():
            logging.info(f"Phase {ph}: images={data['images']} hours={data['hours']} rate={data['rate']} baseline_rate={data['baseline_rate']} vs={data['vs_baseline_pct']}%")

        resp_data = {
            "project": {
                "projectId": project_id,
                "title": title,
                "totalImages": total_images,
                "processedImages": total_processed,
                "percentComplete": percent_complete,
                "totalBilledHours": total_billed_hours
            },
            "phases": phases_current,
            "baseline": baseline,
            "historical_timeline": historical_timeline
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
                document.getElementById('project-subtitle').textContent =
                    `${data.project.processedImages.toLocaleString()} / ${data.project.totalImages.toLocaleString()} images (${data.project.percentComplete}%) • ${data.project.totalBilledHours}h billed`;

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

                let extra = '';
                if (data.historical_timeline) {
                    extra = `
                    <h2 style="margin: 40px 0 20px 0; color: var(--accent);">Historical Productivity</h2>
                    <div class="chart-container" style="background: var(--surface); padding: 20px; border-radius: 12px;">
                        <canvas id="historyChart"></canvas>
                    </div>`;
                }

                document.getElementById('dashboard-content').innerHTML =
                    `<div class="phases-grid">${phasesHTML}</div>${extra}`;
                // Stop loader polling now that content is rendered
                const dc = document.getElementById('dashboard-content');
                if (dc && dc.classList.contains('loading')) {
                    dc.classList.remove('loading');
                }
                if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }

                // Render historical chart
                if (data.historical_timeline) {
                    const labels = data.historical_timeline.map(p => p.name);
                    const rates = data.historical_timeline.map(p => p.rate);
                    const baselineSel = (data.baseline && data.baseline.selection) ? data.baseline.selection.avg_rate : 0;
                    const colors = rates.map(r =>
                        r > baselineSel * 1.1 ? '#51cf66' :
                        r < baselineSel * 0.9 ? '#ff6b6b' : '#4f9dff'
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
                                    label: 'Baseline',
                                    data: Array(labels.length).fill(baselineSel),
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
        threading.Thread(target=launch_browser, args=(host, port, 1.2), daemon=True).start()

    print(f"🚀 Current Project Dashboard starting at http://{host}:{port}")
    print("📊 Process-centric view: Selection • Crop • Sort")
    print("📈 Current vs Historical Baseline")
    print("\nPress Ctrl+C to stop")

    # Avoid caching through proxies/CDNs just in case
    @app.after_request
    def add_no_store(resp):
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
