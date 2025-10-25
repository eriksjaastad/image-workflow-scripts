#!/usr/bin/env python3
"""
Current Project Dashboard V2 - Process-Centric View
===================================================
Tracks time spent per PROCESS (selection, crop, sort) and compares to historical averages.

Key Features:
- Integrates timesheet data for accurate billing hours
- Per-process breakdown: selection, crop, sort
- Current project vs historical baseline
- Real-time progress tracking

Serves on port 8082 by default
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import threading

from flask import Flask, jsonify, render_template_string

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.timesheet_parser import TimesheetParser
from scripts.dashboard.data_engine import DashboardDataEngine
from scripts.utils.companion_file_utils import launch_browser


DATA_DIR = PROJECT_ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"
TIMESHEET_PATH = DATA_DIR / "timesheet.csv"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp to datetime."""
    if not ts:
        return None
    try:
        v = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(v)
    except Exception:
        return None


def find_active_project() -> Optional[Dict[str, Any]]:
    """Find the active project (status='active' or no finishedAt)."""
    if not PROJECTS_DIR.exists():
        return None

    candidates: List[Tuple[datetime, Dict[str, Any]]] = []
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

        # Include if active status or no finish date
        is_active = status == "active"
        not_finished = finished_at in (None, "", [])

        if is_active or not_finished:
            project_data["manifestPath"] = str(manifest_file)
            candidates.append((start_dt, project_data))

    if not candidates:
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
    project: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Get all file operations for a specific project."""
    # Load all file operations
    all_ops = engine.load_file_operations()

    # Filter to this project
    started_at = _parse_iso(project.get("startedAt"))
    finished_at = _parse_iso(project.get("finishedAt"))
    root_hint = (project.get("paths") or {}).get("root") or ""

    project_ops = []
    for op in all_ops:
        # Match by path
        matches_path = False
        if root_hint:
            for key in ("source_dir", "dest_dir", "working_dir"):
                v = str(op.get(key) or "")
                if root_hint in v:
                    matches_path = True
                    break

        # Match by timestamp
        matches_time = False
        ts = op.get("timestamp") or op.get("timestamp_str")
        if ts:
            try:
                op_dt = _parse_iso(ts)
                if op_dt:
                    # Make naive for comparison
                    if op_dt.tzinfo:
                        op_dt = op_dt.replace(tzinfo=None)

                    start_compare = started_at.replace(tzinfo=None) if started_at else None
                    end_compare = finished_at.replace(tzinfo=None) if finished_at else datetime.now()
                    if end_compare.tzinfo:
                        end_compare = end_compare.replace(tzinfo=None)

                    if start_compare and start_compare <= op_dt <= end_compare:
                        matches_time = True
            except Exception:
                pass

        if matches_path or matches_time:
            project_ops.append(op)

    return project_ops


def classify_operation_phase(op: Dict[str, Any]) -> str:
    """Classify a file operation into a phase: selection, crop, or sort."""
    operation = str(op.get("operation") or "").lower()
    dest_dir = str(op.get("dest_dir") or "").lower().strip()

    # Crop phase
    if operation == "crop":
        return "crop"

    # Move operations
    if operation == "move":
        # Selection phase: moved to 'selected'
        if dest_dir == "selected":
            return "selection"

        # Sort phase: moved to character_group_* folders
        if dest_dir.startswith("character_group") or dest_dir.startswith("_"):
            if dest_dir not in {"_trash", "_tmp", "_temp"}:
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
        # Format as M/D/YYYY (no leading zeros)
        date_key = current_dt.strftime("%-m/%-d/%Y")  # macOS/Linux format

        hours_for_day = daily_hours.get(date_key, 0)
        total_hours += hours_for_day

        current_dt += timedelta(days=1)

    return total_hours


def build_historical_baseline(
    timesheet_data: Dict[str, Any],
    engine: DashboardDataEngine
) -> Dict[str, Dict[str, Any]]:
    """Build per-phase baseline from completed projects."""
    baselines = {
        "selection": {"total_hours": 0, "total_images": 0, "projects": 0},
        "crop": {"total_hours": 0, "total_images": 0, "projects": 0},
        "sort": {"total_hours": 0, "total_images": 0, "projects": 0}
    }

    # Find all completed projects
    for manifest_file in PROJECTS_DIR.glob("*.project.json"):
        try:
            project = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Only include finished projects
        if not project.get("finishedAt"):
            continue

        project_id = project.get("projectId")
        if not project_id:
            continue

        # Match with timesheet
        ts_project = match_timesheet_to_project(timesheet_data, project_id)
        if not ts_project:
            continue

        # Get file operations for this project
        ops = get_project_file_operations(engine, project)

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

    # Compute averages
    result = {}
    for phase, data in baselines.items():
        if data["projects"] > 0 and data["total_hours"] > 0:
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
                "projects_count": 0
            }

    return result


def create_app() -> Flask:
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    engine = DashboardDataEngine(str(PROJECT_ROOT))

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_TEMPLATE)

    @app.route("/api/progress")
    def progress_api():
        # Find active project
        active_project = find_active_project()
        if not active_project:
            return jsonify({"error": "No active project found"}), 404

        project_id = active_project.get("projectId")
        title = active_project.get("title") or project_id
        counts = active_project.get("counts") or {}
        total_images = int(counts.get("initialImages") or 0)

        # Load timesheet
        timesheet_data = load_timesheet_data()

        # Build historical baseline
        baseline = build_historical_baseline(timesheet_data, engine)

        # Get file operations for current project
        ops = get_project_file_operations(engine, active_project)

        # Match with timesheet
        ts_project = match_timesheet_to_project(timesheet_data, project_id)
        total_billed_hours = ts_project.get("total_hours", 0) if ts_project else 0

        # Compute per-phase metrics for current project
        phases_current = {}
        for phase in ["selection", "crop", "sort"]:
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
            vs_baseline = round((rate / baseline_rate - 1) * 100, 1) if baseline_rate > 0 else 0

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

        # Overall progress
        total_processed = sum(p["images"] for p in phases_current.values())
        percent_complete = round((total_processed / total_images) * 100, 1) if total_images > 0 else 0

        return jsonify({
            "project": {
                "projectId": project_id,
                "title": title,
                "totalImages": total_images,
                "processedImages": total_processed,
                "percentComplete": percent_complete,
                "totalBilledHours": total_billed_hours
            },
            "phases": phases_current,
            "baseline": baseline
        })

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
            Loading dashboard...
        </div>
    </div>

    <script>
        async function loadDashboard() {
            try {
                const response = await fetch('/api/progress');
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
                    const baseline = data.baseline[phase];

                    const vsBaseline = phaseData.vs_baseline_pct;
                    const vsClass = vsBaseline > 10 ? 'ahead' : (vsBaseline < -10 ? 'behind' : 'ontrack');
                    const vsLabel = vsBaseline > 0 ? `+${vsBaseline}%` : `${vsBaseline}%`;

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
                                <div class="vs-baseline-label">vs Historical Avg (${baseline.avg_rate} img/h)</div>
                                <div class="vs-baseline-value ${vsClass}">${vsLabel}</div>
                            </div>
                        </div>
                    `;
                }).join('');

                document.getElementById('dashboard-content').innerHTML =
                    `<div class="phases-grid">${phasesHTML}</div>`;

            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('dashboard-content').innerHTML =
                    `<div class="error">Failed to load dashboard: ${error.message}</div>`;
            }
        }

        // Load on page load
        loadDashboard();

        // Refresh every 30 seconds
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
"""


def main():
    app = create_app()
    host = "127.0.0.1"
    port = 8082

    # Auto-launch browser
    threading.Thread(target=launch_browser, args=(host, port, 1.2), daemon=True).start()

    print(f"🚀 Current Project Dashboard starting at http://{host}:{port}")
    print("📊 Process-centric view: Selection • Crop • Sort")
    print("📈 Current vs Historical Baseline")
    print("\nPress Ctrl+C to stop")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
