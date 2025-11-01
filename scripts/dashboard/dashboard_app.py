#!/usr/bin/env python3
"""
Unified Dashboard - Tabbed Interface
====================================
Combined dashboard with tabs for Current Project and Productivity views.

USAGE:
------
  python scripts/dashboard/run_dashboard.py [--port 5001] [--debug]

Access the dashboard:
  http://localhost:5001

Tabs:
  1. Current Project - Real-time project progress, process tracking
  2. Productivity - Historical analytics, cross-project metrics
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dashboard engines
from scripts.dashboard.engines.analytics import DashboardAnalytics
from scripts.dashboard.engines.data_engine import DashboardDataEngine
from scripts.dashboard.parsers.timesheet_parser import TimesheetParser


class UnifiedDashboard:
    """Unified dashboard with tabbed interface for Current Project and Productivity views."""

    def __init__(self, data_dir: str = "../.."):
        """Initialize unified dashboard with both view engines."""
        base = Path(data_dir).resolve()

        # Shared data engines
        self.data_engine = DashboardDataEngine(str(base))
        self.analytics = DashboardAnalytics(base)
        self.timesheet_parser = TimesheetParser(base / "data" / "timesheet.csv")

        # Flask app setup
        self.app = Flask(__name__, template_folder=".")
        self.app.config["JSON_SORT_KEYS"] = False
        self.app.config["TEMPLATES_AUTO_RELOAD"] = True

        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes for tabbed interface."""

        @self.app.route("/")
        def dashboard():
            """Main dashboard page with tabs."""
            return render_template_string(self._get_dashboard_template())

        # ===== Productivity Tab API Endpoints =====

        @self.app.route("/api/productivity/data/<time_slice>")
        def get_productivity_data(time_slice):
            """API endpoint for productivity analytics data."""
            lookback_days = request.args.get("lookback_days", 60, type=int)
            project_id = request.args.get("project_id", default=None, type=str)

            try:
                pid = (
                    project_id
                    if (project_id is not None and str(project_id).strip() != "")
                    else None
                )

                # Generate analytics response
                analytics_resp = self.analytics.generate_dashboard_response(
                    time_slice=time_slice, lookback_days=lookback_days, project_id=pid
                )

                # Attach artifact stats
                try:
                    data = self.data_engine.generate_dashboard_data(
                        time_slice=time_slice,
                        lookback_days=lookback_days,
                        project_id=pid,
                    )
                    analytics_resp["artifact_stats"] = data.get("artifact_stats", {})
                except Exception:
                    pass

                resp = jsonify(analytics_resp)
                resp.headers["Cache-Control"] = (
                    "no-store, no-cache, must-revalidate, max-age=0"
                )
                resp.headers["Pragma"] = "no-cache"
                return resp

            except Exception as e:
                import traceback

                print("=" * 70)
                print("ERROR in /api/productivity/data endpoint:")
                traceback.print_exc()
                print("=" * 70)
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/scripts")
        def get_scripts():
            """Get list of discovered scripts."""
            scripts = self.data_engine.discover_scripts()
            return jsonify({"scripts": scripts})

        @self.app.route("/api/script_updates", methods=["GET", "POST"])
        def handle_script_updates():
            """Handle script update tracking."""
            if request.method == "POST":
                data = request.get_json()
                self.data_engine.add_script_update(
                    script=data.get("script"),
                    description=data.get("description"),
                    date=data.get("date"),
                )
                return jsonify({"status": "success"})
            else:
                updates = self.data_engine.load_script_updates()
                return jsonify(updates.to_dict("records"))

        # ===== Current Project Tab API Endpoints =====

        @self.app.route("/api/current-project/progress")
        def get_current_project_progress():
            """API endpoint for current project progress data."""
            try:
                # Import current project logic
                from scripts.dashboard.current_project_dashboard_v2 import (
                    find_active_project,
                    get_directory_status,
                    load_timesheet_data,
                    get_project_file_operations,
                    match_timesheet_to_project,
                    compute_phase_metrics,
                    compute_phase_active_days,
                    compute_phase_hours_by_active_days,
                    build_historical_baseline,
                )

                # Find active project
                active_project = find_active_project()
                if not active_project:
                    return jsonify({"error": "No active project found"}), 404

                project_id = active_project.get("projectId")
                title = active_project.get("title") or project_id
                counts = active_project.get("counts") or {}
                total_images = int(counts.get("initialImages") or 0)
                total_groups = int(counts.get("groupCount") or 0)

                # Get real-time directory counts
                directory_status = get_directory_status(active_project)

                # Load timesheet
                timesheet_data = load_timesheet_data()

                # Preload operations
                all_ops = self.data_engine.load_file_operations()

                # Build historical baseline
                baseline = build_historical_baseline(
                    timesheet_data, self.data_engine, preloaded_ops=all_ops
                )

                # Get project operations
                ops = get_project_file_operations(
                    self.data_engine, active_project, preloaded_ops=all_ops
                )

                # Match with timesheet
                ts_project = match_timesheet_to_project(timesheet_data, project_id)
                total_billed_hours = (
                    ts_project.get("total_hours", 0) if ts_project else 0
                )

                # Compute per-phase metrics
                phases_current = {}
                phases = ["selection", "crop", "sort"]
                active_days_map = {p: compute_phase_active_days(ops, p) for p in phases}
                allocated_hours = (
                    compute_phase_hours_by_active_days(ts_project, active_days_map)
                    if ts_project
                    else {p: 0.0 for p in phases}
                )

                for phase in phases:
                    metrics = compute_phase_metrics(ops, phase)
                    # Override images with directory counts (ground truth)
                    if phase == "selection":
                        images = int(directory_status.get("__selected", 0))
                    elif phase == "crop":
                        images = int(directory_status.get("__cropped", 0))
                    elif phase == "sort":
                        images = int(directory_status.get("sort_images", 0))
                    else:
                        images = metrics.get("images", 0)

                    hours = allocated_hours.get(phase, 0.0)
                    rate = round(images / hours, 1) if hours > 0 else 0
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

                # Calculate overall progress
                selected_only = int(directory_status.get("__selected", 0)) + int(
                    directory_status.get("selected", 0)
                )
                cropped_count = int(directory_status.get("__cropped", 0)) + int(
                    directory_status.get("cropped", 0)
                )
                processed_images = selected_only + cropped_count
                percent_complete = (
                    round((processed_images / total_images) * 100, 1)
                    if total_images > 0
                    else 0
                )

                # Build response payload
                payload = {
                    "project": {
                        "id": project_id,
                        "title": title,
                        "totalImages": total_images,
                        "totalGroups": total_groups,
                        "processedImages": processed_images,
                        "percentComplete": percent_complete,
                        "totalBilledHours": total_billed_hours,
                        "directoryStatus": directory_status,
                    },
                    "phases": phases_current,
                    "baseline": baseline,
                }

                resp = jsonify(payload)
                resp.headers["Cache-Control"] = (
                    "no-store, no-cache, must-revalidate, max-age=0"
                )
                resp.headers["Pragma"] = "no-cache"
                return resp

            except Exception as e:
                import traceback

                print("=" * 70)
                print("ERROR in /api/current-project/progress endpoint:")
                traceback.print_exc()
                print("=" * 70)
                return jsonify({"error": str(e)}), 500

        # ===== Debug Endpoint =====

        @self.app.route("/api/debug")
        def debug_data():
            """Debug endpoint to see raw data structure."""
            try:
                raw_data = self.data_engine.generate_dashboard_data(
                    time_slice="D",
                    lookback_days=60,
                )
                return jsonify(raw_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _get_dashboard_template(self) -> str:
        """Return the HTML template for the tabbed dashboard."""
        # Read the new tabbed template
        template_path = Path(__file__).parent / "dashboard_tabbed_template.html"
        if template_path.exists():
            return template_path.read_text()
        else:
            # Fallback: create a basic tabbed template
            return self._create_tabbed_template()

    def _create_tabbed_template(self) -> str:
        """Create a basic tabbed dashboard template (fallback)."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dashboard - Image Workflow</title>
</head>
<body>
    <h1>Dashboard</h1>
    <p>Template file missing. Please ensure dashboard_tabbed_template.html exists.</p>
</body>
</html>
"""

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        debug: bool = False,
        auto_open: bool = True,
    ):
        """Run the Flask app."""
        if auto_open:
            # Launch browser after a short delay to let server start
            import threading
            import webbrowser

            def open_browser():
                import time

                time.sleep(1.5)  # Wait for server to be ready
                webbrowser.open(f"http://{host}:{port}")

            threading.Thread(target=open_browser, daemon=True).start()

        self.app.run(host=host, port=port, debug=debug, threaded=True)
