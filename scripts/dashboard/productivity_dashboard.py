#!/usr/bin/env python3
"""
Productivity Dashboard - Web Interface
======================================
Flask-based web dashboard for visualizing workflow productivity data.

Features:
- Global and individual graph time controls
- Bar charts with historical average overlays
- Script update markers with hover descriptions
- Modular design for easy script additions
- Dark theme matching existing tools
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
from flask import Flask, render_template, jsonify, request
from data_engine import DashboardDataEngine
from analytics import DashboardAnalytics

class ProductivityDashboard:
    def __init__(self, data_dir: str = "../.."):
        # Honor provided data_dir for testability; resolve relative paths
        base = Path(data_dir).resolve()
        self.data_engine = DashboardDataEngine(str(base))
        # Analytics engine should use the same base
        self.analytics = DashboardAnalytics(base)
        # Set template folder to current directory
        self.app = Flask(__name__, template_folder='.')
        # Preserve dict order in JSON responses (don't sort keys)
        self.app.config['JSON_SORT_KEYS'] = False
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route("/")
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard_template.html')
        
        @self.app.route("/api/data/<time_slice>")
        def get_dashboard_data(time_slice):
            """API endpoint for dashboard data"""
            lookback_days = request.args.get('lookback_days', 60, type=int)  # 60 days to show archived projects
            project_id = request.args.get('project_id', default=None, type=str)
            
            try:
                # Treat empty project_id as None to avoid downstream issues
                pid = project_id if (project_id is not None and str(project_id).strip() != '') else None
                data = self.data_engine.generate_dashboard_data(
                    time_slice=time_slice, 
                    lookback_days=lookback_days,
                    project_id=pid
                )
                
                # Build complete analytics response (single source of truth for table and charts)
                analytics_resp = self.analytics.generate_dashboard_response(
                    time_slice=time_slice,
                    lookback_days=lookback_days,
                    project_id=pid
                )
                chart_data = analytics_resp
                # Attach artifact stats to API payload for UI consumption
                try:
                    chart_data['artifact_stats'] = data.get('artifact_stats', {})
                except Exception:
                    pass
                # Debug provenance marker and no-cache headers
                try:
                    chart_data['metadata']['table_source'] = 'analytics'
                except Exception:
                    pass
                resp = jsonify(chart_data)
                resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                resp.headers['Pragma'] = 'no-cache'
                return resp
            except Exception as e:
                # Log full traceback for debugging
                import traceback
                print("="*70)
                print("ERROR in /api/data endpoint:")
                traceback.print_exc()
                print("="*70)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/scripts")
        def get_scripts():
            """Get list of discovered scripts"""
            scripts = self.data_engine.discover_scripts()
            return jsonify({"scripts": scripts})
        
        @self.app.route("/api/script_updates", methods=["GET", "POST"])
        def handle_script_updates():
            """Handle script update tracking"""
            if request.method == "POST":
                data = request.get_json()
                self.data_engine.add_script_update(
                    script=data.get('script'),
                    description=data.get('description'),
                    date=data.get('date')
                )
                return jsonify({"status": "success"})
            else:
                updates = self.data_engine.load_script_updates()
                return jsonify(updates.to_dict('records'))
        
        @self.app.route("/api/debug")
        def debug_data():
            """Debug endpoint to see raw data structure"""
            try:
                raw_data = self.data_engine.generate_dashboard_data(
                    time_slice='D', 
                    lookback_days=60  # 60 days to show archived projects
                )
                return jsonify(raw_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def transform_for_charts(self, data):
        """Transform raw data into Chart.js format"""
        chart_data = {
            "metadata": data["metadata"],
            "charts": {}
        }
        
        # Script name mapping for display
        script_display_names = {
            'image_version_selector': '01_web_image_selector',
            'character_sorter': '03_web_character_sorter',
            'batch_crop_tool': '04_batch_crop_tool'
        }
        
        # Transform file operations by script
        if data['file_operations_data'].get('by_script'):
            by_script_data = {}
            for record in data['file_operations_data']['by_script']:
                script = record['script']
                display_name = script_display_names.get(script, script)
                date = record['time_slice']
                count = record['file_count']
                
                if display_name not in by_script_data:
                    by_script_data[display_name] = {'dates': [], 'counts': []}
                
                by_script_data[display_name]['dates'].append(date)
                by_script_data[display_name]['counts'].append(count)
            
            # Sort dates and counts for each script to ensure chronological order
            for script_name in by_script_data:
                dates_counts = sorted(zip(by_script_data[script_name]['dates'], 
                                         by_script_data[script_name]['counts']))
                by_script_data[script_name]['dates'] = [d for d, c in dates_counts]
                by_script_data[script_name]['counts'] = [c for d, c in dates_counts]
            
            chart_data['charts']['by_script'] = by_script_data
        
        # Transform file operations by type
        if data['file_operations_data'].get('by_operation'):
            by_operation_data = {}
            for record in data['file_operations_data']['by_operation']:
                operation = record['operation']
                date = record['time_slice']
                count = record['file_count']
                
                if operation not in by_operation_data:
                    by_operation_data[operation] = {'dates': [], 'counts': []}
                
                by_operation_data[operation]['dates'].append(date)
                by_operation_data[operation]['counts'].append(count)
            
            # Sort dates and counts for each operation to ensure chronological order
            for operation_name in by_operation_data:
                dates_counts = sorted(zip(by_operation_data[operation_name]['dates'], 
                                         by_operation_data[operation_name]['counts']))
                by_operation_data[operation_name]['dates'] = [d for d, c in dates_counts]
                by_operation_data[operation_name]['counts'] = [c for d, c in dates_counts]
            
            chart_data['charts']['by_operation'] = by_operation_data
        
        # Project comparison payload (overall and per-tool IPH)
        try:
            pm = data.get('project_metrics', {}) or {}
            comparisons = []
            for pid, rec in pm.items():
                title = rec.get('title') or pid
                iph = float((rec.get('throughput') or {}).get('images_per_hour') or 0)
                base = rec.get('baseline') or {}
                overall_base = float(base.get('overall_iph_baseline') or 0)
                per_tool_base = base.get('per_tool') or {}
                tools = {}
                for tool, stats in (rec.get('tools') or {}).items():
                    tools[tool] = {
                        'iph': float(stats.get('images_per_hour') or 0),
                        'baseline': float(per_tool_base.get(tool) or 0),
                        'images_processed': int(stats.get('images_processed') or 0),
                        'work_time_minutes': float(stats.get('work_time_minutes') or 0)
                    }
                
                # NEW: Add crop rate data from SQLite v3 database
                crop_rate = self._get_crop_rate_for_project(pid)
                
                comparisons.append({
                    'projectId': pid,
                    'title': title,
                    'iph': iph,
                    'baseline_overall': overall_base,
                    'tools': tools,
                    'startedAt': rec.get('startedAt'),
                    'finishedAt': rec.get('finishedAt'),
                    'total_operations': rec.get('totals', {}).get('total_images_processed', 0),
                    'crop_rate': crop_rate  # NEW: Crop vs approve stats
                })
            # sort by startedAt if available
            def _key(x):
                return (x.get('startedAt') or '')
            chart_data['project_comparisons'] = sorted(comparisons, key=_key)
        except Exception:
            chart_data['project_comparisons'] = []

        # Add project productivity table data
        chart_data['project_productivity_table'] = self._build_project_productivity_table(data)

        return chart_data
    
    def _get_crop_rate_for_project(self, project_id: str) -> dict | None:
        """Get crop vs approve statistics from SQLite v3 database."""
        import sqlite3
        from pathlib import Path
        
        db_path = Path(self.data_dir) / 'data' / 'training' / 'ai_training_decisions' / f'{project_id}.db'
        if not db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN user_action = 'crop' THEN 1 ELSE 0 END) as cropped,
                    SUM(CASE WHEN user_action = 'approve' THEN 1 ELSE 0 END) as approved
                FROM ai_decisions
            """)
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] > 0:
                total, cropped, approved = row
                return {
                    'total_decisions': total,
                    'cropped': cropped,
                    'approved_no_crop': approved,
                    'crop_percentage': (cropped / total * 100) if total > 0 else 0,
                    'approve_percentage': (approved / total * 100) if total > 0 else 0
                }
        except Exception as e:
            print(f"[!] Error loading crop rate for {project_id}: {e}")
            return None
        
        return None
    
    def _build_project_productivity_table(self, data):
        """Build project productivity table data"""
        projects = data.get('projects', [])
        project_metrics = data.get('project_metrics', {})
        
        table_data = []
        
        for project in projects:
            project_id = project.get('projectId')
            if not project_id:
                continue
            
            # Get project metrics
            pm = project_metrics.get(project_id, {})
            
            # Start images (from manifest counts)
            counts = project.get('counts', {})
            start_images = counts.get('initialImages', 0)
            
            # End images (from manifest if finished, else from metrics)
            end_images = counts.get('finalImages') or pm.get('totals', {}).get('images_processed', 0)
            
            # Build per-tool breakdown for THIS project
            tools_breakdown = self._build_tools_breakdown_for_project(
                project_id,
                project,
                pm,
                data
            )
            
            table_data.append(OrderedDict([
                ("projectId", project_id),
                ("title", project.get('title') or project_id),
                ("start_images", start_images),
                ("end_images", end_images),
                ("tools", tools_breakdown)
            ]))
        
        return table_data
    
    def _build_tools_breakdown_for_project(self, project_id, project, pm, data):
        """Build per-tool breakdown for a specific project"""
        # Define allowed tools in the desired order
        allowed_tools_order = [
            'Web Image Selector',
            'Web Character Sorter',
            'Multi Crop Tool'
        ]
        
        # Build window from server-provided baseline labels
        meta = data.get('metadata', {})
        cur_slice = meta.get('time_slice')
        baseline = (meta.get('baseline_labels', {}) or {}).get(cur_slice, [])
        start_lbl = baseline[0] if baseline else None
        end_lbl = baseline[-1] if baseline else None
        def _to_yyyymmdd(lbl):
            if not isinstance(lbl, str):
                return None
            if 'T' in lbl:
                lbl = lbl.split('T')[0]
            return lbl.replace('-', '')
        start_date = _to_yyyymmdd(start_lbl)
        end_date = _to_yyyymmdd(end_lbl)

        # Load ALL file operations for lifetime computations
        try:
            window_ops = self.data_engine.load_file_operations()
        except Exception:
            window_ops = []
        
        # Filter to project by date range (startedAt to finishedAt)
        started_at = project.get('startedAt')
        finished_at = project.get('finishedAt')
        
        def _belongs(rec):
            """Filter file operations to this project's date range."""
            if not started_at:
                return False  # Can't match without start date
            
            # Get operation timestamp
            ts = rec.get('timestamp') or rec.get('timestamp_str')
            if not ts:
                return False
            
            try:
                # Parse operation timestamp (naive local time - no timezone info)
                if isinstance(ts, str):
                    # Remove Z if present, but DON'T add timezone offset
                    ts = ts.replace('Z', '')
                    op_dt = datetime.fromisoformat(ts)
                else:
                    op_dt = ts  # Already datetime
                
                # Make sure op_dt is naive (no timezone)
                if op_dt.tzinfo is not None:
                    op_dt = op_dt.replace(tzinfo=None)
                
                # Parse project dates and convert to naive datetime (drop timezone)
                start_str = started_at.replace('Z', '') if isinstance(started_at, str) else started_at
                start_dt = datetime.fromisoformat(start_str)
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.replace(tzinfo=None)
                
                # Check if operation is after project start
                if op_dt < start_dt:
                    return False
                
                # Check if operation is before project end (if project is finished)
                if finished_at:
                    end_str = finished_at.replace('Z', '') if isinstance(finished_at, str) else finished_at
                    end_dt = datetime.fromisoformat(end_str)
                    if end_dt.tzinfo is not None:
                        end_dt = end_dt.replace(tzinfo=None)
                    if op_dt > end_dt:
                        return False
                
                return True
            except Exception:
                return False
        
        proj_ops = [r for r in window_ops if _belongs(r)]

        # Group by display tool name
        grouped = {}
        for r in proj_ops:
            disp = self._get_display_name(r.get('script') or '')
            if disp not in allowed_tools_order:
                continue
            grouped.setdefault(disp, []).append(r)

        temp_breakdown = {}
        from datetime import datetime as _dt
        for disp in allowed_tools_order:
            recs = grouped.get(disp, [])
            if not recs:
                # Fallback to project metrics if no activity in window
                # Find matching tool stats by raw key
                tool_stats = None
                for raw, stats in (pm.get('tools', {}) or {}).items():
                    if self._get_display_name(raw) == disp:
                        tool_stats = stats
                        break
                if not tool_stats:
                    continue
                images = int(tool_stats.get('images_processed', 0) or 0)
                iph = float(tool_stats.get('images_per_hour', 0) or 0)
                hours = round(images / iph) if iph > 0 else 0
                days = self._count_active_days_for_tool(data, project_id, raw)
                if disp == "Web Image Selector":
                    by_dest = (pm.get('totals', {}) or {}).get('operations_by_dest', {})
                    move = by_dest.get('move', {}) if isinstance(by_dest, dict) else {}
                    selected = int(move.get('selected', 0) or 0)
                    cropped = int(move.get('crop', 0) or 0)
                    temp_breakdown[disp] = {
                        "hours": hours,
                        "days": days,
                        "images_total": images,
                        "images_selected": selected,
                        "images_cropped": cropped
                    }
                else:
                    temp_breakdown[disp] = {
                        "hours": hours,
                        "days": days,
                        "images": images
                    }
                continue

            # Hours from file-ops timing
            try:
                metrics = self.data_engine.calculate_file_operation_work_time(recs)
                hours = round(float(metrics.get('work_time_minutes') or 0.0) / 60.0, 1)
            except Exception:
                hours = 0
            # Unique active days
            days_set = set()
            for rec in recs:
                ts = rec.get('timestamp') or rec.get('timestamp_str')
                if not ts:
                    continue
                try:
                    v = ts
                    # Handle datetime objects directly (already parsed by load_file_operations)
                    if isinstance(v, _dt):
                        d = v.date().isoformat()
                    elif isinstance(v, str):
                        # Remove Z and parse string timestamps
                        v = v.replace('Z', '')
                        d = _dt.fromisoformat(v).date().isoformat()
                    else:
                        continue
                    
                    days_set.add(d)
                except Exception:
                    continue
            days = len(days_set)
            images = sum(int(r.get('file_count') or 0) for r in recs)

            if disp == "Web Image Selector":
                selected = sum(int(r.get('file_count') or 0) for r in recs if (r.get('operation') == 'move' and str(r.get('dest_dir') or '').lower() in {'selected','__selected'}))
                cropped = sum(int(r.get('file_count') or 0) for r in recs if (r.get('operation') == 'move' and any(k in str(r.get('dest_dir') or '').lower() for k in ['crop','__crop','__crop_auto','crop_auto'])))
                temp_breakdown[disp] = {
                    "hours": hours,
                    "days": days,
                    "images_total": images,
                    "images_selected": selected,
                    "images_cropped": cropped
                }
            else:
                temp_breakdown[disp] = {
                    "hours": hours,
                    "days": days,
                    "images": images
                }
        
        # Return tools as a list of tuples to preserve order (will be converted to dict in frontend)
        tools_breakdown_list = []
        for tool_name in allowed_tools_order:
            if tool_name in temp_breakdown:
                tools_breakdown_list.append([tool_name, temp_breakdown[tool_name]])
        
        return tools_breakdown_list
    
    def _get_display_name(self, tool_name):
        """Convert tool name to display name"""
        display_names = {
            'image_version_selector': 'Web Image Selector',
            'character_sorter': 'Web Character Sorter',  # Renamed from "Character Sorter"
            'batch_crop_tool': 'Multi Crop Tool',
            'multi_crop_tool': 'Multi Crop Tool',
            'recursive_file_mover': 'Recursive File Mover',
            'test_web_selector': 'Test Web Selector',
            'web_character_sorter': 'Web Character Sorter'
        }
        return display_names.get(tool_name, tool_name)
    
    def _count_active_days_for_tool(self, data, project_id, tool_name):
        """Count unique days a tool was active for a specific project"""
        # Find the project to get its date range
        projects = data.get("projects", [])
        project = next((p for p in projects if p.get("projectId") == project_id), None)
        if not project:
            return 0
        
        started_at = project.get('startedAt')
        finished_at = project.get('finishedAt')
        if not started_at:
            return 0
        
        # Load all file operations and filter by date range and tool
        try:
            all_ops = self.data_engine.load_file_operations()
        except Exception:
            return 0
        
        unique_dates = set()
        
        for rec in all_ops:
            # Match tool name
            rec_script = rec.get("script", "")
            if self._get_display_name(rec_script) != self._get_display_name(tool_name):
                continue
            
            # Filter by project date range
            ts = rec.get('timestamp') or rec.get('timestamp_str')
            if not ts:
                continue
            
            try:
                # Parse operation timestamp (naive local time)
                if isinstance(ts, datetime):
                    op_dt = ts
                elif isinstance(ts, str):
                    ts = ts.replace('Z', '')
                    op_dt = datetime.fromisoformat(ts)
                else:
                    continue
                
                # Make sure op_dt is naive
                if op_dt.tzinfo is not None:
                    op_dt = op_dt.replace(tzinfo=None)
                
                # Parse project dates and convert to naive datetime
                start_str = started_at.replace('Z', '') if isinstance(started_at, str) else started_at
                start_dt = datetime.fromisoformat(start_str)
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.replace(tzinfo=None)
                
                # Check if operation is after project start
                if op_dt < start_dt:
                    continue
                
                # Check if operation is before project end (if project is finished)
                if finished_at:
                    end_str = finished_at.replace('Z', '') if isinstance(finished_at, str) else finished_at
                    end_dt = datetime.fromisoformat(end_str)
                    if end_dt.tzinfo is not None:
                        end_dt = end_dt.replace(tzinfo=None)
                    if op_dt > end_dt:
                        continue
                
                # Add the date (not datetime) to the set
                unique_dates.add(op_dt.date().isoformat())
            except:
                continue
        
        return len(unique_dates)
    
    def run(self, host="127.0.0.1", port=5001, debug=False):
        """Run the dashboard server"""
        import webbrowser
        import threading
        import time
        
        url = f"http://{host}:{port}"
        print(f"🚀 Productivity Dashboard starting at {url}")
        
        # Auto-open browser after a short delay (like your other scripts)
        def open_browser():
            time.sleep(1.5)  # Give server time to start
            try:
                webbrowser.open(url)
                print(f"🌐 Opening browser to {url}")
            except Exception as e:
                print(f"Could not auto-open browser: {e}")
        
        # Start browser opener in background thread
        if not debug:  # Don't auto-open in debug mode
            threading.Thread(target=open_browser, daemon=True).start()
        
        self.app.run(host=host, port=port, debug=debug)

# Dashboard HTML Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Productivity Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            color-scheme: dark;
            --bg: #101014;
            --surface: #181821;
            --surface-alt: #1f1f2c;
            --accent: #4f9dff;
            --accent-soft: rgba(79, 157, 255, 0.2);
            --success: #51cf66;
            --danger: #ff6b6b;
            --warning: #ffd43b;
            --muted: #a0a3b1;
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: white;
            min-height: 100vh;
        }
        
        .dashboard-header {
            background: var(--surface);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .dashboard-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
            margin: 0;
        }
        
        .global-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .time-selector {
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.1);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .time-selector:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .dashboard-content {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-container {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin: 0;
        }
        
        .chart-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .chart-time-btn {
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.1);
            color: var(--muted);
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .chart-time-btn:hover {
            background: var(--accent-soft);
            color: white;
        }
        
        .chart-time-btn.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        
        .chart-canvas {
            height: 300px;
            margin-top: 1rem;
        }
        
        .loading {
            text-align: center;
            color: var(--muted);
            padding: 2rem;
        }
        
        .error {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }
        
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--surface);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: var(--muted);
            font-size: 0.9rem;
        }
        
        .update-marker {
            position: absolute;
            width: 2px;
            background: var(--warning);
            top: 0;
            bottom: 0;
            cursor: pointer;
        }
        
        .update-tooltip {
            position: absolute;
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            display: none;
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1 class="dashboard-title">📊 Productivity Dashboard</h1>
        <div class="global-controls">
            <label for="global-time">Global Time Scale:</label>
            <select id="global-time" class="time-selector">
                <option value="">Individual Controls</option>
                <option value="15min">15 Minutes</option>
                <option value="1H">1 Hour</option>
                <option value="D">Daily</option>
                <option value="W">Weekly</option>
                <option value="M">Monthly</option>
            </select>
            <select id="lookback-days" class="time-selector">
                <option value="7">Last 7 days</option>
                <option value="30" selected>Last 30 days</option>
                <option value="90">Last 90 days</option>
                <option value="365">Last year</option>
            </select>
        </div>
    </div>
    
    <div class="dashboard-content">
        <div class="stats-summary" id="stats-summary">
            <div class="stat-card">
                <div class="stat-value" id="total-active-time">--</div>
                <div class="stat-label">Total Active Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-files">--</div>
                <div class="stat-label">Files Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-efficiency">--</div>
                <div class="stat-label">Average Efficiency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="active-scripts">--</div>
                <div class="stat-label">Active Scripts</div>
            </div>
        </div>
        
        <div class="charts-grid" id="charts-container">
            <div class="loading">Loading dashboard data...</div>
        </div>
    </div>

    <script>
        // Dashboard state
        let dashboardData = null;
        let charts = {};
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            setupGlobalControls();
            loadDashboardData('D'); // Default to daily view
        });
        
        function setupGlobalControls() {
            const globalTimeSelect = document.getElementById('global-time');
            const lookbackSelect = document.getElementById('lookback-days');
            
            globalTimeSelect.addEventListener('change', function() {
                if (this.value) {
                    // Apply to all charts
                    updateAllCharts(this.value);
                }
            });
            
            lookbackSelect.addEventListener('change', function() {
                const currentTimeSlice = globalTimeSelect.value || 'D';
                loadDashboardData(currentTimeSlice, parseInt(this.value));
            });
        }
        
        async function loadDashboardData(timeSlice = 'D', lookbackDays = 60) {  // 60 days to show archived projects
            try {
                const response = await fetch(`/api/data/${timeSlice}?lookback_days=${lookbackDays}`);
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                dashboardData = data;
                updateStatsCards(data);
                renderCharts(data, timeSlice);
                
            } catch (error) {
                showError('Failed to load dashboard data: ' + error.message);
            }
        }
        
        function updateStatsCards(data) {
            // Calculate summary statistics
            let totalActiveTime = 0;
            let totalFiles = 0;
            let totalEfficiency = 0;
            let efficiencyCount = 0;
            
            if (data.activity_data.active_time) {
                data.activity_data.active_time.forEach(record => {
                    totalActiveTime += record.active_time || 0;
                });
            }
            
            if (data.activity_data.files_processed) {
                data.activity_data.files_processed.forEach(record => {
                    totalFiles += record.files_processed || 0;
                });
            }
            
            if (data.activity_data.efficiency) {
                data.activity_data.efficiency.forEach(record => {
                    if (record.efficiency > 0) {
                        totalEfficiency += record.efficiency;
                        efficiencyCount++;
                    }
                });
            }
            
            // Update cards
            document.getElementById('total-active-time').textContent = 
                Math.round(totalActiveTime / 60) + 'm';
            document.getElementById('total-files').textContent = totalFiles;
            document.getElementById('avg-efficiency').textContent = 
                efficiencyCount > 0 ? Math.round(totalEfficiency / efficiencyCount) + '%' : '--';
            document.getElementById('active-scripts').textContent = 
                data.metadata.scripts_found.length;
        }
        
        function renderCharts(data, timeSlice) {
            const container = document.getElementById('charts-container');
            container.innerHTML = '';
            const baselines = (data && data.metadata && data.metadata.baseline_labels) || {};
            const lookbackDays = parseInt(document.getElementById('lookback-days').value);
            const baselineFor = (slice) => {
                const server = (baselines && baselines[slice]) || [];
                if (server && server.length) return server;
                return buildClientBaselineLabels(slice, lookbackDays);
            };
            
            // Render different chart types
            if (data.activity_data.active_time && data.activity_data.active_time.length > 0) {
                renderTimeSeriesChart(container, 'Active Time by Script', data.activity_data.active_time, 
                             'active_time', timeSlice, 'script', baselineFor(timeSlice));
            }
            
            if (data.activity_data.files_processed && data.activity_data.files_processed.length > 0) {
                renderTimeSeriesChart(container, 'Files Processed by Script', data.activity_data.files_processed, 
                             'files_processed', timeSlice, 'script', baselineFor(timeSlice));
            }
            
            if (data.file_operations_data.deletions && data.file_operations_data.deletions.length > 0) {
                renderTimeSeriesChart(container, 'Files Deleted by Script', data.file_operations_data.deletions, 
                             'file_count', timeSlice, 'script', baselineFor(timeSlice));
            }
            
            if (data.file_operations_data.by_operation && data.file_operations_data.by_operation.length > 0) {
                renderTimeSeriesChart(container, 'Operations by Type', data.file_operations_data.by_operation, 
                             'file_count', timeSlice, 'operation', baselineFor(timeSlice));
            }
        }
        
        function renderTimeSeriesChart(container, title, data, valueField, timeSlice, groupField = 'script', baselineLabels = []) {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            
            const chartId = 'chart-' + title.toLowerCase().replace(/[^a-z0-9]/g, '-');
            
            chartDiv.innerHTML = `
                <div class="chart-header">
                    <h3 class="chart-title">${title}</h3>
                    <div class="chart-controls">
                        <button class="chart-time-btn ${timeSlice === '15min' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', '15min')">15m</button>
                        <button class="chart-time-btn ${timeSlice === '1H' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', '1H')">1h</button>
                        <button class="chart-time-btn ${timeSlice === 'D' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'D')">Daily</button>
                        <button class="chart-time-btn ${timeSlice === 'W' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'W')">Weekly</button>
                        <button class="chart-time-btn ${timeSlice === 'M' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'M')">Monthly</button>
                    </div>
                </div>
                <div class="chart-canvas">
                    <canvas id="${chartId}"></canvas>
                </div>
            `;
            
            container.appendChild(chartDiv);
            
            // Create Chart.js chart
            const ctx = document.getElementById(chartId).getContext('2d');
            
            // Use server-provided canonical labels for alignment/padding
            const built = buildTimeSeries(data, valueField, groupField, baselineLabels || []);
            const finalLabels = (baselineLabels && baselineLabels.length) ? baselineLabels : built.labels;
            const palette = [
                '#4f9dff', '#51cf66', '#ffd43b', '#ff6b6b', '#845ef7', '#22b8cf', '#fcc419', '#e64980'
            ];
            // Build datasets strictly aligned to finalLabels, padding zeros
            const seriesNames = Array.from(new Set(data.map(r => (r[groupField] || 'unknown'))));
            const datasets = seriesNames.map((name, idx) => {
                const seriesData = new Array(finalLabels.length).fill(0);
                for (const rec of data) {
                    if ((rec[groupField] || 'unknown') !== name) continue;
                    const t = rec['time_slice'];
                    const v = Number(rec[valueField] || 0);
                    const ix = finalLabels.indexOf(t);
                    if (ix !== -1) seriesData[ix] += v;
                }
                return {
                    label: name,
                    data: seriesData,
                    backgroundColor: palette[idx % palette.length],
                    borderColor: palette[idx % palette.length],
                    borderWidth: 1,
                };
            });

            const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: finalLabels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: 'white' } },
                        tooltip: {
                            callbacks: {
                                title: (items) => {
                                    if (!items || !items.length) return '';
                                    return formatTooltipLabel(items[0].label, timeSlice);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#a0a3b1',
                                callback: (val, index, ticks) => formatTickLabel(built.labels[index], timeSlice)
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#a0a3b1'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        }
                    }
                }
            });
            
            charts[chartId] = {
                chart: chart,
                title: title,
                data: data,
                valueField: valueField,
                groupField: groupField
            };
        }

        // (Project bands overlay removed by request; keeping code minimal and stable.)

        function buildTimeSeries(data, valueField, groupField, baselineLabels = null) {
            const useBaseline = Array.isArray(baselineLabels) && baselineLabels.length > 0;
            const labelsSet = new Set(useBaseline ? baselineLabels : []);
            const seriesMap = new Map(); // name -> Map(label->value)
            data.forEach(rec => {
                const t = rec['time_slice'];
                if (!t) return;
                // If baseline provided, ignore records outside it to keep strict alignment
                if (useBaseline && !labelsSet.has(t)) return;
                if (!useBaseline) labelsSet.add(t);
                const series = rec[groupField] || 'unknown';
                if (!seriesMap.has(series)) seriesMap.set(series, new Map());
                const m = seriesMap.get(series);
                const v = Number(rec[valueField] || 0);
                m.set(t, (m.get(t) || 0) + v);
            });
            const labels = Array.from(labelsSet).sort();
            const series = Array.from(seriesMap.keys());
            const values = {};
            series.forEach(name => {
                const m = seriesMap.get(name);
                values[name] = labels.map(l => m.get(l) || 0);
            });
            return { labels, series, values };
        }

        // Build baseline labels for padding zero-activity periods based on lookback (server fallback)
        function buildBaselineLabels(timeSlice, lookbackDays) {
            try {
                const labels = [];
                if (timeSlice === 'D' || !timeSlice) {
                    const end = new Date();
                    end.setHours(0,0,0,0);
                    const start = new Date(end);
                    start.setDate(end.getDate() - (lookbackDays - 1));
                    const cur = new Date(start);
                    while (cur <= end) {
                        const y = cur.getFullYear();
                        const m = (cur.getMonth()+1).toString().padStart(2,'0');
                        const d = cur.getDate().toString().padStart(2,'0');
                        labels.push(`${y}-${m}-${d}`);
                        cur.setDate(cur.getDate() + 1);
                    }
                }
                return labels;
            } catch(e) {
                return [];
            }
        }

        // Client-side baseline label builder used when server baseline is missing
        function buildClientBaselineLabels(timeSlice, lookbackDays) {
            const out = [];
            const now = new Date();
            if (timeSlice === '15min') {
                const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), Math.floor(now.getMinutes()/15)*15, 0, 0);
                const start = new Date(end.getTime() - (lookbackDays-1)*24*60*60*1000);
                start.setMinutes(Math.floor(start.getMinutes()/15)*15, 0, 0);
                for (let d = new Date(start); d <= end; d = new Date(d.getTime() + 15*60*1000)) out.push(d.toISOString().replace('Z',''));
            } else if (timeSlice === '1H') {
                const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), 0, 0, 0);
                const start = new Date(end.getTime() - (lookbackDays-1)*24*60*60*1000);
                start.setMinutes(0,0,0);
                for (let d = new Date(start); d <= end; d = new Date(d.getTime() + 60*60*1000)) out.push(d.toISOString().replace('Z',''));
            } else if (timeSlice === 'D') {
                const end = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                const start = new Date(end.getTime() - (lookbackDays-1)*24*60*60*1000);
                for (let d = new Date(start); d <= end; d = new Date(d.getTime() + 24*60*60*1000)) out.push(`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`);
            } else if (timeSlice === 'W') {
                const end = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                const endMonday = new Date(end.getTime() - end.getDay()*24*60*60*1000 + 1*24*60*60*1000);
                const start = new Date(endMonday.getTime() - Math.ceil(lookbackDays/7-1)*7*24*60*60*1000);
                for (let d = new Date(start); d <= endMonday; d = new Date(d.getTime() + 7*24*60*60*1000)) out.push(`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`);
            } else if (timeSlice === 'M') {
                const end = new Date(now.getFullYear(), now.getMonth(), 1);
                const start = new Date(end);
                start.setMonth(start.getMonth() - Math.ceil(lookbackDays/30-1));
                for (let d = new Date(start); d <= end; ) {
                    out.push(`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-01`);
                    d.setMonth(d.getMonth()+1);
                }
            }
            return out;
        }

        function formatTickLabel(label, timeSlice) {
            // Bottom axis: show day for daily; show 12h time for intraday; MDY for weekly/monthly
            const d = parseLabelToDate(label, timeSlice);
            if (!d) return label;
            if (timeSlice === '15min' || timeSlice === '1H') {
                let hours = d.getHours();
                const minutes = d.getMinutes().toString().padStart(2, '0');
                const ampm = hours >= 12 ? 'PM' : 'AM';
                hours = hours % 12; if (hours === 0) hours = 12;
                return `${hours}:${minutes} ${ampm}`;
            } else if (timeSlice === 'D') {
                return `${(d.getMonth()+1).toString().padStart(2,'0')}/${d.getDate().toString().padStart(2,'0')}/${d.getFullYear()}`;
            } else {
                return `${(d.getMonth()+1).toString().padStart(2,'0')}/${d.getDate().toString().padStart(2,'0')}/${d.getFullYear()}`;
            }
        }

        function formatTooltipLabel(label, timeSlice) {
            // Hover: always show MDY and 12-hour time when applicable
            const d = parseLabelToDate(label, timeSlice);
            if (!d) return label;
            const mdy = `${(d.getMonth()+1).toString().padStart(2,'0')}/${d.getDate().toString().padStart(2,'0')}/${d.getFullYear()}`;
            if (timeSlice === '15min' || timeSlice === '1H') {
                let hours = d.getHours();
                const minutes = d.getMinutes().toString().padStart(2, '0');
                const ampm = hours >= 12 ? 'PM' : 'AM';
                hours = hours % 12; if (hours === 0) hours = 12;
                return `${mdy} ${hours}:${minutes} ${ampm}`;
            }
            return mdy;
        }

        function parseLabelToDate(label, timeSlice) {
            try {
                if (timeSlice === 'D') {
                    // label is YYYY-MM-DD
                    const [y, m, d] = label.split('-').map(Number);
                    return new Date(y, m - 1, d);
                }
                // ISO strings for intraday/week/month
                const d = new Date(label);
                if (!isNaN(d.getTime())) return d;
            } catch (e) {}
            return null;
        }
        
        function updateChart(chartId, timeSlice) {
            // Update individual chart time slice
            const lookbackDays = parseInt(document.getElementById('lookback-days').value);
            loadDashboardData(timeSlice, lookbackDays);
        }
        
        function updateAllCharts(timeSlice) {
            // Update all charts to the same time slice
            const lookbackDays = parseInt(document.getElementById('lookback-days').value);
            loadDashboardData(timeSlice, lookbackDays);
        }
        
        function showError(message) {
            const container = document.getElementById('charts-container');
            container.innerHTML = `<div class="error">Error: ${message}</div>`;
        }
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Productivity Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data-dir", default="scripts", help="Data directory path")
    
    args = parser.parse_args()
    
    dashboard = ProductivityDashboard(args.data_dir)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
