#!/usr/bin/env python3
"""
Dashboard Analytics Engine
==========================
Transforms raw data from DashboardDataEngine and ProjectMetricsAggregator
into the exact JSON contract expected by the dashboard template.

Key responsibilities:
- Bucket data by time slice with canonical baseline labels
- Align series to baseline (fill gaps with zeros)
- Compute historical averages for "cloud" overlays
- Transform project metrics for comparison charts
- Ensure deterministic label ordering (chronological)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.data_engine import DashboardDataEngine
from scripts.dashboard.project_metrics_aggregator import ProjectMetricsAggregator
from scripts.dashboard.timesheet_parser import TimesheetParser


# Centralized tool order - used across ALL charts, tables, and toggles
# This ensures consistent ordering everywhere in the dashboard
STANDARD_TOOL_ORDER = [
    'Web Image Selector',
    'Web Character Sorter',
    'Multi Crop Tool'
]


class DashboardAnalytics:
    """
    High-level analytics engine that orchestrates data collection and transformation.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.engine = DashboardDataEngine(str(data_dir))
        self.project_agg = ProjectMetricsAggregator(data_dir)
        # Timesheet is in project_root/data/timesheet.csv
        timesheet_path = self.data_dir / "data" / "timesheet.csv"
        self.timesheet_parser = TimesheetParser(timesheet_path)
        print(f"[INIT DEBUG] Analytics data_dir: {self.data_dir}")
        print(f"[INIT DEBUG] Timesheet path: {timesheet_path}")
        
        # PERFORMANCE FIX: Cache file operations to avoid reloading 19+ times
        self._cached_file_ops = None
        self._cached_file_ops_for_daily = None
        self._cache_timestamp = None
        
        # Track timesheet modification time for cache invalidation
        self._timesheet_mtime = None
    
    def generate_dashboard_response(
        self,
        time_slice: str,
        lookback_days: int,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete dashboard response matching UI contract.
        
        Args:
            time_slice: One of "15min", "1H", "D", "W", "M"
            lookback_days: Number of days to look back
            project_id: Optional project ID filter
            
        Returns:
            Dictionary with all dashboard data
        """
        import time
        overall_start = time.time()
        print(f"\n{'='*70}")
        print(f"[TIMING] Dashboard Response Generation Started")
        print(f"  time_slice: {time_slice}, lookback_days: {lookback_days}, project_id: {project_id}")
        print(f"{'='*70}")
        
        # Get raw data from engine
        step_start = time.time()
        raw_data = self.engine.generate_dashboard_data(
            time_slice=time_slice,
            lookback_days=lookback_days,
            project_id=project_id
        )
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ engine.generate_dashboard_data: {step_time:.3f}s")
        
        # Store raw_data for later use (e.g., in _build_billed_vs_actual)
        self.engine._raw_data = raw_data
        
        # Get project metrics (now includes daily summaries merged with logs)
        step_start = time.time()
        project_metrics = self.project_agg.aggregate()
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ project_agg.aggregate: {step_time:.3f}s")
        
        # Build baseline labels for this time slice
        step_start = time.time()
        baseline_labels = self.engine.build_time_labels(time_slice, lookback_days)
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ build_time_labels: {step_time:.3f}s")
        
        # Transform to UI contract
        step_start = time.time()
        response = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_slice": time_slice,
                "lookback_days": lookback_days,
                "hours_scope": "lifetime",  # table hours/days computed over full project lifetime
                "hours_source": "file_operations",  # derived from file operation timing (break-aware)
                "session_source": self.engine.snapshot_loader.get_session_source(),  # derived or legacy_work_timer
                "performance_mode": self.engine.snapshot_loader.is_performance_mode(),
                "standard_tool_order": STANDARD_TOOL_ORDER,  # Centralized tool order for UI consistency
                "baseline_labels": {
                    "15min": self.engine.build_time_labels("15min", lookback_days),
                    "1H": self.engine.build_time_labels("1H", lookback_days),
                    "D": self.engine.build_time_labels("D", lookback_days),
                    "W": self.engine.build_time_labels("W", lookback_days),
                    "M": self.engine.build_time_labels("M", lookback_days)
                },
                "active_project": project_id or ""
            },
            "projects": self._build_projects_list(raw_data),
            "charts": self._build_charts(raw_data, baseline_labels, time_slice),
            "timing_data": self._build_timing_data(raw_data),
            "project_comparisons": self._build_project_comparisons(project_metrics, raw_data, time_slice, lookback_days),
            "project_kpi": self._build_project_kpi(project_id, project_metrics, raw_data),
            "project_metrics": project_metrics,
            "project_markers": self._build_project_markers(project_id, raw_data),
        }
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ build response metadata/projects/charts/etc: {step_time:.3f}s")
        
        # Build detailed productivity table
        step_start = time.time()
        detailed_table = self._build_project_productivity_table(raw_data, project_metrics, time_slice, lookback_days)
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ _build_project_productivity_table: {step_time:.3f}s")
        
        # Build productivity overview from detailed data
        step_start = time.time()
        productivity_overview = self._build_productivity_overview_table(detailed_table)
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ _build_productivity_overview_table: {step_time:.3f}s")
        
        response["project_productivity_table"] = detailed_table
        response["productivity_overview"] = productivity_overview
        
        # Add timesheet data for billing/efficiency tracking
        step_start = time.time()
        timesheet_data = self._load_timesheet_data()
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ _load_timesheet_data: {step_time:.3f}s")
        response["timesheet_data"] = timesheet_data
        
        # Add billed vs actual comparison
        step_start = time.time()
        response["billed_vs_actual"] = self._build_billed_vs_actual(timesheet_data, project_metrics)
        response["billed_vs_actual_timeseries"] = self._build_billed_vs_actual_timeseries(timesheet_data, project_metrics)
        step_time = time.time() - step_start
        print(f"[TIMING] ✓ _build_billed_vs_actual: {step_time:.3f}s")
        
        overall_time = time.time() - overall_start
        print(f"\n{'='*70}")
        print(f"[TIMING] TOTAL DASHBOARD RESPONSE TIME: {overall_time:.3f}s")
        print(f"{'='*70}\n")
        
        return response
    
    def _build_projects_list(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract projects list from raw data."""
        projects = raw_data.get("projects", [])
        return [
            {
                "projectId": p.get("projectId"),
                "title": p.get("title") or p.get("projectId"),
                "status": p.get("status")
            }
            for p in projects
        ]
    
    def _build_charts(
        self,
        raw_data: Dict[str, Any],
        baseline_labels: List[str],
        time_slice: str
    ) -> Dict[str, Any]:
        """
        Build charts data aligned to baseline labels.
        
        Returns:
            {
                "by_script": { "<ToolName>": { "dates": [...], "counts": [...] } },
                "by_operation": { "<OpType>": { "dates": [...], "counts": [...] } },
                "by_project": { "<ProjectName>": { "dates": [...], "counts": [...] } }
            }
        """
        charts = {
            "by_script": {},
            "by_operation": {},
            "by_project": {}
        }
        
        # Extract file operations data
        file_ops_data = raw_data.get("file_operations_data", {})
        
        # Build by_script chart data
        by_script_raw = file_ops_data.get("by_script", [])
        charts["by_script"] = self._aggregate_to_baseline(
            by_script_raw,
            baseline_labels,
            group_field="script",
            value_field="file_count",
            map_display_names=True
        )
        
        # Build by_operation chart data
        by_operation_raw = file_ops_data.get("by_operation", [])
        charts["by_operation"] = self._aggregate_to_baseline(
            by_operation_raw,
            baseline_labels,
            group_field="operation",
            value_field="file_count"
        )
        
        # Build by_project chart data (NEW - for project-to-project comparison)
        by_project_raw = file_ops_data.get("by_project", [])
        if by_project_raw:
            charts["by_project"] = self._aggregate_to_baseline(
                by_project_raw,
                baseline_labels,
                group_field="project",
                value_field="file_count"
            )
        
        return charts
    
    def _aggregate_to_baseline(
        self,
        records: List[Dict[str, Any]],
        baseline_labels: List[str],
        group_field: str,
        value_field: str,
        map_display_names: bool = False
    ) -> Dict[str, Dict[str, List]]:
        """
        Aggregate records to baseline labels, filling gaps with zeros.
        
        Args:
            records: List of aggregated records from data_engine
            baseline_labels: Canonical ordered labels for this time slice
            group_field: Field to group by (e.g., "script", "operation")
            value_field: Field to sum (e.g., "file_count")
            
        Returns:
            Dict mapping group keys to {"dates": [...], "counts": [...]}
        """
        # Build label index for fast lookup
        label_index = {label: idx for idx, label in enumerate(baseline_labels)}
        
        # Group records by group_field
        grouped = defaultdict(lambda: defaultdict(float))
        for record in records:
            time_key = record.get("time_slice")
            group_key = record.get(group_field)
            value = record.get(value_field, 0) or 0
            
            if time_key and group_key:
                # Optionally remap names to display names (charts expect human-friendly labels)
                if map_display_names and group_field == "script":
                    group_key = self.engine.get_display_name(group_key)
                grouped[group_key][time_key] += float(value)
        
        # Align to baseline labels (fill gaps with zeros)
        result = {}
        for group_key, time_values in grouped.items():
            counts = [0.0] * len(baseline_labels)
            for time_key, value in time_values.items():
                if time_key in label_index:
                    idx = label_index[time_key]
                    counts[idx] += value
            
            result[group_key] = {
                "dates": baseline_labels.copy(),
                "counts": [int(c) for c in counts]  # Convert to integers
            }
        
        return result
    
    def _build_timing_data(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Build timing data per tool.
        
        Returns:
            {
                "<ToolName>": {
                    "work_time_minutes": <float>,
                    "timing_method": "file_operations|activity_timer|both"
                }
            }
        """
        return raw_data.get("timing_data", {})
    
    def _build_project_comparisons(
        self,
        project_metrics: Dict[str, Dict[str, Any]],
        raw_data: Dict[str, Any],
        time_slice: str,
        lookback_days: int
    ) -> List[Dict[str, Any]]:
        """
        Build project comparison data for bar charts with tool-level breakdowns.
        
        Returns:
            [
                {
                    "projectId": "abc",
                    "title": "Project ABC",
                    "total_operations": <int>,
                    "operations_by_type": { "crop": <int>, "delete": <int>, ... },
                    "tools": {
                        "Web Image Selector": {
                            "images_processed": <int>,
                            "work_time_minutes": <float>,
                            "images_per_hour": <float>
                        },
                        ...
                    }
                }
            ]
        """
        comparisons = []
        
        # Get projects from raw_data to have access to full project info
        projects_by_id = {p.get("projectId"): p for p in raw_data.get("projects", []) if p.get("projectId")}
        
        for project_id, metrics in project_metrics.items():
            title = metrics.get("title", project_id)
            ops_by_type = metrics.get("totals", {}).get("operations_by_type", {})
            total_ops = sum(ops_by_type.values())
            
            # Build tools breakdown for this project
            project = projects_by_id.get(project_id, {})
            tools_breakdown = self._build_tools_breakdown_for_project(
                project_id,
                project,
                metrics,
                raw_data,
                time_slice,
                lookback_days
            )
            
            # Convert tools breakdown to simpler format for charts
            tools_metrics = {}
            for tool_name, tool_data in tools_breakdown.items():
                hours = tool_data.get("hours", 0) or 0
                
                # Get image count (handle selector vs other tools)
                if tool_name == "Web Image Selector":
                    images = tool_data.get("images_total", 0) or 0
                else:
                    images = tool_data.get("images", 0) or 0
                
                # Calculate images per hour
                img_per_hour = round(images / hours, 1) if hours > 0 and images > 0 else 0
                
                tools_metrics[tool_name] = {
                    "images_processed": images,
                    "work_time_minutes": round(hours * 60, 1),
                    "images_per_hour": img_per_hour
                }
            
            # Get startedAt date for chronological sorting
            started_at = project.get("startedAt", "")
            
            comparisons.append({
                "projectId": project_id,
                "title": title,
                "total_operations": total_ops,
                "operations_by_type": ops_by_type,
                "tools": tools_metrics,
                "startedAt": started_at
            })
        
        # Sort chronologically (oldest to newest) for left-to-right timeline
        comparisons.sort(key=lambda x: x.get("startedAt", "9999-99-99"))  # Projects without dates go to end
        
        return comparisons
    
    def _build_project_kpi(
        self,
        project_id: Optional[str],
        project_metrics: Dict[str, Dict[str, Any]],
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build project KPI card data.
        
        Returns:
            {
                "images_per_hour": <float>,
                "images_processed": <int>
            }
        """
        if not project_id or project_id not in project_metrics:
            # Return aggregate across all projects
            total_images = 0
            total_work_minutes = 0
            
            # Sum from timing_data
            timing_data = raw_data.get("timing_data", {})
            for tool, stats in timing_data.items():
                total_work_minutes += stats.get("work_time_minutes", 0)
            
            # Sum from by_operation totals
            file_ops = raw_data.get("file_operations_data", {})
            by_op = file_ops.get("by_operation", [])
            for record in by_op:
                total_images += record.get("file_count", 0) or 0
            
            iph = (total_images / (total_work_minutes / 60)) if total_work_minutes > 0 else 0
            
            return {
                "images_per_hour": round(iph, 2),
                "images_processed": total_images
            }
        
        # Return specific project KPI
        pm = project_metrics[project_id]
        return {
            "images_per_hour": pm.get("throughput", {}).get("images_per_hour", 0),
            "images_processed": pm.get("totals", {}).get("images_processed", 0)
        }
    
    def _build_project_markers(
        self,
        project_id: Optional[str],
        raw_data: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """
        Build project markers (start/finish timestamps).
        
        Returns:
            {
                "startedAt": "<ISO timestamp or null>",
                "finishedAt": "<ISO timestamp or null>"
            }
        """
        if not project_id:
            return {"startedAt": None, "finishedAt": None}
        
        # Find project in raw_data
        projects = raw_data.get("projects", [])
        project = next((p for p in projects if p.get("projectId") == project_id), None)
        
        if not project:
            return {"startedAt": None, "finishedAt": None}
        
        return {
            "startedAt": project.get("startedAt"),
            "finishedAt": project.get("finishedAt")
        }
    
    def _build_productivity_overview_table(
        self,
        detailed_table_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build high-level productivity overview table (img/h metrics only).
        
        Takes the detailed table data and extracts just productivity metrics
        for a clean summary view.
        
        Returns:
            [
                {
                    "projectId": "mojo1",
                    "title": "Mojo1",
                    "tool_metrics": {
                        "Web Image Selector": 1048,  # img/h
                        "Character Sorter": 930,
                        "Multi Crop Tool": 183
                    },
                    "overall_img_h": 886  # total images / total hours
                }
            ]
        """
        overview_data = []
        
        for project in detailed_table_data:
            project_id = project.get("projectId")
            title = project.get("title")
            tools = project.get("tools", {})
            
            # Extract img/h for each tool
            tool_metrics = {}
            total_images = 0
            total_hours = 0.0
            
            for tool_name in STANDARD_TOOL_ORDER:
                tool_data = tools.get(tool_name)
                if not tool_data:
                    continue
                
                hours = tool_data.get("hours", 0) or 0
                
                # Get image count (handle selector vs other tools)
                if tool_name == "Web Image Selector":
                    images = tool_data.get("images_total", 0) or 0
                else:
                    images = tool_data.get("images", 0) or 0
                
                # Calculate img/h for this tool
                if hours > 0 and images > 0:
                    img_h = round(images / hours)
                    tool_metrics[tool_name] = img_h
                    
                    # Accumulate for overall calculation
                    total_images += images
                    total_hours += hours
            
            # Calculate overall img/h
            overall_img_h = round(total_images / total_hours) if total_hours > 0 else 0
            
            overview_data.append({
                "projectId": project_id,
                "title": title,
                "tool_metrics": tool_metrics,
                "overall_img_h": overall_img_h
            })
        
        return overview_data
    
    def _build_project_productivity_table(
        self,
        raw_data: Dict[str, Any],
        project_metrics: Dict[str, Dict[str, Any]],
        time_slice: str,
        lookback_days: int
    ) -> List[Dict[str, Any]]:
        """
        Build detailed project productivity table data.
        
        Returns data for table showing:
        - Project name
        - Start images / End images
        - Per-tool breakdown: Hours, Days active, Images processed
          - For Image Selector: Selected vs Cropped breakdown
        
        Returns:
            [
                {
                    "projectId": "mojo1",
                    "title": "Mojo1",
                    "start_images": 12450,
                    "end_images": 3358,
                    "tools": {
                        "Web Image Selector": {
                            "hours": 18.2,
                            "days": 12,
                            "images_total": 3358,
                            "images_selected": 2100,
                            "images_cropped": 1258
                        },
                        "Character Sorter": {
                            "hours": 4.1,
                            "days": 7,
                            "images": 3358
                        },
                        "Multi Crop Tool": {
                            "hours": 12.7,
                            "days": 6,
                            "images": 3358
                        }
                    }
                }
            ]
        """
        projects = raw_data.get("projects", [])
        file_ops_records = raw_data.get("file_operations_data", {})
        timing_data = raw_data.get("timing_data", {})
        
        table_data = []
        
        for project in projects:
            project_id = project.get("projectId")
            if not project_id:
                continue
            
            # Get project metrics
            pm = project_metrics.get(project_id, {})
            
            # Start images (from manifest counts)
            counts = project.get("counts", {})
            start_images = counts.get("initialImages", 0)
            
            # End images (from manifest if finished, else from metrics)
            end_images = counts.get("finalImages") or pm.get("totals", {}).get("images_processed", 0)
            
            # Build per-tool breakdown for THIS project
            tools_breakdown = self._build_tools_breakdown_for_project(
                project_id,
                project,
                pm,
                raw_data,
                time_slice,
                lookback_days
            )
            
            table_data.append({
                "projectId": project_id,
                "title": project.get("title") or project_id,
                "startedAt": project.get("startedAt"),
                "start_images": start_images,
                "end_images": end_images,
                "tools": tools_breakdown
            })
        
        # Sort by startedAt (most recent first), then by title
        def sort_key(proj):
            started = proj.get("startedAt")
            if started:
                try:
                    # Parse ISO date and return for sorting (newest first = descending)
                    dt = datetime.fromisoformat(started.replace('Z', ''))
                    return (0, -dt.timestamp())  # negative for descending order
                except:
                    pass
            # Projects without date go to end
            return (1, proj.get("title", ""))
        
        table_data.sort(key=sort_key)
        
        return table_data
    
    def _build_tools_breakdown_for_project(
        self,
        project_id: str,
        project: Dict[str, Any],
        pm: Dict[str, Any],
        raw_data: Dict[str, Any],
        time_slice: str,
        lookback_days: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build per-tool breakdown for a specific project.
        
        Returns tool metrics filtered to this project only.
        """
        tools_breakdown = {}
        # Use centralized tool order
        allowed_tools_order = STANDARD_TOOL_ORDER
        
        # PERFORMANCE FIX: Use cached file operations instead of reloading
        # This was loading 222K records 19 times (once per project + once at start)!
        import time
        if self._cached_file_ops is None:
            cache_start = time.time()
            self._cached_file_ops = self.engine.load_file_operations()
            cache_time = time.time() - cache_start
            print(f"[CACHE] Loaded {len(self._cached_file_ops)} file operations: {cache_time:.3f}s")
        
        window_ops = self._cached_file_ops
        
        # Filter by project date range (startedAt to finishedAt)
        started_at = project.get('startedAt')
        finished_at = project.get('finishedAt')
        
        def belongs(rec: Dict[str, Any]) -> bool:
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
        
        proj_ops = [r for r in window_ops if belongs(r)]

        # Group per tool (display)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in proj_ops:
            display = self.engine.get_display_name(r.get('script') or 'unknown')
            if display not in allowed_tools_order:
                continue
            grouped.setdefault(display, []).append(r)

        for display_name in allowed_tools_order:
            records = grouped.get(display_name, [])
            if not records:
                # Fallback to project metrics when no windowed ops found for this tool
                # Map display_name back to a tool key in pm['tools']
                tool_stats = None
                for tool_key, stats in (pm.get('tools', {}) or {}).items():
                    if self.engine.get_display_name(tool_key) == display_name:
                        tool_stats = stats
                        break
                if not tool_stats:
                    continue
                images = int(tool_stats.get('images_processed', 0) or 0)
                iph = float(tool_stats.get('images_per_hour', 0) or 0)
                hours = round(images / iph, 1) if iph > 0 else 0.0
                days = self._count_active_days_for_tool(raw_data, project_id, tool_key)
                if display_name == "Web Image Selector":
                    by_dest = (pm.get('totals', {}) or {}).get('operations_by_dest', {})
                    move_breakdown = by_dest.get('move', {}) if isinstance(by_dest, dict) else {}
                    selected = int(move_breakdown.get('selected', 0) or 0)
                    cropped = int(move_breakdown.get('crop', 0) or 0)
                    tools_breakdown[display_name] = {
                        'hours': hours,
                        'days': days,
                        'images_total': images,
                        'images_selected': selected,
                        'images_cropped': cropped
                    }
                else:
                    tools_breakdown[display_name] = {
                        'hours': hours,
                        'days': days,
                        'images': images
                    }
                continue
            # Hours from file-op timing
            try:
                metrics = self.engine.calculate_file_operation_work_time(records)
                hours = float(metrics.get('work_time_minutes') or 0.0) / 60.0
            except Exception:
                hours = 0.0
            # Days active in window
            days_set = set()
            from datetime import datetime as _dt
            for rec in records:
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
            
            # Images processed in window - COUNT PNG FILES ONLY
            def count_pngs(rec):
                """Count PNG files only, not companion files."""
                files_list = rec.get('files', [])
                if files_list:
                    return sum(1 for f in files_list if isinstance(f, str) and f.lower().endswith('.png'))
                # Fallback to file_count for daily summary records (which are already PNG-only after backfill)
                return int(rec.get('file_count') or 0)
            
            images = sum(count_pngs(r) for r in records)

            if display_name == "Web Image Selector":
                selected = sum(count_pngs(r) for r in records if (r.get('operation') == 'move' and 'selected' in str(r.get('dest_dir') or '').lower()))
                cropped = sum(count_pngs(r) for r in records if (r.get('operation') == 'move' and 'crop' in str(r.get('dest_dir') or '').lower()))
                tools_breakdown[display_name] = {
                    'hours': round(hours, 1),
                    'days': days,
                    'images_total': images,
                    'images_selected': selected,
                    'images_cropped': cropped
                }
            else:
                tools_breakdown[display_name] = {
                    'hours': round(hours, 1),
                    'days': days,
                    'images': images
                }
        
        return tools_breakdown
    
    def _count_active_days_for_tool(
        self,
        raw_data: Dict[str, Any],
        project_id: str,
        tool_name: str
    ) -> int:
        """
        Count unique calendar days where a tool was used for a project.
        
        Returns number of days with at least one operation.
        """
        # Find the project to get its date range
        projects = raw_data.get("projects", [])
        project = next((p for p in projects if p.get("projectId") == project_id), None)
        if not project:
            return 0
        
        started_at = project.get('startedAt')
        finished_at = project.get('finishedAt')
        if not started_at:
            return 0
        
        # PERFORMANCE FIX: Use cached file operations
        all_ops = self._cached_file_ops if self._cached_file_ops is not None else self.engine.load_file_operations()
        
        unique_dates = set()
        
        for rec in all_ops:
            # Match tool name
            rec_script = rec.get("script", "")
            if self.engine.get_display_name(rec_script) != self.engine.get_display_name(tool_name):
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
    
    def _build_billed_vs_actual(self, timesheet_data: Dict[str, Any], project_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare billed hours (from timesheet) vs actual hours (from timer data).
        
        Returns:
            {
                "projects": [
                    {
                        "name": "mojo-1",
                        "billed_hours": 66,
                        "actual_hours": 70,
                        "difference": 4,
                        "overbilled": False,
                        "started_at": "2024-01-15T10:00:00"
                    },
                    ...
                ]
            }
        """
        comparison = []
        
        # Get timesheet projects
        timesheet_projects = {p['name']: p for p in timesheet_data.get('projects', [])}
        
        print(f"\n[BILLED_VS_ACTUAL DEBUG] Timesheet projects: {list(timesheet_projects.keys())}")
        print(f"[BILLED_VS_ACTUAL DEBUG] Available project metrics: {list(project_metrics.keys())}")
        
        # Get raw data for project dates
        raw_data = self.engine._raw_data if hasattr(self.engine, '_raw_data') else {}
        projects_list = raw_data.get("projects", [])
        projects_by_id = {p.get("projectId"): p for p in projects_list}
        
        print(f"[BILLED_VS_ACTUAL DEBUG] Projects with manifests: {list(projects_by_id.keys())}")
        
        # Match with actual project data
        for ts_name, ts_data in timesheet_projects.items():
            billed_hours = ts_data['total_hours']
            
            # Try to find matching project in metrics
            # Normalize names for matching (lowercase, remove spaces/dashes)
            ts_normalized = ts_name.lower().replace(' ', '').replace('-', '').replace('_', '')
            
            actual_hours = 0
            matched_project = None
            started_at = None
            
            print(f"\n[BILLED_VS_ACTUAL DEBUG] Looking for match for '{ts_name}' (normalized: '{ts_normalized}')")
            
            for proj_id, proj_data in project_metrics.items():
                proj_normalized = proj_id.lower().replace(' ', '').replace('-', '').replace('_', '')
                
                # Try multiple matching strategies (in order of specificity)
                matched = False
                
                # 1. Exact match (highest priority)
                if ts_normalized == proj_normalized:
                    matched = True
                    print(f"  - '{proj_id}' (normalized: '{proj_normalized}') → EXACT MATCH!")
                
                # 2. Check if timesheet name contains project ID as a complete "word"
                # (e.g., "mojo-1" should match "mojo1" but "mojo1-4/mojo-3" should NOT match "mojo1")
                elif proj_normalized in ts_normalized:
                    # Ensure it's not a substring within another word
                    # Check boundaries: start of string or after delimiter, and end of string or before delimiter
                    idx = ts_normalized.find(proj_normalized)
                    if idx != -1:
                        before_ok = (idx == 0) or (ts_normalized[idx-1] in ['/', '_', '-', ' '])
                        after_idx = idx + len(proj_normalized)
                        after_ok = (after_idx >= len(ts_normalized)) or (ts_normalized[after_idx] in ['/', '_', '-', ' '])
                        
                        if before_ok and after_ok:
                            matched = True
                            print(f"  - '{proj_id}' (normalized: '{proj_normalized}') → PARTIAL MATCH with boundaries")
                        else:
                            print(f"  - '{proj_id}' (normalized: '{proj_normalized}') → substring found but NOT a word boundary (rejected)")
                else:
                    print(f"  - '{proj_id}' (normalized: '{proj_normalized}') → no match")
                
                if matched:
                    # Found a match - get actual hours from totals
                    totals = proj_data.get('totals', {})
                    actual_hours = totals.get('work_hours', 0) or 0
                    matched_project = proj_id
                    
                    # Get start date for sorting (must have manifest)
                    if proj_id in projects_by_id:
                        started_at = projects_by_id[proj_id].get('startedAt')
                    
                    print(f"  ✓ MATCHED! Project: {proj_id}, work_hours: {actual_hours}, started_at: {started_at}")
                    break
            
            if not matched_project:
                print(f"  ✗ NO MATCH FOUND for '{ts_name}' (no manifest) - SKIPPING")
                continue  # Skip timesheet projects without manifests
            
            difference = actual_hours - billed_hours
            
            comparison.append({
                'name': ts_name,
                'billed_hours': round(billed_hours, 1),
                'actual_hours': round(actual_hours, 1),
                'difference': round(difference, 1),
                'overbilled': difference < 0,  # Negative = billed more than worked
                'matched_project': matched_project,
                'started_at': started_at  # Add date for frontend sorting
            })
        
        print(f"\n[BILLED_VS_ACTUAL DEBUG] Final comparison: {len(comparison)} projects")
        for c in comparison[:3]:  # Show first 3 as sample
            print(f"  - {c['name']}: billed={c['billed_hours']}h, actual={c['actual_hours']}h, match={c['matched_project']}, started={c.get('started_at')}")
        
        return {
            'projects': comparison
        }
    
    def _build_billed_vs_actual_timeseries(self, timesheet_data: Dict[str, Any], project_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build daily breakdown of billed vs actual hours.
        
        Returns:
            {
                "daily": {
                    "projects": {
                        "mojo-1": {
                            "dates": ["2024-01-15", "2024-01-16", ...],
                            "billed_hours": [1.0, 2.0, ...],
                            "actual_hours": [0.5, 1.2, ...]
                        },
                        ...
                    }
                }
            }
        """
        from datetime import datetime
        from collections import defaultdict
        
        print("\n[BILLED_VS_ACTUAL_TS DEBUG] Building daily time-series...")
        
        daily_data = {}
        
        try:
            # Get raw data for project info
            raw_data = self.engine._raw_data if hasattr(self.engine, '_raw_data') else {}
            projects_list = raw_data.get("projects", [])
            projects_by_id = {p.get("projectId"): p for p in projects_list}
            
            # Get timesheet projects
            timesheet_projects = timesheet_data.get('projects', [])
            
            print(f"[BILLED_VS_ACTUAL_TS DEBUG] Processing {len(timesheet_projects)} timesheet projects")
        except Exception as e:
            print(f"[BILLED_VS_ACTUAL_TS ERROR] Failed to initialize: {e}")
            return {'daily': {'projects': {}}}
        
        for ts_proj in timesheet_projects:
            try:
                ts_name = ts_proj['name']
                total_billed_hours = ts_proj.get('total_hours', 0)
                start_date_str = ts_proj.get('start_date', '')
                last_date_str = ts_proj.get('last_date', '')
                date_count = ts_proj.get('date_count', 0)
                daily_billed_hours = ts_proj.get('daily_hours', {})  # NEW: Get actual daily billed hours
                
                print(f"[BILLED_VS_ACTUAL_TS DEBUG] Project '{ts_name}' has {len(daily_billed_hours)} days with billed hours")
                
                # Try to match with a manifest project
                ts_normalized = ts_name.lower().replace(' ', '').replace('-', '').replace('_', '')
                matched_project = None
                
                for proj_id in project_metrics.keys():
                    proj_normalized = proj_id.lower().replace(' ', '').replace('-', '').replace('_', '')
                    
                    # Use same matching logic as main function
                    if ts_normalized == proj_normalized:
                        matched_project = proj_id
                        break
                    elif proj_normalized in ts_normalized:
                        idx = ts_normalized.find(proj_normalized)
                        if idx != -1:
                            before_ok = (idx == 0) or (ts_normalized[idx-1] in ['/', '_', '-', ' '])
                            after_idx = idx + len(proj_normalized)
                            after_ok = (after_idx >= len(ts_normalized)) or (ts_normalized[after_idx] in ['/', '_', '-', ' '])
                            if before_ok and after_ok:
                                matched_project = proj_id
                                break
                
                if not matched_project:
                    print(f"[BILLED_VS_ACTUAL_TS DEBUG] Skipping '{ts_name}' (no manifest match)")
                    continue
                
                print(f"[BILLED_VS_ACTUAL_TS DEBUG] Processing '{ts_name}' → matched to '{matched_project}'")
                print(f"  Billed: {total_billed_hours}h over {date_count} days ({start_date_str} to {last_date_str})")
                
                # Get project metrics for this project
                proj_metrics = project_metrics.get(matched_project, {})
                total_work_hours = proj_metrics.get('totals', {}).get('work_hours', 0)
                
                print(f"  Project total work_hours: {total_work_hours}h")
                
                # Get project date range from manifest
                project_manifest = projects_by_id.get(matched_project, {})
                project_start = project_manifest.get('startedAt')
                project_end = project_manifest.get('finishedAt')
                
                if not project_start:
                    print(f"  ⚠️  No startedAt in manifest, skipping daily breakdown")
                    continue
                
                # Parse project dates
                try:
                    start_dt = datetime.fromisoformat(project_start.replace('Z', ''))
                    if project_end:
                        end_dt = datetime.fromisoformat(project_end.replace('Z', ''))
                    else:
                        end_dt = datetime.now()
                except Exception as e:
                    print(f"  ⚠️  Failed to parse dates: {e}")
                    continue
                
                # Load file operations for this project (use cached)
                if not hasattr(self, '_cached_file_ops_for_daily') or self._cached_file_ops_for_daily is None:
                    self._cached_file_ops_for_daily = self.engine.load_file_operations()
                    print(f"  Loaded {len(self._cached_file_ops_for_daily)} file operations")
                
                all_ops = self._cached_file_ops_for_daily
                
                # Filter operations for this project by date range
                proj_ops = []
                for op in all_ops:
                    ts = op.get('timestamp') or op.get('timestamp_str')
                    if not ts:
                        continue
                    
                    try:
                        if isinstance(ts, str):
                            op_dt = datetime.fromisoformat(ts.replace('Z', ''))
                        else:
                            op_dt = ts
                        
                        # Check if within project date range
                        if op_dt >= start_dt and op_dt <= end_dt:
                            proj_ops.append((op_dt.date().isoformat(), op))
                    except:
                        continue
                
                print(f"  Found {len(proj_ops)} operations in date range")
                
                # Group by day
                from collections import defaultdict
                ops_by_day = defaultdict(list)
                for date_str, op in proj_ops:
                    ops_by_day[date_str].append(op)
                
                print(f"  Operations span {len(ops_by_day)} days")
                
                # Calculate work hours per day using the SAME method as project_metrics
                from scripts.utils.companion_file_utils import get_file_operation_metrics
                
                dates = []
                billed_by_day = []
                actual_by_day = []
                
                # NO MORE AVERAGING! Use actual daily billed hours from timesheet
                print(f"  Using ACTUAL daily billed hours from timesheet (no averaging)")
                
                daily_work_hours = {}
                for date_key in sorted(ops_by_day.keys()):
                    day_ops = ops_by_day[date_key]
                    
                    # Convert ops to format expected by metrics calculator
                    ops_for_metrics = []
                    for op in day_ops:
                        op_copy = dict(op)
                        ts = op_copy.get('timestamp')
                        if isinstance(ts, datetime):
                            op_copy['timestamp'] = ts.isoformat()
                        elif not isinstance(ts, str):
                            ts_str = op_copy.get('timestamp_str')
                            if isinstance(ts_str, str):
                                op_copy['timestamp'] = ts_str
                        ops_for_metrics.append(op_copy)
                    
                    try:
                        # Use the SAME calculation as project_metrics (no rounding inflation)
                        metrics = get_file_operation_metrics(ops_for_metrics)
                        work_minutes = float(metrics.get('work_time_minutes') or 0.0)
                        # Use 15-min precision like project totals
                        work_hours = round((work_minutes / 60.0) / 0.25) * 0.25
                        daily_work_hours[date_key] = work_hours
                    except Exception as e:
                        print(f"    Warning: Failed to calculate metrics for {date_key}: {e}")
                        daily_work_hours[date_key] = 0.0
                
                # Build arrays with real daily variation
                for date_key in sorted(daily_work_hours.keys()):
                    work_hours = daily_work_hours[date_key]
                    if work_hours > 0:
                        # Get actual billed hours for this day from timesheet
                        # Date format in timesheet is "M/D/YYYY", date_key is "YYYY-MM-DD"
                        # Need to convert date_key to match timesheet format
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(date_key)
                            timesheet_date_key = dt.strftime('%-m/%-d/%Y')  # macOS/Linux format (no leading zeros)
                        except:
                            timesheet_date_key = None
                        
                        billed_hours_for_day = daily_billed_hours.get(timesheet_date_key, 0)
                        
                        if billed_hours_for_day > 0:
                            dates.append(date_key)
                            billed_by_day.append(round(billed_hours_for_day, 2))  # ACTUAL daily billed hours!
                            actual_by_day.append(round(work_hours, 2))
                        else:
                            print(f"  Warning: No billed hours found for {date_key} (timesheet key: {timesheet_date_key})")
                
                # Calculate sum and compare to project total
                daily_sum = sum(actual_by_day)
                billed_sum = sum(billed_by_day)
                
                print(f"  Result: {len(dates)} days with data")
                print(f"  Daily billed sum: {billed_sum:.2f}h vs Timesheet total: {total_billed_hours}h")
                print(f"  Daily actual sum: {daily_sum:.2f}h vs Project total: {total_work_hours}h")
                
                # If there's a significant difference, scale the daily values proportionally
                # This maintains daily variation while matching the project total
                if daily_sum > 0 and abs(daily_sum - total_work_hours) > 0.5:
                    scale_factor = total_work_hours / daily_sum
                    print(f"  Applying scale factor {scale_factor:.3f} to match project total")
                    actual_by_day = [round(h * scale_factor, 2) for h in actual_by_day]
                    final_sum = sum(actual_by_day)
                    print(f"  Final sum after scaling: {final_sum:.2f}h")
                
                daily_data[ts_name] = {
                    'dates': dates,
                    'billed_hours': billed_by_day,
                    'actual_hours': actual_by_day
                }
            except Exception as e:
                print(f"[BILLED_VS_ACTUAL_TS ERROR] Failed to process '{ts_proj.get('name', 'unknown')}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[BILLED_VS_ACTUAL_TS DEBUG] Built daily data for {len(daily_data)} projects")
        
        return {
            'daily': {'projects': daily_data}
        }
    
    def _load_timesheet_data(self) -> Dict[str, Any]:
        """
        Load and parse timesheet CSV for billing/efficiency tracking.
        Checks modification time and invalidates analytics cache if timesheet changed.
        
        Returns:
            {
                "projects": [
                    {
                        "name": "mojo-1",
                        "total_hours": 70,
                        "starting_images": 19183,
                        "images_per_hour": 274.0,
                        "hours_per_image": 0.0036
                    },
                    ...
                ],
                "totals": {
                    "total_hours": 231,
                    "total_projects": 15
                }
            }
        """
        print(f"[TIMESHEET DEBUG] Loading timesheet from: {self.timesheet_parser.csv_path}")
        try:
            if not self.timesheet_parser.csv_path.exists():
                print(f"[TIMESHEET ERROR] CSV file does not exist: {self.timesheet_parser.csv_path}")
                return {"projects": [], "totals": {"total_hours": 0, "total_projects": 0}}
            
            # Check if timesheet was modified - invalidate caches if so
            current_mtime = int(self.timesheet_parser.csv_path.stat().st_mtime)
            if self._timesheet_mtime is not None and self._timesheet_mtime != current_mtime:
                print(f"[TIMESHEET CACHE] Timesheet modified - invalidating caches")
                # Clear file ops cache
                self._cached_file_ops = None
                self._cached_file_ops_for_daily = None
                # Clear project metrics cache
                self.project_agg._cache_key = None
                self.project_agg._cache_value = {}
            
            self._timesheet_mtime = current_mtime
            
            data = self.timesheet_parser.parse()
            print(f"[TIMESHEET DEBUG] Loaded {len(data.get('projects', []))} projects, total hours: {data.get('totals', {}).get('total_hours', 0)}")
            print(f"[TIMESHEET DEBUG] First 3 projects: {[p['name'] for p in data.get('projects', [])[:3]]}")
            return data
        except Exception as e:
            import traceback
            print(f"[TIMESHEET ERROR] Failed to load timesheet data: {e}")
            print(f"[TIMESHEET ERROR] Traceback: {traceback.format_exc()}")
            return {"projects": [], "totals": {"total_hours": 0, "total_projects": 0}}


def main():
    """Test the analytics engine."""
    analytics = DashboardAnalytics(project_root)
    
    print("🔍 Testing analytics engine...")
    
    # Test daily view
    data = analytics.generate_dashboard_response(
        time_slice="D",
        lookback_days=7,
        project_id=None
    )
    
    print(f"\n✅ Generated response with {len(data.get('charts', {}).get('by_script', {}))} tools")
    print(f"✅ Projects: {[p['projectId'] for p in data.get('projects', [])]}")
    print(f"✅ Baseline labels (D): {len(data['metadata']['baseline_labels']['D'])} days")
    
    # Print sample chart data
    if data.get("charts", {}).get("by_script"):
        sample_tool = list(data["charts"]["by_script"].keys())[0]
        sample_data = data["charts"]["by_script"][sample_tool]
        print(f"\n📊 Sample tool: {sample_tool}")
        print(f"   Dates: {len(sample_data['dates'])} labels")
        print(f"   Total files: {sum(sample_data['counts'])}")
    
    # Save sample output
    import json
    output_file = project_root / "dashboard_analytics_sample.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n💾 Sample output saved to: {output_file}")


if __name__ == "__main__":
    main()

