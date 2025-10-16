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


# Centralized tool order - used across ALL charts, tables, and toggles
# This ensures consistent ordering everywhere in the dashboard
STANDARD_TOOL_ORDER = [
    'Desktop Image Selector Crop',
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
        # Get raw data from engine
        raw_data = self.engine.generate_dashboard_data(
            time_slice=time_slice,
            lookback_days=lookback_days,
            project_id=project_id
        )
        
        # Get project metrics (now includes daily summaries merged with logs)
        project_metrics = self.project_agg.aggregate()
        
        # Build baseline labels for this time slice
        baseline_labels = self.engine.build_time_labels(time_slice, lookback_days)
        
        # Transform to UI contract
        response = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_slice": time_slice,
                "lookback_days": lookback_days,
                "hours_scope": "lifetime",  # table hours/days computed over full project lifetime
                "hours_source": "file_operations",  # derived from file operation timing (break-aware)
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
            "project_comparisons": self._build_project_comparisons(project_metrics),
            "project_kpi": self._build_project_kpi(project_id, project_metrics, raw_data),
            "project_metrics": project_metrics,
            "project_markers": self._build_project_markers(project_id, raw_data),
            "project_productivity_table": self._build_project_productivity_table(raw_data, project_metrics, time_slice, lookback_days)
        }
        
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
            value_field="file_count"
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
        value_field: str
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
                # Use display name if group_field is script
                if group_field == "script":
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
        project_metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build project comparison data for simple bar chart.
        
        Returns total operations per project (not images/hour which is broken).
        
        Returns:
            [
                {
                    "projectId": "abc",
                    "title": "Project ABC",
                    "total_operations": <int>,
                    "operations_by_type": { "crop": <int>, "delete": <int>, ... }
                }
            ]
        """
        comparisons = []
        
        for project_id, metrics in project_metrics.items():
            title = metrics.get("title", project_id)
            ops_by_type = metrics.get("totals", {}).get("operations_by_type", {})
            total_ops = sum(ops_by_type.values())
            
            comparisons.append({
                "projectId": project_id,
                "title": title,
                "total_operations": total_ops,
                "operations_by_type": ops_by_type
            })
        
        # Sort by total operations (descending)
        comparisons.sort(key=lambda x: x.get("total_operations", 0), reverse=True)
        
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
        
        # Pull ALL file operations (full project lifetime), then filter to project by date range
        # This matches the requirement: sum total time each tool has been used for the project
        window_ops = self.engine.load_file_operations()
        
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
                if display_name in ("Web Image Selector", "Desktop Image Selector Crop"):
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

            if display_name in ("Web Image Selector", "Desktop Image Selector Crop"):
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
        
        # Load all file operations and filter by date range and tool
        all_ops = self.engine.load_file_operations()
        
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

