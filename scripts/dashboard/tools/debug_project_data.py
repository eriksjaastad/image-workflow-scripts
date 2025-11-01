#!/usr/bin/env python3
"""
Debug Project Data Flow
=======================
Traces data from raw logs → project metrics → dashboard output for ANY project.

Usage:
    python scripts/dashboard/tools/debug_project_data.py [project_id]

Examples:
        python scripts/dashboard/tools/debug_project_data.py mojo2
        python scripts/dashboard/tools/debug_project_data.py mixed-0919
        python scripts/dashboard/tools/debug_project_data.py  # (shows all projects)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from collections import defaultdict

from scripts.dashboard.analytics import DashboardAnalytics
from scripts.dashboard.data_engine import DashboardDataEngine
from scripts.dashboard.project_metrics_aggregator import ProjectMetricsAggregator


def debug_project(project_id=None):
    """Debug data flow for a specific project or all projects."""
    print("=" * 80)
    print("PROJECT DATA FLOW DEBUG")
    if project_id:
        print(f"Project: {project_id}")
    else:
        print("Showing all projects")
    print("=" * 80)

    # Step 1: Raw file operations
    print("\n1️⃣  RAW FILE OPERATIONS")
    print("-" * 80)
    engine = DashboardDataEngine(str(Path.cwd()))
    all_ops = engine.load_file_operations()

    if project_id:
        # Filter to specific project
        project_ops = [
            op
            for op in all_ops
            if project_id.lower() in str(op.get("source_dir", "")).lower()
            or project_id.lower() in str(op.get("dest_dir", "")).lower()
            or project_id.lower() in str(op.get("working_dir", "")).lower()
        ]

        print(f"Total operations: {len(all_ops)}")
        print(f"{project_id} operations: {len(project_ops)}")

        # Group by script
        proj_by_script = defaultdict(list)
        for op in project_ops:
            script = op.get("script", "unknown")
            proj_by_script[script].append(op)

        print(f"\n{project_id} operations by script:")
        for script, ops in sorted(
            proj_by_script.items(), key=lambda x: len(x[1]), reverse=True
        ):
            dates = sorted(set(str(o.get("date", "N/A")) for o in ops))
            file_counts = [o.get("file_count", 0) for o in ops if o.get("file_count")]
            total_files = sum(file_counts) if file_counts else 0
            print(f"  {script}: {len(ops)} operations, {total_files} files")
            if dates:
                print(f"    Date range: {dates[0]} to {dates[-1]}")
    else:
        # Show summary of all projects
        print(f"Total operations: {len(all_ops)}")

    # Step 2: Project Metrics Aggregator
    print("\n2️⃣  PROJECT METRICS AGGREGATOR")
    print("-" * 80)
    pm_agg = ProjectMetricsAggregator(Path.cwd())
    project_metrics = pm_agg.aggregate()

    if project_id:
        if project_id in project_metrics:
            pm = project_metrics[project_id]
            print(f"{project_id} found in project metrics!")
            print(f"  Started: {pm.get('startedAt', 'N/A')}")
            print(f"  Status: {pm.get('status', 'N/A')}")
            print("\n  Tools section:")
            for tool_key, tool_data in pm.get("tools", {}).items():
                print(f"    {tool_key}:")
                print(f"      images_processed: {tool_data.get('images_processed')}")
                print(
                    f"      images_per_hour: {tool_data.get('images_per_hour', 0):.2f}"
                )
        else:
            print(f"❌ {project_id} NOT found in project metrics")
            print(f"\nAvailable projects: {', '.join(sorted(project_metrics.keys()))}")
    else:
        print(f"Total projects: {len(project_metrics)}")
        print(f"Projects: {', '.join(sorted(project_metrics.keys()))}")

    # Step 3: Dashboard Output
    print("\n3️⃣  DASHBOARD OUTPUT")
    print("-" * 80)
    analytics = DashboardAnalytics(Path.cwd())
    response = analytics.generate_dashboard_response("D", 30, None)

    if project_id:
        # Find in productivity overview
        proj_overview = next(
            (
                p
                for p in response.get("productivity_overview", [])
                if p.get("projectId") == project_id
            ),
            None,
        )

        if proj_overview:
            print(f"{project_id} in productivity_overview:")
            print("  tool_metrics:")
            for tool, iph in proj_overview.get("tool_metrics", {}).items():
                print(f"    {tool}: {iph} img/h")
            print(f"  overall_img_h: {proj_overview.get('overall_img_h')}")
        else:
            print(f"❌ {project_id} NOT in productivity_overview")

        # Find in table
        proj_table = next(
            (
                p
                for p in response.get("project_productivity_table", [])
                if p.get("projectId") == project_id
            ),
            None,
        )

        if proj_table:
            print(f"\n{project_id} in project_productivity_table:")
            print(f"  Started: {proj_table.get('startedAt')}")
            print("  Tools:")
            for tool_name, tool_data in proj_table.get("tools", {}).items():
                if "images" in tool_data:
                    print(
                        f"    {tool_name}: {tool_data['images']} images in {tool_data.get('hours', 0)}h ({tool_data.get('days', 0)}d)"
                    )
                elif "images_total" in tool_data:
                    print(
                        f"    {tool_name}: {tool_data['images_total']} images in {tool_data.get('hours', 0)}h ({tool_data.get('days', 0)}d)"
                    )
                    print(
                        f"      Selected: {tool_data.get('images_selected', 0)}, Cropped: {tool_data.get('images_cropped', 0)}"
                    )
        else:
            print(f"❌ {project_id} NOT in project_productivity_table")
    else:
        print(
            f"Projects in productivity_overview: {len(response.get('productivity_overview', []))}"
        )
        print(
            f"Projects in project_productivity_table: {len(response.get('project_productivity_table', []))}"
        )

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    project_id = sys.argv[1] if len(sys.argv) > 1 else None
    debug_project(project_id)
