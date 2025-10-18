#!/usr/bin/env python3
"""
Dashboard Timing Test
====================
Runs the dashboard data generation to capture timing breakdowns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.analytics import DashboardAnalytics

def main():
    print("=" * 70)
    print("DASHBOARD TIMING TEST")
    print("=" * 70)
    print("\nThis will generate dashboard data and show timing for each step.\n")
    
    # Initialize analytics
    print("Initializing DashboardAnalytics...")
    analytics = DashboardAnalytics(project_root)
    
    # Generate dashboard response (this triggers all the timing logs)
    print("\nGenerating dashboard response...\n")
    
    response = analytics.generate_dashboard_response(
        time_slice='D',
        lookback_days=30,
        project_id=None
    )
    
    print("\n" + "=" * 70)
    print("RESPONSE SUMMARY")
    print("=" * 70)
    print(f"Projects: {len(response.get('projects', []))}")
    print(f"Charts: {len(response.get('charts', {}).get('by_script', {}))}")
    print(f"Project comparisons: {len(response.get('project_comparisons', []))}")
    print(f"Timesheet projects: {len(response.get('timesheet_data', {}).get('projects', []))}")
    print("=" * 70)

if __name__ == '__main__':
    main()

