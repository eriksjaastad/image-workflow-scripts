#!/usr/bin/env python3
"""
🚀 Dashboard Launcher - Unified Tabbed Interface
Launch the web-based dashboard with Current Project and Productivity tabs

HOW TO RUN:
===========
1. Activate your virtual environment: source .venv311/bin/activate
2. Navigate to dashboard: cd scripts/dashboard
3. Run dashboard: python scripts/dashboard/run_dashboard.py
4. Open browser: http://localhost:5001

OR run with options:
python run_dashboard.py --port 8080 --host 0.0.0.0

Note: Flask should already be installed in your .venv311 environment.
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dashboard_app import UnifiedDashboard


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Launch Productivity Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py                    # Default: localhost:5001
  python run_dashboard.py --port 8080       # Custom port
  python run_dashboard.py --host 0.0.0.0    # Allow external connections
  
The dashboard shows:
  📊 Files processed by tool (01_web_image_selector, 03_web_character_sorter, 04_batch_crop_tool)
  ⚡ Operations by type (delete, crop, sort, move)
  📈 Time-series analysis with configurable time slices
  🎛️ Interactive controls for daily/weekly/monthly views
        """,
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port", type=int, default=5001, help="Port to bind to (default: 5001)"
    )

    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip pre-start snapshot extraction/aggregation (use existing data)",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to project root directory (default: repository root inferred from this script)",
    )

    args = parser.parse_args()

    # Resolve data directory robustly: default to repo root based on this file's location
    inferred_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else inferred_root

    print("🚀 Starting Unified Dashboard...")
    print(f"📊 Data source: {data_dir}")

    # Auto-update snapshots before dashboard loads
    skip_snapshots = (
        bool(os.environ.get("DASHBOARD_SKIP_SNAPSHOTS")) or args.skip_snapshots
    )
    if not skip_snapshots:
        print("🔄 Updating snapshots from raw logs...")
        try:
            import subprocess

            scripts_dir = data_dir / "scripts" / "data_pipeline"

            # Run extraction scripts (fast, only processes new data)
            subprocess.run(
                [sys.executable, str(scripts_dir / "extract_operation_events_v1.py")],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                [sys.executable, str(scripts_dir / "build_daily_aggregates_v1.py")],
                capture_output=True,
                check=False,
            )
            print("✅ Snapshots updated")
        except Exception as e:
            print(f"⚠️  Snapshot update skipped: {e}")
            print("   (Dashboard will use existing snapshots)")
    else:
        print("⏭  Skipping snapshot updates (test mode)")

    print(f"🌐 URL: http://{args.host}:{args.port}")
    print("🎨 Theme: Dark mode with Erik's style guide")
    print()
    print("📊 Tabs:")
    print("   • Current Project - Real-time progress & phase tracking")
    print("   • Productivity - Historical analytics & cross-project metrics")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)

    try:
        # Create and run dashboard
        dashboard = UnifiedDashboard(data_dir=str(data_dir))
        dashboard.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            auto_open=not args.no_browser,
        )

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
