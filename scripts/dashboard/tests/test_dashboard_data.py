#!/usr/bin/env python3
"""
🧪 Test Dashboard Data Engine
Test the data processing without Flask dependencies
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scripts.dashboard.engines.data_engine import DashboardDataEngine


def main():
    print("🧪 Testing Dashboard Data Engine...")
    print("=" * 50)

    try:
        # Initialize data engine
        engine = DashboardDataEngine("..")  # Point to scripts/ from scripts/dashboard/

        print("📊 Generating dashboard data...")
        data = engine.generate_dashboard_data(time_slice="D", lookback_days=14)

        print("\n✅ Data generated successfully!")
        print(f"📈 Metadata: {data['metadata']}")

        # Show file operations by script
        if data["file_operations_data"].get("by_script"):
            print(
                f"\n📊 Files by Script ({len(data['file_operations_data']['by_script'])} records):"
            )
            for record in data["file_operations_data"]["by_script"][-5:]:  # Show last 5
                script_display = {
                    "image_version_selector": "01_web_image_selector",
                    "character_sorter": "03_web_character_sorter",
                    "batch_crop_tool": "04_batch_crop_tool",
                }.get(record["script"], record["script"])

                print(
                    f"  📅 {record['time_slice']}: {script_display} → {record['file_count']:,} files"
                )

        # Show operations by type
        if data["file_operations_data"].get("by_operation"):
            print(
                f"\n⚡ Operations by Type ({len(data['file_operations_data']['by_operation'])} records):"
            )
            for record in data["file_operations_data"]["by_operation"][
                -5:
            ]:  # Show last 5
                print(
                    f"  🔧 {record['time_slice']}: {record['operation']} → {record['file_count']:,} files"
                )

        # Calculate totals
        total_files = 0
        if data["file_operations_data"].get("by_script"):
            for record in data["file_operations_data"]["by_script"]:
                total_files += record["file_count"] or 0

        print("\n🎯 Summary:")
        print(f"  📁 Total files processed: {total_files:,}")
        print(f"  📊 Time slice: {data['metadata']['time_slice']}")
        data_range = data["metadata"].get("data_range", {})
        if data_range.get("file_ops_start") and data_range.get("file_ops_end"):
            print(
                f"  📅 Date range: {data_range['file_ops_start']} to {data_range['file_ops_end']}"
            )
        else:
            print("  📅 Date range: Available")

        print("\n🚀 Dashboard data engine is working perfectly!")
        print("💡 Ready for Flask web interface (requires: pip install flask)")

    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
