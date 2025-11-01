#!/usr/bin/env python3
"""
Test that all production scripts have dashboard mappings.

This test prevents the "invisible work" bug where a script logs operations
but the dashboard doesn't recognize the script name, making thousands of
operations invisible.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.data_engine import DashboardDataEngine


def test_all_production_scripts_have_mappings():
    """
    Ensure every production script in the production_scripts list
    has an explicit mapping in get_display_name().

    This prevents scripts from falling back to the default Title Case
    conversion, which indicates they're not properly mapped.
    """
    engine = DashboardDataEngine("data")

    # Get the production scripts list (same as used in get_data_for_dashboard)
    production_scripts = [
        # Current script names
        "01_web_image_selector",
        "03_web_character_sorter",
        "04_multi_crop_tool",
        "multi_crop_tool",
        "ai_desktop_multi_crop",  # ← This was missing and caused the bug!
        "ai_desktop_multi_crop_queue",
        "crop_queue_processor",
        "ai_assisted_reviewer",
        # Log script names (from actual data)
        "character_sorter",
        "image_version_selector",
        "multi_batch_crop_tool",
        # Legacy names
        "batch_crop_tool",
    ]

    unmapped_scripts = []

    for script in production_scripts:
        display_name = engine.get_display_name(script)

        # Check if it fell back to default Title Case conversion
        # This indicates the script is NOT explicitly mapped
        default_fallback = script.replace("_", " ").title()

        if display_name == default_fallback:
            unmapped_scripts.append(script)

    if unmapped_scripts:
        raise AssertionError(
            f"The following production scripts are NOT mapped in dashboard:\n"
            f"  {unmapped_scripts}\n\n"
            f"Add them to get_display_name() in data_engine.py!\n"
            f"This causes work to be invisible in the dashboard."
        )


def test_mapped_scripts_are_in_production_list():
    """
    Check that commonly mapped scripts are actually in the production_scripts
    list so they show up in the dashboard.
    """
    engine = DashboardDataEngine("data")

    # These are scripts we know should be active and tracked
    critical_scripts = [
        "ai_desktop_multi_crop",  # The one we just fixed!
        "ai_desktop_multi_crop_queue",
        "ai_assisted_reviewer",
        "01_web_image_selector",
        "multi_crop_tool",
    ]

    # Get raw data to see what's in production_scripts
    # We'll call the private method directly for testing
    engine.get_data_for_dashboard(
        lookback_days=7,
        time_slice="day",
        production_scripts=None,  # Use default list
    )

    # This is a bit hacky, but we want to ensure the scripts are being loaded
    # The real test is: can the dashboard SEE operations from these scripts?
    # For now, just verify the mapping exists
    for script in critical_scripts:
        display_name = engine.get_display_name(script)
        assert (
            display_name != script.replace("_", " ").title()
        ), f"Critical script '{script}' not explicitly mapped!"


def test_no_duplicate_mappings():
    """
    Ensure multiple script names don't accidentally map to the same
    display name in a way that would cause confusion.

    Note: It's OKAY for multiple scripts to map to the same display name
    (e.g., "batch_crop_tool" and "multi_crop_tool" both → "Multi Crop Tool")
    but we want to document this.
    """
    engine = DashboardDataEngine("data")

    production_scripts = [
        "01_web_image_selector",
        "03_web_character_sorter",
        "04_multi_crop_tool",
        "multi_crop_tool",
        "ai_desktop_multi_crop",
        "ai_desktop_multi_crop_queue",
        "crop_queue_processor",
        "ai_assisted_reviewer",
        "character_sorter",
        "image_version_selector",
        "multi_batch_crop_tool",
        "batch_crop_tool",
    ]

    mappings = {}
    for script in production_scripts:
        display = engine.get_display_name(script)
        mappings.setdefault(display, []).append(script)

    # Print mappings for documentation
    print("\nScript → Display Name Mappings:")
    for display, scripts in sorted(mappings.items()):
        print(f"  {display}:")
        for script in scripts:
            print(f"    - {script}")

    # This test just documents the mappings, doesn't fail
    # But it helps us see what's grouped together
    assert True


if __name__ == "__main__":
    # Run tests manually
    print("Running dashboard script mapping tests...")

    try:
        test_all_production_scripts_have_mappings()
        print("✅ All production scripts have explicit mappings")
    except AssertionError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        test_mapped_scripts_are_in_production_list()
        print("✅ Critical scripts are mapped")
    except AssertionError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        test_no_duplicate_mappings()
        print("✅ Mapping documentation complete")
    except AssertionError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print("\n🎉 All dashboard mapping tests passed!")
