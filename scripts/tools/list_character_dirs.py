#!/usr/bin/env python3
"""Quick script to list character subdirectories in __selected/

USAGE:
------
  python scripts/tools/list_character_dirs.py

FEATURES:
---------
• Lists all subdirectories in __selected/
• Counts the number of images in each subdirectory
• Prints the subdirectory name and image count
• Prints the total number of subdirectories
• Prints the command to process one subdirectory

"""

from pathlib import Path

selected_dir = Path("__selected")

if not selected_dir.exists():
    print(f"❌ {selected_dir} doesn't exist")
    exit(1)

subdirs = [d for d in selected_dir.iterdir() if d.is_dir()]
subdirs.sort()

print(f"📁 Found {len(subdirs)} subdirectories in __selected/:\n")

for subdir in subdirs:
    png_count = len(list(subdir.glob("*.png")))
    print(f"  {subdir.name:30s} - {png_count:4d} images")
    # Copy-friendly path on the next line
    print(f"  __selected/{subdir.name}")

print(f"\n✅ Total: {len(subdirs)} directories\n")
print("To process one:")
print("  python scripts/03_web_character_sorter.py __selected/DIRNAME/")
