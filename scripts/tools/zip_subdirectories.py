#!/usr/bin/env python3
"""
Zip all subdirectories in a target directory.

Usage:
    python scripts/tools/zip_subdirectories.py /Volumes/T7Shield/Eros/finished --dry-run
    python scripts/tools/zip_subdirectories.py /Volumes/T7Shield/Eros/finished
"""

import argparse
import shutil
import sys
from pathlib import Path


def zip_subdirectories(target_dir: Path, dry_run: bool = False):
    """Create zip files for each subdirectory in target_dir."""
    if not target_dir.exists():
        print(f"❌ Directory not found: {target_dir}")
        return 1

    if not target_dir.is_dir():
        print(f"❌ Not a directory: {target_dir}")
        return 1

    # Find all subdirectories (not files)
    subdirs = [d for d in target_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"📁 No subdirectories found in {target_dir}")
        return 0

    print(f"📦 Found {len(subdirs)} subdirectories to zip")
    print(f"🧪 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    zipped_count = 0
    skipped_count = 0

    for subdir in sorted(subdirs):
        zip_name = subdir.name
        zip_path = target_dir / f"{zip_name}.zip"

        # Check if zip already exists
        if zip_path.exists():
            print(f"⏭️  SKIP: {zip_name} (zip already exists)")
            skipped_count += 1
            continue

        if dry_run:
            print(f"📦 Would create: {zip_path.name}")
        else:
            try:
                print(f"📦 Zipping: {subdir.name} ...", end=" ", flush=True)
                # Create zip file (without .zip extension, shutil adds it)
                shutil.make_archive(
                    str(target_dir / zip_name),
                    "zip",
                    root_dir=subdir.parent,
                    base_dir=subdir.name,
                )
                print("✅")
                zipped_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    print("=" * 60)
    if dry_run:
        print(f"🧪 DRY RUN: Would create {len(subdirs) - skipped_count} zip files")
        if skipped_count:
            print(f"   Skipped: {skipped_count} (already exist)")
    else:
        print(f"✅ Created {zipped_count} zip files")
        if skipped_count:
            print(f"⏭️  Skipped {skipped_count} (already exist)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Zip all subdirectories in a target directory"
    )
    parser.add_argument(
        "target_dir", type=Path, help="Directory containing subdirectories to zip"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be zipped without creating files",
    )

    args = parser.parse_args()

    return zip_subdirectories(args.target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
