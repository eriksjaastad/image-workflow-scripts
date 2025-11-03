#!/usr/bin/env python3
"""Clean up all sandbox test data.

Safely removes sandbox directories used for testing without affecting
production data. Requires explicit confirmation unless --force is used.

Usage:
    # Interactive cleanup (asks for confirmation)
    python scripts/tools/cleanup_sandbox.py

    # Force cleanup without confirmation
    python scripts/tools/cleanup_sandbox.py --force

    # Dry run (show what would be deleted)
    python scripts/tools/cleanup_sandbox.py --dry-run

Directories cleaned:
    - data/projects/sandbox/
    - data/file_operations_logs/sandbox/
    - data/test_runs/
"""
import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.sandbox_mode import SandboxConfig


def cleanup_sandbox(confirm: bool = True, dry_run: bool = False) -> int:
    """Remove all sandbox test data.

    Args:
        confirm: If True, ask for user confirmation before deleting
        dry_run: If True, only show what would be deleted

    Returns:
        Exit code (0 for success, 1 for cancelled/error)
    """
    # Find project root (2 levels up from this script)
    project_root = Path(__file__).resolve().parents[2]

    # Define sandbox directories to clean
    sandbox_dirs = [
        project_root / "data" / "projects" / "sandbox",
        project_root / "data" / "file_operations_logs" / "sandbox",
        project_root / "data" / "test_runs",
    ]

    # Filter to only existing directories
    existing_dirs = [d for d in sandbox_dirs if d.exists()]

    if not existing_dirs:
        print("✓ No sandbox directories found - nothing to clean")
        return 0

    # Verify all directories have sandbox markers (safety check)
    dirs_without_markers = [d for d in existing_dirs if not SandboxConfig.has_marker_file(d)]

    if dirs_without_markers:
        print("⚠️  ERROR: Some directories are missing .sandbox_marker files:", file=sys.stderr)
        for d in dirs_without_markers:
            print(f"   - {d}", file=sys.stderr)
        print("\nThese directories will NOT be deleted (safety check).", file=sys.stderr)
        print("If you're sure these are sandbox directories, add .sandbox_marker files manually.", file=sys.stderr)

        # Remove unmarked directories from the list
        existing_dirs = [d for d in existing_dirs if SandboxConfig.has_marker_file(d)]

        if not existing_dirs:
            print("\n✗ No safe-to-delete sandbox directories found", file=sys.stderr)
            return 1

    # Show what will be deleted
    print("⚠️  The following sandbox directories will be DELETED:")
    total_size = 0
    for dir_path in existing_dirs:
        try:
            # Calculate size
            size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            file_count = len(list(dir_path.rglob("*")))

            print(f"   - {dir_path}")
            print(f"     ({file_count} files, {size_mb:.2f} MB)")
        except (OSError, PermissionError) as e:
            print(f"   - {dir_path}")
            print(f"     (Could not calculate size: {e})")

    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")

    # Dry run - show only
    if dry_run:
        print("\n[DRY-RUN] No files were deleted")
        return 0

    # Confirmation prompt
    if confirm:
        print("\n⚠️  This action cannot be undone!")
        response = input("Continue? (yes/no): ").strip().lower()
        if response != "yes":
            print("Cancelled.")
            return 1

    # Delete directories
    print()
    for dir_path in existing_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"✓ Deleted {dir_path}")
        except (OSError, PermissionError) as e:
            print(f"✗ Failed to delete {dir_path}: {e}", file=sys.stderr)
            return 1

    print("\n✓ Sandbox cleanup complete")
    return 0


def main() -> int:
    """Main entry point for sandbox cleanup tool.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Clean up sandbox test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    return cleanup_sandbox(confirm=not args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
