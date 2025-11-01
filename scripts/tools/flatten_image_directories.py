#!/usr/bin/env python3
"""Flatten Image Directories (companion-safe)

USAGE:
  python scripts/tools/flatten_image_directories.py __selected
  python scripts/tools/flatten_image_directories.py /path/to/dir --dry-run
  python scripts/tools/flatten_image_directories.py __selected --yes   # skip prompt

WHAT IT DOES:
  - For each immediate subdirectory of the given directory, move all images
    (PNG/JPG/JPEG) from that subdirectory into the parent directory.
  - Moves are COMPANION-SAFE using the shared utilities (PNG + all sidecars).
  - Operations are logged via FileTracker. Conflicting filenames are skipped.

Notes:
  - Only immediate subdirectories are flattened (one level).
  - Use --dry-run to preview; use --yes to skip confirmation.
"""

import argparse
import sys
from pathlib import Path

# Ensure project imports resolve
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from send2trash import send2trash  # type: ignore

    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False

from scripts.file_tracker import FileTracker
from scripts.utils.companion_file_utils import move_multiple_files_with_companions


def _scan_immediate_subdirs(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def _find_images(dir_path: Path) -> list[Path]:
    return (
        sorted([p for p in dir_path.glob("*.png")])
        + sorted([p for p in dir_path.glob("*.jpg")])
        + sorted([p for p in dir_path.glob("*.jpeg")])
    )


def flatten_once(base_dir: Path, dry_run: bool = False) -> dict:
    """Flatten one level of subdirectories into base_dir.

    Returns summary dict with totals.
    """
    tracker = FileTracker("flatten_image_directories")
    subdirs = _scan_immediate_subdirs(base_dir)

    total_images = 0
    moved = 0
    skipped = 0
    errors = 0

    print(f"📂 Base directory: {base_dir}")
    print(f"📁 Subdirectories: {len(subdirs)}\n")

    for i, sd in enumerate(subdirs, 1):
        images = _find_images(sd)
        if not images:
            continue
        total_images += len(images)
        print(
            f"[{i:3d}/{len(subdirs)}] {sd.name}: {len(images)} images → {base_dir.name}"
        )

        res = move_multiple_files_with_companions(
            images, base_dir, dry_run=dry_run, tracker=tracker
        )

        moved += int(res.get("moved", 0))
        skipped += int(res.get("skipped", 0))
        errors += int(res.get("errors", 0))

    # Remove empty immediate subdirectories when LIVE (not dry-run)
    empty_deleted = 0
    if not dry_run:
        if not _SEND2TRASH_AVAILABLE:
            print(
                "[!] send2trash not installed; skipping empty directory cleanup (pip install send2trash)"
            )
        else:
            for sd in subdirs:
                try:
                    entries = list(sd.iterdir())
                    if not entries:
                        # Truly empty
                        send2trash(str(sd))
                        empty_deleted += 1
                        try:
                            # Best-effort logging
                            tracker.log_operation(
                                operation="send_to_trash",
                                source_dir=str(base_dir.name),
                                dest_dir="trash",
                                file_count=0,
                                files=[sd.name],
                                notes="empty directory cleanup",
                            )
                        except Exception:
                            pass
                    else:
                        # Consider macOS ignorable files as empty (.DS_Store, AppleDouble, .localized)
                        ignorable = {".DS_Store", ".localized"}
                        non_ignorable = [
                            p
                            for p in entries
                            if not (p.name in ignorable or p.name.startswith("._"))
                        ]
                        if len(non_ignorable) == 0:
                            # Send ignorable files to Trash first
                            for p in entries:
                                try:
                                    send2trash(str(p))
                                except Exception:
                                    pass
                            # Now send the directory to Trash
                            send2trash(str(sd))
                            empty_deleted += 1
                            try:
                                tracker.log_operation(
                                    operation="send_to_trash",
                                    source_dir=str(base_dir.name),
                                    dest_dir="trash",
                                    file_count=0,
                                    files=[sd.name],
                                    notes="empty directory cleanup (ignorable files)",
                                )
                            except Exception:
                                pass
                except Exception as exc:
                    print(f"[!] Failed to remove empty directory {sd.name}: {exc}")

    return {
        "subdirs": len(subdirs),
        "total_images": total_images,
        "moved": moved,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
        "empty_dirs_deleted": empty_deleted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten image subdirectories into the parent (companion-safe)",
    )
    parser.add_argument("directory", help="Directory to flatten (e.g., __selected)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    base_dir = Path(args.directory).expanduser().resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"❌ Not a directory: {base_dir}")
        sys.exit(1)

    # Guard: do not allow using the same source/destination pattern implicitly —
    # flattening is per-subdir into base_dir, which is safe.

    print("🚀 FLATTEN IMAGE DIRECTORIES")
    print("=" * 50)
    print(f"Target directory: {base_dir}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be moved)")
    print()

    if not args.dry_run and not args.yes:
        resp = (
            input(
                "Proceed to move images from all subdirectories into this directory? [y/N]: "
            )
            .strip()
            .lower()
        )
        if resp not in {"y", "yes"}:
            print("Cancelled.")
            sys.exit(0)

    summary = flatten_once(base_dir, dry_run=args.dry_run)

    print()
    print("📊 SUMMARY")
    print(f"   • Subdirectories scanned: {summary['subdirs']}")
    print(f"   • Images discovered:      {summary['total_images']}")
    print(f"   • Moved successfully:     {summary['moved']}")
    print(f"   • Skipped (conflicts):    {summary['skipped']}")
    if summary["errors"]:
        print(f"   • Errors:                 {summary['errors']}")
    print(
        "   • Mode:                   DRY RUN"
        if summary["dry_run"]
        else "   • Mode:                   LIVE"
    )
    if not summary["dry_run"]:
        print(f"   • Empty dirs deleted:     {summary['empty_dirs_deleted']}")


if __name__ == "__main__":
    main()
