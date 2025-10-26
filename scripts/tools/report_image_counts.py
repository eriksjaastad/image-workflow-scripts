#!/usr/bin/env python3
"""
One-shot Image Count and Companion Integrity Report
===================================================

Purpose:
- Count images across key workflow directories
- Validate companion files (.yaml and .caption) per image
- Summarize recent file operations from FileTracker logs
- Optionally write a JSON report to data/daily_summaries/

Safe by design:
- Read-only for production images and companions
- Creates NEW report files only under data/daily_summaries/

Usage:
  python scripts/tools/report_image_counts.py \
    --days 7 \
    --write-report

Args:
  --dirs: optional list of directories to scan (defaults to common workflow dirs)
  --days: how many days of FileTracker history to summarize (default: 7)
  --write-report: write JSON report to data/daily_summaries/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class DirectoryCounts:
    directory: str
    total_images: int
    with_yaml: int
    missing_yaml: int
    with_caption: int
    missing_caption: int
    images_modified_within_days: int


@dataclass
class FileOperationSummary:
    total_entries: int
    by_operation: Dict[str, int]
    recent_entries: int
    recent_by_operation: Dict[str, int]


def is_image_file(path: Path) -> bool:
    try:
        return path.suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def has_yaml_companion(image_path: Path) -> bool:
    yaml_path = image_path.with_suffix(image_path.suffix + ".yaml")
    if yaml_path.exists():
        return True
    # Common alternative: sometimes companions share stem only
    alt_yaml_path = image_path.with_suffix(".yaml")
    return alt_yaml_path.exists()


def has_caption_companion(image_path: Path) -> bool:
    caption_path = image_path.with_suffix(".caption")
    return caption_path.exists()


def scan_directory(dir_path: Path, recent_cutoff: datetime) -> DirectoryCounts:
    total_images = 0
    with_yaml = 0
    with_caption = 0
    images_modified_within_days = 0

    if not dir_path.exists() or not dir_path.is_dir():
        return DirectoryCounts(
            directory=str(dir_path),
            total_images=0,
            with_yaml=0,
            missing_yaml=0,
            with_caption=0,
            missing_caption=0,
            images_modified_within_days=0,
        )

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue
        if not is_image_file(file_path):
            continue

        total_images += 1
        if has_yaml_companion(file_path):
            with_yaml += 1
        if has_caption_companion(file_path):
            with_caption += 1

        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime >= recent_cutoff:
                images_modified_within_days += 1
        except Exception:
            pass

    return DirectoryCounts(
        directory=str(dir_path),
        total_images=total_images,
        with_yaml=with_yaml,
        missing_yaml=total_images - with_yaml,
        with_caption=with_caption,
        missing_caption=total_images - with_caption,
        images_modified_within_days=images_modified_within_days,
    )


def load_filetracker_summaries(log_dir: Path, days: int) -> Optional[FileOperationSummary]:
    if not log_dir.exists() or not log_dir.is_dir():
        return None

    cutoff = datetime.utcnow() - timedelta(days=days)
    total_entries = 0
    by_operation: Dict[str, int] = {}
    recent_entries = 0
    recent_by_operation: Dict[str, int] = {}

    for log_path in sorted(log_dir.glob("*.log")):
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    op_type = str(entry.get("type") or "").lower()
                    if op_type not in {"file_operation", "session_start", "session_end", "batch_start", "batch_end", "user_action", "metric_mode_update", "directory_state"}:
                        # Unknown or non-standard; still count by presence
                        pass

                    # Track file operations by operation name when present
                    operation = str(entry.get("operation") or "").lower()
                    timestamp_str = entry.get("timestamp")

                    total_entries += 1
                    if operation:
                        by_operation[operation] = by_operation.get(operation, 0) + 1

                    # Recent window
                    if timestamp_str:
                        try:
                            ts = timestamp_str
                            if isinstance(ts, str) and ts.endswith("Z"):
                                ts = ts[:-1] + "+00:00"
                            dt = datetime.fromisoformat(ts)
                            if dt >= cutoff:
                                recent_entries += 1
                                if operation:
                                    recent_by_operation[operation] = recent_by_operation.get(operation, 0) + 1
                        except Exception:
                            pass
        except Exception:
            continue

    return FileOperationSummary(
        total_entries=total_entries,
        by_operation=by_operation,
        recent_entries=recent_entries,
        recent_by_operation=recent_by_operation,
    )


def print_human_summary(dir_results: List[DirectoryCounts], op_summary: Optional[FileOperationSummary], days: int) -> None:
    print("\n" + "=" * 80)
    print("IMAGE COUNTS AND COMPANION INTEGRITY")
    print("=" * 80 + "\n")

    grand_total = sum(d.total_images for d in dir_results)
    print(f"Directories scanned: {len(dir_results)}")
    print(f"Total images:        {grand_total:,}\n")

    for d in dir_results:
        print(f"[{d.directory}]")
        print(f"  images:         {d.total_images:,}")
        print(f"  with YAML:      {d.with_yaml:,}   missing: {d.missing_yaml:,}")
        print(f"  with CAPTION:   {d.with_caption:,}   missing: {d.missing_caption:,}")
        print(f"  modified ≤{days}d: {d.images_modified_within_days:,}\n")

    if op_summary:
        print("-" * 80)
        print("FILETRACKER OPERATIONS SUMMARY")
        print(f"  total entries:  {op_summary.total_entries:,}")
        if op_summary.by_operation:
            print("  by operation:")
            for op, cnt in sorted(op_summary.by_operation.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    - {op}: {cnt:,}")
        print(f"  recent ≤{days}d: {op_summary.recent_entries:,}")
        if op_summary.recent_by_operation:
            print("  recent by operation:")
            for op, cnt in sorted(op_summary.recent_by_operation.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    - {op}: {cnt:,}")
        print()


def write_json_report(dir_results: List[DirectoryCounts], op_summary: Optional[FileOperationSummary]) -> Path:
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "directories": [asdict(d) for d in dir_results],
        "file_operations": asdict(op_summary) if op_summary else None,
    }

    out_dir = Path("data") / "daily_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"image_counts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count images and validate companions; summarize recent file ops.")
    parser.add_argument(
        "--dirs",
        nargs="*",
        help="Directories to scan (defaults to common workflow dirs)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of FileTracker history to summarize (default: 7)",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write JSON report to data/daily_summaries/",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    default_dirs = [
        "crop",
        "crop_auto",
        "selected",
        "__crop",
        "__crop_auto",
        "__selected",
        "__cropped",
        "__delete_staging",
    ]
    dir_strings = args.dirs if args.dirs else default_dirs
    target_dirs: List[Path] = [Path(d).resolve() for d in dir_strings]

    recent_cutoff = datetime.utcnow() - timedelta(days=args.days)

    # Scan directories
    dir_results: List[DirectoryCounts] = []
    for d in target_dirs:
        try:
            dir_results.append(scan_directory(d, recent_cutoff))
        except Exception:
            dir_results.append(
                DirectoryCounts(
                    directory=str(d),
                    total_images=0,
                    with_yaml=0,
                    missing_yaml=0,
                    with_caption=0,
                    missing_caption=0,
                    images_modified_within_days=0,
                )
            )

    # Summarize FileTracker logs
    log_dir = Path("data") / "file_operations_logs"
    op_summary = load_filetracker_summaries(log_dir, args.days)

    # Print human-readable summary
    print_human_summary(dir_results, op_summary, args.days)

    # Optional JSON report
    if args.write_report:
        out_path = write_json_report(dir_results, op_summary)
        print(f"Report written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


