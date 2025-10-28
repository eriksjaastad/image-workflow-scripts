#!/usr/bin/env python3
"""
ZIP Group Scanner (Read-Only)
=============================
Scan one or more zip files without extraction to compute:
- unique group stems (YYYYMMDD_HHMMSS) from .png names
- groups-by-size (2/3/4 images per group stem)
- per-zip totals

Safety:
- Read-only; never modifies archives (per policy)
- Writes NEW outputs to data/daily_summaries/

Usage:
  python scripts/tools/zip_group_scanner.py /path/to/dir-or-zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List

# Ensure project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.companion_file_utils import (
    detect_stage,
    extract_datetime_from_filename,
    extract_timestamp_from_filename,
    find_consecutive_stage_groups,
    get_stage_number,
)

OUT_JSON = _ROOT / "data" / "daily_summaries" / "zip_group_scan.json"


class _ZipEntry:
    def __init__(self, rel_path: str):
        # Full relative path inside zip (may include subdirectories)
        self.rel_path = rel_path
        # Basename for timestamp/stage parsing
        self.name = rel_path.split("/")[-1]
        # Immediate parent directory label for fallback separation
        parts = rel_path.split("/")
        self.parent_dir = parts[-2] if len(parts) >= 2 else ""


def _sort_zip_png_names(rel_paths: List[str]) -> List[_ZipEntry]:
    def _key(rp: str):
        base = rp.split("/")[-1]
        ts = extract_timestamp_from_filename(base) or "99999999_999999"
        stage = detect_stage(base)
        return (ts, get_stage_number(stage), base)

    return [_ZipEntry(rp) for rp in sorted(rel_paths, key=_key)]


def scan_zip(zip_path: Path) -> Tuple[int, Dict[int, int], int, int, str]:
    """
    Return (total_groups, groups_count_by_size, total_images, singles_count).
    Groups are formed using the same logic as AI-Assisted Reviewer (nearest-up by stage).
    Singles are images not included in any formed group.
    """
    rel_png_paths: List[str] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                if not name.lower().endswith(".png"):
                    continue
                # Preserve subdirectory context
                rel_png_paths.append(name)
    except Exception as e:
        print(f"[warn] failed to read {zip_path}: {e}")
        return 0, {}, 0, 0, "error"
    total_images = len(rel_png_paths)

    if total_images == 0:
        return 0, {}, 0, 0, "empty"

    # Sort and group using the shared logic
    entries = _sort_zip_png_names(rel_png_paths)

    # If most files lack recognizable stage markers, fall back to timestamp-only grouping
    unknown_stages = sum(
        1
        for rp in rel_png_paths
        if get_stage_number(detect_stage(rp.split("/")[-1])) == 0.0
    )
    use_timestamp_fallback = (unknown_stages / max(1, len(rel_png_paths))) > 0.8

    def _stage_of(item: _ZipEntry) -> float:
        return float(get_stage_number(detect_stage(item.name)))

    def _dt_of(item: _ZipEntry):
        return extract_datetime_from_filename(item.name)

    groups_count_by_size: Dict[int, int] = Counter()
    singles_count = 0
    mode = "stage"

    if use_timestamp_fallback:
        mode = "timestamp"
        stems: Dict[tuple, int] = Counter()
        no_ts = 0
        for rp in rel_png_paths:
            base = rp.split("/")[-1]
            ts = extract_timestamp_from_filename(base)
            parent = rp.split("/")[-2] if "/" in rp else ""
            if ts:
                stems[(ts, parent)] += 1
            else:
                no_ts += 1
        for cnt in stems.values():
            if cnt >= 2:
                groups_count_by_size[cnt] += 1
            else:
                singles_count += 1
        singles_count += no_ts
    else:
        grouped_lists = find_consecutive_stage_groups(
            entries,
            stage_of=_stage_of,
            dt_of=_dt_of,
            min_group_size=2,
        )

        in_group = set()
        for grp in grouped_lists:
            size = len(grp)
            groups_count_by_size[size] += 1
            for it in grp:
                in_group.add(it.name)

        filtered_images = len(entries)
        singles_count = filtered_images - len(in_group)
        if singles_count > 0:
            groups_count_by_size[1] += singles_count

    total_groups = sum(groups_count_by_size.values())
    return total_groups, dict(groups_count_by_size), total_images, singles_count, mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan zip(s) for group counts (no extraction)"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more zip files and/or directories to scan recursively",
    )
    # CSV output removed; JSON is the single source of truth
    args = parser.parse_args()

    # Gather zips (files or from directories recursively) with container label
    def _gather(paths: List[str]):
        found = []
        for p in paths:
            path = Path(p)
            if path.is_file() and path.suffix.lower() == ".zip":
                found.append((path, path.parent.name or path.parent.as_posix()))
            elif path.is_dir():
                for zp in path.rglob("*.zip"):
                    found.append((zp, zp.parent.name or zp.parent.as_posix()))
        return found

    zip_entries = _gather(args.paths)

    results = []
    total_groups_all = 0
    agg_by_size: Dict[int, int] = Counter()
    totals_by_container: Dict[str, int] = Counter()
    total_images_all = 0
    singles_all = 0

    for zp_path, container in zip_entries:
        total, by_size, num_images, singles, mode = scan_zip(zp_path)
        results.append(
            {
                "zip": str(zp_path),
                "zip_basename": zp_path.name,
                "container": container,
                "total_groups": total,
                "groups_count_by_size": by_size,
                "total_images": num_images,
                "singles": singles,
                "grouping_mode": mode,
            }
        )
        total_groups_all += total
        for k, v in (by_size or {}).items():
            agg_by_size[k] += v
        totals_by_container[container] += total
        total_images_all += num_images
        singles_all += singles

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "results": results,
                "totals": {
                    "groups": total_groups_all,
                    "groups_by_size": dict(agg_by_size),
                    "by_container": dict(totals_by_container),
                    "total_images": total_images_all,
                    "singles": singles_all,
                },
            },
            f,
            indent=2,
        )
    print(f"Wrote {OUT_JSON}")

    # CSV output removed

    # Console summary
    print(f"Total groups across zips: {total_groups_all}")
    print(f"Groups by size: {dict(agg_by_size)}")
    if totals_by_container:
        print(f"By container: {dict(totals_by_container)}")
    print(f"Total images: {total_images_all}, singles: {singles_all}")


if __name__ == "__main__":
    main()
