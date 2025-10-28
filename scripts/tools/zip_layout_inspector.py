#!/usr/bin/env python3
"""
ZIP Layout Inspector (Read-Only)
================================
Scan one or more zip files (or directories of zips) without extraction and
summarize internal folder structure and filename patterns.

Outputs (JSON): data/daily_summaries/zip_layouts.json

Reported per-zip:
- total_entries, total_files, png_files
- top_components: counts of first path segment
- depth_histogram (pngs)
- per_top_component_png: png count per first segment
- sample_paths: up to 10 example PNG relative paths
- pattern_stats: {has_timestamp%, has_stage%}
- notes: simple hints (e.g., likely selected/final/original)

Usage:
  python scripts/tools/zip_layout_inspector.py /path/to/dir-or-zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Ensure project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.companion_file_utils import (
    detect_stage,
    extract_timestamp_from_filename,
    get_stage_number,
)

OUT_JSON = _ROOT / "data" / "daily_summaries" / "zip_layouts.json"


def _gather(paths: List[str]) -> List[Path]:
    found: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".zip":
            found.append(path)
        elif path.is_dir():
            for zp in path.rglob("*.zip"):
                found.append(zp)
    return found


def _depth(rel_path: str) -> int:
    # Count components excluding empty segments
    return len([c for c in rel_path.split("/") if c])


def _top_component(rel_path: str) -> str:
    parts = [c for c in rel_path.split("/") if c]
    return parts[0] if parts else ""


def _infer_notes(top_counts: Dict[str, int]) -> List[str]:
    notes: List[str] = []
    keys = {k.lower() for k in top_counts.keys()}
    if any("selected" in k for k in keys):
        notes.append("contains selected/")
    if any("final" in k for k in keys):
        notes.append("contains final/")
    if any("crop" in k for k in keys):
        notes.append("contains crop/")
    if not notes:
        notes.append("no obvious selected/final/crop top-level")
    return notes


def inspect_zip(zip_path: Path) -> Dict:
    total_entries = 0
    total_files = 0
    png_files = 0
    depth_hist: Dict[int, int] = Counter()
    top_components: Dict[str, int] = Counter()
    per_top_png: Dict[str, int] = Counter()
    sample_paths: List[str] = []

    ts_hits = 0
    stage_hits = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            total_entries = len(infos)
            for info in infos:
                rel = info.filename
                if rel.endswith("/"):
                    # directory entry
                    continue
                total_files += 1
                top = _top_component(rel)
                top_components[top] += 1
                if rel.lower().endswith(".png"):
                    png_files += 1
                    depth_hist[_depth(rel)] += 1
                    per_top_png[top] += 1
                    if len(sample_paths) < 10:
                        sample_paths.append(rel)
                    base = rel.split("/")[-1]
                    if extract_timestamp_from_filename(base):
                        ts_hits += 1
                    if get_stage_number(detect_stage(base)) > 0.0:
                        stage_hits += 1
    except Exception as e:
        return {
            "zip": str(zip_path),
            "error": str(e),
        }

    pattern_stats = {
        "has_timestamp_pct": round((ts_hits / max(1, png_files)) * 100.0, 2),
        "has_stage_pct": round((stage_hits / max(1, png_files)) * 100.0, 2),
    }

    return {
        "zip": str(zip_path),
        "zip_basename": zip_path.name,
        "total_entries": total_entries,
        "total_files": total_files,
        "png_files": png_files,
        "top_components": dict(top_components),
        "depth_histogram": dict(depth_hist),
        "per_top_component_png": dict(per_top_png),
        "pattern_stats": pattern_stats,
        "sample_paths": sample_paths,
        "notes": _infer_notes(top_components),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect zip internal layouts")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more zip files and/or directories to scan recursively",
    )
    args = parser.parse_args()

    zips = _gather(args.paths)
    results: List[Dict] = []
    totals = {
        "zip_count": 0,
        "png_files": 0,
        "top_components": Counter(),
    }

    for zp in zips:
        res = inspect_zip(zp)
        results.append(res)
        totals["zip_count"] += 1
        if not res.get("error"):
            totals["png_files"] += int(res.get("png_files") or 0)
            # merge top-components
            for k, v in (res.get("top_components") or {}).items():
                totals["top_components"][k] += v

    # convert counters
    totals["top_components"] = dict(totals["top_components"])  # type: ignore

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"results": results, "totals": totals}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
