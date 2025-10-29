#!/usr/bin/env python3
"""
Quick Directory Count Checker
==============================
Counts PNGs in workflow directories for the active project.

Usage:
    python scripts/tools/check_directory_counts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = _ROOT / "data" / "projects"


def find_active_project():
    """Find the active project (no finishedAt)."""
    for manifest_file in PROJECTS_DIR.glob("*.project.json"):
        try:
            project = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not project.get("finishedAt"):
                return project, manifest_file
        except Exception:
            continue
    return None, None


def resolve_project_root(manifest_path: Path, root_hint: str) -> Path:
    """Resolve the project root path from manifest."""
    if not root_hint:
        return None

    # Try resolving relative to manifest directory
    try:
        root = (manifest_path.parent / root_hint).resolve()
        if root.exists():
            return root
    except Exception:
        pass

    # Try as absolute path
    try:
        root = Path(root_hint).resolve()
        if root.exists():
            return root
    except Exception:
        pass

    return None


def count_pngs_in_dir(directory: Path) -> int:
    """Count PNG files in a directory."""
    if not directory.exists():
        return 0
    try:
        return len(list(directory.glob("*.png")))
    except Exception:
        return 0


def main():
    project, manifest_path = find_active_project()

    if not project:
        print("❌ No active project found")
        sys.exit(1)

    project_id = project.get("projectId", "unknown")
    title = project.get("title", project_id)

    print(f"📊 Project: {title} ({project_id})")
    print("=" * 60)

    # Resolve root path
    root_hint = (project.get("paths") or {}).get("root", "")
    root = resolve_project_root(manifest_path, root_hint)

    if not root:
        print(f"❌ Could not resolve project root: {root_hint}")
        sys.exit(1)

    print(f"📁 Root: {root}")
    print()

    # Check common workflow directories
    dirs_to_check = [
        ("__selected", "__selected/"),
        ("__crop", "__crop/"),
        ("__crop_auto", "__crop_auto/"),
        ("__cropped", "__cropped/"),
        ("content", "content/"),
    ]

    total_remaining = 0
    total_done = 0

    for label, subdir in dirs_to_check:
        dir_path = root / subdir.rstrip("/")
        count = count_pngs_in_dir(dir_path)

        status = "✓" if dir_path.exists() else "✗"
        print(f"{status} {label:20s} {count:6,d} PNGs")

        # Track remaining work
        if label in ["__crop", "__crop_auto"]:
            total_remaining += count
        elif label == "__cropped":
            total_done = count

    print()
    print("=" * 60)
    print("📈 Progress Summary:")
    print(f"   Remaining to crop:  {total_remaining:,} images")
    print(f"   Already cropped:    {total_done:,} images ✓")

    if total_remaining > 0 and total_done > 0:
        total = total_remaining + total_done
        pct = (total_done / total) * 100
        print(f"   Progress:           {pct:.1f}% complete ({total_done:,} / {total:,})")

    # Estimate batches (assuming 3 images per batch)
    if total_remaining > 0:
        batches = (total_remaining + 2) // 3  # Round up
        print(f"   Estimated batches:  ~{batches:,} batches (at 3 img/batch)")


if __name__ == "__main__":
    main()
