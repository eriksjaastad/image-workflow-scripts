#!/usr/bin/env python3
"""
Generate Dataset Overview Markdown
==================================
Produce Documents/data/DATASET_OVERVIEW.md summarizing available metrics using
manifests, group stems, stager fields, training DB presence, and timesheet totals.

Safe: read-only from data sources; writes a NEW markdown file in Documents/data/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

# Ensure project root
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.companion_file_utils import extract_timestamp_from_filename

PROJECTS_DIR = _ROOT / "data" / "projects"
DOC_OUT = _ROOT / "Documents" / "data" / "DATASET_OVERVIEW.md"
TIMESHEET_CSV = _ROOT / "data" / "timesheet.csv"
TRAINING_DIR = _ROOT / "data" / "training" / "ai_training_decisions"


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _iter_project_manifests() -> Iterable[dict]:
    if not PROJECTS_DIR.exists():
        return []
    for mf in sorted(PROJECTS_DIR.glob("*.project.json")):
        pj = _read_json(mf)
        if not pj:
            continue
        pj.setdefault("paths", {})
        pj.setdefault("counts", {})
        yield pj, mf


def _collect_group_stems(group_dirs: list[Path]) -> int:
    stems = set()
    for d in group_dirs:
        try:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.rglob("*.png"):
                ts = extract_timestamp_from_filename(p.name)
                if ts:
                    stems.add(ts)
        except Exception:
            continue
    return len(stems)


def _groups_by_size_from_training(db_dir: Path) -> dict[int, int]:
    """Approximate groups by size using training DB presence if any (requires DB queries in future).
    For now, return empty counts; placeholder for future enrichment.
    """
    return {}


def _parse_timesheet_hours_total() -> float:
    if not TIMESHEET_CSV.exists():
        return 0.0
    total = 0.0
    try:
        with open(TIMESHEET_CSV) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 6:
                    continue
                hrs = (row[5] or "").strip()
                try:
                    total += float(hrs)
                except ValueError:
                    continue
    except Exception:
        return total
    return total


def build_overview(now_str: str) -> str:
    total_projects = 0
    active_projects = 0
    completed_projects = 0

    total_source_images = 0
    total_groups = 0

    # Selection/crop metrics from manifests' stager if available
    images_selected = 0
    images_cropped = 0

    projects_with_training_db = 0
    known_gaps: list[str] = []

    for pj, manifest_path in _iter_project_manifests():
        total_projects += 1
        status = (pj.get("status") or "").lower()
        if status == "active":
            active_projects += 1
        elif status in ("finished", "completed"):
            completed_projects += 1

        counts = pj.get("counts") or {}
        total_source_images += int(counts.get("initialImages") or 0)

        # groupCount if present; otherwise compute from characterGroups quickly
        group_count = counts.get("groupCount")
        if isinstance(group_count, int):
            total_groups += group_count
        else:
            groups = (pj.get("paths") or {}).get("characterGroups") or []
            gdirs: list[Path] = []
            for rel in groups:
                try:
                    gdirs.append((manifest_path.parent / rel).resolve())
                except Exception:
                    pass
            total_groups += _collect_group_stems(gdirs)

        stager = (pj.get("metrics") or {}).get("stager") or {}
        by_ext = (stager.get("byExtIncluded") or {}) if isinstance(stager, dict) else {}
        # Selected ~= png files staged
        images_selected += int(by_ext.get("png") or 0)
        # Cropped ~= yaml count if present (historically yaml accompanies crops)
        images_cropped += int(by_ext.get("yaml") or 0)

        # Training DB presence
        pid = str(pj.get("projectId") or manifest_path.stem).strip()
        db_path = TRAINING_DIR / f"{pid}.db"
        if db_path.exists():
            projects_with_training_db += 1

    selection_rate = (
        (images_selected / total_source_images * 100.0) if total_source_images else 0.0
    )
    crop_rate = (
        (images_cropped / max(images_selected, 1) * 100.0) if images_selected else 0.0
    )

    total_hours = _parse_timesheet_hours_total()

    # Known issues list (static for now; can be detected later)
    known_gaps.extend(
        [
            "AI-Assisted Reviewer batch sessions undercount real hours",
            "Historical CSV corruption in select/crop logs (pre-v3)",
            "Active projects may not have stable group counts",
        ]
    )

    lines: list[str] = []
    lines.append(f"## Dataset Overview (as of {now_str})")
    lines.append("")
    lines.append("### Projects Summary")
    lines.append(f"- Total projects: {total_projects}")
    lines.append(f"- Active projects: {active_projects}")
    lines.append(f"- Completed projects: {completed_projects}")
    lines.append("")
    lines.append("### Image Volume")
    lines.append(f"- Total source images: {total_source_images}")
    lines.append(f"- Total image groups: {total_groups} (key metric)")
    lines.append("- Groups by size:")
    lines.append("  - Pairs (2 images): N/A")
    lines.append("  - Triplets (3 images): N/A")
    lines.append("  - Quads (4 images): N/A")
    lines.append("")
    lines.append("### Selection Stats")
    lines.append(f"- Images selected (best-of-N): {images_selected}")
    lines.append(
        f"- Selection rate: {round(selection_rate, 2)}% (selected / total source)"
    )
    lines.append(f"- Images cropped: {images_cropped}")
    lines.append(f"- Crop rate: {round(crop_rate, 2)}% (cropped / selected)")
    lines.append("")
    lines.append("### Training Data")
    lines.append("- Total crop decisions logged: N/A")
    lines.append("- Total selection decisions logged: N/A")
    lines.append(f"- Projects with complete training data: {projects_with_training_db}")
    lines.append("- Known data gaps/issues:")
    for g in known_gaps:
        lines.append(f"  - {g}")
    lines.append("")
    lines.append("### Work Time Reality Check")
    lines.append(
        f"- Estimated total hours invested (timesheet): {round(total_hours, 2)}"
    )
    lines.append("- Projects with accurate time tracking: N/A")
    lines.append("- Known tracking issues: [batch processing, sparse logging]")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset overview markdown")
    parser.parse_args()

    now_str = datetime.utcnow().strftime("%Y-%m-%d")
    md = build_overview(now_str)
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {DOC_OUT}")


if __name__ == "__main__":
    main()
