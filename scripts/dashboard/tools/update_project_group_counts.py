#!/usr/bin/env python3
"""
Update Project Group Counts (Safe, Dry-Run by Default)
=====================================================
Computes per-project image group counts (unique timestamp stems across character groups)
and writes them into manifests under counts.groupCount.

Safety:
- Read-only scan of character group directories
- Dry-run by default; use --commit to write changes
- Skips active projects unless --include-active is set
- Logs file operations via FileTracker
- Creates JSON backups in data/ai_data/backups/manifests/<projectId>-YYYYMMDD_HHMMSS.json

Usage:
  python scripts/dashboard/tools/update_project_group_counts.py                 # dry-run all finished projects
  python scripts/dashboard/tools/update_project_group_counts.py --projects mojo1 mojo2
  python scripts/dashboard/tools/update_project_group_counts.py --include-active
  python scripts/dashboard/tools/update_project_group_counts.py --commit        # actually write changes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Ensure project root import path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.file_tracker import FileTracker
from scripts.utils.companion_file_utils import extract_timestamp_from_filename

PROJECTS_DIR = _ROOT / "data" / "projects"
BACKUP_DIR = _ROOT / "data" / "ai_data" / "backups" / "manifests"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_project_paths(project_filter: Optional[Set[str]] = None) -> Iterable[Path]:
    if not PROJECTS_DIR.exists():
        return []
    for mf in sorted(PROJECTS_DIR.glob("*.project.json")):
        if project_filter:
            # safer: read manifest to get projectId
            pj = _read_json(mf) or {}
            if (pj.get("projectId") or "").strip() not in project_filter:
                continue
        yield mf


def _collect_group_stems(group_dirs: List[Path]) -> int:
    stems: Set[str] = set()
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


def _resolve_group_dirs(manifest_path: Path, pj: dict) -> List[Path]:
    groups = (pj.get("paths") or {}).get("characterGroups") or []
    dirs: List[Path] = []
    for rel in groups:
        try:
            dirs.append((manifest_path.parent / rel).resolve())
        except Exception:
            pass
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write counts.groupCount to project manifests"
    )
    parser.add_argument(
        "--projects", nargs="*", help="Optional list of projectIds to include"
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Include active projects (default skips them)",
    )
    parser.add_argument(
        "--commit", action="store_true", help="Actually write changes (default dry-run)"
    )
    args = parser.parse_args()

    project_filter = set(args.projects) if args.projects else None
    tracker = FileTracker("update_project_group_counts", sandbox=False)

    updated = 0
    skipped = 0
    for manifest_path in _iter_project_paths(project_filter):
        pj = _read_json(manifest_path) or {}
        pid = (
            pj.get("projectId") or manifest_path.stem.replace(".project", "")
        ).strip()
        status = (pj.get("status") or "").strip().lower()
        if status == "active" and not args.include_active:
            print(f"[skip] {pid}: active (use --include-active to update)")
            skipped += 1
            continue

        group_dirs = _resolve_group_dirs(manifest_path, pj)
        group_count = _collect_group_stems(group_dirs)

        counts = pj.get("counts") or {}
        current = counts.get("groupCount")
        if current == group_count:
            print(f"[same] {pid}: groupCount already {group_count}")
            continue

        print(f"[update] {pid}: {current} -> {group_count}")

        if args.commit:
            # Backup
            try:
                BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
                backup_path = BACKUP_DIR / f"{pid}-{ts}.json"
                with open(backup_path, "w") as bf:
                    json.dump(pj, bf, indent=2)
                tracker.log_operation(
                    "create",
                    source_dir=str(manifest_path.parent),
                    dest_dir=str(BACKUP_DIR),
                    file_count=1,
                    files=[backup_path.name],
                    notes="manifest backup",
                )
            except Exception as e:
                print(f"[warn] backup failed for {pid}: {e}")

            # Write updated manifest
            try:
                pj.setdefault("counts", {})
                pj["counts"]["groupCount"] = group_count
                with open(manifest_path, "w") as f:
                    json.dump(pj, f, indent=2)
                tracker.log_operation(
                    "create",
                    source_dir=str(manifest_path.parent),
                    dest_dir=str(manifest_path.parent),
                    file_count=1,
                    files=[manifest_path.name],
                    notes="updated manifest with groupCount",
                )
                updated += 1
            except Exception as e:
                print(f"[error] failed to write manifest {pid}: {e}")
        else:
            # dry-run
            continue

    print(f"Done. updated={updated}, skipped={skipped}, dry_run={not args.commit}")


if __name__ == "__main__":
    main()
