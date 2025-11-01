#!/usr/bin/env python3
"""
Daily Backup Utility
====================
Safely archives logs, snapshots, and AI databases to a user-specified backup root.

Default destination (can be overridden with --dest):
  ~/project-data-archives/image-workflow

What gets backed up (read-only sources):
  - data/file_operations_logs/* (both rolling and daily)
  - data/log_archives/*.gz (if present)
  - data/snapshot/** (operation_events_v1, daily_aggregates_v1, derived_sessions_v1)
  - data/ai_data/backups/** and any *.db under data/**

Format:
  <dest>/YYYY-MM-DD/...

Safety:
  - Never deletes sources
  - Creates a manifest.json with counts and sizes
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def copy_tree(src: Path, dst: Path, glob_pattern: str | None = None) -> dict[str, Any]:
    report = {"src": str(src), "dst": str(dst), "files": 0, "bytes": 0}
    if not src.exists():
        return report
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst / src.name)
        report["files"] = 1
        report["bytes"] = (dst / src.name).stat().st_size
        return report
    if glob_pattern:
        files = list(src.rglob(glob_pattern))
    else:
        files = list(src.rglob("*"))
    for f in files:
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(f, out)
            report["files"] += 1
            try:
                report["bytes"] += out.stat().st_size
            except Exception:
                pass
        except Exception:
            # Skip unreadable files quietly
            continue
    return report


def find_project_dbs(root: Path) -> list[Path]:
    dbs: list[Path] = []
    for ext in ("*.db", "*.sqlite", "*.sqlite3"):
        dbs.extend(root.rglob(ext))
    return dbs


def main():
    parser = argparse.ArgumentParser(description="Daily backup of workflow data")
    parser.add_argument(
        "--dest", default=str(Path.home() / "project-data-archives" / "image-workflow")
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    dest_root = Path(args.dest).expanduser()
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = dest_root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "dest": str(out_dir),
        "items": [],
    }

    # 1) file_operations_logs
    manifest["items"].append(
        {
            "category": "file_operations_logs",
            **copy_tree(
                data_root / "file_operations_logs", out_dir / "file_operations_logs"
            ),
        }
    )

    # 2) log_archives (if present)
    manifest["items"].append(
        {
            "category": "log_archives",
            **copy_tree(data_root / "log_archives", out_dir / "log_archives"),
        }
    )

    # 3) snapshots
    manifest["items"].append(
        {
            "category": "snapshot_operation_events_v1",
            **copy_tree(
                data_root / "snapshot" / "operation_events_v1",
                out_dir / "snapshot" / "operation_events_v1",
            ),
        }
    )
    manifest["items"].append(
        {
            "category": "snapshot_daily_aggregates_v1",
            **copy_tree(
                data_root / "snapshot" / "daily_aggregates_v1",
                out_dir / "snapshot" / "daily_aggregates_v1",
            ),
        }
    )
    manifest["items"].append(
        {
            "category": "snapshot_derived_sessions_v1",
            **copy_tree(
                data_root / "snapshot" / "derived_sessions_v1",
                out_dir / "snapshot" / "derived_sessions_v1",
            ),
        }
    )

    # 4) ai_data backups and DBs
    manifest["items"].append(
        {
            "category": "ai_data_backups",
            **copy_tree(
                data_root / "ai_data" / "backups", out_dir / "ai_data" / "backups"
            ),
        }
    )

    # Include any live *.db under data/** for safety
    dbs = find_project_dbs(data_root)
    db_out = out_dir / "ai_data" / "db_snapshot"
    copied = 0
    db_out.mkdir(parents=True, exist_ok=True)
    for db in dbs:
        try:
            rel = db.relative_to(data_root)
        except Exception:
            rel = db.name
        target = db_out / str(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(db, target)
            copied += 1
        except Exception:
            continue
    manifest["items"].append({"category": "ai_data_db_snapshot", "files": copied})

    # Write manifest
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Force output to stdout and stderr
    import sys

    print(f"✅ Backup complete → {out_dir}", file=sys.stdout, flush=True)
    print(
        json.dumps({"summary": manifest["items"]}, indent=2),
        file=sys.stdout,
        flush=True,
    )

    # Also write to a log file for debugging
    log_file = dest_root / "backup_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Backup completed to {out_dir}\n")
        f.write(json.dumps({"summary": manifest["items"]}, indent=2) + "\n")
        f.write("-" * 50 + "\n")


if __name__ == "__main__":
    main()
