#!/usr/bin/env python3
"""
Weekly Rollup Archiver
======================
Packs the last 7 days of daily backups under
  ~/project-data-archives/image-workflow/YYYY-MM-DD/
into a single tar.zst with a manifest, then uploads via rclone to gbackup:weekly-rollups/.

Local retention: keep last 12 weekly archives (older are removed locally only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Create weekly rollup and upload via rclone"
    )
    parser.add_argument(
        "--root", default=str(Path.home() / "project-data-archives" / "image-workflow")
    )
    parser.add_argument("--remote", default="gbackup:weekly-rollups")
    parser.add_argument("--keep", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"No backups found at {root}")
        return

    # Determine week range (today back 6 days)
    end = datetime.now().date()
    start = end - timedelta(days=6)
    days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    existing_days = [d for d in days if (root / d).exists()]
    if not existing_days:
        print("No daily folders to roll up")
        return

    rollup_dir = Path.home() / "project-data-archives" / "image-workflow" / "weekly"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    label = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    tar_path = rollup_dir / f"weekly_{label}.tar.zst"

    # Build manifest
    manifest: dict[str, Any] = {
        "range": {"start": days[0], "end": days[-1]},
        "included": [],
        "created_at": datetime.now().isoformat(),
    }
    total_files = 0
    total_bytes = 0
    for d in existing_days:
        folder = root / d
        count = 0
        size = 0
        for p in folder.rglob("*"):
            if p.is_file():
                count += 1
                try:
                    size += p.stat().st_size
                except Exception:
                    pass
        manifest["included"].append({"day": d, "files": count, "bytes": size})
        total_files += count
        total_bytes += size
    manifest["totals"] = {"files": total_files, "bytes": total_bytes}

    # Write manifest temp and create tar.zst
    tmp_manifest = rollup_dir / f"manifest_{label}.json"
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Use python tarfile with zstd if available; else fallback to system tar + zstd
    try:
        import tarfile

        tar = tarfile.open(tar_path, mode="w|")
        for d in existing_days:
            tar.add(str(root / d), arcname=d)
        tar.add(str(tmp_manifest), arcname="manifest.json")
        tar.close()
        # post compress with zstd
        run(["zstd", "-f", str(tar_path)])
        tar_path = Path(str(tar_path) + ".zst")
    except Exception:
        # system tar + zstd pipeline
        cmd = f"(cd '{root}' && tar -cf - {' '.join(existing_days)} && tar -rf - -C '{rollup_dir}' manifest_{label}.json) | zstd -f -o '{tar_path}'"
        subprocess.check_call(cmd, shell=True)

    try:
        tmp_manifest.unlink()
    except Exception:
        pass

    # Upload
    run(["rclone", "copy", str(tar_path), args.remote])

    # Local retention
    archives = sorted(rollup_dir.glob("weekly_*.tar.zst"))
    if len(archives) > args.keep:
        for old in archives[: -args.keep]:
            try:
                old.unlink()
            except Exception:
                pass

    print(f"✅ Weekly rollup created and uploaded: {tar_path}")


if __name__ == "__main__":
    main()
