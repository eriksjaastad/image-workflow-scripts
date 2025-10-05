#!/usr/bin/env python3
"""
Create a daily integrity manifest for training data files.

Outputs: data/training/manifests/YYYYMMDD.json
Each entry includes sha256, size, row_count (for CSV), and mtime.

Usage:
  python scripts/backup/make_training_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def compute_sha256(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader)
            return max(0, rows - 1)  # exclude header if present
    except Exception:
        return 0


def build_manifest() -> Dict[str, Any]:
    training_dir = Path("data") / "training"
    manifests_dir = training_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    files = []
    if training_dir.exists():
        for p in sorted(training_dir.glob("*.csv")):
            try:
                stat = p.stat()
                entry = {
                    "path": str(p.resolve()),
                    "name": p.name,
                    "size": stat.st_size,
                    "mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
                    "sha256": compute_sha256(p),
                }
                if p.suffix.lower() == ".csv":
                    entry["row_count"] = csv_row_count(p)
                files.append(entry)
            except Exception as exc:
                files.append({
                    "path": str(p.resolve()),
                    "name": p.name,
                    "error": str(exc),
                })

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "date": datetime.utcnow().strftime("%Y%m%d"),
        "files": files,
    }
    out_path = manifests_dir / f"{manifest['date']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[*] Manifest written: {out_path}")
    print(f"[*] Files listed: {len(files)}")

    return manifest


if __name__ == "__main__":
    build_manifest()


