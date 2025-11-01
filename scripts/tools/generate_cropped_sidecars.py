#!/usr/bin/env python3
"""
Generate .cropped Sidecars for PNGs in crop/
--------------------------------------------
Creates empty .cropped files beside each PNG in a directory (recursive),
to signal cropped status in UIs. Safe: idempotent; overwrites with empty file.

Usage:
  python scripts/tools/generate_cropped_sidecars.py --dir /abs/path/to/crop
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def create_sidecars(root: Path) -> int:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            if p.suffix.lower() == ".png":
                sidecar = p.with_suffix(".cropped")
                try:
                    sidecar.write_text("")
                    count += 1
                except OSError:
                    pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate .cropped sidecars for PNGs")
    parser.add_argument("--dir", required=True, help="Directory to scan (e.g., crop/)")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[!] Not a directory: {root}")

    n = create_sidecars(root)
    print(f"[*] Created/updated {n} .cropped sidecars under {root}")


if __name__ == "__main__":
    main()
