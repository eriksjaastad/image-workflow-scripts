#!/usr/bin/env python3
"""
Inventory Allowed Extensions (Per Project)
-----------------------------------------
Scans a content directory and writes the per-project allowed extensions snapshot
to data/projects/<project_id>_allowed_ext.json.

Usage:
  python scripts/tools/inventory_allowed_ext.py --project-id mojo1 --content-dir /abs/path/to/mojo1

Notes:
  - Lowercases extensions and strips leading dots
  - Skips directories and dotfiles
  - Safe: read-only scan; writes a small JSON file under data/projects/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


def is_hidden(p: Path) -> bool:
    return p.name.startswith('.')


def _dir_has_images(dir_path: Path, image_exts: Set[str]) -> bool:
    for dp, dn, fn in os.walk(dir_path):
        dn[:] = [d for d in dn if not d.startswith('.')]
        for name in fn:
            if name.startswith('.'):
                continue
            ext = Path(name).suffix.lower().lstrip('.')
            if ext in image_exts:
                return True
    return False


def collect_extensions(root: Path, image_exts: Set[str]) -> Dict[str, int]:
    """Collect extensions but only within:
    - the root directory, and
    - top-level subdirectories that contain at least one image (recursively).
    Hidden dirs/files are ignored.
    """
    ext_counts: Dict[str, int] = {}

    # Determine top-level subdirectories to scan (those that contain images)
    allowed_topdirs: Set[str] = set()
    skipped_topdirs: List[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith('.'):
            if _dir_has_images(entry, image_exts):
                allowed_topdirs.add(entry.name)
            else:
                skipped_topdirs.append(entry.name)

    # Walk
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # At root: only descend into allowed top-level dirs
        if p == root:
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d in allowed_topdirs]
        else:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        for name in filenames:
            if name.startswith('.'):  # hidden files
                continue
            ext = Path(name).suffix.lower().lstrip('.')
            if not ext:
                continue
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    if skipped_topdirs:
        print("[*] Skipped top-level directories with no images:")
        for d in skipped_topdirs:
            print(f"    {d}")

    return ext_counts


def write_allowlist(project_id: str, content_dir: Path, ext_counts: Dict[str, int]) -> Path:
    project_file = Path('data/projects') / f"{project_id}_allowed_ext.json"
    project_file.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "projectId": project_id,
        "snapshotAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "sourcePath": str(content_dir),
        "allowedExtensions": sorted(list(ext_counts.keys())),
        "clientWhitelistOverrides": [],
        "notes": "Initial inventory of content/ extensions"
    }

    with project_file.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    return project_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory allowed extensions from a content directory")
    parser.add_argument('--project-id', required=True, help='Project identifier (e.g., mojo1)')
    parser.add_argument('--content-dir', required=True, help='Path to the project content directory')
    parser.add_argument('--bans-json', default=str((Path('data/projects') / 'global_bans.json').resolve()), help='Path to global bans JSON')
    parser.add_argument('--fail-on-banned', action='store_true', help='Exit with error if banned extensions are present in content/')
    parser.add_argument('--image-exts', default='png,jpg,jpeg,webp,heic,heif,tif,tiff', help='Comma-separated list of image extensions used to detect image-bearing dirs')
    args = parser.parse_args()

    content_dir = Path(args.content_dir).resolve()
    if not content_dir.exists() or not content_dir.is_dir():
        raise SystemExit(f"[!] content-dir not found or not a directory: {content_dir}")

    image_exts = {e.strip().lower() for e in args.image_exts.split(',') if e.strip()}
    ext_counts = collect_extensions(content_dir, image_exts)
    # Optional banned presence check
    banned_present: Dict[str, int] = {}
    try:
        bans_path = Path(args.bans_json)
        if bans_path.exists():
            bans = json.loads(bans_path.read_text(encoding='utf-8'))
            banned_exts = {str(e).lower() for e in bans.get('bannedExtensions', [])}
            for ext, cnt in ext_counts.items():
                if ext in banned_exts:
                    banned_present[ext] = cnt
    except Exception:
        pass
    out_path = write_allowlist(args.project_id, content_dir, ext_counts)

    total = sum(ext_counts.values())
    print(f"[*] Inventoried {total} files across {len(ext_counts)} extensions from: {content_dir}")
    for ext, count in sorted(ext_counts.items()):
        print(f"    .{ext}: {count}")
    print(f"[*] Wrote allowlist: {out_path}")

    if banned_present:
        print("[!] BANNED EXTENSIONS PRESENT IN SOURCE (review):")
        for ext, cnt in sorted(banned_present.items()):
            print(f"    .{ext}: {cnt}")
        if args.fail_on_banned:
            raise SystemExit(2)


if __name__ == '__main__':
    main()


