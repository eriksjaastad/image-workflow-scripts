#!/usr/bin/env python3
"""
Deterministic subset builder (sandbox-only).

Copies a fraction of groups from a source sandbox tree to a destination sandbox tree,
preserving image+companions. Default is a dry-run that only prints counts.

Deterministic sampling:
- Sort files by timestamp+stage using centralized utilities
- Build groups via nearest-up grouping
- Hash the first file path in each group (SHA1), select by modulus

Safety:
- Non-destructive; copies only under sandbox
- Default dry-run; must pass --commit to copy files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import List

# Ensure project root on sys.path when invoked directly
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.companion_file_utils import (
    sort_image_files_by_timestamp_and_stage,
    find_consecutive_stage_groups,
    find_all_companion_files,
)


def deterministic_group_filter(groups: List[List[Path]], *, fraction: float, modulus: int = 100, offset: int = 0) -> List[List[Path]]:
    modulus = max(1, int(modulus))
    # Convert fraction to a threshold in [0, modulus)
    threshold = max(0, min(modulus, int(round(fraction * modulus))))
    selected: List[List[Path]] = []
    for g in groups:
        if not g:
            continue
        h = hashlib.sha1(str(g[0]).encode("utf-8")).hexdigest()
        bucket = (int(h[:8], 16) + int(offset)) % modulus
        if bucket < threshold:
            selected.append(g)
    return selected


def _score_group_fast(group: List[Path], thumb_size: int = 128) -> float:
    """Higher score means more likely to contain a risky top-stage (challenge).
    Heuristic: compare top stage vs median of group using simple Tenengrad and clip fraction.
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return 0.0
    def metrics(p: Path) -> tuple:
        try:
            with Image.open(p) as img:
                img.thumbnail((thumb_size, thumb_size))
                g = img.convert("L")
                a = np.asarray(g, dtype=np.float32)
                # Sobel
                kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
                ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
                from numpy.lib.stride_tricks import sliding_window_view
                def conv2(x, k):
                    w = sliding_window_view(x, k.shape)
                    return (w * k).sum(axis=(-1,-2))
                gx = conv2(a, kx); gy = conv2(a, ky)
                ten = (gx*gx + gy*gy).mean()
                v = a.reshape(-1)
                clip = ((v <= 1).sum() + (v >= 254).sum()) / float(v.size if v.size else 1)
                return float(ten), float(clip)
        except Exception:
            return 0.0, 0.0
    # compute group metrics
    mets = [metrics(p) for p in group]
    if not mets:
        return 0.0
    ten_vals = [m[0] for m in mets]
    clip_vals = [m[1] for m in mets]
    import numpy as _np
    med_ten = float(_np.median(ten_vals))
    top_ten = ten_vals[-1]
    top_clip = clip_vals[-1]
    # score: penalize top if blurrier than median and/or more clipped
    blur_penalty = max(0.0, (med_ten - top_ten) / (med_ten + 1e-6))
    clip_penalty = top_clip  # already fraction
    return blur_penalty + clip_penalty


def challenge_group_filter(groups: List[List[Path]], *, fraction: float, thumb_size: int = 128) -> List[List[Path]]:
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return groups
    scored = [(g, _score_group_fast(g, thumb_size=thumb_size)) for g in groups if g]
    scored.sort(key=lambda t: t[1], reverse=True)
    k = max(1, int(round(len(scored) * fraction)))
    return [g for g, s in scored[:k]]


def copy_with_companions(src_path: Path, dst_dir: Path, *, dry_run: bool) -> int:
    # Copy main file
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    tgt = dst_dir / src_path.name
    if dry_run:
        copied += 1
    else:
        if not tgt.exists():
            shutil.copy2(str(src_path), str(tgt))
        copied += 1
    # Copy companions
    for c in find_all_companion_files(src_path):
        ctgt = dst_dir / c.name
        if dry_run:
            copied += 1
        else:
            if not ctgt.exists():
                shutil.copy2(str(c), str(ctgt))
            copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser(description="Deterministic subset builder (sandbox-only)")
    ap.add_argument("--source", default="sandbox/mojo2", help="Source sandbox root")
    ap.add_argument("--dest", default="sandbox/mojo2_subset", help="Destination sandbox root (will be created)")
    ap.add_argument("--fraction", type=float, default=0.25, help="Fraction of groups to include (0..1)")
    ap.add_argument("--modulus", type=int, default=100, help="Hash buckets for sampling")
    ap.add_argument("--offset", type=int, default=0, help="Hash offset for deterministic seed-variation")
    ap.add_argument("--challenge", action="store_true", default=False, help="Use risk-scored challenge sampling instead of hash")
    ap.add_argument("--challenge-thumb", type=int, default=128, help="Thumbnail size for challenge scoring")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--commit", action="store_true", default=False)
    ap.add_argument("--max-groups", type=int, default=None, help="Optional cap for groups to process (after sampling)")
    args = ap.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    if not src.exists():
        print(f"❌ Source not found: {src}")
        sys.exit(2)

    exclude_dirs = {"metrics", "reports", "runs", "logs", "selected", "delete", "crop"}
    image_files = [p for p in src.rglob("*.png") if not any(part in exclude_dirs for part in p.parts)]
    image_files = sort_image_files_by_timestamp_and_stage(image_files)

    groups = find_consecutive_stage_groups(image_files)
    if args.challenge:
        sampled = challenge_group_filter(groups, fraction=float(args.fraction), thumb_size=int(args.challenge_thumb))
        mode_used = "challenge"
    else:
        sampled = deterministic_group_filter(groups, fraction=float(args.fraction), modulus=int(args.modulus), offset=int(args.offset))
        mode_used = "hash"
    if args.max_groups:
        sampled = sampled[: int(args.max_groups)]

    # Stats
    total_images = len(image_files)
    total_groups = len(groups)
    sampled_groups = len(sampled)
    sampled_images = sum(len(g) for g in sampled)

    print(
        {
            "source": str(src),
            "dest": str(dst),
            "mode": "dry-run" if args.dry_run else "commit",
            "fraction": args.fraction,
            "modulus": args.modulus,
            "offset": args.offset,
            "sampling": mode_used,
            "images": total_images,
            "groups": total_groups,
            "sampled_groups": sampled_groups,
            "sampled_images": sampled_images,
        }
    )

    # Copy
    if not args.commit:
        print("[DRY RUN] No files copied.")
        return

    copied_files = 0
    for gi, g in enumerate(sampled, 1):
        # Mirror relative subdir of group[0] under dst
        rel = g[0].parent.relative_to(src)
        out_dir = dst / rel
        for p in g:
            copied_files += copy_with_companions(p, out_dir, dry_run=False)
        if gi % 25 == 0:
            print(f"[copy] groups={gi}/{sampled_groups} copied_files~{copied_files}")

    print(f"✅ Copy complete. Groups={sampled_groups} files_copied≈{copied_files}")


if __name__ == "__main__":
    main()


