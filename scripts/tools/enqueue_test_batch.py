#!/usr/bin/env python3
"""
Enqueue Test Batch - Queue N center-crops from a folder for processor testing.

Usage:
  python scripts/tools/enqueue_test_batch.py --dir mojo3 --limit 10 --dest __cropped
"""

import argparse
from pathlib import Path

from PIL import Image

from scripts.utils.crop_queue import CropQueueManager


def collect_images(source_dir: Path, limit: int) -> list[Path]:
    imgs = [p for p in source_dir.glob("*.png")]
    return imgs[:limit] if limit else imgs


essential_safe_destinations = {"__cropped", "__final", "__temp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue N center-crops from a folder")
    parser.add_argument(
        "--dir",
        dest="directory",
        required=True,
        help="Source directory with PNGs (e.g., mojo3)",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Max number of images to enqueue"
    )
    parser.add_argument(
        "--dest",
        dest="dest_dir",
        default="__cropped",
        help="Destination directory for crops",
    )
    parser.add_argument(
        "--session", dest="session_id", default="enqueue_test", help="Session ID"
    )
    parser.add_argument(
        "--project", dest="project_id", default="debug", help="Project ID"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_dir = (repo_root / args.directory).resolve()
    dest_dir = (repo_root / args.dest_dir).resolve()

    assert source_dir.exists(), f"Source directory not found: {source_dir}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source_dir, args.limit)
    assert images, f"No PNGs found in {source_dir}"

    crops = []
    for idx, img in enumerate(images):
        with Image.open(img) as im:
            w, h = im.size
        # Center 80% crop
        x1, y1 = int(w * 0.1), int(h * 0.1)
        x2, y2 = int(w * 0.9), int(h * 0.9)
        crops.append(
            {
                "source_path": str(img.resolve()),
                "crop_rect": [x1, y1, x2, y2],
                "crop_rect_normalized": [0.1, 0.1, 0.9, 0.9],
                "dest_directory": str(dest_dir),
                "image_width": w,
                "image_height": h,
                "index_in_batch": idx,
            }
        )

    cq = CropQueueManager()
    batch_id = cq.enqueue_batch(
        crops, session_id=args.session_id, project_id=args.project_id
    )
    print(f"Queued batch: {batch_id} ({len(crops)} crops) → dest={dest_dir}")


if __name__ == "__main__":
    main()
