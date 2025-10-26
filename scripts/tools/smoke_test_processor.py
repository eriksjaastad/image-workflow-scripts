#!/usr/bin/env python3
"""
Smoke Test - Enqueue 1 image and process queue; verify cropped output exists.

Usage:
  python scripts/tools/smoke_test_processor.py --dir mojo3
"""

import argparse
from pathlib import Path

from PIL import Image
from scripts.utils.crop_queue import CropQueueManager
from scripts.process_crop_queue import CropQueueProcessor, HumanTimingSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the queue processor with one image")
    parser.add_argument("--dir", dest="directory", required=True, help="Source directory with PNGs (e.g., mojo3)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_dir = (repo_root / args.directory).resolve()
    dest_dir = (repo_root / "__cropped").resolve()
    dest_dir.mkdir(exist_ok=True)

    img = next((p for p in source_dir.glob("*.png")), None)
    assert img, f"No PNGs found in {source_dir}"

    with Image.open(img) as im:
        w, h = im.size

    cq = CropQueueManager()
    batch_id = cq.enqueue_batch([
        {
            "source_path": str(img.resolve()),
            "crop_rect": [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)],
            "crop_rect_normalized": [0.1, 0.1, 0.9, 0.9],
            "dest_directory": str(dest_dir),
            "image_width": w,
            "image_height": h,
            "index_in_batch": 0,
        }
    ], session_id="smoke_test", project_id="debug")

    # Process immediately (fast)
    processor = CropQueueProcessor(queue_manager=cq, timing_simulator=None, preview_mode=False)
    processed = processor.run(limit=1, skip_confirmation=True)

    # Verify output exists
    out_path = dest_dir / img.name
    assert out_path.exists(), f"Cropped output not found: {out_path}"
    print(f"✓ Smoke test passed: {out_path}")


if __name__ == "__main__":
    main()
