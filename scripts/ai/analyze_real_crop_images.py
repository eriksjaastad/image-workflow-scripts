#!/usr/bin/env python3
"""
Analyze ACTUAL crop images from crop directories to see real crop patterns.

This bypasses the CSV and looks at actual before/after dimensions.
"""

import statistics
import sys
from pathlib import Path

from PIL import Image


def analyze_real_crops(crop_dir: Path, limit: int = 100):
    """
    Analyze actual crop images to see real patterns.
    Assumes crop directory has images that were cropped from originals.
    """
    # Find all PNG files
    png_files = list(crop_dir.glob("*.png"))[:limit]
    
    if not png_files:
        return
    
    
    # Analyze dimensions
    widths = []
    heights = []
    aspect_ratios = []
    
    for img_path in png_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 1.0)
        except Exception:
            pass
    
    if not widths:
        return
    
    
    # Show sample
    for _i, (img_path, w, h, _ar) in enumerate(zip(png_files[:20], widths[:20], heights[:20], aspect_ratios[:20], strict=False)):
        pass
    
    # Analyze variance
    width_cv = statistics.stdev(widths) / statistics.mean(widths) if len(widths) > 1 else 0
    height_cv = statistics.stdev(heights) / statistics.mean(heights) if len(heights) > 1 else 0
    ar_std = statistics.stdev(aspect_ratios) if len(aspect_ratios) > 1 else 0
    
    
    if width_cv > 0.2 or height_cv > 0.2 or ar_std > 0.15 or width_cv > 0.1 or height_cv > 0.1 or ar_std > 0.08:
        pass
    else:
        pass
    
    # If originals are 2048x2048 or 3072x3072, estimate crop percentages
    for _i, (w, h) in enumerate(zip(widths[:10], heights[:10], strict=False)):
        (w / 2048) * 100
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze actual crop images")
    parser.add_argument("directory", nargs="?", help="Crop directory to analyze")
    parser.add_argument("--limit", type=int, default=100, help="Max images to analyze (default: 100)")
    args = parser.parse_args()
    
    if args.directory:
        crop_dir = Path(args.directory).expanduser().resolve()
    else:
        # Default to project crop directory (double-underscore)
        crop_dir = Path(__file__).parent.parent.parent / "__cropped"
    
    if not crop_dir.exists():
        sys.exit(1)
    
    analyze_real_crops(crop_dir, args.limit)

