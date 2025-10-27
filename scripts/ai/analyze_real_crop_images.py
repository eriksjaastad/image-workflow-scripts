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
    
    print("=" * 80)
    print("REAL CROP IMAGE ANALYSIS")
    print("=" * 80)
    print(f"Directory: {crop_dir}")
    print()
    
    # Find all PNG files
    png_files = list(crop_dir.glob("*.png"))[:limit]
    
    if not png_files:
        print(f"[!] No PNG files found in {crop_dir}")
        return
    
    print(f"Analyzing {len(png_files)} images...")
    print()
    
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
        except Exception as e:
            print(f"   Warning: Could not open {img_path.name}: {e}")
    
    if not widths:
        print("[!] No valid images found")
        return
    
    print("📐 IMAGE DIMENSIONS")
    print("-" * 80)
    print("Width:")
    print(f"  Min:    {min(widths)}px")
    print(f"  Max:    {max(widths)}px")
    print(f"  Mean:   {statistics.mean(widths):.0f}px")
    print(f"  Median: {statistics.median(widths):.0f}px")
    print(f"  StdDev: {statistics.stdev(widths) if len(widths) > 1 else 0:.0f}px")
    print()
    print("Height:")
    print(f"  Min:    {min(heights)}px")
    print(f"  Max:    {max(heights)}px")
    print(f"  Mean:   {statistics.mean(heights):.0f}px")
    print(f"  Median: {statistics.median(heights):.0f}px")
    print(f"  StdDev: {statistics.stdev(heights) if len(heights) > 1 else 0:.0f}px")
    print()
    print("Aspect Ratio (width/height):")
    print(f"  Min:    {min(aspect_ratios):.3f}")
    print(f"  Max:    {max(aspect_ratios):.3f}")
    print(f"  Mean:   {statistics.mean(aspect_ratios):.3f}")
    print(f"  Median: {statistics.median(aspect_ratios):.3f}")
    print(f"  StdDev: {statistics.stdev(aspect_ratios) if len(aspect_ratios) > 1 else 0:.3f}")
    print()
    
    # Show sample
    print("📋 SAMPLE IMAGES (first 20)")
    print("-" * 80)
    for i, (img_path, w, h, ar) in enumerate(zip(png_files[:20], widths[:20], heights[:20], aspect_ratios[:20])):
        print(f"{i+1:2d}. {img_path.name[:50]:50s} {w:4d}×{h:4d}  (AR: {ar:.3f})")
    print()
    
    # Analyze variance
    width_cv = statistics.stdev(widths) / statistics.mean(widths) if len(widths) > 1 else 0
    height_cv = statistics.stdev(heights) / statistics.mean(heights) if len(heights) > 1 else 0
    ar_std = statistics.stdev(aspect_ratios) if len(aspect_ratios) > 1 else 0
    
    print("🔍 DIVERSITY ANALYSIS")
    print("-" * 80)
    print(f"Width Coefficient of Variation: {width_cv:.3f}")
    print(f"Height Coefficient of Variation: {height_cv:.3f}")
    print(f"Aspect Ratio Std Dev: {ar_std:.3f}")
    print()
    
    if width_cv > 0.2 or height_cv > 0.2 or ar_std > 0.15:
        print("✅ HIGH DIVERSITY: Crop sizes vary significantly")
        print("   → Your actual crops are diverse!")
    elif width_cv > 0.1 or height_cv > 0.1 or ar_std > 0.08:
        print("⚠️  MODERATE DIVERSITY: Some variety in crop sizes")
    else:
        print("❌ LOW DIVERSITY: Crops are very uniform")
    print()
    
    # If originals are 2048x2048 or 3072x3072, estimate crop percentages
    print("📊 ESTIMATED CROP PERCENTAGES (if originals were 2048×2048)")
    print("-" * 80)
    for i, (w, h) in enumerate(zip(widths[:10], heights[:10])):
        crop_pct = (w / 2048) * 100
        print(f"{i+1:2d}. {w}×{h} = {crop_pct:.1f}% of 2048 original")
    print()
    
    print("=" * 80)


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
        print(f"[!] Directory not found: {crop_dir}")
        sys.exit(1)
    
    analyze_real_crops(crop_dir, args.limit)

