#!/usr/bin/env python3
"""
Analyze crop training data patterns to understand what the AI learned.

This script will:
1. Load all crop training data (7,194 clean rows)
2. Calculate crop statistics (size, position, variance)
3. Generate visualizations showing crop diversity
4. Identify any patterns or biases in the data
"""

import csv
import statistics
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_crop_data():
    """Analyze crop patterns in training data."""
    
    training_dir = Path(__file__).parent.parent.parent / "data" / "training"
    csv_path = training_dir / "select_crop_log.csv"
    
    if not csv_path.exists():
        print(f"[!] Training data not found: {csv_path}")
        return
    
    print("=" * 80)
    print("CROP TRAINING DATA ANALYSIS")
    print("=" * 80)
    print(f"Source: {csv_path.name}")
    print()
    
    # Read all rows
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total rows: {len(rows)}")
    
    # Filter rows with valid crop coordinates and dimensions
    valid_crops = []
    for row in rows:
        try:
            x1 = float(row.get('crop_x1', 0))
            y1 = float(row.get('crop_y1', 0))
            x2 = float(row.get('crop_x2', 0))
            y2 = float(row.get('crop_y2', 0))
            width = int(row.get('width_0', 0))
            height = int(row.get('height_0', 0))
            
            # Skip invalid data
            if width <= 0 or height <= 0:
                continue
            if x1 >= x2 or y1 >= y2:
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            
            valid_crops.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': width, 'height': height,
                'filename': row.get('chosen_path', ''),
                'project': row.get('directory', '')
            })
        except (ValueError, TypeError):
            continue
    
    print(f"Valid crops: {len(valid_crops)}")
    print()
    
    if len(valid_crops) == 0:
        print("[!] No valid crop data found!")
        return
    
    # Calculate normalized crop statistics
    crop_widths = []  # As percentage of original
    crop_heights = []
    crop_areas = []
    crop_x_centers = []  # Normalized [0, 1]
    crop_y_centers = []
    
    x1_positions = []
    y1_positions = []
    
    for crop in valid_crops:
        # Calculate crop dimensions as percentage
        crop_w = (crop['x2'] - crop['x1']) / crop['width']
        crop_h = (crop['y2'] - crop['y1']) / crop['height']
        crop_area = crop_w * crop_h
        
        # Calculate crop center position (normalized)
        crop_x_center = (crop['x1'] + crop['x2']) / 2 / crop['width']
        crop_y_center = (crop['y1'] + crop['y2']) / 2 / crop['height']
        
        # Top-left corner position (normalized)
        x1_norm = crop['x1'] / crop['width']
        y1_norm = crop['y1'] / crop['height']
        
        crop_widths.append(crop_w * 100)
        crop_heights.append(crop_h * 100)
        crop_areas.append(crop_area * 100)
        crop_x_centers.append(crop_x_center)
        crop_y_centers.append(crop_y_center)
        x1_positions.append(x1_norm)
        y1_positions.append(y1_norm)
    
    print("📊 CROP SIZE DISTRIBUTION")
    print("-" * 80)
    print("Crop Width (% of original):")
    print(f"  Min:    {min(crop_widths):.1f}%")
    print(f"  Max:    {max(crop_widths):.1f}%")
    print(f"  Mean:   {statistics.mean(crop_widths):.1f}%")
    print(f"  Median: {statistics.median(crop_widths):.1f}%")
    print(f"  StdDev: {statistics.stdev(crop_widths):.1f}%")
    print()
    print("Crop Height (% of original):")
    print(f"  Min:    {min(crop_heights):.1f}%")
    print(f"  Max:    {max(crop_heights):.1f}%")
    print(f"  Mean:   {statistics.mean(crop_heights):.1f}%")
    print(f"  Median: {statistics.median(crop_heights):.1f}%")
    print(f"  StdDev: {statistics.stdev(crop_heights):.1f}%")
    print()
    print("Crop Area (% of original):")
    print(f"  Min:    {min(crop_areas):.1f}%")
    print(f"  Max:    {max(crop_areas):.1f}%")
    print(f"  Mean:   {statistics.mean(crop_areas):.1f}%")
    print(f"  Median: {statistics.median(crop_areas):.1f}%")
    print(f"  StdDev: {statistics.stdev(crop_areas):.1f}%")
    print()
    
    print("📍 CROP POSITION DISTRIBUTION")
    print("-" * 80)
    print("Top-Left X Position (normalized [0,1]):")
    print(f"  Min:    {min(x1_positions):.3f}")
    print(f"  Max:    {max(x1_positions):.3f}")
    print(f"  Mean:   {statistics.mean(x1_positions):.3f}")
    print(f"  Median: {statistics.median(x1_positions):.3f}")
    print(f"  StdDev: {statistics.stdev(x1_positions):.3f}")
    print()
    print("Top-Left Y Position (normalized [0,1]):")
    print(f"  Min:    {min(y1_positions):.3f}")
    print(f"  Max:    {max(y1_positions):.3f}")
    print(f"  Mean:   {statistics.mean(y1_positions):.3f}")
    print(f"  Median: {statistics.median(y1_positions):.3f}")
    print(f"  StdDev: {statistics.stdev(y1_positions):.3f}")
    print()
    print("Crop Center X (normalized [0,1]):")
    print(f"  Min:    {min(crop_x_centers):.3f}")
    print(f"  Max:    {max(crop_x_centers):.3f}")
    print(f"  Mean:   {statistics.mean(crop_x_centers):.3f}")
    print(f"  Median: {statistics.median(crop_x_centers):.3f}")
    print(f"  StdDev: {statistics.stdev(crop_x_centers):.3f}")
    print()
    print("Crop Center Y (normalized [0,1]):")
    print(f"  Min:    {min(crop_y_centers):.3f}")
    print(f"  Max:    {max(crop_y_centers):.3f}")
    print(f"  Mean:   {statistics.mean(crop_y_centers):.3f}")
    print(f"  Median: {statistics.median(crop_y_centers):.3f}")
    print(f"  StdDev: {statistics.stdev(crop_y_centers):.3f}")
    print()
    
    # Analyze patterns
    print("🔍 PATTERN ANALYSIS")
    print("-" * 80)
    
    # Check if crops are mostly uniform
    width_variance = statistics.stdev(crop_widths) / statistics.mean(crop_widths)
    height_variance = statistics.stdev(crop_heights) / statistics.mean(crop_heights)
    x_variance = statistics.stdev(x1_positions)
    y_variance = statistics.stdev(y1_positions)
    
    print(f"Width Coefficient of Variation: {width_variance:.3f}")
    print(f"Height Coefficient of Variation: {height_variance:.3f}")
    print(f"X Position Standard Deviation: {x_variance:.3f}")
    print(f"Y Position Standard Deviation: {y_variance:.3f}")
    print()
    
    # Interpretation
    if width_variance < 0.15 and height_variance < 0.15:
        print("⚠️  WARNING: Crop sizes are VERY uniform (low variance)")
        print("   → Model likely learned to output average crop size")
    elif width_variance < 0.25 and height_variance < 0.25:
        print("⚠️  CAUTION: Crop sizes have moderate variance")
        print("   → Model has some diversity but may still overfit")
    else:
        print("✓  GOOD: Crop sizes have high variance")
        print("   → Diverse training data")
    print()
    
    if x_variance < 0.1 and y_variance < 0.1:
        print("⚠️  WARNING: Crop positions are VERY uniform (low variance)")
        print("   → Model likely learned to output average position")
    elif x_variance < 0.15 and y_variance < 0.15:
        print("⚠️  CAUTION: Crop positions have moderate variance")
        print("   → Some diversity but may still overfit")
    else:
        print("✓  GOOD: Crop positions have high variance")
        print("   → Diverse training data")
    print()
    
    # Show some examples
    print("📋 SAMPLE CROPS (first 10)")
    print("-" * 80)
    for i, crop in enumerate(valid_crops[:10]):
        crop_w_pct = (crop['x2'] - crop['x1']) / crop['width'] * 100
        crop_h_pct = (crop['y2'] - crop['y1']) / crop['height'] * 100
        x_center = (crop['x1'] + crop['x2']) / 2 / crop['width']
        y_center = (crop['y1'] + crop['y2']) / 2 / crop['height']
        print(f"{i+1:2d}. Size: {crop_w_pct:4.1f}%w × {crop_h_pct:4.1f}%h | Center: ({x_center:.2f}, {y_center:.2f})")
    print()
    
    # Show extreme examples
    print("🎯 EXTREME EXAMPLES")
    print("-" * 80)
    
    # Smallest crop
    min_area_idx = crop_areas.index(min(crop_areas))
    min_crop = valid_crops[min_area_idx]
    print(f"Smallest Crop ({min(crop_areas):.1f}% area):")
    print(f"  Size: {(min_crop['x2']-min_crop['x1'])/min_crop['width']*100:.1f}%w × {(min_crop['y2']-min_crop['y1'])/min_crop['height']*100:.1f}%h")
    print()
    
    # Largest crop
    max_area_idx = crop_areas.index(max(crop_areas))
    max_crop = valid_crops[max_area_idx]
    print(f"Largest Crop ({max(crop_areas):.1f}% area):")
    print(f"  Size: {(max_crop['x2']-max_crop['x1'])/max_crop['width']*100:.1f}%w × {(max_crop['y2']-max_crop['y1'])/max_crop['height']*100:.1f}%h")
    print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    # Overall assessment
    overall_diversity = (width_variance + height_variance + x_variance + y_variance) / 4
    
    if overall_diversity < 0.15:
        print("❌ LOW DIVERSITY: Training data has very uniform crop patterns")
        print("   → Model is likely overfitting to average crop")
        print("   → Needs more diverse crop examples")
    elif overall_diversity < 0.25:
        print("⚠️  MODERATE DIVERSITY: Training data has some variety")
        print("   → Model may struggle with edge cases")
        print("   → Could benefit from more diverse examples")
    else:
        print("✅ HIGH DIVERSITY: Training data has diverse crop patterns")
        print("   → Model should learn content-aware cropping")
        print("   → Current poor performance may be architecture issue")
    print()


if __name__ == "__main__":
    analyze_crop_data()

