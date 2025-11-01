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
        return
    
    
    # Read all rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    
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
    
    
    if len(valid_crops) == 0:
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
    
    
    
    # Analyze patterns
    
    # Check if crops are mostly uniform
    width_variance = statistics.stdev(crop_widths) / statistics.mean(crop_widths)
    height_variance = statistics.stdev(crop_heights) / statistics.mean(crop_heights)
    x_variance = statistics.stdev(x1_positions)
    y_variance = statistics.stdev(y1_positions)
    
    
    # Interpretation
    if (width_variance < 0.15 and height_variance < 0.15) or (width_variance < 0.25 and height_variance < 0.25):
        pass
    else:
        pass
    
    if (x_variance < 0.1 and y_variance < 0.1) or (x_variance < 0.15 and y_variance < 0.15):
        pass
    else:
        pass
    
    # Show some examples
    for _i, crop in enumerate(valid_crops[:10]):
        (crop['x2'] - crop['x1']) / crop['width'] * 100
        (crop['y2'] - crop['y1']) / crop['height'] * 100
        (crop['x1'] + crop['x2']) / 2 / crop['width']
        (crop['y1'] + crop['y2']) / 2 / crop['height']
    
    # Show extreme examples
    
    # Smallest crop
    min_area_idx = crop_areas.index(min(crop_areas))
    valid_crops[min_area_idx]
    
    # Largest crop
    max_area_idx = crop_areas.index(max(crop_areas))
    valid_crops[max_area_idx]
    
    
    # Overall assessment
    overall_diversity = (width_variance + height_variance + x_variance + y_variance) / 4
    
    if overall_diversity < 0.15 or overall_diversity < 0.25:
        pass
    else:
        pass


if __name__ == "__main__":
    analyze_crop_data()

