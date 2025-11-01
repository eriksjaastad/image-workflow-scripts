#!/usr/bin/env python3
"""
Backfill Missing Crop Dimensions

Problem: 7,189 crop records have coordinates but dimensions are (0,0)
Solution: Read the original image files to get actual width/height
Result: Recover ~7,000 training examples for Crop Proposer

Usage:
    python scripts/ai/backfill_crop_dimensions.py --dry-run
    python scripts/ai/backfill_crop_dimensions.py  # Actually update the CSV
"""

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path("PROJECT_ROOT")
CROP_LOG = PROJECT_ROOT / "data/training/select_crop_log.csv"
BACKUP_LOG = PROJECT_ROOT / "data/training/select_crop_log_backup_before_backfill.csv"

def find_image_file(chosen_path: str) -> Path:
    """
    Find the original image file from the logged path.
    
    The CSV has paths like: "crop/mojo1/20250705_image.png"
    We need to search for the actual file in various locations.
    """
    # Try different possible locations
    filename = Path(chosen_path).name
    
    search_locations = [
        PROJECT_ROOT / "mojo1" / filename,
        PROJECT_ROOT / "mojo1_final" / filename,
        PROJECT_ROOT / "mojo2" / filename,
        PROJECT_ROOT / "mojo2_final" / filename,
        PROJECT_ROOT / "__crop" / filename,
        PROJECT_ROOT / "__selected" / filename,
        PROJECT_ROOT / "training data" / "mojo1" / filename,
        PROJECT_ROOT / "training data" / "mojo1_final" / filename,
        PROJECT_ROOT / "training data" / "mojo2" / filename,
        PROJECT_ROOT / "training data" / "mojo2_final" / filename,
    ]
    
    for location in search_locations:
        if location.exists():
            return location
    
    return None


def get_image_dimensions(image_path: Path) -> tuple:
    """Get (width, height) from an image file."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


def backfill_dimensions(dry_run=True):
    """Backfill missing dimensions in the crop log."""
    rows = []
    with CROP_LOG.open('r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    
    # Find rows with zero dimensions
    zero_dim_rows = []
    for i, row in enumerate(rows):
        try:
            w0 = int(row.get('width_0', 0))
            h0 = int(row.get('height_0', 0))
            w1 = int(row.get('width_1', 0))
            h1 = int(row.get('height_1', 0))
            
            if w0 == 0 or h0 == 0 or w1 == 0 or h1 == 0:
                zero_dim_rows.append(i)
        except Exception:
            pass
    
    
    if len(zero_dim_rows) == 0:
        return
    
    # Process each row that needs backfilling
    found = 0
    not_found = 0
    updated_rows = []
    
    
    for i in zero_dim_rows:
        row = rows[i]
        
        # Try to find the chosen image
        chosen_path = row.get('chosen_path', '')
        if not chosen_path:
            not_found += 1
            continue
        
        image_file = find_image_file(chosen_path)
        
        if image_file:
            width, height = get_image_dimensions(image_file)
            
            if width > 0 and height > 0:
                # Update the dimensions for the chosen image
                chosen_idx = int(row.get('chosen_index', 0))
                
                if chosen_idx == 0:
                    row['width_0'] = str(width)
                    row['height_0'] = str(height)
                elif chosen_idx == 1:
                    row['width_1'] = str(width)
                    row['height_1'] = str(height)
                
                found += 1
                updated_rows.append((i, chosen_path, width, height))
                
                if found % 100 == 0:
                    pass
            else:
                not_found += 1
        else:
            not_found += 1
    
    
    if dry_run:
        for i, _path, _w, _h in updated_rows[:10]:
            pass
    else:
        # Backup original file
        shutil.copy(CROP_LOG, BACKUP_LOG)
        
        # Write updated CSV
        with CROP_LOG.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        


def main():
    parser = argparse.ArgumentParser(description='Backfill missing crop dimensions')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without making changes')
    args = parser.parse_args()
    
    backfill_dimensions(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

