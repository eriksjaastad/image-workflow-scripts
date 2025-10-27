#!/usr/bin/env python3
"""
Backfill Missing Dimensions in mojo1_crop_log.csv

Problem: mojo1_crop_log.csv has 5,157 rows but width_0/height_0 are empty
Solution: Read the original mojo1 images to get actual dimensions
Result: Make mojo1 crop data usable for training

Usage:
    # Test with first 20 rows only (dry-run)
    python scripts/ai/backfill_mojo1_dimensions.py --test --dry-run
    
    # Test with first 20 rows (actually update)
    python scripts/ai/backfill_mojo1_dimensions.py --test
    
    # Process all rows (dry-run)
    python scripts/ai/backfill_mojo1_dimensions.py --dry-run
    
    # Process all rows (full backfill)
    python scripts/ai/backfill_mojo1_dimensions.py
"""

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path("PROJECT_ROOT")
CROP_LOG = PROJECT_ROOT / "data/training/mojo1_crop_log.csv"
BACKUP_LOG = PROJECT_ROOT / "data/training/mojo1_crop_log_backup_before_backfill.csv"
MOJO1_DIR = PROJECT_ROOT / "training data" / "mojo1"

def find_image_in_mojo1(filename: str) -> Path:
    """
    Find an image file in the mojo1 directory.
    
    Args:
        filename: Just the filename (e.g., "20250705_215711_stage2_upscaled.png")
    
    Returns:
        Path to the file, or None if not found
    """
    image_path = MOJO1_DIR / filename
    
    if image_path.exists():
        return image_path
    
    return None


def get_image_dimensions(image_path: Path) -> tuple:
    """Get (width, height) from an image file."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"   ⚠️  Error reading {image_path.name}: {e}")
        return (0, 0)


def backfill_mojo1_dimensions(dry_run=True, test_mode=False):
    """
    Backfill missing dimensions in mojo1_crop_log.csv
    
    Args:
        dry_run: If True, don't write changes to disk
        test_mode: If True, only process first 20 rows
    """
    
    print(f"\n{'='*70}")
    print("🔧 BACKFILL MOJO1 DIMENSIONS")
    print(f"{'='*70}")
    
    if not MOJO1_DIR.exists():
        print("\n❌ ERROR: Mojo1 directory not found!")
        print(f"   Expected: {MOJO1_DIR}")
        print("   Please make sure mojo1 images are in 'training data/mojo1/'")
        return
    
    # Count images in mojo1 directory
    image_count = len(list(MOJO1_DIR.glob("*.png"))) + len(list(MOJO1_DIR.glob("*.jpg")))
    print(f"\n📂 Mojo1 directory: {MOJO1_DIR}")
    print(f"   Images found: {image_count:,}")
    
    print(f"\n📂 Reading crop log: {CROP_LOG}")
    
    # Read CSV
    rows = []
    with CROP_LOG.open('r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"   Total rows: {len(rows):,}")
    
    # Find rows with missing dimensions
    missing_dim_rows = []
    for i, row in enumerate(rows):
        w0 = row.get('width_0', '').strip()
        h0 = row.get('height_0', '').strip()
        
        # Check if width_0 or height_0 is empty or zero
        if not w0 or not h0 or w0 == '0' or h0 == '0':
            missing_dim_rows.append(i)
    
    print(f"   Rows with missing dimensions: {len(missing_dim_rows):,}")
    
    if len(missing_dim_rows) == 0:
        print("✅ No rows need backfilling!")
        return
    
    # Test mode: limit to first 20
    if test_mode:
        print("\n⚠️  TEST MODE: Processing only first 20 rows")
        missing_dim_rows = missing_dim_rows[:20]
    
    # Process each row
    found = 0
    not_found = 0
    updated_rows = []
    
    print("\n🔍 Searching for images and reading dimensions...")
    print(f"   Processing {len(missing_dim_rows):,} rows...")
    
    for idx, i in enumerate(missing_dim_rows):
        row = rows[i]
        
        # Get the chosen image path
        chosen_path = row.get('chosen_path', '').strip()
        if not chosen_path:
            not_found += 1
            continue
        
        # Extract just the filename
        filename = Path(chosen_path).name
        
        # Find the image in mojo1 directory
        image_file = find_image_in_mojo1(filename)
        
        if image_file:
            width, height = get_image_dimensions(image_file)
            
            if width > 0 and height > 0:
                # Update the dimensions for the chosen image
                chosen_idx = int(row.get('chosen_index', 0))
                
                if chosen_idx == 0:
                    row['width_0'] = str(width)
                    row['height_0'] = str(height)
                else:
                    row['width_1'] = str(width)
                    row['height_1'] = str(height)
                
                found += 1
                updated_rows.append((i, filename, width, height))
                
                if (found % 100 == 0) or (test_mode and found % 5 == 0):
                    print(f"   Progress: {found:,} found, {not_found:,} not found")
            else:
                not_found += 1
        else:
            not_found += 1
            if test_mode and not_found <= 5:
                print(f"   ⚠️  Not found: {filename}")
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}")
    print(f"   ✅ Found and updated: {found:,} images")
    print(f"   ❌ Not found: {not_found:,} images")
    
    if found > 0:
        print(f"   📈 Success rate: {found*100/(found+not_found):.1f}%")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN - No changes made to {CROP_LOG.name}")
        print("\n   Sample updates (first 10):")
        for i, filename, w, h in updated_rows[:10]:
            print(f"      Row {i}: {filename[:50]:<50} → {w}x{h}")
        print("\n   ✅ Run without --dry-run to actually update the CSV")
    else:
        # Backup original file
        print("\n💾 Creating backup:")
        print(f"   {BACKUP_LOG.name}")
        shutil.copy(CROP_LOG, BACKUP_LOG)
        
        # Write updated CSV
        print("\n✍️  Writing updated CSV...")
        with CROP_LOG.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Updated {found:,} rows in {CROP_LOG.name}")
        print(f"{'='*70}")
        
        if test_mode:
            print("\n⚠️  This was a TEST RUN (20 rows)")
            print(f"   Run without --test to process all {len(missing_dim_rows):,} rows")


def main():
    parser = argparse.ArgumentParser(description='Backfill mojo1 crop dimensions')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without making changes')
    parser.add_argument('--test', action='store_true',
                        help='Process only first 20 rows (test mode)')
    args = parser.parse_args()
    
    backfill_mojo1_dimensions(dry_run=args.dry_run, test_mode=args.test)


if __name__ == "__main__":
    main()

