#!/usr/bin/env python3
"""
Backfill missing image dimensions in select_crop_log.csv

SAFETY FEATURES:
1. Always creates backup before touching original
2. Writes to temp file, only replaces on success
3. Test mode processes only 20 rows
4. Dry run shows what would happen without writing

Usage:
    # Test on 20 rows, no changes
    python scripts/ai/backfill_select_crop_dimensions.py --test --dry-run
    
    # Test on 20 rows, write to temp file
    python scripts/ai/backfill_select_crop_dimensions.py --test
    
    # Full run, dry run first
    python scripts/ai/backfill_select_crop_dimensions.py --dry-run
    
    # Full run, actually write
    python scripts/ai/backfill_select_crop_dimensions.py
"""

import csv
import shutil
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]
CROP_LOG = WORKSPACE / "data" / "training" / "select_crop_log.csv"
MOJO1_DIR = WORKSPACE / "training data" / "mojo1"
MOJO2_DIR = WORKSPACE / "training data" / "mojo2"

# Project date ranges (from manifest data)
MOJO1_START = datetime(2025, 10, 1)
MOJO1_END = datetime(2025, 10, 11, 23, 59, 59)
MOJO2_START = datetime(2025, 10, 12)
MOJO2_END = datetime(2025, 10, 20, 23, 59, 59)


def get_project_from_timestamp(timestamp_str: str) -> str:
    """Map timestamp to project (mojo1 or mojo2)"""
    try:
        # Parse ISO format timestamp
        ts = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        
        # Extract just the date for comparison (ignore time)
        ts_date = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if MOJO1_START <= ts_date <= MOJO1_END:
            return "mojo1"
        elif MOJO2_START <= ts_date <= MOJO2_END:
            return "mojo2"
        else:
            return "unknown"
    except Exception as e:
        print(f"   ⚠️  Error parsing timestamp '{timestamp_str}': {e}")
        return "unknown"


def find_image_in_training_data(filename: str, project: str) -> Path | None:
    """
    Find the original image in training data directories.
    
    Args:
        filename: e.g. "20250705_230713_stage3_enhanced.png"
        project: "mojo1" or "mojo2"
    
    Returns:
        Path to image if found, None otherwise
    """
    if project == "mojo1":
        search_dir = MOJO1_DIR
    elif project == "mojo2":
        search_dir = MOJO2_DIR
    else:
        return None
    
    if not search_dir.exists():
        return None
    
    # Search recursively for the image
    for img_path in search_dir.rglob(filename):
        if img_path.is_file():
            return img_path
    
    return None


def get_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """Get image dimensions (width, height)"""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"   ⚠️  Error reading {image_path.name}: {e}")
        return None


def backfill_dimensions(dry_run: bool = True, test_mode: bool = False, test_count: int = 20):
    """
    Backfill missing dimensions in select_crop_log.csv
    
    Args:
        dry_run: If True, show what would happen but don't write
        test_mode: If True, only process test_count rows
        test_count: Number of rows to process in test mode (default 20)
    """
    
    if not CROP_LOG.exists():
        print(f"❌ Error: {CROP_LOG} not found!")
        return
    
    print(f"\n{'='*70}")
    print(f"BACKFILL SELECT_CROP_LOG.CSV DIMENSIONS")
    print(f"{'='*70}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'} | {'TEST (' + str(test_count) + ' rows)' if test_mode else 'FULL'}")
    print(f"Log file: {CROP_LOG}")
    print(f"Mojo1 dir: {MOJO1_DIR} {'✅' if MOJO1_DIR.exists() else '❌ NOT FOUND'}")
    print(f"Mojo2 dir: {MOJO2_DIR} {'✅' if MOJO2_DIR.exists() else '❌ NOT FOUND'}")
    
    # Read existing CSV
    print(f"\n📖 Reading CSV...")
    with CROP_LOG.open('r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"   Total rows: {len(rows)}")
    
    # Identify rows with missing dimensions
    rows_to_process = []
    for i, row in enumerate(rows):
        has_crop = (row.get('crop_x1', '') != '' and 
                   row.get('crop_y1', '') != '' and
                   row.get('crop_x2', '') != '' and
                   row.get('crop_y2', '') != '')
        
        # CRITICAL: Validate timestamp to avoid corrupted rows
        # Corrupted rows have garbage in crop columns and missing/invalid timestamps
        timestamp = row.get('timestamp', '')
        has_valid_timestamp = False
        if timestamp and timestamp != 'None' and timestamp.strip() != '':
            try:
                datetime.fromisoformat(timestamp.replace('Z', ''))
                has_valid_timestamp = True
            except:
                pass
        
        # Check if dimensions are missing or zero
        # Handle None, empty string, and '0' as "missing"
        w0 = row.get('width_0')
        h0 = row.get('height_0')
        w1 = row.get('width_1')
        h1 = row.get('height_1')
        
        # Check if w0/h0 are valid (not None, not empty, not '0')
        w0_valid = w0 is not None and w0 != '' and w0 != '0'
        h0_valid = h0 is not None and h0 != '' and h0 != '0'
        
        # Check if w1/h1 are valid (not None, not empty, not '0')
        w1_valid = w1 is not None and w1 != '' and w1 != '0'
        h1_valid = h1 is not None and h1 != '' and h1 != '0'
        
        # Has dims if EITHER (w0 and h0 are valid) OR (w1 and h1 are valid)
        has_dims = ((w0_valid and h0_valid) or (w1_valid and h1_valid))
        
        # ONLY process rows that have:
        # 1. Crop coordinates
        # 2. Valid timestamp (to avoid corrupted rows)
        # 3. Missing dimensions
        if has_crop and has_valid_timestamp and not has_dims:
            rows_to_process.append((i, row))
            if test_mode and len(rows_to_process) >= test_count:
                break
    
    print(f"   Rows with missing dimensions: {len(rows_to_process)}")
    
    if not rows_to_process:
        print("✅ Nothing to backfill!")
        return
    
    # Process rows
    print(f"\n🔍 Processing rows...")
    success_count = 0
    fail_count = 0
    project_counts = {"mojo1": 0, "mojo2": 0, "unknown": 0}
    
    for idx, (row_idx, row) in enumerate(rows_to_process, 1):
        # Get project from timestamp
        timestamp = row.get('timestamp', '')
        project = get_project_from_timestamp(timestamp)
        project_counts[project] += 1
        
        # Extract filename from chosen_path
        chosen_path = row.get('chosen_path', '')
        if not chosen_path:
            print(f"   ⚠️  Row {row_idx+2}: No chosen_path")
            fail_count += 1
            continue
        
        filename = Path(chosen_path).name
        
        # Find image
        img_path = find_image_in_training_data(filename, project)
        
        if not img_path:
            if idx <= 5:  # Only print first 5 failures
                print(f"   ❌ Row {row_idx+2}: {filename} not found in {project}/")
            fail_count += 1
            continue
        
        # Get dimensions
        dims = get_image_dimensions(img_path)
        if not dims:
            fail_count += 1
            continue
        
        width, height = dims
        
        # Update row (fill in ALL dimension columns for consistency)
        rows[row_idx]['width_0'] = str(width)
        rows[row_idx]['height_0'] = str(height)
        
        # Also fill width_1/height_1 if they exist and are empty
        if 'width_1' in rows[row_idx] and rows[row_idx].get('width_1', '') in ['', '0']:
            rows[row_idx]['width_1'] = str(width)
        if 'height_1' in rows[row_idx] and rows[row_idx].get('height_1', '') in ['', '0']:
            rows[row_idx]['height_1'] = str(height)
        
        success_count += 1
        
        # Print progress for first 20
        if idx <= 20:
            print(f"   ✅ Row {row_idx+2}: {filename} → {width}x{height} (from {project}/)")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed:  {fail_count}")
    print(f"   📁 Projects: mojo1={project_counts['mojo1']}, mojo2={project_counts['mojo2']}, unknown={project_counts['unknown']}")
    
    # Write updated CSV
    if not dry_run:
        if not test_mode:
            # Create backup
            backup_path = CROP_LOG.with_name(f"{CROP_LOG.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            print(f"\n💾 Creating backup: {backup_path.name}")
            shutil.copy(CROP_LOG, backup_path)
            
            # Verify backup
            backup_size = backup_path.stat().st_size
            original_size = CROP_LOG.stat().st_size
            if backup_size != original_size:
                print(f"❌ ERROR: Backup size mismatch! {backup_size} != {original_size}")
                return
            print(f"   ✅ Backup verified ({backup_size:,} bytes)")
        
        # Write to temp file first
        temp_path = CROP_LOG.with_suffix('.csv.tmp')
        print(f"\n✍️  Writing to temp file: {temp_path.name}")
        
        with temp_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        # Verify temp file
        if not temp_path.exists():
            print(f"❌ ERROR: Temp file not created!")
            return
        
        temp_size = temp_path.stat().st_size
        print(f"   ✅ Temp file created ({temp_size:,} bytes)")
        
        if not test_mode:
            # Replace original with temp
            print(f"\n🔄 Replacing original with temp file...")
            temp_path.replace(CROP_LOG)
            print(f"   ✅ Done! {CROP_LOG.name} updated")
        else:
            print(f"\n⚠️  TEST MODE: Temp file saved as {temp_path.name}")
            print(f"   Review it, then manually rename to {CROP_LOG.name} if good")
    else:
        print(f"\n⚠️  DRY RUN - No changes made to {CROP_LOG.name}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill missing dimensions in select_crop_log.csv')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without writing')
    parser.add_argument('--test', action='store_true', help='Test mode (use with --count to specify rows)')
    parser.add_argument('--count', type=int, default=20, help='Number of rows to process in test mode (default: 20)')
    
    args = parser.parse_args()
    
    # Determine test count
    test_count = args.count if args.test else None
    
    backfill_dimensions(dry_run=args.dry_run, test_mode=args.test, test_count=test_count)

