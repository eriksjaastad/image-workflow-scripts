#!/usr/bin/env python3
"""
Analyze Mojo 1 crops by comparing file modification times.

Logic:
- Same filename in both raw and final directories
- If final mtime == raw mtime → NOT cropped (just copied)
- If final mtime != raw mtime → CROPPED (edited and saved)
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


RAW_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1")
FINAL_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1_final")


def parse_filename(filename: str) -> Optional[Dict]:
    """Extract timestamp and stage from filename."""
    pattern = r'(\d{8}_\d{6})_stage(\d+(?:\.\d+)?)'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'timestamp': match.group(1),
            'stage': float(match.group(2)),
            'full_name': filename
        }
    return None


def main():
    print("=" * 80)
    print("MOJO 1 CROP ANALYSIS - FILE MODIFICATION TIME COMPARISON")
    print("=" * 80)
    print()
    
    # Build index of raw files by filename
    print("📂 Scanning raw directory...")
    raw_files = {}  # filename -> (path, mtime)
    
    for img_path in RAW_DIR.rglob("*.png"):
        mtime = img_path.stat().st_mtime
        raw_files[img_path.name] = (img_path, mtime)
    
    print(f"   Found {len(raw_files)} raw images")
    
    # Scan final files and compare mtimes
    print("\n📂 Scanning final directory...")
    final_files = list(FINAL_DIR.rglob("*.png"))
    print(f"   Found {len(final_files)} final images")
    
    print("\n🔍 Comparing file modification times...")
    
    not_cropped = []  # Same mtime
    cropped = []      # Different mtime
    not_in_raw = []   # Final file not found in raw
    
    for final_path in final_files:
        final_name = final_path.name
        final_mtime = final_path.stat().st_mtime
        
        if final_name in raw_files:
            raw_path, raw_mtime = raw_files[final_name]
            
            # Compare modification times (allow 1 second tolerance for filesystem precision)
            time_diff = abs(final_mtime - raw_mtime)
            
            if time_diff < 1:
                # Same time = NOT cropped (just copied)
                not_cropped.append({
                    'filename': final_name,
                    'raw_path': raw_path,
                    'final_path': final_path,
                    'raw_mtime': datetime.fromtimestamp(raw_mtime),
                    'final_mtime': datetime.fromtimestamp(final_mtime),
                    'time_diff': time_diff
                })
            else:
                # Different time = CROPPED (edited and saved)
                cropped.append({
                    'filename': final_name,
                    'raw_path': raw_path,
                    'final_path': final_path,
                    'raw_mtime': datetime.fromtimestamp(raw_mtime),
                    'final_mtime': datetime.fromtimestamp(final_mtime),
                    'time_diff': time_diff,
                    'days_diff': time_diff / 86400  # Convert to days
                })
        else:
            # Final file not in raw (shouldn't happen, but track it)
            not_in_raw.append({
                'filename': final_name,
                'final_path': final_path
            })
    
    # Results
    total = len(not_cropped) + len(cropped)
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total selections analyzed: {total}")
    print(f"NOT cropped (mtime match): {len(not_cropped)} ({len(not_cropped)/total*100:.1f}%)")
    print(f"CROPPED (mtime differs): {len(cropped)} ({len(cropped)/total*100:.1f}%)")
    if not_in_raw:
        print(f"Not found in raw: {len(not_in_raw)}")
    print()
    
    # Show examples of NOT cropped
    if not_cropped:
        print("=" * 80)
        print("EXAMPLES - NOT CROPPED (first 10)")
        print("=" * 80)
        for i, case in enumerate(not_cropped[:10], 1):
            print(f"{i}. {case['filename']}")
            print(f"   Raw mtime:   {case['raw_mtime']}")
            print(f"   Final mtime: {case['final_mtime']}")
            print(f"   Diff: {case['time_diff']:.3f} seconds")
            print()
    
    # Show examples of CROPPED
    if cropped:
        print("=" * 80)
        print("EXAMPLES - CROPPED (first 10)")
        print("=" * 80)
        for i, case in enumerate(cropped[:10], 1):
            print(f"{i}. {case['filename']}")
            print(f"   Raw mtime:   {case['raw_mtime']}")
            print(f"   Final mtime: {case['final_mtime']}")
            print(f"   Diff: {case['days_diff']:.1f} days ({case['time_diff']/3600:.1f} hours)")
            print(f"   ✂️  File was modified = cropped and saved")
            print()
    
    # Analyze crop timing distribution
    if cropped:
        print("=" * 80)
        print("CROP TIMING ANALYSIS")
        print("=" * 80)
        
        # Group by days difference
        day_buckets = defaultdict(int)
        for case in cropped:
            days = int(case['days_diff'])
            if days < 7:
                day_buckets['0-7 days'] += 1
            elif days < 30:
                day_buckets['7-30 days'] += 1
            elif days < 90:
                day_buckets['30-90 days'] += 1
            else:
                day_buckets['90+ days'] += 1
        
        for bucket, count in sorted(day_buckets.items()):
            print(f"  {bucket:15s}: {count:5d} crops ({count/len(cropped)*100:.1f}%)")
        print()
    
    # Summary for training
    print("=" * 80)
    print("TRAINING DATA SUMMARY")
    print("=" * 80)
    print(f"Total selections: {total}")
    print(f"  - Selected without cropping: {len(not_cropped)}")
    print(f"  - Selected AND cropped: {len(cropped)}")
    print()
    print(f"💡 Crop rate: {len(cropped)/total*100:.1f}%")
    print()
    print(f"The {len(cropped)} cropped images are valuable training data")
    print("for the crop proposer model.")
    print()


if __name__ == "__main__":
    main()
