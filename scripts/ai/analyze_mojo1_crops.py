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

RAW_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1")
FINAL_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1_final")


def parse_filename(filename: str) -> dict | None:
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
    
    # Build index of raw files by filename
    raw_files = {}  # filename -> (path, mtime)
    
    for img_path in RAW_DIR.rglob("*.png"):
        mtime = img_path.stat().st_mtime
        raw_files[img_path.name] = (img_path, mtime)
    
    
    # Scan final files and compare mtimes
    final_files = list(FINAL_DIR.rglob("*.png"))
    
    
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
    len(not_cropped) + len(cropped)
    
    if not_in_raw:
        pass
    
    # Show examples of NOT cropped
    if not_cropped:
        for _i, case in enumerate(not_cropped[:10], 1):
            pass
    
    # Show examples of CROPPED
    if cropped:
        for _i, case in enumerate(cropped[:10], 1):
            pass
    
    # Analyze crop timing distribution
    if cropped:
        
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
        
        for _bucket, _count in sorted(day_buckets.items()):
            pass
    
    # Summary for training


if __name__ == "__main__":
    main()
