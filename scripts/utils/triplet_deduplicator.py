#!/usr/bin/env python3
"""
Triplet Deduplication Utility - Remove triplets if any stage matches completed work

This utility prevents duplicates by checking incoming images against files already
in Kiara_Average_Completed. If ANY stage of a triplet matches an existing file,
the ENTIRE triplet is removed from the new batch.

Usage:
    python scripts/util_triplet_deduplicator.py <new_images_directory>

Example:
    python scripts/util_triplet_deduplicator.py "Raw_Images_New"
"""

import os
import shutil
import argparse
from pathlib import Path
import re
from send2trash import send2trash

def extract_base_timestamp(filename):
    """Extract the base timestamp from a filename (without stage suffix)."""
    # Match pattern like 20250726_010033
    match = re.match(r'(\d{8}_\d{6})', filename)
    return match.group(1) if match else None

def build_completed_database(completed_dir):
    """Build a set of all base timestamps from completed work."""
    completed_path = Path(completed_dir)
    completed_timestamps = set()
    
    if not completed_path.exists():
        print(f"⚠️  Completed directory not found: {completed_dir}")
        return completed_timestamps
    
    # Find all PNG files in completed directory
    png_files = list(completed_path.rglob("*.png"))
    
    for png_file in png_files:
        base_timestamp = extract_base_timestamp(png_file.name)
        if base_timestamp:
            completed_timestamps.add(base_timestamp)
    
    print(f"📊 Found {len(completed_timestamps)} unique timestamps in completed work")
    return completed_timestamps

def find_triplets_in_directory(directory):
    """Find all triplets in the given directory."""
    dir_path = Path(directory)
    triplets = {}
    
    # Find all PNG files and group by base timestamp
    png_files = list(dir_path.glob("*.png"))
    
    for png_file in png_files:
        base_timestamp = extract_base_timestamp(png_file.name)
        if not base_timestamp:
            continue
            
        if base_timestamp not in triplets:
            triplets[base_timestamp] = {}
        
        # Determine stage type
        if "_stage1_generated.png" in png_file.name:
            triplets[base_timestamp]['stage1'] = png_file
        elif "_stage1.5_face_swapped.png" in png_file.name:
            triplets[base_timestamp]['stage1_5'] = png_file
        elif "_stage2_upscaled.png" in png_file.name:
            triplets[base_timestamp]['stage2'] = png_file
    
    return triplets

def remove_triplet_files(triplet_files):
    """Remove all files in a triplet (PNG + YAML)."""
    removed_files = []
    
    for stage, png_file in triplet_files.items():
        if png_file and png_file.exists():
            # Remove PNG file
            try:
                send2trash(str(png_file))
                removed_files.append(png_file.name)
                print(f"    🗑️  {png_file.name}")
            except Exception as e:
                print(f"    ❌ Error removing {png_file.name}: {e}")
            
            # Remove corresponding YAML file
            yaml_file = png_file.parent / f"{png_file.stem}.yaml"
            if yaml_file.exists():
                try:
                    send2trash(str(yaml_file))
                    removed_files.append(yaml_file.name)
                    print(f"    🗑️  {yaml_file.name}")
                except Exception as e:
                    print(f"    ❌ Error removing {yaml_file.name}: {e}")
    
    return removed_files

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate triplets based on completed work')
    parser.add_argument('new_images_dir', help='Directory containing new images to deduplicate')
    parser.add_argument('--completed-dir', default='Kiara_Average_Completed',
                       help='Directory containing completed work (default: Kiara_Average_Completed)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed without actually removing files')
    
    args = parser.parse_args()
    
    new_images_path = Path(args.new_images_dir)
    if not new_images_path.exists():
        print(f"❌ New images directory not found: {args.new_images_dir}")
        return
    
    print(f"🔍 Scanning for duplicates...")
    print(f"   📂 New images: {args.new_images_dir}")
    print(f"   📂 Completed work: {args.completed_dir}")
    
    # Build database of completed work
    completed_timestamps = build_completed_database(args.completed_dir)
    
    if not completed_timestamps:
        print("✅ No completed work found - no deduplication needed")
        return
    
    # Find triplets in new images
    print(f"\n🔍 Analyzing triplets in new images...")
    new_triplets = find_triplets_in_directory(args.new_images_dir)
    
    if not new_triplets:
        print("✅ No triplets found in new images")
        return
    
    print(f"📊 Found {len(new_triplets)} potential triplets in new images")
    
    # Check for duplicates
    duplicates_found = []
    
    for base_timestamp, triplet_files in new_triplets.items():
        if base_timestamp in completed_timestamps:
            duplicates_found.append((base_timestamp, triplet_files))
    
    if not duplicates_found:
        print("✅ No duplicates found!")
        return
    
    print(f"\n⚠️  Found {len(duplicates_found)} duplicate triplets:")
    
    total_removed = 0
    
    for base_timestamp, triplet_files in duplicates_found:
        print(f"\n🔍 Duplicate triplet: {base_timestamp}")
        
        if args.dry_run:
            print("    💡 DRY RUN - would remove:")
            for stage, png_file in triplet_files.items():
                if png_file and png_file.exists():
                    print(f"    🗑️  {png_file.name}")
                    yaml_file = png_file.parent / f"{png_file.stem}.yaml"
                    if yaml_file.exists():
                        print(f"    🗑️  {yaml_file.name}")
        else:
            print("    🗑️  Removing:")
            removed_files = remove_triplet_files(triplet_files)
            total_removed += len(removed_files)
    
    if args.dry_run:
        print(f"\n💡 DRY RUN: Would remove {len(duplicates_found)} duplicate triplets")
    else:
        print(f"\n✅ Removed {len(duplicates_found)} duplicate triplets ({total_removed} files)")
        print(f"🎯 New images directory is now deduplicated against completed work!")

if __name__ == "__main__":
    main()
