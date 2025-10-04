#!/usr/bin/env python3
"""
Simple script to move orphaned caption files to their matching directories.
This is much cheaper than re-running the image grouper.
"""

import os
import shutil
from pathlib import Path

def find_matching_png_directory(caption_file: Path, clustered_dir: Path):
    """Find where the matching PNG file is located."""
    base_name = caption_file.stem  # Remove .caption extension
    png_name = f"{base_name}.png"
    
    # Search for the PNG file in the clustered directory
    for png_file in clustered_dir.rglob(png_name):
        return png_file.parent
    
    return None

def main():
    # Configuration
    mojo1_dir = Path("mojo1")
    clustered_dir = Path("mojo1_clustered")
    
    if not mojo1_dir.exists() or not clustered_dir.exists():
        print("❌ Required directories not found!")
        return
    
    # Find all caption files in mojo1
    caption_files = list(mojo1_dir.glob("*.caption"))
    print(f"Found {len(caption_files)} caption files to process...")
    
    moved_count = 0
    not_found_count = 0
    
    for caption_file in caption_files:
        # Find where the matching PNG is located
        target_dir = find_matching_png_directory(caption_file, clustered_dir)
        
        if target_dir:
            # Move the caption file to the same directory as the PNG
            target_path = target_dir / caption_file.name
            if not target_path.exists():
                shutil.move(str(caption_file), str(target_path))
                print(f"✅ Moved: {caption_file.name} → {target_dir.name}/")
                moved_count += 1
            else:
                print(f"⚠️  Already exists: {target_path}")
        else:
            print(f"❌ No matching PNG found for: {caption_file.name}")
            not_found_count += 1
    
    print(f"\n📊 Results:")
    print(f"  ✅ Moved: {moved_count} caption files")
    print(f"  ❌ Not found: {not_found_count} caption files")
    print(f"  📁 Total processed: {len(caption_files)}")

if __name__ == "__main__":
    main()
