#!/usr/bin/env python3
"""
Character Sorter - Single Image Classification Tool with Notes Panel

## To activate the virtual environment:
source venv/bin/activate

## To deactivate when done:
deactivate

This script displays images one at a time with a notes panel and allows you to sort them into:
- character_group_1 directory (press 1)
- character_group_2 directory (press 2)
- character_group_3 directory (press 3)
- Delete (press d)

Features:
- Side panel with persistent review notes and reminders
- Notes stay visible throughout the entire sorting session
- Both PNG and YAML files are moved together

Usage:
    python scripts/03_character_sorter.py <directory>

Example:
    python scripts/03_character_sorter.py face_group_1/

Keys in viewer:
  1 = character_group_1 • 2 = character_group_2 • 3 = character_group_3 • d = delete • s = skip • b = back • q = quit • h = help

To customize notes:
  Edit the get_review_notes() function in this file to add your own persistent
  review reminders and checklists.
"""

import argparse
import csv
import os
import sys
import shutil
from pathlib import Path
from typing import List
from file_tracker import FileTracker

# Matplotlib for display & key handling
try:
    import matplotlib
    try:
        matplotlib.use("MacOSX")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except Exception:
    print("[!] matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    raise

# Pillow
try:
    from PIL import Image
except Exception:
    print("[!] Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise

# Optional Trash
_SEND2TRASH_AVAILABLE = False
try:
    from send2trash import send2trash
    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False


def human_err(msg: str):
    print(f"[!] {msg}", file=sys.stderr)


def info(msg: str):
    print(f"[*] {msg}")


def open_image(path: Path):
    """Open and convert image to RGB."""
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        human_err(f"Failed to open {path.name}: {e}")
        return None


def scan_images(folder: Path) -> List[Path]:
    """Scan folder for PNG files."""
    png_files = sorted(folder.glob("*.png"))
    return png_files


def get_review_notes() -> str:
    """Get persistent review notes that stay visible throughout the sorting session.
    Edit this function to customize your review reminders."""
    
    # ===== EDIT YOUR PERSISTENT NOTES BELOW =====
    
    return """REVIEW CHECKLIST:

🔍 ANATOMY CHECK:
• 

⚖️ QUALITY ASSESSMENT:
• The double underwear not ok
• crop more
• more critical of consistent face shapes

📏 CHARACTER CLASSIFICATION:
• character_group_1 (1): First category
• character_group_2 (2): Second category
• character_group_3 (3): Third category
• Delete (d): Poor quality, anomalies

⚠️ RED FLAGS:
• 

💡 TIPS:
• Take your time on each image
• When in doubt, compare similar poses
• Trust your instincts on quality
• Skip if unsure, come back later

CONTROLS:
1=Group_1 | 2=Group_2 | 3=Group_3 | d=Delete
s=Skip | b=Back | q=Quit | h=Help"""
    
    # ===== END NOTES SECTION =====




def show_image(img_path: Path, current_idx: int, total_count: int) -> int:
    """Display single image and handle user input with notes panel."""
    img = open_image(img_path)
    
    # Create figure with two subplots: image (left) and notes (right)
    fig, (ax_img, ax_notes) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    
    # Image panel (left)
    ax_img.axis('off')
    if img is not None:
        ax_img.imshow(img)
    
    # Title with progress and filename
    title = f"Image {current_idx + 1}/{total_count}: {img_path.name}"
    ax_img.set_title(title, fontsize=12, pad=10)

    # Notes panel (right)
    ax_notes.axis('off')
    notes_text = get_review_notes()
    ax_notes.text(0.05, 0.95, "REVIEW NOTES:", fontsize=12, fontweight='bold', 
                  transform=ax_notes.transAxes, verticalalignment='top')
    ax_notes.text(0.05, 0.88, notes_text, fontsize=9, 
                  transform=ax_notes.transAxes, verticalalignment='top',
                  wrap=True, fontfamily='monospace')
    
    # Add border around notes panel
    ax_notes.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                    fill=False, edgecolor='gray', linewidth=1, 
                                    transform=ax_notes.transAxes))

    # Instructions
    fig.suptitle("1=character_group_1 | 2=character_group_2 | 3=character_group_3 | d=Delete | s=Skip | b=Back | q=Quit | h=Help", 
                 fontsize=11, y=0.02)

    result = {"choice": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key == "1":
            result["choice"] = 1  # character_group_1
            plt.close(fig)
        elif key == "2":
            result["choice"] = 2  # character_group_2
            plt.close(fig)
        elif key == "3":
            result["choice"] = 3  # character_group_3
            plt.close(fig)
        elif key == "d":
            result["choice"] = -1  # Delete
            plt.close(fig)
        elif key == "s":
            result["choice"] = 0   # Skip
            plt.close(fig)
        elif key == "b":
            result["choice"] = -2  # Back
            plt.close(fig)
        elif key == "q":
            result["choice"] = -3  # Quit
            plt.close(fig)
        elif key == "h":
            print("Keys: 1=character_group_1 | 2=character_group_2 | 3=character_group_3 | d=Delete | s=Skip | b=Back | q=Quit")

    # Disable matplotlib's default key shortcuts to prevent save dialog
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return result["choice"] if result["choice"] is not None else 0


def move_files(png_path: Path, target_dir: Path):
    """Move PNG and corresponding YAML file to target directory."""
    target_dir.mkdir(exist_ok=True)
    
    # Get base name without extension
    base_name = png_path.stem
    
    # Find corresponding YAML file
    yaml_path = png_path.parent / f"{base_name}.yaml"
    
    moved_files = []
    
    # Move PNG file
    png_moved = False
    try:
        target_png = target_dir / png_path.name
        shutil.move(str(png_path), str(target_png))
        moved_files.append(png_path.name)
        print(f"✓ Moved: {png_path.name} → {target_dir.name}")
        png_moved = True
    except Exception as e:
        print(f"❌ Error moving {png_path.name}: {e}")
    
    # Move YAML file if it exists - regardless of PNG move success
    if yaml_path.exists():
        try:
            target_yaml = target_dir / yaml_path.name
            shutil.move(str(yaml_path), str(target_yaml))
            moved_files.append(yaml_path.name)
            print(f"✓ Moved: {yaml_path.name} → {target_dir.name}")
        except Exception as e:
            print(f"❌ Error moving {yaml_path.name}: {e}")
    
    # Only return success if PNG was moved (YAML is optional)
    if not png_moved:
        return []
    
    return moved_files


def delete_files(png_path: Path, use_trash: bool = True):
    """Delete PNG and corresponding YAML file."""
    base_name = png_path.stem
    yaml_path = png_path.parent / f"{base_name}.yaml"
    
    files_to_delete = [png_path]
    if yaml_path.exists():
        files_to_delete.append(yaml_path)
    
    for file_path in files_to_delete:
        try:
            if use_trash and _SEND2TRASH_AVAILABLE:
                send2trash(str(file_path))
                print(f"🗑️  Sent to Trash: {file_path.name}")
            else:
                file_path.unlink()
                print(f"🗑️  Deleted: {file_path.name}")
        except Exception as e:
            print(f"❌ Error deleting {file_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Sort images into character_group_1, character_group_2, character_group_3, or delete')
    parser.add_argument('directory', help='Directory containing PNG files to sort')
    parser.add_argument('--hard-delete', action='store_true', 
                       help='Permanently delete files instead of sending to Trash')
    
    args = parser.parse_args()
    
    # Initialize file tracker
    tracker = FileTracker("character_sorter")
    
    source_dir = Path(args.directory).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        human_err(f"Directory not found: {source_dir}")
        sys.exit(1)
    
    # Set up target directories
    character_group_1 = source_dir.parent / "character_group_1"
    character_group_2 = source_dir.parent / "character_group_2"
    character_group_3 = source_dir.parent / "character_group_3"
    
    # Check if target directories exist
    if not character_group_1.exists():
        print(f"⚠️  Target directory {character_group_1} does not exist. Creating it.")
        character_group_1.mkdir(exist_ok=True)
    
    if not character_group_2.exists():
        print(f"⚠️  Target directory {character_group_2} does not exist. Creating it.")
        character_group_2.mkdir(exist_ok=True)
    
    if not character_group_3.exists():
        print(f"⚠️  Target directory {character_group_3} does not exist. Creating it.")
        character_group_3.mkdir(exist_ok=True)
    
    # Scan for images
    png_files = scan_images(source_dir)
    if not png_files:
        human_err(f"No PNG files found in {source_dir}")
        sys.exit(1)
    
    info(f"Found {len(png_files)} PNG files in {source_dir}")
    info(f"Target directories: {character_group_1.name}, {character_group_2.name}, {character_group_3.name}")
    
    if not args.hard_delete and not _SEND2TRASH_AVAILABLE:
        print("⚠️  send2trash not installed. Deleted files will be permanently removed.")
        print("Install send2trash with: pip install send2trash")
        print("Or use --hard-delete flag to acknowledge permanent deletion.")
    
    # Main sorting loop
    index = 0
    history = []
    changes_made = False
    
    while 0 <= index < len(png_files):
        current_file = png_files[index]
        print(f"\n=== Image {index + 1}/{len(png_files)}: {current_file.name} ===")
        
        choice = show_image(current_file, index, len(png_files))
        
        if choice == 1:  # character_group_1
            moved_files = move_files(current_file, character_group_1)
            if moved_files:
                history.append((current_file, "character_group_1"))
                changes_made = True
                tracker.log_operation("move", source_dir.name, "character_group_1", 
                                    len(moved_files), moved_files, f"User selected group 1")
                # Remove from list since it's been moved
                png_files.pop(index)
                if index >= len(png_files) and png_files:
                    index = len(png_files) - 1
            else:
                index += 1
                
        elif choice == 2:  # character_group_2
            moved_files = move_files(current_file, character_group_2)
            if moved_files:
                history.append((current_file, "character_group_2"))
                changes_made = True
                tracker.log_operation("move", source_dir.name, "character_group_2", 
                                    len(moved_files), moved_files, f"User selected group 2")
                # Remove from list since it's been moved
                png_files.pop(index)
                if index >= len(png_files) and png_files:
                    index = len(png_files) - 1
            else:
                index += 1
                
        elif choice == 3:  # character_group_3
            moved_files = move_files(current_file, character_group_3)
            if moved_files:
                history.append((current_file, "character_group_3"))
                changes_made = True
                tracker.log_operation("move", source_dir.name, "character_group_3", 
                                    len(moved_files), moved_files, f"User selected group 3")
                # Remove from list since it's been moved
                png_files.pop(index)
                if index >= len(png_files) and png_files:
                    index = len(png_files) - 1
            else:
                index += 1
                
        elif choice == -1:  # Delete
            yaml_file = current_file.with_suffix('.yaml')
            delete_files(current_file, use_trash=not args.hard_delete)
            history.append((current_file, "DELETED"))
            changes_made = True
            tracker.log_operation("delete", source_dir.name, None, 2, 
                                [current_file.name, yaml_file.name], f"User chose to delete")
            # Remove from list since it's been deleted
            png_files.pop(index)
            if index >= len(png_files) and png_files:
                index = len(png_files) - 1
                
        elif choice == 0:  # Skip
            index += 1
            
        elif choice == -2:  # Back
            if index > 0:
                index -= 1
            else:
                print("Already at the beginning.")
                
        elif choice == -3:  # Quit
            print("Quitting sorter.")
            break
        else:
            # Default to next image
            index += 1
    
    # Summary
    if changes_made:
        print(f"\n✅ Sorting complete! Processed {len(history)} actions.")
        
        # Count actions
        group1_count = sum(1 for _, action in history if action == "character_group_1")
        group2_count = sum(1 for _, action in history if action == "character_group_2")
        group3_count = sum(1 for _, action in history if action == "character_group_3")
        deleted_count = sum(1 for _, action in history if action == "DELETED")
        
        print(f"📊 Summary:")
        print(f"   → character_group_1: {group1_count}")
        print(f"   → character_group_2: {group2_count}")
        print(f"   → character_group_3: {group3_count}")
        print(f"   → Deleted: {deleted_count}")
        print(f"   → Remaining in source: {len(png_files)}")
    else:
        print("\n✅ No changes made.")


if __name__ == "__main__":
    main()

