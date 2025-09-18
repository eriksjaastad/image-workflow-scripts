#!/usr/bin/env python3
"""
Pair Compare - Side-by-Side Image Comparison Tool
=================================================
Displays two images side-by-side for direct comparison and allows you to
delete one or skip both. Perfect for choosing between similar images.

USAGE:
------
Activate virtual environment first:
  source venv/bin/activate

Run on directories containing images:
  python scripts/04_pair_compare.py face_group_1

CONTROLS:
---------
1 = Delete LEFT image (keep right)
2 = Delete RIGHT image (keep left)  
s = Skip both (move both to cropped/ - perfect as-is)
q = Quit

The script automatically moves through pairs of images, allowing you to
quickly eliminate duplicates or choose between similar options.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import shutil
from file_tracker import FileTracker

# Check for send2trash
try:
    from send2trash import send2trash
    _SEND2TRASH_AVAILABLE = True
except ImportError:
    _SEND2TRASH_AVAILABLE = False

# Matplotlib for display & key handling
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    print("❌ Missing required packages. Install with:")
    print("pip install matplotlib pillow")
    sys.exit(1)


def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get all image files from directory."""
    if extensions is None:
        extensions = ['png', 'jpg', 'jpeg']
    
    files = []
    exts_lower = [ext.lower() for ext in extensions]
    
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower().lstrip('.') in exts_lower:
            files.append(p)
    return sorted(files)


def open_image(path: Path):
    """Open and return PIL image."""
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open {path.name}: {e}")
        return None


def show_image_pair(img1_path: Path, img2_path: Path, pair_num: int, remaining_pairs: int) -> int:
    """Display two images side-by-side and handle user input."""
    img1 = open_image(img1_path)
    img2 = open_image(img2_path)
    
    # Create figure with two subplots side-by-side, taller aspect ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    
    # Left image
    ax1.axis('off')
    if img1 is not None:
        ax1.imshow(img1)
    ax1.set_title(f"LEFT: {img1_path.name}", fontsize=12, pad=10)
    
    # Right image  
    ax2.axis('off')
    if img2 is not None:
        ax2.imshow(img2)
    ax2.set_title(f"RIGHT: {img2_path.name}", fontsize=12, pad=10)
    
    # Main title with instructions and progress
    pairs_left = remaining_pairs - 1  # Current pair is being processed, so subtract 1
    fig.suptitle(f"1=Delete LEFT | 2=Delete RIGHT | 3=Delete BOTH | x=Move LEFT to crop/ | c=Move RIGHT to crop/ | s=Skip BOTH to cropped/ | q=Quit | Pair {pair_num} | ({pairs_left} remaining)", 
                 fontsize=11, y=0.95)
    
    result = {"choice": None}
    
    def on_key(event):
        key = (event.key or "").lower()
        if key == "1":
            result["choice"] = 1  # Delete left image
            plt.close(fig)
        elif key == "2":
            result["choice"] = 2  # Delete right image
            plt.close(fig)
        elif key == "3":
            result["choice"] = 3  # Delete both images
            plt.close(fig)
        elif key == "x":
            result["choice"] = 4  # Move left to crop
            plt.close(fig)
        elif key == "c":
            result["choice"] = 5  # Move right to crop
            plt.close(fig)
        elif key == "s":
            result["choice"] = 0  # Skip both
            plt.close(fig)
        elif key == "q":
            result["choice"] = -1  # Quit
            plt.close(fig)
        elif key == "h":
            print("Controls: 1=delete left | 2=delete right | 3=delete both | x=move left to crop/ | c=move right to crop/ | s=skip both to cropped/ | q=quit")
    
    # Disable matplotlib's default key shortcuts to prevent save dialog
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    
    return result["choice"] if result["choice"] is not None else 0


def delete_file_with_yaml(png_path: Path, use_trash: bool = True, tracker: FileTracker = None):
    """Delete PNG file and its corresponding YAML file."""
    base_name = png_path.stem
    yaml_path = png_path.parent / f"{base_name}.yaml"
    
    files_to_delete = [png_path]
    if yaml_path.exists():
        files_to_delete.append(yaml_path)
    
    deleted_files = []
    for file_path in files_to_delete:
        try:
            if use_trash and _SEND2TRASH_AVAILABLE:
                send2trash(str(file_path))
                print(f"🗑️  Sent to trash: {file_path.name}")
                deleted_files.append(file_path.name)
            else:
                file_path.unlink()
                print(f"❌ Deleted: {file_path.name}")
                deleted_files.append(file_path.name)
        except Exception as e:
            print(f"⚠️  Error deleting {file_path.name}: {e}")
    
    # Log the deletion operation
    if tracker and deleted_files:
        operation_type = "send_to_trash" if (use_trash and _SEND2TRASH_AVAILABLE) else "delete"
        tracker.log_operation(
            operation=operation_type,
            source_dir=str(png_path.parent.name),
            file_count=len(deleted_files),
            files=deleted_files,
            notes="User chose to delete via pair comparison"
        )


def move_file_to_cropped(png_path: Path, tracker: FileTracker = None):
    """Move PNG file and its corresponding YAML file to top-level cropped/ folder."""
    # Find the project root (containing venv/ and scripts/ directories)
    current_dir = png_path.parent
    while current_dir.parent != current_dir:  # Not at filesystem root
        if (current_dir / "venv").exists() and (current_dir / "scripts").exists():
            break
        current_dir = current_dir.parent
    
    # Create cropped directory at the project root level
    cropped_dir = current_dir / "cropped"
    cropped_dir.mkdir(exist_ok=True)
    
    base_name = png_path.stem
    yaml_path = png_path.parent / f"{base_name}.yaml"
    
    files_to_move = [png_path]
    if yaml_path.exists():
        files_to_move.append(yaml_path)
    
    moved_files = []
    for file_path in files_to_move:
        try:
            target_path = cropped_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            print(f"✓ Moved to cropped/: {file_path.name}")
            moved_files.append(file_path.name)
        except Exception as e:
            print(f"⚠️  Error moving {file_path.name}: {e}")
    
    # Log the move operation
    if tracker and moved_files:
        tracker.log_operation(
            operation="move",
            source_dir=str(png_path.parent.name),
            dest_dir="cropped",
            file_count=len(moved_files),
            files=moved_files,
            notes="User skipped - moved directly to cropped (perfect as-is)"
        )

def move_file_to_crop(png_path: Path, tracker: FileTracker = None):
    """Move PNG file and its corresponding YAML file to top-level crop/ folder."""
    # Find the project root (containing venv/ and scripts/ directories)
    current_dir = png_path.parent
    while current_dir.parent != current_dir:  # Not at filesystem root
        if (current_dir / "venv").exists() and (current_dir / "scripts").exists():
            break
        current_dir = current_dir.parent
    
    # Create crop directory at the project root level
    crop_dir = current_dir / "crop"
    crop_dir.mkdir(exist_ok=True)
    
    base_name = png_path.stem
    yaml_path = png_path.parent / f"{base_name}.yaml"
    
    files_to_move = [png_path]
    if yaml_path.exists():
        files_to_move.append(yaml_path)
    
    moved_files = []
    for file_path in files_to_move:
        try:
            target_path = crop_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            print(f"📁 Moved to crop/: {file_path.name}")
            moved_files.append(file_path.name)
        except Exception as e:
            print(f"⚠️  Error moving {file_path.name}: {e}")
    
    # Log the move operation
    if tracker and moved_files:
        tracker.log_operation(
            operation="move",
            source_dir=str(png_path.parent.name),
            dest_dir="crop",
            file_count=len(moved_files),
            files=moved_files,
            notes="User chose to move to crop via pair comparison"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare image pairs side-by-side")
    parser.add_argument("directory", help="Directory containing images to compare")
    parser.add_argument("--hard-delete", action="store_true", 
                       help="Permanently delete files instead of moving to trash")
    parser.add_argument("--extensions", default="png,jpg,jpeg", 
                       help="Comma-separated list of file extensions (default: png,jpg,jpeg)")
    
    args = parser.parse_args()
    
    # Initialize file tracker
    tracker = FileTracker("pair_compare")
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    # Check for send2trash if not using hard delete
    if not args.hard_delete and not _SEND2TRASH_AVAILABLE:
        print("❌ send2trash not installed. Install it with: pip install send2trash")
        print("Or use --hard-delete to permanently delete files (dangerous).")
        sys.exit(1)
    
    # Get image files
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    image_files = get_image_files(directory, extensions)
    
    if len(image_files) < 2:
        print(f"❌ Need at least 2 images in directory. Found: {len(image_files)}")
        sys.exit(1)
    
    print(f"🖼️  Found {len(image_files)} images in {directory}")
    print(f"📊 Will compare {len(image_files) // 2} pairs")
    print("Controls: 1=delete left | 2=delete right | 3=delete both | x=move left to crop/ | c=move right to crop/ | s=skip | q=quit")
    print()
    
    # Process pairs
    index = 0
    pairs_processed = 0
    
    while index < len(image_files) - 1:
        pairs_processed += 1
        remaining_pairs = (len(image_files) - index) // 2
        
        img1_path = image_files[index]
        img2_path = image_files[index + 1]
        
        print(f"=== Pair {pairs_processed} ===")
        print(f"LEFT:  {img1_path.name}")
        print(f"RIGHT: {img2_path.name}")
        
        choice = show_image_pair(img1_path, img2_path, pairs_processed, remaining_pairs)
        
        if choice == 1:  # Delete left image
            delete_file_with_yaml(img1_path, use_trash=not args.hard_delete, tracker=tracker)
            # Remove from list so we don't process it again
            image_files.remove(img1_path)
            # Don't increment index - next image moves into current position
            
        elif choice == 2:  # Delete right image
            delete_file_with_yaml(img2_path, use_trash=not args.hard_delete, tracker=tracker)
            # Remove from list
            image_files.remove(img2_path)
            # Don't increment index - next image moves into current position
            
        elif choice == 3:  # Delete both images
            delete_file_with_yaml(img1_path, use_trash=not args.hard_delete, tracker=tracker)
            delete_file_with_yaml(img2_path, use_trash=not args.hard_delete, tracker=tracker)
            # Remove both from list
            image_files.remove(img1_path)
            image_files.remove(img2_path)
            # Don't increment index - next images move into current position
            
        elif choice == 4:  # Move left image to crop
            move_file_to_crop(img1_path, tracker=tracker)
            # Remove from list
            image_files.remove(img1_path)
            # Don't increment index - next image moves into current position
            
        elif choice == 5:  # Move right image to crop
            move_file_to_crop(img2_path, tracker=tracker)
            # Remove from list
            image_files.remove(img2_path)
            # Don't increment index - next image moves into current position
            
        elif choice == 0:  # Skip both - move to cropped directory (perfect as-is)
            move_file_to_cropped(img1_path, tracker=tracker)
            move_file_to_cropped(img2_path, tracker=tracker)
            # Remove both from list since they're moved
            image_files.remove(img1_path)
            image_files.remove(img2_path)
            # Don't increment index - next pair moves into current position
            
        elif choice == -1:  # Quit
            print("👋 Quitting...")
            break
    
    print(f"\n✅ Completed! Processed {pairs_processed} pairs.")
    print(f"📁 Remaining images: {len(image_files)}")


if __name__ == "__main__":
    main()
