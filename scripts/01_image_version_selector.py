#!/usr/bin/env python3
"""
Image Version Selector (Step 1 of Workflow)
===========================================
Displays triplets of AI-generated images side-by-side for manual review.
Helps you pick the best version from each set of 3 related images:
  stage1_generated.png      (initial generation)
  stage1.5_face_swapped.png (face-swapped version)  
  stage2_upscaled.png       (upscaled final version)

USAGE:
------
Activate virtual environment first:
  source venv/bin/activate

Run on normal_images directories (after quality filtering):
  python scripts/01_image_version_selector.py "normal_images"
  python scripts/01_image_version_selector.py "normal_images_2"

WORKFLOW POSITION:
------------------
1. Run scripts/01_image_version_selector.py → Pick best versions (YOU ARE HERE)
2. Run scripts/02_face_grouper.py → Group similar faces
3. Run scripts/03_character_sorter.py → Sort by body type
4. Run scripts/04_pair_comparator.py → Compare pairs
5. Run scripts/05_crop_tool.py → Final cropping

HOW IT WORKS:
-------------
• Files are sorted by name and grouped when they appear in correct stage order
• Shows 3 versions side-by-side for easy comparison
• When images are deleted, their corresponding YAML files are also deleted automatically
• Creates a log file with your selections for future reference

CONTROLS:
---------
  1/2/3 = Keep selected image, delete others
  4     = Delete all three images  
  s     = Skip this triplet (keep all)
  b     = Go back to previous triplet
  q     = Quit and save progress
  h     = Show help

EXAMPLE OUTPUT:
---------------
Shows three images in a grid layout with filenames and allows you to pick the best one.
Perfect for removing poor quality versions while keeping the best of each triplet.
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
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

# Optional HEIC
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    print("[*] HEIC/HEIF support enabled via pillow-heif.")
except Exception:
    pass

# Optional Trash
_SEND2TRASH_AVAILABLE = False
try:
    from send2trash import send2trash
    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False

# Memory monitoring
try:
    import psutil
    import gc
    _MEMORY_MONITORING_AVAILABLE = True
except Exception:
    _MEMORY_MONITORING_AVAILABLE = False
    print("[*] psutil not available - memory monitoring disabled. Install with: pip install psutil")


def human_err(msg: str):
    print(f"[!] {msg}", file=sys.stderr)


def info(msg: str):
    print(f"[*] {msg}")


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    if not _MEMORY_MONITORING_AVAILABLE:
        return 0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0


def check_memory_warning(warning_threshold_mb=3000, critical_threshold_mb=4000):
    """Check memory usage and return warning level: 0=OK, 1=Warning, 2=Critical"""
    if not _MEMORY_MONITORING_AVAILABLE:
        return 0
        
    current_mb = get_memory_usage_mb()
    
    if current_mb > critical_threshold_mb:
        return 2  # Critical
    elif current_mb > warning_threshold_mb:
        return 1  # Warning
    else:
        return 0  # OK


def format_memory_display():
    """Format memory usage for display."""
    if not _MEMORY_MONITORING_AVAILABLE:
        return ""
    
    current_mb = get_memory_usage_mb()
    if current_mb > 1024:
        return f" • Memory: {current_mb/1024:.1f}GB"
    else:
        return f" • Memory: {current_mb:.0f}MB"


STAGE_NAMES = ("stage1_generated", "stage1.5_face_swapped", "stage2_upscaled")


def detect_stage(name: str) -> str:
    low = name.lower()
    for s in STAGE_NAMES:
        if s in low:
            return s
    return ""


def scan_images(folder: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    exts_lower = {e.lower().lstrip('.') for e in exts}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip('.') in exts_lower:
            files.append(p)
    return sorted(files)


def extract_timestamp(filename):
    """Extract timestamp from filename (YYYYMMDD_HHMMSS format)."""
    import re
    match = re.search(r'(\d{8}_\d{6})', filename)
    return match.group(1) if match else None

def timestamp_to_minutes(timestamp_str):
    """Convert timestamp to minutes since midnight for easy comparison."""
    if not timestamp_str or len(timestamp_str) != 15:  # YYYYMMDD_HHMMSS
        return None
    try:
        time_part = timestamp_str.split('_')[1]  # HHMMSS
        hours = int(time_part[:2])
        minutes = int(time_part[2:4])
        seconds = int(time_part[4:6])
        return hours * 60 + minutes + seconds / 60.0
    except:
        return None

def get_date_from_timestamp(timestamp_str):
    """Extract date part from timestamp (YYYYMMDD)."""
    if not timestamp_str or len(timestamp_str) != 15:
        return None
    return timestamp_str.split('_')[0]

def find_triplets(files: List[Path]) -> List[Tuple[Path, Path, Path]]:
    """Find triplets based on timestamp proximity and stage sequence."""
    triplets = []
    
    # Group files by stage type with timestamps
    stage1_files = []
    stage15_files = []
    stage2_files = []
    
    for file in files:
        stage = detect_stage(file.name)
        timestamp = extract_timestamp(file.name)
        timestamp_minutes = timestamp_to_minutes(timestamp)
        
        if stage == "stage1_generated" and timestamp_minutes is not None:
            stage1_files.append((file, timestamp_minutes, timestamp))
        elif stage == "stage1.5_face_swapped" and timestamp_minutes is not None:
            stage15_files.append((file, timestamp_minutes, timestamp))
        elif stage == "stage2_upscaled" and timestamp_minutes is not None:
            stage2_files.append((file, timestamp_minutes, timestamp))
    
    # Sort by timestamp
    stage1_files.sort(key=lambda x: x[1])
    stage15_files.sort(key=lambda x: x[1])
    stage2_files.sort(key=lambda x: x[1])
    
    # Match triplets with tighter time windows
    MAX_GAP_STAGE1_TO_15 = 0.5    # 30 seconds max between stage1 and stage1.5
    MAX_GAP_STAGE15_TO_2 = 2      # 2 minutes max between stage1.5 and stage2
    
    for stage1_file, stage1_time, stage1_ts in stage1_files:
        stage1_date = get_date_from_timestamp(stage1_ts)
        if not stage1_date:
            continue
            
        # Find matching stage1.5 within time window AND same date
        stage15_match = None
        for stage15_file, stage15_time, stage15_ts in stage15_files:
            stage15_date = get_date_from_timestamp(stage15_ts)
            if (stage15_date == stage1_date and 
                stage1_time <= stage15_time <= stage1_time + MAX_GAP_STAGE1_TO_15):
                stage15_match = (stage15_file, stage15_time, stage15_ts)
                break
        
        if not stage15_match:
            continue
            
        stage15_file, stage15_time, stage15_ts = stage15_match
        
        # Find matching stage2 within time window AND same date
        stage2_match = None
        for stage2_file, stage2_time, stage2_ts in stage2_files:
            stage2_date = get_date_from_timestamp(stage2_ts)
            if (stage2_date == stage1_date and 
                stage15_time <= stage2_time <= stage15_time + MAX_GAP_STAGE15_TO_2):
                stage2_match = (stage2_file, stage2_time, stage2_ts)
                break
        
        if not stage2_match:
            continue
            
        stage2_file, stage2_time, stage2_ts = stage2_match
        
        # Found a valid triplet
        triplets.append((stage1_file, stage15_file, stage2_file))
        
        # Remove used files to prevent double-matching
        stage15_files = [f for f in stage15_files if f[2] != stage15_ts]
        stage2_files = [f for f in stage2_files if f[2] != stage2_ts]
    
    return triplets


def open_image(path: Path):
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        human_err(f"Failed to open {path.name}: {e}")
        return None


def show_triplet(img_paths: Tuple[Path, Path, Path], current: int = 0, total: int = 0) -> int:
    imgs = [open_image(p) for p in img_paths]
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, top=0.85, bottom=0.1)

    titles = [p.name for p in img_paths]
    for ax, img, title in zip(axes, imgs, titles):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=6)

    remaining = total - current if total > 0 else 0
    progress_text = f"({current}/{total} • {remaining} remaining)" if total > 0 else ""
    memory_text = format_memory_display()
    fig.suptitle(f"Press 1/2/3 to KEEP | 4=delete all | q=quit | h=help {progress_text}{memory_text}", fontsize=11)

    result = {"choice": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in ["1", "2", "3"]:
            result["choice"] = int(key) - 1
            plt.close(fig)
        elif key == "4":
            result["choice"] = -4  # New code: -4 means delete all
            plt.close(fig)


        elif key == "q":
            result["choice"] = -3
            plt.close(fig)
        elif key == "h":
            print("Keys: 1/2/3 keep | 4 delete all | q quit")

    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return result["choice"] if result["choice"] is not None else -1


def safe_delete(paths: List[Path], hard_delete: bool = False, tracker: FileTracker = None):
    """Delete PNG files and their corresponding YAML files."""
    # Collect all files to delete (PNGs + their YAMLs)
    all_files_to_delete = []
    
    for png_path in paths:
        # Add the PNG file
        all_files_to_delete.append(png_path)
        
        # Find corresponding YAML file
        base_name = png_path.stem
        yaml_path = png_path.parent / f"{base_name}.yaml"
        if yaml_path.exists():
            all_files_to_delete.append(yaml_path)
    
    # Delete all files
    deleted_files = []
    if hard_delete:
        for p in all_files_to_delete:
            try:
                p.unlink()
                info(f"Deleted: {p.name}")
                deleted_files.append(p.name)
            except Exception as e:
                human_err(f"Failed to delete {p}: {e}")
    else:
        if not _SEND2TRASH_AVAILABLE:
            raise RuntimeError(
                "send2trash is not installed. Install it with:\n    pip install send2trash\n"
                "Or run with --hard-delete to permanently delete files."
            )
        for p in all_files_to_delete:
            try:
                send2trash(str(p))
                info(f"Sent to Trash: {p.name}")
                deleted_files.append(p.name)
            except Exception as e:
                human_err(f"Failed to send to Trash {p}: {e}")
    
    # Log the deletion operation
    if tracker and deleted_files:
        operation_type = "delete" if hard_delete else "send_to_trash"
        source_dir = str(paths[0].parent.name) if paths else "unknown"
        tracker.log_operation(
            operation=operation_type,
            source_dir=source_dir,
            file_count=len(deleted_files),
            files=deleted_files,
            notes="Rejected images from triplet selection"
        )


def write_log(csv_path: Path, action: str, kept: Path, deleted: List[Path]):
    header_needed = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["action", "kept", "deleted1", "deleted2"])
        row = [action, str(kept) if kept else "", str(deleted[0]) if deleted else "", str(deleted[1]) if len(deleted) > 1 else ""]
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Preview 3 images; keep 1, Trash/delete the other 2.")
    parser.add_argument("folder", type=str, help="Folder containing images")
    parser.add_argument("--exts", type=str, default="png", help="Comma-separated list of extensions to include")
    parser.add_argument("--print-triplets", action="store_true", help="Print grouped triplets and exit (debugging aid)")
    parser.add_argument("--hard-delete", action="store_true", help="Permanently delete files instead of sending to Trash (USE WITH CARE)")
    args = parser.parse_args()

    # Initialize file tracker
    tracker = FileTracker("image_version_selector")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        human_err(f"Folder not found: {folder}")
        sys.exit(1)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = scan_images(folder, exts)
    if not files:
        human_err("No images found. Check --exts or folder path.")
        sys.exit(1)

    triplets = find_triplets(files)
    if not triplets:
        human_err("No triplets found with the current grouping. Try a different flag or no grouping.")
        sys.exit(1)

    if args.print_triplets:
        for idx, t in enumerate(triplets, 1):
            print(f"\nTriplet {idx}:")
            for p in t:
                print("  -", p.name)
        print(f"\nTotal triplets: {len(triplets)}")
        return

    log_path = folder / "triplet_culler_log.csv"
    index = 0
    made_changes = False  # Track if we've made any changes that need logging

    if not args.hard_delete and not _SEND2TRASH_AVAILABLE:
        human_err("send2trash not installed. Install it with: pip install send2trash")
        human_err("Or rerun with --hard-delete to permanently delete files (dangerous).")

    info(f"Found {len(triplets)} triplets. Starting at {folder}")

    while 0 <= index < len(triplets):
        t = triplets[index]
        remaining = len(triplets) - index
        
        # Check memory usage before processing
        memory_level = check_memory_warning()
        memory_display = format_memory_display()
        
        if memory_level == 2:  # Critical
            print(f"\n⚠️  CRITICAL MEMORY WARNING{memory_display}")
            print("💡 Memory usage is very high! Recommend quitting (q) and restarting the script.")
            print("💾 Your progress has been saved. You can resume from where you left off.")
        elif memory_level == 1:  # Warning
            print(f"\n⚠️  Memory Warning{memory_display}")
            print("💡 Memory getting high. Consider quitting (q) soon to restart fresh.")
        
        print(f"\n=== Triplet {index+1}/{len(triplets)} • {remaining} remaining{memory_display} ===")
        for i, p in enumerate(t, start=1):
            print(f"{i}. {p.name}")

        # Force garbage collection to free up memory
        if _MEMORY_MONITORING_AVAILABLE and memory_level > 0:
            gc.collect()

        choice = show_triplet(t, current=index+1, total=len(triplets))

        if choice in (0, 1, 2):
            kept = t[choice]
            deleted = [p for i, p in enumerate(t) if i != choice]
            try:
                # Create Reviewed directory if it doesn't exist
                reviewed_dir = folder.parent / "Reviewed"
                reviewed_dir.mkdir(exist_ok=True)
                
                # Move the chosen image and its YAML to Reviewed
                kept_yaml = kept.parent / f"{kept.stem}.yaml"
                new_kept_path = reviewed_dir / kept.name
                new_yaml_path = reviewed_dir / f"{kept.stem}.yaml"
                
                moved_files = []
                shutil.move(str(kept), str(new_kept_path))
                moved_files.append(kept.name)
                if kept_yaml.exists():
                    shutil.move(str(kept_yaml), str(new_yaml_path))
                    moved_files.append(kept_yaml.name)
                    info(f"Moved to Reviewed: {kept.name} + {kept_yaml.name}")
                else:
                    info(f"Moved to Reviewed: {kept.name} (no YAML found)")
                
                # Log the move operation
                tracker.log_operation(
                    operation="move",
                    source_dir=str(folder.name),
                    dest_dir="Reviewed",
                    file_count=len(moved_files),
                    files=moved_files,
                    notes=f"Selected image {choice + 1} from triplet"
                )
                
                # Delete the non-chosen images
                safe_delete(deleted, hard_delete=args.hard_delete, tracker=tracker)
                write_log(log_path, "keep_one", new_kept_path, deleted)
                made_changes = True
                index += 1  # Move to next triplet after successful keep/delete
            except RuntimeError as e:
                human_err(str(e))
                print("Aborting due to deletion method issue.")
                break
        elif choice == -4:  # Delete all three images
            try:
                safe_delete(list(t), hard_delete=args.hard_delete, tracker=tracker)
                write_log(log_path, "delete_all", None, list(t))
                made_changes = True
                index += 1  # Move to next triplet after successful delete
            except RuntimeError as e:
                human_err(str(e))
                print("Aborting due to deletion method issue.")
                break

        elif choice == -3:  # quit
            print("Quitting.")
            break
        else:
            index += 1

    if made_changes:
        print("\nDone. Log saved to:", log_path)
    else:
        print("\nDone. No changes were made, no log file created.")


if __name__ == "__main__":
    main()
