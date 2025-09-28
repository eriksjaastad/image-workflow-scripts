#!/usr/bin/env python3
"""
Desktop Image Selector + Crop Tool - Select and Crop in One Workflow
====================================================================
Combines triplet selection with immediate cropping for maximum efficiency.
Select which image to keep from each triplet, crop it, and move to the next.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Process directories containing image triplets:
  python scripts/01_desktop_image_selector_crop.py XXX_CONTENT/
  python scripts/01_desktop_image_selector_crop.py XXX_CONTENT/ --aspect-ratio 16:9

FEATURES:
---------
• Advanced triplet detection with precise timestamp matching
• Select which image to keep from each triplet (1, 2, or 3)
• Immediate cropping of selected image with interactive rectangles
• Large, easy-to-see images with generous crop handles
• Configurable aspect ratios (1:1, 16:9, 4:3, free)
• Integrated ActivityTimer and FileTracker logging
• Session progress tracking and navigation
• Safe deletion with send2trash (recoverable)

WORKFLOW:
---------
1. View 3 images from triplet side-by-side
2. Click or press [1][2][3] to select which image to keep
3. Crop the selected image by dragging rectangle
4. Press [Enter] to crop selected image, delete others, next triplet
5. Or press [R] to reset row (all back to delete), or [←] to go back

WORKFLOW POSITION:
------------------
Step 1: Image Selection + Cropping → THIS SCRIPT (scripts/01_desktop_image_selector_crop.py)
Step 2: Character Sorting → scripts/03_web_character_sorter.py
Step 3: Basic Review → scripts/05_multi_directory_viewer.py

FILE HANDLING:
--------------
• Selected & cropped images: Moved to selected/ directory
• Unselected images: Deleted (moved to trash)
• YAML files: Moved with their corresponding images

CONTROLS:
---------
Selection:  [1] [2] [3] Select Image (auto-shows crop rectangle)
Reset:      [R] Reset Row (all back to delete, clear crops)
Navigation: [←] Previous Triplet  [Enter] Submit & Next Triplet
Global:     [Space] Aspect Ratio Toggle  [Q] Quit  [H] Help

NOTE: Moving any crop handle automatically selects that image!
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

try:
    matplotlib.use('Qt5Agg', force=True)
    print("[*] Using Qt5Agg backend (PyQt5) - full interactivity available")
    backend_interactive = True
except Exception as e:
    print(f"[!] Qt5Agg backend failed: {e}")
    matplotlib.use('Agg', force=True)
    print("[!] Using Agg backend - limited interactivity")
    backend_interactive = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image
import numpy as np
import os
import shutil
from send2trash import send2trash

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from scripts.file_tracker import FileTracker
from utils.activity_timer import ActivityTimer

# Triplet detection constants and functions (from web image selector)
STAGE_NAMES = [
    "stage1_generated",
    "stage1.5_face_swapped", 
    "stage2_upscaled",
    "stage2.5_inpainted",
    "stage3_final"
]


def detect_stage(name: str) -> str:
    """Detect stage from filename."""
    low = name.lower()
    for stage in STAGE_NAMES:
        if stage in low:
            return stage
    return ""


def get_stage_number(stage: str) -> float:
    """Convert stage name to numeric value for sorting."""
    stage_map = {
        "stage1_generated": 1.0,
        "stage1.5_face_swapped": 1.5,
        "stage2_upscaled": 2.0,
        "stage2.5_inpainted": 2.5,
        "stage3_final": 3.0
    }
    return stage_map.get(stage, 0.0)


def extract_timestamp(filename: str) -> Optional[str]:
    """Extract timestamp from filename (YYYYMMDD_HHMMSS format)."""
    match = re.search(r"(\d{8}_\d{6})", filename)
    return match.group(1) if match else None


def scan_images(folder: Path, exts: List[str], recursive: bool = True) -> List[Path]:
    """Scan directory for image files with specified extensions."""
    results = []
    
    def scan_dir(directory: Path):
        try:
            for entry in directory.iterdir():
                if entry.is_file() and any(entry.name.lower().endswith(f'.{ext}') for ext in exts):
                    results.append(entry)
                elif entry.is_dir() and recursive:
                    scan_dir(entry)
        except PermissionError:
            pass
    
    scan_dir(folder)
    return results


def find_flexible_groups(files: List[Path]) -> List[Tuple[Path, ...]]:
    """Find triplet groups by detecting stage number decreases."""
    groups: List[Tuple[Path, ...]] = []
    
    # Sort by timestamp first, then stage order, then name
    def sort_key(path):
        timestamp = extract_timestamp(path.name) or "99999999_999999"
        stage = detect_stage(path.name)
        stage_num = get_stage_number(stage)
        return (timestamp, stage_num, path.name)
    
    sorted_files = sorted(files, key=sort_key)
    
    if not sorted_files:
        return groups
    
    current_group = []
    prev_stage_num = None
    
    for file in sorted_files:
        stage = detect_stage(file.name)
        if not stage:
            continue  # Skip files that don't match expected patterns
            
        stage_num = get_stage_number(stage)
        
        # If this is the first file or stage equal/decreased, start a new group
        if prev_stage_num is None or (stage_num <= prev_stage_num and current_group):
            # Finish current group if it exists and has enough files
            if current_group and len(current_group) >= 2:
                groups.append(tuple(current_group))
            current_group = [file]  # Start new group with current file
        else:
            # Continue current group
            current_group.append(file)
        
        prev_stage_num = stage_num
    
    # Add the final group if it has 2+ images
    if len(current_group) >= 2:
        groups.append(tuple(current_group))
    
    # Filter out orphan groups (no ascending stage progression)
    filtered_groups = []
    for group in groups:
        # Check if group has ascending stage progression
        stage_nums = [get_stage_number(detect_stage(f.name)) for f in group]
        has_progression = any(stage_nums[i] < stage_nums[i+1] for i in range(len(stage_nums)-1))
        
        if has_progression:
            filtered_groups.append(group)
    
    return filtered_groups


@dataclass
class TripletRecord:
    """Represents a triplet of images for processing."""
    index: int
    paths: Tuple[Path, ...]  # Can be 2 or 3 paths
    relative_dir: str
    
    @property
    def display_name(self) -> str:
        """Get display name for this triplet."""
        group_type = "Triplet" if len(self.paths) == 3 else "Pair"
        return f"{group_type} {self.index + 1}"


class TripletProgressTracker:
    """Manages progress tracking for triplet processing."""
    
    def __init__(self, triplets: List[TripletRecord]):
        self.triplets = triplets
        self.current_triplet_index = 0
        self.selected_count = 0
        self.deleted_count = 0
        self.processed_count = 0
    
    def get_current_triplet(self) -> Optional[TripletRecord]:
        """Get current triplet being processed."""
        if self.current_triplet_index < len(self.triplets):
            return self.triplets[self.current_triplet_index]
        return None
    
    def advance_triplet(self):
        """Move to next triplet."""
        self.current_triplet_index += 1
        self.processed_count += 1
    
    def previous_triplet(self):
        """Move to previous triplet."""
        if self.current_triplet_index > 0:
            self.current_triplet_index -= 1
            self.processed_count -= 1
    
    def has_more_triplets(self) -> bool:
        """Check if there are more triplets to process."""
        return self.current_triplet_index < len(self.triplets)
    
    def can_go_back(self) -> bool:
        """Check if we can go back to previous triplet."""
        return self.current_triplet_index > 0
    
    def get_progress_info(self) -> Dict:
        """Get current progress information."""
        remaining = len(self.triplets) - self.current_triplet_index
        return {
            'current_triplet': self.current_triplet_index + 1,
            'total_triplets': len(self.triplets),
            'remaining': remaining,
            'selected': self.selected_count,
            'deleted': self.deleted_count,
            'processed': self.processed_count
        }


class MultiDirectoryProgressTracker:
    """Manages progress tracking across multiple directories with session persistence."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.progress_dir = Path("scripts/crop_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # Create progress file name based on base directory
        safe_name = str(base_directory).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
        
        self.directories = []
        self.current_directory_index = 0
        self.current_file_index = 0
        self.session_data = {}
        
        self.discover_directories()
        self.load_progress()
    
    def discover_directories(self):
        """Discover all subdirectories containing PNG files, sorted alphabetically."""
        subdirs = []
        
        for item in self.base_directory.iterdir():
            if item.is_dir():
                # Check if directory contains PNG files
                png_files = list(item.glob("*.png"))
                if png_files:
                    subdirs.append({
                        'path': item,
                        'name': item.name,
                        'file_count': len(png_files)
                    })
        
        # Sort alphabetically by directory name
        subdirs.sort(key=lambda x: x['name'].lower())
        self.directories = subdirs
        
        print(f"[*] Discovered {len(self.directories)} directories with images:")
        for i, dir_info in enumerate(self.directories):
            print(f"    {i+1}. {dir_info['name']} ({dir_info['file_count']} images)")
    
    def load_progress(self):
        """Load existing progress from file."""
        if not self.progress_file.exists():
            self.initialize_progress()
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                self.session_data = json.load(f)
            
            # Validate and restore state
            if 'current_directory_index' in self.session_data:
                self.current_directory_index = self.session_data['current_directory_index']
            if 'current_file_index' in self.session_data:
                self.current_file_index = self.session_data['current_file_index']
            
            # Ensure indices are valid
            if self.current_directory_index >= len(self.directories):
                self.current_directory_index = 0
                self.current_file_index = 0
            
            print(f"[*] Resumed session from: {self.progress_file}")
            self.print_resume_info()
            
        except Exception as e:
            print(f"[!] Error loading progress: {e}")
            self.initialize_progress()
    
    def initialize_progress(self):
        """Initialize new progress tracking session."""
        self.session_data = {
            'base_directory': str(self.base_directory),
            'session_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_directory_index': 0,
            'current_file_index': 0,
            'directories': {
                dir_info['name']: {
                    'status': 'pending',
                    'files_processed': 0,
                    'total_files': dir_info['file_count']
                } for dir_info in self.directories
            }
        }
        self.save_progress()
    
    def save_progress(self):
        """Save current progress to file."""
        self.session_data['current_directory_index'] = self.current_directory_index
        self.session_data['current_file_index'] = self.current_file_index
        self.session_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"[!] Error saving progress: {e}")
    
    def get_current_directory(self) -> Optional[Dict]:
        """Get current directory info."""
        if self.current_directory_index < len(self.directories):
            return self.directories[self.current_directory_index]
        return None
    
    def get_current_files(self) -> List[Path]:
        """Get ALL PNG files from current directory (not sliced)."""
        current_dir = self.get_current_directory()
        if not current_dir:
            return []
        
        # Return ALL files - let load_batch() handle the indexing
        all_files = sorted(current_dir['path'].glob("*.png"))
        return all_files
    
    def advance_file(self, count: int = 1):
        """Advance file index by count."""
        self.current_file_index += count
        self.save_progress()
    
    def advance_directory(self):
        """Move to next directory."""
        if self.current_directory_index < len(self.directories):
            # Mark current directory as completed
            current_dir = self.get_current_directory()
            if current_dir:
                self.session_data['directories'][current_dir['name']]['status'] = 'completed'
        
        self.current_directory_index += 1
        self.current_file_index = 0
        
        # Mark new directory as in progress
        current_dir = self.get_current_directory()
        if current_dir:
            self.session_data['directories'][current_dir['name']]['status'] = 'in_progress'
        
        self.save_progress()
    
    def has_more_directories(self) -> bool:
        """Check if there are more directories to process."""
        return self.current_directory_index < len(self.directories)
    
    def get_progress_info(self) -> Dict:
        """Get current progress information for display."""
        current_dir = self.get_current_directory()
        if not current_dir:
            return {
                'current_directory': None,
                'files_remaining': 0,
                'directories_remaining': 0,
                'total_directories': len(self.directories)
            }
        
        files_remaining = current_dir['file_count'] - self.current_file_index
        directories_remaining = len(self.directories) - self.current_directory_index - 1
        
        return {
            'current_directory': current_dir['name'],
            'files_remaining': files_remaining,
            'directories_remaining': directories_remaining,
            'total_directories': len(self.directories),
            'directory_index': self.current_directory_index + 1
        }
    
    def print_resume_info(self):
        """Print information about resumed session."""
        progress = self.get_progress_info()
        if progress['current_directory']:
            print(f"[*] Resuming in directory: {progress['current_directory']}")
            print(f"[*] Files remaining in this directory: {progress['files_remaining']}")
            print(f"[*] Directories remaining after this: {progress['directories_remaining']}")
        
    def cleanup_completed_session(self):
        """Remove progress file when session is complete."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                print(f"[*] Session complete - removed progress file: {self.progress_file}")
        except Exception as e:
            print(f"[!] Error cleaning up progress file: {e}")


class DesktopImageSelectorCropTool:
    """Enhanced batch crop tool with multi-directory support."""
    
    def __init__(self, directory, aspect_ratio=None, exts=["png"]):
        self.base_directory = Path(directory)
        self.aspect_ratio = self._parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
        self.aspect_ratio_locked = True
        self.tracker = FileTracker("desktop_image_selector_crop")
        
        # Initialize activity timer
        self.activity_timer = ActivityTimer("01_desktop_image_selector_crop")
        self.activity_timer.start_session()
        
        # Scan for images and detect triplets
        print(f"[*] Scanning {self.base_directory} for image triplets...")
        files = scan_images(self.base_directory, exts, recursive=True)
        if not files:
            raise ValueError(f"No images found in {self.base_directory}")
        
        # Find triplet groups
        group_paths = find_flexible_groups(files)
        if not group_paths:
            raise ValueError("No triplet groups found. Check filenames and timestamps.")
        
        # Create triplet records
        self.triplets = []
        for idx, group in enumerate(group_paths):
            first_parent = group[0].parent
            try:
                relative = str(first_parent.relative_to(self.base_directory))
            except ValueError:
                relative = str(first_parent)
            
            self.triplets.append(
                TripletRecord(
                    index=idx,
                    paths=group,
                    relative_dir=relative if relative != "." else ""
                )
            )
        
        # Initialize progress tracker
        self.progress_tracker = TripletProgressTracker(self.triplets)
        
        print(f"[*] Found {len(self.triplets)} triplet groups")
        triplet_count = sum(1 for t in self.triplets if len(t.paths) == 3)
        pair_count = sum(1 for t in self.triplets if len(t.paths) == 2)
        print(f"[*] {triplet_count} triplets, {pair_count} pairs")
        
        # Initialize current triplet state
        self.current_triplet = None
        self.current_images = []  # Will hold 3 different images from current triplet
        self.image_states = []   # State for each image: {'status': 'delete'|'selected', 'crop_coords': None}
        
        # No old directory logic needed - triplet-based processing only
        
        # State for current batch of 3 images
        self.current_images = []
        self.image_states = []  # List of dicts with crop coords, action, etc.
        self.selectors = []
        self.axes = []
        self.has_pending_changes = False
        self.quit_confirmed = False
        self.previous_batch_confirmed = False
        
        # No longer using a separate cropped directory - files stay in place
        
        # This tool always works with triplets, no single directory mode
        self.single_directory_mode = False
        
        # Setup matplotlib
        self.fig = None
        self.setup_display()
    
    def _is_single_directory_mode(self) -> bool:
        """Determine if we're processing a single directory or multiple directories."""
        # Check if the base directory contains PNG files directly
        png_files = list(self.base_directory.glob("*.png"))
        if png_files:
            return True
        
        # Check if it has subdirectories with PNG files
        has_subdirs_with_images = any(
            list(item.glob("*.png")) for item in self.base_directory.iterdir() 
            if item.is_dir()
        )
        
        return not has_subdirs_with_images
    
    def _parse_aspect_ratio(self, ratio_str: str) -> float:
        """Parse aspect ratio string like '16:9' into float."""
        try:
            if ':' in ratio_str:
                width, height = map(float, ratio_str.split(':'))
                return width / height
            else:
                return float(ratio_str)
        except:
            print(f"⚠️  Invalid aspect ratio: {ratio_str}. Using original image ratio.")
            return None
    
    def setup_display(self):
        """Setup the matplotlib figure with 3 subplots optimized for screen space"""
        if self.fig:
            plt.close(self.fig)
            
        # Get screen dimensions and calculate optimal figure size
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except ImportError:
            screen_width = 1920
            screen_height = 1080
        
        # Calculate optimal figure size
        max_width = screen_width * 0.9 / 100
        max_height = (screen_height * 0.85) / 100
        
        # Create figure with optimized dimensions
        self.fig, self.axes = plt.subplots(1, 3, figsize=(max_width, max_height))
        self.fig.suptitle("", fontsize=14, y=0.98)
        
        # Hide the matplotlib toolbar completely
        try:
            self.fig.canvas.toolbar_visible = False
            if hasattr(self.fig.canvas, 'toolbar'):
                self.fig.canvas.toolbar = None
        except:
            pass
        
        # Minimize spacing between subplots
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04, wspace=0.03)
        
        # Connect keyboard and mouse events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_image_click)
        
        # Initialize selectors list
        self.selectors = [None, None, None]
    
    def load_current_triplet(self):
        """Load the current triplet for processing"""
        # Get current triplet from progress tracker
        self.current_triplet = self.progress_tracker.get_current_triplet()
        if not self.current_triplet:
            print("🎉 All triplets processed!")
            self.activity_timer.end_session()
            plt.close('all')
            sys.exit(0)
        
        # Get the image files from the triplet
        triplet_files = list(self.current_triplet.paths)
        
        # Debug: Show which triplet is being loaded
        progress = self.progress_tracker.get_progress_info()
        print(f"[DEBUG] Loading {self.current_triplet.display_name} ({progress['current_triplet']}/{progress['total_triplets']}):")
        for i, file_path in enumerate(triplet_files):
            stage = detect_stage(file_path.name) or f"image {i+1}"
            print(f"  Image {i+1}: {file_path.name} ({stage})")
        
        self.current_images = []
        self.image_states = []
        
        # Reset flags for new batch
        self.has_pending_changes = False
        self.quit_confirmed = False
        self.previous_batch_confirmed = False
        
        # Clear previous selectors
        for selector in self.selectors:
            if selector:
                selector.set_active(False)
        self.selectors = [None, None, None]
        
        # Load and display each image from the triplet
        for i, png_path in enumerate(triplet_files):
            try:
                # Load image
                img = Image.open(png_path)
                img_array = np.array(img)
                
                self.current_images.append({
                    'path': png_path,
                    'image': img_array,
                    'original_size': img.size
                })
                
                # Calculate this image's aspect ratio
                img_width, img_height = img.size
                image_aspect_ratio = img_width / img_height
                
                # Initialize state for this image (triplet selection logic)
                self.image_states.append({
                    'status': 'delete',  # 'delete' (default) or 'selected'
                    'crop_coords': None,
                    'has_selection': False,
                    'image_aspect_ratio': image_aspect_ratio
                })
                
                # Display image
                ax = self.axes[i]
                ax.clear()
                ax.imshow(img_array, aspect='equal')
                
                # Set title with stage information
                stage = detect_stage(png_path.name) or f"Image {i+1}"
                ax.set_title(f"{stage.replace('_', ' ').title()}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Set border color based on selection status
                border_color = 'red' if self.image_states[i]['status'] == 'delete' else 'green'
                border_width = 3
                
                # Configure axis spines for visual feedback
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)
                ax.margins(0)
                
                # Create RectangleSelector
                selector = RectangleSelector(
                    ax, 
                    lambda eclick, erelease, idx=i: self.on_crop_select(eclick, erelease, idx),
                    useblit=True,
                    button=[1],
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='none', edgecolor='red', linewidth=2),
                    drag_from_anywhere=False,
                    use_data_coordinates=False,
                    grab_range=120,
                    handle_props=dict(markersize=48, markerfacecolor='none', markeredgecolor='red', markeredgewidth=3)
                )
                
                self.selectors[i] = selector
                
                # Set initial crop selection to full image
                selector.extents = (0, img_width, 0, img_height)
                
                # Initialize the crop coordinates
                self.image_states[i]['crop_coords'] = (0, 0, img_width, img_height)
                self.image_states[i]['has_selection'] = True
                
                # Update title
                aspect_str = f" [{image_aspect_ratio:.2f}:1]" if self.aspect_ratio_locked else ""
                self.axes[i].set_title(f"Image {i+1}: {img_width}×{img_height}{aspect_str} [Full Image]", 
                                     fontsize=10, color='green')
                
            except Exception as e:
                print(f"Error loading {png_path}: {e}")
                # Add placeholder for failed image to maintain index consistency
                self.current_images.append({
                    'path': png_path,
                    'image': None,
                    'original_size': (0, 0),
                    'load_error': str(e)
                })
                
                # Initialize state for failed image
                self.image_states.append({
                    'action': 'skip',  # Auto-skip failed images
                    'crop_coords': None,
                    'has_selection': False,
                    'image_aspect_ratio': 1.0,
                    'load_failed': True
                })
                
                # Show error in the subplot
                ax = self.axes[i]
                ax.clear()
                ax.text(0.5, 0.5, f'LOAD ERROR\n{png_path.name}\n{str(e)}', 
                       ha='center', va='center', fontsize=10, color='red',
                       transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                ax.set_title(f"Image {i+1}: LOAD ERROR", fontsize=10, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(len(triplet_files), 3):
            self.axes[i].clear()
            self.axes[i].set_visible(False)
        
        # Update title with progress information
        self.update_title()
        self.update_control_labels()
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.draw()
    
    def update_title(self):
        """Update the title with current progress information."""
        if self.single_directory_mode:
            # Single directory mode - original behavior
            remaining_images = len(self.png_files) - (self.current_batch * 3)
            lock_str = "🔒" if self.aspect_ratio_locked else "🔓"
            aspect_info = f" • [{lock_str} Space] Aspect Ratio" if self.aspect_ratio else ""
            # Debug: Show first filename to verify position
            first_file = self.png_files[self.progress_tracker.current_file_index] if self.progress_tracker.current_file_index < len(self.png_files) else None
            filename_debug = f" • DEBUG: {first_file.name[:25]}..." if first_file else " • DEBUG: N/A"
            title = f"Batch {self.current_batch + 1}/{self.total_batches} • {remaining_images} images remaining • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}{filename_debug}"
        else:
            # Triplet mode - enhanced progress display
            progress = self.progress_tracker.get_progress_info()
            lock_str = "🔒" if self.aspect_ratio_locked else "🔓"
            aspect_info = f" • [{lock_str} Space] Aspect" if self.aspect_ratio else ""
            
            # Count current selections
            selected_count = sum(1 for state in self.image_states if state['status'] == 'selected')
            delete_count = len(self.image_states) - selected_count
            
            triplet_info = f"📸 Triplet {progress['current_triplet']}/{progress['total_triplets']} • {self.current_triplet.display_name if hasattr(self, 'current_triplet') else 'Loading...'}"
            selection_info = f"Selected: {selected_count} • Delete: {delete_count}"
            
            title = f"{triplet_info} • {selection_info} • [1,2,3] Select • [R] Reset • [Enter] Submit • [Q] Quit{aspect_info}"
        
        self.fig.suptitle(title, fontsize=12, y=0.98)
    
    def select_image(self, image_idx: int):
        """Select an image (change status from delete to selected)"""
        if image_idx >= len(self.image_states):
            return
        
        # Change status to selected
        self.image_states[image_idx]['status'] = 'selected'
        
        # Update visual feedback
        self.update_visual_feedback()
        self.update_control_labels()
        self.update_title()
        
        # Mark as having pending changes
        self.has_pending_changes = True
        
        print(f"[*] Image {image_idx + 1} selected for cropping")
    
    def on_image_click(self, event):
        """Handle clicking on an image to select it"""
        if event.inaxes is None:
            return
        
        # Find which image was clicked
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax and i < len(self.current_images):
                self.select_image(i)
                break
    
    def update_visual_feedback(self):
        """Update visual feedback for all images based on selection status"""
        for i, ax in enumerate(self.axes):
            if i < len(self.image_states):
                status = self.image_states[i]['status']
                border_color = 'green' if status == 'selected' else 'red'
                border_width = 3
                
                # Update border color
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)
        
        # Refresh the display
        self.fig.canvas.draw()
    
    def reset_entire_row(self):
        """Reset entire row to default state (all delete, clear all crops)"""
        for i in range(len(self.image_states)):
            self.image_states[i]['status'] = 'delete'
            self.image_states[i]['crop_coords'] = None
            self.image_states[i]['has_selection'] = False
            
            # Clear crop rectangle
            if i < len(self.selectors) and self.selectors[i]:
                self.selectors[i].set_visible(False)
        
        # Update display
        self.update_visual_feedback()
        self.update_control_labels()
        self.update_title()
        
        self.has_pending_changes = False
        print("[*] Entire row reset to default (all DELETE)")
    
    def previous_triplet(self):
        """Move to previous triplet with confirmation if changes pending"""
        if self.has_pending_changes:
            print("[!] You have unsaved changes. Press [Enter] to submit or [R] to reset before navigating.")
            return
        
        if not self.progress_tracker.can_go_back():
            print("Already at first triplet")
            return
        
        self.progress_tracker.previous_triplet()
        self.load_current_triplet()
        print("[*] Moved to previous triplet")
    
    def update_control_labels(self):
        """Update the control labels below each image"""
        for i, ax in enumerate(self.axes):
            if i < len(self.current_images):
                status = self.image_states[i]['status']
                if status == 'selected':
                    control_text = f"[{i+1}] ✅ SELECTED  [X/C/V] Reset"
                else:
                    control_text = f"[{i+1}] ❌ DELETE  [X/C/V] Reset"
                ax.set_xlabel(control_text, fontsize=10)
    
    def on_crop_select(self, eclick, erelease, image_idx):
        """Handle crop rectangle selection for a specific image"""
        self.activity_timer.mark_activity()
        if image_idx >= len(self.current_images):
            return
            
        # Skip crop selection for failed images
        if self.current_images[image_idx].get('image') is None:
            print(f"Cannot crop image {image_idx + 1}: Load failed")
            return
        
        # Auto-select this image when crop handle is moved
        self.select_image(image_idx)
            
        # Get crop coordinates
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Apply aspect ratio constraint if locked
        active_aspect_ratio = self.aspect_ratio if self.aspect_ratio else self.image_states[image_idx]['image_aspect_ratio']
        
        if self.aspect_ratio_locked and active_aspect_ratio:
            sel_width = x2 - x1
            sel_height = y2 - y1
            
            if sel_width > 0 and sel_height > 0:
                target_height_from_width = sel_width / active_aspect_ratio
                target_width_from_height = sel_height * active_aspect_ratio
                
                if target_height_from_width <= sel_height:
                    new_height = target_height_from_width
                    height_diff = sel_height - new_height
                    y1 += height_diff / 2
                    y2 = y1 + new_height
                else:
                    new_width = target_width_from_height
                    width_diff = sel_width - new_width
                    x1 += width_diff / 2
                    x2 = x1 + new_width
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                self.selectors[image_idx].extents = (x1, x2, y1, y2)
                plt.draw()
        
        # Store crop coordinates
        self.image_states[image_idx]['crop_coords'] = (x1, y1, x2, y2)
        self.image_states[image_idx]['has_selection'] = True
        self.image_states[image_idx]['action'] = None
        self.has_pending_changes = True
        
        # Update title
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_aspect = crop_width/crop_height if crop_height > 0 else 0
        aspect_str = f" ({crop_aspect:.2f}:1)" if crop_height > 0 else ""
        
        if self.aspect_ratio_locked:
            ratio_source = f"Global {self.aspect_ratio:.2f}:1" if self.aspect_ratio else f"Original {active_aspect_ratio:.2f}:1"
            lock_str = f" [🔒 {ratio_source}]"
        else:
            lock_str = " [🔓 Free]"
        
        self.axes[image_idx].set_title(f"Image {image_idx + 1}: {crop_width}×{crop_height}{aspect_str}{lock_str}", 
                                       fontsize=10, color='green')
        plt.draw()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        self.activity_timer.mark_activity()
        key = event.key.lower()
        
        # Arrow key detection confirmed working
        
        # Global controls
        if key == 'q':
            self.quit()
            return
        elif key == 'escape':
            self.quit()
            return
        elif key == 'enter':
            self.submit_batch()
            return
        elif key == ' ':
            self.aspect_ratio_locked = not self.aspect_ratio_locked
            lock_str = "🔒 locked" if self.aspect_ratio_locked else "🔓 unlocked"
            print(f"Aspect ratio {lock_str}")
            return
        elif key == 'n' and not self.single_directory_mode:
            self.next_directory()
            return
        elif key == 'p' and not self.single_directory_mode:
            self.previous_directory()
            return
        elif key in ['left', 'arrow_left', 'leftarrow']:
            self.previous_triplet()
            return
        elif key == 'r':
            self.reset_entire_row()
            return
            
        # Image selection and reset controls
        if key in ['1', '2', '3']:
            image_idx = int(key) - 1
            if image_idx < len(self.current_images):
                self.select_image(image_idx)
        elif key in ['x', 'c', 'v']:
            # Reset individual crop
            reset_map = {'x': 0, 'c': 1, 'v': 2}
            image_idx = reset_map[key]
            if image_idx < len(self.current_images):
                self.reset_image_crop(image_idx)
    
    def next_directory(self):
        """Move to next directory (multi-directory mode only)."""
        if self.single_directory_mode:
            return
        
        print("Moving to next directory...")
        self.progress_tracker.advance_directory()
        
        if self.progress_tracker.has_more_directories():
            self.png_files = self.progress_tracker.get_current_files()
            # current_batch will be set by load_batch()
            self.total_batches = (len(self.png_files) + 2) // 3
            self.load_batch()
        else:
            self.show_completion()
    
    def previous_directory(self):
        """Move to previous directory (multi-directory mode only)."""
        if self.single_directory_mode or self.progress_tracker.current_directory_index == 0:
            print("Already at first directory")
            return
        
        print("Moving to previous directory...")
        self.progress_tracker.current_directory_index -= 1
        self.progress_tracker.current_file_index = 0
        self.progress_tracker.save_progress()
        
        self.png_files = self.progress_tracker.get_current_files()
        # When going to previous directory, start at beginning (batch 0)
        self.current_batch = 0
        self.total_batches = (len(self.png_files) + 2) // 3
        self.load_batch()
    
    def previous_batch(self):
        """Go back to the previous batch within the current directory"""
        # Check if we can go back (need at least 3 files back)
        if self.progress_tracker.current_file_index < 3:
            print("Already at first batch")
            return
            
        # Check for uncommitted changes
        if self.has_pending_changes:
            print("\n⚠️  WARNING: You have uncommitted changes in current batch!")
            print("   - Press [Enter] to commit changes first, or")
            print("   - Press [←] again to go back anyway (changes will be lost)")
            if not hasattr(self, 'previous_batch_confirmed'):
                self.previous_batch_confirmed = False
            if not self.previous_batch_confirmed:
                self.previous_batch_confirmed = True
                return
            else:
                self.previous_batch_confirmed = False
        
        # Move back one batch (3 files)
        old_file_index = self.progress_tracker.current_file_index
        self.progress_tracker.current_file_index = max(0, old_file_index - 3)
        self.progress_tracker.save_progress()
        
        print(f"Going back from file {old_file_index} to {self.progress_tracker.current_file_index}")
        
        # Load the previous batch
        self.load_batch()
        print(f"Moved back to batch {self.current_batch + 1}/{self.total_batches}")
    
    def reset_image_crop(self, image_idx):
        """Reset the crop selection for a specific image back to full image"""
        if image_idx >= len(self.current_images):
            return
            
        image_info = self.current_images[image_idx]
        
        # Skip reset for failed images
        if image_info.get('image') is None:
            print(f"Cannot reset image {image_idx + 1}: Load failed")
            return
            
        img = Image.open(image_info['path'])
        img_width, img_height = img.size
        
        if self.selectors[image_idx]:
            self.selectors[image_idx].extents = (0, img_width, 0, img_height)
            
        self.image_states[image_idx]['crop_coords'] = (0, 0, img_width, img_height)
        self.image_states[image_idx]['has_selection'] = True
        self.image_states[image_idx]['action'] = None
        
        image_aspect_ratio = self.image_states[image_idx]['image_aspect_ratio']
        aspect_str = f" [{image_aspect_ratio:.2f}:1]" if self.aspect_ratio_locked else ""
        self.axes[image_idx].set_title(f"Image {image_idx + 1}: {img_width}×{img_height}{aspect_str} [RESET TO FULL]", 
                                       fontsize=10, color='green')
        plt.draw()
    
    def set_image_action(self, image_idx, action):
        """Set action (skip/delete) for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        self.image_states[image_idx]['action'] = action
        self.has_pending_changes = True
        
        color = 'orange' if action == 'skip' else 'red'
        action_text = action.upper()
        
        self.axes[image_idx].set_title(f"Image {image_idx + 1} [{action_text}]", 
                                       fontsize=12, color=color)
        plt.draw()
    
    def submit_batch(self):
        """Process current triplet - crop selected image, delete others"""
        if not self.current_images:
            return
        
        progress = self.progress_tracker.get_progress_info()
        self.activity_timer.mark_batch(f"Triplet {progress['current_triplet']}")
        self.activity_timer.mark_activity()
        
        print(f"\nProcessing {self.current_triplet.display_name}...")
        
        selected_images = []
        deleted_images = []
        
        # Process each image based on selection status
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            png_path = image_info['path']
            yaml_path = png_path.with_suffix('.yaml')
            
            try:
                if state['status'] == 'selected':
                    # Crop and save selected image
                    if state['has_selection'] and state['crop_coords']:
                        self.crop_and_save(image_info, state['crop_coords'])
                        selected_images.append(png_path.name)
                        self.tracker.log_action("crop", str(png_path))
                    else:
                        # Selected but no crop - just keep the original
                        selected_images.append(png_path.name)
                        self.tracker.log_action("keep", str(png_path))
                else:
                    # Delete unselected images
                    self.safe_delete(png_path, yaml_path)
                    deleted_images.append(png_path.name)
                    self.tracker.log_action("delete", str(png_path))
                    
            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
        
        # Update progress tracker stats
        self.progress_tracker.selected_count += len(selected_images)
        self.progress_tracker.deleted_count += len(deleted_images)
        
        print(f"Selected: {len(selected_images)}, Deleted: {len(deleted_images)}")
        if selected_images:
            print(f"  Kept: {', '.join(selected_images)}")
        if deleted_images:
            print(f"  Deleted: {', '.join(deleted_images)}")
        
        # Clear pending changes flag after successful submission
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # Advance to next triplet
        self.progress_tracker.advance_triplet()
        
        self.activity_timer.end_batch(f"Completed triplet processing")
        
        # Move to next triplet or finish
        next_triplet = self.progress_tracker.get_current_triplet()
        if next_triplet:
            self.load_current_triplet()
        else:
            self.show_completion()
    
    def crop_and_save(self, image_info, crop_coords):
        """Crop image and save over the original file in place"""
        png_path = image_info['path']
        yaml_path = png_path.with_suffix('.yaml')
        
        x1, y1, x2, y2 = crop_coords
        
        # Load and crop image
        img = Image.open(png_path)
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Save over the original file (in place)
        cropped_img.save(png_path)
        
        # YAML file stays unchanged in the same directory
        
        # Log the operation
        self.tracker.log_operation(
            "crop", str(png_path.parent), str(png_path.parent), 1,
            f"Cropped in place to ({x1},{y1},{x2},{y2})", 
            [png_path.name]
        )
        
        self.activity_timer.log_operation("crop", file_count=1)
        
        print(f"Cropped and saved in place: {png_path.name}")
    
    def move_to_cropped(self, png_path, yaml_path, reason):
        """Mark files as processed (skipped) but leave them in original directory"""
        # Files stay in place - no actual moving
        
        file_count = 2 if yaml_path.exists() else 1
        files = [png_path.name, yaml_path.name] if yaml_path.exists() else [png_path.name]
        
        self.tracker.log_operation(
            "skip", str(png_path.parent), str(png_path.parent), file_count,
            f"Image {reason} (left in place)", files
        )
        
        self.activity_timer.log_operation("skip", file_count=1)
        print(f"Skipped (left in place): {png_path.name} ({reason})")
    
    def safe_delete(self, png_path, yaml_path):
        """Safely delete image files"""
        files_deleted = []
        
        send2trash(str(png_path))
        files_deleted.append(png_path.name)
        
        if yaml_path.exists():
            send2trash(str(yaml_path))
            files_deleted.append(yaml_path.name)
            
        self.tracker.log_operation(
            "delete", str(png_path.parent), "trash", len(files_deleted),
            "Image deleted", files_deleted
        )
        
        self.activity_timer.log_operation("delete", file_count=1)
        print(f"Deleted: {png_path.name}")
    
    def has_next_batch(self):
        """Check if there are more batches to process in current directory"""
        next_start = (self.current_batch + 1) * 3
        return next_start < len(self.png_files)
    
    def show_completion(self):
        """Show completion message"""
        plt.clf()
        
        if not self.single_directory_mode:
            self.progress_tracker.cleanup_completed_session()
        
        plt.text(0.5, 0.5, "🎉 All images processed!\n\nBatch cropping complete.", 
                 ha='center', va='center', fontsize=20, 
                 bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen"))
        plt.axis('off')
        plt.title("Multi-Directory Batch Crop Tool - Complete", fontsize=16)
        plt.draw()
        
        print("\n🎉 All images processed! Multi-directory batch cropping complete.")
    
    def quit(self):
        """Quit the application"""
        if self.has_pending_changes and not self.quit_confirmed:
            print("\n⚠️  WARNING: You have uncommitted changes!")
            print("   - You've made crop selections or set actions (skip/delete)")
            print("   - These changes will be LOST if you quit now")
            print("   - Press [Enter] to commit changes, or [Q] again to quit anyway")
            self.quit_confirmed = True
            return
            
        print("Quitting multi-directory batch crop tool...")
        
        if not self.single_directory_mode:
            print(f"[*] Progress saved to: {self.progress_tracker.progress_file}")
            print("[*] Resume by running the same command again")
        
        self.activity_timer.end_session()
        plt.close('all')
        sys.exit(0)
    
    def run(self):
        """Main execution loop"""
        progress = self.progress_tracker.get_progress_info()
        print(f"Starting Desktop Image Selector + Crop Tool on {self.base_directory}")
        print(f"Found {len(self.triplets)} triplet groups to process")
        print(f"Current: Triplet {progress['current_triplet']}/{progress['total_triplets']}")
        
        print("\nControls:")
        print("  Selection: [1] [2] [3] Select image (default: all DELETE)")
        print("  Reset:     [R] Reset entire row  [X/C/V] Reset individual crop")
        print("  Submit:    [Enter] Crop selected image, delete others, advance")
        print("  Navigate:  [←] Previous triplet  [Q] Quit")
        print("  Aspect:    [Space] Toggle aspect ratio lock")
        
        if self.aspect_ratio:
            print(f"\n📐 Global aspect ratio override enabled: {self.aspect_ratio:.2f}:1 (locked by default)")
        else:
            print("\n📐 Using each image's natural aspect ratio (locked by default)")
            print("   Use [Space] to toggle aspect ratio lock on/off")
            
        print("\nStarting first triplet...\n")
        
        # Load first triplet
        self.load_current_triplet()
        
        # Show the plot
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Directory Batch Crop Tool - Process multiple character directories with session persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-directory mode (auto-discover subdirectories)
  python scripts/04_batch_crop_tool_multi.py selected/
  
  # Single directory mode (process one directory)
  python scripts/04_batch_crop_tool_multi.py selected/kelly_mia/
  
  # With aspect ratio constraint
  python scripts/04_batch_crop_tool_multi.py selected/ --aspect-ratio 16:9

Session Persistence:
  Progress is automatically saved to scripts/crop_progress/
  Resume by running the same command - it will continue where you left off
        """
    )
    parser.add_argument("directory", help="Directory containing PNG images or subdirectories to crop")
    parser.add_argument("--aspect-ratio", help="Target aspect ratio (e.g., '16:9', '4:3', '1:1')")
    
    args = parser.parse_args()
    
    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
        
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)
    
    try:
        tool = DesktopImageSelectorCropTool(directory, args.aspect_ratio)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
