#!/usr/bin/env python3
"""
Enhanced Multi-Directory Batch Crop Tool - Intelligent Directory Processing
===========================================================================
Process multiple character directories in one continuous session with automatic
discovery, progress tracking, and session persistence.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Multi-directory mode (NEW):
  python scripts/04_batch_crop_tool_multi.py selected/
  python scripts/04_batch_crop_tool_multi.py selected/ --aspect-ratio 16:9

Single directory mode (legacy):
  python scripts/04_batch_crop_tool_multi.py selected/kelly_mia/

FEATURES:
---------
• Auto-discover character subdirectories (kelly_mia/, astrid_kelly/, etc.)
• Alphabetical directory processing order
• Session persistence - resume exactly where you left off
• Clean progress tracking: files remaining + directories left
• Process 3 images at once in side-by-side layout
• Individual crop rectangles for each image
• Intuitive hotkey system (W-S-X, E-D-C, R-F-V)
• Batch submission reduces overhead dramatically
• Large, easy-to-grab handles with generous click zones
• Configurable aspect ratios (1:1, 16:9, 4:3, free)
• Real-time crop preview with zoom capabilities

PROGRESS TRACKING:
------------------
• Files: "587 files remaining in kelly_mia"
• Directories: "3 directories left after this one"
• Session persistence stored in scripts/crop_progress/

WORKFLOW POSITION:
------------------
Step 1: Character Processing → scripts/utils/character_processor.py
Step 2: Final Cropping → THIS SCRIPT (scripts/04_batch_crop_tool_multi.py)
Step 3: Basic Review → scripts/05_multi_directory_viewer.py

FILE HANDLING:
--------------
• Cropped images: Saved over original files in place (emily/image.png → emily/image.png)
• Skipped images: Left unchanged in original directory
• Deleted images: Moved to trash (no longer in directory)

CONTROLS:
---------
Image 1: [1] Delete  [X] Reset crop
Image 2: [2] Delete  [C] Reset crop  
Image 3: [3] Delete  [V] Reset crop

Global: [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Q] Quit
        [N] Next Directory  [P] Previous Directory  [←] Previous Batch
"""

import argparse
import json
import sys
import time
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
        """Get PNG files from current directory, starting from current file index."""
        current_dir = self.get_current_directory()
        if not current_dir:
            return []
        
        all_files = sorted(current_dir['path'].glob("*.png"))
        return all_files[self.current_file_index:]
    
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


class MultiDirectoryBatchCropTool:
    """Enhanced batch crop tool with multi-directory support."""
    
    def __init__(self, directory, aspect_ratio=None):
        self.base_directory = Path(directory)
        self.aspect_ratio = self._parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
        self.aspect_ratio_locked = True
        self.tracker = FileTracker("multi_batch_crop_tool")
        
        # Initialize activity timer
        self.activity_timer = ActivityTimer("04_batch_crop_tool_multi")
        self.activity_timer.start_session()
        
        # Initialize progress tracker
        self.progress_tracker = MultiDirectoryProgressTracker(self.base_directory)
        
        # Check if we're in single directory mode or multi-directory mode
        self.single_directory_mode = self._is_single_directory_mode()
        
        if self.single_directory_mode:
            print("[*] Single directory mode detected")
            self.png_files = sorted([f for f in self.base_directory.glob("*.png")])
            if not self.png_files:
                raise ValueError(f"No PNG files found in {directory}")
        else:
            print("[*] Multi-directory mode enabled")
            if not self.progress_tracker.has_more_directories():
                print("🎉 All directories completed!")
                self.progress_tracker.cleanup_completed_session()
                sys.exit(0)
            
            self.png_files = self.progress_tracker.get_current_files()
            if not self.png_files:
                # Check if we're truly at the end or if there's a tracking issue
                current_dir = self.progress_tracker.get_current_directory()
                if current_dir:
                    all_files = sorted(current_dir['path'].glob("*.png"))
                    if all_files:
                        print(f"[!] Progress tracking issue detected in {current_dir['name']}")
                        print(f"[!] Directory has {len(all_files)} files but current_file_index is {self.progress_tracker.current_file_index}")
                        print("[!] Resetting to start of directory to prevent data loss")
                        self.progress_tracker.current_file_index = 0
                        self.progress_tracker.save_progress()
                        self.png_files = self.progress_tracker.get_current_files()
                    else:
                        print("Directory is truly empty, advancing...")
                        self.progress_tracker.advance_directory()
                        if self.progress_tracker.has_more_directories():
                            self.png_files = self.progress_tracker.get_current_files()
                        else:
                            print("🎉 All directories completed!")
                            self.progress_tracker.cleanup_completed_session()
                            sys.exit(0)
                else:
                    print("🎉 All directories completed!")
                    self.progress_tracker.cleanup_completed_session()
                    sys.exit(0)
        
        self.current_batch = 0
        self.total_batches = (len(self.png_files) + 2) // 3  # Round up division
        
        # Debug output to help track the issue
        if not self.single_directory_mode:
            current_dir = self.progress_tracker.get_current_directory()
            print(f"[DEBUG] Starting in directory: {current_dir['name'] if current_dir else 'None'}")
            print(f"[DEBUG] Current file index: {self.progress_tracker.current_file_index}")
            print(f"[DEBUG] Files available: {len(self.png_files)}")
            print(f"[DEBUG] Total batches: {self.total_batches}")
        
        # State for current batch of 3 images
        self.current_images = []
        self.image_states = []  # List of dicts with crop coords, action, etc.
        self.selectors = []
        self.axes = []
        self.has_pending_changes = False
        self.quit_confirmed = False
        self.previous_batch_confirmed = False
        
        # No longer using a separate cropped directory - files stay in place
        
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
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initialize selectors list
        self.selectors = [None, None, None]
    
    def load_batch(self):
        """Load the current batch of up to 3 images"""
        start_idx = self.current_batch * 3
        end_idx = min(start_idx + 3, len(self.png_files))
        
        # Get images for this batch
        batch_files = self.png_files[start_idx:end_idx]
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
        
        # Load and display each image
        for i, png_path in enumerate(batch_files):
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
                
                # Initialize state for this image
                self.image_states.append({
                    'action': None,  # 'skip', 'delete', or None (crop)
                    'crop_coords': None,
                    'has_selection': False,
                    'image_aspect_ratio': image_aspect_ratio
                })
                
                # Display image
                ax = self.axes[i]
                ax.clear()
                ax.imshow(img_array, aspect='equal')
                ax.set_title(f"Image {i+1}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Remove axis spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
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
        for i in range(len(batch_files), 3):
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
            title = f"Batch {self.current_batch + 1}/{self.total_batches} • {remaining_images} images remaining • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}"
        else:
            # Multi-directory mode - enhanced progress display
            progress = self.progress_tracker.get_progress_info()
            lock_str = "🔒" if self.aspect_ratio_locked else "🔓"
            aspect_info = f" • [{lock_str} Space] Aspect" if self.aspect_ratio else ""
            
            dir_info = f"📁 {progress['current_directory']} • {progress['files_remaining']} files remaining"
            batch_info = f"Batch {self.current_batch + 1}/{self.total_batches}"
            dirs_info = f"{progress['directories_remaining']} directories left"
            
            title = f"{dir_info} • {batch_info} • {dirs_info} • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}"
        
        self.fig.suptitle(title, fontsize=12, y=0.98)
    
    def update_control_labels(self):
        """Update the control labels below each image"""
        controls = [
            "[1] Delete  [X] Reset",
            "[2] Delete  [C] Reset", 
            "[3] Delete  [V] Reset"
        ]
        
        for i, (ax, control_text) in enumerate(zip(self.axes, controls)):
            if i < len(self.current_images):
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
        
        # Debug: Print key for troubleshooting
        if key in ['left', 'arrow_left', 'leftarrow']:
            print(f"DEBUG: Arrow key detected: '{event.key}' -> '{key}'")
        
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
        elif key in ['left', 'arrow_left', 'leftarrow'] or key == 'b':
            self.previous_batch()
            return
            
        # Image-specific controls
        image_actions = {
            '1': (0, 'delete'), 'x': (0, 'reset'),
            '2': (1, 'delete'), 'c': (1, 'reset'),
            '3': (2, 'delete'), 'v': (2, 'reset'),
        }
        
        if key in image_actions:
            image_idx, action = image_actions[key]
            
            if image_idx < len(self.current_images):
                if action == 'reset':
                    self.reset_image_crop(image_idx)
                else:
                    self.set_image_action(image_idx, action)
    
    def next_directory(self):
        """Move to next directory (multi-directory mode only)."""
        if self.single_directory_mode:
            return
        
        print("Moving to next directory...")
        self.progress_tracker.advance_directory()
        
        if self.progress_tracker.has_more_directories():
            self.png_files = self.progress_tracker.get_current_files()
            self.current_batch = 0
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
        self.current_batch = 0
        self.total_batches = (len(self.png_files) + 2) // 3
        self.load_batch()
    
    def previous_batch(self):
        """Go back to the previous batch within the current directory"""
        if self.current_batch <= 0:
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
        
        print(f"Going back to batch {self.current_batch}...")
        
        # Move back one batch
        self.current_batch -= 1
        
        # Reverse the progress tracking - subtract the files from the previous batch
        if not self.single_directory_mode:
            # Calculate how many files were in the batch we're going back to
            batch_start = self.current_batch * 3
            batch_end = min(batch_start + 3, len(self.png_files))
            files_in_batch = batch_end - batch_start
            
            # Subtract those files from the progress
            self.progress_tracker.current_file_index -= files_in_batch
            if self.progress_tracker.current_file_index < 0:
                self.progress_tracker.current_file_index = 0
            self.progress_tracker.save_progress()
            
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
        """Process all images in the current batch"""
        if not self.current_images:
            return
            
        self.activity_timer.mark_batch(f"Crop batch {self.current_batch + 1}")
        self.activity_timer.mark_activity()
            
        print(f"\nProcessing batch {self.current_batch + 1}...")
        
        processed_count = 0
        
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            png_path = image_info['path']
            yaml_path = png_path.with_suffix('.yaml')
            
            try:
                if state['action'] == 'skip':
                    self.move_to_cropped(png_path, yaml_path, "skipped")
                    processed_count += 1
                elif state['action'] == 'delete':
                    self.safe_delete(png_path, yaml_path)
                    processed_count += 1
                elif state['has_selection'] and state['crop_coords']:
                    self.crop_and_save(image_info, state['crop_coords'])
                    processed_count += 1
                else:
                    print(f"Image {i + 1}: No action specified, skipping...")
                    
            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
        
        print(f"Processed {processed_count}/{len(self.current_images)} images in batch")
        
        # Clear pending changes flag after successful submission
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # Update progress tracking
        if not self.single_directory_mode:
            self.progress_tracker.advance_file(processed_count)
        
        self.activity_timer.end_batch(f"Completed {processed_count} operations")
        
        # Move to next batch or directory
        if self.has_next_batch():
            self.current_batch += 1
            self.load_batch()
        elif not self.single_directory_mode and self.progress_tracker.has_more_directories():
            self.next_directory()
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
        if self.single_directory_mode:
            print(f"Starting single directory batch crop tool on {self.base_directory}")
            print(f"Found {len(self.png_files)} images")
        else:
            progress = self.progress_tracker.get_progress_info()
            print(f"Starting multi-directory batch crop tool on {self.base_directory}")
            print(f"Directory {progress['directory_index']} of {progress['total_directories']}: {progress['current_directory']}")
            print(f"Files remaining in this directory: {progress['files_remaining']}")
            print(f"Directories remaining after this: {progress['directories_remaining']}")
        
        print(f"Will process in {self.total_batches} batches of up to 3 images each")
        print("\nControls:")
        print("  Image 1: [1] Delete  [X] Reset")
        print("  Image 2: [2] Delete  [C] Reset") 
        print("  Image 3: [3] Delete  [V] Reset")
        print("  Global:  [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Q] Quit")
        print("  Navigation: [←] Previous Batch")
        
        if not self.single_directory_mode:
            print("  Multi-Directory: [N] Next Directory  [P] Previous Directory")
        
        if self.aspect_ratio:
            print(f"\n📐 Global aspect ratio override enabled: {self.aspect_ratio:.2f}:1 (locked by default)")
        else:
            print("\n📐 Using each image's natural aspect ratio (locked by default)")
            print("   Use [Space] to toggle aspect ratio lock on/off")
            
        print("\nStarting first batch...\n")
        
        # Load first batch
        self.load_batch()
        
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
        tool = MultiDirectoryBatchCropTool(directory, args.aspect_ratio)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
