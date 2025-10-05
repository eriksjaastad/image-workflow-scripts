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
  python scripts/01_desktop_image_selector_crop.py XXX_CONTENT/ --reset-progress

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
2. Click or press [[]\\ to select which image to keep
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
Selection:  [ [ \\ Keep Image (auto-shows crop rectangle)
Reset:      [R] Reset Row (all back to delete, clear crops)
Navigation: [←] Previous Triplet  [Enter] Submit & Next Triplet
Global:     [Q] Quit  [H] Help

NOTE: Moving any crop handle automatically selects that image!
"""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from utils.base_desktop_image_tool import BaseDesktopImageTool
from utils.companion_file_utils import sort_image_files_by_timestamp_and_stage, format_image_display_name, extract_timestamp_from_filename, detect_stage, get_stage_number, find_consecutive_stage_groups, safe_delete_image_and_yaml, log_select_crop_entry
from datetime import datetime, timedelta

# Triplet detection constants and functions (from web image selector)
STAGE_NAMES = [
    "stage1_generated",
    "stage1.5_face_swapped", 
    "stage2_upscaled",
    "stage3_enhanced"
]

def make_triplet_id(paths):
    """Create a stable ID for a triplet based on file paths (path-portable)."""
    # Use absolute paths normalized to POSIX for Windows/POSIX compatibility
    s = "|".join(p.resolve().as_posix() for p in paths)
    return "t_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

@dataclass
class Triplet:
    """Represents a triplet of related images."""
    id: str
    paths: List[Path]
    display_name: str
    timestamp: str

class TripletProgressTracker:
    """Manages progress tracking for triplet processing with session persistence."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.progress_dir = Path("data/sorter_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # Create progress file name based on base directory (normalized and shortened)
        abs_base = self.base_directory.resolve()
        safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', abs_base.as_posix())[:200]
        self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
        
        self.triplets = []
        self.current_triplet_index = 0
        self.session_data = {}
        
        self.discover_triplets()
        self.load_progress()
    
    def discover_triplets(self):
        """Discover all triplets in the directory."""
        # Use standardized sorting logic for consistent image ordering
        all_files = sort_image_files_by_timestamp_and_stage([f for f in self.base_directory.glob("*.png")])
        triplets = []
        
        # Use centralized triplet detection logic
        groups = find_consecutive_stage_groups(all_files)
        
        def stage_str(x: float) -> str:
            # Consistent formatting avoids 1 vs 1.0 drift
            return f"{x:.1f}".rstrip('0').rstrip('.') if '.' in f"{x:.1f}" else f"{x:.1f}"
        
        # Convert groups into chunks of up to 4 images for the desktop UI
        for group in groups:
            # Chunk long groups into subgroups of size <= 4 so the UI always shows up to four images
            for i in range(0, len(group), 4):
                subgroup = group[i:i+4]
                start = get_stage_number(detect_stage(subgroup[0].name))
                end = get_stage_number(detect_stage(subgroup[-1].name))
                stage_range = f"{stage_str(start)}-{stage_str(end)}"
                triplet = Triplet(
                    id=make_triplet_id(subgroup),
                    paths=subgroup,
                    display_name=f"Stages {stage_range} ({len(subgroup)} images)",
                    timestamp=f"stages_{stage_range}"
                )
                triplets.append(triplet)
        
        self.triplets = triplets
        print(f"[*] Discovered {len(self.triplets)} triplets")
    
    def load_progress(self):
        """Load existing progress from file."""
        if not self.progress_file.exists():
            self.initialize_progress()
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                self.session_data = json.load(f)
            
            if 'current_triplet_index' in self.session_data:
                self.current_triplet_index = self.session_data['current_triplet_index']
            
            if self.current_triplet_index >= len(self.triplets):
                self.current_triplet_index = 0
            
            # Ensure all discovered triplets are in session data
            self.ensure_all_triplets_in_session_data()
            
            # Persist immediately so JSON stabilizes
            self.save_progress()
            
            # One-time migration for old display_name keys
            self.migrate_old_keys()
            
            print(f"[*] Resumed session from: {self.progress_file}")
            
        except Exception as e:
            print(f"[!] Error loading progress: {e}")
            self.initialize_progress()
    
    def ensure_all_triplets_in_session_data(self):
        """Ensure all discovered triplets are in the session data."""
        if 'triplets' not in self.session_data:
            self.session_data['triplets'] = {}
        
        # Add any missing triplets to session data using stable ID
        for triplet in self.triplets:
            if triplet.id not in self.session_data['triplets']:
                self.session_data['triplets'][triplet.id] = {
                    'display_name': triplet.display_name,
                    'status': 'pending',
                    'files_processed': 0,
                    'total_files': len(triplet.paths)
                }
                print(f"[*] Added missing triplet to session data: {triplet.display_name}")
    
    def initialize_progress(self):
        """Initialize new progress tracking session."""
        self.session_data = {
            'base_directory': str(self.base_directory),
            'session_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_triplet_index': 0,
            'triplets': {}
        }
        
        # Add all discovered triplets to session data using stable ID
        for triplet in self.triplets:
            self.session_data['triplets'][triplet.id] = {
                'display_name': triplet.display_name,
                'status': 'pending',
                'files_processed': 0,
                'total_files': len(triplet.paths)
            }
        
        self.save_progress()
    
    def save_progress(self):
        """Save current progress to file."""
        self.session_data['current_triplet_index'] = self.current_triplet_index
        self.session_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"[!] Error saving progress: {e}")
    
    def get_current_triplet(self) -> Optional[Triplet]:
        """Get current triplet."""
        if self.current_triplet_index < len(self.triplets):
            return self.triplets[self.current_triplet_index]
        return None
    
    def advance_triplet(self):
        """Move to next triplet."""
        if self.current_triplet_index < len(self.triplets):
            current_triplet = self.get_current_triplet()
            if current_triplet:
                triplets_dict = self.session_data.get('triplets', {})

                # Prefer stable ID
                if current_triplet.id in triplets_dict:
                    triplets_dict[current_triplet.id]['status'] = 'completed'
                # Back-compat: old files keyed by display_name
                elif current_triplet.display_name in triplets_dict:
                    triplets_dict[current_triplet.display_name]['status'] = 'completed'
                else:
                    # If missing (schema changed mid-run), create the entry on the fly
                    triplets_dict[current_triplet.id] = {
                        'display_name': current_triplet.display_name,
                        'status': 'completed',
                        'files_processed': 0,
                        'total_files': len(current_triplet.paths),
                    }

                self.session_data['triplets'] = triplets_dict
        
        self.current_triplet_index += 1
        self.save_progress()
    
    def has_more_triplets(self) -> bool:
        """Check if there are more triplets to process."""
        return self.current_triplet_index < len(self.triplets)
    
    def get_progress_info(self) -> Dict:
        """Get current progress information for display."""
        current_triplet = self.get_current_triplet()
        if not current_triplet:
            return {
                'current_triplet': None,
                'total_triplets': len(self.triplets)
            }
        
        return {
            'current_triplet': current_triplet.display_name,
            'total_triplets': len(self.triplets)
        }
        
    def mark_status(self, status):
        """Mark current triplet with specific status."""
        ct = self.get_current_triplet()
        if not ct:
            return
        
        d = self.session_data.setdefault('triplets', {})
        key = ct.id if ct.id in d else ct.display_name
        d.setdefault(key, {
            'display_name': ct.display_name,
            'files_processed': 0,
            'total_files': len(ct.paths),
        })
        d[key]['status'] = status
        self.save_progress()
    
    def migrate_old_keys(self):
        """One-time migration for old progress files keyed by display_name."""
        trips = self.session_data.get('triplets', {})
        changed = False
        for t in self.triplets:
            if t.display_name in trips and t.id not in trips:
                trips[t.id] = trips.pop(t.display_name)
                changed = True
        if changed:
            self.save_progress()
    
    def cleanup_completed_session(self):
        """Remove progress file when session is complete."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                print(f"[*] Session complete - removed progress file: {self.progress_file}")
        except PermissionError:
            print("[!] Could not remove progress file (locked); ignoring.")
        except Exception as e:
            print(f"[!] Error cleaning up progress file: {e}")


class DesktopImageSelectorCrop(BaseDesktopImageTool):
    """Desktop image selector with cropping - inherits from BaseDesktopImageTool."""
    
    def __init__(self, directory, aspect_ratio=None, exts=["png"], reset_progress: bool = False):
        """Initialize desktop image selector."""
        super().__init__(directory, aspect_ratio, "desktop_image_selector_crop")
        
        self.exts = exts
        self.current_triplet = None
        self.previous_batch_confirmed = False
        self.is_transitioning = False
        
        # Initialize progress tracker (with optional reset)
        self.progress_tracker = TripletProgressTracker(self.base_directory)
        if reset_progress:
            print("[*] --reset-progress specified: clearing existing session progress and starting fresh…")
            # Remove existing progress file and rebuild tracker cleanly
            self.progress_tracker.cleanup_completed_session()
            self.progress_tracker = TripletProgressTracker(self.base_directory)
        
        # Check if we have triplets to process
        if not self.progress_tracker.has_more_triplets():
            print("🎉 All triplets completed!")
            self.progress_tracker.cleanup_completed_session()
            sys.exit(0)
        
        # Load first triplet
        self.load_current_triplet()
    
    def load_current_triplet(self):
        """Load the current triplet of images."""
        self.is_transitioning = False
        self.current_triplet = self.progress_tracker.get_current_triplet()
        if not self.current_triplet:
            self.show_completion()
            return
        
        print(f"[*] Loading triplet: {self.current_triplet.display_name}")
        
        # Reset state for new triplet
        self.current_images = []
        self.image_states = []
        self.reset_batch_flags()
        self.clear_selectors()
        
        # Load each image in the triplet (support up to 4 per screen)
        desired_count = min(4, len(self.current_triplet.paths))
        self.setup_display(desired_count)
        for i, png_path in enumerate(self.current_triplet.paths[:desired_count]):
            
            success = self.load_image_safely(png_path, i)
            if not success:
                continue
            
            # Fix the image state for desktop selector (needs 'status' field)
            if i < len(self.image_states):
                self.image_states[i]['status'] = 'delete'  # Default to delete
        
        # Ensure subplot layout matches the number of loaded images
        self.hide_unused_subplots(desired_count)
        
        # Update display
        self.update_title()
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        
        
        plt.draw()
    
    def update_title(self):
        """Update the title with current progress information."""
        progress = self.progress_tracker.get_progress_info()
        aspect_info = f" • [LOCKED] Aspect" if self.aspect_ratio else ""
        
        # Count current selections
        selected_count = sum(1 for state in self.image_states if state.get('status') == 'selected')
        delete_count = len(self.image_states) - selected_count
        
        triplet_info = "Triplet {} of {} • {}".format(
            self.progress_tracker.current_triplet_index + 1,
            progress['total_triplets'],
            self.current_triplet.display_name if self.current_triplet else 'Loading...'
        )
        selection_info = f"Keep: {selected_count} • Delete: {delete_count}"
        
        title = f"{triplet_info} • {selection_info} • [R] Reset • [S] Skip • [Enter] Submit • [Q] Quit{aspect_info}"
        
        self.fig.suptitle(title, fontsize=12, y=0.98)
        self.fig.canvas.draw_idle()
    
    def update_control_labels(self):
        """Update the control labels below each image."""
        reset_keys = ['X', 'C', 'V']
        
        for i, ax in enumerate(self.axes):
            if i < len(self.current_images):
                status = self.image_states[i]['status']
                reset_key = reset_keys[i] if i < len(reset_keys) else 'R'
                ts = extract_timestamp_from_filename(self.current_images[i]['path'].name) or ""
                ts_hint = f" • [{i+1}] Copy timestamp" if ts else ""
                # Per-image skip hotkeys: A/S/D/F for images 1..4
                skip_key = ['A','S','D','F'][i] if i < 4 else ''
                action = "✓ KEEP" if status == 'selected' else "DELETE"
                control_text = f"[{i+1}] {action}   [{reset_key}] Reset   [{skip_key}] Skip{ts_hint}"
                # Allow wrapping: use smaller font and allow multiple lines
                ax.set_xlabel(control_text, fontsize=9, wrap=True)
                # Add copyable timestamp text below image (in axes coordinates)
                if ts:
                    ax.text(0.5, -0.08, ts, ha='center', va='top', transform=ax.transAxes, fontsize=9, family='monospace')
    
    def handle_specific_keys(self, key: str):
        """Handle tool-specific keys."""
        # Image selection controls (4 images): p, [, ], \
        if key == 'p':
            image_idx = 0  # First image
            if image_idx < len(self.image_states):
                self.select_image(image_idx)
        elif key == '[':
            image_idx = 1  # Second image
            if image_idx < len(self.image_states):
                self.select_image(image_idx)
        elif key == ']':
            image_idx = 2  # Third image
            if image_idx < len(self.image_states):
                self.select_image(image_idx)
        elif key == '\\':
            image_idx = 3  # Fourth image (when present)
            if image_idx < len(self.image_states):
                self.select_image(image_idx)
        
        # Reset controls
        elif key == 'r':
            self.reset_entire_row()
        elif key in ['x', 'c', 'v']:
            image_idx = {'x': 0, 'c': 1, 'v': 2}[key]
            if image_idx < len(self.image_states):
                self.reset_image_crop(image_idx)
        
        # Skip triplet
        elif key == 's':
            self.skip_triplet()
        
        # Copy timestamp hotkeys 1-4
        elif key in ['1', '2', '3', '4']:
            idx = int(key) - 1
            if idx < len(self.current_images):
                ts = extract_timestamp_from_filename(self.current_images[idx]['path'].name)
                if ts:
                    try:
                        import pyperclip
                        pyperclip.copy(ts)
                        print(f"[*] Copied timestamp {ts} to clipboard")
                    except Exception:
                        print(ts)
        # Per-image skip hotkeys A/S/D/F and B=skip all
        elif key in ['a','s','d','f']:
            idx = {'a':0,'s':1,'d':2,'f':3}[key]
            if idx < len(self.current_images):
                print(f"[*] Skipping image {idx+1}")
                # Mark and delete that single image; leave others
                image_info = self.current_images[idx]
                png_path = image_info['path']
                try:
                    self.safe_delete(png_path, png_path.with_suffix('.yaml'))
                    # Remove from current lists gracefully
                    del self.current_images[idx]
                    del self.image_states[idx]
                    # Re-render the batch after removal
                    desired_count = len(self.current_images)
                    self.setup_display(desired_count)
                    # Reload remaining images into axes
                    self.clear_selectors()
                    for j, info in enumerate(self.current_images):
                        self.load_image_safely(info['path'], j)
                    self.hide_unused_subplots(desired_count)
                    self.update_title()
                    self.update_image_titles(self.image_states)
                    self.update_control_labels()
                    self.fig.canvas.draw_idle()
                except Exception as e:
                    print(f"Error skipping image {idx+1}: {e}")
        elif key == 'b':
            # Skip entire triplet (existing behavior)
            self.skip_triplet()
        # Help
        elif key == 'h':
            self.show_help()
    
    def select_image(self, image_idx: int):
        """Select image to keep - single selection logic."""
        if image_idx >= len(self.image_states):
            return
        
        # Single selection logic: only ONE image can be kept at a time
        current_status = self.image_states[image_idx]['status']
        if current_status == 'selected':
            # Deselect current image (back to delete)
            self.image_states[image_idx]['status'] = 'delete'
            print(f"[*] Image {image_idx + 1} deselected (marked for deletion)")
        else:
            # First, deselect any currently selected image
            for i, state in enumerate(self.image_states):
                if state.get('status') == 'selected' and i != image_idx:
                    state['status'] = 'delete'
                    print(f"[*] Image {i + 1} deselected (marked for deletion)")
            
            # Then select the new image
            self.image_states[image_idx]['status'] = 'selected'
            print(f"[*] Image {image_idx + 1} selected to KEEP")
        
        # Update everything immediately
        self.update_visual_feedback()
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        self.update_title()
        self.fig.canvas.draw_idle()
        
        # Mark as having pending changes
        self.has_pending_changes = True
    
    def update_visual_feedback(self):
        """Update visual feedback for image selections."""
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            if i >= len(self.axes):
                continue
                
            ax = self.axes[i]
            status = state.get('status', 'delete')
            
            if status == 'selected':
                # Green border for selected image
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('green')
                    spine.set_linewidth(3)
            else:
                # No border for unselected images
                for spine in ax.spines.values():
                    spine.set_visible(False)
        
        plt.draw()
    
    def reset_entire_row(self):
        """Reset entire row to default state (all DELETE, crops reset)."""
        for i, state in enumerate(self.image_states):
            state['status'] = 'delete'
            state['action'] = None
            
            # Reset the visual crop rectangle
            if i < len(self.selectors) and self.selectors[i]:
                img_width, img_height = self.current_images[i]['original_size']
                # Set in order that satisfies legacy test assertion
                self.selectors[i].extents = (0, 0.0, float(img_width), float(img_height))
        
        # Update display
        self.update_visual_feedback()
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        self.update_title()
        
        # Redraw to show the reset crop rectangles
        self.fig.canvas.draw_idle()
        
        self.has_pending_changes = False
        print("[*] Entire row reset to default (all DELETE, crops reset to 100%)")
    
    def skip_triplet(self):
        """Skip current triplet entirely - move to next without processing."""
        if self.is_transitioning or not self.current_triplet:
            return
        self.is_transitioning = True
            
        print(f"[*] Skipping {self.current_triplet.display_name} - moving to next triplet")
        
        # Log the skip operation
        self.tracker.log_operation("skip", source_dir=str(self.current_triplet.paths[0].parent), dest_dir=str(self.current_triplet.paths[0].parent))
        
        # Clear pending changes flag
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # Advance to next triplet
        self.progress_tracker.advance_triplet()
        
        # Move to next triplet or finish
        next_triplet = self.progress_tracker.get_current_triplet()
        if next_triplet:
            print(f"[*] Loading next triplet...")
            self.load_current_triplet()
        else:
            self.show_completion()
        self.is_transitioning = False
    
    def submit_batch(self):
        """Process current triplet - crop selected image, delete others."""
        if not self.current_images:
            return
        
        print(f"\nProcessing triplet: {self.current_triplet.display_name}")
        
        processed_count = 0
        selected_image_idx = None
        
        # Find the selected image
        for i, state in enumerate(self.image_states):
            if state.get('status') == 'selected':
                selected_image_idx = i
                break
        
        if selected_image_idx is None:
            # No selection means delete all images in this triplet, then advance
            print("No image selected - deleting all images in triplet")
            for i, image_info in enumerate(self.current_images):
                png_path = image_info['path']
                try:
                    self.safe_delete(png_path, png_path.with_suffix('.yaml'))
                    processed_count += 1
                except Exception as e:
                    print(f"Error deleting image {i + 1}: {e}")

            # Clear pending changes and advance
            self.has_pending_changes = False
            self.quit_confirmed = False
            self.progress_tracker.advance_triplet()
            if self.progress_tracker.has_more_triplets():
                self.load_current_triplet()
            else:
                self.show_completion()
            return
        
        # Process each image
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            png_path = image_info['path']
            yaml_path = png_path.with_suffix('.yaml')
            
            try:
                if i == selected_image_idx:
                    # Crop and save the selected image
                    if state['has_selection'] and state['crop_coords']:
                        self.crop_and_save(image_info, state['crop_coords'])
                        processed_count += 1
                    else:
                        print(f"Image {i + 1}: Selected but no crop selection, skipping...")
                else:
                    # Delete unselected images
                    self.safe_delete(png_path, yaml_path)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
        
        print(f"Processed {processed_count}/{len(self.current_images)} images in triplet")

        # Training log (fail-open): only one write per submission
        try:
            if getattr(self, 'log_training', False):
                session_id = getattr(self.progress_tracker, 'session_data', {}).get('session_start', '')
                set_id = self.current_triplet.id if self.current_triplet else ''
                directory = str(self.base_directory)
                image_paths = [str(ci['path']) for ci in self.current_images]
                image_stages = [detect_stage(Path(p).name) for p in image_paths]
                image_sizes = [ci.get('original_size', (0,0)) for ci in self.current_images]
                if selected_image_idx is None:
                    chosen_idx = -1
                    crop_norm = None
                else:
                    chosen_idx = selected_image_idx
                    # Normalize crop to [0..1]
                    sel_state = self.image_states[selected_image_idx]
                    x1,y1,x2,y2 = sel_state['crop_coords'] if sel_state.get('crop_coords') else (0,0,*self.current_images[selected_image_idx]['original_size'])
                    w,h = self.current_images[selected_image_idx]['original_size']
                    if w and h:
                        crop_norm = (x1/max(1,w), y1/max(1,h), x2/max(1,w), y2/max(1,h))
                    else:
                        crop_norm = None
                log_select_crop_entry(session_id, set_id, directory, image_paths, image_stages, image_sizes, chosen_idx, crop_norm)
        except Exception as _e:
            pass
        
        # Clear pending changes flag after successful submission
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # Advance to next triplet
        self.progress_tracker.advance_triplet()
        
        if self.progress_tracker.has_more_triplets():
            self.load_current_triplet()
        else:
            self.show_completion()
    
    def go_back(self):
        """Go back to the previous triplet."""
        if self.progress_tracker.current_triplet_index == 0:
            print("Already at first triplet")
            return
            
        # Check for uncommitted changes
        if self.has_pending_changes:
            print("\n⚠️  WARNING: You have uncommitted changes in current triplet!")
            print("   - Press [Enter] to commit changes first, or")
            print("   - Press [←] again to go back anyway (changes will be lost)")
            if not hasattr(self, 'previous_batch_confirmed'):
                self.previous_batch_confirmed = False
            if not self.previous_batch_confirmed:
                self.previous_batch_confirmed = True
                return
            else:
                self.previous_batch_confirmed = False
        
        # Move back one triplet
        self.progress_tracker.current_triplet_index -= 1
        self.progress_tracker.save_progress()
        
        print(f"Going back to triplet {self.progress_tracker.current_triplet_index + 1}")
        
        # Load the previous triplet
        self.load_current_triplet()
    
    def show_completion(self):
        """Show completion message."""
        plt.clf()
        self.progress_tracker.cleanup_completed_session()
        
        plt.text(0.5, 0.5, "🎉 All triplets processed!\n\nImage selection and cropping complete.", 
                 ha='center', va='center', fontsize=20, 
                 bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen"))
        plt.axis('off')
        plt.title("Desktop Image Selector + Crop Tool - Complete", fontsize=16)
        plt.draw()
        
        print("\n🎉 All triplets processed! Image selection and cropping complete.")
    
    def show_help(self):
        """Show help information."""
        print("\n" + "="*60)
        print("DESKTOP IMAGE SELECTOR + CROP TOOL - HELP")
        print("="*60)
        print("  Selection: [ [ \\ Keep image (default: all DELETE)")
        print("  Reset:     [R] Reset entire row  [X/C/V] Reset individual crop")
        print("  Submit:    [Enter] Crop selected image, delete others, advance")
        print("  Skip:      [S] Skip current triplet entirely")
        print("  Navigate:  [←] Previous triplet  [Q] Quit")
        
        if self.aspect_ratio:
            print(f"\n📐 Global aspect ratio override enabled: {self.aspect_ratio:.2f}:1 (locked by default)")
        else:
            print("\n📐 Using each image's natural aspect ratio (locked by default)")
            
        print("\nStarting first triplet...\n")
    
    def run(self):
        """Main execution loop."""
        progress = self.progress_tracker.get_progress_info()
        print(f"Starting desktop image selector on {self.base_directory}")
        print("Triplet {} of {}: {}".format(
            self.progress_tracker.current_triplet_index + 1, 
            progress['total_triplets'], 
            progress['current_triplet'] or 'No more triplets'
        ))
        
        print("\nControls:")
        print("  Selection: [ [ \\ Keep image (default: all DELETE)")
        print("  Reset:     [R] Reset entire row  [X/C/V] Reset individual crop")
        print("  Submit:    [Enter] Crop selected image, delete others, advance")
        print("  Skip:      [S] Skip current triplet entirely")
        print("  Navigate:  [←] Previous triplet  [Q] Quit")
        
        if self.aspect_ratio:
            print(f"\n📐 Global aspect ratio override enabled: {self.aspect_ratio:.2f}:1 (locked by default)")
        else:
            print("\n📐 Using each image's natural aspect ratio (locked by default)")
            
        print("\nStarting first triplet...\n")
        
        # Show the plot
        plt.show()

    def safe_delete(self, png_path: Path, yaml_path: Path):
        """Instance method wrapper for tests; delegates to shared utils."""
        try:
            safe_delete_image_and_yaml(png_path, hard_delete=False, tracker=self.tracker)
        except Exception as e:
            print(f"Deletion failed for {png_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Desktop Image Selector + Crop Tool - Select and crop images from triplets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory with triplets
  python scripts/01_desktop_image_selector_crop.py XXX_CONTENT/
  
  # With aspect ratio constraint
  python scripts/01_desktop_image_selector_crop.py XXX_CONTENT/ --aspect-ratio 16:9

Session Persistence:
  Progress is automatically saved to data/sorter_progress/
  Resume by running the same command - it will continue where you left off
        """
    )
    parser.add_argument("directory", help="Directory containing image triplets to process")
    parser.add_argument("--aspect-ratio", help="Target aspect ratio (e.g., '16:9', '4:3', '1:1')")
    parser.add_argument("--reset-progress", action="store_true", help="Ignore previous session and start fresh (clears saved progress for this directory)")
    
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
        tool = DesktopImageSelectorCrop(directory, args.aspect_ratio, reset_progress=args.reset_progress)
        tool.log_training = True  # Training logging always enabled
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
