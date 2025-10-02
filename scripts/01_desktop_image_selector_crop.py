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
2. Click or press [[]\ to select which image to keep
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
Selection:  [ [ \ Keep Image (auto-shows crop rectangle)
Reset:      [R] Reset Row (all back to delete, clear crops)
Navigation: [←] Previous Triplet  [Enter] Submit & Next Triplet
Global:     [Q] Quit  [H] Help

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

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from utils.base_desktop_image_tool import BaseDesktopImageTool

# Triplet detection constants and functions (from web image selector)
STAGE_NAMES = [
    "stage1_generated",
    "stage1.5_face_swapped", 
    "stage2_upscaled",
    "stage3_enhanced"
]

@dataclass
class Triplet:
    """Represents a triplet of related images."""
    paths: List[Path]
    display_name: str
    timestamp: str

class TripletProgressTracker:
    """Manages progress tracking for triplet processing with session persistence."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.progress_dir = Path("data/sorter_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # Create progress file name based on base directory
        safe_name = str(base_directory).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
        
        self.triplets = []
        self.current_triplet_index = 0
        self.session_data = {}
        
        self.discover_triplets()
        self.load_progress()
    
    def discover_triplets(self):
        """Discover all triplets in the directory."""
        # Implementation from original file
        all_files = sorted([f for f in self.base_directory.glob("*.png")])
        triplets = []
        
        # Group files by timestamp (first 14 characters of filename)
        timestamp_groups = {}
        for file_path in all_files:
            timestamp = file_path.stem[:14]
            if timestamp not in timestamp_groups:
                timestamp_groups[timestamp] = []
            timestamp_groups[timestamp].append(file_path)
        
        # Create triplets from groups
        for timestamp, files in timestamp_groups.items():
            if len(files) >= 2:  # At least 2 images to form a triplet
                triplet = Triplet(
                    paths=sorted(files),
                    display_name=f"{timestamp} ({len(files)} images)",
                    timestamp=timestamp
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
            
            print(f"[*] Resumed session from: {self.progress_file}")
            
        except Exception as e:
            print(f"[!] Error loading progress: {e}")
            self.initialize_progress()
    
    def initialize_progress(self):
        """Initialize new progress tracking session."""
        self.session_data = {
            'base_directory': str(self.base_directory),
            'session_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_triplet_index': 0,
            'triplets': {
                triplet.display_name: {
                    'status': 'pending',
                    'files_processed': 0,
                    'total_files': len(triplet.paths)
                } for triplet in self.triplets
            }
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
                self.session_data['triplets'][current_triplet.display_name]['status'] = 'completed'
        
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
        
    def cleanup_completed_session(self):
        """Remove progress file when session is complete."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                print(f"[*] Session complete - removed progress file: {self.progress_file}")
        except Exception as e:
            print(f"[!] Error cleaning up progress file: {e}")


class DesktopImageSelectorCrop(BaseDesktopImageTool):
    """Desktop image selector with cropping - inherits from BaseDesktopImageTool."""
    
    def __init__(self, directory, aspect_ratio=None, exts=["png"]):
        """Initialize desktop image selector."""
        super().__init__(directory, aspect_ratio, "desktop_image_selector_crop")
        
        self.exts = exts
        self.current_triplet = None
        self.previous_batch_confirmed = False
        
        # Initialize progress tracker
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
        
        # Load each image in the triplet
        for i, png_path in enumerate(self.current_triplet.paths):
            if i >= 3:  # Only process up to 3 images
                break
            
            success = self.load_image_safely(png_path, i)
            if not success:
                continue
            
            # Fix the image state for desktop selector (needs 'status' field)
            if i < len(self.image_states):
                self.image_states[i]['status'] = 'delete'  # Default to delete
        
        # Hide unused subplots
        self.hide_unused_subplots(len(self.current_triplet.paths))
        
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
        
        triplet_info = f"Triplet {self.progress_tracker.current_triplet_index + 1}/{progress['total_triplets']} • {self.current_triplet.display_name if self.current_triplet else 'Loading...'}"
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
                
                # Get image filename and trim it
                image_path = self.current_images[i]['path']
                filename = image_path.stem  # Remove extension
                
                # Trim timestamp from beginning if it's too long (first 14 chars are usually timestamp)
                if len(filename) > 20 and filename[:14].replace('_', '').isdigit():
                    display_name = filename[14:]  # Remove timestamp
                else:
                    display_name = filename
                
                # Limit display name length
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                
                if status == 'selected':
                    control_text = f"{display_name} • [{i+1}] ✓ KEEP  [{reset_key}] Reset"
        else:
                    control_text = f"{display_name} • [{i+1}] DELETE  [{reset_key}] Reset"
                ax.set_xlabel(control_text, fontsize=10)
    
    def handle_specific_keys(self, key: str):
        """Handle tool-specific keys."""
        # Image selection controls
        if key in ['1', '2', '3']:
            image_idx = int(key) - 1
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
                    self.selectors[i].extents = (0, img_width, 0, img_height)
        
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
        if not self.current_triplet:
            return
            
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
    
    def submit_batch(self):
        """Process current triplet - crop selected image, delete others."""
        if not self.current_images:
            return
        
        self.activity_timer.mark_batch(f"Triplet {self.progress_tracker.current_triplet_index + 1}")
        self.activity_timer.mark_activity()
        
        print(f"\nProcessing triplet: {self.current_triplet.display_name}")
        
        processed_count = 0
        selected_image_idx = None
        
        # Find the selected image
        for i, state in enumerate(self.image_states):
            if state.get('status') == 'selected':
                selected_image_idx = i
                break
        
        if selected_image_idx is None:
            print("No image selected - skipping triplet")
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
        print(f"Triplet {self.progress_tracker.current_triplet_index + 1} of {progress['total_triplets']}: {progress['current_triplet']}")
        
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
        tool = DesktopImageSelectorCrop(directory, args.aspect_ratio)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
