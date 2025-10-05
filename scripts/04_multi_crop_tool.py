#!/usr/bin/env python3
"""
Multi-Directory Crop Tool - Intelligent Directory Processing
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
  python scripts/04_multi_crop_tool.py selected/
  python scripts/04_multi_crop_tool.py selected/ --aspect-ratio 16:9

Single directory mode (legacy):
  python scripts/04_multi_crop_tool.py selected/kelly_mia/

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
Step 2: Final Cropping → THIS SCRIPT (scripts/04_multi_crop_tool.py)
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

Global: [Enter] Submit Batch  [Q] Quit
        [N] Next Directory  [P] Previous Directory  [←] Previous Batch
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from utils.base_desktop_image_tool import BaseDesktopImageTool
from utils.companion_file_utils import (
    format_image_display_name, 
    sort_image_files_by_timestamp_and_stage,
    log_select_crop_entry,
    extract_timestamp_from_filename,
    detect_stage
)


class MultiDirectoryProgressTracker:
    """Manages progress tracking across multiple directories with session persistence."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.progress_dir = Path("data/crop_progress")
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
        
        # Return ALL files (centrally sorted) - let load_batch() handle the indexing
        all_files = list(current_dir['path'].glob("*.png"))
        return sort_image_files_by_timestamp_and_stage(all_files)
    
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


class MultiCropTool(BaseDesktopImageTool):
    """Multi-directory crop tool - inherits from BaseDesktopImageTool."""
    
    def __init__(self, directory, aspect_ratio=None):
        """Initialize multi-directory crop tool."""
        super().__init__(directory, aspect_ratio, "multi_crop_tool")
        
        # Initialize progress tracker
        self.progress_tracker = MultiDirectoryProgressTracker(self.base_directory)
        
        # Check if we're in single directory mode or multi-directory mode
        self.single_directory_mode = self._is_single_directory_mode()
        
        if self.single_directory_mode:
            print("[*] Single directory mode detected")
            self.png_files = sort_image_files_by_timestamp_and_stage([f for f in self.base_directory.glob("*.png")])
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
                    all_files = sort_image_files_by_timestamp_and_stage(list(current_dir['path'].glob("*.png")))
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
        
        # Initialize batch tracking (load_batch will set current_batch correctly)
        self.current_batch = 0  # Will be set by load_batch()
        self.total_batches = (len(self.png_files) + 2) // 3  # Round up division
        
        # Debug output to help track the issue
        if not self.single_directory_mode:
            current_dir = self.progress_tracker.get_current_directory()
            print(f"[DEBUG] Starting in directory: {current_dir['name'] if current_dir else 'None'}")
            print(f"[DEBUG] Current file index: {self.progress_tracker.current_file_index}")
            print(f"[DEBUG] Calculated current batch: {self.current_batch}")
            print(f"[DEBUG] Files available: {len(self.png_files)}")
            print(f"[DEBUG] Total batches: {self.total_batches}")
        
        # Load first batch
        self.load_batch()
    
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
    
    def load_batch(self):
        """Load the current batch of up to 3 images"""
        # Use progress tracker's file index as the source of truth
        start_idx = self.progress_tracker.current_file_index
        end_idx = min(start_idx + 3, len(self.png_files))
        
        # Update current_batch to match the actual position
        self.current_batch = start_idx // 3
        
        # Get images for this batch
        batch_files = self.png_files[start_idx:end_idx]
        
        # Debug: Show which files are being loaded
        print(f"[DEBUG] Loading batch at file index {start_idx}:")
        for i, file_path in enumerate(batch_files):
            print(f"  Image {i+1}: {file_path.name}")
        
        # Reset state for new batch
        self.current_images = []
        self.image_states = []
        self.reset_batch_flags()
        self.clear_selectors()
        
        # Load and display each image
        for i, png_path in enumerate(batch_files):
            success = self.load_image_safely(png_path, i)
            if not success:
                continue
        
        # Set default status to 'selected' for multi crop tool (keep all unless specifically deleted)
        for i in range(len(self.image_states)):
            self.image_states[i]['status'] = 'selected'
        
        # Hide unused subplots
        self.hide_unused_subplots(len(batch_files))
        
        # Update title with progress information
        self.update_title()
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.draw()
    
    def update_title(self):
        """Update the title with current progress information."""
        if self.single_directory_mode:
            # Single directory mode - original behavior
            remaining_images = len(self.png_files) - (self.current_batch * 3)
            aspect_info = f" • [🔒 LOCKED] Aspect Ratio" if self.aspect_ratio else ""
            # Debug: Show first filename to verify position
            first_file = self.png_files[self.progress_tracker.current_file_index] if self.progress_tracker.current_file_index < len(self.png_files) else None
            filename_debug = f" • DEBUG: {first_file.name[:25]}..." if first_file else " • DEBUG: N/A"
            title = f"Batch {self.current_batch + 1}/{self.total_batches} • {remaining_images} images remaining • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}{filename_debug}"
        else:
            # Multi-directory mode - enhanced progress display
            progress = self.progress_tracker.get_progress_info()
            aspect_info = f" • [🔒 LOCKED] Aspect" if self.aspect_ratio else ""
            
            dir_info = f"📁 {progress['current_directory']} • {progress['files_remaining']} files remaining"
            batch_info = f"Batch {self.current_batch + 1}/{self.total_batches}"
            dirs_info = f"{progress['directories_remaining']} directories left"
            
            # Debug: Show first filename to verify position
            first_file = self.png_files[self.progress_tracker.current_file_index] if self.progress_tracker.current_file_index < len(self.png_files) else None
            filename_debug = f" • DEBUG: {first_file.name[:25]}..." if first_file else " • DEBUG: N/A"
            title = f"{dir_info} • {batch_info} • {dirs_info} • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}"
        
        self.fig.suptitle(title, fontsize=12, y=0.98)
        self.fig.canvas.draw_idle()
    
    def update_control_labels(self):
        """Update the control labels below each image"""
        reset_keys = ['X', 'C', 'V']
        
        for i, ax in enumerate(self.axes):
            if i < len(self.current_images):
                reset_key = reset_keys[i] if i < len(reset_keys) else 'R'
                
                # Get image filename and trim it
                image_path = self.current_images[i]['path']
                filename = image_path.stem  # Remove extension
                
                # Use centralized image name formatting
                display_name = format_image_display_name(filename, max_length=25, context="desktop")
                
                status = self.image_states[i].get('status', 'selected')
                if status == 'selected':
                    control_text = f"{display_name} • [{i+1}] ✓ KEEP  [{reset_key}] Reset"
                else:
                    control_text = f"{display_name} • [{i+1}] DELETE  [{reset_key}] Reset"
                ax.set_xlabel(control_text, fontsize=10)
    
    def handle_specific_keys(self, key: str):
        """Handle tool-specific keys."""
        # Directory navigation (multi-directory mode only)
        if key == 'n' and not self.single_directory_mode:
            self.next_directory()
            return
        elif key == 'p' and not self.single_directory_mode:
            self.previous_directory()
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
    
    def set_image_action(self, image_idx, action):
        """Toggle between selected and delete for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        # Toggle between selected and delete
        current_status = self.image_states[image_idx].get('status', 'selected')
        if current_status == 'selected':
            self.image_states[image_idx]['status'] = 'delete'
            print(f"[*] Image {image_idx + 1} marked for deletion")
        else:
            self.image_states[image_idx]['status'] = 'selected'
            print(f"[*] Image {image_idx + 1} marked to keep")
            
        self.has_pending_changes = True
        
        # Update display
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        self.fig.canvas.draw_idle()
    
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
    
    def go_back(self):
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
    
    def submit_batch(self):
        """Process all images in the current batch"""
        if not self.current_images:
            return
            
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
        else:
            # In single directory mode, advance both batch counter and progress tracker
            self.current_batch += 1
            # Also update progress tracker since load_batch uses it as source of truth
            self.progress_tracker.current_file_index = self.current_batch * 3
        
        # Move to next batch or directory
        if self.has_next_batch():
            # Progress was already advanced above by processed_count
            # Just load the next batch
            self.load_batch()
        elif not self.single_directory_mode and self.progress_tracker.has_more_directories():
            self.next_directory()
        else:
            self.show_completion()
    
    def has_next_batch(self):
        """Check if there are more batches to process in current directory"""
        next_start = (self.current_batch + 1) * 3
        return next_start < len(self.png_files)
    
    def show_completion(self):
        """Show completion message"""
        plt.clf()
        
        if not self.single_directory_mode:
            self.progress_tracker.cleanup_completed_session()
        
        plt.text(0.5, 0.5, "🎉 All images processed!\n\nMulti-directory cropping complete.", 
                 ha='center', va='center', fontsize=20, 
                 bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen"))
        plt.axis('off')
        plt.title("Multi-Directory Crop Tool - Complete", fontsize=16)
        plt.draw()
        
        print("\n🎉 All images processed! Multi-directory cropping complete.")
    
    def crop_and_save(self, image_info: Dict, crop_coords: Tuple[int, int, int, int]):
        """Override to add training logging for multi-crop tool."""
        # Call parent implementation for actual cropping
        super().crop_and_save(image_info, crop_coords)
        
        # Log training data
        try:
            png_path = image_info['path']
            session_id = getattr(self.progress_tracker, 'session_data', {}).get('session_start', '')
            set_id = f"{png_path.parent.name}_{png_path.stem}"  # Use directory + filename as set_id
            directory = str(png_path.parent)
            
            # For multi-crop, we process images individually, so treat each as its own "set"
            image_paths = [str(png_path)]
            image_stages = [detect_stage(png_path.name)]
            image_sizes = [image_info.get('size', (0, 0))]
            chosen_idx = 0  # Always 0 since we're logging individual crops
            
            # Normalize crop coordinates
            w, h = image_info.get('size', (1, 1))
            x1, y1, x2, y2 = crop_coords
            crop_norm = (x1/max(1,w), y1/max(1,h), x2/max(1,w), y2/max(1,h))
            
            log_select_crop_entry(session_id, set_id, directory, image_paths, image_stages, image_sizes, chosen_idx, crop_norm)
        except Exception as e:
            # Don't let logging errors break the workflow
            pass
    
    def run(self):
        """Main execution loop"""
        if self.single_directory_mode:
            print(f"Starting single directory multi crop tool on {self.base_directory}")
            print(f"Found {len(self.png_files)} images")
        else:
            progress = self.progress_tracker.get_progress_info()
            print(f"Starting multi-directory multi crop tool on {self.base_directory}")
            print(f"Directory {progress['directory_index']} of {progress['total_directories']}: {progress['current_directory']}")
            print(f"Files remaining in this directory: {progress['files_remaining']}")
            print(f"Directories remaining after this: {progress['directories_remaining']}")
        
        print(f"Will process in {self.total_batches} batches of up to 3 images each")
        print("\nControls:")
        print("  Image 1: [1] Delete  [X] Reset")
        print("  Image 2: [2] Delete  [C] Reset") 
        print("  Image 3: [3] Delete  [V] Reset")
        print("  Global:  [Enter] Submit Batch  [Q] Quit")
        print("  Navigation: [←] Previous Batch")
        
        if not self.single_directory_mode:
            print("  Multi-Directory: [N] Next Directory  [P] Previous Directory")
        
        if self.aspect_ratio:
            print(f"\n📐 Global aspect ratio override enabled: {self.aspect_ratio:.2f}:1 (locked by default)")
        else:
            print("\n📐 Using each image's natural aspect ratio (locked by default)")
            
        print("\nStarting first batch...\n")
        
        # Show the plot
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Directory Crop Tool - Process multiple character directories with session persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-directory mode (auto-discover subdirectories)
  python scripts/04_multi_crop_tool.py selected/
  
  # Single directory mode (process one directory)
  python scripts/04_multi_crop_tool.py selected/kelly_mia/
  
  # With aspect ratio constraint
  python scripts/04_multi_crop_tool.py selected/ --aspect-ratio 16:9

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
        tool = MultiCropTool(directory, args.aspect_ratio)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
