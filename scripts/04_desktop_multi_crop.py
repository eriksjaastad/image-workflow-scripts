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
  python scripts/04_desktop_multi_crop.py selected/
  python scripts/04_desktop_multi_crop.py crop/

Single directory mode (legacy):
  python scripts/04_desktop_multi_crop.py selected/kelly_mia/

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
Step 1: Character Processing → scripts/02_character_processor.py
Step 2: Final Cropping → THIS SCRIPT (scripts/04_desktop_multi_crop.py)
Step 3: Basic Review → scripts/05_web_multi_directory_viewer.py

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
from matplotlib.widgets import Button
from PIL import Image
import numpy as np

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from utils.base_desktop_image_tool import BaseDesktopImageTool
from utils.companion_file_utils import (
    format_image_display_name, 
    sort_image_files_by_timestamp_and_stage,
    log_select_crop_entry,
    detect_stage,
    move_file_with_all_companions,
    extract_timestamp_from_filename
)

# AI Training (optional, non-blocking)
try:
    from scripts.ai.training_snapshot import capture_crop_decision
    HAS_AI_LOGGING = True
except ImportError:
    HAS_AI_LOGGING = False
    capture_crop_decision = None
"""
NOTE: Focus timer durations can be edited via FOCUS_TIMER_WORK_MIN and
FOCUS_TIMER_REST_MIN constants below. Search this file for
FOCUS_TIMER_WORK_MIN to locate the configuration block quickly.
"""

# Focus timer configuration (minutes)
FOCUS_TIMER_WORK_MIN: int = 15
FOCUS_TIMER_REST_MIN: int = 5


class _FocusTimer:
    """Lightweight Pomodoro-style focus timer anchored to a figure top strip.

    UI elements (top-right):
      [Work mm:ss] [Rest mm:ss] [Start/Pause] [Reset]

    The timer toggles work → rest → work automatically. Only Start/Pause and
    Reset are user-facing. A subtle tint is applied to the timer strip during
    the rest phase. No keyboard bindings are used.
    """

    def __init__(self, fig: plt.Figure, work_min: int, rest_min: int) -> None:
        self.fig = fig
        self.work_total = max(1, int(work_min)) * 60
        self.rest_total = max(1, int(rest_min)) * 60
        self.phase = 'work'  # 'work' | 'rest'
        self.remaining = self.work_total
        self._running = False
        self._last_tick = None  # type: Optional[float]
        self._timer = None

        # Axes/controls placeholders
        self.bg_ax = None
        self.work_txt = None
        self.rest_txt = None
        self.start_btn = None
        self.reset_btn = None

    def init_ui(self) -> None:
        # Reserve a slim band near the very top, above the subplot area
        # Keep within [0.983, 1.0] to avoid overlapping tight_layout top=0.98
        band_bottom = 0.983
        band_height = 0.015
        left_base = 0.58
        self.bg_ax = self.fig.add_axes([left_base, band_bottom, 0.40, band_height])
        self.bg_ax.set_xticks([])
        self.bg_ax.set_yticks([])
        self.bg_ax.set_facecolor('white')
        for spine in self.bg_ax.spines.values():
            spine.set_visible(False)

        # Text readouts inside bg_ax (use axes coords)
        self.work_txt = self.bg_ax.text(0.02, 0.5, self._fmt("Work", self.remaining),
                                        va='center', ha='left', fontsize=9)
        self.rest_txt = self.bg_ax.text(0.28, 0.5, self._fmt("Rest", self.rest_total),
                                        va='center', ha='left', fontsize=9, color='#2f9e44')

        # Buttons (tiny) – place to the right within the band
        self.start_ax = self.fig.add_axes([left_base + 0.26, band_bottom + 0.001, 0.08, band_height - 0.002])
        self.reset_ax = self.fig.add_axes([left_base + 0.35, band_bottom + 0.001, 0.07, band_height - 0.002])
        self.start_btn = Button(self.start_ax, 'Start', color='#e9ecef', hovercolor='#dee2e6')
        self.reset_btn = Button(self.reset_ax, 'Reset', color='#e9ecef', hovercolor='#dee2e6')
        self.start_btn.on_clicked(lambda _evt: self.start_pause())
        self.reset_btn.on_clicked(lambda _evt: self.reset())

        # Create a 1s UI timer (matplotlib Timer)
        self._timer = self.fig.canvas.new_timer(interval=1000)
        self._timer.add_callback(self._tick)
        self._apply_phase_style()

    def start_pause(self) -> None:
        if not self._running:
            self._running = True
            self._last_tick = time.monotonic()
            try:
                self.start_btn.label.set_text('Pause')
            except Exception:
                pass
            if self._timer:
                self._timer.start()
        else:
            self._running = False
            try:
                self.start_btn.label.set_text('Start')
            except Exception:
                pass
            if self._timer:
                self._timer.stop()

        self.fig.canvas.draw_idle()

    def reset(self) -> None:
        self.phase = 'work'
        self.remaining = self.work_total
        self._running = False
        if self._timer:
            self._timer.stop()
        try:
            self.start_btn.label.set_text('Start')
        except Exception:
            pass
        self._update_texts()
        self._apply_phase_style()
        self.fig.canvas.draw_idle()

    def _advance_phase(self) -> None:
        if self.phase == 'work':
            self.phase = 'rest'
            self.remaining = self.rest_total
        else:
            self.phase = 'work'
            self.remaining = self.work_total
        self._apply_phase_style()

    def _tick(self) -> None:
        if not self._running:
            return
        now = time.monotonic()
        dt = 1.0
        if self._last_tick is not None:
            dt = max(0.5, min(5.0, now - self._last_tick))
        self._last_tick = now

        self.remaining -= int(round(dt))
        if self.remaining <= 0:
            self._advance_phase()
            self.remaining = max(0, self.remaining)
            # Immediately continue counting in the next phase if still running
        self._update_texts()
        self.fig.canvas.draw_idle()

    def _fmt(self, label: str, secs: int) -> str:
        m = max(0, int(secs)) // 60
        s = max(0, int(secs)) % 60
        return f"{label} {m:02d}:{s:02d}"

    def _update_texts(self) -> None:
        # Work readout shows current phase time if phase is work; rest readout remains constant label
        if self.phase == 'work':
            self.work_txt.set_text(self._fmt("Work", self.remaining))
            self.rest_txt.set_text(self._fmt("Rest", self.rest_total))
        else:
            self.work_txt.set_text(self._fmt("Work", self.work_total))
            self.rest_txt.set_text(self._fmt("Rest", self.remaining))

    def _apply_phase_style(self) -> None:
        try:
            if self.phase == 'rest':
                self.bg_ax.set_facecolor('#fff4e6')  # soft amber tint
            else:
                self.bg_ax.set_facecolor('white')
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._timer:
            try:
                self._timer.stop()
            except Exception:
                pass



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
        """Discover all subdirectories containing PNG files, sorted alphabetically.
        Skips directories ending with '_cropped' as those contain processed files."""
        subdirs = []
        
        for item in self.base_directory.iterdir():
            if item.is_dir():
                # Skip directories ending with _cropped (processed files)
                if item.name.endswith('_cropped'):
                    continue
                
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
        """Load existing progress from file - uses directory name and rescans files."""
        if not self.progress_file.exists():
            self.initialize_progress()
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                self.session_data = json.load(f)
            
            # Get saved directory name
            saved_dir_name = self.session_data.get('current_directory_name')
            
            # Find the directory by name in our discovered directories
            self.current_directory_index = 0
            if saved_dir_name:
                for i, dir_info in enumerate(self.directories):
                    if dir_info['name'] == saved_dir_name:
                        self.current_directory_index = i
                        break
                else:
                    # Saved directory not found - it might be complete or renamed
                    print(f"[*] Saved directory '{saved_dir_name}' not found - starting at first available directory")
                    self.current_directory_index = 0
            
            # Always start at file index 0 - we rescan to see what's actually left
            self.current_file_index = 0
            
            # Ensure current directory has files
            current_dir = self.get_current_directory()
            if current_dir:
                actual_files = list(current_dir['path'].glob("*.png"))
                if not actual_files:
                    # Current directory is empty, advance to next
                    print(f"[*] Directory '{current_dir['name']}' is empty, advancing...")
                    self.current_directory_index += 1
                    if self.current_directory_index >= len(self.directories):
                        self.current_directory_index = 0
            
            # Update/merge directories dictionary
            if 'directories' not in self.session_data:
                self.session_data['directories'] = {}
            
            for dir_info in self.directories:
                dir_name = dir_info['name']
                if dir_name not in self.session_data['directories']:
                    self.session_data['directories'][dir_name] = {
                        'status': 'pending',
                        'total_files_initial': dir_info['file_count']
                    }
            
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
            'current_directory_name': self.directories[0]['name'] if self.directories else None,
            'directories': {
                dir_info['name']: {
                    'status': 'pending',
                    'total_files_initial': dir_info['file_count']
                } for dir_info in self.directories
            }
        }
        self.current_directory_index = 0
        self.current_file_index = 0
        self.save_progress()
    
    def save_progress(self):
        """Save current progress to file - only saves directory name, not file indices."""
        current_dir = self.get_current_directory()
        if current_dir:
            self.session_data['current_directory_name'] = current_dir['name']
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
            if current_dir and current_dir['name'] in self.session_data['directories']:
                self.session_data['directories'][current_dir['name']]['status'] = 'completed'
        
        self.current_directory_index += 1
        self.current_file_index = 0
        
        # Mark new directory as in progress
        current_dir = self.get_current_directory()
        if current_dir and current_dir['name'] in self.session_data['directories']:
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
    
    def __init__(self, directory, aspect_ratio=None, enable_ai_logging=True):
        """Initialize multi-directory crop tool."""
        super().__init__(directory, aspect_ratio, "multi_crop_tool")
        # Focus timer instance will be initialized after first batch load when fig exists
        self._focus_timer: Optional[_FocusTimer] = None
        
        # Configure panel bounds for this tool: support 1–3 images per batch
        self.min_panels = 1
        self.max_panels = 3

        # AI logging (optional, non-blocking)
        self.enable_ai_logging = enable_ai_logging and HAS_AI_LOGGING
        if self.enable_ai_logging:
            print("[AI] 🤖 Training data logging enabled (background, zero impact)")

        # Initialize progress tracker
        self.progress_tracker = MultiDirectoryProgressTracker(self.base_directory)
        
        # Check if we're in single directory mode or multi-directory mode
        self.single_directory_mode = self._is_single_directory_mode()
        
        if self.single_directory_mode:
            print("[*] Single directory mode detected")
            all_png_files = list(self.base_directory.glob("*.png"))
            print(f"[DEBUG] Found {len(all_png_files)} PNG files before sorting")
            
            if all_png_files:
                print(f"[DEBUG] Files found:")
                for f in all_png_files:
                    print(f"  - {f.name} (size: {f.stat().st_size} bytes)")
            
            self.png_files = sort_image_files_by_timestamp_and_stage(all_png_files)
            print(f"[DEBUG] After sorting: {len(self.png_files)} files")
            
            if self.png_files:
                print(f"[DEBUG] Sorted files:")
                for f in self.png_files:
                    print(f"  - {f.name}")
            
            if not self.png_files:
                print(f"[ERROR] No PNG files found in {directory}")
                print(f"[ERROR] Checked path: {self.base_directory.absolute()}")
                print(f"[ERROR] Directory exists: {self.base_directory.exists()}")
                print(f"[ERROR] Is directory: {self.base_directory.is_dir()}")
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
        
        # Load first batch (file_index always starts at 0 after rescan in load_progress)
        self.load_batch()
    
    def _is_single_directory_mode(self) -> bool:
        """Determine if we're processing a single directory or multiple directories."""
        # Check if the base directory contains PNG files directly
        png_files = list(self.base_directory.glob("*.png"))
        print(f"[DEBUG] Checking {self.base_directory}")
        print(f"[DEBUG] Found {len(png_files)} PNG files directly in directory")
        
        if png_files:
            print(f"[DEBUG] → Single directory mode (PNG files present)")
            print(f"[DEBUG] Files: {[f.name for f in png_files[:10]]}")  # Show first 10
            return True
        
        # Check if it has subdirectories with PNG files
        subdirs_with_images = []
        for item in self.base_directory.iterdir():
            if item.is_dir():
                item_pngs = list(item.glob("*.png"))
                if item_pngs:
                    subdirs_with_images.append((item.name, len(item_pngs)))
        
        print(f"[DEBUG] Found {len(subdirs_with_images)} subdirectories with PNG files")
        if subdirs_with_images:
            print(f"[DEBUG] Subdirectories: {subdirs_with_images[:5]}")  # Show first 5
            print(f"[DEBUG] → Multi-directory mode")
        else:
            print(f"[DEBUG] → Single directory mode (no subdirectories with images)")
        
        return len(subdirs_with_images) == 0
    
    def get_cropped_directory(self, source_dir: Path) -> Path:
        """Get the corresponding _cropped directory for a source directory.
        Example: crop/dalia_hannah/ → crop/dalia_hannah_cropped/"""
        parent = source_dir.parent
        dir_name = source_dir.name
        cropped_name = f"{dir_name}_cropped"
        return parent / cropped_name
    
    def load_batch(self):
        """Load the current batch of up to 3 images - optimized"""
        # Use progress tracker's file index as the source of truth
        start_idx = self.progress_tracker.current_file_index
        end_idx = min(start_idx + 3, len(self.png_files))
        
        # Update current_batch to match the actual position
        self.current_batch = start_idx // 3
        
        # Get images for this batch
        batch_files = self.png_files[start_idx:end_idx]
        
        # Safety check: if no files in batch, don't try to load
        if not batch_files:
            return
        
        # IMPORTANT: Setup display for correct number of images BEFORE loading
        # This prevents hide_unused_subplots from clearing already-loaded images
        self.hide_unused_subplots(len(batch_files))
        
        # Reset state for new batch
        self.current_images = []
        self.image_states = []
        self.reset_batch_flags()
        self.clear_selectors()
        
        # Load and display each image (optimized: no print per image)
        for i, png_path in enumerate(batch_files):
            success = self.load_image_safely(png_path, i)
            if not success:
                continue
        
        # Set default action to None (crop) for multi crop tool - matches base class pattern
        # Images will be cropped unless explicitly deleted
        for i in range(len(self.image_states)):
            self.image_states[i]['action'] = None  # None means crop (default)
        
        # Update title with progress information
        self.update_title()
        self.update_image_titles(self.image_states)
        self.update_control_labels()
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # Initialize focus timer UI once (top-right strip)
        if self._focus_timer is None:
            try:
                self._focus_timer = _FocusTimer(self.fig, FOCUS_TIMER_WORK_MIN, FOCUS_TIMER_REST_MIN)
                self._focus_timer.init_ui()
            except Exception as e:
                print(f"[timer] init failed: {e}")
        plt.draw()
    
    def update_title(self):
        """Update the title with current progress information."""
        if self.single_directory_mode:
            # Single directory mode - simple format
            images_done = self.progress_tracker.current_file_index
            total_images = len(self.png_files)
            aspect_info = f" • [🔒 LOCKED] Aspect Ratio" if self.aspect_ratio else ""
            title = f"{images_done}/{total_images} images • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}"
        else:
            # Multi-directory mode - simple format
            progress = self.progress_tracker.get_progress_info()
            aspect_info = f" • [🔒 LOCKED] Aspect" if self.aspect_ratio else ""
            
            images_done = self.progress_tracker.current_file_index
            total_images = len(self.png_files)
            dirs_info = f"{progress['directories_remaining']} directories left" if progress['directories_remaining'] > 0 else "Last directory!"
            
            title = f"📁 {progress['current_directory']} • {images_done}/{total_images} images • {dirs_info} • [1,2,3] Delete • [Enter] Submit • [Q] Quit{aspect_info}"
        
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
                
                # Extract timestamp for display at bottom
                timestamp = extract_timestamp_from_filename(filename) or "NO_TIMESTAMP"
                
                # Use centralized image name formatting
                display_name = format_image_display_name(filename, max_length=25, context="desktop")
                
                # Check action field (matches base class pattern: None = crop, 'delete' = delete)
                action = self.image_states[i].get('action')
                if action == 'delete':
                    control_text = f"{display_name}\n{timestamp} • [{i+1}] DELETE  [{reset_key}] Reset"
                else:
                    control_text = f"{display_name}\n{timestamp} • [{i+1}] ✓ KEEP  [{reset_key}] Reset"
                ax.set_xlabel(control_text, fontsize=10)

    def update_image_titles(self, image_states):
        """Override to ensure default state is KEEP/SELECTED, not DELETE."""
        if not image_states:
            return
        
        for i, state in enumerate(image_states):
            if i >= len(self.axes) or i >= len(self.current_images) or not isinstance(state, dict):
                continue
            action = state.get('action')
            has_selection = state.get('has_selection', False)
            name = format_image_display_name(self.current_images[i]['path'].name, context='desktop')
            if action == 'delete':
                self.axes[i].set_title(f"Image {i + 1}: DELETE - {name}", fontsize=12, color='red')
            elif has_selection:
                self.axes[i].set_title(f"Image {i + 1}: SELECTED - {name}", fontsize=12, color='blue', weight='bold')
            else:
                self.axes[i].set_title(f"Image {i + 1}: READY - {name}", fontsize=12, color='black')
    
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
        """Toggle between keep (None/crop) and delete for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        # Toggle between None (crop) and 'delete' - matches base class pattern
        current_action = self.image_states[image_idx].get('action')
        if current_action == 'delete':
            self.image_states[image_idx]['action'] = None  # None means crop
            print(f"[*] Image {image_idx + 1} marked to keep/crop")
        else:
            self.image_states[image_idx]['action'] = 'delete'
            print(f"[*] Image {image_idx + 1} marked for deletion")
            
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
        """Process all images in the current batch - optimized for speed"""
        import traceback
        print(f"\n[DEBUG submit_batch] CALLED! Stack trace:")
        for line in traceback.format_stack()[:-1]:
            print(line.strip())
        
        if not self.current_images:
            print("[DEBUG submit_batch] No current images to process")
            return
        
        print(f"\n[DEBUG submit_batch] Processing batch with {len(self.current_images)} images")
        print(f"[DEBUG submit_batch] Image states:")
        for i, state in enumerate(self.image_states):
            print(f"  Image {i+1}: action={state.get('action')}, has_selection={state.get('has_selection')}, crop_coords={state.get('crop_coords')}")
            
        # Collect operations first (faster than processing one-by-one)
        operations = []
        processed_count = 0
        
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            png_path = image_info['path']
            yaml_path = png_path.with_suffix('.yaml')
            action = state.get('action')
            
            print(f"[DEBUG] Image {i+1} ({png_path.name}):")
            print(f"  - action: {action}")
            print(f"  - has_selection: {state.get('has_selection')}")
            print(f"  - crop_coords: {state.get('crop_coords')}")
            
            if action == 'skip':
                print(f"  → Will SKIP")
                operations.append(('skip', png_path, yaml_path, None))
                processed_count += 1
            elif action == 'delete':
                print(f"  → Will DELETE")
                operations.append(('delete', png_path, yaml_path, None))
                processed_count += 1
            elif action is None and state['has_selection'] and state['crop_coords']:
                print(f"  → Will CROP with coords: {state['crop_coords']}")
                operations.append(('crop', png_path, yaml_path, (image_info, state['crop_coords'])))
                processed_count += 1
            else:
                print(f"  → NO ACTION (will be left untouched)")
        
        print(f"\n[DEBUG submit_batch] Total operations queued: {processed_count}")
        
        # If nothing is queued, don't advance; stay on the same batch
        if processed_count == 0:
            print("⚠️  No operations queued (no crops/deletes). Not advancing.")
            return
        
        # Clear pending flags before heavy I/O
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # 1) Process file operations FIRST
        print(f"[DEBUG] Processing {len(operations)} file operations...")
        for op_type, png_path, yaml_path, extra in operations:
            try:
                if op_type == 'skip':
                    self.move_to_cropped(png_path, yaml_path, "skipped")
                elif op_type == 'delete':
                    self.safe_delete(png_path, yaml_path)
                elif op_type == 'crop':
                    image_info, crop_coords = extra
                    self.crop_and_save(image_info, crop_coords)
            except Exception as e:
                print(f"Error processing {png_path.name}: {e}")
        
        print(f"✓ Processed {processed_count} images")
        
        # 2) Advance progress by the actual count processed (not by full batch size)
        if not self.single_directory_mode:
            self.progress_tracker.advance_file(processed_count)
        else:
            self.progress_tracker.current_file_index += processed_count
            self.current_batch = self.progress_tracker.current_file_index // 3
            self.progress_tracker.save_progress()
        
        # 3) Refresh list from disk (files may have been moved/deleted)
        self.png_files = self.progress_tracker.get_current_files()
        
        # 4) Decide where to go next
        if self.has_next_batch():
            self.load_batch()
        elif (not self.single_directory_mode) and self.progress_tracker.has_more_directories():
            self.next_directory()
        else:
            self.show_completion()
            return
    
    def has_next_batch(self):
        """Check if there are more batches to process in current directory"""
        # Use unified logic: look at the progress tracker's file index
        next_start = self.progress_tracker.current_file_index
        remaining = len(self.png_files) - next_start
        print(f"[DEBUG has_next_batch] current_file_index={self.progress_tracker.current_file_index}, remaining={remaining}")
        return remaining > 0
    
    def show_completion(self):
        """Show completion message"""
        plt.clf()
        # Stop focus timer if running
        try:
            if self._focus_timer:
                self._focus_timer.stop()
        except Exception:
            pass
        
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
        """Override to crop, save, and move to _cropped directory."""
        png_path = image_info['path']
        source_dir = png_path.parent
        
        # Call parent implementation for actual cropping (saves in place)
        super().crop_and_save(image_info, crop_coords)
        
        # Move the cropped file (and all companions) to the _cropped directory
        try:
            cropped_dir = self.get_cropped_directory(source_dir)
            moved_files = move_file_with_all_companions(png_path, cropped_dir, dry_run=False)
            print(f"[*] Moved {len(moved_files)} file(s) to {cropped_dir.name}/")
        except Exception as e:
            print(f"[!] Error moving files to {cropped_dir.name}: {e}")
            # Continue despite move errors
        
        # Log training data (existing CSV logging)
        try:
            session_id = getattr(self.progress_tracker, 'session_data', {}).get('session_start', '')
            set_id = f"{source_dir.name}_{png_path.stem}"  # Use directory + filename as set_id
            directory = str(source_dir)
            
            # For multi-crop, we process images individually, so treat each as its own "set"
            image_paths = [str(png_path)]
            image_stages = [detect_stage(png_path.name)]
            image_sizes = [image_info.get('original_size', (0, 0))]  # Fixed: was 'size', should be 'original_size'
            chosen_idx = 0  # Always 0 since we're logging individual crops
            
            # Normalize crop coordinates
            w, h = image_info.get('original_size', (1, 1))  # Fixed: was 'size', should be 'original_size'
            x1, y1, x2, y2 = crop_coords
            crop_norm = (x1/max(1,w), y1/max(1,h), x2/max(1,w), y2/max(1,h))
            
            log_select_crop_entry(session_id, set_id, directory, image_paths, image_stages, image_sizes, chosen_idx, crop_norm)
            
            # AI training snapshot (NEW - async, non-blocking)
            if self.enable_ai_logging and capture_crop_decision:
                capture_crop_decision(
                    image_path=png_path,
                    crop_coords=crop_norm,
                    action="cropped",
                    image_size=(w, h),
                    session_id=session_id,
                    tool="multi_crop_tool"
                )
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
    parser.add_argument("--no-ai-logging", action="store_true", 
                       help="Disable AI training data capture (saves images + decisions for future model training)")
    
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
        tool = MultiCropTool(directory, args.aspect_ratio, enable_ai_logging=not args.no_ai_logging)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
