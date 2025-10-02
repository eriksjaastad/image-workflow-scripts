#!/usr/bin/env python3
"""
Base Desktop Image Tool - Shared functionality for desktop image processing tools
================================================================================
Common functionality for desktop image tools including matplotlib setup, crop handling,
keyboard events, file operations, and progress tracking.

This base class provides shared functionality that both DesktopImageSelectorCrop
and MultiDirectoryBatchCropTool can inherit from, reducing code duplication and
making maintenance easier.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
from send2trash import send2trash

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.file_tracker import FileTracker
from utils.activity_timer import ActivityTimer


class BaseDesktopImageTool:
    """Base class for desktop image processing tools with shared functionality."""
    
    def __init__(self, directory: Path, aspect_ratio: Optional[float] = None, tool_name: str = "base_tool"):
        """Initialize base desktop image tool with common functionality."""
        self.base_directory = Path(directory)
        self.aspect_ratio = aspect_ratio
        self.aspect_ratio_locked = True  # Always locked as per user preference
        self.tool_name = tool_name
        
        # Initialize trackers
        self.tracker = FileTracker(tool_name)
        self.activity_timer = ActivityTimer(tool_name)
        self.activity_timer.start_session()
        
        # Common state
        self.current_images = []
        self.image_states = []  # List of dicts with crop coords, action, etc.
        self.selectors = []
        self.axes = []
        self.has_pending_changes = False
        self.quit_confirmed = False
        
        # Matplotlib setup
        self.fig = None
        self.setup_display()
    
    def setup_display(self):
        """Setup the matplotlib figure with 3 subplots optimized for screen space."""
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
        
        # Center images with minimal margins and good spacing
        # Keep images large while centering them properly
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.04, wspace=0.08)
        print(f"🔧 Layout set: left=0.05, right=0.95, wspace=0.08")
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initialize selectors list
        self.selectors = [None, None, None]
    
    def create_crop_selector(self, ax, image_idx: int, img_width: int, img_height: int):
        """Create a RectangleSelector for the given axis and image."""
        selector = RectangleSelector(
            ax, 
            lambda eclick, erelease, idx=image_idx: self.on_crop_select(eclick, erelease, idx),
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
        
        # Set initial crop selection to full image
        selector.extents = (0, img_width, 0, img_height)
        
        return selector
    
    def on_crop_select(self, eclick, erelease, image_idx: int):
        """Handle crop rectangle selection for a specific image."""
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
        
        # Auto-select this image when cropping (crop tool = "keep this image")
        self.image_states[image_idx]['status'] = 'selected'
        print(f"🔍 DEBUG: Auto-selected image {image_idx + 1} for cropping")
        
        self.has_pending_changes = True
        
        # Update title to show selection status instead of crop info
        self.update_image_titles(self.image_states)
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """Handle keyboard input - common keys handled here, specific keys delegated to subclasses."""
        self.activity_timer.mark_activity()
        key = event.key.lower()
        
        # Common global controls
        if key == 'q':
            self.handle_quit()
            return
        elif key == 'escape':
            self.handle_quit()
            return
        elif key == 'enter':
            self.submit_batch()
            return
        elif key in ['left', 'arrow_left', 'leftarrow'] or key == 'b':
            self.go_back()
            return
        
        # Delegate specific key handling to subclasses
        self.handle_specific_keys(key)
    
    def handle_specific_keys(self, key: str):
        """Handle tool-specific keys - override in subclasses."""
        pass
    
    def handle_quit(self):
        """Handle quit with confirmation for uncommitted changes."""
        if self.has_pending_changes and not self.quit_confirmed:
            print("\n⚠️  WARNING: You have uncommitted changes!")
            print("   - You've made crop selections or set actions (skip/delete)")
            print("   - These changes will be LOST if you quit now")
            print("   - Press [Enter] to commit changes, or [Q] again to quit anyway")
            self.quit_confirmed = True
            return
            
        print(f"Quitting {self.tool_name}...")
        self.activity_timer.end_session()
        plt.close('all')
        sys.exit(0)
    
    def update_title(self):
        """Update the title with current progress information - override in subclasses."""
        # Base implementation - subclasses should override with specific progress info
        title = f"{self.tool_name} - Ready"
        self.fig.suptitle(title, fontsize=12, y=0.98)
        self.fig.canvas.draw_idle()
    
    def update_control_labels(self):
        """Update the control labels below each image - override in subclasses."""
        # Base implementation - subclasses should override with specific controls
        for i, ax in enumerate(self.axes):
            if i < len(self.current_images):
                ax.set_xlabel(f"Image {i+1} Controls", fontsize=10)
    
    def reset_image_crop(self, image_idx: int):
        """Reset the crop selection for a specific image back to full image."""
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
        # Update title to show selection status
        self.update_image_titles(self.image_states)
        plt.draw()
    
    def crop_and_save(self, image_info: Dict, crop_coords: Tuple[int, int, int, int]):
        """Crop image and save over the original file in place."""
        png_path = image_info['path']
        yaml_path = png_path.with_suffix('.yaml')
        
        x1, y1, x2, y2 = crop_coords
        
        # Load and crop image
        img = Image.open(png_path)
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Save over the original file (in place)
        cropped_img.save(png_path)
        
        # Log the operation
        self.tracker.log_operation(
            "crop", str(png_path.parent), str(png_path.parent), 1,
            f"Cropped in place to ({x1},{y1},{x2},{y2})", 
            [png_path.name]
        )
        
        self.activity_timer.log_operation("crop", file_count=1)
        
        print(f"Cropped and saved in place: {png_path.name}")
    
    def safe_delete(self, png_path: Path, yaml_path: Path):
        """Safely delete image files."""
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
    
    def move_to_cropped(self, png_path: Path, yaml_path: Path, reason: str):
        """Mark files as processed (skipped) but leave them in original directory."""
        # Files stay in place - no actual moving
        
        file_count = 2 if yaml_path.exists() else 1
        files = [png_path.name, yaml_path.name] if yaml_path.exists() else [png_path.name]
        
        self.tracker.log_operation(
            "skip", str(png_path.parent), str(png_path.parent), file_count,
            f"Image {reason} (left in place)", files
        )
        
        self.activity_timer.log_operation("skip", file_count=1)
        print(f"Skipped (left in place): {png_path.name} ({reason})")
    
    def load_image_safely(self, png_path: Path, image_idx: int) -> bool:
        """Load an image safely and handle errors."""
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
            ax = self.axes[image_idx]
            ax.clear()
            ax.imshow(img_array, aspect='equal')
            ax.set_title(f"Image {image_idx + 1}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove axis spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.margins(0)
            
            # Create RectangleSelector
            selector = self.create_crop_selector(ax, image_idx, img_width, img_height)
            self.selectors[image_idx] = selector
            
            # Initialize the crop coordinates
            self.image_states[image_idx]['crop_coords'] = (0, 0, img_width, img_height)
            self.image_states[image_idx]['has_selection'] = True
            
            # Update title to show selection status
            self.update_image_titles(self.image_states)
            
            return True
            
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
            ax = self.axes[image_idx]
            ax.clear()
            ax.text(0.5, 0.5, f'LOAD ERROR\n{png_path.name}\n{str(e)}', 
                   ha='center', va='center', fontsize=10, color='red',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            ax.set_title(f"Image {image_idx + 1}: LOAD ERROR", fontsize=10, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            
            return False
    
    def hide_unused_subplots(self, num_images: int):
        """Hide unused subplots and center the visible ones."""
        # Hide unused subplots
        for i in range(num_images, 3):
            self.axes[i].clear()
            self.axes[i].set_visible(False)
        
        # Center the visible subplots
        if num_images == 2:
            # For 2 images, center them by adjusting subplot positions
            # Give more room for title at top - subplot height reduced to 0.88
            self.axes[0].set_position([0.05, 0.04, 0.41, 0.88])  # left, bottom, width, height
            self.axes[1].set_position([0.54, 0.04, 0.41, 0.88])  # left, bottom, width, height
            print(f"🎯 Centered 2 images: pos1=[0.05, 0.04, 0.41, 0.88], pos2=[0.54, 0.04, 0.41, 0.88]")
        elif num_images == 1:
            # For 1 image, center it
            self.axes[0].set_position([0.3, 0.04, 0.4, 0.88])  # left, bottom, width, height
            print(f"🎯 Centered 1 image: pos=[0.3, 0.04, 0.4, 0.88]")
        # For 3 images, use the default layout (already centered)
    
    def update_image_titles(self, image_states):
        """Update image titles to show selection status instead of crop info."""
        if not image_states:
            return
            
        for i, state in enumerate(image_states):
            if i < len(self.axes) and isinstance(state, dict):
                status = state.get('status', 'delete')
                if status in ['keep', 'selected']:
                    self.axes[i].set_title(f"Image {i + 1}: SELECTED", fontsize=12, color='blue', weight='bold')
                elif status == 'delete':
                    self.axes[i].set_title(f"Image {i + 1}: DELETE", fontsize=12, color='red')
                else:
                    self.axes[i].set_title(f"Image {i + 1}: {status.upper()}", fontsize=12, color='orange')
    
    def clear_selectors(self):
        """Clear previous selectors."""
        for selector in self.selectors:
            if selector:
                selector.set_active(False)
        self.selectors = [None, None, None]
    
    def reset_batch_flags(self):
        """Reset flags for new batch."""
        self.has_pending_changes = False
        self.quit_confirmed = False
    
    # Abstract methods that subclasses must implement
    def submit_batch(self):
        """Submit current batch - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement submit_batch()")
    
    def go_back(self):
        """Go back to previous item - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement go_back()")
    
    def run(self):
        """Main execution loop - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
