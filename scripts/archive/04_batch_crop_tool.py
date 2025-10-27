#!/usr/bin/env python3
"""
Step 4: Enhanced Batch Crop Tool - 3-Image Interactive Cropping
================================================================
Efficient batch processing tool for cropping character images.
Process 3 images simultaneously with individual crop controls.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run on directories containing images:
  python scripts/04_batch_crop_tool.py crop/
  python scripts/04_batch_crop_tool.py face_groups/person_0001/
  python scripts/04_batch_crop_tool.py crop/ --aspect-ratio 16:9

FEATURES:
---------
• Process 3 images at once in side-by-side layout
• Individual crop rectangles for each image
• Intuitive hotkey system (W-S-X, E-D-C, R-F-V)
• Batch submission reduces overhead dramatically
• Large, easy-to-grab handles with generous click zones
• Configurable aspect ratios (1:1, 16:9, 4:3, free)
• Real-time crop preview with zoom capabilities
• Comprehensive progress tracking and statistics

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_web_image_selector.py
Step 2: Face Grouping → scripts/02_face_grouper.py
Step 3: Character Sorting → scripts/03_web_character_sorter.py
Step 4: Final Cropping → THIS SCRIPT (scripts/04_batch_crop_tool.py)
Step 5: Basic Review → scripts/05_multi_directory_viewer.py

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/util_similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

CONTROLS:
---------
Image 1: [W] Skip  [S] Delete  [X] Reset crop
Image 2: [E] Skip  [D] Delete  [C] Reset crop  
Image 3: [R] Skip  [F] Delete  [V] Reset crop

Global: [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Q] Quit

FEATURES:
---------
• 3-image side-by-side layout for batch processing
• Individual RectangleSelector for each image
• Clean outlined crop rectangles (transparent fill, no visual clutter)
• Large outlined handles (48px markers, 120px grab range) for easy interaction
• Dynamic aspect ratio (uses each image's natural ratio) or global override
• Aspect ratio lock/unlock with [Space] - constrains crops to maintain proportions
• Batch submission processes all 3 at once
• FileTracker integration for operation logging
• Automatic PNG + companion file handling
• Progress tracking (Batch X/Y format)
• Significant speed improvement over single-image tool
"""

import argparse
import sys
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib

# Disable toolbar before setting backend
matplotlib.rcParams['toolbar'] = 'None'

# Try to use Qt5Agg first (PyQt5), fall back to Agg
try:
    matplotlib.use('Qt5Agg', force=True)
    print("[*] Using Qt5Agg backend (PyQt5) - full interactivity available")
    backend_interactive = True
except Exception as e:
    print(f"[!] Qt5Agg backend failed: {e}")
    matplotlib.use('Agg', force=True)
    print("[!] Using Agg backend - limited interactivity")
    print("[!] For full interactive features, ensure PyQt5 is properly installed")
    backend_interactive = False

import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image
from send2trash import send2trash

# Add the project root to the path for importing
sys.path.append(str(Path(__file__).parent.parent))
from scripts.file_tracker import FileTracker
from util_activity_timer import ActivityTimer


class BatchCropTool:
    def __init__(self, directory, aspect_ratio=None):
        self.directory = Path(directory)
        self.aspect_ratio = self._parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
        self.aspect_ratio_locked = True
        self.tracker = FileTracker("batch_crop_tool")
        
        # Initialize activity timer
        self.activity_timer = ActivityTimer("04_batch_crop_tool")
        self.activity_timer.start_session()
        
        # Get all PNG files
        self.png_files = sorted([f for f in self.directory.glob("*.png")])
        if not self.png_files:
            raise ValueError(f"No PNG files found in {directory}")
        
        self.current_batch = 0
        self.total_batches = (len(self.png_files) + 2) // 3  # Round up division
        
        # State for current batch of 3 images
        self.current_images = []
        self.image_states = []  # List of dicts with crop coords, action, etc.
        self.selectors = []
        self.axes = []
        
        # Create cropped directory if it doesn't exist
        self.cropped_dir = Path.cwd() / "cropped"
        self.cropped_dir.mkdir(exist_ok=True)
        
        # Setup matplotlib
        self.fig = None
        self.setup_display()
        
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
        # Disable toolbar globally before creating figure
        import matplotlib
        matplotlib.rcParams['toolbar'] = 'None'
        
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
            # Fallback to reasonable defaults when tkinter is not available
            print("[!] tkinter not available, using default screen dimensions")
            screen_width = 1920  # Common default
            screen_height = 1080
        
        # Calculate optimal figure size (use 90% of screen, account for OS bars)
        max_width = screen_width * 0.9 / 100  # Convert pixels to inches (approx 100 DPI)
        max_height = (screen_height * 0.85) / 100  # Leave space for title bars, dock, etc.
        
        # Create figure with optimized dimensions
        self.fig, self.axes = plt.subplots(1, 3, figsize=(max_width, max_height))
        self.fig.suptitle("", fontsize=14, y=0.98)  # Higher title position
        
        # Hide the matplotlib toolbar completely
        try:
            # Multiple approaches to disable toolbar
            self.fig.canvas.toolbar_visible = False
            if hasattr(self.fig.canvas, 'toolbar'):
                self.fig.canvas.toolbar = None
            # Set matplotlib to not show toolbar globally
            import matplotlib
            matplotlib.rcParams['toolbar'] = 'None'
        except:
            pass
        
        # Minimize spacing between subplots to maximize image space - tighter layout
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04, wspace=0.03)
        
        # Disable matplotlib's default key bindings to prevent conflicts
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
            self.fig.canvas.toolbar = None  # Remove toolbar shortcuts
        
        # Disable specific matplotlib shortcuts
        try:
            # Remove matplotlib's default 's' save shortcut
            if hasattr(plt.rcParams, 'keymap.save'):
                plt.rcParams['keymap.save'] = []
            # Remove other potentially conflicting shortcuts
            conflicting_keys = ['keymap.save', 'keymap.quit', 'keymap.fullscreen', 'keymap.grid']
            for key in conflicting_keys:
                if key in plt.rcParams:
                    plt.rcParams[key] = []
        except:
            pass  # Ignore if rcParams keys don't exist
        
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
                
                # Display image optimized for maximum size
                ax = self.axes[i]
                ax.clear()
                
                # Display image with aspect ratio preserved, filling available space
                ax.imshow(img_array, aspect='equal')
                ax.set_title(f"Image {i+1}", fontsize=12)  # Reduced padding
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Remove axis spines to maximize image space
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Tight layout for this subplot
                ax.margins(0)
                
                # Create RectangleSelector with enhanced handles
                selector = RectangleSelector(
                    ax, 
                    lambda eclick, erelease, idx=i: self.on_crop_select(eclick, erelease, idx),
                    useblit=True,
                    button=[1],  # Only left mouse button
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='none', edgecolor='red', linewidth=2),  # Transparent fill with red outline
                    drag_from_anywhere=False,  # Disable rotation/dragging from center
                    use_data_coordinates=False,  # Use pixel coordinates, no rotation
                    grab_range=120,  # Much larger grab range (4x bigger)
                    handle_props=dict(markersize=48, markerfacecolor='none', markeredgecolor='red', markeredgewidth=3)  # Outlined handles only
                )
                
                self.selectors[i] = selector
                
                # Set initial crop selection to full image using actual image dimensions
                # The image is displayed with extents from 0 to width and 0 to height
                selector.extents = (0, img_width, 0, img_height)
                
                # Initialize the crop coordinates for this image (full image)
                self.image_states[i]['crop_coords'] = (0, 0, img_width, img_height)
                self.image_states[i]['has_selection'] = True
                
                # Update the title to show the initial selection
                aspect_str = f" [{image_aspect_ratio:.2f}:1]" if self.aspect_ratio_locked else ""
                self.axes[i].set_title(f"Image {i+1}: {img_width}×{img_height}{aspect_str} [Full Image]", 
                                     fontsize=10, color='green')
                
            except Exception as e:
                print(f"Error loading {png_path}: {e}")
                continue
        
        # Hide unused subplots and maximize space for active ones
        for i in range(len(batch_files), 3):
            self.axes[i].clear()
            self.axes[i].set_visible(False)
            
        # If we have fewer than 3 images, adjust subplot layout for larger display
        if len(batch_files) == 1:
            # Center single image with more space - tighter layout
            self.fig.subplots_adjust(left=0.2, right=0.8, top=0.96, bottom=0.04)
        elif len(batch_files) == 2:
            # Center two images with more space each - tighter layout
            self.fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.04, wspace=0.05)
            
        # Update title with controls in header
        remaining_images = len(self.png_files) - (start_idx)
        lock_str = "🔒" if self.aspect_ratio_locked else "🔓"
        aspect_info = f" • [{lock_str} Space] Aspect Ratio" if self.aspect_ratio else ""
        title = f"Batch {self.current_batch + 1}/{self.total_batches} • {remaining_images} images remaining • [Enter] Submit • [Q] Quit{aspect_info}"
        self.fig.suptitle(title, fontsize=12, y=0.98)
        
        # Update control labels
        self.update_control_labels()
        
        # Force tight layout and redraw for maximum space utilization
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Minimal space reservations
        plt.draw()
        
    def update_control_labels(self):
        """Update the control labels below each image"""
        controls = [
            "[W] Skip  [S] Delete  [X] Reset",
            "[E] Skip  [D] Delete  [C] Reset", 
            "[R] Skip  [F] Delete  [V] Reset"
        ]
        
        for i, (ax, control_text) in enumerate(zip(self.axes, controls)):
            if i < len(self.current_images):
                ax.set_xlabel(control_text, fontsize=10)
                
        # Controls are now in the header - no bottom text needed
        
    def on_crop_select(self, eclick, erelease, image_idx):
        """Handle crop rectangle selection for a specific image"""
        # Mark activity for timer
        self.activity_timer.mark_activity()
        if image_idx >= len(self.current_images):
            return
            
        # Get crop coordinates
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Apply aspect ratio constraint if locked
        # Use global aspect ratio if set, otherwise use this image's natural aspect ratio
        active_aspect_ratio = self.aspect_ratio if self.aspect_ratio else self.image_states[image_idx]['image_aspect_ratio']
        
        if self.aspect_ratio_locked and active_aspect_ratio:
            sel_width = x2 - x1
            sel_height = y2 - y1
            
            if sel_width > 0 and sel_height > 0:
                # Calculate what the dimensions should be to maintain aspect ratio
                target_height_from_width = sel_width / active_aspect_ratio
                target_width_from_height = sel_height * active_aspect_ratio
                
                if target_height_from_width <= sel_height:
                    # Use width as constraint, adjust height
                    new_height = target_height_from_width
                    height_diff = sel_height - new_height
                    y1 += height_diff / 2
                    y2 = y1 + new_height
                else:
                    # Use height as constraint, adjust width
                    new_width = target_width_from_height
                    width_diff = sel_width - new_width
                    x1 += width_diff / 2
                    x2 = x1 + new_width
                
                # Ensure coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Update the selector to reflect the aspect ratio adjustment
                self.selectors[image_idx].extents = (x1, x2, y1, y2)
                plt.draw()  # Force visual update
        
        # Store crop coordinates
        self.image_states[image_idx]['crop_coords'] = (x1, y1, x2, y2)
        self.image_states[image_idx]['has_selection'] = True
        self.image_states[image_idx]['action'] = None  # Reset to crop mode
        
        # Update title to show crop info
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_aspect = crop_width/crop_height if crop_height > 0 else 0
        aspect_str = f" ({crop_aspect:.2f}:1)" if crop_height > 0 else ""
        
        # Show what aspect ratio is being used
        if self.aspect_ratio_locked:
            ratio_source = f"Global {self.aspect_ratio:.2f}:1" if self.aspect_ratio else f"Original {active_aspect_ratio:.2f}:1"
            lock_str = f" [🔒 {ratio_source}]"
        else:
            lock_str = " [🔓 Free]"
        
        self.axes[image_idx].set_title(f"Image {image_idx + 1}: {crop_width}×{crop_height}{aspect_str}{lock_str}", 
                                       fontsize=10, color='green')
        plt.draw()
        
        print(f"Image {image_idx + 1} crop selected: ({x1}, {y1}) to ({x2}, {y2})")
        
    def on_key_press(self, event):
        """Handle keyboard input"""
        # Mark activity for timer
        self.activity_timer.mark_activity()
        
        key = event.key.lower()
        
        # Explicitly disable matplotlib's default save functionality
        if key == 's' and (event.key == 'ctrl+s' or event.key == 'cmd+s'):
            return  # Ignore save shortcuts
        
        # Disable any other matplotlib default shortcuts that might interfere
        if key in ['ctrl+s', 'cmd+s']:
            return
        
        # Global controls
        if key == 'q':  # Q is now quit
            self.quit()
            return
        elif key == 'escape':  # Keep escape as backup quit
            self.quit()
            return
        elif key == 'enter':
            self.submit_batch()
            return
        elif key == ' ':  # Space to toggle aspect ratio lock
            self.aspect_ratio_locked = not self.aspect_ratio_locked
            lock_str = "🔒 locked" if self.aspect_ratio_locked else "🔓 unlocked"
            print(f"Aspect ratio {lock_str}")
            return
            
        # Image-specific controls - WSX EDC RFV layout
        image_actions = {
            'w': (0, 'skip'), 's': (0, 'delete'), 'x': (0, 'reset'),
            'e': (1, 'skip'), 'd': (1, 'delete'), 'c': (1, 'reset'),
            'r': (2, 'skip'), 'f': (2, 'delete'), 'v': (2, 'reset'),
        }
        
        if key in image_actions:
            image_idx, action = image_actions[key]
            
            if image_idx < len(self.current_images):
                if action == 'reset':
                    self.reset_image_crop(image_idx)
                else:
                    self.set_image_action(image_idx, action)
                    
    def reset_image_crop(self, image_idx):
        """Reset the crop selection for a specific image back to full image"""
        if image_idx >= len(self.current_images):
            return
            
        # Get the original image dimensions
        image_info = self.current_images[image_idx]
        img = Image.open(image_info['path'])
        img_width, img_height = img.size
        
        # Reset selector to full image
        if self.selectors[image_idx]:
            self.selectors[image_idx].extents = (0, img_width, 0, img_height)
            
        # Reset state to full image coordinates
        self.image_states[image_idx]['crop_coords'] = (0, 0, img_width, img_height)
        self.image_states[image_idx]['has_selection'] = True
        self.image_states[image_idx]['action'] = None
        
        # Update title to show reset to full image
        image_aspect_ratio = self.image_states[image_idx]['image_aspect_ratio']
        aspect_str = f" [{image_aspect_ratio:.2f}:1]" if self.aspect_ratio_locked else ""
        self.axes[image_idx].set_title(f"Image {image_idx + 1}: {img_width}×{img_height}{aspect_str} [RESET TO FULL]", 
                                       fontsize=10, color='green')
        plt.draw()
        
        print(f"Image {image_idx + 1} crop reset to full image ({img_width}×{img_height})")
        
    def set_image_action(self, image_idx, action):
        """Set action (skip/delete) for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        self.image_states[image_idx]['action'] = action
        
        # Update visual feedback
        color = 'orange' if action == 'skip' else 'red'
        action_text = action.upper()
        
        self.axes[image_idx].set_title(f"Image {image_idx + 1} [{action_text}]", 
                                       fontsize=12, color=color)
        plt.draw()
        
        print(f"Image {image_idx + 1} marked for {action}")
        
    def submit_batch(self):
        """Process all images in the current batch"""
        if not self.current_images:
            return
            
        # Mark batch processing activity
        self.activity_timer.mark_batch(f"Crop batch {self.current_batch + 1}")
        self.activity_timer.mark_activity()
            
        print(f"\nProcessing batch {self.current_batch + 1}...")
        
        processed_count = 0
        
        for i, (image_info, state) in enumerate(zip(self.current_images, self.image_states)):
            png_path = image_info['path']
            yaml_path = png_path.with_suffix('.yaml')
            
            try:
                if state['action'] == 'skip':
                    # Move to cropped directory without cropping
                    self.move_to_cropped(png_path, yaml_path, "skipped")
                    processed_count += 1
                    
                elif state['action'] == 'delete':
                    # Delete the image
                    self.safe_delete(png_path, yaml_path)
                    processed_count += 1
                    
                elif state['has_selection'] and state['crop_coords']:
                    # Crop and save
                    self.crop_and_save(image_info, state['crop_coords'])
                    processed_count += 1
                    
                else:
                    print(f"Image {i + 1}: No action specified, skipping...")
                    
            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
                
        print(f"Processed {processed_count}/{len(self.current_images)} images in batch")
        
        # End batch tracking
        self.activity_timer.end_batch(f"Completed {processed_count} operations")
        
        # Move to next batch
        if self.has_next_batch():
            self.current_batch += 1
            self.load_batch()
        else:
            self.show_completion()
            
    def crop_and_save(self, image_info, crop_coords):
        """Crop image and save to cropped directory"""
        png_path = image_info['path']
        yaml_path = png_path.with_suffix('.yaml')
        
        x1, y1, x2, y2 = crop_coords
        
        # Load and crop image
        img = Image.open(png_path)
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Save to cropped directory
        output_png = self.cropped_dir / png_path.name
        output_yaml = self.cropped_dir / yaml_path.name
        
        cropped_img.save(output_png)
        
        # Copy YAML file if it exists
        if yaml_path.exists():
            shutil.copy2(yaml_path, output_yaml)
            
        # Log the operation
        self.tracker.log_operation(
            "crop", str(png_path.parent), "cropped", 2,
            f"Cropped to ({x1},{y1},{x2},{y2})", 
            [png_path.name, yaml_path.name]
        )
        
        # Log operation in activity timer
        self.activity_timer.log_operation("crop", file_count=1)
        
        # Delete original files
        png_path.unlink()
        if yaml_path.exists():
            yaml_path.unlink()
            
        print(f"Cropped and saved: {png_path.name}")
        
    def move_to_cropped(self, png_path, yaml_path, reason):
        """Move files to cropped directory without modification"""
        output_png = self.cropped_dir / png_path.name
        output_yaml = self.cropped_dir / yaml_path.name
        
        # Move files
        shutil.move(str(png_path), str(output_png))
        if yaml_path.exists():
            shutil.move(str(yaml_path), str(output_yaml))
            
        # Log the operation
        file_count = 2 if yaml_path.exists() else 1
        files = [png_path.name, yaml_path.name] if yaml_path.exists() else [png_path.name]
        
        self.tracker.log_operation(
            "move", str(png_path.parent), "cropped", file_count,
            f"Image {reason}", files
        )
        
        # Log operation in activity timer
        self.activity_timer.log_operation("skip", file_count=1)
        
        print(f"Moved to cropped: {png_path.name} ({reason})")
        
    def safe_delete(self, png_path, yaml_path):
        """Safely delete image files"""
        files_deleted = []
        
        # Delete PNG
        send2trash(str(png_path))
        files_deleted.append(png_path.name)
        
        # Delete YAML if exists
        if yaml_path.exists():
            send2trash(str(yaml_path))
            files_deleted.append(yaml_path.name)
            
        # Log the operation
        self.tracker.log_operation(
            "delete", str(png_path.parent), "trash", len(files_deleted),
            "Image deleted", files_deleted
        )
        
        # Log operation in activity timer
        self.activity_timer.log_operation("delete", file_count=1)
        
        print(f"Deleted: {png_path.name}")
        
    def has_next_batch(self):
        """Check if there are more batches to process"""
        next_start = (self.current_batch + 1) * 3
        return next_start < len(self.png_files)
        
    def show_completion(self):
        """Show completion message"""
        plt.clf()
        plt.text(0.5, 0.5, "🎉 All images processed!\n\nBatch cropping complete.", 
                 ha='center', va='center', fontsize=20, 
                 bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen"))
        plt.axis('off')
        plt.title("Batch Crop Tool - Complete", fontsize=16)
        plt.draw()
        
        print("\n🎉 All images processed! Batch cropping complete.")
        
    def quit(self):
        """Quit the application"""
        print("Quitting batch crop tool...")
        
        # End activity timer session
        self.activity_timer.end_session()
        
        plt.close('all')
        sys.exit(0)
        
    def run(self):
        """Main execution loop"""
        print(f"Starting batch crop tool on {self.directory}")
        print(f"Found {len(self.png_files)} images")
        print(f"Will process in {self.total_batches} batches of up to 3 images each")
        print("\nControls:")
        print("  Image 1: [W] Skip  [S] Delete  [X] Reset")
        print("  Image 2: [E] Skip  [D] Delete  [C] Reset") 
        print("  Image 3: [R] Skip  [F] Delete  [V] Reset")
        print("  Global:  [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Q] Quit")
        
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

def parse_aspect_ratio(ratio_str):
    """Parse aspect ratio string like '16:9' into float"""
    if not ratio_str:
        return None
    try:
        w, h = map(float, ratio_str.split(':'))
        return w / h
    except:
        raise ValueError(f"Invalid aspect ratio format: {ratio_str}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Batch Crop Tool - Process 3 images at once")
    parser.add_argument("directory", help="Directory containing PNG images to crop")
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
        tool = BatchCropTool(directory, args.aspect_ratio)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
