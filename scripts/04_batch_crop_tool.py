#!/usr/bin/env python3
"""
Enhanced Batch Crop Tool - 3-Image Interactive Cropping
======================================================
A batch-processing version of the crop tool for maximum efficiency:
- Process 3 images at once in side-by-side layout
- Individual crop rectangles for each image
- Intuitive hotkey system (1-Q-A, 2-W-S, 3-E-D)
- Batch submission reduces overhead dramatically
- Maintains precision with improved matplotlib handles

USAGE:
------
Activate virtual environment first:
  source venv/bin/activate

Run on directories containing images:
  python scripts/04_batch_crop_tool.py crop/
  python scripts/04_batch_crop_tool.py face_group_1
  python scripts/04_batch_crop_tool.py crop/ --aspect-ratio 16:9

CONTROLS:
---------
Image 1: [1] Skip  [Q] Delete  [A] Reset crop
Image 2: [2] Skip  [W] Delete  [S] Reset crop  
Image 3: [3] Skip  [E] Delete  [D] Reset crop

Global: [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Esc] Quit

FEATURES:
---------
• 3-image side-by-side layout for batch processing
• Individual RectangleSelector for each image
• Enhanced handles (24px markers, 30px grab range)
• Dynamic aspect ratio (uses each image's natural ratio) or global override
• Aspect ratio lock/unlock with [Space] - constrains crops to maintain proportions
• Batch submission processes all 3 at once
• FileTracker integration for operation logging
• Automatic PNG+YAML file handling
• Progress tracking (Batch X/Y format)
• Significant speed improvement over single-image tool
"""

import argparse
import sys
from pathlib import Path
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

class BatchCropTool:
    def __init__(self, directory, aspect_ratio=None):
        self.directory = Path(directory)
        self.aspect_ratio = self._parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
        self.aspect_ratio_locked = True
        self.tracker = FileTracker("batch_crop_tool")
        
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
        if self.fig:
            plt.close(self.fig)
            
        # Get screen dimensions and calculate optimal figure size
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Calculate optimal figure size (use 90% of screen, account for OS bars)
        max_width = screen_width * 0.9 / 100  # Convert pixels to inches (approx 100 DPI)
        max_height = (screen_height * 0.85) / 100  # Leave space for title bars, dock, etc.
        
        # Create figure with optimized dimensions
        self.fig, self.axes = plt.subplots(1, 3, figsize=(max_width, max_height))
        self.fig.suptitle("", fontsize=14, y=0.98)  # Higher title position
        
        # Minimize spacing between subplots to maximize image space
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.05)
        
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
                ax.set_title(f"Image {i+1}", fontsize=12, pad=5)  # Reduced padding
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
                    props=dict(facecolor='red', alpha=0.2),
                    rectprops=dict(facecolor='red', alpha=0.2),
                    handle_props=dict(markersize=24),  # Larger handles
                    grab_range=30  # Larger grab range
                )
                
                self.selectors[i] = selector
                
            except Exception as e:
                print(f"Error loading {png_path}: {e}")
                continue
        
        # Hide unused subplots and maximize space for active ones
        for i in range(len(batch_files), 3):
            self.axes[i].clear()
            self.axes[i].set_visible(False)
            
        # If we have fewer than 3 images, adjust subplot layout for larger display
        if len(batch_files) == 1:
            # Center single image with more space
            self.fig.subplots_adjust(left=0.25, right=0.75, top=0.92, bottom=0.08)
        elif len(batch_files) == 2:
            # Center two images with more space each
            self.fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.08, wspace=0.1)
            
        # Update title
        remaining_images = len(self.png_files) - (start_idx)
        title = f"Batch {self.current_batch + 1}/{self.total_batches} • {remaining_images} images remaining"
        self.fig.suptitle(title, fontsize=14, y=0.95)
        
        # Update control labels
        self.update_control_labels()
        
        # Force tight layout and redraw for maximum space utilization
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for controls
        plt.draw()
        
    def update_control_labels(self):
        """Update the control labels below each image"""
        controls = [
            "[1] Skip  [Q] Delete  [A] Reset",
            "[2] Skip  [W] Delete  [S] Reset", 
            "[3] Skip  [E] Delete  [D] Reset"
        ]
        
        for i, (ax, control_text) in enumerate(zip(self.axes, controls)):
            if i < len(self.current_images):
                ax.set_xlabel(control_text, fontsize=10, pad=5)
                
        # Add global controls at bottom
        lock_str = "🔒" if self.aspect_ratio_locked else "🔓"
        aspect_info = f" • [{lock_str} Space] Aspect Ratio" if self.aspect_ratio else ""
        self.fig.text(0.5, 0.02, f"[Enter] Submit Batch  •  [Esc] Quit{aspect_info}", 
                     ha='center', fontsize=12, weight='bold')
        
    def on_crop_select(self, eclick, erelease, image_idx):
        """Handle crop rectangle selection for a specific image"""
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
                                       fontsize=10, color='green', pad=10)
        plt.draw()
        
        print(f"Image {image_idx + 1} crop selected: ({x1}, {y1}) to ({x2}, {y2})")
        
    def on_key_press(self, event):
        """Handle keyboard input"""
        key = event.key.lower()
        
        # Global controls
        if key == 'escape':
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
            
        # Image-specific controls
        image_actions = {
            '1': (0, 'skip'), 'q': (0, 'delete'), 'a': (0, 'reset'),
            '2': (1, 'skip'), 'w': (1, 'delete'), 's': (1, 'reset'),
            '3': (2, 'skip'), 'e': (2, 'delete'), 'd': (2, 'reset'),
        }
        
        if key in image_actions:
            image_idx, action = image_actions[key]
            
            if image_idx < len(self.current_images):
                if action == 'reset':
                    self.reset_image_crop(image_idx)
                else:
                    self.set_image_action(image_idx, action)
                    
    def reset_image_crop(self, image_idx):
        """Reset the crop selection for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        # Clear the selection
        if self.selectors[image_idx]:
            # Clear the rectangle by setting it to empty
            self.selectors[image_idx].set_visible(False)
            self.selectors[image_idx].set_visible(True)
            
        # Reset state
        self.image_states[image_idx]['crop_coords'] = None
        self.image_states[image_idx]['has_selection'] = False
        self.image_states[image_idx]['action'] = None
        
        # Update title to show reset
        self.axes[image_idx].set_title(f"Image {image_idx + 1} [RESET]", 
                                       fontsize=12, color='blue', pad=10)
        plt.draw()
        
        print(f"Image {image_idx + 1} crop reset")
        
    def set_image_action(self, image_idx, action):
        """Set action (skip/delete) for a specific image"""
        if image_idx >= len(self.current_images):
            return
            
        self.image_states[image_idx]['action'] = action
        
        # Update visual feedback
        color = 'orange' if action == 'skip' else 'red'
        action_text = action.upper()
        
        self.axes[image_idx].set_title(f"Image {image_idx + 1} [{action_text}]", 
                                       fontsize=12, color=color, pad=10)
        plt.draw()
        
        print(f"Image {image_idx + 1} marked for {action}")
        
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
        plt.title("Batch Crop Tool - Complete", fontsize=16, pad=20)
        plt.draw()
        
        print("\n🎉 All images processed! Batch cropping complete.")
        
    def quit(self):
        """Quit the application"""
        print("Quitting batch crop tool...")
        plt.close('all')
        sys.exit(0)
        
    def run(self):
        """Main execution loop"""
        print(f"Starting batch crop tool on {self.directory}")
        print(f"Found {len(self.png_files)} images")
        print(f"Will process in {self.total_batches} batches of up to 3 images each")
        print("\nControls:")
        print("  Image 1: [1] Skip  [Q] Delete  [A] Reset")
        print("  Image 2: [2] Skip  [W] Delete  [S] Reset") 
        print("  Image 3: [3] Skip  [E] Delete  [D] Reset")
        print("  Global:  [Enter] Submit Batch  [Space] Toggle Aspect Ratio  [Esc] Quit")
        
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
