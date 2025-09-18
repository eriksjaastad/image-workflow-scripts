#!/usr/bin/env python3
"""
Advanced Crop Tool - Interactive Image Cropping with Aspect Ratio Control
========================================================================
A better alternative to Preview's crop tool with these features:
- Click and drag to select crop area
- Maintains aspect ratio (or allows free cropping)
- Enter to crop and save
- Automatically moves to next image
- Shows preview of crop area in real-time
- Saves original aspect ratio by default
- Moves PNG+YAML files together to cropped/ directory
- Deletes original files after successful cropping

USAGE:
------
Activate virtual environment first:
  source venv/bin/activate

Run on directories containing images:
  python scripts/05_crop_tool.py crop/
  python scripts/05_crop_tool.py face_group_1
  python scripts/05_crop_tool.py face_group_5 --aspect-ratio 16:9

CONTROLS:
---------
• Click and drag to select crop area
• Enter = Crop and save to cropped/, delete original
• r = Reset crop selection
• s = Skip image and move to cropped/ (no cropping)
• d = Delete image (move to trash)
• q = Quit
• Space = Toggle aspect ratio lock on/off
• + or = = Zoom in
• - = Zoom out
• 0 = Reset zoom to fit image

FEATURES:
---------
• Real-time crop preview with red rectangle
• Maintains original aspect ratio by default
• Option to set custom aspect ratios (16:9, 4:3, 1:1, etc.)
• Cropped images saved to cropped/ directory (at project root)
• Skipped images moved to cropped/ directory (at project root)
• Deletes original PNG+YAML files after successful cropping
• Moves PNG+YAML files for skipped images
• Full file operation tracking via FileTracker
• Shows progress and remaining count
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
from file_tracker import FileTracker

# Required imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import RectangleSelector
    from PIL import Image
    import numpy as np
except ImportError as e:
    print("❌ Missing required packages. Install with:")
    print("pip install matplotlib pillow numpy")
    print(f"Error: {e}")
    sys.exit(1)

class InteractiveCropTool:
    def __init__(self, aspect_ratio: Optional[str] = None):
        self.image = None
        self.fig = None
        self.ax = None
        self.selector = None
        self.crop_coords = None
        self.current_image_path = None
        self.aspect_ratio_locked = True
        self.aspect_ratio = self._parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
        self.result = {"action": None}
        self.zoom_level = 1.0
        self.original_xlim = None
        self.original_ylim = None
        
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
    
    def zoom_in(self, factor: float = 1.5):
        """Zoom in by the specified factor."""
        if not self.ax:
            return
            
        # Get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate center point
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        
        # Calculate new zoom range
        current_width = xlim[1] - xlim[0]
        current_height = ylim[1] - ylim[0]
        new_width = current_width / factor
        new_height = current_height / factor
        
        # Set new limits centered on the same point
        self.ax.set_xlim(center_x - new_width/2, center_x + new_width/2)
        self.ax.set_ylim(center_y - new_height/2, center_y + new_height/2)
        
        self.zoom_level *= factor
        self._update_zoom_title()
        self.fig.canvas.draw()
    
    def zoom_out(self, factor: float = 1.5):
        """Zoom out by the specified factor."""
        if not self.ax:
            return
            
        # Get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate center point
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        
        # Calculate new zoom range
        current_width = xlim[1] - xlim[0]
        current_height = ylim[1] - ylim[0]
        new_width = current_width * factor
        new_height = current_height * factor
        
        # Don't zoom out beyond original image bounds
        if self.original_xlim and self.original_ylim:
            max_width = self.original_xlim[1] - self.original_xlim[0]
            max_height = self.original_ylim[1] - self.original_ylim[0]
            
            if new_width > max_width * 1.1:  # Allow slight overzoom
                new_width = max_width * 1.1
            if new_height > max_height * 1.1:
                new_height = max_height * 1.1
        
        # Set new limits centered on the same point
        self.ax.set_xlim(center_x - new_width/2, center_x + new_width/2)
        self.ax.set_ylim(center_y - new_height/2, center_y + new_height/2)
        
        self.zoom_level /= factor
        self._update_zoom_title()
        self.fig.canvas.draw()
    
    def reset_zoom(self):
        """Reset zoom to show full image."""
        if not self.ax or not self.original_xlim or not self.original_ylim:
            return
            
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.zoom_level = 1.0
        self._update_zoom_title()
        self.fig.canvas.draw()
    
    def _update_zoom_title(self):
        """Update the title to show current zoom level."""
        zoom_text = f" [Zoom: {self.zoom_level:.1f}x]"
        current_title = self.ax.get_title()
        
        # Remove existing zoom text if present
        if "[Zoom:" in current_title:
            current_title = current_title.split("[Zoom:")[0].strip()
        
        self.ax.set_title(current_title + zoom_text, fontsize=10, pad=10)
    
    def load_image(self, image_path: Path) -> bool:
        """Load image and set up the display."""
        try:
            self.current_image_path = image_path
            self.image = Image.open(image_path)
            
            # Convert to RGB if needed
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')
            
            # Use original aspect ratio if none specified
            if self.aspect_ratio is None:
                width, height = self.image.size
                self.aspect_ratio = width / height
            
            return True
        except Exception as e:
            print(f"❌ Error loading {image_path.name}: {e}")
            return False
    
    def onselect(self, eclick, erelease):
        """Handle rectangle selection."""
        # Get selection coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Ensure coordinates are valid
        if None in (x1, y1, x2, y2):
            return
            
        # Make sure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Apply aspect ratio constraint if locked
        if self.aspect_ratio_locked and self.aspect_ratio:
            # Calculate current selection dimensions
            sel_width = x2 - x1
            sel_height = y2 - y1
            
            if sel_width > 0 and sel_height > 0:
                # Calculate what the dimensions should be to maintain aspect ratio
                # We'll use the selection that gives us the largest area while maintaining ratio
                target_height_from_width = sel_width / self.aspect_ratio
                target_width_from_height = sel_height * self.aspect_ratio
                
                if target_height_from_width <= sel_height:
                    # Use width as constraint, adjust height
                    new_height = target_height_from_width
                    height_diff = sel_height - new_height
                    y1 += height_diff / 2
                    y2 -= height_diff / 2
                else:
                    # Use height as constraint, adjust width
                    new_width = target_width_from_height
                    width_diff = sel_width - new_width
                    x1 += width_diff / 2
                    x2 -= width_diff / 2
                
                # Update the selector with the corrected coordinates
                self.selector.extents = (x1, x2, y1, y2)
        
        # Store crop coordinates (convert to image pixel coordinates)
        img_height, img_width = np.array(self.image).shape[:2]
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))
        
        self.crop_coords = (int(x1), int(y1), int(x2), int(y2))
        
        # Update title to show crop dimensions
        if self.crop_coords:
            crop_width = self.crop_coords[2] - self.crop_coords[0]
            crop_height = self.crop_coords[3] - self.crop_coords[1]
            aspect_str = f" (AR: {crop_width/crop_height:.2f})" if crop_height > 0 else ""
            lock_str = "[LOCKED]" if self.aspect_ratio_locked else "[UNLOCKED]"
            self.ax.set_title(f"Crop: {crop_width}×{crop_height}{aspect_str} {lock_str} | Enter=crop | r=reset | s=skip | Space=toggle lock | q=quit", 
                            fontsize=10, pad=10)
        
        self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard input."""
        key = event.key.lower() if event.key else ""
        
        if key == 'enter':
            if self.crop_coords:
                self.result["action"] = "crop"
            else:
                print("⚠️  No crop area selected. Draw a selection first.")
                return
        elif key == 'r':
            # Reset crop to 100% (full image)
            if self.image:
                img_height, img_width = self.image.size[1], self.image.size[0]
                self.crop_coords = (0, 0, img_width, img_height)
                self.selector.extents = (0, img_width, 0, img_height)
                lock_str = "[LOCKED]" if self.aspect_ratio_locked else "[UNLOCKED]"
                self.ax.set_title(f"Crop: {img_width}×{img_height} (100% - Full Image) {lock_str} | Drag to resize | Enter=crop | r=reset | s=skip | d=delete | Space=toggle lock | +=zoom in | -=zoom out | 0=reset zoom | q=quit", 
                                fontsize=10, pad=10)
                self.fig.canvas.draw()
            return
        elif key == 's':
            self.result["action"] = "skip"
        elif key == 'd':
            self.result["action"] = "delete"
        elif key == 'q':
            self.result["action"] = "quit"
        elif key == ' ':  # Space to toggle aspect ratio lock
            self.aspect_ratio_locked = not self.aspect_ratio_locked
            lock_str = "🔒 locked" if self.aspect_ratio_locked else "🔓 unlocked"
            print(f"Aspect ratio {lock_str}")
            # Update title if we have a selection
            if self.crop_coords:
                crop_width = self.crop_coords[2] - self.crop_coords[0]
                crop_height = self.crop_coords[3] - self.crop_coords[1]
                aspect_str = f" (AR: {crop_width/crop_height:.2f})" if crop_height > 0 else ""
                lock_str = "[LOCKED]" if self.aspect_ratio_locked else "[UNLOCKED]"
                self.ax.set_title(f"Crop: {crop_width}×{crop_height}{aspect_str} {lock_str} | Enter=crop | r=reset | s=skip | d=delete | Space=toggle lock | +=zoom in | -=zoom out | 0=reset zoom | q=quit", 
                                fontsize=10, pad=10)
                self.fig.canvas.draw()
            return
        elif key in ['+', '=']:  # Zoom in (+ or = key)
            self.zoom_in()
            return
        elif key == '-':  # Zoom out
            self.zoom_out()
            return
        elif key == '0':  # Reset zoom
            self.reset_zoom()
            return
        else:
            return
        
        plt.close(self.fig)
    
    def show_image(self, progress_text: str = "") -> str:
        """Display image with interactive crop selector."""
        self.result = {"action": None}
        
        # Calculate optimal figure size based on image aspect ratio
        img_width, img_height = self.image.size
        img_aspect = img_width / img_height
        
        # Target size - make it at least as big as pair comparator (16x10) but fit screen
        # Account for macOS Dock at bottom, so reduce height significantly
        min_width = 16
        min_height = 8
        max_width = 18
        max_height = 9
        
        # Calculate figure size maintaining image proportions
        if img_aspect > 1:  # Landscape
            fig_width = min(max_width, max(min_width, min_height * img_aspect))
            fig_height = fig_width / img_aspect
            if fig_height < min_height:
                fig_height = min_height
                fig_width = fig_height * img_aspect
        else:  # Portrait or square
            fig_height = min(max_height, max(min_height, min_width / img_aspect))
            fig_width = fig_height * img_aspect
            if fig_width < min_width:
                fig_width = min_width
                fig_height = fig_width / img_aspect
        
        # Ensure we don't exceed maximum bounds
        fig_width = min(fig_width, max_width)
        fig_height = min(fig_height, max_height)
        
        # Create figure with calculated size
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        self.ax.imshow(self.image)
        
        # Store original zoom limits for reset
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.zoom_level = 1.0
        
        # Set initial crop coordinates to full image (100%)
        img_height, img_width = np.array(self.image).shape[:2]
        self.crop_coords = (0, 0, img_width, img_height)
        
        # Optimize layout for better image display
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        
        # Update title to show initial crop info
        lock_str = "[LOCKED]" if self.aspect_ratio_locked else "[UNLOCKED]"
        self.ax.set_title(f"Crop: {img_width}×{img_height} (100% - Full Image) {lock_str} | Drag to resize | Enter=crop | r=reset | s=skip | d=delete | Space=toggle lock | +=zoom in | -=zoom out | 0=reset zoom | q=quit {progress_text}", 
                         fontsize=10, pad=10)
        
        # Create rectangle selector with interactive controls and bigger handles
        self.selector = RectangleSelector(
            self.ax, 
            self.onselect,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, 
            minspany=5,
            spancoords='pixels',
            interactive=True,  # Enable interactive controls for resizing
            grab_range=30,  # Increase click tolerance for handles (default is 10)
            props=dict(facecolor='red', alpha=0.2),
            # Make handles bigger and more visible for easier clicking
            handle_props=dict(markersize=24, markerfacecolor='red', markeredgecolor='red', markeredgewidth=1)
        )
        
        # Set initial selection to full image
        self.selector.extents = (0, img_width, 0, img_height)
        
        # Disable matplotlib's default key shortcuts to prevent save dialog
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        
        # Connect our custom keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        return self.result.get("action", "skip")
    
    def move_to_cropped(self, tracker: FileTracker = None) -> bool:
        """Move skipped image and YAML to cropped directory."""
        if not self.current_image_path:
            return False
        
        try:
            # Always move to cropped directory (at project root level)
            project_root = Path.cwd()  # Should be /Users/eriksjaastad/projects/Eros Mate
            cropped_dir = project_root / "cropped"
            cropped_dir.mkdir(exist_ok=True)
            
            # Move PNG file to cropped directory
            png_target = cropped_dir / self.current_image_path.name
            shutil.move(str(self.current_image_path), str(png_target))
            
            # Move corresponding YAML file if it exists
            yaml_path = self.current_image_path.parent / f"{self.current_image_path.stem}.yaml"
            yaml_moved = False
            if yaml_path.exists():
                yaml_target = cropped_dir / yaml_path.name
                shutil.move(str(yaml_path), str(yaml_target))
                yaml_moved = True
            
            # Track the move operation
            if tracker:
                files_moved = [self.current_image_path.name]
                if yaml_moved:
                    files_moved.append(yaml_path.name)
                tracker.log_operation(
                    operation="move",
                    source_dir=str(self.current_image_path.parent.name),
                    dest_dir="cropped",
                    file_count=len(files_moved),
                    files=files_moved,
                    notes=f"Skipped image moved to cropped directory"
                )
            
            # Success message
            files_processed = "PNG" + (" + YAML" if yaml_moved else "")
            print(f"✓ Moved {files_processed} to cropped/: {self.current_image_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error moving {self.current_image_path.name}: {e}")
            return False

    def perform_crop(self, tracker: FileTracker = None) -> bool:
        """Perform the actual crop and save to cropped directory, then delete original."""
        if not self.crop_coords or not self.current_image_path:
            return False
        
        try:
            # Crop the image
            x1, y1, x2, y2 = self.crop_coords
            cropped_image = self.image.crop((x1, y1, x2, y2))
            
            # Always save to cropped directory (at project root level)
            project_root = Path.cwd()  # Should be /Users/eriksjaastad/projects/Eros Mate
            cropped_dir = project_root / "cropped"
            cropped_dir.mkdir(exist_ok=True)
            
            # Save cropped image to cropped directory
            output_path = cropped_dir / self.current_image_path.name
            cropped_image.save(output_path, quality=95)
            
            # Move corresponding YAML file if it exists
            yaml_path = self.current_image_path.parent / f"{self.current_image_path.stem}.yaml"
            yaml_moved = False
            if yaml_path.exists():
                yaml_output = cropped_dir / yaml_path.name
                shutil.move(str(yaml_path), str(yaml_output))
                yaml_moved = True
            
            # Track the crop operation
            if tracker:
                files_moved = [self.current_image_path.name]
                if yaml_moved:
                    files_moved.append(yaml_path.name)
                tracker.log_operation(
                    operation="crop_and_move",
                    source_dir=str(self.current_image_path.parent.name),
                    dest_dir="cropped",
                    file_count=len(files_moved),
                    files=files_moved,
                    notes=f"Cropped {self.current_image_path.name} and moved to cropped directory"
                )
            
            # Delete the original PNG file
            original_deleted = False
            try:
                self.current_image_path.unlink()  # Delete original PNG
                original_deleted = True
                if tracker:
                    tracker.log_operation(
                        operation="delete",
                        source_dir=str(self.current_image_path.parent.name),
                        file_count=1,
                        files=[self.current_image_path.name],
                        notes="Original file deleted after cropping"
                    )
            except Exception as e:
                print(f"⚠️  Warning: Could not delete original PNG {self.current_image_path.name}: {e}")
            
            # Success message
            files_processed = "PNG" + (" + YAML" if yaml_moved else "")
            print(f"✓ Cropped and saved {files_processed} to cropped/: {self.current_image_path.name}")
            if original_deleted:
                print(f"✓ Deleted original: {self.current_image_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cropping {self.current_image_path.name}: {e}")
            return False
    
    def delete_image(self, use_trash: bool = True, tracker: FileTracker = None) -> bool:
        """Delete the current image and its YAML file."""
        if not self.current_image_path:
            return False
        
        try:
            # Import here to avoid global import issues
            from send2trash import send2trash
            
            base_name = self.current_image_path.stem
            yaml_path = self.current_image_path.parent / f"{base_name}.yaml"
            
            files_to_delete = [self.current_image_path]
            if yaml_path.exists():
                files_to_delete.append(yaml_path)
            
            deleted_files = []
            for file_path in files_to_delete:
                if use_trash:
                    send2trash(str(file_path))
                    print(f"🗑️  Sent to trash: {file_path.name}")
                    deleted_files.append(file_path.name)
                else:
                    file_path.unlink()
                    print(f"❌ Deleted: {file_path.name}")
                    deleted_files.append(file_path.name)
            
            # Log the deletion operation
            if tracker and deleted_files:
                operation_type = "delete" if not use_trash else "send_to_trash"
                tracker.log_operation(
                    operation=operation_type,
                    source_dir=str(self.current_image_path.parent.name),
                    file_count=len(deleted_files),
                    files=deleted_files,
                    notes="User chose to delete via crop tool"
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting {self.current_image_path.name}: {e}")
            return False


def get_image_files(directory: Path) -> List[Path]:
    """Get all PNG files from directory."""
    return sorted(directory.glob("*.png"))


def find_project_root() -> Path:
    """Find the project root directory (containing venv/ and scripts/)."""
    current_dir = Path.cwd()
    while current_dir.parent != current_dir:  # Not at filesystem root
        if (current_dir / "venv").exists() and (current_dir / "scripts").exists():
            return current_dir
        current_dir = current_dir.parent
    return Path.cwd()  # Fallback to current directory


def main():
    parser = argparse.ArgumentParser(description="Interactive crop tool with aspect ratio control")
    parser.add_argument("directory", help="Directory containing images to crop")
    parser.add_argument("--aspect-ratio", help="Target aspect ratio (e.g., '16:9', '4:3', '1:1')")
    
    args = parser.parse_args()
    
    # Initialize file tracker
    tracker = FileTracker("crop_tool")
    
    # Validate source directory
    source_dir = Path(args.directory)
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"❌ Directory not found: {source_dir}")
        sys.exit(1)
    
    # Note: Cropped files will always be saved to cropped/ directory at project root
    
    # Get image files
    image_files = get_image_files(source_dir)
    if not image_files:
        print(f"❌ No PNG files found in {source_dir}")
        sys.exit(1)
    
    print(f"🖼️  Found {len(image_files)} images in {source_dir}")
    print(f"📁 Output directory: cropped/ (at project root)")
    if args.aspect_ratio:
        print(f"📐 Target aspect ratio: {args.aspect_ratio}")
    else:
        print(f"📐 Using original image aspect ratios")
    print()
    print("Controls:")
    print("  • Click and drag to select crop area")
    print("  • Enter = Crop and save, move to next")
    print("  • r = Reset selection")
    print("  • s = Skip image")
    print("  • d = Delete image (move to trash)")
    print("  • Space = Toggle aspect ratio lock")
    print("  • + or = = Zoom in")
    print("  • - = Zoom out")
    print("  • 0 = Reset zoom to fit image")
    print("  • q = Quit")
    print()
    
    # Initialize crop tool
    crop_tool = InteractiveCropTool(aspect_ratio=args.aspect_ratio)
    
    # Process images
    index = 0
    cropped_count = 0
    skipped_count = 0
    deleted_count = 0
    
    while index < len(image_files):
        current_file = image_files[index]
        remaining = len(image_files) - index
        progress_text = f"({index + 1}/{len(image_files)} • {remaining} remaining)"
        
        print(f"\n=== Image {index + 1}/{len(image_files)}: {current_file.name} ===")
        
        # Load image
        if not crop_tool.load_image(current_file):
            index += 1
            continue
        
        # Show interactive crop interface
        action = crop_tool.show_image(progress_text)
        
        if action == "crop":
            if crop_tool.perform_crop(tracker):
                cropped_count += 1
            index += 1
        elif action == "skip":
            if crop_tool.move_to_cropped(tracker):
                print(f"⏭️  Skipped and moved to cropped/: {current_file.name}")
                skipped_count += 1
            else:
                print(f"❌ Failed to move skipped file: {current_file.name}")
            index += 1
            
            # Check if only one file remains after skip
            if len(image_files) - index == 1:
                print(f"\n🎯 Only one image remaining! Auto-loading final image...")
                
        elif action == "delete":
            if crop_tool.delete_image(use_trash=True, tracker=tracker):
                deleted_count += 1
                # Remove from list so we don't process it again
                image_files.remove(current_file)
                # Don't increment index - next image moves into current position
                
                # Check if only one file remains after delete
                if len(image_files) - index == 1:
                    print(f"\n🎯 Only one image remaining! Auto-loading final image...")
                    
            else:
                index += 1  # Move on if delete failed
        elif action == "quit":
            print("👋 Quitting crop tool...")
            break
        else:
            # Default to skip if unknown action
            index += 1
    
    # Summary
    print(f"\n✅ Crop session complete!")
    print(f"📊 Summary:")
    print(f"   • Images cropped: {cropped_count}")
    print(f"   • Images skipped: {skipped_count}")
    print(f"   • Images deleted: {deleted_count}")
    print(f"   • Cropped files saved to: cropped/")


if __name__ == "__main__":
    main()
