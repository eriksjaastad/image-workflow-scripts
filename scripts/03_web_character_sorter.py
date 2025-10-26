#!/usr/bin/env python3
"""
Step 3: Web Character Sorter - Modern Browser Edition
======================================================
Interactive web tool for sorting images into character groups.
Features auto-advance between person directories and similarity-based layout.

🎨 STYLE GUIDE:
---------------
This web interface follows the project style guide for consistent design:
  📁 WEB_STYLE_GUIDE.md
Colors, spacing, typography, and interaction patterns are defined there.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

OPTIONAL FLAGS:
---------------
  --similarity-map  Directory with similarity map files for intelligent layout
  --hard-delete     Permanently delete instead of using trash
  --host/--port     Web server binding (default: 127.0.0.1:5000)
  --no-browser      Don't auto-launch browser

USAGE:
------
Multi-directory character sorting (RECOMMENDED):
  python scripts/03_web_character_sorter.py selected
  # Automatically processes all subdirectories (emily/, mia/, etc.) with auto-advance

Basic character sorting (single directory):
  python scripts/03_web_character_sorter.py selected/emily
  python scripts/03_web_character_sorter.py _sort_again
  python scripts/03_web_character_sorter.py face_groups/person_0001

Enhanced sorting with similarity maps (RECOMMENDED after face grouper):
  python scripts/03_web_character_sorter.py face_groups/person_0001 --similarity-map face_groups

FEATURES:
---------
• Modern browser interface with clickable action buttons
• Auto-advance to next person directory (no server restarts!)
• Similarity-based spatial layout for easier visual sorting
• Row-level actions: bulk G1/G2/G3/Delete for 6-image rows
• Individual image actions for fine-grained control
• Smart map cleanup: refresh similarity data after deletions
• Manual "Next Directory" button for early advancement
• Integrated FileTracker logging for complete audit trail
• Safe deletion with send2trash (recoverable from system trash)

WORKFLOW:
---------
1. Click one of the character group buttons (1, 2, or 3) to sort the image
2. Click Delete to remove poor quality images
3. Click Skip to move to next image without action
4. Use Back to return to previous image
5. The script will:
   • Move chosen images (+ companion files) to character_group_1/2/3/ directories in project root
   • Delete images using send2trash (recoverable)
   • Log all actions in FileTracker logs

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_ai_assisted_reviewer.py
Step 2: Face Grouping → scripts/02_face_grouper.py
Step 3: Character Sorting → THIS SCRIPT (scripts/03_web_character_sorter.py)
Step 4: Final Cropping → scripts/04_batch_crop_tool.py
Step 5: Basic Review → scripts/05_multi_directory_viewer.py

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/utils/similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

⚠️ IMPORTANT: This script works best with similarity maps from step 2 face grouper!

WHAT HAPPENS:
-------------
• Script starts a local web server (usually http://127.0.0.1:5000)
• Browser automatically opens to show your images one by one
• Click action buttons to make decisions
• All file operations are logged for recovery and audit
• Work at your own pace - server stays running until you're done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from file_tracker import FileTracker
from utils.companion_file_utils import find_all_companion_files, move_file_with_all_companions, safe_delete_image_and_yaml, launch_browser, generate_thumbnail, get_error_display_html, format_image_display_name, sort_image_files_by_timestamp_and_stage

try:
    from flask import Flask, Response, jsonify, render_template_string, request, redirect
except Exception:  # pragma: no cover - import guard for clearer error
    print("[!] Flask is required. Install with: pip install flask", file=sys.stderr)
    raise

try:
    from PIL import Image
except Exception:
    print("[!] Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise

try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    print("[*] HEIC/HEIF support enabled via pillow-heif.")
except Exception:
    pass

_SEND2TRASH_AVAILABLE = False
try:
    from send2trash import send2trash
    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False

THUMBNAIL_MAX_DIM = 800


class MultiDirectoryProgressTracker:
    """Manages progress tracking across multiple directories with session persistence."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.progress_dir = Path("data/sorter_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress file name based on base directory
        safe_name = str(base_directory).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
        
        self.directories = []
        self.current_directory_index = 0
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
        """Initialize fresh progress tracking session (always starts clean)."""
        # Always initialize fresh - no need to persist progress across restarts
        self.initialize_progress()
    
    def initialize_progress(self):
        """Initialize new progress tracking session."""
        self.session_data = {
            'base_directory': str(self.base_directory),
            'session_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_directory_index': 0,
            'directories': {
                dir_info['name']: {
                    'status': 'pending',
                    'total_files': dir_info['file_count']
                } for dir_info in self.directories
            }
        }
        self.save_progress()
    
    def save_progress(self):
        """Save current progress to file."""
        self.session_data['current_directory_index'] = self.current_directory_index
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
    
    def _find_next_directory_with_images(self, start_index: int) -> int:
        """Find the next directory that actually has images, starting from start_index."""
        for i in range(start_index, len(self.directories)):
            dir_path = self.base_directory / self.directories[i]['name']
            if dir_path.exists() and dir_path.is_dir():
                # Check if directory has images
                image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg"))
                if image_files:
                    print(f"[*] Found directory with images: {self.directories[i]['name']} ({len(image_files)} images)")
                    return i
        
        # If no directory with images found from start_index, search from beginning
        for i in range(0, start_index):
            dir_path = self.base_directory / self.directories[i]['name']
            if dir_path.exists() and dir_path.is_dir():
                image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg"))
                if image_files:
                    print(f"[*] Found directory with images (wrapped): {self.directories[i]['name']} ({len(image_files)} images)")
                    return i
        
        # Fallback to start_index if no images found anywhere
        print(f"[!] No directories with images found, using start index: {start_index}")
        return start_index
    
    def advance_directory(self):
        """Move to next directory."""
        if self.current_directory_index < len(self.directories):
            # Mark current directory as completed
            current_dir = self.get_current_directory()
            if current_dir:
                # Ensure directory exists in session data (might have been added after session started)
                if current_dir['name'] not in self.session_data['directories']:
                    self.session_data['directories'][current_dir['name']] = {
                        'status': 'pending',
                        'total_files': current_dir['file_count']
                    }
                self.session_data['directories'][current_dir['name']]['status'] = 'completed'
        
        self.current_directory_index += 1
        
        # Mark new directory as in progress
        current_dir = self.get_current_directory()
        if current_dir:
            # Ensure directory exists in session data (might have been added after session started)
            if current_dir['name'] not in self.session_data['directories']:
                self.session_data['directories'][current_dir['name']] = {
                    'status': 'pending',
                    'total_files': current_dir['file_count']
                }
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
                'directories_remaining': 0,
                'total_directories': len(self.directories),
                'progress_text': "All directories completed"
            }
        
        directories_remaining = len(self.directories) - self.current_directory_index
        
        return {
            'current_directory': current_dir['name'],
            'directories_remaining': directories_remaining,
            'total_directories': len(self.directories),
            'progress_text': f"{current_dir['name']} • {directories_remaining} directories left • {self.current_directory_index + 1}/{len(self.directories)}"
        }
    
    def print_resume_info(self):
        """Print resume information to console."""
        current_dir = self.get_current_directory()
        if current_dir:
            print(f"    Current directory: {current_dir['name']} ({self.current_directory_index + 1}/{len(self.directories)})")
            remaining = len(self.directories) - self.current_directory_index
            print(f"    Directories remaining: {remaining}")
        else:
            print("    All directories completed!")
    
    def cleanup_completed_session(self):
        """Clean up progress file after completing all directories."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                print(f"[*] Cleaned up progress file: {self.progress_file}")
        except Exception as e:
            print(f"[!] Error cleaning up progress file: {e}")


def human_err(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr)

def info(msg: str) -> None:
    print(f"[*] {msg}")

def scan_images(folder: Path) -> List[Path]:
    """Scan directory for PNG files."""
    allowed = {"png", "jpg", "jpeg", "heic", "heif"}
    results: List[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower().lstrip(".") in allowed:
            results.append(entry)
    return sort_image_files_by_timestamp_and_stage(results)

def load_similarity_neighbors(similarity_map_dir: Path) -> Dict[str, Dict]:
    """Load similarity neighbor data from face_groups/neighbors.jsonl"""
    neighbors = {}
    neighbors_file = similarity_map_dir / "neighbors.jsonl"
    
    if not neighbors_file.exists():
        info(f"No similarity map found at {neighbors_file}")
        return neighbors
    
    try:
        with open(neighbors_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                neighbors[data['filename']] = data
        info(f"Loaded similarity data for {len(neighbors)} images")
    except Exception as e:
        info(f"Failed to load similarity map: {e}")
    
    return neighbors

def similarity_sort_images(images: List[Path], neighbors_data: Dict[str, Dict]) -> List[Path]:
    """Sort images by similarity to create visual neighborhoods."""
    if len(images) <= 1 or not neighbors_data:
        return images
    
    # Convert to filename-based lookup
    image_names = [img.name for img in images]
    image_set = set(image_names)
    name_to_path = {img.name: img for img in images}
    
    # Build similarity graph for this set of images
    similarity_graph = {}
    for img_name in image_names:
        similarity_graph[img_name] = []
        if img_name in neighbors_data:
            # Only include neighbors that are also in this directory
            for neighbor in neighbors_data[img_name]['neighbors']:
                if neighbor['filename'] in image_set:
                    similarity_graph[img_name].append({
                        'filename': neighbor['filename'],
                        'similarity': neighbor['sim']
                    })
            # Sort neighbors by similarity (highest first)
            similarity_graph[img_name].sort(key=lambda x: x['similarity'], reverse=True)
    
    # Use a greedy approach to create spatial neighborhoods
    sorted_names = []
    used = set()
    
    # Start with the image that has the most high-similarity connections
    if similarity_graph:
        start_img = max(image_names, key=lambda img: len([n for n in similarity_graph[img] if n['similarity'] > 0.5]))
    else:
        start_img = image_names[0]
    
    current = start_img
    sorted_names.append(current)
    used.add(current)
    
    # Greedily add the most similar unused neighbor
    while len(sorted_names) < len(image_names):
        best_next = None
        best_similarity = -1
        
        # Look for the best unused neighbor of the current image
        for neighbor in similarity_graph.get(current, []):
            if neighbor['filename'] not in used and neighbor['similarity'] > best_similarity:
                best_next = neighbor['filename']
                best_similarity = neighbor['similarity']
        
        # If no good neighbor found, jump to the unused image with most connections
        if best_next is None:
            remaining = [img for img in image_names if img not in used]
            if remaining:
                best_next = max(remaining, key=lambda img: len(similarity_graph.get(img, [])))
        
        if best_next:
            sorted_names.append(best_next)
            used.add(best_next)
            current = best_next
        else:
            # Fallback: add any remaining images
            remaining = [img for img in image_names if img not in used]
            if remaining:
                sorted_names.extend(remaining)
            break
    
    # Convert back to Path objects
    return [name_to_path[name] for name in sorted_names]

def get_review_notes() -> str:
    """Return HTML-formatted review notes for the classification panel."""
    return """
    <div class="review-notes">
        <h3>📋 Character Classification Guide</h3>
        
        <div class="classification-groups">
            <div class="group-item">
                <strong>Character Group 1:</strong>
                <div class="description">First category - define your criteria</div>
            </div>
            
            <div class="group-item">
                <strong>Character Group 2:</strong>
                <div class="description">Second category - define your criteria</div>
            </div>
            
            <div class="group-item">
                <strong>Character Group 3:</strong>
                <div class="description">Third category - define your criteria</div>
            </div>
        </div>
        
        <div class="red-flags">
            <h4>⚠️ Red Flags (Delete):</h4>
            <ul>
                <li>Poor image quality</li>
                <li>Visual anomalies</li>
                <li>Unclear classification</li>
                <li>Does not fit any category</li>
            </ul>
        </div>
        
        <div class="workflow-tips">
            <h4>💡 Workflow Tips:</h4>
            <ul>
                <li>Take your time - accuracy over speed</li>
                <li>Use Skip if unsure</li>
                <li>Use Back to review previous decisions</li>
                <li>All actions are logged and recoverable</li>
            </ul>
        </div>
    </div>
    """

@lru_cache(maxsize=100)
def _generate_thumbnail(image_path: str, mtime_ns: int, size: int) -> bytes:
    """Generate thumbnail using shared function."""
    return generate_thumbnail(image_path, mtime_ns, size, max_dim=THUMBNAIL_MAX_DIM, quality=85)

def safe_delete(png_path: Path, hard_delete: bool = False, tracker: Optional[FileTracker] = None) -> None:
    """Delete image and ALL companion sidecars (yaml, caption, etc.) via shared utility."""
    # Delegate to shared companion-aware delete to ensure nothing is stranded
    safe_delete_image_and_yaml(png_path, hard_delete=hard_delete, tracker=tracker)

def move_with_metadata(src_path: Path, dest_dir: Path, tracker: FileTracker, group_name: str) -> List[str]:
    """Move PNG and ALL corresponding companion files to destination directory."""
    dest_dir.mkdir(exist_ok=True)
    
    # Use wildcard logic to move PNG and ALL companion files
    moved_files = move_file_with_all_companions(src_path, dest_dir, dry_run=False)
    
    # Log the operation
    tracker.log_operation(
        operation="move",
        source_dir=src_path.parent.name,
        dest_dir=group_name,
        file_count=len(moved_files),
        files=moved_files,
        notes=f"User selected {group_name}"
    )
    
    return moved_files

def launch_browser(host: str, port: int):
    """Launch browser after a short delay."""
    from utils.companion_file_utils import launch_browser as shared_launch_browser
    shared_launch_browser(host, port, delay=1.5)

def detect_face_groups_context(folder: Path) -> dict:
    """Detect context when working in face_groups structure."""
    face_groups_root = folder.parent
    person_dirs = sorted([d for d in face_groups_root.iterdir() 
                         if d.is_dir() and d.name.startswith("person_")])
    
    current_index = next((i for i, d in enumerate(person_dirs) if d == folder), 0)
    
    return {
        "is_face_groups": True,
        "face_groups_root": face_groups_root,
        "current_dir": folder.name,
        "position": current_index + 1,
        "total": len(person_dirs),
        "person_dirs": person_dirs,
        "current_index": current_index
    }

def find_next_person_directory(face_groups_info: dict) -> Optional[Path]:
    """Find the next person directory with images."""
    person_dirs = face_groups_info["person_dirs"]
    current_index = face_groups_info["current_index"]
    
    # Look for next directory with images
    for i in range(current_index + 1, len(person_dirs)):
        next_dir = person_dirs[i]
        if any(next_dir.glob("*.png")):  # Has images
            return next_dir
    
    return None

def clean_similarity_maps(folder: Path, similarity_map_dir: Path) -> bool:
    """Clean similarity maps by removing references to deleted files."""
    try:
        import json
        
        # Get current images in directory
        current_images = {img.name for img in folder.glob("*.png")}
        
        neighbors_file = similarity_map_dir / "neighbors.jsonl"
        if not neighbors_file.exists():
            return False
        
        # Clean neighbors.jsonl
        cleaned_entries = []
        with open(neighbors_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                filename = entry.get("filename", "")
                
                # Keep entry if image still exists
                if filename in current_images:
                    # Clean the neighbors list - remove references to deleted files
                    if "neighbors" in entry:
                        entry["neighbors"] = [n for n in entry["neighbors"] 
                                           if n.get("filename", "") in current_images]
                    cleaned_entries.append(entry)
        
        # Write cleaned neighbors.jsonl
        with open(neighbors_file, 'w') as f:
            for entry in cleaned_entries:
                f.write(json.dumps(entry) + '\n')
        
        info(f"Cleaned similarity maps - kept {len(cleaned_entries)} entries")
        return True
        
    except Exception as e:
        info(f"Failed to clean similarity maps: {e}")
        return False

def create_app(folder: Path, hard_delete: bool = False, similarity_map_dir: Optional[Path] = None, multi_directory_tracker: Optional[MultiDirectoryProgressTracker] = None) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    
    # Detect if we're working in face_groups structure
    is_face_groups_mode = "face_groups" in str(folder) and folder.parent.name == "face_groups"
    face_groups_info = None
    
    if is_face_groups_mode:
        face_groups_info = detect_face_groups_context(folder)
        info(f"Face Groups Mode: {face_groups_info['current_dir']} ({face_groups_info['position']}/{face_groups_info['total']})")
    
    # Scan for images
    images = scan_images(folder)
    if not images:
        human_err(f"No images found in {folder}")
        sys.exit(1)
    
    info(f"Found {len(images)} images in {folder}")
    
    # Apply similarity-based sorting if available
    if similarity_map_dir:
        neighbors_data = load_similarity_neighbors(similarity_map_dir)
        if neighbors_data:
            original_count = len(images)
            images = similarity_sort_images(images, neighbors_data)
            moved_count = sum(1 for i, img in enumerate(images) if i >= original_count or img != scan_images(folder)[i])
            info(f"Applied similarity layout: {moved_count}/{original_count} images repositioned")
        else:
            info("Similarity map not found, using alphabetical order")
    
    # Group images into batches of 6 for grid display
    IMAGES_PER_BATCH = 6
    image_batches = []
    for i in range(0, len(images), IMAGES_PER_BATCH):
        batch = images[i:i + IMAGES_PER_BATCH]
        image_batches.append({
            'id': i // IMAGES_PER_BATCH,
            'images': [{'index': j, 'path': img, 'name': img.name} for j, img in enumerate(batch)],
            'start_index': i
        })
    
    # Initialize FileTracker
    tracker = FileTracker("character_sorter")
    
    # Set up target directories using centralized standard paths
    try:
        from utils.standard_paths import get_character_group_dirs
        character_group_1, character_group_2, character_group_3 = get_character_group_dirs()
    except Exception:
        project_root = Path(__file__).parent.parent  # fallback
        character_group_1 = project_root / "__character_group_1"
        character_group_2 = project_root / "__character_group_2"  
        character_group_3 = project_root / "__character_group_3"
    
    # Create directories if they don't exist
    for group_dir in [character_group_1, character_group_2, character_group_3]:
        group_dir.mkdir(exist_ok=True)
    
    # App state
    app.config["IMAGES"] = images
    app.config["IMAGE_BATCHES"] = image_batches
    app.config["CURRENT_BATCH"] = 0
    app.config["TRACKER"] = tracker
    app.config["FOLDER"] = folder
    app.config["HARD_DELETE"] = hard_delete
    app.config["SIMILARITY_MAP_DIR"] = similarity_map_dir
    app.config["IS_FACE_GROUPS_MODE"] = is_face_groups_mode
    app.config["FACE_GROUPS_INFO"] = face_groups_info
    app.config["MULTI_DIRECTORY_TRACKER"] = multi_directory_tracker
    app.config["IS_MULTI_DIRECTORY_MODE"] = multi_directory_tracker is not None
    
    # Set up directory navigation for face groups mode
    if is_face_groups_mode and face_groups_info:
        app.config["ALL_PERSON_DIRS"] = face_groups_info["person_dirs"]
        app.config["CURRENT_DIR"] = face_groups_info["current_dir"]
    else:
        app.config["ALL_PERSON_DIRS"] = []
        app.config["CURRENT_DIR"] = folder
    app.config["TARGET_DIRS"] = {
        "group1": character_group_1,
        "group2": character_group_2,
        "group3": character_group_3
    }
    app.config["HISTORY"] = []  # Track actions for undo/back functionality
    app.config["BATCH_DECISIONS"] = {}  # Track decisions for current batch
    
    @app.route("/")
    def index():
        """Main page with viewport-based progressive image sorter interface."""
        images = app.config["IMAGES"]
        
        if not images:
            # Check for auto-advance in multi-directory mode
            if app.config["IS_MULTI_DIRECTORY_MODE"] and app.config["MULTI_DIRECTORY_TRACKER"]:
                tracker = app.config["MULTI_DIRECTORY_TRACKER"]
                if tracker.has_more_directories():
                    current_dir_info = tracker.get_current_directory()
                    tracker.advance_directory()
                    next_dir_info = tracker.get_current_directory()
                    
                    if next_dir_info:
                        print(f"DEBUG: Multi-directory mode - advancing to {next_dir_info['name']}")
                        # Direct redirect to next directory instead of showing countdown page
                        return redirect(f'/switch-directory/{next_dir_info["name"]}')
            
            # Check for auto-advance in face groups mode
            elif app.config["IS_FACE_GROUPS_MODE"] and app.config["FACE_GROUPS_INFO"]:
                print(f"DEBUG: Face groups mode detected, checking for next directory...")
                next_dir = find_next_person_directory(app.config["FACE_GROUPS_INFO"])
                if next_dir:
                    print(f"DEBUG: Found next directory: {next_dir.name}")
                    # Direct redirect to next directory instead of showing countdown page
                    return redirect(f'/switch-directory/{next_dir.name}')
                else:
                    print(f"DEBUG: No next directory found, showing completion")
            else:
                print(f"DEBUG: Single directory mode")
            
            return render_template_string(COMPLETION_TEMPLATE, 
                                        history=app.config["HISTORY"])
        
        # Group images into rows for display
        image_rows = []
        for i in range(0, len(images), IMAGES_PER_BATCH):
            row_images = images[i:i + IMAGES_PER_BATCH]
            row_list = []
            for j, img in enumerate(row_images):
                cropped = img.with_suffix('.cropped').exists()
                row_list.append({
                    'index': j,
                    'path': img,
                    'name': img.name,
                    'stage_name': format_image_display_name(img.name, context='web'),
                    'global_index': i + j,
                    'cropped': cropped
                })
            image_rows.append({
                'id': i // IMAGES_PER_BATCH,
                'images': row_list,
                'start_index': i
            })
        
        # Calculate current directory index for previous button state
        current_dir_index = 0
        if app.config["IS_FACE_GROUPS_MODE"] and "ALL_PERSON_DIRS" in app.config:
            try:
                current_dir_index = app.config["ALL_PERSON_DIRS"].index(app.config["CURRENT_DIR"])
            except (ValueError, KeyError):
                current_dir_index = 0
        
        # Get progress info for multi-directory mode
        progress_info = None
        if app.config["IS_MULTI_DIRECTORY_MODE"] and app.config["MULTI_DIRECTORY_TRACKER"]:
            progress_info = app.config["MULTI_DIRECTORY_TRACKER"].get_progress_info()
        
        return render_template_string(
            VIEWPORT_TEMPLATE,
            image_rows=image_rows,
            total_images=len(images),
            total_rows=len(image_rows),
            folder_name=folder.name,
            is_face_groups_mode=app.config["IS_FACE_GROUPS_MODE"],
            face_groups_info=app.config["FACE_GROUPS_INFO"],
            is_multi_directory_mode=app.config["IS_MULTI_DIRECTORY_MODE"],
            progress_info=progress_info,
            has_similarity_maps=app.config["SIMILARITY_MAP_DIR"] is not None,
            current_dir_index=current_dir_index,
            error_display_html=get_error_display_html()
        )
    
    @app.route("/image/<int:global_index>")
    def serve_image(global_index: int):
        """Serve image thumbnail by global index."""
        images = app.config["IMAGES"]
        
        if global_index >= len(images):
            return Response(status=404)
        
        path = images[global_index]
        if not path.exists():
            return Response(status=404)
        
        stat = path.stat()
        data = _generate_thumbnail(str(path), int(stat.st_mtime_ns), stat.st_size)
        return Response(data, mimetype="image/jpeg")
    
    @app.route("/action", methods=["POST"])
    def handle_action():
        """Handle user actions (group1, group2, group3, delete, skip, back)."""
        data = request.get_json() or {}
        action = data.get("action")
        
        current_idx = app.config["CURRENT_INDEX"]
        images = app.config["IMAGES"]
        tracker = app.config["TRACKER"]
        target_dirs = app.config["TARGET_DIRS"]
        history = app.config["HISTORY"]
        
        if current_idx >= len(images):
            return jsonify({"status": "error", "message": "No more images"}), 400
        
        current_image = images[current_idx]
        
        if action == "back":
            if current_idx > 0:
                app.config["CURRENT_INDEX"] = current_idx - 1
            return jsonify({"status": "ok", "action": "back"})
        
        elif action == "skip":
            app.config["CURRENT_INDEX"] = current_idx + 1
            return jsonify({"status": "ok", "action": "skip"})
        
        elif action in ["group1", "group2", "group3"]:
            target_dir = target_dirs[action]
            group_name = f"character_{action}"
            
            try:
                moved_files = move_with_metadata(current_image, target_dir, tracker, group_name)
                history.append((current_image.name, group_name, moved_files))
                
                # Remove from images list
                images.pop(current_idx)
                if current_idx >= len(images) and images:
                    app.config["CURRENT_INDEX"] = len(images) - 1
                
                return jsonify({
                    "status": "ok", 
                    "action": action,
                    "moved_files": moved_files,
                    "remaining": len(images)
                })
                
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500
        
        elif action == "delete":
            try:
                yaml_file = current_image.with_suffix('.yaml')
                files_to_delete = [current_image.name]
                if yaml_file.exists():
                    files_to_delete.append(yaml_file.name)
                
                safe_delete(current_image, app.config["HARD_DELETE"], tracker)
                history.append((current_image.name, "DELETED", files_to_delete))
                
                # Remove from images list
                images.pop(current_idx)
                if current_idx >= len(images) and images:
                    app.config["CURRENT_INDEX"] = len(images) - 1
                
                return jsonify({
                    "status": "ok",
                    "action": "delete", 
                    "deleted_files": files_to_delete,
                    "remaining": len(images)
                })
                
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500
        
        else:
            return jsonify({"status": "error", "message": "Unknown action"}), 400
    
    def process_shutdown():
        """Graceful shutdown."""
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
    
    @app.route("/complete", methods=["POST"])
    def complete_session():
        """Complete the sorting session and shutdown."""
        history = app.config["HISTORY"]
        remaining = len(app.config["IMAGES"])
        
        # Generate summary
        group1_count = sum(1 for _, action, _ in history if action == "character_group1")
        group2_count = sum(1 for _, action, _ in history if action == "character_group2") 
        group3_count = sum(1 for _, action, _ in history if action == "character_group3")
        deleted_count = sum(1 for _, action, _ in history if action == "DELETED")
        
        message = (
            f"Character sorting complete! "
            f"Group 1: {group1_count}, Group 2: {group2_count}, Group 3: {group3_count}, "
            f"Deleted: {deleted_count}, Remaining: {remaining}"
        )
        
        info(message)
        
        # Shutdown server after response
        threading.Thread(target=process_shutdown, daemon=True).start()
        
        return jsonify({"status": "ok", "message": message})
    
    @app.route("/process-batch", methods=["POST"])
    def process_batch():
        """Process the current batch of reviewed images."""
        data = request.get_json() or {}
        reviewed_count = data.get("reviewed_count", 0)
        
        if reviewed_count == 0:
            return jsonify({"status": "error", "message": "No images in batch"}), 400
        
        history = app.config["HISTORY"]
        remaining = len(app.config["IMAGES"])
        
        # Generate summary for the batch that was processed
        # (the actual processing was done incrementally as user made decisions)
        group1_count = sum(1 for _, action, _ in history if action == "character_group1")
        group2_count = sum(1 for _, action, _ in history if action == "character_group2") 
        group3_count = sum(1 for _, action, _ in history if action == "character_group3")
        deleted_count = sum(1 for _, action, _ in history if action == "DELETED")
        
        message = (
            f"Processed {reviewed_count} images - "
            f"Group 1: {group1_count}, Group 2: {group2_count}, Group 3: {group3_count}, "
            f"Deleted: {deleted_count}"
        )
        
        info(f"Batch processed: {message}")
        
        return jsonify({
            "status": "ok", 
            "message": message,
            "remaining": remaining,
            "processed": reviewed_count
        })
    
    @app.route("/process-viewport-batch", methods=["POST"])
    def process_viewport_batch():
        """Process decisions for images in the current viewport batch."""
        data = request.get_json() or {}
        decisions = data.get("decisions", {})
        
        if not decisions:
            return jsonify({"status": "error", "message": "No decisions to process"}), 400
        
        tracker = app.config["TRACKER"]
        target_dirs = app.config["TARGET_DIRS"]
        hard_delete = app.config["HARD_DELETE"]
        images = app.config["IMAGES"]
        history = app.config["HISTORY"]
        
        processed_count = 0
        processed_indices = []
        
        try:
            for image_index_str, action in decisions.items():
                image_index = int(image_index_str)
                
                if image_index >= len(images):
                    continue
                
                image_path = images[image_index]
                processed_indices.append(image_index)
                
                if action in ["group1", "group2", "group3"]:
                    target_dir = target_dirs[action]
                    group_name = f"character_{action}"
                    moved_files = move_with_metadata(image_path, target_dir, tracker, group_name)
                    history.append((image_path.name, group_name, moved_files))
                    processed_count += 1
                    
                elif action == "delete":
                    files_to_delete = [image_path.name]
                    yaml_file = image_path.with_suffix('.yaml')
                    if yaml_file.exists():
                        files_to_delete.append(yaml_file.name)
                    
                    safe_delete(image_path, hard_delete, tracker)
                    history.append((image_path.name, "DELETED", files_to_delete))
                    processed_count += 1
                    
                elif action == "crop":
                    # Move to central __crop directory for further processing
                    try:
                        from utils.standard_paths import get_crop_dir
                        crop_dir = get_crop_dir()
                        crop_dir.mkdir(exist_ok=True)
                    except Exception:
                        crop_dir = Path.cwd() / "__crop"
                        crop_dir.mkdir(exist_ok=True)
                    moved_files = move_with_metadata(image_path, crop_dir, tracker, "crop_processing")
                    history.append((image_path.name, "SENT_TO_CROP", moved_files))
                    processed_count += 1
        
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        
        # Remove processed images from the list (in reverse order to maintain indices)
        for index in sorted(processed_indices, reverse=True):
            if index < len(images):
                images.pop(index)
        
        app.config["IMAGES"] = images
        
        message = f"Processed {processed_count} images from viewport batch"
        
        # If no images remain, trigger page refresh to check for auto-advance
        if len(images) == 0:
            return jsonify({"status": "ok", "message": message, "remaining": 0, "refresh": True})
        
        return jsonify({"status": "ok", "message": message, "remaining": len(images)})
    
    @app.route("/complete")
    def completion_page():
        """Show completion page when all images are processed."""
        return render_template_string(COMPLETION_TEMPLATE, history=app.config["HISTORY"])
    
    @app.route("/refresh-layout", methods=["POST"])
    def refresh_layout():
        """Clean similarity maps and refresh the layout."""
        if not app.config["SIMILARITY_MAP_DIR"]:
            return jsonify({"status": "error", "message": "No similarity maps available"}), 400
        
        folder = app.config["FOLDER"]
        similarity_map_dir = app.config["SIMILARITY_MAP_DIR"]
        
        success = clean_similarity_maps(folder, similarity_map_dir)
        
        if success:
            return jsonify({"status": "success", "message": "Layout refreshed successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to refresh layout"}), 500
    
    @app.route("/prev-directory", methods=["POST"])
    def prev_directory():
        """Get the previous person directory for manual navigation."""
        if not app.config["IS_FACE_GROUPS_MODE"]:
            return jsonify({"status": "error", "message": "Not in face groups mode"}), 400
        
        current_dir = app.config["CURRENT_DIR"]
        all_dirs = app.config["ALL_PERSON_DIRS"]
        
        try:
            current_index = all_dirs.index(current_dir)
            if current_index == 0:
                return jsonify({"status": "first_directory", "message": "Already at the first directory"})
            
            prev_dir = all_dirs[current_index - 1]
            
            # Build redirect URL for previous directory
            base_url = request.url_root.rstrip('/')
            redirect_url = f"{base_url}/?dir={prev_dir.name}"
            
            return jsonify({
                "status": "success",
                "prev_directory": prev_dir.name,
                "redirect_url": redirect_url,
                "current_index": current_index - 1,
                "total_directories": len(all_dirs)
            })
            
        except ValueError:
            return jsonify({"status": "error", "message": "Current directory not found in list"}), 500
    
    @app.route("/next-directory", methods=["POST"])
    def next_directory():
        """Get the next person directory for manual advance."""
        if not app.config["IS_FACE_GROUPS_MODE"]:
            return jsonify({"status": "error", "message": "Not in face groups mode"}), 400
        
        face_groups_info = app.config["FACE_GROUPS_INFO"]
        next_dir = find_next_person_directory(face_groups_info)
        
        if next_dir:
            return jsonify({
                "status": "success", 
                "next_directory": next_dir.name,
                "current_position": face_groups_info["position"],
                "total": face_groups_info["total"]
            })
        else:
            return jsonify({"status": "complete", "message": "All directories processed"})
    
    @app.route("/switch-directory/<directory_name>")
    def switch_directory(directory_name):
        """Switch to a new directory without restarting the server."""
        # Handle multi-directory mode
        if app.config["IS_MULTI_DIRECTORY_MODE"]:
            tracker = app.config["MULTI_DIRECTORY_TRACKER"]
            
            # Find the directory by name
            target_dir = None
            for dir_info in tracker.directories:
                if dir_info['name'] == directory_name:
                    target_dir = dir_info['path']
                    break
            
            if not target_dir:
                return jsonify({"status": "error", "message": f"Directory {directory_name} not found"}), 404
            
            # Scan for images in new directory
            new_images = scan_images(target_dir)
            if not new_images:
                return jsonify({"status": "error", "message": f"No images found in {directory_name}"}), 404
            
            # Apply similarity sorting if available
            similarity_map_dir = app.config["SIMILARITY_MAP_DIR"]
            if similarity_map_dir:
                neighbors_data = load_similarity_neighbors(similarity_map_dir)
                if neighbors_data:
                    new_images = similarity_sort_images(new_images, neighbors_data)
            
            # Group images into batches
            new_image_batches = []
            for i in range(0, len(new_images), IMAGES_PER_BATCH):
                batch = new_images[i:i + IMAGES_PER_BATCH]
                new_image_batches.append({
                    'id': i // IMAGES_PER_BATCH,
                    'images': [{'index': j, 'path': img, 'name': img.name} for j, img in enumerate(batch)],
                    'start_index': i
                })
            
            # Update app config
            app.config["IMAGES"] = new_images
            app.config["IMAGE_BATCHES"] = new_image_batches
            app.config["CURRENT_BATCH"] = 0
            app.config["FOLDER"] = target_dir
            app.config["HISTORY"] = []
            app.config["BATCH_DECISIONS"] = {}
            
            # Redirect to main page to show the new directory
            return redirect('/')
        
        # Handle face groups mode
        elif app.config["IS_FACE_GROUPS_MODE"]:
            if not app.config["FACE_GROUPS_INFO"]:
                return jsonify({"status": "error", "message": "Not in face groups mode"}), 400
            
            # Get the face groups root from current config
            face_groups_info = app.config["FACE_GROUPS_INFO"]
            face_groups_root = face_groups_info["face_groups_root"]
            new_directory = face_groups_root / directory_name
            
            if not new_directory.exists():
                return jsonify({"status": "error", "message": f"Directory {directory_name} not found"}), 404
            
            # Scan for images in new directory
            new_images = scan_images(new_directory)
            if not new_images:
                return jsonify({"status": "error", "message": f"No images found in {directory_name}"}), 404
            
            # Apply similarity sorting if available
            similarity_map_dir = app.config["SIMILARITY_MAP_DIR"]
            if similarity_map_dir:
                neighbors_data = load_similarity_neighbors(similarity_map_dir)
                if neighbors_data:
                    new_images = similarity_sort_images(new_images, neighbors_data)
            
            # Update app config with new directory
            new_face_groups_info = detect_face_groups_context(new_directory)
            app.config["FOLDER"] = new_directory
            app.config["IMAGES"] = new_images
            app.config["FACE_GROUPS_INFO"] = new_face_groups_info
            app.config["HISTORY"] = []  # Reset history for new directory
            
            # Update directory navigation config
            app.config["ALL_PERSON_DIRS"] = new_face_groups_info["person_dirs"]
            app.config["CURRENT_DIR"] = new_face_groups_info["current_dir"]
            
            return jsonify({
                "status": "success",
                "message": f"Switched to {directory_name}",
                "image_count": len(new_images)
            })
        
        # If we get here, neither multi-directory nor face groups mode is active
        else:
            return jsonify({"status": "error", "message": "No multi-directory mode active"}), 400
    
    return app

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Web Character Sorter</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101014;
      --surface: #181821;
      --surface-alt: #1f1f2c;
      --accent: #4f9dff;
      --accent-soft: rgba(79, 157, 255, 0.2);
      --danger: #ff6b6b;
      --success: #51cf66;
      --warning: #ffd43b;
      --muted: #a0a3b1;
    }
    
    * { box-sizing: border-box; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: white;
      min-height: 100vh;
      display: flex;
    }
    
    .main-container {
      display: flex;
      width: 100%;
      min-height: 100vh;
    }
    
    .image-panel {
      flex: 2;
      display: flex;
      flex-direction: column;
      padding: 2rem;
      align-items: center;
      justify-content: center;
    }
    
    .notes-panel {
      flex: 1;
      background: var(--surface);
      padding: 2rem;
      border-left: 1px solid rgba(255,255,255,0.1);
      overflow-y: auto;
      max-width: 400px;
    }
    
    .progress-bar {
      width: 100%;
      height: 8px;
      background: var(--surface);
      border-radius: 4px;
      margin-bottom: 2rem;
      overflow: hidden;
    }
    
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--success));
      width: {{ progress_percent }}%;
      transition: width 0.3s ease;
    }
    
    .image-container {
      text-align: center;
      margin-bottom: 2rem;
    }
    
    .image-container img {
      max-width: 100%;
      max-height: 60vh;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .image-info {
      margin: 1rem 0;
      color: var(--muted);
    }
    
    .actions {
      display: flex;
      gap: 0.6rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    
    .btn {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      min-width: 120px;
    }
    
    /* Button highlight for active/clicked state */
    .btn.btn-active {
      background: var(--accent) !important;
      color: white !important;
      box-shadow: 0 0 0 2px var(--accent), 0 0 12px rgba(79, 157, 255, 0.6);
      transform: translateY(-1px);
    }
    
    .btn-group1 {
      background: var(--accent);
      color: white;
    }
    
    .btn-group1:hover {
      background: #3d8bff;
      transform: translateY(-2px);
    }
    
    .btn-group2 {
      background: var(--success);
      color: white;
    }
    
    .btn-group2:hover {
      background: #40c057;
      transform: translateY(-2px);
    }
    
    .btn-group3 {
      background: var(--warning);
      color: black;
    }
    
    .btn-group3:hover {
      background: #ffec99;
      transform: translateY(-2px);
    }
    
    .btn-delete {
      background: var(--danger);
      color: white;
    }
    
    .btn-delete:hover {
      background: #ff5252;
      transform: translateY(-2px);
    }
    
    .btn-secondary {
      background: var(--surface-alt);
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.1);
    }
    
    .btn-secondary:hover {
      background: var(--surface);
      color: white;
    }
    
    .review-notes h3 {
      color: var(--accent);
      margin-bottom: 0.5rem;
    }
    
    .group-item {
      margin-bottom: 0.5rem;
      padding: 1rem;
      background: var(--surface-alt);
      border-radius: 8px;
    }
    
    .group-item strong {
      display: block;
      margin-bottom: 0.5rem;
    }
    
    .description {
      color: var(--muted);
      font-size: 0.9rem;
    }
    
    .red-flags, .workflow-tips {
      margin-top: 2rem;
    }
    
    .red-flags h4, .workflow-tips h4 {
      color: var(--warning);
      margin-bottom: 0.5rem;
    }
    
    .red-flags ul, .workflow-tips ul {
      margin: 0;
      padding-left: 1.2rem;
    }
    
    .red-flags li, .workflow-tips li {
      margin-bottom: 0.3rem;
      color: var(--muted);
      font-size: 0.9rem;
    }
    
    .status {
      position: fixed;
      top: 1rem;
      right: 1rem;
      background: var(--surface);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.1);
      color: var(--muted);
    }
    
    .loading {
      opacity: 0.5;
      pointer-events: none;
    }
    
    .batch-controls {
      margin-top: 2rem;
      text-align: center;
      padding: 1.5rem;
      background: var(--surface-alt);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    
    .btn-batch {
      background: linear-gradient(135deg, var(--warning) 0%, #ffb700 100%);
      color: black;
      margin-bottom: 0.5rem;
    }
    
    .btn-batch:hover {
      background: linear-gradient(135deg, #ffec99 0%, #ffd43b 100%);
      transform: translateY(-2px);
    }
    
    .batch-info {
      display: flex;
      justify-content: center;
      gap: 0.6rem;
      color: var(--muted);
      font-size: 0.9rem;
    }
    
    .batch-info strong {
      color: var(--warning);
    }
  </style>
</head>
<body>
  <div class="main-container">
    <div class="image-panel">
      <div class="status">
        {{ current_index }} / {{ total_images }} • {{ folder_name }}
      </div>
      
      <div class="progress-bar">
        <div class="progress-fill"></div>
      </div>
      
      <div class="image-container">
        <img src="/image" alt="{{ current_image }}" loading="eager">
        <div class="image-info">{{ current_image }}</div>
      </div>
      
      <div class="actions">
        <button class="btn btn-group1" onclick="handleAction('group1')">
          Group 1
        </button>
        <button class="btn btn-group2" onclick="handleAction('group2')">
          Group 2  
        </button>
        <button class="btn btn-group3" onclick="handleAction('group3')">
          Group 3
        </button>
        <button class="btn btn-delete" onclick="handleAction('delete')">
          Delete
        </button>
        <button class="btn btn-secondary" onclick="handleAction('skip')">
          Skip
        </button>
        <button class="btn btn-secondary" onclick="handleAction('back')">
          Back
        </button>
      </div>
      
      <div class="batch-controls">
        <button class="btn btn-batch" onclick="processBatch()">
          Process Current Batch
        </button>
        <div class="batch-info">
          <span>Reviewed: <strong id="batch-count">0</strong> images</span>
          <span>•</span>
          <span>Remaining: <strong id="remaining-count">{{ total_images }}</strong></span>
        </div>
      </div>
    </div>
    
    <div class="notes-panel">
      {{ review_notes|safe }}
    </div>
  </div>

  <script>
    let isLoading = false;
    
    // Progressive batch tracking
    let maxIndexReached = 0;
    let reviewedIndices = new Set();
    
    function updateBatchInfo() {
      const batchCount = document.getElementById('batch-count');
      const remainingCount = document.getElementById('remaining-count');
      
      if (batchCount) batchCount.textContent = reviewedIndices.size.toString();
      if (remainingCount) {
        // Get current remaining count from server response
        const currentRemaining = parseInt(remainingCount.textContent) || 0;
        remainingCount.textContent = currentRemaining.toString();
      }
    }
    
    function highlightButton(action) {
      // Remove active class from all buttons
      document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('btn-active'));
      
      // Add active class to the clicked button
      const buttonClass = action === 'skip' || action === 'back' ? 'btn-secondary' : `btn-${action}`;
      const button = document.querySelector(`.${buttonClass}`);
      if (button) {
        button.classList.add('btn-active');
        
        // Remove highlight after 1 second
        setTimeout(() => {
          button.classList.remove('btn-active');
        }, 1000);
      }
    }
    
    async function handleAction(action) {
      if (isLoading) return;
      
      // Highlight the clicked button
      highlightButton(action);
      
      isLoading = true;
      document.body.classList.add('loading');
      
      try {
        const response = await fetch('/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
          // Track progress for batch processing
          if (action !== 'back') {
            const currentIndex = maxIndexReached;
            reviewedIndices.add(currentIndex);
            maxIndexReached = Math.max(maxIndexReached + 1, 0);
          }
          
          if (result.remaining === 0) {
            // Refresh page to trigger auto-advance check in face groups mode
            window.location.reload();
          } else {
            // Update batch info before reload
            updateBatchInfo();
            // Reload page for next image
            window.location.reload();
          }
        } else {
          showError('Error: ' + result.message);
        }
      } catch (error) {
        showError('Network error: ' + error.message);
      } finally {
        isLoading = false;
        document.body.classList.remove('loading');
      }
    }
    
    async function processBatch() {
      if (isLoading) return;
      
      if (reviewedIndices.size === 0) {
        showError('No images reviewed yet. Make some decisions first!');
        return;
      }
      
      if (!confirm(`Process current batch of ${reviewedIndices.size} reviewed images? This will apply your decisions and continue with remaining images.`)) {
        return;
      }
      
      isLoading = true;
      document.body.classList.add('loading');
      
      try {
        const response = await fetch('/process-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            reviewed_count: reviewedIndices.size,
            batch_mode: true 
          })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
          // Reset batch tracking
          reviewedIndices.clear();
          maxIndexReached = 0;
          
          if (result.remaining === 0) {
            // Refresh page to trigger auto-advance check if in face groups mode
            if (result.refresh) {
              window.location.reload();
            } else {
              document.body.innerHTML = '<div style="text-align:center;padding:4rem;"><h1>✅ All Images Processed!</h1><p>Character sorting complete.</p></div>';
            }
          } else {
            // Batch processed successfully, reload page
            window.location.reload();
          }
        } else {
          showError('Error processing batch: ' + result.message);
        }
      } catch (error) {
        showError('Network error: ' + error.message);
      } finally {
        isLoading = false;
        document.body.classList.remove('loading');
      }
    }
    
    // Initialize batch info on page load
    updateBatchInfo();
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (isLoading) return;
      
      switch(e.key) {
        case '1': handleAction('group1'); break;
        case '2': handleAction('group2'); break;
        case '3': handleAction('group3'); break;
        case 'd': case 'Delete': handleAction('delete'); break;
        case 's': handleAction('skip'); break;
        case 'b': handleAction('back'); break;
      }
    });
  </script>
</body>
</html>
"""

GRID_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Web Character Sorter - Grid View</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101014;
      --surface: #181821;
      --surface-alt: #1f1f2c;
      --accent: #4f9dff;
      --accent-soft: rgba(79, 157, 255, 0.2);
      --danger: #ff6b6b;
      --success: #51cf66;
      --warning: #ffd43b;
      --muted: #a0a3b1;
    }
    
    * { box-sizing: border-box; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: white;
      min-height: 100vh;
    }
    
    .main-container {
      display: flex;
      width: 100%;
      min-height: 100vh;
    }
    
    .content-panel {
      flex: 3;
      padding: 2rem;
    }
    
    .notes-panel {
      flex: 1;
      background: var(--surface);
      padding: 2rem;
      border-left: 1px solid rgba(255,255,255,0.1);
      overflow-y: auto;
      max-width: 400px;
    }
    
    .header {
      margin-bottom: 2rem;
    }
    
    .progress-bar {
      width: 100%;
      height: 8px;
      background: var(--surface);
      border-radius: 4px;
      margin-bottom: 0.5rem;
      overflow: hidden;
    }
    
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--success));
      width: {{ progress_percent }}%;
      transition: width 0.3s ease;
    }
    
    .batch-info {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 2rem;
    }
    
    .image-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 0.6rem;
      margin-bottom: 2rem;
    }
    
    .image-item {
      background: var(--surface);
      border-radius: 12px;
      padding: 1rem;
      text-align: center;
      border: 2px solid transparent;
      transition: border-color 0.3s ease;
    }
    
    .image-item.selected-group1 { border-color: var(--accent); }
    .image-item.selected-group2 { border-color: var(--success); }
    .image-item.selected-group3 { border-color: var(--warning); }
    .image-item.selected-delete { border-color: var(--danger); }
    
    .image-container {
      margin-bottom: 0.5rem;
    }
    
    .image-container img {
      width: 100%;
      height: 200px;
      object-fit: cover;
      border-radius: 8px;
    }
    
    .image-name {
      font-size: 0.8rem;
      color: white;
      margin-bottom: 0.5rem;
      word-break: break-all;
      font-weight: 500;
    }
    
    .action-buttons {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    
    .btn {
      padding: 0.5rem;
      border: none;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .btn-group1 {
      background: var(--accent);
      color: white;
    }
    
    .btn-group1:hover {
      background: #3d8bff;
    }
    
    .btn-group2 {
      background: var(--success);
      color: white;
    }
    
    .btn-group2:hover {
      background: #40c057;
    }
    
    .btn-group3 {
      background: var(--warning);
      color: black;
    }
    
    .btn-group3:hover {
      background: #ffec99;
    }
    
    .btn-delete {
      background: var(--danger);
      color: white;
    }
    
    .btn-delete:hover {
      background: #ff5252;
    }
    
    .btn-skip {
      background: var(--surface-alt);
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.1);
    }
    
    .btn-skip:hover {
      background: var(--surface);
      color: white;
    }
    
    .controls {
      display: flex;
      justify-content: center;
      gap: 0.6rem;
      margin-bottom: 2rem;
    }
    
    .control-btn {
      padding: 0.75rem 2rem;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .btn-process {
      background: linear-gradient(135deg, var(--warning) 0%, #ffb700 100%);
      color: black;
    }
    
    .btn-process:hover {
      background: linear-gradient(135deg, #ffec99 0%, #ffd43b 100%);
      transform: translateY(-2px);
    }
    
    .btn-next {
      background: linear-gradient(135deg, var(--accent) 0%, #3d8bff 100%);
      color: white;
    }
    
    .btn-next:hover {
      background: linear-gradient(135deg, #3d8bff 0%, var(--accent) 100%);
      transform: translateY(-2px);
    }
    
    .btn-prev {
      background: linear-gradient(135deg, var(--muted) 0%, #8a8ca0 100%);
      color: white;
      margin-right: 0.5rem;
    }
    
    .btn-prev:hover:not(:disabled) {
      background: linear-gradient(135deg, #8a8ca0 0%, var(--muted) 100%);
      transform: translateY(-2px);
    }
    
    .btn-prev:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none;
    }
    
    .btn-back {
      background: var(--surface-alt);
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.1);
    }
    
    .btn-back:hover {
      background: var(--surface);
      color: white;
    }
    
    .loading {
      opacity: 0.5;
      pointer-events: none;
    }
    
    .review-notes h3 {
      color: var(--accent);
      margin-bottom: 0.5rem;
    }
    
    .group-item {
      margin-bottom: 0.5rem;
      padding: 1rem;
      background: var(--surface-alt);
      border-radius: 8px;
    }
    
    .group-item strong {
      display: block;
      margin-bottom: 0.5rem;
    }
    
    .description {
      color: var(--muted);
      font-size: 0.9rem;
    }
    
    .red-flags, .workflow-tips {
      margin-top: 2rem;
    }
    
    .red-flags h4, .workflow-tips h4 {
      color: var(--warning);
      margin-bottom: 0.5rem;
    }
    
    .red-flags ul, .workflow-tips ul {
      margin: 0;
      padding-left: 1.2rem;
    }
    
    .red-flags li, .workflow-tips li {
      margin-bottom: 0.3rem;
      color: var(--muted);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="main-container">
    <div class="content-panel">
      <div class="header">
        <div class="progress-bar">
          <div class="progress-fill"></div>
        </div>
        <div class="batch-info">
          <span>Batch {{ batch_id }} / {{ total_batches }}</span>
          <span>{{ folder_name }}</span>
          <span>{{ processed_images }} / {{ total_images }} processed</span>
        </div>
      </div>
      
      <div class="image-grid">
        {% for image in current_batch.images %}
        <div class="image-item" data-image-index="{{ image.index }}" data-batch-id="{{ current_batch.id }}">
          <div class="image-container">
            <img src="/image/{{ current_batch.id }}/{{ image.index }}" alt="{{ image.name }}" loading="lazy">
          </div>
          <div class="image-name">{{ image.stage_name }}</div>
          <div class="action-buttons">
            <button class="btn btn-group1" onclick="selectAction({{ current_batch.id }}, {{ image.index }}, 'group1')">
              Group 1
            </button>
            <button class="btn btn-group2" onclick="selectAction({{ current_batch.id }}, {{ image.index }}, 'group2')">
              Group 2
            </button>
            <button class="btn btn-group3" onclick="selectAction({{ current_batch.id }}, {{ image.index }}, 'group3')">
              Group 3
            </button>
            <button class="btn btn-delete" onclick="selectAction({{ current_batch.id }}, {{ image.index }}, 'delete')">
              Delete
            </button>
            <button class="btn btn-skip" onclick="selectAction({{ current_batch.id }}, {{ image.index }}, 'skip')">
              Skip
            </button>
          </div>
        </div>
        {% endfor %}
      </div>
      
      <div class="controls">
        <button class="control-btn btn-back" onclick="previousBatch()">Previous Batch</button>
        <button class="control-btn btn-process" onclick="processBatch()">Process Batch</button>
        <button class="control-btn btn-next" onclick="nextBatch()">Next Batch</button>
      </div>
    </div>
    
    <div class="notes-panel">
      {{ review_notes|safe }}
    </div>
  </div>

  <script>
    let batchDecisions = {};
    let isLoading = false;
    
    function selectAction(batchId, imageIndex, action) {
      const key = `${batchId}-${imageIndex}`;
      const imageItem = document.querySelector(`[data-batch-id="${batchId}"][data-image-index="${imageIndex}"]`);
      
      // Remove previous selection classes
      imageItem.classList.remove('selected-group1', 'selected-group2', 'selected-group3', 'selected-delete');
      
      // Add new selection
      if (action !== 'skip') {
        imageItem.classList.add(`selected-${action}`);
        batchDecisions[key] = action;
      } else {
        delete batchDecisions[key];
      }
    }
    
    async function processBatch() {
      if (isLoading) return;
      
      const decisions = Object.keys(batchDecisions).length;
      if (decisions === 0) {
        showError('Make some decisions first! Select actions for the images you want to process.');
        return;
      }
      
      // Confirmation dialog removed - proceed directly
      
      isLoading = true;
      document.body.classList.add('loading');
      
      try {
        const response = await fetch('/process-decisions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ decisions: batchDecisions })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
          if (result.remaining === 0) {
            // Refresh page to trigger auto-advance check in face groups mode
            window.location.reload();
          } else {
            // Batch processed successfully, reload page
            batchDecisions = {};
            window.location.reload();
          }
        } else {
          showError('Error: ' + result.message);
        }
      } catch (error) {
        showError('Network error: ' + error.message);
      } finally {
        isLoading = false;
        document.body.classList.remove('loading');
      }
    }
    
    function nextBatch() {
      window.location.href = '/next';
    }
    
    function previousBatch() {
      window.location.href = '/previous';
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (isLoading) return;
      
      switch(e.key) {
        case 'ArrowRight': nextBatch(); break;
        case 'ArrowLeft': previousBatch(); break;
        case 'Enter': processBatch(); break;
      }
    });
  </script>
</body>
</html>
"""

VIEWPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Web Character Sorter - Viewport Progressive</title>
  {{ error_display_html|safe }}
  <style>
    :root {
      color-scheme: dark;
      --bg: #101014;
      --surface: #181821;
      --surface-alt: #1f1f2c;
      --accent: #4f9dff;
      --accent-soft: rgba(79, 157, 255, 0.2);
      --danger: #ff6b6b;
      --success: #51cf66;
      --warning: #ffd43b;
      --muted: #a0a3b1;
    }
    
    * { box-sizing: border-box; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: white;
      min-height: 100vh;
    }
    
    .main-container {
      width: 100%;
      min-height: 100vh;
    }
    
    .content-panel {
      padding: 0 2rem 2rem 2rem;
      overflow-y: auto;
      max-height: 100vh;
      max-width: 1800px;
      margin: 0 auto;
    }
    
    .header {
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: 100;
      padding: 1rem 0 1rem 0;
      margin-bottom: 2rem;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .progress-info {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 0.5rem;
    }
    
    .batch-info {
      color: var(--warning);
      font-weight: 600;
    }
    
    .controls {
      display: flex;
      justify-content: center;
      gap: 0.6rem;
    }
    
    .btn-process {
      background: linear-gradient(135deg, var(--warning) 0%, #ffb700 100%);
      color: black;
      border: none;
      padding: 0.75rem 2rem;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .btn-process:hover {
      background: linear-gradient(135deg, #ffec99 0%, #ffd43b 100%);
      transform: translateY(-2px);
    }
    
    .btn-process:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }
    
    .btn-refresh, .btn-next {
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-right: 0.5rem;
    }
    
    .btn-refresh:hover {
      background: #3a8ce6;
      transform: translateY(-1px);
    }
    
    .btn-next {
      background: var(--success);
    }
    
    .btn-next:hover {
      background: #40c653;
      transform: translateY(-1px);
    }
    
    .face-groups-progress {
      font-size: 0.9rem;
      color: var(--accent);
      font-weight: 500;
    }
    
    .multi-directory-progress {
      font-size: 0.9rem;
      color: var(--accent);
      font-weight: 500;
    }
    
    .image-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-bottom: 2rem;
      padding: 0.5rem;
      background: var(--surface);
      border-radius: 12px;
      border: 2px solid transparent;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .image-row.in-batch {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(79,157,255,0.2);
    }
    
    .image-row.reviewed {
      border-color: var(--success);
      box-shadow: 0 0 0 1px rgba(81,207,102,0.2);
    }
    
    .image-item {
      background: var(--surface-alt);
      border-radius: 8px;
      padding: 0.2rem;
      text-align: center;
      border: 2px solid transparent;
      transition: all 0.3s ease;
      flex: 1;
      min-width: 0;
      max-width: none;
    }
    
    .image-item.selected-group1 { border-color: var(--accent); }
    .image-item.selected-group2 { border-color: var(--success); }
    .image-item.selected-group3 { border-color: var(--warning); }
    .image-item.selected-crop { border-color: var(--secondary); }
    .image-item.selected-delete { 
      border: 4px solid #cc0000;
      background: rgba(204, 0, 0, 0.1);
      box-shadow: 0 0 0 2px rgba(204, 0, 0, 0.3);
    }
    
    .image-container {
      margin-bottom: 0.5rem;
    }
    
    .image-container img {
      width: 100%;
      height: auto;
      object-fit: contain;
      border-radius: 6px;
      cursor: pointer;
      transition: opacity 0.2s ease;
      display: block;
    }
    
    
    .image-name {
      font-size: 0.7rem;
      color: white;
      margin-bottom: 0.8rem;
      word-break: break-all;
      font-weight: 500;
    }
    
    .action-buttons {
      display: flex;
      gap: 0.2rem;
    }
    
    .btn {
      padding: 0.3rem 0.4rem;
      border: none;
      border-radius: 4px;
      font-size: 0.65rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
      flex: 1;
    }
    
    .btn-group1 {
      background: var(--accent);
      color: white;
    }
    
    .btn-group1:hover {
      background: #3d8bff;
    }
    
    .btn-group2 {
      background: var(--success);
      color: white;
    }
    
    .btn-group2:hover {
      background: #40c057;
    }
    
    .btn-group3 {
      background: var(--warning);
      color: black;
    }
    
    .btn-group3:hover {
      background: #ffec99;
    }
    
    .btn-crop {
      background: var(--secondary);
      color: white;
    }
    
    .btn-crop:hover {
      background: #868e96;
    }
    
    .btn-delete {
      background: var(--danger);
      color: white;
    }
    
    .btn-delete:hover {
      background: #ff5252;
    }

    /* Row-level button styles */
    .row-container {
      margin-bottom: 1rem;
    }

    .row-buttons {
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    
    /* Row button active state */
    .btn-row.btn-active {
      background: var(--accent) !important;
      color: white !important;
      box-shadow: 0 0 0 2px var(--accent), 0 0 8px rgba(79, 157, 255, 0.5);
    }

    .btn-row {
      padding: 0.2rem 0.8rem;
      height: 24px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s ease;
      min-width: 32px;
    }

    .btn-row-group1 {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .btn-row-group1:hover {
      background: #5b9cff;
      border-color: #5b9cff;
    }

    .btn-row-group2 {
      background: var(--success);
      border-color: var(--success);
      color: white;
    }

    .btn-row-group2:hover {
      background: #6bc56d;
      border-color: #6bc56d;
    }

    .btn-row-group3 {
      background: var(--warning);
      border-color: var(--warning);
      color: white;
    }

    .btn-row-group3:hover {
      background: #ffec99;
      border-color: #ffec99;
      color: #333;
    }

    .btn-row-delete {
      background: var(--danger);
      border-color: var(--danger);
      color: white;
    }

    .btn-row-delete:hover {
      background: #ff6b6b;
      border-color: #ff6b6b;
    }
    
    .loading {
      opacity: 0.5;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div class="main-container">
    <div class="content-panel">
      <div class="header">
        <div class="progress-info">
          <span>{{ total_images }} images • {{ folder_name }}</span>
          {% if is_multi_directory_mode and progress_info %}
          <span class="multi-directory-progress">{{ progress_info.progress_text }}</span>
          {% elif is_face_groups_mode and face_groups_info %}
          <span class="face-groups-progress">Directory {{ face_groups_info.position }}/{{ face_groups_info.total }}</span>
          {% endif %}
          <span class="batch-info" id="batch-info">Review batch: 0 rows</span>
        </div>
        <div class="controls">
          {% if is_face_groups_mode and has_similarity_maps %}
          <button class="btn-refresh" id="refresh-layout" onclick="refreshLayout()">
            🔄 Refresh Layout
          </button>
          {% endif %}
          {% if is_face_groups_mode %}
          <button class="btn-prev" id="prev-directory" onclick="skipToPreviousDirectory()" {% if current_dir_index == 0 %}disabled{% endif %}>
            ⏮️ Previous Directory
          </button>
          <button class="btn-next" id="next-directory" onclick="skipToNextDirectory()">
            ⏭️ Next Directory
          </button>
          {% endif %}
          <button class="btn-process" id="process-batch" onclick="processViewportBatch()" disabled>
            Process Current Batch
          </button>
        </div>
      </div>
      
      {% for row in image_rows %}
      <div class="row-container">
        <div class="row-buttons">
          <button class="btn-row btn-row-group1" onclick="selectRowAction({{ row.id }}, 'group1')">G1</button>
          <button class="btn-row btn-row-group2" onclick="selectRowAction({{ row.id }}, 'group2')">G2</button>
          <button class="btn-row btn-row-group3" onclick="selectRowAction({{ row.id }}, 'group3')">G3</button>
          <button class="btn-row btn-row-delete" onclick="selectRowAction({{ row.id }}, 'delete')">Del</button>
        </div>
        <div class="image-row" data-row-id="{{ row.id }}">
          {% for image in row.images %}
        <div class="image-item" data-global-index="{{ image.global_index }}">
          <div class="image-container">
            <img src="/image/{{ image.global_index }}" alt="{{ image.name }}" loading="lazy" onclick="selectAction({{ image.global_index }}, 'delete')">
            {% if image.cropped %}
            <div style="position:absolute; top:6px; left:6px; background: white; color: black; padding: 2px 6px; border-radius: 6px; font-size: 10px; font-weight: 700;">Cropped</div>
            {% endif %}
          </div>
          <div class="image-name">{{ image.stage_name }}</div>
          <div class="action-buttons">
            <button class="btn btn-group1" onclick="selectAction({{ image.global_index }}, 'group1')">
              G1
            </button>
            <button class="btn btn-group2" onclick="selectAction({{ image.global_index }}, 'group2')">
              G2
            </button>
            <button class="btn btn-group3" onclick="selectAction({{ image.global_index }}, 'group3')">
              G3
            </button>
            <button class="btn btn-crop" onclick="selectAction({{ image.global_index }}, 'crop')">
              Crop
            </button>
          </div>
        </div>
        {% endfor %}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <script>
    let reviewBatchRows = new Set();
    let imageDecisions = {};
    let isLoading = false;
    
    function updateBatchInfo() {
      const batchInfo = document.getElementById('batch-info');
      const processBtn = document.getElementById('process-batch');
      
      // Count decisions by type
      const counts = { group1: 0, group2: 0, group3: 0, delete: 0, crop: 0 };
      Object.values(imageDecisions).forEach(action => {
        if (counts.hasOwnProperty(action)) {
          counts[action]++;
        }
      });
      
      // Calculate remaining images
      const totalImages = {{ total_images }};
      const decisionsCount = Object.keys(imageDecisions).length;
      const remainingImages = totalImages - decisionsCount;
      
      batchInfo.innerHTML = `${remainingImages} left • <span style="color: #4f9dff">G1: ${counts.group1}</span> • <span style="color: #51cf66">G2: ${counts.group2}</span> • <span style="color: #ffd43b">G3: ${counts.group3}</span> • <span style="color: #868e96">Crop: ${counts.crop}</span> • <span style="color: #cc0000">Del: ${counts.delete}</span>`;
      
      processBtn.disabled = decisionsCount === 0;
    }
    
    function updateViewportBatch() {
      const viewportTop = window.scrollY;
      const viewportBottom = viewportTop + window.innerHeight;
      const buffer = 100;
      
      document.querySelectorAll('.image-row').forEach((row) => {
        const rect = row.getBoundingClientRect();
        const elementTop = rect.top + window.scrollY;
        const elementBottom = elementTop + rect.height;
        
        // Check if row is in or has been in viewport
        if (elementTop < viewportBottom + buffer && elementBottom > viewportTop - buffer) {
          const rowId = parseInt(row.dataset.rowId);
          if (!reviewBatchRows.has(rowId)) {
            reviewBatchRows.add(rowId);
            row.classList.add('in-batch');
          }
        }
      });
      
      updateBatchInfo();
    }
    
    function selectAction(globalIndex, action) {
      const imageItem = document.querySelector(`[data-global-index="${globalIndex}"]`);
      const row = imageItem.closest('.image-row');
      
      // Check if this action is already selected - if so, deselect it
      if (imageItem.classList.contains(`selected-${action}`)) {
        // Deselect - remove all selection classes and decision
        imageItem.classList.remove('selected-group1', 'selected-group2', 'selected-group3', 'selected-crop', 'selected-delete');
        delete imageDecisions[globalIndex];
      } else {
        // Remove previous selection classes
        imageItem.classList.remove('selected-group1', 'selected-group2', 'selected-group3', 'selected-crop', 'selected-delete');
        
        // Add new selection (skip is implicit - no action needed)
        imageItem.classList.add(`selected-${action}`);
        imageDecisions[globalIndex] = action;
      }
      
      // Update row state based on decisions made in this row
      const rowImages = row.querySelectorAll('[data-global-index]');
      const rowDecisions = Array.from(rowImages).some(img => {
        const idx = img.dataset.globalIndex;
        return imageDecisions.hasOwnProperty(idx);
      });
      
      if (rowDecisions) {
        row.classList.add('reviewed');
        row.classList.remove('in-batch');
      } else {
        row.classList.add('in-batch');
        row.classList.remove('reviewed');
      }
      
      updateBatchInfo();
    }
    
    function highlightRowButton(rowId, action) {
      // Remove active class from all row buttons in this row
      const rowButtons = document.querySelector(`[data-row-id="${rowId}"]`).closest('.row-container').querySelectorAll('.btn-row');
      rowButtons.forEach(btn => btn.classList.remove('btn-active'));
      
      // Add active class to the clicked row button
      const button = document.querySelector(`[onclick="selectRowAction(${rowId}, '${action}')"]`);
      if (button) {
        button.classList.add('btn-active');
        
        // Remove highlight after 1 second
        setTimeout(() => {
          button.classList.remove('btn-active');
        }, 1000);
      }
    }

    function selectRowAction(rowId, action) {
      // Highlight the clicked row button
      highlightRowButton(rowId, action);
      
      const row = document.querySelector(`[data-row-id="${rowId}"]`);
      const imageItems = row.querySelectorAll('[data-global-index]');
      
      // Check if most unselected images in this row already have this action
      const unselectedImages = Array.from(imageItems).filter(imageItem => {
        const globalIndex = imageItem.dataset.globalIndex;
        return !imageDecisions.hasOwnProperty(globalIndex);
      });
      
      const alreadySelectedWithAction = Array.from(imageItems).filter(imageItem => {
        return imageItem.classList.contains(`selected-${action}`);
      });
      
      // If more than half the row already has this action, toggle it off (deselect)
      if (alreadySelectedWithAction.length > imageItems.length / 2) {
        imageItems.forEach(imageItem => {
          const globalIndex = imageItem.dataset.globalIndex;
          if (imageItem.classList.contains(`selected-${action}`)) {
            // Deselect this action
            imageItem.classList.remove('selected-group1', 'selected-group2', 'selected-group3', 'selected-crop', 'selected-delete');
            delete imageDecisions[globalIndex];
          }
        });
      } else {
        // Apply action to all images in the row that don't already have a selection
        unselectedImages.forEach(imageItem => {
          const globalIndex = imageItem.dataset.globalIndex;
          
          // Remove previous selection classes
          imageItem.classList.remove('selected-group1', 'selected-group2', 'selected-group3', 'selected-crop', 'selected-delete');
          
          // Add new selection
          imageItem.classList.add(`selected-${action}`);
          imageDecisions[globalIndex] = action;
        });
      }
      
      // Update row state
      row.classList.add('reviewed');
      row.classList.remove('in-batch');
      
      updateBatchInfo();
    }
    
    async function processViewportBatch() {
      if (isLoading) return;
      
      const decisionsCount = Object.keys(imageDecisions).length;
      if (decisionsCount === 0) {
        showError('No decisions made yet! Make some selections first.');
        return;
      }
      
      // Confirmation dialog removed - proceed directly
      
      isLoading = true;
      document.body.classList.add('loading');
      
      try {
        const response = await fetch('/process-viewport-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ decisions: imageDecisions })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
          if (result.remaining === 0) {
            // Refresh page to trigger auto-advance check in face groups mode
            window.location.reload();
          } else {
            // Batch processed successfully, reload page
            imageDecisions = {};
            window.location.reload();
          }
        } else {
          showError('Error: ' + result.message);
        }
      } catch (error) {
        showError('Network error: ' + error.message);
      } finally {
        isLoading = false;
        document.body.classList.remove('loading');
      }
    }
    
    // Initialize viewport tracking
    window.addEventListener('scroll', updateViewportBatch);
    window.addEventListener('resize', updateViewportBatch);
    updateViewportBatch(); // Initial check
    
    // Face Groups functions
    async function refreshLayout() {
      const refreshBtn = document.getElementById('refresh-layout');
      const originalText = refreshBtn.innerHTML;
      refreshBtn.innerHTML = '🔄 Refreshing...';
      refreshBtn.disabled = true;
      
      try {
        const response = await fetch('/refresh-layout', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'}
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
          // Reload page to get updated layout
          window.location.reload();
        } else {
          showError('Failed to refresh layout: ' + result.message);
        }
      } catch (error) {
        showError('Error refreshing layout: ' + error.message);
      } finally {
        refreshBtn.innerHTML = originalText;
        refreshBtn.disabled = false;
      }
    }
    
    async function skipToPreviousDirectory() {
      try {
        const response = await fetch('/prev-directory', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'}
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
          // Navigate to previous directory
          window.location.href = result.redirect_url;
        } else if (result.status === 'first_directory') {
          showError('Already at the first directory');
        } else {
          showError('Error: ' + result.message);
        }
      } catch (error) {
        showError('Error going to previous directory: ' + error.message);
      }
    }
    
    async function skipToNextDirectory() {
      try {
        const response = await fetch('/next-directory', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'}
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
          // Switch to the next directory using the server route
          const switchResponse = await fetch('/switch-directory/' + result.next_directory);
          const switchResult = await switchResponse.json();
          
          if (switchResult.status === 'success') {
            // Successfully switched directories, reload the page
            window.location.reload();
          } else {
            showError('Error switching to next directory: ' + switchResult.message);
          }
        } else if (result.status === 'complete') {
          showError('All directories have been processed!');
        } else {
          showError('Error: ' + result.message);
        }
      } catch (error) {
        showError('Error getting next directory: ' + error.message);
      }
    }
    
    // Track user activity
    function markActivity() {
        // Activity tracking removed - using file-operation-based timing instead
    }
    
    // Activity tracking events
    document.addEventListener('click', markActivity);
    document.addEventListener('keydown', markActivity);
    document.addEventListener('scroll', markActivity);
    
    // Throttled mouse movement tracking
    let mouseThrottle = false;
    document.addEventListener('mousemove', function() {
        if (!mouseThrottle) {
            markActivity();
            mouseThrottle = true;
            setTimeout(() => mouseThrottle = false, 2000);
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (isLoading) return;
      
      if (e.key === 'Enter') {
        processViewportBatch();
      }
    });
  </script>
</body>
</html>
"""

COMPLETION_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Character Sorting Complete</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #101014;
      color: white;
      text-align: center;
      padding: 4rem;
    }
    .summary {
      background: #181821;
      padding: 2rem;
      border-radius: 12px;
      margin: 2rem auto;
      max-width: 600px;
    }
  </style>
</head>
<body>
  <h1>✅ Character Sorting Complete!</h1>
  <div class="summary">
    <h2>Session Summary</h2>
    <p>All images have been processed. Check the FileTracker logs for complete details.</p>
    <p>Server shutting down...</p>
  </div>
</body>
</html>
"""

def extract_stage_name(filename: str) -> str:
    """Extract stage name from filename after second underscore."""
    parts = filename.split('_')
    if len(parts) >= 3:
        # Join everything after the second underscore
        stage_name = '_'.join(parts[2:])
        # Remove file extension
        stage_name = stage_name.rsplit('.', 1)[0]
        return stage_name
    return filename.rsplit('.', 1)[0]  # Fallback to filename without extension

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort images into character groups using a modern web interface.",
    )
    parser.add_argument("directory", help="Directory containing images to sort")
    
    parser.add_argument("--hard-delete", action="store_true", 
                       help="Permanently delete files instead of using trash")
    parser.add_argument("--similarity-map", type=str, 
                       help="Directory containing similarity map files (neighbors.jsonl, etc.)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=8080, type=int, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", 
                       help="Don't automatically open browser")
    
    args = parser.parse_args()
    
    folder = Path(args.directory).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        human_err(f"{folder} is not a directory")
        sys.exit(1)
    
    # Check if we're in multi-directory mode (directory contains subdirectories with images)
    multi_directory_tracker = None
    is_multi_directory_mode = False
    
    # Look for subdirectories with PNG files
    subdirs_with_images = []
    for item in folder.iterdir():
        if item.is_dir() and list(item.glob("*.png")):
            subdirs_with_images.append(item)
    
    # If we found subdirectories with images, enable multi-directory mode
    if len(subdirs_with_images) > 1:
        is_multi_directory_mode = True
        multi_directory_tracker = MultiDirectoryProgressTracker(folder)
        
        # Get the current directory to process
        current_dir_info = multi_directory_tracker.get_current_directory()
        if not current_dir_info:
            info("All directories completed!")
            sys.exit(0)
        
        folder = current_dir_info['path']  # Process the current subdirectory
        info(f"Multi-directory mode: Processing {current_dir_info['name']} ({multi_directory_tracker.current_directory_index + 1}/{len(multi_directory_tracker.directories)})")
    
    # Handle similarity map directory
    similarity_map_dir = None
    if args.similarity_map:
        similarity_map_dir = Path(args.similarity_map).expanduser().resolve()
        if not similarity_map_dir.exists() or not similarity_map_dir.is_dir():
            info(f"Similarity map directory {similarity_map_dir} not found - using alphabetical order")
            similarity_map_dir = None
    
    app = create_app(folder, args.hard_delete, similarity_map_dir, multi_directory_tracker)
    
    if not args.no_browser:
        threading.Thread(target=launch_browser, args=(args.host, args.port), daemon=True).start()
    
    try:
        info(f"Character sorter running at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except OSError as exc:
        human_err(f"Failed to start server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
