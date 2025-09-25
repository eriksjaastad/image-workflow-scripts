#!/usr/bin/env python3
"""
Step 1: Web Image Selector - Modern Browser Edition
====================================================
Modern web-based tool for selecting the best image from each triplet set.
Replaces old Matplotlib interface with fast, scrollable browser interface.

🎨 STYLE GUIDE:
---------------
This web interface follows the project style guide for consistent design:
  📁 WEB_STYLE_GUIDE.md
Colors, spacing, typography, and interaction patterns are defined there.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run on directories containing triplets (after quality filtering):
  python scripts/01_web_image_selector.py XXX_CONTENT/

FEATURES:
---------
• Modern browser interface with clickable thumbnails
• Fast performance with cached image loading
• Advanced triplet detection with precise timestamp matching
• Integrated FileTracker logging for complete audit trail
• Selected images are automatically moved to selected/ folder for processing
• Safe deletion with send2trash (recoverable from system trash)
• Support for PNG and HEIC/HEIF formats

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → THIS SCRIPT (scripts/01_web_image_selector.py)
Step 2: Face Grouping → scripts/02_face_grouper.py
Step 3: Character Sorting → scripts/03_web_character_sorter.py (uses similarity maps from step 2)
Step 4: Final Cropping → scripts/04_batch_crop_tool.py
Step 5: Basic Review → scripts/05_multi_directory_viewer.py

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/utils/similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

HOW IT WORKS:
---------
1. Click exactly one image in each row to keep it
2. Leave a row untouched if you want all three images deleted
3. Selected images are automatically moved to selected/ folder
4. Press **Finalize selections** when done
5. The script will:
   • Move chosen images (+ YAML files) to selected/
   • Delete unselected images using send2trash (recoverable)
   • Log all actions in triplet_culler_log.csv and FileTracker logs

OPTIONAL FLAGS:
---------------
  --exts            File extensions to include (default: png)
  --hard-delete     Permanently delete instead of using trash
  --host/--port     Web server binding (default: 127.0.0.1:8080)
  --no-browser      Don't auto-launch browser
  --print-triplets  Show triplet groupings and exit (debug)

HOW IT WORKS:
-------------
1. Click exactly one image in each row to keep it
2. Leave a row untouched if you want all three images deleted
3. Selected images are automatically moved to selected/ folder
4. Press **Finalize selections** when done

WHAT HAPPENS:
-------------
• Script starts a local web server (usually http://127.0.0.1:5000)
• Browser automatically opens to show your image triplets
• Click thumbnails to make selections
• All file operations are logged for recovery and audit
• Work at your own pace - server stays running until you're done
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from file_tracker import FileTracker
from utils.activity_timer import ActivityTimer

try:
    from flask import Flask, Response, jsonify, render_template_string, request
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

STAGE_NAMES = ("stage1_generated", "stage1.5_face_swapped", "stage2_upscaled")
THUMBNAIL_MAX_DIM = 768


def human_err(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[*] {msg}")


def detect_stage(name: str) -> str:
    low = name.lower()
    for stage in STAGE_NAMES:
        if stage in low:
            return stage
    return ""


def scan_images(folder: Path, exts: Iterable[str], recursive: bool = True) -> List[Path]:
    """Scan for image files, optionally recursively"""
    allowed = {e.lower().lstrip(".") for e in exts}
    results: List[Path] = []
    
    if recursive:
        # Use rglob for recursive scanning (** pattern)
        for ext in allowed:
            pattern = f"**/*.{ext}"
            results.extend(folder.glob(pattern))
        # Remove duplicates and sort
        results = sorted(set(results))
    else:
        # Original non-recursive behavior
        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.suffix.lower().lstrip(".") in allowed:
                results.append(entry)
    
    return results


def extract_timestamp(filename: str) -> Optional[str]:
    match = re.search(r"(\d{8}_\d{6})", filename)
    return match.group(1) if match else None


def timestamp_to_minutes(timestamp_str: Optional[str]) -> Optional[float]:
    if not timestamp_str or len(timestamp_str) != 15:
        return None
    try:
        time_part = timestamp_str.split("_")[1]
        hours = int(time_part[:2])
        minutes = int(time_part[2:4])
        seconds = int(time_part[4:6])
        return hours * 60 + minutes + seconds / 60.0
    except Exception:
        return None


def get_date_from_timestamp(timestamp_str: Optional[str]) -> Optional[str]:
    if not timestamp_str or len(timestamp_str) != 15:
        return None
    return timestamp_str.split("_")[0]


@dataclass
class GroupRecord:
    index: int
    paths: Tuple[Path, ...]  # Can be 2 or 3 paths
    relative_dir: str

    def as_payload(self, total: int) -> Dict[str, object]:
        images = []
        for idx, path in enumerate(self.paths):
            stage = detect_stage(path.name) or f"image {idx + 1}"
            images.append(
                {
                    "index": idx,
                    "name": path.name,
                    "stage": stage.replace("_", " "),
                    "src": f"/image/{self.index}/{idx}",
                }
            )
        return {
            "id": self.index,
            "display_index": self.index + 1,
            "relative_dir": self.relative_dir,
            "images": images,
            "total": total,
        }


STAGE_TOKENS = (
    "stage1_generated",
    "stage1.5_face_swapped", 
    "stage2_upscaled",
    "stage3_enhanced",
)

def _basekey(p: Path) -> str:
    """Filename minus stage tokens & punctuation → used to detect runs."""
    n = p.stem.lower()
    for t in STAGE_TOKENS:
        n = n.replace(t, "")
    # collapse separators and trim
    n = re.sub(r"[._\-]+", "_", n).strip("_- .")
    return n

def get_stage_number(stage: str) -> float:
    """Convert stage name to numeric value for comparison."""
    stage_map = {
        "stage1_generated": 1.0,
        "stage1.5_face_swapped": 1.5,
        "stage2_upscaled": 2.0,
        "stage3_enhanced": 3.0
    }
    return stage_map.get(stage, 0.0)


def find_flexible_groups(files: List[Path]) -> List[Tuple[Path, ...]]:
    """Find groups by detecting stage number decreases (new group when stage < previous stage)."""
    groups: List[Tuple[Path, ...]] = []
    
    # Sort by timestamp first, then stage order, then name (ChatGPT recommendation)
    def sort_key(path):
        timestamp = extract_timestamp(path.name) or "99999999_999999"
        stage = detect_stage(path.name)
        stage_num = get_stage_number(stage)
        return (timestamp, stage_num, path.name)
    
    sorted_files = sorted(files, key=sort_key)
    
    if not sorted_files:
        return groups
    
    current_group = []
    prev_stage_num = None  # Start with None to handle first file properly
    
    for file in sorted_files:
        stage = detect_stage(file.name)
        if not stage:
            continue  # Skip files that don't match expected patterns
            
        stage_num = get_stage_number(stage)
        
        # If this is the first file or stage equal/decreased, start a new group
        if prev_stage_num is None or (stage_num <= prev_stage_num and current_group):
            # Finish current group if it exists and has enough files
            if current_group and len(current_group) >= 2:
                groups.append(tuple(current_group))
            current_group = [file]  # Start new group with current file
        else:
            # Continue current group
            current_group.append(file)
        
        prev_stage_num = stage_num
    
    # Add the final group if it has 2+ images
    if len(current_group) >= 2:
        groups.append(tuple(current_group))
    
    # Filter out orphan groups (no ascending stage progression)
    filtered_groups = []
    for group in groups:
        # Check if group has ascending stage progression
        stage_nums = [get_stage_number(detect_stage(f.name)) for f in group]
        has_progression = any(stage_nums[i] < stage_nums[i+1] for i in range(len(stage_nums)-1))
        
        if has_progression:
            filtered_groups.append(group)
    
    return filtered_groups


def safe_delete(paths: Iterable[Path], hard_delete: bool = False, tracker: Optional[FileTracker] = None) -> None:
    all_files_to_delete: List[Path] = []

    for png_path in paths:
        if not png_path.exists():
            continue
        all_files_to_delete.append(png_path)
        yaml_path = png_path.parent / f"{png_path.stem}.yaml"
        if yaml_path.exists():
            all_files_to_delete.append(yaml_path)

    if not all_files_to_delete:
        return

    deleted_files: List[str] = []

    if hard_delete:
        for path in all_files_to_delete:
            try:
                path.unlink()
                info(f"Deleted: {path.name}")
                deleted_files.append(path.name)
            except Exception as exc:
                human_err(f"Failed to delete {path}: {exc}")
    else:
        if not _SEND2TRASH_AVAILABLE:
            raise RuntimeError(
                "send2trash is not installed. Install with: pip install send2trash\n"
                "Or rerun with --hard-delete to permanently delete files."
            )
        for path in all_files_to_delete:
            try:
                send2trash(str(path))
                info(f"Sent to Trash: {path.name}")
                deleted_files.append(path.name)
            except Exception as exc:
                human_err(f"Failed to send to Trash {path}: {exc}")

    if tracker and deleted_files:
        source_dir = str(all_files_to_delete[0].parent.name) if all_files_to_delete else "unknown"
        tracker.log_operation(
            operation="delete" if hard_delete else "send_to_trash",
            source_dir=source_dir,
            file_count=len(deleted_files),
            files=deleted_files,
            notes="Rejected images from triplet selection",
        )


def write_log(csv_path: Path, action: str, kept: Optional[Path], deleted: List[Path]) -> None:
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if header_needed:
            writer.writerow(["action", "kept", "deleted1", "deleted2", "deleted3"])
        row = [
            action,
            str(kept) if kept else "",
            str(deleted[0]) if len(deleted) > 0 else "",
            str(deleted[1]) if len(deleted) > 1 else "",
            str(deleted[2]) if len(deleted) > 2 else "",
        ]
        writer.writerow(row)


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / "scripts").exists():
            return current
        current = current.parent
    return start.resolve()


def move_with_yaml(
    src_path: Path,
    destination: Path,
    tracker: FileTracker,
    dest_label: str,
    notes: str,
) -> Path:
    destination.mkdir(exist_ok=True)
    moved_files: List[str] = []

    target_png = destination / src_path.name
    shutil.move(str(src_path), str(target_png))
    moved_files.append(src_path.name)

    yaml_path = src_path.parent / f"{src_path.stem}.yaml"
    if yaml_path.exists():
        target_yaml = destination / yaml_path.name
        shutil.move(str(yaml_path), str(target_yaml))
        moved_files.append(yaml_path.name)

    tracker.log_operation(
        operation="move",
        source_dir=str(src_path.parent.name),
        dest_dir=dest_label,
        file_count=len(moved_files),
        files=moved_files,
        notes=notes,
    )

    return target_png


# Removed move_to_reviewed function - all selections now go to crop


def move_to_selected(src_path: Path, tracker: FileTracker) -> Path:
    project_root = find_project_root(src_path.parent)
    selected_dir = project_root / "selected"
    return move_with_yaml(
        src_path,
        selected_dir,
        tracker,
        dest_label="selected",
        notes="Selected during triplet selection",
    )


@lru_cache(maxsize=2048)
def _generate_thumbnail(path_str: str, mtime_ns: int, file_size: int) -> bytes:
    path = Path(path_str)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def build_app(
    groups: List[GroupRecord],
    base_folder: Path,
    tracker: FileTracker,
    log_path: Path,
    hard_delete: bool,
    batch_size: int = 50,
    batch_start: int = 0,
    activity_timer: Optional[ActivityTimer] = None,
) -> Flask:
    app = Flask(__name__)

    # Calculate batch boundaries
    total_groups = len(groups)
    batch_end = min(batch_start + batch_size, total_groups)
    current_batch_groups = groups[batch_start:batch_end]
    current_batch_payload = [record.as_payload(total=total_groups) for record in current_batch_groups]
    
    # Batch info
    batch_info = {
        "current_batch": (batch_start // batch_size) + 1,
        "total_batches": (total_groups + batch_size - 1) // batch_size,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "batch_size": batch_size,
        "total_groups": total_groups,
        "current_batch_size": len(current_batch_groups)
    }

    app.config["ALL_GROUPS"] = groups  # Keep full list for batch switching
    app.config["GROUPS"] = current_batch_groups  # Current batch only
    app.config["GROUP_PAYLOAD"] = current_batch_payload
    app.config["BATCH_INFO"] = batch_info
    app.config["BASE_FOLDER"] = base_folder
    app.config["TRACKER"] = tracker
    app.config["LOG_PATH"] = log_path
    app.config["HARD_DELETE"] = hard_delete
    app.config["PROCESSED"] = False
    app.config["ACTIVITY_TIMER"] = activity_timer

    page_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Image Version Selector</title>
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
          margin: 0;
          font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
          background: var(--bg);
          color: #f8f9ff;
          display: flex;
          flex-direction: column;
        }
        .main-layout {
          display: flex;
          flex: 1;
          min-height: 100vh;
        }
        .content-area {
          flex: 1;
          overflow-y: auto;
        }
        .row-buttons {
          display: flex;
          flex-direction: column;
          gap: 0.3rem;
          margin-left: 1rem;
          align-self: center;
        }
        .action-btn {
          padding: 0.5rem;
          background: var(--surface);
          border: 2px solid rgba(255,255,255,0.1);
          border-radius: 6px;
          color: white;
          font-size: 0.8rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          text-align: center;
          min-height: 32px;
          min-width: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .action-btn:hover {
          background: var(--accent);
          border-color: var(--accent);
          transform: translateY(-1px);
        }
        .action-btn.crop-active {
          background: white;
          border-color: white;
          color: black;
        }
        .action-btn.image-active {
          background: var(--accent);
          border-color: var(--accent);
          color: white;
          transform: translateY(-1px);
        }
        .action-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
          transform: none;
        }
        header.toolbar {
          background: var(--bg);
          padding: 1rem 2rem;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          position: fixed;
          top: 0;
          z-index: 150;
          left: 0;
          right: 0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          margin-bottom: 1rem;
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          align-items: center;
          justify-content: space-between;
        }
        header.toolbar h1 {
          margin: 0;
          font-size: clamp(1.2rem, 2vw, 1.6rem);
        }
        header.toolbar p {
          margin: 0.35rem 0 0;
          color: var(--muted);
          font-size: 0.9rem;
        }
        header.toolbar .summary {
          display: flex;
          gap: 1rem;
          font-size: 0.95rem;
          align-items: center;
          flex-wrap: wrap;
        }
        header.toolbar .summary span strong {
          font-variant-numeric: tabular-nums;
          color: var(--accent);
        }
        header.toolbar button {
          background: var(--accent);
          color: #0b1221;
          border: none;
          border-radius: 999px;
          padding: 0.65rem 1.5rem;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
        }
        header.toolbar button[disabled] {
          opacity: 0.6;
          cursor: not-allowed;
        }
        header.toolbar button:hover:not([disabled]) {
          transform: translateY(-1px);
          box-shadow: 0 12px 24px rgba(79,157,255,0.35);
        }
        .process-batch {
          background: linear-gradient(135deg, var(--warning) 0%, #ffb700 100%) !important;
          color: black !important;
          box-shadow: 0 4px 12px rgba(255,212,59,0.3) !important;
        }
        .batch-info {
          color: var(--muted);
          font-size: 0.9rem;
          margin-left: 1rem;
        }
        .batch-info strong {
          color: var(--warning);
        }
        #status {
          width: 100%;
          font-size: 0.95rem;
          font-weight: 500;
          color: var(--muted);
        }
        #status.success { color: #4dd78a; }
        #status.error { color: var(--danger); }
        main {
          padding: 120px 1rem 1rem 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          max-width: 1800px;
          margin: 0 auto;
        }
        section.group {
          background: var(--surface);
          border-radius: 12px;
          padding: 0.75rem;
          box-shadow: 0 8px 16px rgba(0,0,0,0.15);
          border: 2px solid transparent;
          transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        section.group.in-batch {
          border-color: var(--accent);
          box-shadow: 0 20px 40px rgba(0,0,0,0.25), 0 0 0 1px rgba(79,157,255,0.2);
        }
        section.group:nth-of-type(odd) {
          background: var(--surface-alt);
        }
        section.group header.group-header {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          font-size: 0.7rem;
          color: rgba(160,163,177,0.5);
          margin-bottom: 0.25rem;
        }
        section.group header.group-header .location {
          font-size: 0.85rem;
          color: rgba(255,255,255,0.6);
        }
        .image-row {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem;
          margin-bottom: 0.25rem;
          background: var(--surface);
          border-radius: 8px;
          border: 2px solid transparent;
          transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .images-container {
          display: grid;
          grid-auto-flow: column;
          grid-auto-columns: 1fr;
          justify-content: center;
          gap: 0.5rem;
          flex: 1;
        }
        figure.image-card {
          margin: 0;
          background: var(--surface-alt);
          border-radius: 8px;
          padding: 0.5rem;
          text-align: center;
          border: 2px solid transparent;
          transition: all 0.3s ease;
          cursor: pointer;
          display: flex;
          flex-direction: column;
          align-items: center;
          position: relative;
        }
        figure.image-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 16px 32px rgba(0,0,0,0.25);
        }
        figure.image-card img {
          max-width: 100%;
          max-height: 60vh;
          width: auto;
          height: auto;
          object-fit: contain;
          border-radius: 8px;
          margin-bottom: 0.5rem;
        }
        figure.image-card .stage {
          display: none;
          background: rgba(0,0,0,0.6);
          border-radius: 999px;
          padding: 0.3rem 0.7rem;
          font-size: 0.75rem;
          letter-spacing: 0.02em;
          text-transform: uppercase;
        }
        figure.image-card .filename {
          font-size: 0.65rem;
          color: rgba(160,163,177,0.6);
          margin-bottom: 0.25rem;
          word-break: break-all;
          line-height: 1.2;
        }
        figure.image-card.selected {
          border-color: var(--accent);
          box-shadow: 0 0 0 2px rgba(79,157,255,0.3);
        }
        figure.image-card.selected.crop-selected {
          border-color: white;
          box-shadow: 0 0 0 2px rgba(255,255,255,0.6);
        }
        figure.image-card.delete-hint {
          border-color: var(--danger);
          box-shadow: 0 0 0 2px rgba(255,107,107,0.3);
        }
        footer.page-end {
          text-align: center;
          padding: 2rem;
          color: var(--muted);
        }
        @media (max-width: 720px) {
          header.toolbar {
            flex-direction: column;
            align-items: flex-start;
          }
          header.toolbar .summary {
            width: 100%;
            justify-content: space-between;
          }
        }
      </style>
    </head>
    <body>
      <div class="main-layout">
        <div class="content-area">
      <header class="toolbar">
        <div>
          <h1>Image version selector</h1>
              <p>Use right sidebar or keys: 1,2,3 (select) • Enter (next) • ↑ (back)</p>
        </div>
        <div class="summary" id="summary">
          <span>Selected: <strong id="summary-selected">0</strong></span>
          <span>Selected: <strong id="summary-selected-count">0</strong></span>
          <span>Skipped: <strong id="summary-skipped">0</strong></span>
              <span>Deleting: <strong id="summary-delete">{{ groups|length }}</strong></span>
        </div>
        <button id="process-batch" class="process-batch">Process Current Batch</button>
        <div id="batch-info" class="batch-info">
              <span>Batch {{ batch_info.current_batch }}/{{ batch_info.total_batches }}: <strong id="batch-count">{{ batch_info.current_batch_size }}</strong> groups</span>
        </div>
        <div id="status"></div>
      </header>
      <main>
            {% for group in groups %}
            <section class="group" data-group-id="{{ group.id }}" id="group-{{ group.id }}">
              <header class="group-header">
                <span>{% if group.images|length == 3 %}Triplet{% else %}Pair{% endif %} {{ group.display_index }} / {{ total_groups }}</span>
                {% if group.relative_dir %}
                <span class="location">{{ group.relative_dir }}</span>
            {% endif %}
          </header>
          <div class="image-row">
            <div class="images-container">
                  {% for image in group.images %}
              <figure class="image-card" data-image-index="{{ image.index }}" onclick="handleImageClick(this)">
              <div class="stage">{{ image.stage }}</div>
              <img src="{{ image.src }}" alt="{{ image.name }}" loading="lazy">
              <figcaption class="filename">{{ image.name }}</figcaption>
            </figure>
            {% endfor %}
          </div>
            <div class="row-buttons">
              <button class="action-btn row-btn-1" onclick="selectImage(0, '{{ group.id }}')">1</button>
              <button class="action-btn row-btn-2" onclick="selectImage(1, '{{ group.id }}')">2</button>
              {% if group.images|length >= 3 %}
              <button class="action-btn row-btn-3" onclick="selectImage(2, '{{ group.id }}')">3</button>
              {% endif %}
              <button class="action-btn row-btn-next" onclick="nextGroup()">▼</button>
            </div>
          </div>
        </section>
        {% endfor %}
      </main>
      <footer class="page-end">
        Re-run the script to continue with newly added images.
      </footer>
        </div>
      </div>
      <script>
        const groups = Array.from(document.querySelectorAll('section.group'));
        const summarySelected = document.getElementById('summary-selected');
        const summarySelectedCount = document.getElementById('summary-selected-count');
        const summarySkipped = document.getElementById('summary-skipped');
        const summaryDelete = document.getElementById('summary-delete');
        const processBatchButton = document.getElementById('process-batch');
        const batchCount = document.getElementById('batch-count');
        const statusBox = document.getElementById('status');
        
        let groupStates = {}; // { groupId: { selectedImage: 0|1|2, skipped: boolean } }

        function updateSummary() {
          const selectedCount = Object.values(groupStates).filter(state => state.selectedImage !== undefined && !state.skipped).length;
          const skippedCount = Object.values(groupStates).filter(state => state.skipped).length;
          const deleteCount = groups.length - selectedCount - skippedCount;
          
          summarySelected.textContent = selectedCount;
          summarySelectedCount.textContent = selectedCount;
          summarySkipped.textContent = skippedCount;
          summaryDelete.textContent = Math.max(0, deleteCount);
          batchCount.textContent = selectedCount;
        }
        
        function updateButtonStates(groupId) {
          const group = document.querySelector(`section.group[data-group-id="${groupId}"]`);
          if (!group) return;
          
          const state = groupStates[groupId];
          const imageCount = group.querySelectorAll('.image-card').length;
          
          // Reset button states for this group
          group.querySelectorAll('.action-btn').forEach(btn => {
            btn.classList.remove('crop-active', 'image-active');
          });
          
          if (state) {
            // Highlight selected image button
            if (state.selectedImage !== undefined) {
              const imageButton = group.querySelector(`.row-btn-${state.selectedImage + 1}`);
              if (imageButton) {
                imageButton.classList.add('image-active');
              }
            }
            
            // Crop button removed - all selections go to crop automatically
          }
        }
        
        function updateVisualState() {
          groups.forEach((group, index) => {
            const groupId = group.dataset.groupId;
            const state = groupStates[groupId];
            
            // Clear all visual states
            group.querySelectorAll('.image-card').forEach(card => {
              card.classList.remove('selected', 'delete-hint', 'crop-selected');
            });
            
            if (state && state.skipped) {
              // Skipped group - no visual indication (neutral state)
              // Images have no outline, indicating they'll be left alone
            } else if (state && state.selectedImage !== undefined) {
              // Show selected image
              const selectedCard = group.querySelectorAll('.image-card')[state.selectedImage];
              if (selectedCard) {
                selectedCard.classList.add('selected');
                // All selected images automatically go to selected
                selectedCard.classList.add('crop-selected');
              }
              
              // Show delete hint on other images
              group.querySelectorAll('.image-card').forEach((card, cardIndex) => {
                if (cardIndex !== state.selectedImage) {
                  card.classList.add('delete-hint');
                }
              });
            } else {
              // No selection = all images will be deleted
              group.querySelectorAll('.image-card').forEach(card => {
                card.classList.add('delete-hint');
              });
            }
            
            // Update button states for this group
            updateButtonStates(groupId);
          });
        }
        
        function selectImage(imageIndex, groupId) {
          const group = document.querySelector(`section.group[data-group-id="${groupId}"]`);
          if (!group) return;
          
          const imageCount = group.querySelectorAll('.image-card').length;
          if (imageIndex >= imageCount) return;
          
          // BUTTON TOGGLE: If same image already selected, deselect it
          const currentState = groupStates[groupId];
          if (currentState && currentState.selectedImage === imageIndex) {
            delete groupStates[groupId]; // Deselect (back to delete)
          } else {
            // Update state - select image (automatically goes to selected)
            groupStates[groupId] = { selectedImage: imageIndex, skipped: false };
          }
          
          // Update everything immediately
          updateVisualState();
          updateSummary();
        }
        
        // toggleCrop function removed - all selections automatically go to crop
        
        // selectImageAndCrop function removed - regular selectImage now handles all cases
        
        function handleImageClick(imageCard) {
          const group = imageCard.closest('section.group');
          const groupId = group.dataset.groupId;
          const imageIndex = parseInt(imageCard.dataset.imageIndex);
          const currentState = groupStates[groupId];
          
          if (!currentState || (!currentState.skipped && currentState.selectedImage === undefined)) {
            // First click: skip this group (leave in directory)
            groupStates[groupId] = { skipped: true };
          } else if (currentState.skipped) {
            // Second click: back to delete mode
            delete groupStates[groupId];
          } else if (currentState.selectedImage === imageIndex) {
            // UNSELECT FUNCTIONALITY: Clicking selected image deselects it (back to delete)
            delete groupStates[groupId];
          }
          // If different image is selected, clicking does nothing (use buttons for that)
          
          updateVisualState();
          updateSummary();
        }
        
        function nextGroup() {
          // Find the currently visible group using the same logic as keyboard shortcuts
          const currentGroupId = getCurrentVisibleGroupId();
          if (!currentGroupId) return;
          
          const currentGroupElement = document.querySelector(`section.group[data-group-id="${currentGroupId}"]`);
          if (!currentGroupElement) return;
          
          const allGroups = Array.from(document.querySelectorAll('section.group'));
          const currentIndex = allGroups.indexOf(currentGroupElement);
          const nextGroupElement = allGroups[currentIndex + 1];
          
          if (nextGroupElement) {
            nextGroupElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
        
        function previousGroup() {
          // UP ARROW NAVIGATION: Go back one row (opposite of nextGroup)
          const currentGroupId = getCurrentVisibleGroupId();
          if (!currentGroupId) return;
          
          const currentGroupElement = document.querySelector(`section.group[data-group-id="${currentGroupId}"]`);
          if (!currentGroupElement) return;
          
          const allGroups = Array.from(document.querySelectorAll('section.group'));
          const currentIndex = allGroups.indexOf(currentGroupElement);
          const previousGroupElement = allGroups[currentIndex - 1];
          
          if (previousGroupElement) {
            previousGroupElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
        
        // Get the currently visible group ID for keyboard shortcuts
        function getCurrentVisibleGroupId() {
          const viewportCenter = window.scrollY + window.innerHeight / 2;
          let closestGroup = null;
          let closestDistance = Infinity;
          
          groups.forEach((group) => {
            const rect = group.getBoundingClientRect();
            const groupCenter = rect.top + window.scrollY + rect.height / 2;
            const distance = Math.abs(groupCenter - viewportCenter);
            
            if (distance < closestDistance) {
              closestDistance = distance;
              closestGroup = group;
            }
          });
          
          return closestGroup ? closestGroup.dataset.groupId : null;
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
          // Ignore if user is typing in an input
          if (e.target.matches('input, textarea, select')) return;
          
          const currentGroupId = getCurrentVisibleGroupId();
          if (!currentGroupId) return;
          
          switch(e.key) {
            case '1':
              e.preventDefault();
              selectImage(0, currentGroupId);
              break;
            case '2':
              e.preventDefault();
              selectImage(1, currentGroupId);
              break;
            case '3':
              e.preventDefault();
              selectImage(2, currentGroupId);
              break;
            case 'Enter':
              e.preventDefault();
              nextGroup(); // ENTER: Go forward (main navigation key)
              break;
            case 'ArrowUp':
              e.preventDefault();
              previousGroup(); // UP ARROW: Go back one row
              break;
          }
        });
        
        function setStatus(message, type = '') {
          statusBox.textContent = message;
          statusBox.className = type ? type : '';
        }

        // PROCESS BUTTON SAFETY: Only enable after scrolling to bottom
        let hasScrolledToBottom = false;
        
        function checkScrollPosition() {
          const scrollPercent = (window.scrollY + window.innerHeight) / document.body.scrollHeight;
          if (scrollPercent >= 0.9 && !hasScrolledToBottom) {
            hasScrolledToBottom = true;
            processBatchButton.disabled = false;
            processBatchButton.style.opacity = '1';
            processBatchButton.title = 'Ready to process batch';
          }
        }
        
        // Initialize process button as disabled
        processBatchButton.disabled = true;
        processBatchButton.style.opacity = '0.5';
        processBatchButton.title = 'Scroll to bottom of page to enable';
        
        // Listen for scroll events
        window.addEventListener('scroll', checkScrollPosition);
        
        processBatchButton.addEventListener('click', async () => {
          if (Object.keys(groupStates).length === 0) {
            setStatus('No groups selected in current batch to process', 'error');
            return;
          }
          
          if (!confirm(`Process current batch of ${Object.keys(groupStates).length} selected groups? This will move/delete files immediately.`)) {
            return;
          }
          
          processBatchButton.disabled = true;
          setStatus('Processing batch…');
          
          // For batch processing, only send selections for current batch groups
          const selections = groups.map(group => {
            const groupId = parseInt(group.dataset.groupId);
            const state = groupStates[groupId];
            return {
              groupId: groupId,
              selectedIndex: state ? state.selectedImage : null,
              selected: state ? (state.selectedImage !== undefined) : false, // All selections go to selected
              skipped: state ? state.skipped : false,
            };
          });
          
          try {
            const response = await fetch('/submit', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ selections: selections, batch_mode: true })
            });
            const payload = await response.json();
            if (!response.ok) {
              throw new Error(payload.message || 'Server error');
            }
            
            setStatus(`Batch processed! ${payload.moved || 0} moved, ${payload.deleted || 0} deleted`, 'success');
            
            // Check if this batch is complete and auto-advance
            if (payload.remaining === 0) {
              // Batch is empty, check for next batch
              setTimeout(async () => {
                try {
                  const nextResponse = await fetch('/next-batch', { method: 'POST' });
                  const nextResult = await nextResponse.json();
                  
                  if (nextResult.status === 'success') {
                    // Switch to next batch
                    const switchResponse = await fetch('/switch-batch/' + nextResult.next_batch);
                    const switchResult = await switchResponse.json();
                    
                    if (switchResult.status === 'success') {
                      window.scrollTo(0, 0); // Scroll to top before reload
                      window.location.reload();
                    }
                  } else if (nextResult.status === 'complete') {
                    setStatus('All batches processed! 🎉', 'success');
                  }
                } catch (error) {
                  // Fallback to simple reload
                  window.scrollTo(0, 0); // Scroll to top before reload
                  window.location.reload();
                }
              }, 1000);
            } else {
              // Still have groups in current batch, just reload
              setTimeout(() => {
                window.scrollTo(0, 0); // Scroll to top before reload
                window.location.reload();
              }, 1000);
            }
            
          } catch (error) {
            console.error(error);
            processBatchButton.disabled = false;
            setStatus(error.message || 'Unable to process batch', 'error');
          }
        });
        
        // Initialize
        updateVisualState();
        updateSummary();
        
        // Add simple activity timer status to header
        const header = document.querySelector('h1');
        if (header) {
            const timerStatus = document.createElement('span');
            timerStatus.id = 'activity-status';
            timerStatus.style.cssText = `
                margin-left: 1rem;
                font-size: 0.8rem;
                font-weight: normal;
                color: var(--text-secondary);
            `;
            timerStatus.innerHTML = '🟢 Active';
            header.appendChild(timerStatus);
        }

        // Update timer display every 5 seconds
        setInterval(updateTimerDisplay, 5000);
        
        // Track user activity
        function markActivity() {
            fetch('/mark_activity', { method: 'POST' });
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
        
        function updateTimerDisplay() {
            fetch('/timer_stats')
                .then(response => response.json())
                .then(data => {
                    if (data.active_time !== undefined) {
                        // Update simple status indicator in header
                        const statusElement = document.getElementById('activity-status');
                        if (statusElement) {
                            statusElement.innerHTML = data.is_active ? '🟢 Active' : '⚫ Inactive';
                        }
                    }
                })
                .catch(err => console.log('Timer update failed:', err));
        }
        
        // Initial timer update
        updateTimerDisplay();
        
        // Scroll to first group
        if (groups.length > 0) {
          groups[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      </script>
    </body>
    </html>
    """

    AUTO_ADVANCE_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Batch Complete - Auto Advance</title>
      <style>
        :root { color-scheme: dark; --bg: #101014; --surface: #1a1a20; --accent: #4f9dff; }
        body { font-family: system-ui; background: var(--bg); color: white; text-align: center; padding: 4rem; }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { color: var(--accent); margin-bottom: 2rem; }
        .info { background: var(--surface); padding: 2rem; border-radius: 12px; margin: 2rem 0; }
        button { background: var(--accent); color: white; border: none; padding: 1rem 2rem; border-radius: 8px; font-size: 1.1rem; cursor: pointer; margin: 0.5rem; }
        button:hover { opacity: 0.9; }
        .timer { font-size: 1.2rem; margin: 1rem 0; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>✅ Batch {{ current_batch }} Complete!</h1>
        <div class="info">
          <p>Batch {{ current_batch }} of {{ total_batches }} finished.</p>
          <p>Ready to advance to <strong>Batch {{ next_batch }}</strong>?</p>
          <div class="timer">Auto-advancing in <span id="countdown">10</span> seconds...</div>
        </div>
        <button onclick="advanceNow()">Go Now</button>
        <button onclick="skipToNextBatch()">Next Batch</button>
      </div>
      
      <script>
        let countdown = 10;
        const timer = setInterval(() => {
          countdown--;
          document.getElementById('countdown').textContent = countdown;
          if (countdown <= 0) {
            advanceNow();
          }
        }, 1000);
        
        async function advanceNow() {
          clearInterval(timer);
          try {
            const response = await fetch('/switch-batch/{{ next_batch }}');
            const result = await response.json();
            
            if (result.status === 'success') {
              window.location.reload();
            } else {
              alert('Error switching batches: ' + result.message);
            }
          } catch (error) {
            alert('Error advancing to next batch: ' + error.message);
          }
        }
        
        async function skipToNextBatch() {
          try {
            const response = await fetch('/next-batch', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'}
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
              const switchResponse = await fetch('/switch-batch/' + result.next_batch);
              const switchResult = await switchResponse.json();
              
              if (switchResult.status === 'success') {
                window.location.reload();
              } else {
                alert('Error switching to next batch: ' + switchResult.message);
              }
            } else if (result.status === 'complete') {
              alert('All batches have been processed!');
            } else {
              alert('Error: ' + result.message);
            }
          } catch (error) {
            alert('Error getting next batch: ' + error.message);
          }
        }
      </script>
    </body>
    </html>
    """

    COMPLETION_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>All Batches Complete</title>
      <style>
        :root { color-scheme: dark; --bg: #101014; --surface: #1a1a20; --success: #51cf66; }
        body { font-family: system-ui; background: var(--bg); color: white; text-align: center; padding: 4rem; }
        h1 { color: var(--success); }
      </style>
    </head>
    <body>
      <h1>🎉 All Batches Complete!</h1>
      <p>Image selection finished for all groups.</p>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        batch_info = app.config["BATCH_INFO"]
        current_groups = app.config["GROUP_PAYLOAD"]
        
        # Check if batch is complete and handle auto-advance
        if not current_groups:
            # Batch is empty, check for next batch
            if batch_info["current_batch"] < batch_info["total_batches"]:
                # Show auto-advance screen
                next_batch_num = batch_info["current_batch"] + 1
                return render_template_string(AUTO_ADVANCE_TEMPLATE, 
                                            next_batch=next_batch_num,
                                            current_batch=batch_info["current_batch"],
                                            total_batches=batch_info["total_batches"])
            else:
                # All batches complete
                return render_template_string(COMPLETION_TEMPLATE)
        
        return render_template_string(
            page_template,
            groups=current_groups,
            total_groups=batch_info["total_groups"],
            batch_info=batch_info,
        )

    @app.route("/image/<int:group_id>/<int:image_index>")
    def serve_image(group_id: int, image_index: int):
        try:
            # group_id is global index, resolve from ALL_GROUPS
            record: GroupRecord = app.config["ALL_GROUPS"][group_id]
            path = record.paths[image_index]
        except (IndexError, KeyError):
            return Response(status=404)
        if not path.exists():
            return Response(status=404)
        stat = path.stat()
        data = _generate_thumbnail(str(path), int(stat.st_mtime_ns), stat.st_size)
        return Response(data, mimetype="image/jpeg")

    @app.route("/mark_activity", methods=["POST"])
    def mark_activity():
        """Mark user activity for timer tracking"""
        activity_timer: Optional[ActivityTimer] = app.config.get("ACTIVITY_TIMER")
        if activity_timer:
            activity_timer.mark_activity()
        return jsonify({"status": "ok"})
    
    @app.route("/timer_stats")
    def timer_stats():
        """Get current timer statistics"""
        activity_timer: Optional[ActivityTimer] = app.config.get("ACTIVITY_TIMER")
        if activity_timer:
            stats = activity_timer.get_current_stats()
            return jsonify(stats)
        else:
            return jsonify({
                "active_time": 0,
                "total_time": 0,
                "efficiency": 100,
                "is_active": False,
                "files_processed": 0,
                "total_operations": 0
            })

    def process_shutdown():
        # Graceful shutdown - just exit the process
        # The Flask dev server will handle cleanup
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)

    @app.route("/submit", methods=["POST"])
    def submit():
        if app.config.get("PROCESSED"):
            return jsonify({"status": "error", "message": "Selections already applied."}), 400

        payload = request.get_json(silent=True) or {}
        selections = payload.get("selections")
        batch_mode = payload.get("batch_mode", False)
        
        if not isinstance(selections, list):
            return jsonify({"status": "error", "message": "Selection payload missing or invalid."}), 400
        
        # Only batch mode is supported now (finalize mode removed for safety)
        current_groups = app.config["GROUPS"]
        all_groups = app.config["ALL_GROUPS"]
        
        if not batch_mode:
            return jsonify({"status": "error", "message": "Only batch processing is supported."}), 400
            
        # For batch processing, expect selections for current batch only
        expected_count = len(current_groups)
            
        if len(selections) != expected_count:
            return jsonify({
                "status": "error", 
                "message": f"Selection payload missing or invalid. Expected {expected_count}, got {len(selections)}."
            }), 400

        tracker: FileTracker = app.config["TRACKER"]
        base_folder: Path = app.config["BASE_FOLDER"]
        log_path: Path = app.config["LOG_PATH"]
        hard_delete: bool = app.config["HARD_DELETE"]
        activity_timer: Optional[ActivityTimer] = app.config.get("ACTIVITY_TIMER")

        kept_count = 0
        crop_count = 0
        
        # Mark batch processing activity
        if activity_timer:
            activity_timer.mark_batch(f"Processing batch {batch_info['current_batch']}")
            activity_timer.mark_activity()

        try:
            for i, selection in enumerate(selections):
                group_id = selection.get("groupId")
                try:
                    # For batch mode, group_id is global but we need to find it in current batch
                    record = None
                    for group_record in current_groups:
                        if group_record.index == group_id:
                            record = group_record
                            break
                    if record is None:
                        raise ValueError(f"Group id {group_id} not found in current batch")
                except (TypeError, KeyError, IndexError):
                    raise ValueError(f"Invalid group id: {group_id}")

                selected_index = selection.get("selectedIndex")
                crop_flag = bool(selection.get("crop"))
                skipped = bool(selection.get("skipped"))

                if skipped:
                    # Skip this group - leave files in place
                    write_log(log_path, "skip", None, list(record.paths))
                    continue

                if selected_index is None:
                    safe_delete(record.paths, hard_delete=hard_delete, tracker=tracker)
                    write_log(log_path, "delete_all", None, list(record.paths))
                    continue

                max_index = len(record.paths) - 1
                if not isinstance(selected_index, int) or selected_index < 0 or selected_index > max_index:
                    raise ValueError(f"Invalid selection index for group {group_id}")

                selected_path = record.paths[selected_index]
                others = [p for idx, p in enumerate(record.paths) if idx != selected_index]

                # Always move selected images to selected directory
                target_path = move_to_selected(selected_path, tracker=tracker)
                crop_count += 1
                action_label = "keep_one_to_selected"

                safe_delete(others, hard_delete=hard_delete, tracker=tracker)
                write_log(log_path, action_label, target_path, others)
                kept_count += 1

        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

        deleted_count = len(selections) - kept_count
        
        # Remove ALL processed groups from current batch (both selected and deleted)
        processed_group_ids = [s.get("groupId") for s in selections]
        remaining_groups = [group for group in app.config["GROUPS"] if group.index not in processed_group_ids]
        app.config["GROUPS"] = remaining_groups
        app.config["GROUP_PAYLOAD"] = [record.as_payload(total=len(app.config["ALL_GROUPS"])) for record in remaining_groups]
        
        # Update batch info
        batch_info = app.config["BATCH_INFO"]
        batch_info["current_batch_size"] = len(remaining_groups)
        
        message = f"Batch processed — kept {kept_count}, sent {crop_count} to selected/, deleted {deleted_count}."
        return jsonify({
                "status": "ok", 
                "message": message,
                "moved": kept_count,
            "deleted": deleted_count,
            "remaining": len(remaining_groups)
        })

    @app.route("/switch-batch/<int:batch_num>")
    def switch_batch(batch_num):
        """Switch to a new batch without restarting the server."""
        batch_info = app.config["BATCH_INFO"]
        all_groups = app.config["ALL_GROUPS"]
        
        if batch_num < 1 or batch_num > batch_info["total_batches"]:
            return jsonify({"status": "error", "message": f"Invalid batch number {batch_num}"}), 404
        
        # Calculate new batch boundaries
        batch_size = batch_info["batch_size"]
        new_batch_start = (batch_num - 1) * batch_size
        new_batch_end = min(new_batch_start + batch_size, len(all_groups))
        new_batch_groups = all_groups[new_batch_start:new_batch_end]
        new_batch_payload = [record.as_payload(total=len(all_groups)) for record in new_batch_groups]
        
        # Update batch info
        new_batch_info = {
            "current_batch": batch_num,
            "total_batches": batch_info["total_batches"],
            "batch_start": new_batch_start,
            "batch_end": new_batch_end,
            "batch_size": batch_size,
            "total_groups": len(all_groups),
            "current_batch_size": len(new_batch_groups)
        }
        
        # Update app config with new batch
        app.config["GROUPS"] = new_batch_groups
        app.config["GROUP_PAYLOAD"] = new_batch_payload
        app.config["BATCH_INFO"] = new_batch_info
        
        return jsonify({
            "status": "success", 
            "message": f"Switched to batch {batch_num}",
            "group_count": len(new_batch_groups),
            "batch": batch_num,
            "total_batches": new_batch_info["total_batches"]
        })

    @app.route("/next-batch", methods=["POST"])
    def next_batch():
        """Get the next batch for auto-advance."""
        batch_info = app.config["BATCH_INFO"]
        
        if batch_info["current_batch"] < batch_info["total_batches"]:
            # More batches available
            next_batch_num = batch_info["current_batch"] + 1
            return jsonify({
                "status": "success", 
                "next_batch": next_batch_num,
                "current_batch": batch_info["current_batch"],
                "total_batches": batch_info["total_batches"]
            })
        else:
            # All batches complete
            return jsonify({"status": "complete", "message": "All batches processed"})
    return app


def launch_browser(host: str, port: int) -> None:
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return
    url = f"http://{host}:{port}/"
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scroll through image triplets in a browser and batch apply decisions.",
    )
    parser.add_argument("folder", type=str, help="Folder containing image triplets")
    parser.add_argument("--exts", type=str, default="png", help="Comma-separated list of extensions to include")
    parser.add_argument("--print-triplets", action="store_true", help="Print grouped triplets and exit")
    parser.add_argument("--hard-delete", action="store_true", help="Permanently delete files instead of send2trash")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of groups to process per batch (default: 100)")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for the local web server")
    parser.add_argument("--port", type=int, default=8080, help="Port for the local web server")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories recursively")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        human_err(f"Folder not found: {folder}")
        sys.exit(1)

    tracker = FileTracker("image_version_selector")

    exts = [ext.strip() for ext in args.exts.split(",") if ext.strip()]
    recursive = not args.no_recursive
    files = scan_images(folder, exts, recursive)
    if not files:
        human_err("No images found. Check --exts or folder path.")
        sys.exit(1)

    group_paths = find_flexible_groups(files)
    if not group_paths:
        human_err("No groups found with the current grouping. Try adjusting filenames or timestamps.")
        sys.exit(1)

    if args.print_triplets:
        for idx, group in enumerate(group_paths, 1):
            group_type = "Triplet" if len(group) == 3 else "Pair"
            print(f"\n{group_type} {idx}:")
            for path in group:
                print("  -", path.name)
        triplet_count = sum(1 for group in group_paths if len(group) == 3)
        pair_count = sum(1 for group in group_paths if len(group) == 2)
        print(f"\nTotal: {triplet_count} triplets, {pair_count} pairs ({len(group_paths)} groups)")
        return

    if not args.hard_delete and not _SEND2TRASH_AVAILABLE:
        human_err("send2trash not installed. Install it with: pip install send2trash")
        human_err("Or rerun with --hard-delete to permanently delete files (dangerous).")
        sys.exit(1)

    records: List[GroupRecord] = []
    for idx, group in enumerate(group_paths):
        first_parent = group[0].parent
        try:
            relative = str(first_parent.relative_to(folder))
        except ValueError:
            relative = str(first_parent)
        records.append(
            GroupRecord(
                index=idx,
                paths=group,
                relative_dir=relative if relative != "." else "",
            )
        )

    # Initialize activity timer
    activity_timer = ActivityTimer("01_web_image_selector")
    activity_timer.start_session()
    
    log_path = folder / "triplet_culler_log.csv"
    app = build_app(records, folder, tracker, log_path, hard_delete=args.hard_delete, batch_size=args.batch_size, activity_timer=activity_timer)

    # Calculate batch info for logging
    total_groups = len(records)
    total_batches = (total_groups + args.batch_size - 1) // args.batch_size
    first_batch_size = min(args.batch_size, total_groups)

    url = f"http://{args.host}:{args.port}"
    info(f"Found {total_groups} groups. Starting with batch 1/{total_batches} ({first_batch_size} groups). Launching browser UI at {url}")
    info("Use CTRL+C to stop the server after it reports that processing is complete.")

    if not args.no_browser:
        threading.Thread(target=launch_browser, args=(args.host, args.port), daemon=True).start()

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except OSError as exc:
        human_err(f"Failed to start server: {exc}")
        sys.exit(1)
    finally:
        # End activity timer session
        if 'activity_timer' in locals():
            activity_timer.end_session()


if __name__ == "__main__":
    main()