#!/usr/bin/env python3
"""
Step 1: Web Image Selector - Modern Browser Edition
====================================================
Modern web-based tool for selecting the best image from each triplet set.
Replaces old Matplotlib interface with fast, scrollable browser interface.

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
• Option to send selections directly to crop/ or Reviewed/ folders
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
   scripts/util_similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

HOW IT WORKS:
---------
1. Click exactly one image in each row to keep it
2. Leave a row untouched if you want all three images deleted
3. Optionally enable **Send to crop/** toggle to move winners to crop/ instead of Reviewed/
4. Press **Finalize selections** when done
5. The script will:
   • Move chosen images (+ YAML files) to Reviewed/ or crop/
   • Delete unselected images using send2trash (recoverable)
   • Log all actions in triplet_culler_log.csv and FileTracker logs

OPTIONAL FLAGS:
---------------
  --exts            File extensions to include (default: png)
  --hard-delete     Permanently delete instead of using trash
  --host/--port     Web server binding (default: 127.0.0.1:5000)
  --no-browser      Don't auto-launch browser
  --print-triplets  Show triplet groupings and exit (debug)

HOW IT WORKS:
-------------
1. Click exactly one image in each row to keep it
2. Leave a row untouched if you want all three images deleted
3. Optionally enable **Send to crop/** toggle to move winners to crop/ instead of Reviewed/
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


def scan_images(folder: Path, exts: Iterable[str]) -> List[Path]:
    allowed = {e.lower().lstrip(".") for e in exts}
    results: List[Path] = []
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
class TripletRecord:
    index: int
    paths: Tuple[Path, Path, Path]
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


def find_triplets(files: List[Path]) -> List[Tuple[Path, Path, Path]]:
    triplets: List[Tuple[Path, Path, Path]] = []
    stage1_files = []
    stage15_files = []
    stage2_files = []

    for file in files:
        stage = detect_stage(file.name)
        timestamp = extract_timestamp(file.name)
        timestamp_minutes = timestamp_to_minutes(timestamp)

        if stage == "stage1_generated" and timestamp_minutes is not None:
            stage1_files.append((file, timestamp_minutes, timestamp))
        elif stage == "stage1.5_face_swapped" and timestamp_minutes is not None:
            stage15_files.append((file, timestamp_minutes, timestamp))
        elif stage == "stage2_upscaled" and timestamp_minutes is not None:
            stage2_files.append((file, timestamp_minutes, timestamp))

    stage1_files.sort(key=lambda x: x[1])
    stage15_files.sort(key=lambda x: x[1])
    stage2_files.sort(key=lambda x: x[1])

    max_gap_stage1_to_15 = 0.5
    max_gap_stage15_to_2 = 2

    for stage1_file, stage1_time, stage1_ts in stage1_files:
        stage1_date = get_date_from_timestamp(stage1_ts)
        if not stage1_date:
            continue

        stage15_match = None
        for stage15_file, stage15_time, stage15_ts in stage15_files:
            stage15_date = get_date_from_timestamp(stage15_ts)
            if (
                stage15_date == stage1_date
                and stage1_time <= stage15_time <= stage1_time + max_gap_stage1_to_15
            ):
                stage15_match = (stage15_file, stage15_time, stage15_ts)
                break

        if not stage15_match:
            continue

        stage15_file, stage15_time, stage15_ts = stage15_match

        stage2_match = None
        for stage2_file, stage2_time, stage2_ts in stage2_files:
            stage2_date = get_date_from_timestamp(stage2_ts)
            if (
                stage2_date == stage1_date
                and stage15_time <= stage2_time <= stage15_time + max_gap_stage15_to_2
            ):
                stage2_match = (stage2_file, stage2_time, stage2_ts)
                break

        if not stage2_match:
            continue

        stage2_file, _, stage2_ts = stage2_match
        triplets.append((stage1_file, stage15_file, stage2_file))

        stage15_files = [f for f in stage15_files if f[2] != stage15_ts]
        stage2_files = [f for f in stage2_files if f[2] != stage2_ts]

    return triplets


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


def move_to_reviewed(src_path: Path, base_folder: Path, tracker: FileTracker) -> Path:
    reviewed_dir = base_folder.parent / "Reviewed"
    return move_with_yaml(
        src_path,
        reviewed_dir,
        tracker,
        dest_label="Reviewed",
        notes="Selected image from triplet selector",
    )


def move_to_crop(src_path: Path, tracker: FileTracker) -> Path:
    project_root = find_project_root(src_path.parent)
    crop_dir = project_root / "crop"
    return move_with_yaml(
        src_path,
        crop_dir,
        tracker,
        dest_label="crop",
        notes="Marked for crop during triplet selection",
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
    triplets: List[TripletRecord],
    base_folder: Path,
    tracker: FileTracker,
    log_path: Path,
    hard_delete: bool,
) -> Flask:
    app = Flask(__name__)

    triplet_payload = [record.as_payload(total=len(triplets)) for record in triplets]

    app.config["TRIPLETS"] = triplets
    app.config["TRIPLET_PAYLOAD"] = triplet_payload
    app.config["BASE_FOLDER"] = base_folder
    app.config["TRACKER"] = tracker
    app.config["LOG_PATH"] = log_path
    app.config["HARD_DELETE"] = hard_delete
    app.config["PROCESSED"] = False

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
          --muted: #a0a3b1;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
          background: var(--bg);
          color: #f8f9ff;
        }
        header.toolbar {
          position: sticky;
          top: 0;
          z-index: 10;
          background: linear-gradient(180deg, rgba(16,16,20,0.95), rgba(16,16,20,0.85));
          backdrop-filter: blur(16px);
          padding: 1.25rem clamp(1rem, 3vw, 2rem);
          border-bottom: 1px solid rgba(255,255,255,0.05);
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
          padding: clamp(1rem, 4vw, 2rem);
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
          max-width: 1400px;
          margin: 0 auto;
        }
        section.triplet {
          background: var(--surface);
          border-radius: 16px;
          padding: 1rem clamp(1rem, 4vw, 1.5rem);
          box-shadow: 0 20px 40px rgba(0,0,0,0.25);
          border: 2px solid transparent;
          transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        section.triplet.in-batch {
          border-color: var(--accent);
          box-shadow: 0 20px 40px rgba(0,0,0,0.25), 0 0 0 1px rgba(79,157,255,0.2);
        }
        section.triplet.reviewed {
          border-color: var(--success);
          box-shadow: 0 20px 40px rgba(0,0,0,0.25), 0 0 0 1px rgba(81,207,102,0.2);
        }
        section.triplet:nth-of-type(odd) {
          background: var(--surface-alt);
        }
        section.triplet header.triplet-header {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          font-size: 0.95rem;
          color: var(--muted);
          margin-bottom: 0.75rem;
        }
        section.triplet header.triplet-header .location {
          font-size: 0.85rem;
          color: rgba(255,255,255,0.6);
        }
        .image-row {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 0.75rem;
        }
        figure.image-card {
          margin: 0;
          border-radius: 14px;
          overflow: hidden;
          position: relative;
          cursor: pointer;
          background: rgba(255,255,255,0.04);
          transition: transform 0.12s ease, box-shadow 0.12s ease;
          border: 2px solid transparent;
        }
        figure.image-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 16px 32px rgba(0,0,0,0.25);
        }
        figure.image-card img {
          width: 100%;
          display: block;
          background: #050508;
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
          padding: 0.65rem 0.85rem;
          font-size: 0.82rem;
          color: rgba(255,255,255,0.72);
          word-break: break-word;
        }
        figure.image-card.selected {
          border-color: var(--accent);
          box-shadow: 0 0 0 2px rgba(79,157,255,0.2), 0 20px 30px rgba(79,157,255,0.2);
        }
        figure.image-card.selected::after {
          content: "Chosen";
          position: absolute;
          bottom: 12px;
          right: 12px;
          background: var(--accent);
          color: #0b1221;
          font-weight: 600;
          padding: 0.25rem 0.6rem;
          border-radius: 999px;
          font-size: 0.7rem;
          letter-spacing: 0.02em;
        }
        figure.image-card.delete-hint::after {
          content: "Will delete";
          position: absolute;
          bottom: 12px;
          right: 12px;
          background: rgba(255,107,107,0.18);
          color: rgba(255,171,171,0.9);
          font-weight: 500;
          padding: 0.25rem 0.6rem;
          border-radius: 999px;
          font-size: 0.7rem;
          letter-spacing: 0.02em;
        }
        .crop-toggle {
          margin-top: 0.75rem;
          padding: 0;
          border-radius: 12px;
          background: rgba(79,157,255,0.08);
          color: rgba(255,255,255,0.85);
          font-size: 0.9rem;
          cursor: pointer;
          transition: background-color 0.2s ease;
        }
        .crop-toggle:hover {
          background: rgba(79, 157, 255, 0.15);
          border: 1px solid rgba(79, 157, 255, 0.3);
        }
        .crop-toggle label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 0.85rem;
          cursor: pointer;
          width: 100%;
          margin: 0;
        }
        .crop-toggle.hidden {
          display: none;
        }
        .crop-toggle input[type="checkbox"] {
          width: 1.1rem;
          height: 1.1rem;
          accent-color: var(--accent);
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
      <header class="toolbar">
        <div>
          <h1>Image version selector</h1>
          <p>Click one image per row to keep it. Leave a row untouched to delete all three.</p>
        </div>
        <div class="summary" id="summary">
          <span>Selected: <strong id="summary-selected">0</strong></span>
          <span>To crop: <strong id="summary-crop">0</strong></span>
          <span>Deleting: <strong id="summary-delete">{{ triplets|length }}</strong></span>
        </div>
        <button id="finalize">Finalize selections</button>
        <button id="process-batch" class="process-batch">Process Current Batch</button>
        <div id="batch-info" class="batch-info">
          <span>Review batch: <strong id="batch-count">0</strong> triplets</span>
        </div>
        <div id="status"></div>
      </header>
      <main>
        {% for triplet in triplets %}
        <section class="triplet" data-triplet-id="{{ triplet.id }}">
          <header class="triplet-header">
            <span>Triplet {{ triplet.display_index }} / {{ total_triplets }}</span>
            {% if triplet.relative_dir %}
            <span class="location">{{ triplet.relative_dir }}</span>
            {% endif %}
          </header>
          <div class="image-row">
            {% for image in triplet.images %}
            <figure class="image-card" data-image-index="{{ image.index }}">
              <div class="stage">{{ image.stage }}</div>
              <img src="{{ image.src }}" alt="{{ image.name }}" loading="lazy">
              <figcaption class="filename">{{ image.name }}</figcaption>
            </figure>
            {% endfor %}
          </div>
          <div class="crop-toggle hidden">
            <label>
              <input type="checkbox" class="crop-checkbox">
              Send selected image to <code>crop/</code> instead of <code>Reviewed/</code>
            </label>
          </div>
        </section>
        {% endfor %}
      </main>
      <footer class="page-end">
        Re-run the script to continue with newly added images.
      </footer>
      <script>
        const triplets = Array.from(document.querySelectorAll('section.triplet'));
        const summarySelected = document.getElementById('summary-selected');
        const summaryCrop = document.getElementById('summary-crop');
        const summaryDelete = document.getElementById('summary-delete');
        const finalizeButton = document.getElementById('finalize');
        const processBatchButton = document.getElementById('process-batch');
        const batchCount = document.getElementById('batch-count');
        const statusBox = document.getElementById('status');
        const totalTriplets = triplets.length;
        
        // Progressive commit state
        let reviewBatchTriplets = new Set();
        let lastScrollTop = 0;

        function updateSummary() {
          let selectedCount = 0;
          let cropCount = 0;
          triplets.forEach(triplet => {
            const selectedIndex = triplet.dataset.selectedIndex;
            const cards = triplet.querySelectorAll('.image-card');
            const cropCheckbox = triplet.querySelector('.crop-checkbox');
            cards.forEach(card => {
              if (selectedIndex === undefined || selectedIndex === '') {
                card.classList.remove('selected');
                card.classList.add('delete-hint');
              } else if (card.dataset.imageIndex === selectedIndex) {
                card.classList.add('selected');
                card.classList.remove('delete-hint');
              } else {
                card.classList.remove('selected');
                card.classList.add('delete-hint');
              }
            });
            if (selectedIndex !== undefined && selectedIndex !== '') {
              selectedCount += 1;
              if (cropCheckbox.checked) {
                cropCount += 1;
              }
            }
          });
          summarySelected.textContent = selectedCount.toString();
          summaryCrop.textContent = cropCount.toString();
          summaryDelete.textContent = (totalTriplets - selectedCount).toString();
          
          // Update batch count
          batchCount.textContent = reviewBatchTriplets.size.toString();
        }
        
        function updateViewportBatch() {
          const viewportTop = window.scrollY;
          const viewportBottom = viewportTop + window.innerHeight;
          const buffer = 100; // Add some buffer for smooth transitions
          
          triplets.forEach((triplet, index) => {
            const rect = triplet.getBoundingClientRect();
            const elementTop = rect.top + window.scrollY;
            const elementBottom = elementTop + rect.height;
            
            // Check if element is in or has been in viewport
            if (elementTop < viewportBottom + buffer && elementBottom > viewportTop - buffer) {
              if (!reviewBatchTriplets.has(index)) {
                reviewBatchTriplets.add(index);
                triplet.classList.add('in-batch');
                // If user has made a selection, mark as reviewed
                if (triplet.dataset.selectedIndex !== undefined && triplet.dataset.selectedIndex !== '') {
                  triplet.classList.add('reviewed');
                  triplet.classList.remove('in-batch');
                }
              }
            }
          });
          
          updateSummary();
        }

        triplets.forEach(triplet => {
          const cards = triplet.querySelectorAll('.image-card');
          const cropToggle = triplet.querySelector('.crop-toggle');
          const cropCheckbox = triplet.querySelector('.crop-checkbox');
          cards.forEach(card => {
            card.addEventListener('click', () => {
              const current = triplet.dataset.selectedIndex;
              const idx = card.dataset.imageIndex;
              if (current === idx) {
                delete triplet.dataset.selectedIndex;
                cropCheckbox.checked = false;
                cropToggle.classList.add('hidden');
              } else {
                triplet.dataset.selectedIndex = idx;
                cropToggle.classList.remove('hidden');
              }
              
              // Update reviewed state for progressive commit
              const tripletIndex = triplets.indexOf(triplet);
              if (reviewBatchTriplets.has(tripletIndex)) {
                if (triplet.dataset.selectedIndex !== undefined && triplet.dataset.selectedIndex !== '') {
                  triplet.classList.add('reviewed');
                  triplet.classList.remove('in-batch');
                } else {
                  triplet.classList.add('in-batch');
                  triplet.classList.remove('reviewed');
                }
              }
              
              updateSummary();
            });
          });
          cropCheckbox.addEventListener('change', () => {
            if (!(triplet.dataset.selectedIndex)) {
              cropCheckbox.checked = false;
              cropToggle.classList.add('hidden');
            }
            updateSummary();
          });
          updateSummary();
        });

        // Initialize viewport batch detection
        window.addEventListener('scroll', updateViewportBatch);
        window.addEventListener('resize', updateViewportBatch);
        
        // Initial viewport batch update
        updateViewportBatch();

        function setStatus(message, type = '') {
          statusBox.textContent = message;
          statusBox.className = type ? type : '';
        }

        finalizeButton.addEventListener('click', async () => {
          if (!confirm('Finalize selections? This will move/delete files immediately.')) {
            return;
          }
          finalizeButton.disabled = true;
          setStatus('Processing…');
          const selections = triplets.map(triplet => {
            const selectedIndex = triplet.dataset.selectedIndex;
            const cropCheckbox = triplet.querySelector('.crop-checkbox');
            return {
              tripletId: Number(triplet.dataset.tripletId),
              selectedIndex: selectedIndex === undefined ? null : Number(selectedIndex),
              crop: selectedIndex === undefined ? false : cropCheckbox.checked,
            };
          });
          try {
          const response = await fetch('/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selections, batch_mode: false }),
            });
            const payload = await response.json();
            if (!response.ok || payload.status !== 'ok') {
              throw new Error(payload.message || 'Server error');
            }
            setStatus(payload.message, 'success');
            finalizeButton.disabled = true;
            finalizeButton.textContent = 'Selections applied';
          } catch (error) {
            console.error(error);
            finalizeButton.disabled = false;
            setStatus(error.message || 'Unable to finalize selections', 'error');
          }
        });

        processBatchButton.addEventListener('click', async () => {
          const batchTriplets = Array.from(reviewBatchTriplets).map(index => triplets[index]);
          
          if (batchTriplets.length === 0) {
            setStatus('No triplets in current batch to process', 'error');
            return;
          }
          
          if (!confirm(`Process current batch of ${batchTriplets.length} triplets? This will move/delete files immediately.`)) {
            return;
          }
          
          processBatchButton.disabled = true;
          setStatus('Processing batch…');
          
          const batchSelections = batchTriplets.map(triplet => {
            const selectedIndex = triplet.dataset.selectedIndex;
            const cropCheckbox = triplet.querySelector('.crop-checkbox');
            return {
              tripletId: parseInt(triplet.dataset.tripletId),
              selectedIndex: selectedIndex ? parseInt(selectedIndex) : null,
              crop: cropCheckbox ? cropCheckbox.checked : false
            };
          });
          
          try {
            const response = await fetch('/submit', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ selections: batchSelections, batch_mode: true })
            });
            const payload = await response.json();
            if (!response.ok) {
              throw new Error(payload.message || 'Server error');
            }
            
            // Remove processed triplets from DOM and batch set
            batchTriplets.forEach((triplet, i) => {
              const index = triplets.indexOf(triplet);
              reviewBatchTriplets.delete(index);
              triplet.remove();
              triplets.splice(index, 1);
            });
            
            // Re-index remaining triplets
            triplets.forEach((triplet, newIndex) => {
              triplet.querySelector('.triplet-header span').textContent = 
                `Triplet ${newIndex + 1} / ${triplets.length}`;
            });
            
            setStatus(`Batch processed! ${payload.moved || 0} moved, ${payload.deleted || 0} deleted`, 'success');
            processBatchButton.disabled = false;
            
            if (triplets.length === 0) {
              setStatus('All triplets processed! 🎉', 'success');
              processBatchButton.textContent = 'All Complete';
              processBatchButton.disabled = true;
            }
            
            updateSummary();
            
          } catch (error) {
            console.error(error);
            processBatchButton.disabled = false;
            setStatus(error.message || 'Unable to process batch', 'error');
          }
        });
      </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return render_template_string(
            page_template,
            triplets=app.config["TRIPLET_PAYLOAD"],
            total_triplets=len(triplets),
        )

    @app.route("/image/<int:triplet_id>/<int:image_index>")
    def serve_image(triplet_id: int, image_index: int):
        try:
            record: TripletRecord = app.config["TRIPLETS"][triplet_id]
            path = record.paths[image_index]
        except (IndexError, KeyError):
            return Response(status=404)
        if not path.exists():
            return Response(status=404)
        stat = path.stat()
        data = _generate_thumbnail(str(path), int(stat.st_mtime_ns), stat.st_size)
        return Response(data, mimetype="image/jpeg")

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
        
        # In batch mode, we only process a subset of triplets
        if not batch_mode and len(selections) != len(triplets):
            return jsonify({"status": "error", "message": "Selection payload missing or invalid."}), 400

        tracker: FileTracker = app.config["TRACKER"]
        base_folder: Path = app.config["BASE_FOLDER"]
        log_path: Path = app.config["LOG_PATH"]
        hard_delete: bool = app.config["HARD_DELETE"]

        kept_count = 0
        crop_count = 0

        try:
            for selection in selections:
                triplet_id = selection.get("tripletId")
                try:
                    record: TripletRecord = app.config["TRIPLETS"][triplet_id]
                except (TypeError, KeyError, IndexError):
                    raise ValueError(f"Invalid triplet id: {triplet_id}")

                selected_index = selection.get("selectedIndex")
                crop_flag = bool(selection.get("crop"))

                if selected_index is None:
                    safe_delete(record.paths, hard_delete=hard_delete, tracker=tracker)
                    write_log(log_path, "delete_all", None, list(record.paths))
                    continue

                if not isinstance(selected_index, int) or selected_index not in (0, 1, 2):
                    raise ValueError(f"Invalid selection index for triplet {triplet_id}")

                selected_path = record.paths[selected_index]
                others = [p for idx, p in enumerate(record.paths) if idx != selected_index]

                if crop_flag:
                    target_path = move_to_crop(selected_path, tracker=tracker)
                    crop_count += 1
                    action_label = "keep_one_to_crop"
                else:
                    target_path = move_to_reviewed(selected_path, base_folder, tracker)
                    action_label = "keep_one"

                safe_delete(others, hard_delete=hard_delete, tracker=tracker)
                write_log(log_path, action_label, target_path, others)
                kept_count += 1

            if not batch_mode:
                app.config["PROCESSED"] = True
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

        deleted_count = len(selections) - kept_count
        
        if batch_mode:
            message = f"Batch processed — kept {kept_count}, sent {crop_count} to crop/, deleted {deleted_count}."
            return jsonify({
                "status": "ok", 
                "message": message,
                "moved": kept_count,
                "deleted": deleted_count
            })
        else:
            message = (
                f"Processed {len(triplets)} triplets — kept {kept_count}, "
                f"sent {crop_count} to crop/, deleted {deleted_count}."
            )
            threading.Thread(target=process_shutdown, daemon=True).start()
            return jsonify({"status": "ok", "message": message})

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
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for the local web server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the local web server")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        human_err(f"Folder not found: {folder}")
        sys.exit(1)

    tracker = FileTracker("image_version_selector")

    exts = [ext.strip() for ext in args.exts.split(",") if ext.strip()]
    files = scan_images(folder, exts)
    if not files:
        human_err("No images found. Check --exts or folder path.")
        sys.exit(1)

    triplet_paths = find_triplets(files)
    if not triplet_paths:
        human_err("No triplets found with the current grouping. Try adjusting filenames or timestamps.")
        sys.exit(1)

    if args.print_triplets:
        for idx, triplet in enumerate(triplet_paths, 1):
            print(f"\nTriplet {idx}:")
            for path in triplet:
                print("  -", path.name)
        print(f"\nTotal triplets: {len(triplet_paths)}")
        return

    if not args.hard_delete and not _SEND2TRASH_AVAILABLE:
        human_err("send2trash not installed. Install it with: pip install send2trash")
        human_err("Or rerun with --hard-delete to permanently delete files (dangerous).")
        sys.exit(1)

    records: List[TripletRecord] = []
    for idx, triplet in enumerate(triplet_paths):
        first_parent = triplet[0].parent
        try:
            relative = str(first_parent.relative_to(folder))
        except ValueError:
            relative = str(first_parent)
        records.append(
            TripletRecord(
                index=idx,
                paths=triplet,
                relative_dir=relative if relative != "." else "",
            )
        )

    log_path = folder / "triplet_culler_log.csv"
    app = build_app(records, folder, tracker, log_path, hard_delete=args.hard_delete)

    url = f"http://{args.host}:{args.port}"
    info(f"Found {len(records)} triplets. Launching browser UI at {url}")
    info("Use CTRL+C to stop the server after it reports that processing is complete.")

    if not args.no_browser:
        threading.Thread(target=launch_browser, args=(args.host, args.port), daemon=True).start()

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except OSError as exc:
        human_err(f"Failed to start server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()