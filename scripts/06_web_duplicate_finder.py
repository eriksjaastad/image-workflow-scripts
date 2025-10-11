#!/usr/bin/env python3
"""
Step 6: Web Duplicate Finder
=============================
Side-by-side directory viewer for finding and removing duplicate images.
Simple split-screen interface for visual duplicate detection.

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
Compare two directories for duplicates:
  python scripts/06_web_duplicate_finder.py left_directory right_directory
  python scripts/06_web_duplicate_finder.py mixed-0919/duplicate_group_001 mixed-0919/duplicate_group_002
  python scripts/06_web_duplicate_finder.py sorted/unknown character_group_1

FEATURES:
---------
• Split-screen layout: left directory (view-only) vs right directory (interactive)
• 4 images per row on each side for easy comparison
• Left side: First directory - view-only reference images
• Right side: Second directory - interactive delete functionality
• Click image → Select for deletion (red highlight)
• Click again → Deselect
• Both sides scroll independently
• Perfect for comparing two directories and removing duplicates
• FileTracker integration for complete audit trail

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_web_image_selector.py
Step 2: Face Grouping → scripts/02_face_grouper.py
Step 3: Character Sorting → scripts/03_web_character_sorter.py
Step 4: Final Cropping → scripts/04_batch_crop_tool.py
Step 5: Basic Review → THIS SCRIPT (scripts/05_multi_directory_viewer.py)

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/utils/similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

FEATURES:
---------
- 8 images per row (compact thumbnails)
- Directory headers as visual separators
- Interactive crop/delete actions
- Shows all subdirectories in one view
- Perfect for quick clustering assessment and cleanup
"""

import os
import sys
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import webbrowser
from utils.companion_file_utils import launch_browser, generate_thumbnail, logger, get_error_display_html, get_training_dir, _append_csv_row, safe_delete_image_and_yaml
from datetime import datetime
import threading
import time
import shutil

# Configuration
THUMBNAIL_MAX_DIM = 200
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.file_tracker import FileTracker

_SEND2TRASH_AVAILABLE = False
try:
    from send2trash import send2trash
    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False

def find_image_directories(output_dir):
    """Find all subdirectories containing images, or treat as single directory if images are directly in it."""
    # Convert to absolute path from script's parent directory
    if not Path(output_dir).is_absolute():
        output_path = Path(__file__).parent.parent / output_dir
    else:
        output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Directory not found: {output_path}")
        return []
    
    directories = []
    
    # First check if there are images directly in the main directory
    direct_images = list(output_path.glob("*.png"))
    if direct_images:
        # Treat the main directory as a single image directory
        directories.append({
            'name': output_path.name,
            'path': output_path,
            'images': sorted([img.name for img in direct_images]),
            'count': len(direct_images)
        })
        return directories
    
    # Otherwise, look for subdirectories containing images
    for subdir in sorted(output_path.iterdir()):
        if subdir.is_dir():
            # Find PNG files in this directory
            images = list(subdir.glob("*.png"))
            if images:
                directories.append({
                    'name': subdir.name,
                    'path': subdir,
                    'images': sorted([img.name for img in images]),
                    'count': len(images)
                })
    
    return directories

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Duplicate Finder - {{ output_dir }}</title>
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
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: #f8f9ff;
            line-height: 1.4;
            height: 100vh;
            overflow: hidden;
        }
        
        .split-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            height: 100vh;
            gap: 2px;
        }
        
        .panel {
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        
        .panel-header {
            background: var(--surface);
            padding: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            flex-shrink: 0;
        }
        
        .panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }
        
        .left-panel {
            background: var(--surface-alt);
        }
        
        .right-panel {
            background: var(--surface);
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            padding: 1rem 0;
        }
        
        .image-item {
            position: relative;
            background: var(--bg);
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s ease;
        }
        
        .image-container {
            position: relative;
        }
        
        .right-panel .image-container {
            cursor: pointer;
        }
        
        .image-container img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }
        
        .image-container.delete-selected {
            border: 3px solid var(--danger);
            box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.3);
        }
        
        
        header.toolbar {
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--surface);
            border-bottom: 1px solid var(--surface-alt);
            padding: 1rem 1.5rem;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .toolbar-left h1 {
            margin: 0 0 4px 0;
            color: #f8f9ff;
            font-size: 20px;
            font-weight: 600;
        }
        
        .toolbar-left p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }
        
        .toolbar-center {
            text-align: center;
            color: var(--muted);
            font-size: 14px;
            font-weight: 500;
        }
        
        .toolbar-center #deleteCount {
            color: var(--danger);
        }
        
        .toolbar-center #cropCount {
            color: var(--accent);
        }
        
        .toolbar-right {
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: flex-end;
        }
        
        header.toolbar button {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        
        .summary {
            background: var(--surface-alt);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid var(--accent);
            color: #f8f9ff;
        }
        
        .directory-section {
            background: var(--surface);
            margin-bottom: 30px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            border: 1px solid var(--surface-alt);
        }
        
        .directory-header {
            background: var(--accent);
            color: white;
            padding: 15px 20px;
            font-size: 18px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .directory-count {
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .image-grid {
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        
        .image-container {
            border-radius: 8px;
            overflow: hidden;
            background: var(--surface-alt);
            transition: transform 0.2s ease, border 0.2s ease;
            cursor: pointer;
            border: 3px solid transparent;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 300px;
        }
        
        
        .image-container.delete-selected {
            border-color: var(--danger);
            box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.3);
        }
        
        .image-container.crop-selected {
            border-color: white;
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.6);
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 300px;
            object-fit: contain;
            display: block;
        }
        
        .crop-button {
            width: 100%;
            padding: 6px;
            margin-top: 4px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.2s ease;
        }
        
        .crop-button:hover {
            background: var(--accent);
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .crop-button.active {
            background: white;
            color: var(--bg);
            transform: translateY(-1px);
        }
        
        .status-bar {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--surface);
            color: #f8f9ff;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 1000;
            border: 1px solid var(--surface-alt);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .status-bar.show {
            opacity: 1;
        }
        
        .status-bar.success {
            background: var(--success);
            color: var(--bg);
        }
        
        .status-bar.error {
            background: var(--danger);
            color: white;
        }
        
        .no-images {
            padding: 40px;
            text-align: center;
            color: #6e6e73;
            font-style: italic;
        }
        
        @media (max-width: 800px) {
            .image-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        @media (max-width: 500px) {
            .image-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="status-bar" id="statusBar"></div>
    
    <div class="split-container">
        <!-- Left Panel - View Only -->
        <div class="panel left-panel">
            <div class="panel-header">
                <h2>{{ left_dir_name }}</h2>
                <p>{{ left_images|length }} images (view-only)</p>
            </div>
            <div class="panel-content">
                <div class="image-grid">
                    {% for image in left_images %}
                    <div class="image-item">
                        <div class="image-container">
                            <img src="/image/{{ left_dir_name }}/{{ image }}" alt="{{ image }}" loading="lazy">
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <!-- Right Panel - Interactive -->
        <div class="panel right-panel">
            <div class="panel-header">
                <h2>{{ right_dir_name }}</h2>
                <div id="selectionStats">
                    <span id="deleteCount">0 selected</span>
                    <button id="processButton" onclick="processSelections()" style="margin-left: 1rem;">Delete Selected</button>
                </div>
            </div>
            <div class="panel-content">
                <div class="image-grid">
                    {% for image in right_images %}
                    <div class="image-item" data-directory="{{ right_dir_name }}" data-image="{{ image }}">
                        <div class="image-container" onclick="toggleDelete('{{ right_dir_name }}', '{{ image }}')">
                            <img src="/image/{{ right_dir_name }}/{{ image }}" alt="{{ image }}" loading="lazy">
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

JAVASCRIPT_CODE = """
<script>
    function showStatus(message, type = 'info') {
        const statusBar = document.getElementById('statusBar');
        statusBar.textContent = message;
        statusBar.className = `status-bar show ${type}`;
        
        setTimeout(() => {
            statusBar.classList.remove('show');
        }, 3000);
    }
    
    // Track image states: 'delete', 'crop', or null
    let imageStates = {};
    
    function getImageKey(directory, image) {
        return `${directory}/${image}`;
    }
    
    function updateImageVisual(directory, image) {
        const key = getImageKey(directory, image);
        const container = document.querySelector(`[data-directory="${directory}"][data-image="${image}"] .image-container`);
        
        if (!container) return;
        
        // Clear all states
        container.classList.remove('delete-selected');
        
        // Apply current state
        const state = imageStates[key];
        if (state === 'delete') {
            container.classList.add('delete-selected');
        }
        
        // Update selection stats
        updateSelectionStats();
    }
    
    function updateSelectionStats() {
        const deleteCount = Object.values(imageStates).filter(state => state === 'delete').length;
        
        document.getElementById('deleteCount').textContent = `${deleteCount} selected`;
    }
    
    function toggleDelete(directory, image) {
        const key = getImageKey(directory, image);
        const currentState = imageStates[key];
        
        if (currentState === 'delete') {
            // Toggle off delete
            delete imageStates[key];
        } else {
            // Set to delete (overrides crop)
            imageStates[key] = 'delete';
        }
        
        updateImageVisual(directory, image);
    }
    
    
    async function processSelections() {
        const selections = Object.keys(imageStates).map(key => {
            const [directory, image] = key.split('/');
            return {
                directory,
                image,
                action: imageStates[key]
            };
        });
        
        if (selections.length === 0) {
            showStatus('No selections made', 'error');
            return;
        }
        
        const deleteCount = selections.filter(s => s.action === 'delete').length;
        
        if (!confirm(`Delete ${deleteCount} selected images?`)) {
            return;
        }
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ selections })
            });
            
            const result = await response.json();
            
            if (result.success) {
                showStatus(`✅ Processed ${result.processed} operations - Reloading page...`, 'success');
                
                // Clear all selections immediately
                imageStates = {};
                updateSelectionStats();
                
                // Remove all visual highlights
                document.querySelectorAll('.image-container.delete-selected').forEach(container => {
                    container.classList.remove('delete-selected');
                });
                
                // Reload the page after a short delay to show the updated file list
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
                
            } else {
                showStatus(`❌ Error: ${result.message}`, 'error');
            }
        } catch (error) {
            showStatus(`❌ Network error: ${error.message}`, 'error');
        }
    }
</script>
"""

def create_app(left_dir, right_dir):
    app = Flask(__name__)
    
    # Convert to absolute paths
    left_path = Path(left_dir).resolve()
    right_path = Path(right_dir).resolve()
    
    # Initialize FileTracker
    tracker = FileTracker("multi_directory_viewer")
    
    @app.route('/')
    def index():
        # Scan directories fresh each time to reflect deletions
        left_images = sorted([img.name for img in left_path.glob("*.png")])
        right_images = sorted([img.name for img in right_path.glob("*.png")])
        
        return render_template_string(
            HTML_TEMPLATE + JAVASCRIPT_CODE,
            left_images=left_images,
            right_images=right_images,
            left_dir_name=left_path.name,
            right_dir_name=right_path.name,
            left_dir=left_dir,
            right_dir=right_dir,
            error_display_html=get_error_display_html()
        )
    
    @app.route('/image/<directory>/<filename>')
    def serve_image(directory, filename):
        """Serve image thumbnails from the directories."""
        from flask import Response
        
        # Determine which directory to serve from
        if directory == left_path.name:
            image_path = left_path / filename
        elif directory == right_path.name:
            image_path = right_path / filename
        else:
            return "Directory not found", 404
        
        if image_path.exists():
            try:
                # Generate thumbnail using shared function
                stat = image_path.stat()
                thumbnail_data = generate_thumbnail(
                    str(image_path), 
                    int(stat.st_mtime_ns), 
                    stat.st_size,
                    max_dim=THUMBNAIL_MAX_DIM,
                    quality=85
                )
                return Response(thumbnail_data, mimetype='image/jpeg')
            except Exception as e:
                logger.error_with_exception(f"Error generating thumbnail for {filename}", e)
                return "Error generating thumbnail", 500
        
        return "Image not found", 404
    
    @app.route('/process', methods=['POST'])
    def process_selections():
        """Process batch of delete selections."""
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'message': 'No selections provided'})
        
        processed = 0
        errors = []
        
        for selection in selections:
            directory = selection.get('directory')
            image = selection.get('image')
            action = selection.get('action')
            
            if not all([directory, image, action]):
                errors.append(f"Invalid selection: {selection}")
                continue
            
            # Only allow deletions from the right directory
            if directory != right_path.name:
                errors.append(f"Can only delete from right directory: {directory}")
                continue
                
            source_path = right_path / image
            if not source_path.exists():
                errors.append(f"Image not found: {image}")
                continue
            
            try:
                if action == 'delete':
                    # Use shared companion-aware delete for image and ALL companions
                    try:
                        _ = safe_delete_image_and_yaml(source_path, hard_delete=False, tracker=tracker)
                    except Exception as e:
                        errors.append(f"send2trash not available or delete failed for {image}: {e}")
                        continue
                    
                    # Log training data for duplicate detection
                    try:
                        training_dir = get_training_dir()
                        log_path = training_dir / "duplicate_detection_log.csv"
                        header = ['timestamp', 'session_id', 'left_directory', 'right_directory', 'deleted_image']
                        row = {
                            'timestamp': datetime.now().isoformat(),
                            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                            'left_directory': str(left_path),
                            'right_directory': str(right_path),
                            'deleted_image': str(source_path)
                        }
                        _append_csv_row(log_path, header, row)
                    except Exception:
                        pass  # Don't let logging errors break the workflow
                
                processed += 1
                
            except Exception as e:
                errors.append(f"Error processing {image}: {str(e)}")
        
        if errors:
            return jsonify({
                'success': False, 
                'message': f"Processed {processed}, errors: {'; '.join(errors[:3])}"
            })
        else:
            return jsonify({
                'success': True, 
                'processed': processed,
                'message': f"Successfully processed {processed} operations"
            })
    
    return app

def open_browser(url):
    """Open browser after a short delay."""
    # Extract host and port from URL for shared function
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5000
    launch_browser(host, port, delay=1.5)

def main():
    parser = argparse.ArgumentParser(description="Web duplicate finder - compare two directories side by side")
    parser.add_argument("left_dir", help="Left directory (view-only reference)")
    parser.add_argument("right_dir", help="Right directory (interactive, can delete)")
    parser.add_argument("--port", type=int, default=5005, help="Port to run the server on (default: 5005)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")
    
    args = parser.parse_args()
    
    # Validate both directories exist
    left_path = Path(args.left_dir)
    right_path = Path(args.right_dir)
    
    if not left_path.exists():
        logger.directory_not_found(args.left_dir)
        sys.exit(1)
    
    if not right_path.exists():
        logger.directory_not_found(args.right_dir)
        sys.exit(1)
    
    # Get images from both directories
    left_images = list(left_path.glob("*.png"))
    right_images = list(right_path.glob("*.png"))
    
    if not left_images:
        print(f"❌ Error: No PNG images found in left directory '{args.left_dir}'")
        sys.exit(1)
    
    if not right_images:
        print(f"❌ Error: No PNG images found in right directory '{args.right_dir}'")
        sys.exit(1)
    
    print(f"🚀 Starting Web Duplicate Finder...")
    print(f"📂 Left directory: {args.left_dir} ({len(left_images)} images)")
    print(f"📂 Right directory: {args.right_dir} ({len(right_images)} images)")
    
    app = create_app(args.left_dir, args.right_dir)
    
    url = f"http://localhost:{args.port}"
    print(f"🌐 Server starting at: {url}")
    
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
