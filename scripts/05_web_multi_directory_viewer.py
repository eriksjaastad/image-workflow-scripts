#!/usr/bin/env python3
"""
Step 5: Multi-Directory Image Viewer
=====================================
Interactive viewer for quick assessment and cleanup of clustering results.
Shows all images from subdirectories with crop and delete functionality.

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
Run on clustering output directories:
  python scripts/05_web_multi_directory_viewer.py face_groups
  python scripts/05_web_multi_directory_viewer.py character_group_1

FEATURES:
---------
• 8 images per row (compact thumbnails)
• Directory headers as visual separators
• Interactive crop and delete functionality
• Click image → Crop (send to crop/ directory)
• Right-click image → Delete (send to trash)
• Shows all subdirectories in one consolidated view
• Perfect for quick clustering quality assessment and cleanup
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
from utils.companion_file_utils import launch_browser, generate_thumbnail, get_error_display_html, format_image_display_name
import threading
import time
import shutil

# Configuration
THUMBNAIL_MAX_DIM = 200
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.file_tracker import FileTracker

try:
    import send2trash
    _SEND2TRASH_AVAILABLE = True
except ImportError:
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
    <title>Multi-Directory Image Viewer - {{ output_dir }}</title>
    {{ error_display_html }}
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
            padding: 20px;
            background: var(--bg);
            color: #f8f9ff;
            line-height: 1.4;
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
            grid-template-columns: repeat(8, 1fr);
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
        
        @media (max-width: 1200px) {
            .image-grid {
                grid-template-columns: repeat(6, 1fr);
            }
        }
        
        @media (max-width: 800px) {
            .image-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        
        @media (max-width: 500px) {
            .image-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="status-bar" id="statusBar"></div>
    
    <header class="toolbar">
        <div class="toolbar-left">
            <h1>Multi-Directory Image Viewer</h1>
            <p>{{ output_dir }} • {{ total_directories }} directories, {{ total_images }} images total</p>
        </div>
        <div class="toolbar-center">
            <span id="selectionStats">
                <span id="deleteCount">0 delete</span> • 
                <span id="cropCount">0 crop</span>
            </span>
        </div>
        <div class="toolbar-right">
            <button id="processButton" onclick="processSelections()">Process Selections</button>
        </div>
    </header>
    
    {% for directory in directories %}
    <div class="directory-section">
        <div class="directory-header">
            <span>{{ directory.name }}</span>
            <span class="directory-count">{{ directory.count }} images</span>
        </div>
        
        {% if directory.images %}
        <div class="image-grid">
            {% for image in directory.images %}
            <div class="image-item" data-directory="{{ directory.name }}" data-image="{{ image }}">
                <div class="image-container" onclick="toggleDelete('{{ directory.name }}', '{{ image }}')">
                    <img src="/image/{{ directory.name }}/{{ image }}" alt="{{ image }}" loading="lazy">
                </div>
                <button class="crop-button" onclick="toggleCrop('{{ directory.name }}', '{{ image }}')">Crop</button>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-images">No images found</div>
        {% endif %}
    </div>
    {% endfor %}
    
    {% if not directories %}
    <div class="directory-section">
        <div class="no-images">
            <h3>No directories with images found</h3>
            <p>Make sure the output directory contains subdirectories with PNG files.</p>
        </div>
    </div>
    {% endif %}
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
    const imageStates = {};
    
    function getImageKey(directory, image) {
        return `${directory}/${image}`;
    }
    
    function updateImageVisual(directory, image) {
        const key = getImageKey(directory, image);
        const container = document.querySelector(`[data-directory="${directory}"][data-image="${image}"] .image-container`);
        const button = document.querySelector(`[data-directory="${directory}"][data-image="${image}"] .crop-button`);
        
        if (!container || !button) return;
        
        // Clear all states
        container.classList.remove('delete-selected', 'crop-selected');
        button.classList.remove('active');
        
        // Apply current state
        const state = imageStates[key];
        if (state === 'delete') {
            container.classList.add('delete-selected');
        } else if (state === 'crop') {
            container.classList.add('crop-selected');
            button.classList.add('active');
        }
        
        // Update selection stats
        updateSelectionStats();
    }
    
    function updateSelectionStats() {
        const deleteCount = Object.values(imageStates).filter(state => state === 'delete').length;
        const cropCount = Object.values(imageStates).filter(state => state === 'crop').length;
        
        document.getElementById('deleteCount').textContent = `${deleteCount} delete`;
        document.getElementById('cropCount').textContent = `${cropCount} crop`;
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
    
    function toggleCrop(directory, image) {
        const key = getImageKey(directory, image);
        const currentState = imageStates[key];
        
        if (currentState === 'crop') {
            // Toggle off crop
            delete imageStates[key];
        } else {
            // Set to crop (overrides delete)
            imageStates[key] = 'crop';
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
        
        const cropCount = selections.filter(s => s.action === 'crop').length;
        const deleteCount = selections.filter(s => s.action === 'delete').length;
        
        if (!confirm(`Process ${cropCount} crop and ${deleteCount} delete operations?`)) {
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

def create_app(output_dir):
    app = Flask(__name__)
    
    # Convert to absolute path from script's parent directory  
    if not Path(output_dir).is_absolute():
        output_path = Path(__file__).parent.parent / output_dir
    else:
        output_path = Path(output_dir)
    
    # Find all directories with images
    directories = find_image_directories(output_dir)
    total_images = sum(d['count'] for d in directories)
    
    print(f"📁 Found {len(directories)} directories with {total_images} total images")
    for d in directories:
        print(f"   • {d['name']}: {d['count']} images")
    
    # Initialize FileTracker
    tracker = FileTracker("multi_directory_viewer")
    
    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE + JAVASCRIPT_CODE,
            directories=directories,
            total_directories=len(directories),
            total_images=total_images,
            output_dir=output_dir,
            error_display_html=get_error_display_html()
        )
    
    @app.route('/image/<directory>/<filename>')
    def serve_image(directory, filename):
        """Serve image thumbnails from the directories."""
        from flask import Response
        
        # Find the correct directory structure
        for dir_info in directories:
            if dir_info['name'] == directory:
                image_path = dir_info['path'] / filename
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
                        print(f"[!] Error generating thumbnail for {filename}: {e}")
                        return "Error generating thumbnail", 500
                break
        
        return "Image not found", 404
    
    @app.route('/process', methods=['POST'])
    def process_selections():
        """Process batch of crop and delete selections."""
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
            
            source_path = output_path / directory / image
            if not source_path.exists():
                errors.append(f"Image not found: {image}")
                continue
            
            try:
                if action == 'crop':
                    # Move to crop directory
                    crop_dir = output_path.parent / 'crop'
                    crop_dir.mkdir(exist_ok=True)
                    
                    dest_path = crop_dir / image
                    
                    # Handle name conflicts
                    counter = 1
                    while dest_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        dest_path = crop_dir / f"{stem}_{counter:03d}{suffix}"
                        counter += 1
                    
                    shutil.move(str(source_path), str(dest_path))
                    
                    # Also move metadata file if it exists (.yaml or .caption)
                    yaml_source = source_path.with_suffix('.yaml')
                    if not yaml_source.exists():
                        yaml_source = source_path.with_suffix('.caption')
                    
                    if yaml_source.exists():
                        yaml_dest = dest_path.with_suffix(yaml_source.suffix)
                        shutil.move(str(yaml_source), str(yaml_dest))
                    
                    tracker.log_move(source_path, dest_path, "crop_action")
                    
                elif action == 'delete':
                    # Move to trash
                    if _SEND2TRASH_AVAILABLE:
                        send2trash.send2trash(str(source_path))
                        
                        # Also delete metadata file if it exists (.yaml or .caption)
                        yaml_source = source_path.with_suffix('.yaml')
                        if not yaml_source.exists():
                            yaml_source = source_path.with_suffix('.caption')
                        
                        if yaml_source.exists():
                            send2trash.send2trash(str(yaml_source))
                        
                        tracker.log_delete(source_path, "delete_action")
                    else:
                        errors.append(f"send2trash not available for {image}")
                        continue
                
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
    parser = argparse.ArgumentParser(description="Multi-directory image viewer for clustering results")
    parser.add_argument("output_dir", help="Directory containing subdirectories with images (e.g., out_k7_final)")
    parser.add_argument("--port", type=int, default=5005, help="Port to run the server on (default: 5005)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")
    
    args = parser.parse_args()
    
    if not Path(args.output_dir).exists():
        print(f"❌ Error: Directory '{args.output_dir}' not found")
        sys.exit(1)
    
    print(f"🚀 Starting Multi-Directory Image Viewer...")
    print(f"📂 Scanning: {args.output_dir}")
    
    app = create_app(args.output_dir)
    
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
