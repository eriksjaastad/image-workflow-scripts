#!/usr/bin/env python3
"""
Step 5: Multi-Directory Image Viewer
=====================================
Simple read-only viewer for quick assessment of clustering results.
Shows all images from subdirectories in a single page with clear separators.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run on clustering output directories:
  python scripts/05_multi_directory_viewer.py face_groups
  python scripts/05_multi_directory_viewer.py character_group_1

FEATURES:
---------
• 8 images per row (compact thumbnails)
• Directory headers as visual separators
• Read-only interface (no interactions needed)
• Shows all subdirectories in one consolidated view
• Perfect for quick clustering quality assessment
• Lightweight and fast for large image sets

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_web_image_selector.py
Step 2: Face Grouping → scripts/02_face_grouper.py
Step 3: Character Sorting → scripts/03_web_character_sorter.py
Step 4: Final Cropping → scripts/04_batch_crop_tool.py
Step 5: Basic Review → THIS SCRIPT (scripts/05_multi_directory_viewer.py)

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/util_similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

FEATURES:
---------
- 8 images per row (compact thumbnails)
- Directory headers as visual separators
- Read-only (no interaction needed)
- Shows all subdirectories in one view
- Perfect for quick clustering assessment
"""

import os
import sys
import argparse
from pathlib import Path
from flask import Flask, render_template_string
import webbrowser
import threading
import time

def find_image_directories(output_dir):
    """Find all subdirectories containing images."""
    # Convert to absolute path from script's parent directory
    if not Path(output_dir).is_absolute():
        output_path = Path(__file__).parent.parent / output_dir
    else:
        output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Directory not found: {output_path}")
        return []
    
    directories = []
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
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
            line-height: 1.4;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0 0 10px 0;
            color: #1d1d1f;
            font-size: 28px;
            font-weight: 600;
        }
        
        .header p {
            margin: 0;
            color: #6e6e73;
            font-size: 16px;
        }
        
        .summary {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #007aff;
        }
        
        .directory-section {
            background: white;
            margin-bottom: 30px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .directory-header {
            background: #007aff;
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
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            background: #f0f0f0;
            transition: transform 0.2s ease;
        }
        
        .image-container:hover {
            transform: scale(1.05);
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
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
    <div class="header">
        <h1>Multi-Directory Image Viewer</h1>
        <p>Clustering results: {{ output_dir }}</p>
    </div>
    
    <div class="summary">
        <strong>📊 Summary:</strong> {{ total_directories }} directories, {{ total_images }} images total
    </div>
    
    {% for directory in directories %}
    <div class="directory-section">
        <div class="directory-header">
            <span>{{ directory.name }}</span>
            <span class="directory-count">{{ directory.count }} images</span>
        </div>
        
        {% if directory.images %}
        <div class="image-grid">
            {% for image in directory.images %}
            <div class="image-container">
                <img src="/image/{{ directory.name }}/{{ image }}" alt="{{ image }}" loading="lazy">
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
    
    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            directories=directories,
            total_directories=len(directories),
            total_images=total_images,
            output_dir=output_dir
        )
    
    @app.route('/image/<directory>/<filename>')
    def serve_image(directory, filename):
        """Serve images from the directories."""
        from flask import send_file
        image_path = output_path / directory / filename
        if image_path.exists():
            return send_file(str(image_path))
        else:
            return "Image not found", 404
    
    return app

def open_browser(url):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(url)

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
