#!/usr/bin/env python3
"""
Utility: Similarity Viewer - Optional Analysis Tool
====================================================
READ-ONLY diagnostic tool for analyzing face grouper clustering results.
Shows similarity connections and quality indicators between images.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run on clustering output with similarity maps:
  python scripts/utils/similarity_viewer.py face_groups

FEATURES:
---------
• 8 images per row (compact thumbnails)
• Click image → highlight its most similar neighbors
• Color-coded similarity connections (green=same cluster, red=cross-cluster)
• Similarity scores on hover
• Cluster quality indicators
• Interactive similarity exploration

⚠️ IMPORTANT: This is an OPTIONAL utility, not a core workflow step!
   Use between steps 2-3 to analyze face grouper results before manual sorting.

RELATED WORKFLOW:
-----------------
Step 2: Face Grouping → scripts/02_face_grouper.py (creates similarity maps)
Step 3: Character Sorting → scripts/03_web_character_sorter.py (uses similarity maps)
- Color-coded similarity connections (green=same cluster, red=cross-cluster)
- Similarity scores on hover
- Cluster quality indicators
- Interactive similarity exploration
"""

import argparse
import csv
import json
import sys
import threading
from pathlib import Path

from companion_file_utils import launch_browser
from flask import Flask, jsonify, render_template_string


def load_similarity_data(output_dir):
    """Load nodes, edges, and neighbors data from similarity map files."""
    output_path = Path(output_dir)

    # Load nodes (image -> cluster mapping)
    nodes = {}
    nodes_file = output_path / "nodes.csv"
    if nodes_file.exists():
        with open(nodes_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                nodes[row["filename"]] = {
                    "index": int(row["index"]),
                    "label": int(row["label"]),
                    "filename": row["filename"],
                }

    # Load edges (similarity connections)
    edges = []
    edges_file = output_path / "edges.csv"
    if edges_file.exists():
        with open(edges_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append(
                    {
                        "src_idx": int(row["src_idx"]),
                        "dst_idx": int(row["dst_idx"]),
                        "src_label": int(row["src_label"]),
                        "dst_label": int(row["dst_label"]),
                        "similarity": float(row["sim"]),
                        "distance": float(row["dist"]),
                        "src_file": row["src_file"],
                        "dst_file": row["dst_file"],
                    }
                )

    # Load neighbors (top-K similar images per image)
    neighbors = {}
    neighbors_file = output_path / "neighbors.jsonl"
    if neighbors_file.exists():
        with open(neighbors_file) as f:
            for line in f:
                data = json.loads(line.strip())
                neighbors[data["filename"]] = data

    return nodes, edges, neighbors


def similarity_sort_images(images, neighbors_data):
    """Sort images within a directory by similarity to create visual neighborhoods."""
    if len(images) <= 1:
        return images

    # Build similarity graph for this set of images
    image_set = set(images)
    similarity_graph = {}

    for img in images:
        similarity_graph[img] = []
        if img in neighbors_data:
            # Only include neighbors that are also in this directory
            for neighbor in neighbors_data[img]["neighbors"]:
                if neighbor["filename"] in image_set:
                    similarity_graph[img].append(
                        {
                            "filename": neighbor["filename"],
                            "similarity": neighbor["sim"],
                        }
                    )
            # Sort neighbors by similarity (highest first)
            similarity_graph[img].sort(key=lambda x: x["similarity"], reverse=True)

    # Use a greedy approach to create spatial neighborhoods
    sorted_images = []
    used = set()

    # Start with the image that has the most high-similarity connections
    start_img = max(
        images,
        key=lambda img: len(
            [n for n in similarity_graph[img] if n["similarity"] > 0.5]
        ),
    )

    current = start_img
    sorted_images.append(current)
    used.add(current)

    # Greedily add the most similar unused neighbor
    while len(sorted_images) < len(images):
        best_next = None
        best_similarity = -1

        # Look for the best unused neighbor of the current image
        for neighbor in similarity_graph[current]:
            if (
                neighbor["filename"] not in used
                and neighbor["similarity"] > best_similarity
            ):
                best_next = neighbor["filename"]
                best_similarity = neighbor["similarity"]

        # If no good neighbor found, jump to the unused image with most connections
        if best_next is None:
            remaining = [img for img in images if img not in used]
            if remaining:
                best_next = max(remaining, key=lambda img: len(similarity_graph[img]))

        if best_next:
            sorted_images.append(best_next)
            used.add(best_next)
            current = best_next
        else:
            # Fallback: add any remaining images
            remaining = [img for img in images if img not in used]
            if remaining:
                sorted_images.extend(remaining)
            break

    return sorted_images


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
                directories.append(
                    {
                        "name": subdir.name,
                        "path": subdir,
                        "images": sorted(
                            [img.name for img in images]
                        ),  # Will be re-sorted by similarity later
                        "count": len(images),
                    }
                )

    return directories


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Similarity-Enhanced Image Viewer - {{ output_dir }}</title>
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
        
        .controls {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #007aff;
        }
        
        .controls button {
            background: #007aff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin-right: 10px;
            font-size: 14px;
        }
        
        .controls button:hover {
            background: #0056cc;
        }
        
        .controls button.active {
            background: #ff3b30;
        }
        
        .similarity-info {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            display: none;
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
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
            border: 3px solid transparent;
        }
        
        
        .image-container.selected {
            border-color: #007aff;
            box-shadow: 0 0 20px rgba(0, 122, 255, 0.5);
        }
        
        .image-container.neighbor-same-cluster {
            border-color: #34c759;
            box-shadow: 0 0 15px rgba(52, 199, 89, 0.4);
        }
        
        .image-container.neighbor-cross-cluster {
            border-color: #ff3b30;
            box-shadow: 0 0 15px rgba(255, 59, 48, 0.4);
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        
        .similarity-score {
            position: absolute;
            top: 4px;
            right: 4px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            display: none;
        }
        
        .legend {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 8px;
            border: 2px solid;
        }
        
        .legend-same { border-color: #34c759; }
        .legend-cross { border-color: #ff3b30; }
        .legend-selected { border-color: #007aff; }
        
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
        <h1>Similarity-Enhanced Image Viewer</h1>
        <p>Clustering results with similarity analysis and spatial layout: {{ output_dir }}</p>
        <p style="font-size: 14px; color: #86868b; margin-top: 8px;">
            ✨ Images are arranged by similarity within each cluster - similar images appear near each other
        </p>
    </div>
    
    <div class="controls">
        <button id="clear-selection">Clear Selection</button>
        <button id="toggle-scores">Show Similarity Scores</button>
        <strong>📊 Summary:</strong> {{ total_directories }} directories, {{ total_images }} images
    </div>
    
    <div class="similarity-info" id="similarity-info">
        <h3>Similarity Analysis</h3>
        <div id="similarity-details"></div>
    </div>
    
    <div class="legend" id="legend">
        <h4 style="margin-top: 0;">Connection Types</h4>
        <div class="legend-item">
            <div class="legend-color legend-selected"></div>
            <span>Selected Image</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-same"></div>
            <span>Similar (Same Cluster)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-cross"></div>
            <span>Similar (Different Cluster)</span>
        </div>
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
            <div class="image-container" 
                 data-filename="{{ image }}" 
                 data-directory="{{ directory.name }}"
                 onclick="selectImage('{{ image }}')">
                <img src="/image/{{ directory.name }}/{{ image }}" alt="{{ image }}" loading="lazy">
                <div class="similarity-score"></div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endfor %}
    
    <script>
        let currentSelection = null;
        let showingScores = false;
        let similarityData = {};
        
        // Load similarity data
        fetch('/similarity-data')
            .then(response => response.json())
            .then(data => {
                similarityData = data;
                console.log('Loaded similarity data:', Object.keys(data.neighbors).length, 'images');
            });
        
        function selectImage(filename) {
            clearSelection();
            currentSelection = filename;
            
            const container = document.querySelector(`[data-filename="${filename}"]`);
            container.classList.add('selected');
            
            // Show similarity info
            showSimilarityInfo(filename);
            
            // Highlight neighbors
            highlightNeighbors(filename);
            
            // Show legend
            document.getElementById('legend').style.display = 'block';
        }
        
        function clearSelection() {
            // Clear all highlights
            document.querySelectorAll('.image-container').forEach(container => {
                container.classList.remove('selected', 'neighbor-same-cluster', 'neighbor-cross-cluster');
                container.querySelector('.similarity-score').style.display = 'none';
            });
            
            // Hide info and legend
            document.getElementById('similarity-info').style.display = 'none';
            document.getElementById('legend').style.display = 'none';
            
            currentSelection = null;
        }
        
        function showSimilarityInfo(filename) {
            const neighborData = similarityData.neighbors[filename];
            if (!neighborData) return;
            
            const nodeData = similarityData.nodes[filename];
            const cluster = nodeData ? nodeData.label : 'unknown';
            
            let html = `
                <p><strong>Selected:</strong> ${filename} (Cluster ${cluster})</p>
                <p><strong>Neighbors found:</strong> ${neighborData.neighbors.length}</p>
            `;
            
            if (neighborData.neighbors.length > 0) {
                html += '<h4>Top Similar Images:</h4><ul>';
                neighborData.neighbors.slice(0, 5).forEach(neighbor => {
                    const neighborNode = similarityData.nodes[neighbor.filename];
                    const sameCluster = neighborNode && neighborNode.label === cluster;
                    const clusterInfo = sameCluster ? '✅ Same cluster' : '❌ Different cluster';
                    html += `<li><strong>${neighbor.filename}</strong> - Similarity: ${neighbor.sim.toFixed(3)} ${clusterInfo}</li>`;
                });
                html += '</ul>';
            }
            
            document.getElementById('similarity-details').innerHTML = html;
            document.getElementById('similarity-info').style.display = 'block';
        }
        
        function highlightNeighbors(filename) {
            const neighborData = similarityData.neighbors[filename];
            if (!neighborData) return;
            
            const nodeData = similarityData.nodes[filename];
            const sourceCluster = nodeData ? nodeData.label : null;
            
            neighborData.neighbors.forEach(neighbor => {
                const neighborContainer = document.querySelector(`[data-filename="${neighbor.filename}"]`);
                if (neighborContainer) {
                    const neighborNode = similarityData.nodes[neighbor.filename];
                    const sameCluster = neighborNode && neighborNode.label === sourceCluster;
                    
                    if (sameCluster) {
                        neighborContainer.classList.add('neighbor-same-cluster');
                    } else {
                        neighborContainer.classList.add('neighbor-cross-cluster');
                    }
                    
                    // Show similarity score
                    const scoreElement = neighborContainer.querySelector('.similarity-score');
                    scoreElement.textContent = neighbor.sim.toFixed(2);
                    if (showingScores) {
                        scoreElement.style.display = 'block';
                    }
                }
            });
        }
        
        // Controls
        document.getElementById('clear-selection').onclick = clearSelection;
        
        document.getElementById('toggle-scores').onclick = function() {
            showingScores = !showingScores;
            this.textContent = showingScores ? 'Hide Similarity Scores' : 'Show Similarity Scores';
            this.classList.toggle('active', showingScores);
            
            document.querySelectorAll('.similarity-score').forEach(score => {
                if (showingScores && score.textContent) {
                    score.style.display = 'block';
                } else {
                    score.style.display = 'none';
                }
            });
        };
    </script>
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

    # Load similarity data
    nodes, edges, neighbors = load_similarity_data(output_dir)

    # Find all directories with images
    directories = find_image_directories(output_dir)

    # Sort images within each directory by similarity
    print("🔄 Applying similarity-based spatial layout...")
    for directory in directories:
        original_order = directory["images"].copy()
        directory["images"] = similarity_sort_images(directory["images"], neighbors)

        # Count how many images moved positions
        moved_count = sum(
            1
            for i, img in enumerate(directory["images"])
            if i >= len(original_order) or img != original_order[i]
        )
        print(
            f"   • {directory['name']}: {moved_count}/{directory['count']} images repositioned"
        )

    total_images = sum(d["count"] for d in directories)

    print(f"📁 Found {len(directories)} directories with {total_images} total images")
    print(
        f"📊 Loaded similarity data: {len(nodes)} nodes, {len(edges)} edges, {len(neighbors)} neighbor sets"
    )
    for d in directories:
        print(f"   • {d['name']}: {d['count']} images")

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            directories=directories,
            total_directories=len(directories),
            total_images=total_images,
            output_dir=output_dir,
        )

    @app.route("/similarity-data")
    def similarity_data():
        """Serve similarity data as JSON for the frontend."""
        return jsonify({"nodes": nodes, "edges": edges, "neighbors": neighbors})

    @app.route("/image/<directory>/<filename>")
    def serve_image(directory, filename):
        """Serve images from the directories."""
        from flask import send_file

        image_path = output_path / directory / filename
        if image_path.exists():
            return send_file(str(image_path))
        return "Image not found", 404

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
    parser = argparse.ArgumentParser(
        description="Similarity-enhanced image viewer for clustering analysis"
    )
    parser.add_argument(
        "output_dir", help="Directory containing clustered images and similarity maps"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to run the server on (default: 5006)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )

    args = parser.parse_args()

    if not Path(args.output_dir).exists():
        print(f"❌ Error: Directory '{args.output_dir}' not found")
        sys.exit(1)

    # Check for similarity map files
    output_path = Path(args.output_dir)
    required_files = ["nodes.csv", "edges.csv", "neighbors.jsonl"]
    missing_files = [f for f in required_files if not (output_path / f).exists()]

    if missing_files:
        print(f"❌ Missing similarity map files: {missing_files}")
        print("   Run hybrid_grouper.py with --emit-map to generate these files")
        sys.exit(1)

    print("🚀 Starting Similarity-Enhanced Image Viewer...")
    print(f"📂 Analyzing: {args.output_dir}")

    app = create_app(args.output_dir)

    url = f"http://localhost:{args.port}"
    print(f"🌐 Server starting at: {url}")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()
