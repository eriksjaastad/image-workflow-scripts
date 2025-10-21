#!/usr/bin/env python3
"""
01 AI-Assisted Reviewer
========================

PHASE 3: Rule-Based Review Tool with AI Future

This tool reviews image groups and makes recommendations using:
  Step 1: SELECT best image from group (currently rule-based: pick highest stage)
  Step 2: CROP recommendation (currently: no crop needed)

Future: AI models will replace rules after Phase 2 training completes.

WORKFLOW:
---------
1. Groups images by timestamp (same logic as web image selector)
2. For each group:
   - AI/Rule recommends best image
   - User reviews: Approve (A), Override (1/2/3/4), Skip (S), Reject (R)
3. Logs decisions to sidecar .decision files (single source of truth)
4. NO file moves - this is review only

SIDECAR DECISION FILES:
-----------------------
For each image group, creates a .decision sidecar file:
  20250719_143022.decision  (JSON)
  
Content:
  {
    "group_id": "20250719_143022",
    "images": ["stage1.png", "stage2.png", "stage3.png"],
    "ai_recommendation": {
      "selected_image": "stage3.png",
      "selected_index": 2,
      "reason": "Highest stage (rule-based)",
      "confidence": 1.0,
      "crop_needed": false
    },
    "user_decision": {
      "action": "approve",  // or "override", "reject", "skip"
      "selected_image": "stage3.png",  // if override, user's choice
      "selected_index": 2,
      "timestamp": "2025-10-20T12:00:00Z"
    }
  }

KEYS:
-----
A - Approve AI recommendation
R - Reject (keep all images)
S - Skip (review later)
1/2/3/4 - Override: select different image
Enter/↓ - Next group
↑ - Previous group
Shift+Enter - Submit batch

USAGE:
------
  python scripts/01_ai_assisted_reviewer.py sandbox/mojo2/selected/

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse existing grouping logic - NO reinventing the wheel!
sys.path.insert(0, str(Path(__file__).parent))
from utils.companion_file_utils import (
    extract_datetime_from_filename,
    find_consecutive_stage_groups,
    get_stage_number,
    detect_stage,
    sort_image_files_by_timestamp_and_stage,
)

try:
    from flask import Flask, Response, jsonify, render_template_string, request
except Exception:
    print("[!] Flask is required. Install with: pip install flask", file=sys.stderr)
    raise

try:
    from PIL import Image
except Exception:
    print("[!] Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise


@dataclass
class ImageGroup:
    """Represents a group of images with same timestamp."""
    group_id: str  # timestamp identifier
    images: List[Path]  # sorted by stage
    directory: Path  # parent directory


def scan_images(directory: Path) -> List[Path]:
    """Scan directory for PNG images."""
    if not directory.exists():
        return []
    return list(directory.rglob("*.png"))


def group_images_by_timestamp(images: List[Path]) -> List[ImageGroup]:
    """
    Group images using EXACT same logic as web image selector.
    Reuses find_consecutive_stage_groups from companion_file_utils.
    """
    # Sort first (required by grouping logic)
    sorted_images = sort_image_files_by_timestamp_and_stage(images)
    
    # Group by timestamp and stage progression
    grouped = find_consecutive_stage_groups(sorted_images, min_group_size=2)
    
    # Convert to ImageGroup objects
    result = []
    for group_paths in grouped:
        if not group_paths:
            continue
        
        # Use first image timestamp as group ID
        first_img = group_paths[0]
        dt = extract_datetime_from_filename(first_img.name)
        if dt:
            group_id = dt.strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback: use stem of first file
            group_id = first_img.stem.split('_stage')[0]
        
        result.append(ImageGroup(
            group_id=group_id,
            images=group_paths,
            directory=first_img.parent
        ))
    
    return result


def get_rule_based_recommendation(group: ImageGroup) -> Dict:
    """
    Rule-based recommendation (Phase 3 temporary, before AI training).
    
    Rule: Pick highest stage image (stage3 > stage2 > stage1.5 > stage1)
    """
    best_image = group.images[-1]  # Last image = highest stage (already sorted)
    best_index = len(group.images) - 1
    
    stage = detect_stage(best_image.name) or "unknown"
    
    return {
        "selected_image": best_image.name,
        "selected_index": best_index,
        "reason": f"Highest stage: {stage} (rule-based)",
        "confidence": 1.0,
        "crop_needed": False,
        "crop_coords": None
    }


def load_or_create_decision_file(group: ImageGroup) -> Dict:
    """
    Load existing decision or create new one.
    Decision files are stored alongside images with .decision extension.
    """
    decision_path = group.directory / f"{group.group_id}.decision"
    
    if decision_path.exists():
        with open(decision_path, 'r') as f:
            return json.load(f)
    
    # Create new decision
    recommendation = get_rule_based_recommendation(group)
    
    return {
        "group_id": group.group_id,
        "images": [img.name for img in group.images],
        "ai_recommendation": recommendation,
        "user_decision": None  # Not reviewed yet
    }


def save_decision_file(group: ImageGroup, decision_data: Dict) -> None:
    """
    Save decision to sidecar .decision file.
    This is the SINGLE SOURCE OF TRUTH for user decisions.
    """
    decision_path = group.directory / f"{group.group_id}.decision"
    
    with open(decision_path, 'w') as f:
        json.dump(decision_data, f, indent=2)


def build_app(groups: List[ImageGroup], base_dir: Path) -> Flask:
    """Build Flask app for reviewing image groups."""
    app = Flask(__name__)
    app.config["GROUPS"] = groups
    app.config["BASE_DIR"] = base_dir
    app.config["CURRENT_INDEX"] = 0
    app.config["DECISIONS"] = {}  # Track decisions in session
    
    # HTML template (full implementation with JavaScript)
    page_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>AI-Assisted Reviewer</title>
      <style>
        :root {
          color-scheme: dark;
          --bg: #101014;
          --surface: #181821;
          --surface-alt: #1f1f2c;
          --accent: #4f9dff;
          --success: #51cf66;
          --danger: #ff6b6b;
          --warning: #ffd43b;
          --muted: #a0a3b1;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Inter", "Segoe UI", system-ui, sans-serif;
          background: var(--bg);
          color: #f8f9ff;
        }
        header {
          background: var(--bg);
          padding: 1rem 2rem;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        header h1 {
          margin: 0 0 0.5rem 0;
          font-size: 1.5rem;
        }
        .progress {
          display: flex;
          gap: 1rem;
          align-items: center;
          color: var(--muted);
          font-size: 0.9rem;
        }
        .progress strong {
          color: var(--accent);
        }
        #status {
          margin-top: 0.5rem;
          font-weight: 500;
        }
        #status.success { color: var(--success); }
        #status.error { color: var(--danger); }
        main {
          padding: 120px 2rem 2rem;
          max-width: 1600px;
          margin: 0 auto;
        }
        .group-card {
          background: var(--surface);
          padding: 2rem;
          border-radius: 12px;
          margin-bottom: 2rem;
          box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        .group-header {
          margin-bottom: 1.5rem;
        }
        .group-header h2 {
          margin: 0 0 0.5rem 0;
          color: var(--accent);
        }
        .group-header .meta {
          color: var(--muted);
          font-size: 0.9rem;
        }
        .recommendation {
          background: rgba(81, 207, 102, 0.15);
          border: 2px solid rgba(81, 207, 102, 0.3);
          padding: 1rem 1.5rem;
          border-radius: 8px;
          margin-bottom: 2rem;
        }
        .recommendation strong {
          color: var(--success);
          display: block;
          margin-bottom: 0.5rem;
        }
        .recommendation .reason {
          color: #f8f9ff;
          margin-bottom: 0.3rem;
        }
        .recommendation .confidence {
          color: var(--muted);
          font-size: 0.9rem;
        }
        .images-row {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }
        .image-card {
          background: var(--surface-alt);
          padding: 1rem;
          border: 3px solid transparent;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          position: relative;
        }
        .image-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(79, 157, 255, 0.2);
        }
        .image-card.ai-pick {
          border-color: var(--success);
          background: rgba(81, 207, 102, 0.08);
        }
        .image-card.ai-pick::before {
          content: "AI PICK";
          position: absolute;
          top: 0.5rem;
          right: 0.5rem;
          background: var(--success);
          color: black;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.05em;
        }
        .image-card.user-override {
          border-color: var(--warning);
          background: rgba(255, 212, 59, 0.08);
        }
        .image-card img {
          width: 100%;
          height: auto;
          border-radius: 6px;
          margin-bottom: 0.5rem;
          display: block;
        }
        .image-card .filename {
          font-size: 0.75rem;
          color: var(--muted);
          margin-bottom: 0.5rem;
          word-break: break-all;
        }
        .image-card .stage {
          background: rgba(79, 157, 255, 0.2);
          padding: 0.3rem 0.6rem;
          border-radius: 4px;
          font-size: 0.8rem;
          display: inline-block;
          margin-bottom: 0.5rem;
        }
        .image-card .select-btn {
          width: 100%;
          padding: 0.5rem;
          background: var(--accent);
          color: black;
          border: none;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        .image-card .select-btn:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(79, 157, 255, 0.4);
        }
        .actions {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }
        .btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s;
          flex: 1;
          min-width: 150px;
        }
        .btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .btn-approve { 
          background: var(--success); 
          color: black; 
        }
        .btn-reject { 
          background: var(--danger); 
          color: white; 
        }
        .btn-skip { 
          background: var(--muted); 
          color: white; 
        }
        .btn-nav {
          background: var(--surface-alt);
          color: white;
          border: 2px solid rgba(255,255,255,0.1);
          flex: 0 0 auto;
          min-width: 100px;
        }
        .help {
          background: var(--surface-alt);
          padding: 1rem;
          border-radius: 8px;
          margin-top: 2rem;
          color: var(--muted);
          font-size: 0.9rem;
        }
        .help strong {
          color: #f8f9ff;
        }
      </style>
    </head>
    <body>
      <header>
        <h1>🤖 AI-Assisted Reviewer</h1>
        <div class="progress">
          <span>Group <strong id="current-num">{{ current + 1 }}</strong> of <strong>{{ total }}</strong></span>
          <span>•</span>
          <span>Reviewed: <strong id="reviewed-count">0</strong></span>
          <span>•</span>
          <span>Approved: <strong id="approved-count">0</strong></span>
          <span>•</span>
          <span>Overridden: <strong id="override-count">0</strong></span>
        </div>
        <div id="status"></div>
      </header>
      <main>
        <div class="group-card">
          <div class="group-header">
            <h2>Group: {{ group.group_id }}</h2>
            <div class="meta">{{ group.images|length }} images • {{ group.directory.name }} directory</div>
          </div>
          
          <div class="recommendation">
            <strong>🎯 AI Recommendation</strong>
            <div class="reason">{{ recommendation.reason }}</div>
            <div class="confidence">Confidence: {{ "%.0f"|format(recommendation.confidence * 100) }}%</div>
          </div>
          
          <div class="images-row">
            {% for img in group.images %}
            <div class="image-card {% if loop.index0 == recommendation.selected_index %}ai-pick{% endif %}"
                 data-index="{{ loop.index0 }}"
                 onclick="selectImage({{ loop.index0 }})">
              <div class="stage">{{ img.name.split('_')[2] if '_' in img.name else 'stage?' }}</div>
              <img src="/image/{{ group.group_id }}/{{ loop.index0 }}" 
                   alt="{{ img.name }}"
                   loading="lazy">
              <div class="filename">{{ img.name }}</div>
              <button class="select-btn" onclick="event.stopPropagation(); selectImage({{ loop.index0 }})">
                Select [{{ loop.index }}]
              </button>
            </div>
            {% endfor %}
          </div>
          
          <div class="actions">
            <button class="btn btn-approve" onclick="approve()">
              ✓ Approve [A]
            </button>
            <button class="btn btn-reject" onclick="reject()">
              ✗ Reject All [R]
            </button>
            <button class="btn btn-skip" onclick="skip()">
              ⊙ Skip [S]
            </button>
            <button class="btn btn-nav" onclick="navigate(-1)">
              ← Prev [↑]
            </button>
            <button class="btn btn-nav" onclick="navigate(1)">
              Next [↓] →
            </button>
          </div>
        </div>
        
        <div class="help">
          <strong>Keyboard Shortcuts:</strong><br>
          <strong>A</strong> - Approve AI recommendation &nbsp;|&nbsp;
          <strong>R</strong> - Reject (keep all images) &nbsp;|&nbsp;
          <strong>S</strong> - Skip (review later)<br>
          <strong>1-4</strong> - Override: select different image &nbsp;|&nbsp;
          <strong>↑/↓ or Enter</strong> - Navigate groups
        </div>
      </main>
      
      <script>
        const groupId = "{{ group.group_id }}";
        const aiRecommendation = {{ recommendation.selected_index }};
        const totalImages = {{ group.images|length }};
        let userSelection = null;
        
        // Stats tracking
        let stats = {
          reviewed: 0,
          approved: 0,
          overridden: 0,
          rejected: 0,
          skipped: 0
        };
        
        function updateStats() {
          document.getElementById('reviewed-count').textContent = stats.reviewed;
          document.getElementById('approved-count').textContent = stats.approved;
          document.getElementById('override-count').textContent = stats.overridden;
        }
        
        function setStatus(message, type = '') {
          const statusEl = document.getElementById('status');
          statusEl.textContent = message;
          statusEl.className = type;
          if (message) {
            setTimeout(() => {
              statusEl.textContent = '';
              statusEl.className = '';
            }, 3000);
          }
        }
        
        function selectImage(index) {
          // Remove all override highlights
          document.querySelectorAll('.image-card').forEach(card => {
            card.classList.remove('user-override');
          });
          
          // Highlight selected
          const cards = document.querySelectorAll('.image-card');
          if (index >= 0 && index < cards.length) {
            cards[index].classList.add('user-override');
            userSelection = index;
            setStatus(`Selected image ${index + 1} (overriding AI)`, 'success');
          }
        }
        
        async function submitDecision(action, selectedIndex = null) {
          const decision = {
            group_id: groupId,
            action: action,
            selected_index: selectedIndex,
            ai_index: aiRecommendation
          };
          
          try {
            const response = await fetch('/submit', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(decision)
            });
            
            const result = await response.json();
            
            if (result.status === 'ok') {
              stats.reviewed++;
              if (action === 'approve') stats.approved++;
              if (action === 'override') stats.overridden++;
              if (action === 'reject') stats.rejected++;
              if (action === 'skip') stats.skipped++;
              updateStats();
              
              setStatus(result.message, 'success');
              
              // Auto-advance after short delay
              setTimeout(() => {
                window.location.href = '/next';
              }, 500);
            } else {
              setStatus(result.message || 'Error submitting decision', 'error');
            }
          } catch (error) {
            setStatus('Network error: ' + error.message, 'error');
          }
        }
        
        function approve() {
          submitDecision('approve', aiRecommendation);
        }
        
        function reject() {
          if (confirm('Reject all images in this group? They will remain unprocessed.')) {
            submitDecision('reject', null);
          }
        }
        
        function skip() {
          submitDecision('skip', null);
        }
        
        function navigate(direction) {
          const url = direction > 0 ? '/next' : '/prev';
          window.location.href = url;
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
          // Ignore if typing in input
          if (e.target.matches('input, textarea')) return;
          
          const key = e.key.toLowerCase();
          
          switch(key) {
            case 'a':
              e.preventDefault();
              approve();
              break;
            case 'r':
              e.preventDefault();
              reject();
              break;
            case 's':
              e.preventDefault();
              skip();
              break;
            case '1':
            case '2':
            case '3':
            case '4':
              e.preventDefault();
              const index = parseInt(key) - 1;
              if (index < totalImages) {
                selectImage(index);
                // Auto-submit override after selection
                setTimeout(() => submitDecision('override', index), 300);
              }
              break;
            case 'enter':
            case 'arrowdown':
              e.preventDefault();
              navigate(1);
              break;
            case 'arrowup':
              e.preventDefault();
              navigate(-1);
              break;
          }
        });
        
        // Load stats from session storage
        const savedStats = sessionStorage.getItem('reviewer_stats');
        if (savedStats) {
          stats = JSON.parse(savedStats);
          updateStats();
        }
        
        // Save stats on page unload
        window.addEventListener('beforeunload', () => {
          sessionStorage.setItem('reviewer_stats', JSON.stringify(stats));
        });
      </script>
    </body>
    </html>
    """
    
    @app.route("/")
    def index():
        """Show current group for review."""
        groups = app.config["GROUPS"]
        current = app.config["CURRENT_INDEX"]
        
        if current >= len(groups):
            return """
            <html>
            <head><title>Review Complete</title></head>
            <body style="background: #101014; color: #f8f9ff; font-family: sans-serif; 
                         text-align: center; padding: 4rem;">
              <h1>🎉 All Groups Reviewed!</h1>
              <p>You've completed reviewing all image groups.</p>
              <p><a href="/stats" style="color: #4f9dff;">View Summary Statistics</a></p>
            </body>
            </html>
            """
        
        group = groups[current]
        decision_data = load_or_create_decision_file(group)
        
        return render_template_string(
            page_template,
            group=group,
            recommendation=decision_data["ai_recommendation"],
            total=len(groups),
            current=current
        )
    
    @app.route("/submit", methods=["POST"])
    def submit():
        """Handle decision submission."""
        try:
            data = request.get_json()
            group_id = data.get("group_id")
            action = data.get("action")
            selected_index = data.get("selected_index")
            ai_index = data.get("ai_index")
            
            # Find group
            groups = app.config["GROUPS"]
            group = next((g for g in groups if g.group_id == group_id), None)
            
            if not group:
                return jsonify({"status": "error", "message": "Group not found"}), 404
            
            # Load decision file
            decision_data = load_or_create_decision_file(group)
            
            # Update with user decision
            decision_data["user_decision"] = {
                "action": action,
                "selected_image": group.images[selected_index].name if selected_index is not None else None,
                "selected_index": selected_index,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Save decision file
            save_decision_file(group, decision_data)
            
            # Track in session
            app.config["DECISIONS"][group_id] = decision_data
            
            # Build response message
            if action == "approve":
                msg = f"✓ Approved AI pick: {group.images[ai_index].name}"
            elif action == "override":
                msg = f"⚡ Override: Selected {group.images[selected_index].name}"
            elif action == "reject":
                msg = "✗ Rejected: All images kept"
            elif action == "skip":
                msg = "⊙ Skipped for later review"
            else:
                msg = "Decision recorded"
            
            return jsonify({"status": "ok", "message": msg})
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/next")
    def next_group():
        """Navigate to next group."""
        current = app.config["CURRENT_INDEX"]
        app.config["CURRENT_INDEX"] = min(current + 1, len(app.config["GROUPS"]))
        return index()
    
    @app.route("/prev")
    def prev_group():
        """Navigate to previous group."""
        current = app.config["CURRENT_INDEX"]
        app.config["CURRENT_INDEX"] = max(0, current - 1)
        return index()
    
    @app.route("/image/<group_id>/<int:index>")
    def serve_image(group_id: str, index: int):
        """Serve image thumbnail."""
        groups = app.config["GROUPS"]
        group = next((g for g in groups if g.group_id == group_id), None)
        
        if not group or index >= len(group.images):
            return "Not found", 404
        
        img_path = group.images[index]
        
        try:
            with Image.open(img_path) as img:
                # Create thumbnail (max 600px to maintain quality)
                img.thumbnail((600, 600), Image.Resampling.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format='PNG', optimize=True)
                buf.seek(0)
                return Response(buf.read(), mimetype='image/png')
        except Exception as e:
            return f"Error loading image: {e}", 500
    
    @app.route("/stats")
    def stats():
        """Show summary statistics."""
        decisions = app.config["DECISIONS"]
        
        approved = sum(1 for d in decisions.values() 
                      if d.get("user_decision", {}).get("action") == "approve")
        overridden = sum(1 for d in decisions.values() 
                        if d.get("user_decision", {}).get("action") == "override")
        rejected = sum(1 for d in decisions.values() 
                      if d.get("user_decision", {}).get("action") == "reject")
        skipped = sum(1 for d in decisions.values() 
                     if d.get("user_decision", {}).get("action") == "skip")
        
        total = len(app.config["GROUPS"])
        reviewed = len(decisions)
        
        return f"""
        <html>
        <head><title>Review Statistics</title></head>
        <body style="background: #101014; color: #f8f9ff; font-family: sans-serif; 
                     padding: 2rem; max-width: 800px; margin: 0 auto;">
          <h1>📊 Review Statistics</h1>
          <div style="background: #181821; padding: 2rem; border-radius: 12px; margin: 2rem 0;">
            <p><strong>Total Groups:</strong> {total}</p>
            <p><strong>Reviewed:</strong> {reviewed} ({reviewed*100//total if total else 0}%)</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p><strong>✓ Approved:</strong> {approved}</p>
            <p><strong>⚡ Overridden:</strong> {overridden}</p>
            <p><strong>✗ Rejected:</strong> {rejected}</p>
            <p><strong>⊙ Skipped:</strong> {skipped}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p><strong>AI Agreement Rate:</strong> {approved*100//(approved+overridden) if (approved+overridden) else 0}%</p>
          </div>
          <p><a href="/" style="color: #4f9dff;">← Back to Review</a></p>
        </body>
        </html>
        """
    
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review image groups with AI assistance (rule-based for Phase 3)"
    )
    parser.add_argument("directory", type=str, help="Directory containing images to review")
    parser.add_argument("--host", default="127.0.0.1", help="Host for web server")
    parser.add_argument("--port", type=int, default=8081, help="Port for web server")
    args = parser.parse_args()
    
    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists():
        print(f"[!] Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    
    # Scan and group images
    print(f"[*] Scanning {directory}...")
    images = scan_images(directory)
    print(f"[*] Found {len(images)} images")
    
    print(f"[*] Grouping images by timestamp...")
    groups = group_images_by_timestamp(images)
    print(f"[*] Found {len(groups)} groups")
    
    if not groups:
        print("[!] No image groups found. Check directory and file naming.", file=sys.stderr)
        sys.exit(1)
    
    # Build and run Flask app
    app = build_app(groups, directory)
    print(f"\n[*] Starting reviewer on http://{args.host}:{args.port}")
    print(f"[*] Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

