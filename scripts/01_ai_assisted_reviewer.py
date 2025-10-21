#!/usr/bin/env python3
"""
01 AI-Assisted Reviewer
========================

REPLACES: Web Image Selector (01) + Desktop Multi-Crop (04)

This tool combines image selection and crop proposal into ONE integrated workflow.

PHASE 3: Rule-Based Review Tool with AI Future

This tool reviews image groups and makes recommendations using:
  Step 1: SELECT best image from group (currently rule-based: pick highest stage)
  Step 2: CROP recommendation (currently: no crop needed)

Future: AI models will replace rules after Phase 2 training completes.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate
  
USAGE:
------
  python scripts/01_ai_assisted_reviewer.py <raw_images_directory>/

Example:
  python scripts/01_ai_assisted_reviewer.py character_group_1/



WORKFLOW:
---------
1. Groups images by timestamp (same logic as web image selector)
2. For each group:
   - AI/Rule recommends best image
   - User reviews: Approve (A), Override (1/2/3/4), Manual Crop (C), Reject (R)
3. FILE OPERATIONS executed immediately:
   - Approve/Override → Move to selected/
   - Manual Crop → Move to crop/ (for manual cropping later)
   - Reject → Move ALL to delete_staging/ (fast deletion staging)
4. Training data logged automatically (selection + crop decisions)
5. Logs decisions to sidecar .decision files (single source of truth)

FILE ROUTING:
-------------
| Action       | Selected Image → | Other Images →    |
|--------------|------------------|-------------------|
| Approve      | selected/        | delete_staging/   |
| Override     | selected/        | delete_staging/   |
| Manual Crop  | crop/            | delete_staging/   |
| Reject       | delete_staging/  | delete_staging/   |

NOTE: All images MUST be processed. Directory should be empty when done.

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
      "action": "approve",  // or "override", "manual_crop", "reject"
      "selected_image": "stage3.png",
      "selected_index": 2,
      "timestamp": "2025-10-21T12:00:00Z"
    }
  }

KEYS:
-----
A - Approve AI recommendation
C - Send selected to crop directory (manual crop)
R - Reject (delete all images in group)
1/2/3/4 - Override: select different image
Enter/↓ - Next group
↑ - Previous group

DIRECTORY STRUCTURE:
--------------------
The script will automatically create and use these directories at the project root:
  selected/         - Final selected images (ready for next step)
  crop/             - Images that need manual cropping
  delete_staging/   - Fast deletion staging (move to Trash later)
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
    move_file_with_all_companions,
    log_selection_only_entry,
    log_crop_decision,  # NEW: Minimal schema crop logging
)
from file_tracker import FileTracker

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

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    print("[!] PyTorch not available - will use rule-based recommendations only")
    TORCH_AVAILABLE = False

try:
    import open_clip
    CLIP_AVAILABLE = True
except Exception:
    print("[!] CLIP not available - will use rule-based recommendations only")
    CLIP_AVAILABLE = False


@dataclass
class ImageGroup:
    """Represents a group of images with same timestamp."""
    group_id: str  # timestamp identifier
    images: List[Path]  # sorted by stage
    directory: Path  # parent directory


# ============================
# AI Model Architecture
# ============================

class RankerModel(nn.Module):
    """MLP ranking model - picks best image from group (matches train_ranker_v3.py)."""
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CropProposerModel(nn.Module):
    """MLP crop proposer - predicts crop coordinates."""
    def __init__(self, input_dim=514):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Output normalized [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)


def load_ai_models(models_dir: Path) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[any]]:
    """
    Load Ranker v3 and Crop Proposer v2 models.
    
    Returns: (ranker_model, crop_model, clip_model) or (None, None, None) if unavailable
    """
    if not TORCH_AVAILABLE or not CLIP_AVAILABLE:
        print("[!] AI models not available (PyTorch or CLIP missing)")
        return None, None, None
    
    try:
        # Load CLIP for embeddings
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[*] Loading CLIP model on {device}...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        # Load Ranker v3
        ranker_path = models_dir / "ranker_v3_w10.pt"
        if ranker_path.exists():
            print(f"[*] Loading Ranker v3 from {ranker_path}...")
            ranker = RankerModel(input_dim=512).to(device)
            ranker.load_state_dict(torch.load(ranker_path, map_location=device))
            ranker.eval()
            print("[✓] Ranker v3 loaded successfully")
        else:
            print(f"[!] Ranker v3 not found at {ranker_path}")
            ranker = None
        
        # Load Crop Proposer v2
        crop_path = models_dir / "crop_proposer_v2.pt"
        if crop_path.exists():
            print(f"[*] Loading Crop Proposer v2 from {crop_path}...")
            crop_proposer = CropProposerModel(input_dim=514).to(device)
            crop_proposer.load_state_dict(torch.load(crop_path, map_location=device))
            crop_proposer.eval()
            print("[✓] Crop Proposer v2 loaded successfully")
        else:
            print(f"[!] Crop Proposer v2 not found at {crop_path}")
            crop_proposer = None
        
        return ranker, crop_proposer, (clip_model, preprocess, device)
        
    except Exception as e:
        print(f"[!] Error loading AI models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_image_embedding(image_path: Path, clip_model, preprocess, device) -> Optional[torch.Tensor]:
    """Get CLIP embedding for an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)
            embedding = embedding.float()
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze(0)  # Remove batch dimension
    
    except Exception as e:
        print(f"[!] Error getting embedding for {image_path.name}: {e}")
        return None


def get_ai_recommendation(group: ImageGroup, ranker_model, crop_model, clip_info) -> Dict:
    """
    Get AI recommendation using Ranker v3 and Crop Proposer v2.
    
    Returns dict with:
        - selected_image: filename
        - selected_index: int
        - reason: str
        - confidence: float
        - crop_needed: bool
        - crop_coords: tuple or None (normalized [0,1] coordinates)
    """
    if ranker_model is None or clip_info is None:
        # Fall back to rule-based
        return get_rule_based_recommendation(group)
    
    clip_model, preprocess, device = clip_info
    
    try:
        # Get embeddings for all images in group
        embeddings = []
        for img_path in group.images:
            emb = get_image_embedding(img_path, clip_model, preprocess, device)
            if emb is None:
                # Fall back to rule-based if any embedding fails
                return get_rule_based_recommendation(group)
            embeddings.append(emb)
        
        # Stack embeddings and run through ranker
        embeddings_tensor = torch.stack(embeddings).to(device)
        
        with torch.no_grad():
            scores = ranker_model(embeddings_tensor).squeeze(-1)  # Shape: (num_images,)
        
        # Pick highest scoring image
        best_idx = scores.argmax().item()
        best_image = group.images[best_idx]
        best_score = scores[best_idx].item()
        confidence = torch.sigmoid(scores[best_idx]).item()  # Convert to probability
        
        # Build detailed reason with all scores
        stage = detect_stage(best_image.name) or "unknown"
        score_details = []
        for idx, (img, score) in enumerate(zip(group.images, scores)):
            img_stage = detect_stage(img.name) or f"img{idx+1}"
            marker = " ✓" if idx == best_idx else ""
            score_details.append(f"{img_stage}: {score.item():.2f}{marker}")
        
        reason = f"AI picked {stage} (score: {best_score:.2f}) | " + " • ".join(score_details)
        
        # Get crop proposal if model available
        crop_coords = None
        crop_needed = False
        
        if crop_model is not None:
            try:
                # Get dimensions of best image
                with Image.open(best_image) as img:
                    width, height = img.size
                
                # Prepare input: embedding + NORMALIZED dimensions (as in training)
                best_embedding = embeddings[best_idx]
                dims = torch.tensor([width / 2048.0, height / 2048.0], dtype=torch.float32).to(device)
                crop_input = torch.cat([best_embedding, dims]).unsqueeze(0)  # Add batch dim
                
                with torch.no_grad():
                    crop_output = crop_model(crop_input).squeeze(0)  # Remove batch dim
                
                # Extract normalized coordinates
                x1, y1, x2, y2 = crop_output.cpu().numpy()
                
                # Check if crop is meaningful (not ~full image)
                crop_area = (x2 - x1) * (y2 - y1)
                if crop_area < 0.95:  # If cropping more than 5%
                    crop_needed = True
                    crop_coords = (float(x1), float(y1), float(x2), float(y2))
                
            except Exception as e:
                print(f"[!] Error getting crop proposal: {e}")
                crop_needed = False
                crop_coords = None
        
        return {
            "selected_image": best_image.name,
            "selected_index": best_idx,
            "reason": reason,
            "confidence": confidence,
            "crop_needed": crop_needed,
            "crop_coords": crop_coords
        }
        
    except Exception as e:
        print(f"[!] Error in AI recommendation: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to rule-based
        return get_rule_based_recommendation(group)


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


def load_or_create_decision_file(group: ImageGroup, ranker_model=None, crop_model=None, clip_info=None) -> Dict:
    """
    Load existing decision or create new one.
    Decision files are stored alongside images with .decision extension.
    """
    decision_path = group.directory / f"{group.group_id}.decision"
    
    if decision_path.exists():
        with open(decision_path, 'r') as f:
            return json.load(f)
    
    # Create new decision with AI recommendation (or rule-based fallback)
    if ranker_model is not None and clip_info is not None:
        recommendation = get_ai_recommendation(group, ranker_model, crop_model, clip_info)
    else:
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


def find_project_root(directory: Path) -> Path:
    """
    Find project root directory by looking for specific markers.
    Falls back to current directory if not found.
    """
    current = directory.resolve()
    
    # Look for project markers
    markers = ["scripts", "data", "selected", "crop"]
    
    for _ in range(5):  # Search up to 5 levels
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    # Fallback: use directory itself
    return directory.resolve()


def get_current_project_id() -> str:
    """
    Get project ID from the CURRENT active project manifest.
    
    Reads from data/projects/*.project.json files and returns the projectId
    from the project where finishedAt is null (the active project).
    
    This is the CORRECT way to get project ID - reading from the project
    manifest system, not guessing from directory names!
    
    Returns:
        Project ID string (e.g., 'mojo3') or 'unknown' if no active project found
    """
    import json
    
    # Find the project manifest directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent / "data" / "projects"
    
    if not project_dir.exists():
        print(f"Warning: Project directory not found: {project_dir}")
        return "unknown"
    
    # Look for active project (finishedAt is null)
    for project_file in project_dir.glob("*.project.json"):
        try:
            with open(project_file, 'r') as f:
                data = json.load(f)
                
                # Active project has no finish date
                if data.get("finishedAt") is None:
                    project_id = data.get("projectId", "unknown")
                    print(f"[*] Found active project: {project_id} (from {project_file.name})")
                    return project_id
        except Exception as e:
            print(f"Warning: Failed to read {project_file.name}: {e}")
            continue
    
    print("Warning: No active project found (no project with finishedAt=null)")
    return "unknown"


def perform_file_operations(group: ImageGroup, action: str, selected_index: Optional[int],
                            crop_coords: Optional[Tuple[float, float, float, float]],
                            tracker: FileTracker, selected_dir: Path, crop_dir: Path, 
                            delete_staging_dir: Path, project_id: str = "unknown") -> str:
    """
    Execute file operations based on user decision.
    
    Args:
        project_id: Project identifier (e.g., 'mojo1', 'mojo3') for training data
    
    Returns: Summary message of what was done
    """
    if action == "reject":
        # Delete all images in group - move to delete_staging
        moved_count = 0
        for img_path in group.images:
            try:
                move_file_with_all_companions(img_path, delete_staging_dir, dry_run=False)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {img_path.name} to delete staging: {e}")
        
        tracker.log_operation(
            "stage_delete",
            str(group.directory),
            str(delete_staging_dir.name),
            moved_count,
            f"Rejected group {group.group_id}",
            [img.name for img in group.images[:5]]
        )
        return f"Rejected: {moved_count} images moved to delete staging"
    
    if action == "reject_single":
        # Delete just one image - move to delete_staging
        if selected_index is None:
            return "Error: No image selected for rejection"
        
        selected_image = group.images[selected_index]
        try:
            move_file_with_all_companions(selected_image, delete_staging_dir, dry_run=False)
            tracker.log_operation(
                "stage_delete",
                str(group.directory),
                str(delete_staging_dir.name),
                1,
                f"Rejected single image from group {group.group_id}",
                [selected_image.name]
            )
            return f"Rejected: {selected_image.name} moved to delete staging"
        except Exception as e:
            return f"Error: {e}"
    
    if selected_index is None:
        return "Error: No image selected"
    
    selected_image = group.images[selected_index]
    other_images = [img for i, img in enumerate(group.images) if i != selected_index]
    
    # Determine destination based on action
    if action == "manual_crop":
        # User wants to crop manually later
        dest_dir = crop_dir
        dest_label = "crop directory (manual crop needed)"
    else:
        # approve or override - goes to selected
        dest_dir = selected_dir
        dest_label = "selected directory"
    
    try:
        # Move selected image
        move_file_with_all_companions(selected_image, dest_dir, dry_run=False)
        
        # Move others to delete staging
        for img_path in other_images:
            try:
                move_file_with_all_companions(img_path, delete_staging_dir, dry_run=False)
            except Exception as e:
                print(f"Error moving {img_path.name} to delete staging: {e}")
        
        # Log training data
        negative_paths = other_images
        try:
            # Always log selection
            log_selection_only_entry(
                session_id=f"ai_reviewer_{datetime.utcnow().strftime('%Y%m%d')}",
                set_id=group.group_id,
                chosen_path=str(selected_image),
                negative_paths=[str(p) for p in negative_paths]
            )
            
            # If we have crop coordinates, log them too (NEW SCHEMA!)
            if crop_coords is not None and action == "approve":
                try:
                    from PIL import Image
                    img = Image.open(selected_image)
                    width, height = img.size
                    img.close()
                    
                    log_crop_decision(
                        project_id=project_id,
                        filename=selected_image.name,
                        crop_coords=crop_coords,
                        width=width,
                        height=height
                    )
                except Exception as e:
                    print(f"Warning: Failed to log crop decision: {e}")
        except Exception as e:
            print(f"Warning: Failed to log training data: {e}")
        
        tracker.log_operation(
            "move",
            str(group.directory),
            dest_dir.name,
            1,
            f"Selected image from group {group.group_id}",
            [selected_image.name]
        )
        
        return f"Moved {selected_image.name} to {dest_label}, {len(other_images)} to delete staging"
        
    except Exception as e:
        return f"Error during file operations: {e}"


def build_app(groups: List[ImageGroup], base_dir: Path, tracker: FileTracker, 
               selected_dir: Path, crop_dir: Path, delete_staging_dir: Path,
               ranker_model=None, crop_model=None, clip_info=None, batch_size: int = 20) -> Flask:
    """Build Flask app for reviewing image groups."""
    app = Flask(__name__)
    app.config["ALL_GROUPS"] = groups  # Full list
    app.config["BATCH_SIZE"] = batch_size
    app.config["CURRENT_BATCH"] = 0
    
    # Calculate batch
    total_groups = len(groups)
    batch_start = 0
    batch_end = min(batch_start + batch_size, total_groups)
    current_batch_groups = groups[batch_start:batch_end]
    
    app.config["GROUPS"] = current_batch_groups  # Current batch only
    app.config["BASE_DIR"] = base_dir
    app.config["CURRENT_INDEX"] = 0
    app.config["DECISIONS"] = {}  # Track decisions in session
    app.config["TRACKER"] = tracker
    app.config["SELECTED_DIR"] = selected_dir
    app.config["CROP_DIR"] = crop_dir
    app.config["DELETE_STAGING_DIR"] = delete_staging_dir
    app.config["PROJECT_ID"] = get_current_project_id()  # Read from project manifest!
    app.config["RANKER_MODEL"] = ranker_model
    app.config["CROP_MODEL"] = crop_model
    app.config["CLIP_INFO"] = clip_info
    
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
          padding: 0.5rem 1rem;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        header h1 {
          margin: 0;
          font-size: 1.1rem;
          display: inline-block;
          margin-right: 1.5rem;
        }
        .progress {
          display: inline-block;
          color: var(--muted);
          font-size: 0.85rem;
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
          padding: 50px 0.5rem 0.5rem;
          max-width: 100%;
          margin: 0;
        }
        .group-card {
          padding: 0;
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
          gap: 0.75rem;
          margin-bottom: 2rem;
        }
        .image-card {
          background: var(--surface-alt);
          padding: 0.5rem;
          border: 3px solid transparent;
          border-radius: 8px;
          cursor: pointer;
          position: relative;
        }
        .image-card.ai-pick {
          border-color: var(--success);
          background: rgba(81, 207, 102, 0.08);
        }
        .image-card.user-override {
          border-color: var(--warning);
          background: rgba(255, 212, 59, 0.08);
        }
        .image-card img {
          width: 100%;
          height: auto;
          margin-bottom: 0.5rem;
          display: block;
        }
        .image-card .image-header {
          display: flex;
          gap: 0.4rem;
          margin-bottom: 0.3rem;
          min-height: 22px;
          align-items: center;
        }
        .image-card .stage-badge {
          background: rgba(79, 157, 255, 0.85);
          color: white;
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 0.7rem;
          font-weight: 600;
        }
        .image-card .ai-pick-badge {
          background: #51cf66;
          color: black;
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.05em;
        }
        .image-card .crop-badge {
          background: #ff6b6b;
          color: white;
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 0.7rem;
          font-weight: 700;
        }
        .image-card .image-container {
          position: relative;
          display: inline-block;
          line-height: 0;
          width: 100%;
        }
        .image-card .crop-overlay {
          position: absolute;
          border: 2px solid #51cf66;
          pointer-events: none;
          z-index: 10;
        }
        .image-card .filename {
          font-size: 0.75rem;
          color: var(--muted);
          margin-bottom: 0.3rem;
          word-break: break-all;
        }
        .image-card .image-actions {
          display: flex;
          gap: 0.3rem;
          flex-direction: row;
        }
        .image-card .img-btn {
          flex: 1;
          padding: 0.5rem 0.3rem;
          border: none;
          border-radius: 4px;
          font-weight: 600;
          font-size: 0.75rem;
          cursor: pointer;
          white-space: nowrap;
        }
        .img-btn-approve {
          background: var(--success);
          color: black;
        }
        .img-btn-crop {
          background: #ffa500;
          color: black;
        }
        .img-btn-reject {
          background: var(--danger);
          color: white;
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
          flex: 1;
          min-width: 150px;
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
        <h1>🤖 AI Reviewer</h1>
        <span class="progress">
          Group <strong id="current-num">{{ current + 1 }}</strong>/{{ total }} •
          Reviewed: <strong id="reviewed-count">0</strong> •
          Approved: <strong id="approved-count">0</strong>
        </span>
        <div id="status"></div>
      </header>
      <main>
        <div class="group-card">
          <div class="images-row">
            {% for img in group.images %}
            <div class="image-card {% if loop.index0 == recommendation.selected_index %}ai-pick{% endif %}"
                 data-index="{{ loop.index0 }}"
                 onclick="selectImage({{ loop.index0 }})">
              <div class="image-header">
                <span class="stage-badge">{{ img.name.split('_')[2] if '_' in img.name else 'stage?' }}</span>
                {% if loop.index0 == recommendation.selected_index %}
                <span class="ai-pick-badge">AI PICK</span>
                {% endif %}
                {% if loop.index0 == recommendation.selected_index and recommendation.crop_needed %}
                <span class="crop-badge">CROP</span>
                {% endif %}
              </div>
              <div class="image-container">
                <img src="/image/{{ group.group_id }}/{{ loop.index0 }}" 
                     alt="{{ img.name }}"
                     loading="lazy"
                     id="img-{{ loop.index0 }}"
                     onload="drawCropOverlay({{ loop.index0 }}, {{ recommendation.selected_index }}, {{ recommendation.crop_needed|lower }}, {{ recommendation.crop_coords|tojson if recommendation.crop_coords else 'null' }})">
                <div id="crop-overlay-{{ loop.index0 }}" class="crop-overlay" style="display: none;"></div>
              </div>
              <div class="filename">{{ img.name }}</div>
              <div class="image-actions">
                <button class="img-btn img-btn-approve" onclick="event.stopPropagation(); submitImageDecision('approve', {{ loop.index0 }})">
                  ✓ Approve [{{ loop.index }}]
                </button>
                <button class="img-btn img-btn-crop" onclick="event.stopPropagation(); submitImageDecision('crop', {{ loop.index0 }})">
                  ✂️ Crop
                </button>
                <button class="img-btn img-btn-reject" onclick="event.stopPropagation(); rejectAll()">
                  ✗ Reject All
                </button>
              </div>
            </div>
            {% endfor %}
          </div>
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
          const reviewedEl = document.getElementById('reviewed-count');
          const approvedEl = document.getElementById('approved-count');
          
          if (reviewedEl) reviewedEl.textContent = stats.reviewed;
          if (approvedEl) approvedEl.textContent = stats.approved;
        }
        
        function drawCropOverlay(imageIndex, selectedIndex, cropNeeded, cropCoords) {
          // Only draw crop overlay on the AI-selected image if crop is needed
          if (imageIndex !== selectedIndex || !cropNeeded || !cropCoords) {
            return;
          }
          
          const img = document.getElementById('img-' + imageIndex);
          const overlay = document.getElementById('crop-overlay-' + imageIndex);
          
          if (!img || !overlay) return;
          
          // Wait for image to fully load
          if (!img.complete || img.naturalWidth === 0) {
            img.addEventListener('load', function() {
              drawCropOverlay(imageIndex, selectedIndex, cropNeeded, cropCoords);
            });
            return;
          }
          
          // Get displayed image dimensions
          const displayWidth = img.clientWidth;
          const displayHeight = img.clientHeight;
          
          // Crop coords are normalized [0, 1]
          const [x1_norm, y1_norm, x2_norm, y2_norm] = cropCoords;
          
          // Convert to pixel coordinates
          const x1 = x1_norm * displayWidth;
          const y1 = y1_norm * displayHeight;
          const width = (x2_norm - x1_norm) * displayWidth;
          const height = (y2_norm - y1_norm) * displayHeight;
          
          // Position overlay
          overlay.style.left = x1 + 'px';
          overlay.style.top = y1 + 'px';
          overlay.style.width = width + 'px';
          overlay.style.height = height + 'px';
          overlay.style.display = 'block';
          
          console.log('Crop overlay drawn:', {imageIndex, displayWidth, displayHeight, x1, y1, width, height});
        }
        
        function setStatus(message, type = '') {
          const statusEl = document.getElementById('status');
          if (statusEl) {
            statusEl.textContent = message;
            statusEl.className = type;
            // Don't auto-clear - let it stay until next action
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
          console.log('[submitDecision] START:', {action, selectedIndex, groupId, aiRecommendation});
          
          const decision = {
            group_id: groupId,
            action: action,
            selected_index: selectedIndex,
            ai_index: aiRecommendation
          };
          
          console.log('[submitDecision] Sending decision:', decision);
          
          try {
            console.log('[submitDecision] Fetching /submit...');
            const response = await fetch('/submit', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(decision)
            });
            
            console.log('[submitDecision] Response status:', response.status);
            
            const result = await response.json();
            console.log('[submitDecision] Result:', result);
            
            if (result.status === 'ok') {
              console.log('[submitDecision] Success! Updating stats...');
              stats.reviewed++;
              if (action === 'approve') stats.approved++;
              if (action === 'override') stats.overridden++;
              if (action === 'reject') stats.rejected++;
              if (action === 'skip') stats.skipped++;
              updateStats();
              
              console.log('[submitDecision] Navigating to /next...');
              window.location.href = '/next';
            } else {
              console.error('[submitDecision] Server returned error:', result.message);
              setStatus(result.message || 'Error submitting decision', 'error');
            }
          } catch (error) {
            console.error('[submitDecision] EXCEPTION:', error);
            console.error('[submitDecision] Error stack:', error.stack);
            setStatus('Network error: ' + error.message, 'error');
          }
        }
        
        function approve() {
          submitDecision('approve', aiRecommendation);
        }
        
        function submitImageDecision(action, imageIndex) {
          console.log('[submitImageDecision] Called:', {action, imageIndex});
          
          // Direct action on a specific image
          if (action === 'approve') {
            console.log('[submitImageDecision] Calling submitDecision for APPROVE');
            // Approve this specific image
            submitDecision('approve', imageIndex);
          } else if (action === 'crop') {
            console.log('[submitImageDecision] Calling submitDecision for MANUAL_CROP');
            // Send to crop directory
            submitDecision('manual_crop', imageIndex);
          } else if (action === 'reject_image') {
            console.log('[submitImageDecision] Asking confirmation for REJECT');
            // Ask for confirmation before rejecting just this image
            if (confirm(`Reject this image? It will be moved to delete staging.`)) {
              console.log('[submitImageDecision] Confirmed, calling submitDecision for REJECT_SINGLE');
              submitDecision('reject_single', imageIndex);
            } else {
              console.log('[submitImageDecision] User cancelled reject');
            }
          }
        }
        
        function manualCrop() {
          // Get currently selected image (or AI recommendation if none selected)
          const selectedIndex = userSelection !== null ? userSelection : aiRecommendation;
          
          // Visual feedback: highlight selected card with crop indicator
          const cards = document.querySelectorAll('.image-card');
          if (selectedIndex >= 0 && selectedIndex < cards.length) {
            cards[selectedIndex].style.borderColor = '#ff6b6b';
            cards[selectedIndex].style.background = 'rgba(255, 107, 107, 0.15)';
          }
          
          // Update button to show confirmation
          const btn = event.target;
          btn.textContent = '⏳ Queuing for crop...';
          btn.style.background = '#ff6b6b';
          
          submitDecision('manual_crop', selectedIndex);
        }
        
        function rejectAll() {
          // No confirmation - just reject immediately
          submitDecision('reject', null);
        }
        
        function navigate(direction) {
          const url = direction > 0 ? '/next' : '/prev';
          window.location.href = url;
        }
        
        // Keyboard shortcuts
        let pageReady = false;
        window.addEventListener('load', () => {
          pageReady = true;
        });
        
        document.addEventListener('keydown', function(e) {
          // Ignore if page not ready or typing in input
          if (!pageReady || e.target.matches('input, textarea')) return;
          
          // Ignore if modifier keys pressed (Cmd+R, Ctrl+R, etc.)
          if (e.metaKey || e.ctrlKey || e.altKey) return;
          
          const key = e.key.toLowerCase();
          
          switch(key) {
            case '1':
            case '2':
            case '3':
            case '4':
              e.preventDefault();
              const index = parseInt(key) - 1;
              if (index < totalImages) {
                selectImage(index);
              }
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
              <h1>🎉 Batch Complete!</h1>
              <p>You've completed reviewing this batch.</p>
              <p><a href="/stats" style="color: #4f9dff;">View Summary Statistics</a></p>
              <p><a href="/next-batch" style="color: #4f9dff;">Load Next Batch</a></p>
            </body>
            </html>
            """
        
        group = groups[current]
        
        # Get AI models from config
        ranker = app.config.get("RANKER_MODEL")
        crop_model = app.config.get("CROP_MODEL")
        clip_info = app.config.get("CLIP_INFO")
        
        decision_data = load_or_create_decision_file(group, ranker, crop_model, clip_info)
        
        return render_template_string(
            page_template,
            group=group,
            recommendation=decision_data["ai_recommendation"],
            total=len(groups),
            current=current
        )
    
    @app.route("/submit", methods=["POST"])
    def submit():
        """Handle decision submission and perform file operations."""
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
            
            # Load decision file (get AI models from config)
            ranker = app.config.get("RANKER_MODEL")
            crop_model = app.config.get("CROP_MODEL")
            clip_info = app.config.get("CLIP_INFO")
            
            decision_data = load_or_create_decision_file(group, ranker, crop_model, clip_info)
            
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
            
            # Perform file operations based on action
            tracker = app.config["TRACKER"]
            selected_dir = app.config["SELECTED_DIR"]
            crop_dir = app.config["CROP_DIR"]
            delete_staging_dir = app.config["DELETE_STAGING_DIR"]
            project_id = app.config.get("PROJECT_ID", "unknown")
            
            file_ops_msg = perform_file_operations(
                group, action, selected_index, None,  # crop_coords=None for now
                tracker, selected_dir, crop_dir, delete_staging_dir, project_id
            )
            
            # Build response message
            if action == "approve":
                msg = f"✓ Approved: {group.images[selected_index].name}\n{file_ops_msg}"
            elif action == "override":
                msg = f"⚡ Override: Selected {group.images[selected_index].name}\n{file_ops_msg}"
            elif action == "manual_crop":
                msg = f"✂️ Manual crop: {group.images[selected_index].name}\n{file_ops_msg}"
            elif action == "reject":
                msg = f"✗ Rejected all images\n{file_ops_msg}"
            elif action == "reject_single":
                msg = f"✗ Rejected: {group.images[selected_index].name}\n{file_ops_msg}"
            else:
                msg = f"Decision recorded\n{file_ops_msg}"
            
            return jsonify({"status": "ok", "message": msg})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/next")
    def next_group():
        """Navigate to next group."""
        current = app.config["CURRENT_INDEX"]
        groups = app.config["GROUPS"]
        app.config["CURRENT_INDEX"] = min(current + 1, len(groups) - 1)
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
        description="Review image groups with AI assistance (Ranker v3 + Crop Proposer v2)"
    )
    parser.add_argument("directory", type=str, help="Directory containing images to review")
    parser.add_argument("--host", default="127.0.0.1", help="Host for web server")
    parser.add_argument("--port", type=int, default=8081, help="Port for web server")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of groups per batch (default: 20)")
    args = parser.parse_args()
    
    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists():
        print(f"[!] Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    
    # Find project root and create necessary directories
    project_root = find_project_root(directory)
    print(f"[*] Project root: {project_root}")
    
    selected_dir = project_root / "selected"
    crop_dir = project_root / "crop"
    delete_staging_dir = project_root / "delete_staging"
    models_dir = project_root / "data" / "ai_data" / "models"
    
    # Ensure directories exist
    selected_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)
    delete_staging_dir.mkdir(exist_ok=True)
    
    print(f"[*] Selected directory: {selected_dir}")
    print(f"[*] Crop directory: {crop_dir}")
    print(f"[*] Delete staging directory: {delete_staging_dir}")
    
    # Load AI models
    print(f"\n[*] Loading AI models from {models_dir}...")
    ranker_model, crop_model, clip_info = load_ai_models(models_dir)
    
    if ranker_model is not None:
        print("[✓] AI models loaded - using Ranker v3 + Crop Proposer v2")
    else:
        print("[!] AI models not available - using rule-based fallback")
    
    # Initialize FileTracker
    tracker = FileTracker("ai_assisted_reviewer")
    
    # Scan and group images
    print(f"\n[*] Scanning {directory}...")
    images = scan_images(directory)
    print(f"[*] Found {len(images)} images")
    
    print(f"[*] Grouping images...")
    groups = group_images_by_timestamp(images)
    print(f"[*] Found {len(groups)} groups")
    
    if not groups:
        print("[!] No image groups found. Check directory and file naming.", file=sys.stderr)
        sys.exit(1)
    
    # Calculate batches
    total_batches = (len(groups) + args.batch_size - 1) // args.batch_size
    print(f"[*] Processing in batches of {args.batch_size} ({total_batches} total batches)")
    
    # Pre-compute AI recommendations for first batch to show crop stats
    if ranker_model is not None and crop_model is not None and clip_info is not None:
        print(f"\n[*] Computing AI recommendations for first batch...")
        first_batch = groups[:args.batch_size]
        crop_needed_count = 0
        no_crop_count = 0
        
        for group in first_batch:
            try:
                rec = get_ai_recommendation(group, ranker_model, crop_model, clip_info)
                if rec.get("crop_needed", False):
                    crop_needed_count += 1
                else:
                    no_crop_count += 1
            except Exception as e:
                print(f"   Warning: Failed to get recommendation for {group.group_id}: {e}")
                no_crop_count += 1
        
        print(f"[✓] First batch stats:")
        print(f"    • {crop_needed_count} images need cropping ({crop_needed_count*100//len(first_batch)}%)")
        print(f"    • {no_crop_count} images need no crop ({no_crop_count*100//len(first_batch)}%)")
    
    
    # Build and run Flask app
    app = build_app(groups, directory, tracker, selected_dir, crop_dir, delete_staging_dir,
                   ranker_model, crop_model, clip_info, batch_size=args.batch_size)
    print(f"\n[*] Starting reviewer on http://{args.host}:{args.port}")
    print(f"[*] Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

