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

🎨 STYLE GUIDE: See WEB_STYLE_GUIDE.md for UI/UX standards

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
  python scripts/01_ai_assisted_reviewer.py <raw_images_directory>/

FLAGS:
------
  --batch-size N    Number of groups to process per batch (default: 100)
  --host HOST       Host for web server (default: 127.0.0.1)
  --port PORT       Port for web server (default: 8081)


WORKFLOW:
---------
1. Groups images by timestamp (same logic as web image selector)
2. For each group:
   - AI/Rule recommends best image
   - User reviews: Approve (A), Override (1/2/3/4), Manual Crop (C), Reject (R)
3. FILE OPERATIONS executed immediately:
   - Approve/Override → Move to selected/
   - Manual Crop → Move to __crop/ (for manual cropping later)
   - Reject → Move ALL to delete_staging/ (fast deletion staging)
4. Training data logged automatically (selection + crop decisions)
5. Logs decisions to sidecar .decision files (single source of truth)

UNSELECT FUNCTIONALITY: Not implemented - all selections are final

FILE ROUTING:
-------------
| Action       | Selected Image →   | Other Images →    |
|--------------|--------------------|-------------------|
| Approve      | __selected/        | delete_staging/   |
| Override     | __selected/        | delete_staging/   |
| Manual Crop  | __crop/            | delete_staging/   |
| Reject       | __delete_staging/  | delete_staging/   |

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
1/2/3/4 - Accept image AS-IS:
            • If AI crop showing → keeps crop → __crop_auto/
            • If no AI crop → no crop → __selected/
A/S/D/F - Remove AI crop and select → __selected/ (always no crop)
Q/W/E/R - Select with manual crop → __crop/
Enter/Space - Next group
↑ - Previous group

All image selection keys auto-advance to next group for fast workflow!

DIRECTORY STRUCTURE:
--------------------
The script will automatically create and use these directories at the project root:
  __selected/         - Final selected images (ready for next step)
  __crop/             - Images that need manual cropping
  __delete_staging/   - Fast deletion staging (move to Trash later)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# SQLite v3 Training Data (NEW!)
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.ai_training_decisions_v3 import (
    generate_group_id,
    init_decision_db,
    log_ai_decision,
)

# Reuse existing grouping logic - NO reinventing the wheel!
sys.path.insert(0, str(Path(__file__).parent))
from file_tracker import FileTracker
from utils.companion_file_utils import (
    detect_stage,
    extract_datetime_from_filename,
    find_consecutive_stage_groups,
    log_crop_decision,  # NEW: Minimal schema crop logging
    log_selection_only_entry,
    move_file_with_all_companions,
    sort_image_files_by_timestamp_and_stage,
)

# Flask import deferred until needed (after argument parsing)
flask_available = False

try:
    from PIL import Image
except Exception:
    raise

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import open_clip

    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False


# ==============================================================================
# CONFIGURATION (Easy to find and modify!)
# ==============================================================================
DEFAULT_BATCH_SIZE = 100  # Number of groups to process per batch (match web selector)
THUMBNAIL_MAX_DIM = 768  # Thumbnail size for web display
STAGE_NAMES = ("stage1_generated", "stage1.5_face_swapped", "stage2_upscaled")
# ==============================================================================


@dataclass
class ImageGroup:
    """Represents a group of images with same timestamp."""

    group_id: str  # timestamp identifier
    images: List[Path]  # sorted by stage
    directory: Path  # parent directory


# ============================
# AI Model Architecture
# ============================

if TORCH_AVAILABLE:

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
                nn.Linear(64, 1),
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
                nn.Sigmoid(),  # Output normalized [0, 1]
            )

        def forward(self, x):
            return self.net(x)

else:
    # Define dummy classes when torch is not available
    class RankerModel:
        pass

    class CropProposerModel:
        pass


def load_ai_models(
    models_dir: Path,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Load Ranker v3 and Crop Proposer v2 models.

    Returns: (ranker_model, crop_model, clip_model) or (None, None, None) if unavailable
    """
    if not TORCH_AVAILABLE or not CLIP_AVAILABLE:
        return None, None, None

    try:
        # Load CLIP for embeddings
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        clip_model = clip_model.to(device)
        clip_model.eval()

        # Load Ranker v3
        ranker_path = models_dir / "ranker_v3_w10.pt"
        if ranker_path.exists():
            ranker = RankerModel(input_dim=512).to(device)
            ranker.load_state_dict(torch.load(ranker_path, map_location=device))
            ranker.eval()
        else:
            ranker = None

        # Load Crop Proposer v2
        crop_path = models_dir / "crop_proposer_v2.pt"
        if crop_path.exists():
            crop_proposer = CropProposerModel(input_dim=514).to(device)
            crop_proposer.load_state_dict(torch.load(crop_path, map_location=device))
            crop_proposer.eval()
        else:
            crop_proposer = None

        return ranker, crop_proposer, (clip_model, preprocess, device)

    except Exception:
        import traceback

        traceback.print_exc()
        return None, None, None


def get_image_embedding(
    image_path: Path, clip_model, preprocess, device
) -> Optional[torch.Tensor]:
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

    except Exception:
        return None


def get_ai_recommendation(
    group: ImageGroup, ranker_model, crop_model, clip_info
) -> Dict:
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
        for idx, (img, score) in enumerate(zip(group.images, scores, strict=False)):
            img_stage = detect_stage(img.name) or f"img{idx + 1}"
            marker = " ✓" if idx == best_idx else ""
            score_details.append(f"{img_stage}: {score.item():.2f}{marker}")

        reason = f"AI picked {stage} (score: {best_score:.2f}) | " + " • ".join(
            score_details
        )

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
                dims = torch.tensor(
                    [width / 2048.0, height / 2048.0], dtype=torch.float32
                ).to(device)
                crop_input = torch.cat([best_embedding, dims]).unsqueeze(
                    0
                )  # Add batch dim

                # DEBUG: Log input to crop model

                with torch.no_grad():
                    crop_output = crop_model(crop_input).squeeze(0)  # Remove batch dim

                # Extract normalized coordinates
                x1, y1, x2, y2 = crop_output.cpu().numpy()

                # DEBUG: Log crop output
                (x2 - x1) * 100
                (y2 - y1) * 100

                # Check if crop is meaningful (not ~full image)
                crop_area = (x2 - x1) * (y2 - y1)

                if crop_area < 0.95:  # If cropping more than 5%
                    crop_needed = True
                    crop_coords = (float(x1), float(y1), float(x2), float(y2))
                else:
                    pass

            except Exception:
                crop_needed = False
                crop_coords = None

        return {
            "selected_image": best_image.name,
            "selected_index": best_idx,
            "reason": reason,
            "confidence": confidence,
            "crop_needed": crop_needed,
            "crop_coords": crop_coords,
        }

    except Exception:
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
            group_id = first_img.stem.split("_stage")[0]

        result.append(
            ImageGroup(
                group_id=group_id, images=group_paths, directory=first_img.parent
            )
        )

    return result


def print_groups_summary(groups: List[ImageGroup]) -> None:
    """Print summary of image groups (for testing)"""
    triplet_count = 0
    pair_count = 0
    singleton_count = 0

    for group in groups:
        size = len(group.images)
        if size == 3:
            triplet_count += 1
        elif size == 2:
            pair_count += 1
        else:
            singleton_count += 1

    total_groups = len(groups)
    print(
        f"\nTotal: {triplet_count} triplets, {pair_count} pairs, {singleton_count} singletons ({total_groups} groups)"
    )


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
        "crop_coords": None,
    }


def load_or_create_decision_file(
    group: ImageGroup, ranker_model=None, crop_model=None, clip_info=None
) -> Dict:
    """
    Load existing decision or create new one.
    Decision files are stored alongside images with .decision extension.
    """
    decision_path = group.directory / f"{group.group_id}.decision"

    if decision_path.exists():
        with open(decision_path, "r") as f:
            return json.load(f)

    # Create new decision with AI recommendation (or rule-based fallback)
    if ranker_model is not None and clip_info is not None:
        recommendation = get_ai_recommendation(
            group, ranker_model, crop_model, clip_info
        )
    else:
        recommendation = get_rule_based_recommendation(group)

    return {
        "group_id": group.group_id,
        "images": [img.name for img in group.images],
        "ai_recommendation": recommendation,
        "user_decision": None,  # Not reviewed yet
    }


def save_decision_file(group: ImageGroup, decision_data: Dict) -> None:
    """
    Save decision to sidecar .decision file.
    This is the SINGLE SOURCE OF TRUTH for user decisions.
    """
    decision_path = group.directory / f"{group.group_id}.decision"

    with open(decision_path, "w") as f:
        json.dump(decision_data, f, indent=2)


def find_project_root(directory: Path) -> Path:
    """
    Find project root directory by looking for specific markers.
    Falls back to current directory if not found.
    """
    current = directory.resolve()

    # Look for project markers (support legacy and new double-underscore dirs)
    markers = ["scripts", "data", "selected", "crop", "__selected", "__crop"]

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
        return "unknown"

    # Look for active project (finishedAt is null)
    for project_file in project_dir.glob("*.project.json"):
        try:
            with open(project_file, "r") as f:
                data = json.load(f)

                # Active project has no finish date
                if data.get("finishedAt") is None:
                    project_id = data.get("projectId", "unknown")
                    return project_id
        except Exception:
            continue

    return "unknown"


def delete_group_images(
    group: ImageGroup, delete_staging_dir: Path, tracker: FileTracker, reason: str = ""
) -> int:
    """
    Delete all images in a group by moving them to delete_staging.

    Args:
        group: ImageGroup to delete
        delete_staging_dir: Destination directory for deleted files
        tracker: FileTracker for logging
        reason: Reason for deletion (for logging)

    Returns:
        Number of images deleted
    """
    deleted_count = 0
    for img in group.images:
        try:
            moved_files = move_file_with_all_companions(
                img, delete_staging_dir, dry_run=False
            )
            tracker.log_operation(
                "delete",
                source_dir=str(img.parent),
                dest_dir=delete_staging_dir.name,
                file_count=len(moved_files),
                files=moved_files,
                notes=f"{reason} - group {group.group_id}",
            )
            deleted_count += 1
        except Exception:
            pass
    return deleted_count


def perform_file_operations(
    group: ImageGroup,
    action: str,
    selected_index: Optional[int],
    crop_coords: Optional[Tuple[float, float, float, float]],
    tracker: FileTracker,
    selected_dir: Path,
    crop_dir: Path,
    delete_staging_dir: Path,
    project_id: str = "unknown",
) -> Dict[str, any]:
    """
    Execute file operations based on user decision.

    Args:
        project_id: Project identifier (e.g., 'mojo1', 'mojo3') for training data

    Returns: Structured result dict with counts and summary message
        {
          "moved_selected": int,  # 1 if selected image moved to selected/
          "moved_crop": int,      # 1 if selected image moved to crop/
          "deleted_images": int,  # number of images moved to delete staging
          "message": str
        }
    """
    if action == "reject":
        # Delete all images in group - move to delete_staging
        moved_count = 0
        for img_path in group.images:
            try:
                move_file_with_all_companions(
                    img_path, delete_staging_dir, dry_run=False
                )
                moved_count += 1
            except Exception:
                pass

        tracker.log_operation(
            "stage_delete",
            str(group.directory),
            str(delete_staging_dir.name),
            moved_count,
            f"Rejected group {group.group_id}",
            [img.name for img in group.images[:5]],
        )
        return {
            "moved_selected": 0,
            "moved_crop": 0,
            "deleted_images": moved_count,
            "message": f"Rejected: {moved_count} images moved to delete staging",
        }

    if action == "reject_single":
        # Delete just one image - move to delete_staging
        if selected_index is None:
            return {
                "moved_selected": 0,
                "moved_crop": 0,
                "deleted_images": 0,
                "message": "Error: No image selected for rejection",
            }

        selected_image = group.images[selected_index]
        try:
            move_file_with_all_companions(
                selected_image, delete_staging_dir, dry_run=False
            )
            tracker.log_operation(
                "stage_delete",
                str(group.directory),
                str(delete_staging_dir.name),
                1,
                f"Rejected single image from group {group.group_id}",
                [selected_image.name],
            )
            return {
                "moved_selected": 0,
                "moved_crop": 0,
                "deleted_images": 1,
                "message": f"Rejected: {selected_image.name} moved to delete staging",
            }
        except Exception as e:
            return {
                "moved_selected": 0,
                "moved_crop": 0,
                "deleted_images": 0,
                "message": f"Error: {e}",
            }

    if selected_index is None:
        return {
            "moved_selected": 0,
            "moved_crop": 0,
            "deleted_images": 0,
            "message": "Error: No image selected",
        }

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

        # Move other images to delete_staging (consolidated deletion logic)
        for img_path in other_images:
            try:
                moved_files = move_file_with_all_companions(
                    img_path, delete_staging_dir, dry_run=False
                )

                # Log each deletion
                tracker.log_operation(
                    "delete",
                    source_dir=str(group.directory),
                    dest_dir=str(delete_staging_dir.name),
                    file_count=len(moved_files),
                    files=moved_files,  # List of filenames
                    notes=f"Deselected image from group {group.group_id}",
                )
            except Exception:
                import traceback

                traceback.print_exc()

        # Log training data
        negative_paths = other_images
        try:
            # Always log selection
            log_selection_only_entry(
                session_id=f"ai_reviewer_{datetime.utcnow().strftime('%Y%m%d')}",
                set_id=group.group_id,
                chosen_path=str(selected_image),
                negative_paths=[str(p) for p in negative_paths],
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
                        height=height,
                    )
                except Exception:
                    pass
        except Exception:
            pass

        tracker.log_operation(
            "move",
            str(group.directory),
            dest_dir.name,
            1,
            f"Selected image from group {group.group_id}",
            [selected_image.name],
        )
        result = {
            "moved_selected": 1 if dest_dir == selected_dir else 0,
            "moved_crop": 1 if dest_dir == crop_dir else 0,
            "deleted_images": len(other_images),
            "message": f"Moved {selected_image.name} to {dest_label}, {len(other_images)} to delete staging",
        }
        return result

    except Exception as e:
        return {
            "moved_selected": 0,
            "moved_crop": 0,
            "deleted_images": 0,
            "message": f"Error during file operations: {e}",
        }


def detect_artifact(group: ImageGroup) -> Tuple[bool, List[str]]:
    """Detect artifact conditions for a group.
    Rules:
      - Multi-directory: images span multiple parent directories
      - Mismatched stems: base stem before '_stage' differs across images
    Returns (is_artifact, reasons)
    """
    reasons: List[str] = []
    parents = {str(p.parent) for p in group.images}
    if len(parents) > 1:
        reasons.append("multi_directory")

    # Extract base stems by splitting before '_stage'
    def base_stem(name: str) -> str:
        parts = name.split("_stage")
        return parts[0] if parts else Path(name).stem

    stems = {base_stem(p.name) for p in group.images}
    if len(stems) > 1:
        reasons.append("mismatched_stems")
    return (len(reasons) > 0, reasons)


def build_app(
    groups: List[ImageGroup],
    base_dir: Path,
    tracker: FileTracker,
    selected_dir: Path,
    crop_dir: Path,
    delete_staging_dir: Path,
    ranker_model=None,
    crop_model=None,
    clip_info=None,
    batch_size: int = 20,
):
    """Build Flask app for reviewing image groups."""
    if not flask_available:
        raise RuntimeError("Flask not available - cannot build web app")

    app = Flask(__name__)
    app.config["ALL_GROUPS"] = groups  # Full list
    app.config["BATCH_SIZE"] = batch_size
    app.config["CURRENT_BATCH"] = 0

    # Calculate batch (match web_image_selector.py structure)
    total_groups = len(groups)
    batch_start = 0
    batch_end = min(batch_start + batch_size, total_groups)
    current_batch_groups = groups[batch_start:batch_end]

    # Batch info (match web selector structure)
    batch_info = {
        "current_batch": 1,  # 1-indexed
        "total_batches": (total_groups + batch_size - 1) // batch_size,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "batch_size": batch_size,
        "total_groups": total_groups,
        "current_batch_size": len(current_batch_groups),
    }

    app.config["GROUPS"] = current_batch_groups  # Current batch only
    app.config["BATCH_INFO"] = batch_info
    app.config["BASE_DIR"] = base_dir
    app.config["CURRENT_INDEX"] = 0
    app.config["DECISIONS"] = {}  # Track decisions in session
    app.config["TRACKER"] = tracker
    app.config["SELECTED_DIR"] = selected_dir
    app.config["CROP_DIR"] = crop_dir
    app.config["DELETE_STAGING_DIR"] = delete_staging_dir
    app.config["PROJECT_ID"] = get_current_project_id()  # Read from project manifest!
    # Configure log archives directory for batch JSON logger
    try:
        project_root = find_project_root(base_dir)
    except Exception:
        project_root = Path(__file__).parent.parent
    log_archives_dir = project_root / "data" / "log_archives"
    log_archives_dir.mkdir(parents=True, exist_ok=True)
    app.config["LOG_ARCHIVES_DIR"] = log_archives_dir

    # Initialize SQLite database for training data (NEW v3!)
    project_id = app.config["PROJECT_ID"]
    try:
        db_path = init_decision_db(project_id)
        app.config["DB_PATH"] = db_path
    except Exception:
        app.config["DB_PATH"] = None
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
          gap: 0.4rem;  /* Reduced from 0.75rem for tighter spacing with 4 images */
          margin-bottom: 2rem;
        }
        .image-card {
          background: var(--surface-alt);
          padding: 0.3rem;  /* Reduced from 0.5rem for tighter spacing */
          border: 3px solid transparent;
          border-radius: 8px;
          cursor: pointer;
          position: relative;
        }
        .image-card.ai-pick {
          /* No visual styling - .ai-pick is just a marker for JS to find AI's choice */
          /* Visual state is controlled by .selected / .crop-selected / .delete-hint */
        }
        .image-card.user-override {
          border-color: var(--warning);
          background: rgba(255, 212, 59, 0.08);
        }
        .image-card img {
          width: 100%;
          max-height: 60vh;  /* Limit height to 60% of viewport (matches web_image_selector) */
          height: auto;
          object-fit: contain;  /* Maintain aspect ratio, prevent distortion */
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
        
        /* Batch processing visual states (match web selector) */
        .image-card.selected {
          border: 3px solid var(--success);
          background: rgba(81, 207, 102, 0.1);
        }
        .image-card.crop-selected {
          border-color: var(--warning);
          background: rgba(255, 212, 59, 0.1);
        }
        .image-card.delete-hint {
          /* No visual styling - delete is the default/neutral state */
          /* Only selected/crop images get colored borders */
        }
        
        /* Batch actions at bottom */
        .batch-actions {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          background: var(--bg);
          border-top: 1px solid rgba(255,255,255,0.1);
          padding: 1rem;
          text-align: center;
          box-shadow: 0 -2px 8px rgba(0,0,0,0.3);
          z-index: 100;
        }
        .btn-primary {
          background: var(--accent);
          color: white;
          border: none;
          padding: 1rem 2rem;
          font-size: 1.1rem;
          font-weight: 600;
          border-radius: 8px;
          cursor: pointer;
        }
        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        /* Group sections */
        .group {
          margin-bottom: 3rem;
          padding-bottom: 2rem;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
      </style>
    </head>
    <body>
      <header>
        <h1>🤖 AI Reviewer</h1>
        <span class="progress">
          Batch {{ batch_info.current_batch }}/{{ batch_info.total_batches }}: <strong id="batch-count">{{ batch_info.current_batch_size }}</strong> groups •
          Selected: <strong id="summary-selected">0</strong> •
          Delete: <strong id="summary-delete">0</strong>
        </span>
        <div id="status"></div>
      </header>
      <main>
        {% for group in groups %}
        <section class="group" data-group-id="{{ group.group_id }}" id="group-{{ group.group_id }}">
          <div class="images-row">
            {% set ai_rec = ai_recommendations.get(group.group_id, {}) %}
            {% for img in group.images %}
            <div class="image-card {% if loop.index0 == ai_rec.get('selected_index') %}ai-pick{% endif %}"
                 data-image-index="{{ loop.index0 }}"
                 data-group-id="{{ group.group_id }}"
                 {% if loop.index0 == ai_rec.get('selected_index') and ai_rec.get('crop_coords') %}
                 data-ai-crop-coords="{{ ai_rec.get('crop_coords')|tojson }}"
                 {% endif %}>
              <div class="image-container">
                <img src="/image/{{ group.group_id }}/{{ loop.index0 }}" 
                     alt="{{ img.name }}"
                     loading="lazy"
                     id="img-{{ group.group_id }}-{{ loop.index0 }}"
                     onload="{% if loop.index0 == ai_rec.get('selected_index') %}drawCropOverlay('{{ group.group_id }}', {{ loop.index0 }}, {{ ai_rec.get('crop_needed', False)|lower }}, {{ ai_rec.get('crop_coords')|tojson if ai_rec.get('crop_coords') else 'null' }}){% endif %}">
                <div id="crop-overlay-{{ group.group_id }}-{{ loop.index0 }}" class="crop-overlay" style="display: none;"></div>
              </div>
              <div class="filename">{{ img.name }}</div>
              <div class="image-actions">
                {% if loop.index0 == ai_rec.get('selected_index') and ai_rec.get('crop_coords') %}
                {# AI-selected image with crop: Show BOTH toggle AND crop buttons #}
                <button class="img-btn img-btn-crop" 
                        id="toggle-crop-{{ group.group_id }}-{{ loop.index0 }}"
                        onclick="toggleAICrop('{{ group.group_id }}', {{ loop.index0 }})">
                  🚫 Remove Crop
                </button>
                <button class="img-btn img-btn-crop" onclick="selectImageWithCrop({{ loop.index0 }}, '{{ group.group_id }}')">
                  ✂️ [{% if loop.index0 == 0 %}Q{% elif loop.index0 == 1 %}W{% elif loop.index0 == 2 %}E{% else %}R{% endif %}]
                </button>
                {% else %}
                {# Regular image: Show crop button with hotkey #}
                <button class="img-btn img-btn-crop" onclick="selectImageWithCrop({{ loop.index0 }}, '{{ group.group_id }}')">
                  ✂️ [{% if loop.index0 == 0 %}Q{% elif loop.index0 == 1 %}W{% elif loop.index0 == 2 %}E{% else %}R{% endif %}]
                </button>
                {% endif %}
              </div>
            </div>
            {% endfor %}
          </div>
        </section>
        {% endfor %}
        
        <div class="batch-actions">
          <button id="process-batch" class="btn btn-primary" disabled>
            Finalize selections
          </button>
        </div>
      </main>
      
      <script>
        // BATCH PROCESSING MODEL (clone of web_image_selector.py)
        const groups = Array.from(document.querySelectorAll('section.group'));
        console.log('[init] AI reviewer loaded. groups=', groups.length);
        
        const summarySelected = document.getElementById('summary-selected');
        const summaryDelete = document.getElementById('summary-delete');
        const processBatchButton = document.getElementById('process-batch');
        const batchCount = document.getElementById('batch-count');
        const statusBox = document.getElementById('status');
        
        // Queue decisions in memory (don't execute immediately!)
        let groupStates = {}; // { groupId: { selectedImage: idx, crop: bool, aiCropAccepted: bool } }
        
        function setStatus(message, type = '') {
          if (statusBox) {
            statusBox.textContent = message;
            statusBox.className = type;
          }
        }
        
        function drawCropOverlay(groupId, imageIndex, cropNeeded, cropCoords) {
          // Draw crop overlay on AI-selected image if crop is needed
          if (!cropNeeded || !cropCoords) {
            return;
          }
          
          const img = document.getElementById('img-' + groupId + '-' + imageIndex);
          const overlay = document.getElementById('crop-overlay-' + groupId + '-' + imageIndex);
          
          if (!img || !overlay) return;
          
          // Wait for image to fully load
          if (!img.complete || img.naturalWidth === 0) {
            img.addEventListener('load', function() {
              drawCropOverlay(groupId, imageIndex, cropNeeded, cropCoords);
            });
            return;
          }
          
          // Calculate actual displayed image dimensions (accounting for object-fit: contain)
          const naturalWidth = img.naturalWidth;
          const naturalHeight = img.naturalHeight;
          const containerWidth = img.clientWidth;
          const containerHeight = img.clientHeight;
          
          const naturalRatio = naturalWidth / naturalHeight;
          const containerRatio = containerWidth / containerHeight;
          
          let displayWidth, displayHeight, offsetX, offsetY;
          
          if (naturalRatio > containerRatio) {
            // Image is wider - letterboxed top/bottom
            displayWidth = containerWidth;
            displayHeight = containerWidth / naturalRatio;
            offsetX = 0;
            offsetY = (containerHeight - displayHeight) / 2;
          } else {
            // Image is taller - letterboxed left/right
            displayHeight = containerHeight;
            displayWidth = containerHeight * naturalRatio;
            offsetX = (containerWidth - displayWidth) / 2;
            offsetY = 0;
          }
          
          // Crop coords are normalized [0, 1]
          const [x1_norm, y1_norm, x2_norm, y2_norm] = cropCoords;
          
          // Convert to pixel coordinates on the displayed image
          const x1 = x1_norm * displayWidth + offsetX;
          const y1 = y1_norm * displayHeight + offsetY;
          const width = (x2_norm - x1_norm) * displayWidth;
          const height = (y2_norm - y1_norm) * displayHeight;
          
          // Position overlay
          overlay.style.left = x1 + 'px';
          overlay.style.top = y1 + 'px';
          overlay.style.width = width + 'px';
          overlay.style.height = height + 'px';
          overlay.style.display = 'block';
          
          console.log('Crop overlay drawn:', {groupId, imageIndex, displayWidth, displayHeight, x1, y1, width, height});
        }
        
        // Toggle AI crop on/off for AI-selected image
        function toggleAICrop(groupId, imageIndex) {
          console.log('[toggleAICrop]', groupId, imageIndex);
          
          const overlay = document.getElementById('crop-overlay-' + groupId + '-' + imageIndex);
          const button = document.getElementById('toggle-crop-' + groupId + '-' + imageIndex);
          const imageCard = document.querySelector(`section.group[data-group-id="${groupId}"] .image-card[data-image-index="${imageIndex}"]`);
          
          if (!overlay || !button || !imageCard) return;
          
          // Get AI crop coords from data attribute
          const aiCropCoordsAttr = imageCard.getAttribute('data-ai-crop-coords');
          if (!aiCropCoordsAttr) {
            console.log('[toggleAICrop] No AI crop coords found');
            return;
          }
          
          const aiCropCoords = JSON.parse(aiCropCoordsAttr);
          
          // Toggle overlay visibility
          if (overlay.style.display === 'none') {
            // Show crop overlay (ADD CROP)
            const img = document.getElementById('img-' + groupId + '-' + imageIndex);
            if (img && img.complete) {
              drawCropOverlay(groupId, imageIndex, true, aiCropCoords);
            }
            button.textContent = '🚫 Remove Crop';
            button.classList.remove('img-btn-approve');
            button.classList.add('img-btn-crop');
            console.log('[toggleAICrop] Crop ENABLED');
            // Mark AI crop accepted and set selection to this image (suggestion path)
            groupStates[groupId] = { selectedImage: imageIndex, crop: false, aiCropAccepted: true };
            updateVisualState();
            updateSummary();
          } else {
            // Hide crop overlay (REMOVE CROP)
            overlay.style.display = 'none';
            button.textContent = '✅ Add Crop';
            button.classList.remove('img-btn-crop');
            button.classList.add('img-btn-approve');
            console.log('[toggleAICrop] Crop DISABLED');
            // Disable AI crop acceptance (keep selection if any)
            const state = groupStates[groupId];
            if (state) {
              groupStates[groupId] = { selectedImage: state.selectedImage, crop: false, aiCropAccepted: false };
              updateVisualState();
              updateSummary();
            }
          }
        }
        
        function updateSummary() {
          let keptFiles = 0;
          let deleteFiles = 0;
          
          groups.forEach((group) => {
            const groupId = group.dataset.groupId;
            const state = groupStates[groupId];
            const imageCount = group.querySelectorAll('.image-card').length;
            
            // selected: state ? (state.selectedImage !== undefined) : false
            if (state && state.selectedImage !== undefined) {
              // One image kept, others deleted
              keptFiles += 1;
              deleteFiles += Math.max(0, imageCount - 1);
            } else {
              // No state: all images deleted by default
              deleteFiles += imageCount;
            }
          });
          
          summarySelected.textContent = keptFiles;
          summaryDelete.textContent = Math.max(0, deleteFiles);
        }
        
        function updateVisualState() {
          groups.forEach((group) => {
            const groupId = group.dataset.groupId;
            const state = groupStates[groupId];
            
            // Clear all visual states (including ai-pick!)
            group.querySelectorAll('.image-card').forEach(card => {
              card.classList.remove('selected', 'delete-hint', 'crop-selected');
            });
            
            if (state && state.selectedImage !== undefined) {
              // Show selected image
              const selectedCard = group.querySelectorAll('.image-card')[state.selectedImage];
              if (selectedCard) {
                selectedCard.classList.add('selected');
                // Indicate crop intent when flagged
                if (state.crop) {
                  selectedCard.classList.add('crop-selected');
                }
              }
              
              // Show delete hint on other images (red border)
              group.querySelectorAll('.image-card').forEach((card, idx) => {
                if (idx !== state.selectedImage) {
                  card.classList.add('delete-hint');
                }
              });
            } else {
              // No selection: all will be deleted (red borders)
              group.querySelectorAll('.image-card').forEach(card => {
                card.classList.add('delete-hint');
              });
            }
          });
        }
        
        // Select image - checks if AI crop overlay is visible and accepts it if so (1234 keys)
        function selectImage(imageIndex, groupId) {
          console.log('[selectImage]', imageIndex, groupId);
          const group = document.querySelector(`section.group[data-group-id="${groupId}"]`);
          if (!group) return;
          
          const imageCount = group.querySelectorAll('.image-card').length;
          if (imageIndex >= imageCount) return;
          
          // Check if AI crop overlay is visible for this image
          const overlay = document.getElementById('crop-overlay-' + groupId + '-' + imageIndex);
          const hasVisibleAICrop = overlay && overlay.style.display !== 'none';
          
          // BUTTON TOGGLE: If same image already selected, deselect it
          const currentState = groupStates[groupId];
          if (currentState && currentState.selectedImage === imageIndex && !currentState.crop) {
            delete groupStates[groupId]; // Deselect (back to delete)
            console.log('[selectImage] deselect', groupId, imageIndex);
            updateVisualState();
            updateSummary();
            // NO auto-advance on deselect
          } else {
            // Update state - select image WITH or WITHOUT AI crop depending on visibility
            groupStates[groupId] = { 
              selectedImage: imageIndex, 
              crop: false, 
              aiCropAccepted: hasVisibleAICrop 
            };
            console.log('[selectImage] set', groupStates[groupId], 'aiCrop:', hasVisibleAICrop);
            updateVisualState();
            updateSummary();
            
            // AUTO-ADVANCE to next group after selection!
            scrollToNextGroup(group);
          }
        }
        
        // Select image WITHOUT crop (ASDF keys) - explicitly removes AI crop if showing
        function selectImageWithoutCrop(imageIndex, groupId) {
          console.log('[selectImageWithoutCrop]', imageIndex, groupId);
          const group = document.querySelector(`section.group[data-group-id="${groupId}"]`);
          if (!group) return;
          
          const imageCount = group.querySelectorAll('.image-card').length;
          if (imageIndex >= imageCount) return;
          
          // Hide AI crop overlay if it exists for this image
          const overlay = document.getElementById('crop-overlay-' + groupId + '-' + imageIndex);
          if (overlay) {
            overlay.style.display = 'none';
          }
          
          // Update toggle button if it exists
          const toggleBtn = document.getElementById('toggle-crop-' + groupId + '-' + imageIndex);
          if (toggleBtn) {
            toggleBtn.textContent = '✅ Add Crop';
            toggleBtn.classList.remove('img-btn-crop');
            toggleBtn.classList.add('img-btn-approve');
          }
          
          // BUTTON TOGGLE: If same image already selected without crop, deselect it
          const currentState = groupStates[groupId];
          if (currentState && currentState.selectedImage === imageIndex && !currentState.crop && !currentState.aiCropAccepted) {
            delete groupStates[groupId]; // Deselect (back to delete)
            console.log('[selectImageWithoutCrop] deselect', groupId, imageIndex);
            updateVisualState();
            updateSummary();
            // NO auto-advance on deselect
          } else {
            // Update state - select image WITHOUT crop (explicitly no AI crop)
            groupStates[groupId] = { selectedImage: imageIndex, crop: false, aiCropAccepted: false };
            console.log('[selectImageWithoutCrop] set (no crop)', groupStates[groupId]);
            updateVisualState();
            updateSummary();
            
            // AUTO-ADVANCE to next group after selection!
            scrollToNextGroup(group);
          }
        }
        
        // Select image WITH crop - EXACT COPY from web_image_selector.py + AUTO-ADVANCE
        function selectImageWithCrop(imageIndex, groupId) {
          console.log('[selectImageWithCrop]', imageIndex, groupId);
          const group = document.querySelector(`section.group[data-group-id="${groupId}"]`);
          if (!group) return;
          
          const imageCount = group.querySelectorAll('.image-card').length;
          if (imageIndex >= imageCount) return;
          
          // TOGGLE: If same image already selected with crop, deselect it
          const currentState = groupStates[groupId];
          if (currentState && currentState.selectedImage === imageIndex && currentState.crop) {
            delete groupStates[groupId]; // Deselect (back to delete)
            console.log('[selectImageWithCrop] deselect', groupId, imageIndex);
            updateVisualState();
            updateSummary();
            // NO auto-advance on deselect
          } else {
            // Update state - select image WITH crop
            groupStates[groupId] = { selectedImage: imageIndex, crop: true, aiCropAccepted: false };
            console.log('[selectImageWithCrop] set', groupStates[groupId]);
            updateVisualState();
            updateSummary();
            
            // AUTO-ADVANCE to next group after selection!
            scrollToNextGroup(group);
          }
        }
        
        // Helper function to scroll to next group (EXACT COPY from web_image_selector.py)
        function scrollToGroupEl(groupEl) {
          try {
            const headerH = getHeaderHeight();
            const rect = groupEl.getBoundingClientRect();
            const top = rect.top + window.scrollY - headerH - 12; // small margin
            window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
          } catch (e) {
            groupEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
        
        function getHeaderHeight() {
          const header = document.querySelector('header');
          return header ? header.offsetHeight : 60; // fallback to 60px
        }
        
        function scrollToNextGroup(currentGroup) {
          const allGroups = Array.from(document.querySelectorAll('section.group'));
          const currentIndex = allGroups.indexOf(currentGroup);
          if (currentIndex >= 0 && currentIndex < allGroups.length - 1) {
            const nextGroup = allGroups[currentIndex + 1];
            scrollToGroupEl(nextGroup);
            console.log('[auto-advance] Scrolled to next group');
          } else {
            console.log('[auto-advance] Already at last group');
          }
        }
        
        // Handle image click - EXACT COPY from web_image_selector.py
        function handleImageClick(imageCard) {
          const group = imageCard.closest('section.group');
          const groupId = group.dataset.groupId;
          const imageIndex = parseInt(imageCard.dataset.imageIndex);
          const currentState = groupStates[groupId];
          console.log('[handleImageClick]', { groupId, imageIndex, currentState });
          
          if (!currentState || currentState.selectedImage === undefined) {
            // First click on any image: select it (no crop)
            groupStates[groupId] = { selectedImage: imageIndex, crop: false, aiCropAccepted: false };
          } else if (currentState.selectedImage === imageIndex) {
            // UNSELECT: Clicking selected image deselects it (back to delete)
            delete groupStates[groupId];
          } else {
            // Clicking different image: switch selection to that image
            groupStates[groupId] = { selectedImage: imageIndex, crop: false, aiCropAccepted: false };
          }
          
          updateVisualState();
          updateSummary();
        }
        
        // Keyboard shortcuts (match web_image_selector.py exactly!)
        document.addEventListener('keydown', function(e) {
          // Ignore if typing in input
          if (e.target.matches('input, textarea')) return;
          
          // Ignore if modifier keys pressed (Cmd+R, Ctrl+R, etc.)
          if (e.metaKey || e.ctrlKey || e.altKey) return;
          
          const key = e.key.toLowerCase();
          
          // Get current visible group (simple: first group in viewport)
          let currentGroupId = null;
          for (let group of groups) {
            const rect = group.getBoundingClientRect();
            if (rect.top >= 0 && rect.top < window.innerHeight) {
              currentGroupId = group.dataset.groupId;
              break;
            }
          }
          
          if (!currentGroupId) return;
          
          // Match web_image_selector.py hotkeys exactly!
          switch(key) {
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
            case '4':
              e.preventDefault();
              selectImage(3, currentGroupId);
              break;
            case 'a':
              e.preventDefault();
              selectImageWithoutCrop(0, currentGroupId);
              break;
            case 's':
              e.preventDefault();
              selectImageWithoutCrop(1, currentGroupId);
              break;
            case 'd':
              e.preventDefault();
              selectImageWithoutCrop(2, currentGroupId);
              break;
            case 'f':
              e.preventDefault();
              selectImageWithoutCrop(3, currentGroupId);
              break;
            case 'q':
              e.preventDefault();
              selectImageWithCrop(0, currentGroupId);
              break;
            case 'w':
              e.preventDefault();
              selectImageWithCrop(1, currentGroupId);
              break;
            case 'e':
              e.preventDefault();
              selectImageWithCrop(2, currentGroupId);
              break;
            case 'r':
              e.preventDefault();
              selectImageWithCrop(3, currentGroupId);
              break;
            case 'enter':
            case ' ': // Spacebar
              e.preventDefault();
              // Scroll to next group
              const currentIndex = groups.findIndex(g => g.dataset.groupId === currentGroupId);
              if (currentIndex >= 0 && currentIndex < groups.length - 1) {
                groups[currentIndex + 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
              break;
            case 'arrowup':  // ArrowUp navigation
              e.preventDefault();
              // Scroll to previous group
              const prevIndex = groups.findIndex(g => g.dataset.groupId === currentGroupId);
              if (prevIndex > 0) {
                groups[prevIndex - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
              break;
          }
        });
        
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
        
        // Finalize selections button
        processBatchButton.addEventListener('click', async () => {
          // Allow processing even with no selections (means delete all unselected groups)
          // if (Object.keys(groupStates).length === 0) {
          //   setStatus('No groups selected in current batch to process', 'error');
          //   return;
          // }
          
          // No confirmation popup - just process immediately!
          processBatchButton.disabled = true;
          setStatus('Processing batch…');
          
          // Convert groupStates to selections array
          const selections = Object.keys(groupStates).map(groupId => ({
            groupId: groupId,
            selectedImage: groupStates[groupId].selectedImage,
            crop: groupStates[groupId].crop || false,
            aiCropAccepted: groupStates[groupId].aiCropAccepted || false
          }));
          
          try {
            const response = await fetch('/process-batch', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ selections })
            });
            
            const result = await response.json();
            
            if (result.status === 'ok') {
              setStatus(result.message || 'Batch processed!', 'success');
              
              // Check if there are remaining groups in current batch
              if (result.remaining > 0) {
                // Reload to show remaining groups in batch
                setTimeout(() => {
                  window.location.reload();
                }, 1000);
              } else {
                // Batch complete - auto-load next batch if available
                setStatus('Batch complete! Loading next batch...', 'success');
                setTimeout(() => {
                  // Try to load next batch, or show completion if no more batches
                  window.location.href = '/next-batch';
                }, 1000);
              }
            } else {
              setStatus(result.message || 'Error processing batch', 'error');
            }
          } catch (error) {
            setStatus('Network error: ' + error.message, 'error');
          }
        });
        
        // Initialize
        // DON'T call updateVisualState() here - wait for AI pre-selection first!
        // updateVisualState();
        // updateSummary();
        
        // Add click handlers to all image cards (like web selector)
        document.querySelectorAll('.image-card').forEach(card => {
          card.addEventListener('click', function() {
            handleImageClick(this);
          });
        });
        
        // AI RECOMMENDATIONS: Show crop overlay but DON'T pre-select
        // Erik: "I would like it to have the crop, but it is deselected by default"
        // Crop overlays are drawn via template's onload callback (line ~1111)
        // No pre-selection needed - user makes the choice via hotkeys
        window.addEventListener('load', function() {
          // Just initialize the visual state and summary with no selections
          console.log('[AI reviewer] Page loaded. Showing crop overlays without pre-selection.');
          updateVisualState();
          updateSummary();
        });
      </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        """Show all groups in current batch (batch processing mode)."""
        groups = app.config["GROUPS"]
        batch_info = app.config.get("BATCH_INFO", {})

        if len(groups) == 0:
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

        # Get AI recommendations for each group
        ranker = app.config.get("RANKER_MODEL")
        crop_model = app.config.get("CROP_MODEL")
        clip_info = app.config.get("CLIP_INFO")

        # Build AI recommendations dict for template
        ai_recommendations = {}
        for group in groups:
            try:
                rec = get_ai_recommendation(group, ranker, crop_model, clip_info)
                ai_recommendations[group.group_id] = rec
            except Exception:
                # Fallback recommendation
                ai_recommendations[group.group_id] = {
                    "selected_index": 0,
                    "selected_image": group.images[0].name if group.images else "",
                    "reason": "Error getting AI recommendation",
                    "confidence": 0.0,
                    "crop_needed": False,
                    "crop_coords": None,
                }

        # Render ALL groups in batch with AI recommendations
        return render_template_string(
            page_template,
            groups=groups,
            batch_info=batch_info,
            ai_recommendations=ai_recommendations,
        )

    @app.route("/process-batch", methods=["POST"])
    def process_batch():
        """Process all queued selections (clone of web_image_selector.py logic)"""
        try:
            data = request.get_json()
            selections = data.get("selections", [])

            # Allow empty selections - means delete all groups
            # if not selections:
            #     return jsonify({"status": "error", "message": "No selections to process"}), 400

            tracker = app.config["TRACKER"]
            selected_dir = app.config["SELECTED_DIR"]
            crop_dir = app.config["CROP_DIR"]
            delete_staging_dir = app.config["DELETE_STAGING_DIR"]
            project_id = app.config.get("PROJECT_ID", "unknown")
            groups = app.config["GROUPS"]
            db_path = app.config.get("DB_PATH")
            batch_info = app.config.get("BATCH_INFO", {})
            batch_number = batch_info.get("current_batch", 1)

            # Get AI models for recommendations
            ranker = app.config.get("RANKER_MODEL")
            crop_model = app.config.get("CROP_MODEL")
            clip_info = app.config.get("CLIP_INFO")

            kept_count = 0
            crop_count = 0
            deleted_count = 0
            per_group_results: Dict[str, Dict[str, Any]] = {}

            for idx, selection in enumerate(selections):
                group_id = selection.get("groupId")
                selected_idx = selection.get("selectedImage")
                crop = selection.get("crop", False)
                ai_crop_accepted = selection.get("aiCropAccepted", False)

                # Find group
                group = next((g for g in groups if g.group_id == group_id), None)
                if not group:
                    continue

                # Handle explicit "delete all" (matches web_image_selector.py line 1669)
                if selected_idx is None:
                    num_deleted = delete_group_images(
                        group,
                        delete_staging_dir,
                        tracker,
                        reason="User explicitly deleted all",
                    )
                    deleted_count += num_deleted
                    continue  # Skip to next selection

                # Get AI recommendation for this group
                ai_rec = None
                ai_selected_idx = None
                ai_crop_coords = None
                ai_confidence = None

                if ranker is not None:
                    try:
                        ai_rec = get_ai_recommendation(
                            group, ranker, crop_model, clip_info
                        )
                        ai_selected_idx = ai_rec.get("selected_index", 0)
                        ai_confidence = ai_rec.get("confidence", 0.0)
                        if ai_rec.get("crop_needed") and ai_rec.get("crop_coords"):
                            ai_crop_coords = ai_rec["crop_coords"]  # Already normalized
                    except Exception:
                        pass

                # Determine user action (Phase 3: Two-Action Crop Flow - suggestion path only)
                # If user accepts AI crop but does not request manual crop, we will route to crop_auto later via sidecar flag
                user_action = "crop" if crop else "approve"

                # Generate unique group ID for database
                timestamp = (
                    datetime.utcnow()
                    .isoformat()
                    .replace("-", "")
                    .replace(":", "")
                    .replace(".", "")
                )
                db_group_id = generate_group_id(
                    project_id, timestamp, batch_number, idx
                )

                # Log to SQLite (NEW v3!)
                if db_path:
                    try:
                        # Get image dimensions (use first image as reference)
                        selected_image = group.images[selected_idx]
                        try:
                            from PIL import Image

                            with Image.open(selected_image) as img:
                                image_width, image_height = img.size
                        except Exception:
                            image_width, image_height = 2048, 2048  # Fallback

                        log_ai_decision(
                            db_path=db_path,
                            group_id=db_group_id,
                            project_id=project_id,
                            images=[img.name for img in group.images],
                            ai_selected_index=(
                                ai_selected_idx if ai_selected_idx is not None else 0
                            ),
                            user_selected_index=selected_idx,
                            user_action=user_action,
                            image_width=image_width,
                            image_height=image_height,
                            ai_crop_coords=ai_crop_coords,
                            ai_confidence=ai_confidence,
                            directory=str(group.images[0].parent),
                            batch_number=batch_number,
                        )

                        # Create .decision sidecar file for Desktop Multi-Crop
                        # Create .decision sidecar for crop destinations
                        if crop or ai_crop_accepted:
                            # manual crop → crop/ ; AI crop suggestion → crop_auto/
                            target_dir = (
                                crop_dir
                                if crop
                                else (
                                    app.config.get("CROP_AUTO_DIR")
                                    or (app.config["BASE_DIR"] / "__crop_auto")
                                )
                            )
                            Path(target_dir).mkdir(parents=True, exist_ok=True)
                            decision_file = (
                                Path(target_dir) / f"{selected_image.stem}.decision"
                            )
                            decision_data = {
                                "group_id": db_group_id,
                                "project_id": project_id,
                                "needs_crop": True,
                                "ai_route": (
                                    "suggestion"
                                    if ai_crop_accepted and ai_crop_coords
                                    else "manual"
                                ),
                                "ai_crop_coords": (
                                    ai_crop_coords
                                    if (ai_crop_accepted and ai_crop_coords)
                                    else None
                                ),
                            }
                            with open(decision_file, "w") as f:
                                json.dump(decision_data, f, indent=2)
                    except Exception:
                        pass

                # Perform file operations
                try:
                    # Artifact detection prior to file ops
                    is_artifact, reasons = detect_artifact(group)
                    if is_artifact:
                        pass
                    # Include artifact flags in tracker notes via log_operation extra? We attach in JSONL below; avoid altering FileTracker schema.
                    ops_result = perform_file_operations(
                        group,
                        "manual_crop" if crop else "approve",
                        selected_idx,
                        None,
                        tracker,
                        selected_dir,
                        crop_dir,
                        delete_staging_dir,
                        project_id,
                    )
                    # Accumulate exact counts
                    kept_count += int(ops_result.get("moved_selected", 0)) + int(
                        ops_result.get("moved_crop", 0)
                    )
                    crop_count += int(ops_result.get("moved_crop", 0))
                    deleted_count += int(ops_result.get("deleted_images", 0))
                    per_group_results[group_id] = ops_result
                except Exception:
                    continue

            # Handle unselected groups (ALL images go to delete_staging)
            selected_group_ids = [s.get("groupId") for s in selections]
            unselected_groups = [
                g for g in groups if g.group_id not in selected_group_ids
            ]

            for group in unselected_groups:
                try:
                    num_deleted = delete_group_images(
                        group, delete_staging_dir, tracker, reason="No selection made"
                    )
                    deleted_count += num_deleted
                except Exception:
                    pass

            # Remove ALL processed groups (selected + unselected) from current batch
            processed_group_ids = selected_group_ids + [
                g.group_id for g in unselected_groups
            ]
            remaining_groups = [
                g for g in groups if g.group_id not in processed_group_ids
            ]
            app.config["GROUPS"] = remaining_groups

            # Update batch info
            batch_info = app.config.get("BATCH_INFO", {})
            batch_info["current_batch_size"] = len(remaining_groups)

            # Lightweight JSON line logger to data/log_archives/
            try:
                logs_dir: Path = app.config.get("LOG_ARCHIVES_DIR")
                if logs_dir:
                    log_path = logs_dir / "reviewer_batch_summaries.jsonl"
                    summary_record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "project_id": project_id,
                        "batch_number": batch_number,
                        "kept": kept_count,
                        "crop": crop_count,
                        "deleted": deleted_count,
                        "selected_group_ids": selected_group_ids,
                        "unselected_group_ids": [g.group_id for g in unselected_groups],
                        # Include artifact candidates seen during this batch for quick visibility
                        "artifact_candidates": [
                            {"group_id": g.group_id, "reasons": detect_artifact(g)[1]}
                            for g in groups
                            if any(detect_artifact(g))
                        ],
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(summary_record) + "\n")
            except Exception:
                pass

            message = f"Batch processed — kept {kept_count}, sent {crop_count} to crop/, deleted {deleted_count}."
            return jsonify(
                {
                    "status": "ok",
                    "message": message,
                    "kept": kept_count,
                    "crop": crop_count,
                    "deleted": deleted_count,
                    "remaining": len(remaining_groups),
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/submit", methods=["POST"])
    def submit():
        """Handle decision submission and perform file operations."""
        try:
            data = request.get_json()
            group_id = data.get("group_id")
            action = data.get("action")
            selected_index = data.get("selected_index")
            data.get("ai_index")

            # Find group
            groups = app.config["GROUPS"]
            group = next((g for g in groups if g.group_id == group_id), None)

            if not group:
                return jsonify({"status": "error", "message": "Group not found"}), 404

            # Load decision file (get AI models from config)
            ranker = app.config.get("RANKER_MODEL")
            crop_model = app.config.get("CROP_MODEL")
            clip_info = app.config.get("CLIP_INFO")

            decision_data = load_or_create_decision_file(
                group, ranker, crop_model, clip_info
            )

            # Update with user decision
            decision_data["user_decision"] = {
                "action": action,
                "selected_image": (
                    group.images[selected_index].name
                    if selected_index is not None
                    else None
                ),
                "selected_index": selected_index,
                "timestamp": datetime.utcnow().isoformat() + "Z",
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

            file_ops_result = perform_file_operations(
                group,
                action,
                selected_index,
                None,  # crop_coords=None for now
                tracker,
                selected_dir,
                crop_dir,
                delete_staging_dir,
                project_id,
            )

            # Build response message
            if action == "approve":
                msg = f"✓ Approved: {group.images[selected_index].name}\n{file_ops_result.get('message', '')}"
            elif action == "override":
                msg = f"⚡ Override: Selected {group.images[selected_index].name}\n{file_ops_result.get('message', '')}"
            elif action == "manual_crop":
                msg = f"✂️ Manual crop: {group.images[selected_index].name}\n{file_ops_result.get('message', '')}"
            elif action == "reject":
                msg = f"✗ Rejected all images\n{file_ops_result.get('message', '')}"
            elif action == "reject_single":
                msg = f"✗ Rejected: {group.images[selected_index].name}\n{file_ops_result.get('message', '')}"
            else:
                msg = f"Decision recorded\n{file_ops_result.get('message', '')}"

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
                img.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                return Response(buf.read(), mimetype="image/png")
        except Exception as e:
            return f"Error loading image: {e}", 500

    @app.route("/stats")
    def stats():
        """Show summary statistics."""
        decisions = app.config["DECISIONS"]

        approved = sum(
            1
            for d in decisions.values()
            if d.get("user_decision", {}).get("action") == "approve"
        )
        overridden = sum(
            1
            for d in decisions.values()
            if d.get("user_decision", {}).get("action") == "override"
        )
        rejected = sum(
            1
            for d in decisions.values()
            if d.get("user_decision", {}).get("action") == "reject"
        )
        skipped = sum(
            1
            for d in decisions.values()
            if d.get("user_decision", {}).get("action") == "skip"
        )

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
            <p><strong>Reviewed:</strong> {reviewed} ({reviewed * 100 // total if total else 0}%)</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p><strong>✓ Approved:</strong> {approved}</p>
            <p><strong>⚡ Overridden:</strong> {overridden}</p>
            <p><strong>✗ Rejected:</strong> {rejected}</p>
            <p><strong>⊙ Skipped:</strong> {skipped}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p><strong>AI Agreement Rate:</strong> {approved * 100 // (approved + overridden) if (approved + overridden) else 0}%</p>
          </div>
          <p><a href="/" style="color: #4f9dff;">← Back to Review</a></p>
        </body>
        </html>
        """

    @app.route("/next-batch")
    def next_batch():
        """Load next batch of groups."""
        all_groups = app.config.get("ALL_GROUPS", [])
        batch_size = app.config.get("BATCH_SIZE", 100)
        current_batch = app.config.get("CURRENT_BATCH", 0)

        # Calculate next batch
        next_batch_num = current_batch + 1
        batch_start = next_batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(all_groups))

        if batch_start >= len(all_groups):
            # No more batches - show completion
            return """
            <html>
            <head><title>All Batches Complete!</title></head>
            <body style="background: #101014; color: #f8f9ff; font-family: sans-serif; 
                         text-align: center; padding: 4rem;">
              <h1>🎉 All Batches Complete!</h1>
              <p>You've reviewed all groups!</p>
              <p><a href="/stats" style="color: #4f9dff;">View Final Statistics</a></p>
            </body>
            </html>
            """

        # Load next batch
        next_batch_groups = all_groups[batch_start:batch_end]
        app.config["GROUPS"] = next_batch_groups
        app.config["CURRENT_BATCH"] = next_batch_num

        # Update batch info
        batch_info = {
            "current_batch": next_batch_num + 1,  # 1-indexed for display
            "total_batches": (len(all_groups) + batch_size - 1) // batch_size,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "batch_size": batch_size,
            "total_groups": len(all_groups),
            "current_batch_size": len(next_batch_groups),
        }
        app.config["BATCH_INFO"] = batch_info

        # Redirect to main page to show new batch
        from flask import redirect

        return redirect("/")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review image groups with AI assistance (Ranker v3 + Crop Proposer v2)"
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing images to review"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for web server")
    parser.add_argument("--port", type=int, default=8081, help="Port for web server")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,  # DEFAULT_BATCH_SIZE = 100
        help="Number of groups per batch (default: 100)",
    )
    parser.add_argument(
        "--print-triplets",
        action="store_true",
        help="Print grouping information and exit (for testing)",
    )
    args = parser.parse_args()

    # Import Flask only if needed (not for print-triplets mode)
    if not args.print_triplets:
        try:
            pass
        except Exception:
            raise

    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists():
        sys.exit(1)

    # Find project root and create necessary directories
    project_root = find_project_root(directory)

    # Use centralized standard paths (double-underscore directories)
    try:
        from utils.standard_paths import (
            get_crop_dir,
            get_delete_staging_dir,
            get_selected_dir,
        )

        selected_dir = get_selected_dir()
        crop_dir = get_crop_dir()
        delete_staging_dir = get_delete_staging_dir()
    except Exception:
        # Fallback to legacy names if helper unavailable
        selected_dir = project_root / "__selected"
        crop_dir = project_root / "__crop"
        delete_staging_dir = project_root / "__delete_staging"
    models_dir = project_root / "data" / "ai_data" / "models"

    # Ensure directories exist
    selected_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)
    delete_staging_dir.mkdir(exist_ok=True)

    # Load AI models
    ranker_model, crop_model, clip_info = load_ai_models(models_dir)

    if ranker_model is not None:
        pass
    else:
        pass

    # Initialize FileTracker
    tracker = FileTracker("ai_assisted_reviewer")

    # Scan and group images
    images = scan_images(directory)

    groups = group_images_by_timestamp(images)

    if not groups:
        sys.exit(1)

    # Handle print-triplets mode (for testing)
    if args.print_triplets:
        print_groups_summary(groups)
        return

    # Calculate batches
    (len(groups) + args.batch_size - 1) // args.batch_size

    # Pre-compute AI recommendations for first batch to show crop stats
    if ranker_model is not None and crop_model is not None and clip_info is not None:
        first_batch = groups[: args.batch_size]
        crop_needed_count = 0
        no_crop_count = 0

        for group in first_batch:
            try:
                rec = get_ai_recommendation(group, ranker_model, crop_model, clip_info)
                if rec.get("crop_needed", False):
                    crop_needed_count += 1
                else:
                    no_crop_count += 1
            except Exception:
                no_crop_count += 1

    # Build and run Flask app
    app = build_app(
        groups,
        directory,
        tracker,
        selected_dir,
        crop_dir,
        delete_staging_dir,
        ranker_model,
        crop_model,
        clip_info,
        batch_size=args.batch_size,
    )

    url = f"http://{args.host}:{args.port}"

    # Auto-open browser after short delay (give server time to start)
    def open_browser():
        import time

        time.sleep(1.5)
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    # Launch browser in background thread
    import threading

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
