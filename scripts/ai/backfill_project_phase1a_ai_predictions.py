#!/usr/bin/env python3
r"""
Phase 1A: Create Temp Database with AI Predictions

This script:
1. Groups original images from specified directory
2. Loads trained AI models (ranker + crop proposer)
3. Runs AI predictions on every group
4. Stores AI predictions in temp database (user fields NULL)

Output: {project_id}_backfill_temp.db (with AI predictions, no user data yet)

Usage:
    python scripts/ai/backfill_project_phase1a_ai_predictions.py \\
        --project-id mojo3 \\
        --original-dir /Volumes/T7Shield/Eros/original/mojo3 \\
        --output-db data/training/ai_training_decisions/mojo3_backfill_temp.db
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

# Add project root to path
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

# Import grouping utilities
from scripts.utils.companion_file_utils import (
    extract_timestamp_from_filename,
    find_consecutive_stage_groups,
    sort_image_files_by_timestamp_and_stage,
)

# Default model paths (can be overridden via CLI args)
DEFAULT_RANKER_MODEL = "ranker_v4.pt"  # Latest ranker
DEFAULT_CROP_MODEL = "crop_proposer_v3.pt"  # Latest crop proposer (v3)
MODELS_DIR = WORKSPACE / "data" / "ai_data" / "models"
EMBEDDINGS_CACHE = WORKSPACE / "data" / "ai_data" / "cache" / "processed_images.jsonl"
EMBEDDINGS_DIR = WORKSPACE / "data" / "ai_data" / "embeddings"

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Model architectures (must match training)
class RankingModel(nn.Module):
    """MLP that scores images: 512 → 256 → 64 → 1"""

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


class CropProposer(nn.Module):
    """MLP that predicts crop coords: (512 + 2 dims) → 256 → 128 → 4"""

    def __init__(self, input_dim=514):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Load pre-computed CLIP embeddings from cache."""
    embeddings = {}
    if EMBEDDINGS_CACHE.exists():
        with open(EMBEDDINGS_CACHE) as f:
            for line in f:
                entry = json.loads(line)
                filename = Path(entry["filename"]).name
                embeddings[filename] = np.array(entry["embedding"])
    return embeddings


def compute_clip_embedding(image_path: Path, clip_model, preprocess) -> np.ndarray:
    """Compute CLIP embedding for an image on-the-fly."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    except Exception:
        return None


def group_original_images(original_dir: Path) -> list[tuple[str, list[Path]]]:
    """
    Group original images using the standard grouping logic.

    Returns:
        List of (group_id, [image_paths]) tuples
    """
    # Find all images recursively
    image_extensions = {".png", ".jpg", ".jpeg"}
    all_images = []
    for ext in image_extensions:
        all_images.extend(original_dir.rglob(f"*{ext}"))

    if not all_images:
        return []


    # Sort by timestamp and stage
    sorted_images = sort_image_files_by_timestamp_and_stage(all_images)

    # Group into consecutive stage groups
    raw_groups = find_consecutive_stage_groups(sorted_images)

    # Generate group_ids from timestamps
    groups_with_ids = []
    for group_images in raw_groups:
        first_img = group_images[0]
        timestamp = extract_timestamp_from_filename(first_img.name)
        if not timestamp:
            continue

        group_id = f"{first_img.parent.name}_group_{timestamp}"
        groups_with_ids.append((group_id, group_images))

    return groups_with_ids


def run_ai_predictions(
    group: list[Path],
    ranker: nn.Module,
    cropper: nn.Module,
    embedding_cache: dict[str, np.ndarray],
    clip_model,
    preprocess,
) -> tuple[int, list[float] | None, float]:
    """
    Run AI models on a group of images.

    Returns:
        (ai_selected_index, ai_crop_coords, ai_confidence)
    """
    # Get or compute embeddings
    embeddings = []
    valid_indices = []

    for i, img_path in enumerate(group):
        filename = img_path.name

        # Try cache first
        if filename in embedding_cache:
            emb = embedding_cache[filename]
        else:
            # Compute on-the-fly
            emb = compute_clip_embedding(img_path, clip_model, preprocess)
            if emb is None:
                continue

        embeddings.append(emb)
        valid_indices.append(i)

    if not embeddings:
        return (0, None, 0.0)

    # Run ranker
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).to(
        device
    )

    with torch.no_grad():
        scores = ranker(embeddings_tensor)

        # Handle single image case
        if len(scores.shape) == 0:
            best_idx = 0
            confidence = 1.0
        else:
            best_idx = int(scores.argmax().item())
            confidence = float(torch.softmax(scores, dim=0)[best_idx].item())

        # Map back to original group index
        ai_selected_index = valid_indices[best_idx]
        best_embedding = embeddings[best_idx]

    # Run crop proposer on selected image
    try:
        # Get image dimensions
        with Image.open(group[ai_selected_index]) as img:
            width, height = img.size

        # Normalize dimensions to [0, 1]
        norm_width = width / 10000.0
        norm_height = height / 10000.0

        # Concatenate embedding + dimensions
        features = np.concatenate([best_embedding, [norm_width, norm_height]])
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Predict crop
        with torch.no_grad():
            crop_coords = cropper(features_tensor).squeeze().cpu().numpy().tolist()

        return ai_selected_index, crop_coords, confidence

    except Exception:
        return ai_selected_index, None, confidence


def create_temp_database(db_path: Path, project_id: str):
    """Create temporary database with schema."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_decisions (
            group_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            project_id TEXT NOT NULL,
            directory TEXT,
            batch_number INTEGER,
            images TEXT NOT NULL,
            ai_selected_index INTEGER,
            ai_crop_coords TEXT,
            ai_confidence REAL,
            user_selected_index INTEGER,
            user_action TEXT,
            final_crop_coords TEXT,
            crop_timestamp TEXT,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            selection_match BOOLEAN,
            crop_match BOOLEAN
        )
    """
    )

    conn.commit()
    conn.close()


def insert_ai_prediction(
    db_path: Path,
    group_id: str,
    project_id: str,
    directory: str,
    images: list[Path],
    ai_selected_index: int,
    ai_crop_coords: list[float] | None,
    ai_confidence: float,
):
    """Insert AI prediction into temp database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get image dimensions from selected image
    with Image.open(images[ai_selected_index]) as img:
        width, height = img.size

    # Convert images to relative filenames
    image_filenames = [img.name for img in images]

    cursor.execute(
        """
        INSERT INTO ai_decisions (
            group_id, timestamp, project_id, directory, batch_number,
            images, ai_selected_index, ai_crop_coords, ai_confidence,
            user_selected_index, user_action, final_crop_coords,
            crop_timestamp, image_width, image_height,
            selection_match, crop_match
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            group_id,
            datetime.utcnow().isoformat() + "Z",
            project_id,
            str(directory),
            None,  # batch_number
            json.dumps(image_filenames),
            ai_selected_index,
            json.dumps(ai_crop_coords) if ai_crop_coords else None,
            ai_confidence,
            None,  # user_selected_index (filled in Phase 1B)
            None,  # user_action (filled in Phase 1B)
            None,  # final_crop_coords (filled in Phase 1B)
            None,  # crop_timestamp
            width,
            height,
            None,  # selection_match (filled in Phase 1B)
            None,  # crop_match (filled in Phase 1B)
        ),
    )

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1A: Generate AI predictions for backfill"
    )
    parser.add_argument("--project-id", required=True, help="Project ID (e.g., mojo3)")
    parser.add_argument(
        "--original-dir", required=True, help="Directory containing original images"
    )
    parser.add_argument(
        "--output-db", required=True, help="Path to output temp database"
    )
    parser.add_argument(
        "--ranker-model",
        default=DEFAULT_RANKER_MODEL,
        help=f"Ranker model filename (default: {DEFAULT_RANKER_MODEL})",
    )
    parser.add_argument(
        "--crop-model",
        default=DEFAULT_CROP_MODEL,
        help=f"Crop proposer model filename (default: {DEFAULT_CROP_MODEL})",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    output_db = Path(args.output_db)
    project_id = args.project_id

    # Resolve model paths
    ranker_model_path = MODELS_DIR / args.ranker_model
    crop_model_path = MODELS_DIR / args.crop_model


    # Verify paths
    if not original_dir.exists():
        return

    if not ranker_model_path.exists():
        return

    if not crop_model_path.exists():
        return

    # Load models
    ranker = RankingModel().to(device)
    ranker.load_state_dict(torch.load(ranker_model_path, map_location=device))
    ranker.eval()

    cropper = CropProposer().to(device)
    cropper.load_state_dict(torch.load(crop_model_path, map_location=device))
    cropper.eval()

    # Load CLIP model for on-the-fly embeddings
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Load embedding cache
    embedding_cache = load_embedding_cache()

    # Group original images
    groups = group_original_images(original_dir)

    if not groups:
        return


    # Create temp database
    if output_db.exists():
        response = input("Overwrite? (y/n): ")
        if response.lower() != "y":
            return
        output_db.unlink()

    output_db.parent.mkdir(parents=True, exist_ok=True)
    create_temp_database(output_db, project_id)

    # Process all groups

    for group_id, group_images in tqdm(groups, desc="Processing"):
        ai_selected_index, ai_crop_coords, ai_confidence = run_ai_predictions(
            group_images, ranker, cropper, embedding_cache, clip_model, preprocess
        )

        insert_ai_prediction(
            output_db,
            group_id,
            project_id,
            str(group_images[0].parent),
            group_images,
            ai_selected_index,
            ai_crop_coords,
            ai_confidence,
        )



if __name__ == "__main__":
    main()
