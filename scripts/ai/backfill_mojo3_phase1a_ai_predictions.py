#!/usr/bin/env python3
"""
Phase 1A: Create Temp Database with AI Predictions

This script:
1. Groups original images from /Volumes/T7Shield/Eros/original/mojo3
2. Loads trained AI models (ranker + crop proposer)
3. Runs AI predictions on every group
4. Stores AI predictions in temp database (user fields NULL)

Output: mojo3_backfill_temp.db (with AI predictions, no user data yet)

Usage:
    python scripts/ai/backfill_mojo3_phase1a_ai_predictions.py
"""

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

# Paths
ORIGINAL_DIR = Path("/Volumes/T7Shield/Eros/original/mojo3")
TEMP_DB = (
    WORKSPACE / "data" / "training" / "ai_training_decisions" / "mojo3_backfill_temp.db"
)
PROJECT_ID = "mojo3"

# Model paths
RANKER_MODEL_PATH = WORKSPACE / "data" / "ai_data" / "models" / "ranker_v3_w10.pt"
CROP_MODEL_PATH = WORKSPACE / "data" / "ai_data" / "models" / "crop_proposer_v2.pt"
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
    """MLP that predicts crop box from image embedding + dimensions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(514, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_embeddings_cache() -> tuple[dict, dict]:
    """Load embeddings cache and create filename lookup."""
    cache = {}
    filename_cache = {}

    if not EMBEDDINGS_CACHE.exists():
        return cache, filename_cache

    with open(EMBEDDINGS_CACHE) as f:
        for line in f:
            data = json.loads(line)
            img_path = data["image_path"]
            emb_file = Path(data["embedding_file"])

            if emb_file.exists():
                emb = np.load(emb_file)
                cache[img_path] = emb

                # Also cache by filename
                filename = Path(img_path).name
                if filename not in filename_cache:
                    filename_cache[filename] = []
                filename_cache[filename].append((img_path, emb))

    return cache, filename_cache


def load_ai_models():
    """Load trained AI models."""
    if not RANKER_MODEL_PATH.exists():
        return None, None

    if not CROP_MODEL_PATH.exists():
        return None, None

    # Load ranker
    ranker = RankingModel().to(device)
    ranker.load_state_dict(torch.load(RANKER_MODEL_PATH, map_location=device))
    ranker.eval()

    # Load crop proposer
    cropper = CropProposer().to(device)
    cropper.load_state_dict(torch.load(CROP_MODEL_PATH, map_location=device))
    cropper.eval()

    return ranker, cropper


def load_clip_model():
    """Load CLIP model for computing embeddings on the fly."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def get_embedding(
    image_path: Path,
    cache: dict,
    filename_cache: dict,
    clip_model=None,
    clip_preprocess=None,
) -> np.ndarray | None:
    """Get embedding for image, from cache or compute on the fly."""
    # Try exact path match
    if str(image_path) in cache:
        return cache[str(image_path)]

    # Try filename match (for images in different directories)
    filename = image_path.name
    if filename in filename_cache:
        # Return first match (assumes same filename = same image)
        return filename_cache[filename][0][1]

    # Compute on the fly if CLIP model available
    if clip_model is not None and clip_preprocess is not None:
        try:
            image = Image.open(image_path).convert("RGB")
            image = clip_preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = clip_model.encode_image(image)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().squeeze()
        except Exception:
            return None

    return None


def run_ai_predictions(
    group: list[Path],
    ranker,
    cropper,
    cache: dict,
    filename_cache: dict,
    clip_model=None,
    clip_preprocess=None,
) -> tuple[int | None, list[float] | None, float | None]:
    """Run AI models on a group to get selection and crop predictions."""
    # Get embeddings for all images in group
    embeddings = []
    valid_indices = []

    for idx, img_path in enumerate(group):
        emb = get_embedding(
            img_path, cache, filename_cache, clip_model, clip_preprocess
        )
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)

    if not embeddings:
        return None, None, None

    # Run ranker to select best image
    with torch.no_grad():
        emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).to(device)
        scores = ranker(emb_tensor).squeeze()

        if len(embeddings) == 1:
            best_idx = 0
            confidence = float(scores.item())
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


def create_temp_database(db_path: Path):
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
            crop_match BOOLEAN,
            ai_crop_accepted BOOLEAN
        )
    """
    )

    conn.commit()
    conn.close()


def group_original_images(original_dir: Path) -> list[tuple[str, list[Path]]]:
    """Group original images using the grouping logic."""
    # Get all image files
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    all_images = []
    for pattern in patterns:
        all_images.extend(original_dir.glob(pattern))


    # Sort by timestamp and stage
    sorted_images = sort_image_files_by_timestamp_and_stage(all_images)

    # Group consecutive images
    raw_groups = find_consecutive_stage_groups(sorted_images)

    # Add group IDs
    groups_with_ids = []
    for group_images in raw_groups:
        # Generate group_id from first image's timestamp
        first_img = group_images[0]
        timestamp = extract_timestamp_from_filename(first_img.name)
        group_id = f"{PROJECT_ID}_group_{timestamp}"
        groups_with_ids.append((group_id, group_images))


    return groups_with_ids


def process_groups_with_ai(
    groups: list[tuple[str, list[Path]]],
    ranker,
    cropper,
    cache: dict,
    filename_cache: dict,
    clip_model=None,
    clip_preprocess=None,
):
    """Process all groups with AI models."""
    records = []

    stats = {
        "total_groups": len(groups),
        "ai_predictions_success": 0,
        "ai_predictions_failed": 0,
    }


    for _i, (group_id, group_images) in enumerate(tqdm(groups, desc="Processing")):
        # Run AI predictions
        ai_selected_index, ai_crop_coords, ai_confidence = run_ai_predictions(
            group_images,
            ranker,
            cropper,
            cache,
            filename_cache,
            clip_model,
            clip_preprocess,
        )

        if ai_selected_index is not None:
            stats["ai_predictions_success"] += 1
        else:
            stats["ai_predictions_failed"] += 1
            continue

        # Get image dimensions from first image
        try:
            with Image.open(group_images[0]) as img:
                image_width, image_height = img.size
        except Exception:
            continue

        # Create record with AI predictions, user fields NULL
        record = {
            "group_id": group_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "project_id": PROJECT_ID,
            "directory": str(group_images[0].parent),
            "batch_number": None,
            "images": json.dumps([img.name for img in group_images]),
            "ai_selected_index": ai_selected_index,
            "ai_crop_coords": json.dumps(ai_crop_coords) if ai_crop_coords else None,
            "ai_confidence": ai_confidence,
            "user_selected_index": None,  # Will be filled in Phase 1B
            "user_action": None,  # Will be filled in Phase 1B
            "final_crop_coords": None,  # Will be filled in Phase 1B
            "crop_timestamp": None,
            "image_width": image_width,
            "image_height": image_height,
            "selection_match": None,  # Will be calculated in Phase 1B
            "crop_match": None,  # Will be calculated in Phase 1B
            "ai_crop_accepted": None,
        }

        records.append(record)

    return records, stats


def write_records_to_database(records: list[dict], db_path: Path):
    """Write records to temporary database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for record in records:
        cursor.execute(
            """
            INSERT INTO ai_decisions (
                group_id, timestamp, project_id, directory, batch_number,
                images, ai_selected_index, ai_crop_coords, ai_confidence,
                user_selected_index, user_action, final_crop_coords, crop_timestamp,
                image_width, image_height, selection_match, crop_match, ai_crop_accepted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record["group_id"],
                record["timestamp"],
                record["project_id"],
                record["directory"],
                record["batch_number"],
                record["images"],
                record["ai_selected_index"],
                record["ai_crop_coords"],
                record["ai_confidence"],
                record["user_selected_index"],
                record["user_action"],
                record["final_crop_coords"],
                record["crop_timestamp"],
                record["image_width"],
                record["image_height"],
                record["selection_match"],
                record["crop_match"],
                record["ai_crop_accepted"],
            ),
        )

    conn.commit()
    conn.close()



def main():

    # Check directories exist
    if not ORIGINAL_DIR.exists():
        return

    # Load embeddings cache
    cache, filename_cache = load_embeddings_cache()

    # Load AI models
    ranker, cropper = load_ai_models()
    if ranker is None or cropper is None:
        return

    # Load CLIP model for computing missing embeddings
    clip_model, clip_preprocess = load_clip_model()

    # Create temporary database
    if TEMP_DB.exists():
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() != "yes":
            return
        TEMP_DB.unlink()

    TEMP_DB.parent.mkdir(parents=True, exist_ok=True)
    create_temp_database(TEMP_DB)

    # Group original images
    groups = group_original_images(ORIGINAL_DIR)

    # Process groups with AI
    records, _stats = process_groups_with_ai(
        groups, ranker, cropper, cache, filename_cache, clip_model, clip_preprocess
    )

    # Write to database
    write_records_to_database(records, TEMP_DB)

    # Print statistics



if __name__ == "__main__":
    main()
