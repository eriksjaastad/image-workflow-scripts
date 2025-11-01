#!/usr/bin/env python3
"""
Batch AI Predictions - Add predictions to existing databases (via temp + merge)

This script:
1. Reads image groups from existing databases
2. Extracts zips to get actual images
3. Runs AI predictions (ranker v4 + crop proposer v3)
4. Creates temp databases with ONLY AI predictions
5. You review, then merge using Phase 2 script

Safety: Never touches existing databases until you run the merge step.
"""

import argparse
import json
import shutil
import sqlite3
import sys
import time
import zipfile
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

# Paths
ORIGINAL_DIR = Path("/Volumes/T7Shield/Eros/original")
DB_DIR = WORKSPACE / "data" / "training" / "ai_training_decisions"
TMP_DIR = Path("/tmp/ai_predictions_batch")
LOG_FILE = WORKSPACE / "data" / "ai_data" / "batch_predictions_log.jsonl"

# Models
DEFAULT_RANKER_MODEL = "ranker_v4.pt"
DEFAULT_CROP_MODEL = "crop_proposer_v3.pt"
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
    """MLP that predicts crop coords: (512 + 2 dims) → 512 → 256 → 128 → 4

    Architecture for v3 model (deeper network than v2).
    """

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


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Load pre-computed CLIP embeddings from cache."""
    embeddings = {}
    if EMBEDDINGS_CACHE.exists():
        with open(EMBEDDINGS_CACHE) as f:
            for line in f:
                entry = json.loads(line)
                filename = Path(entry["image_path"]).name
                emb_file = EMBEDDINGS_DIR / entry["embedding_file"]
                if emb_file.exists():
                    embeddings[filename] = np.load(emb_file)
    return embeddings


def get_embedding(
    image_path: Path,
    cache: dict,
    clip_model,
    preprocess,
) -> np.ndarray | None:
    """Get CLIP embedding from cache or compute on-the-fly."""
    filename = image_path.name

    # Check cache first
    if filename in cache:
        return cache[filename]

    # Compute on-the-fly
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding
    except Exception:
        return None


def run_ai_predictions(
    image_paths: list[Path],
    ranker,
    cropper,
    cache: dict,
    clip_model,
    preprocess,
) -> tuple[int | None, list[float] | None, float | None]:
    """Run AI models on a group to get selection and crop predictions."""
    # Get embeddings for all images in group
    embeddings = []
    valid_indices = []

    for idx, img_path in enumerate(image_paths):
        emb = get_embedding(img_path, cache, clip_model, preprocess)
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
        with Image.open(image_paths[ai_selected_index]) as img:
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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_decisions (
            group_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            images TEXT NOT NULL,
            ai_selected_index INTEGER,
            ai_crop_coords TEXT,
            ai_confidence REAL
        )
    """
    )

    conn.commit()
    conn.close()


def process_project(
    project_id: str,
    zip_file: str,
    ranker,
    cropper,
    clip_model,
    preprocess,
    embedding_cache: dict,
) -> tuple[bool, int]:
    """Process one project: extract images, run predictions, save to temp DB.

    Handles two cases:
    1. Database exists: Read image groups from DB, match to extracted images
    2. Database doesn't exist: Group images from zip using standard grouping logic
    """
    # Paths
    existing_db = DB_DIR / f"{project_id}.db"
    temp_db = DB_DIR / f"{project_id}_ai_predictions_temp.db"
    project_tmp = TMP_DIR / project_id

    has_existing_db = existing_db.exists()

    if has_existing_db:
        pass
    else:
        pass

    # Extract zip
    project_tmp.mkdir(parents=True, exist_ok=True)

    zip_path = ORIGINAL_DIR / zip_file
    if not zip_path.exists():
        return False, 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(project_tmp)


    # Find image directory in extracted files
    image_files = (
        list(project_tmp.rglob("*.png"))
        + list(project_tmp.rglob("*.jpg"))
        + list(project_tmp.rglob("*.jpeg"))
    )
    if not image_files:
        shutil.rmtree(project_tmp)
        return False, 0


    if has_existing_db:
        # CASE 1: Database exists - read image groups from it
        conn = sqlite3.connect(existing_db)
        cursor = conn.cursor()
        cursor.execute("SELECT group_id, images FROM ai_decisions")
        rows = cursor.fetchall()
        conn.close()


        if len(rows) == 0:
            shutil.rmtree(project_tmp)
            return False, 0

        # Build filename -> path mapping
        file_map = {f.name: f for f in image_files}

        # Create temp database
        create_temp_database(temp_db, project_id)

        # Process each group from existing DB
        temp_conn = sqlite3.connect(temp_db)
        temp_cursor = temp_conn.cursor()

        processed = 0
        skipped = 0

        for group_id, images_json in tqdm(rows, desc=f"  {project_id}"):
            images = json.loads(images_json)

            # Map image names to paths
            image_paths = []
            for img_name in images:
                if img_name in file_map:
                    image_paths.append(file_map[img_name])
                else:
                    # Try just the filename without directory
                    base_name = Path(img_name).name
                    if base_name in file_map:
                        image_paths.append(file_map[base_name])

            if len(image_paths) == 0:
                skipped += 1
                continue

            # Run AI predictions
            ai_selected_index, ai_crop_coords, ai_confidence = run_ai_predictions(
                image_paths, ranker, cropper, embedding_cache, clip_model, preprocess
            )

            if ai_selected_index is None:
                skipped += 1
                continue

            # Store in temp database
            temp_cursor.execute(
                """
                INSERT INTO ai_decisions (group_id, project_id, images, ai_selected_index, ai_crop_coords, ai_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    group_id,
                    project_id,
                    images_json,
                    ai_selected_index,
                    json.dumps(ai_crop_coords) if ai_crop_coords else None,
                    ai_confidence,
                ),
            )

            processed += 1

        temp_conn.commit()
        temp_conn.close()

    else:
        # CASE 2: No database - group images ourselves using standard logic
        from scripts.utils.companion_file_utils import (
            find_consecutive_stage_groups,
            sort_image_files_by_timestamp_and_stage,
        )


        # Sort and group images
        sorted_images = sort_image_files_by_timestamp_and_stage(image_files)
        groups = find_consecutive_stage_groups(sorted_images)


        # Create temp database
        create_temp_database(temp_db, project_id)

        # Process each group
        temp_conn = sqlite3.connect(temp_db)
        temp_cursor = temp_conn.cursor()

        processed = 0
        skipped = 0

        for group_idx, group in enumerate(tqdm(groups, desc=f"  {project_id}")):
            # Create group_id
            if group:
                first_img = group[0]
                timestamp = first_img.stem.split("_")[0]
                group_id = f"{project_id}_group_{timestamp}_{group_idx}"
            else:
                skipped += 1
                continue

            # Convert group paths to relative names
            images_json = json.dumps([img.name for img in group])

            # Run AI predictions
            ai_selected_index, ai_crop_coords, ai_confidence = run_ai_predictions(
                group, ranker, cropper, embedding_cache, clip_model, preprocess
            )

            if ai_selected_index is None:
                skipped += 1
                continue

            # Store in temp database
            temp_cursor.execute(
                """
                INSERT INTO ai_decisions (group_id, project_id, images, ai_selected_index, ai_crop_coords, ai_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    group_id,
                    project_id,
                    images_json,
                    ai_selected_index,
                    json.dumps(ai_crop_coords) if ai_crop_coords else None,
                    ai_confidence,
                ),
            )

            processed += 1

        temp_conn.commit()
        temp_conn.close()

    # Clean up extracted files
    shutil.rmtree(project_tmp)


    return True, processed


def main():
    parser = argparse.ArgumentParser(
        description="Batch AI predictions - creates temp databases for existing projects"
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

    # Projects to process (database_name, zip_filename)
    # See: data/ai_data/ZIP_DATABASE_MAPPING.md for full verified mapping
    # Includes ALL 20 projects - will create temp DBs for those without existing DBs
    projects = [
        # Small projects first (fast - under 3k images)
        ("dalia", "dalia.zip"),  # No existing DB - will group from zip
        ("Patricia", "Average Patricia.zip"),  # No existing DB - will group from zip
        ("mixed-0919", "mixed-0919.zip"),  # No existing DB - will group from zip
        ("agent-1003", "agent-1003.zip"),
        ("agent-1002", "agent-1002.zip"),
        ("agent-1001", "agent-1001.zip"),
        ("1013", "1013.zip"),
        ("1010", "1010.zip"),  # No existing DB - will group from zip
        ("1102", "1102.zip"),  # No existing DB - will group from zip
        # Medium projects (3k-10k images)
        ("1011", "1011.zip"),
        ("1012", "1012.zip"),
        ("Aiko", "Aiko_raw.zip"),  # DB: Aiko, Zip: Aiko_raw
        ("Eleni", "Eleni_raw.zip"),  # DB: Eleni, Zip: Eleni_raw
        ("Kiara_Slender", "Slender Kiara.zip"),  # Different naming
        ("1100", "1100.zip"),
        ("1101_Hailey", "1101.zip"),  # DB has _Hailey suffix
        # Large projects (10k+ images - save for last)
        ("tattersail-0918", "tattersail-0918.zip"),
        ("jmlimages-random", "jmlimages-random.zip"),
        ("mojo2", "mojo2.zip"),
        ("mojo1", "mojo1.zip"),
    ]

    # Note: mojo3 skipped (already has AI predictions from earlier backfill)
    # Note: 5 projects without existing DBs will use standard grouping logic


    # Load models
    ranker_path = MODELS_DIR / args.ranker_model
    crop_path = MODELS_DIR / args.crop_model

    if not ranker_path.exists():
        return

    if not crop_path.exists():
        return

    ranker = RankingModel().to(device)
    ranker.load_state_dict(torch.load(ranker_path, map_location=device))
    ranker.eval()

    cropper = CropProposer().to(device)
    cropper.load_state_dict(torch.load(crop_path, map_location=device))
    cropper.eval()

    # Load CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Load embedding cache
    embedding_cache = load_embedding_cache()

    # Create directories
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Process each project
    start_time = time.time()
    success_count = 0
    failed_count = 0
    total_groups = 0

    for project_id, zip_file in projects:
        try:
            success, groups = process_project(
                project_id,
                zip_file,
                ranker,
                cropper,
                clip_model,
                preprocess,
                embedding_cache,
            )

            if success:
                success_count += 1
                total_groups += groups

                # Log success
                with open(LOG_FILE, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "project_id": project_id,
                                "status": "success",
                                "groups_processed": groups,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                        + "\n"
                    )
            else:
                failed_count += 1

                # Log failure
                with open(LOG_FILE, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "project_id": project_id,
                                "status": "failed",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                        + "\n"
                    )

        except Exception as e:
            failed_count += 1

            with open(LOG_FILE, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "project_id": project_id,
                            "status": "error",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                    )
                    + "\n"
                )

    # Final summary
    elapsed = time.time() - start_time
    int(elapsed // 3600)
    int((elapsed % 3600) // 60)
    int(elapsed % 60)



if __name__ == "__main__":
    main()
