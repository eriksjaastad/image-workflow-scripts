#!/usr/bin/env python3
"""
Feature Extraction for Crop Training Data
==========================================

READ-ONLY script that extracts features from training images.
NEVER modifies source images, YAML files, or any existing content.

Creates NEW sidecar files:
- data/ai_data/embeddings/{image_hash}.npy (CLIP embeddings)
- data/ai_data/cache/processed_files.jsonl (tracking)

Usage:
    python scripts/ai/compute_training_features.py [--max-images N]
"""

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import AI dependencies
try:
    import open_clip
    import torch

    HAS_AI_DEPS = True
except ImportError:
    HAS_AI_DEPS = False
    sys.exit(1)

# Verify MPS (Apple GPU) availability
if not torch.backends.mps.is_available():
    DEVICE = "cpu"
else:
    DEVICE = "mps"


class FeatureExtractor:
    """Read-only feature extractor - NEVER modifies source files"""

    def __init__(self, output_dir: Path, device: str = "mps"):
        self.output_dir = output_dir
        self.embeddings_dir = output_dir / "embeddings"
        self.cache_dir = output_dir / "cache"
        self.device = device

        # Create output directories (NEW files only)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load cache of already-processed files
        self.cache_file = self.cache_dir / "processed_files.jsonl"
        self.processed = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        """Load cache of already-processed files"""
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        cache[entry["file_path"]] = entry["embedding_hash"]
        return cache

    def _append_to_cache(
        self, file_path: str, embedding_hash: str, processing_time: float
    ):
        """Append to cache (NEW writes only)"""
        with open(self.cache_file, "a") as f:
            entry = {
                "file_path": file_path,
                "embedding_hash": embedding_hash,
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "processing_time_sec": round(processing_time, 3),
            }
            f.write(json.dumps(entry) + "\n")

    def _get_image_hash(self, image_path: Path) -> str:
        """Generate stable hash for image (read-only)"""
        # Use file path + mtime for hash (no file content modification)
        stat = image_path.stat()
        hash_input = f"{image_path.name}_{stat.st_size}_{stat.st_mtime_ns}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _extract_embedding(self, image_path: Path) -> np.ndarray | None:
        """Extract CLIP embedding (read-only operation)"""
        try:
            # Read image (NO modifications)
            image = Image.open(image_path).convert("RGB")

            # Preprocess and extract features
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                # Normalize for similarity comparisons
                features = features / features.norm(dim=-1, keepdim=True)

            # Convert to numpy (CPU)
            embedding = features.cpu().numpy()[0]
            return embedding

        except Exception:
            return None

    def process_image(self, image_path: Path, force: bool = False) -> bool:
        """Process single image (read-only)"""
        # Check cache (skip if already processed)
        file_path_str = str(image_path)
        if not force and file_path_str in self.processed:
            return True  # Already processed

        # Generate hash for output filename
        image_hash = self._get_image_hash(image_path)
        embedding_path = self.embeddings_dir / f"{image_hash}.npy"

        # Skip if embedding already exists
        if not force and embedding_path.exists():
            return True

        # Extract embedding (read-only operation)
        start_time = time.time()
        embedding = self._extract_embedding(image_path)
        processing_time = time.time() - start_time

        if embedding is None:
            return False

        # Save embedding (NEW file)
        np.save(embedding_path, embedding)

        # Update cache (NEW writes only)
        self._append_to_cache(file_path_str, image_hash, processing_time)

        return True

    def find_training_images_from_csv(self, training_dir: Path) -> list[Path]:
        """Find all images referenced in training CSV logs (read-only)"""
        import csv

        unique_images = set()
        project_root = training_dir.parent.parent  # data/training -> project root

        # Read select_crop_log.csv
        select_crop_log = training_dir / "select_crop_log.csv"
        if select_crop_log.exists():
            with open(select_crop_log) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Get chosen path
                    if row.get("chosen_path"):
                        img_path = project_root / row["chosen_path"]
                        if img_path.exists():
                            unique_images.add(img_path)

                    # Get all images in the set (image_0_path, image_1_path, etc.)
                    i = 0
                    while f"image_{i}_path" in row:
                        img_path_str = row[f"image_{i}_path"]
                        if img_path_str:
                            img_path = project_root / img_path_str
                            if img_path.exists():
                                unique_images.add(img_path)
                        i += 1

        # Read selection_only_log.csv
        selection_log = training_dir / "selection_only_log.csv"
        if selection_log.exists():
            with open(selection_log) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Get chosen path
                    if row.get("chosen_path"):
                        img_path = project_root / row["chosen_path"]
                        if img_path.exists():
                            unique_images.add(img_path)

                    # Get negative paths
                    neg_paths_str = row.get("negative_paths", "")
                    if neg_paths_str:
                        # negative_paths is a JSON array string
                        try:
                            import json

                            neg_paths = (
                                json.loads(neg_paths_str) if neg_paths_str else []
                            )
                            for path_str in neg_paths:
                                img_path = project_root / path_str
                                if img_path.exists():
                                    unique_images.add(img_path)
                        except Exception:
                            pass

        return sorted(unique_images)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from training images (read-only)"
    )
    parser.add_argument(
        "--max-images", type=int, help="Limit number of images to process (for testing)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process already-cached images"
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "training",
        help="Training CSV directory (default: data/training)",
    )
    args = parser.parse_args()

    # Initialize extractor
    output_dir = PROJECT_ROOT / "data" / "ai_data"
    extractor = FeatureExtractor(output_dir, device=DEVICE)

    # Find training images from CSV logs (read-only)
    training_images = extractor.find_training_images_from_csv(args.training_dir)

    if not training_images:
        return

    # Limit if requested
    if args.max_images:
        training_images = training_images[: args.max_images]

    # Process images
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, image_path in enumerate(training_images, 1):
        # Show progress
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start_time
            i / elapsed if elapsed > 0 else 0

        # Process (read-only)
        success = extractor.process_image(image_path, force=args.force)

        if success:
            # Check if it was newly processed or skipped
            if str(image_path) in extractor.processed:
                processed_count += 1
            else:
                skipped_count += 1
        else:
            error_count += 1

    # Final stats
    time.time() - start_time


if __name__ == "__main__":
    main()
