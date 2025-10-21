#!/usr/bin/env python3
"""
CLIP Embedding Extraction for Training Images
==============================================
Extracts CLIP embeddings from images referenced in training logs.

Usage:
    python scripts/ai/compute_embeddings.py
    python scripts/ai/compute_embeddings.py --max-images 100  # Test mode
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set, List
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import hashlib
import numpy as np

def get_image_hash(image_path: Path) -> str:
    """Generate a hash for the image path to use as filename."""
    return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]

def load_training_image_paths(data_dir: Path) -> Set[Path]:
    """Load all unique image paths from training logs."""
    image_paths = set()
    import csv
    import json as json_lib
    
    # Load from selection log
    selection_log = data_dir / "training" / "selection_only_log.csv"
    if selection_log.exists():
        print(f"Reading {selection_log}...")
        with open(selection_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add chosen path
                chosen = row.get('chosen_path', '').strip()
                if chosen and Path(chosen).exists():
                    image_paths.add(Path(chosen))
                
                # Add negative paths
                neg_paths_str = row.get('neg_paths', '').strip()
                if neg_paths_str:
                    try:
                        neg_paths = json_lib.loads(neg_paths_str.replace('""', '"'))
                        for path in neg_paths:
                            if path and Path(path).exists():
                                image_paths.add(Path(path))
                    except:
                        pass
    
    # Load from crop log
    crop_log = data_dir / "training" / "select_crop_log.csv"
    if crop_log.exists():
        print(f"Reading {crop_log}...")
        with open(crop_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add chosen path
                chosen = row.get('chosen_path', '').strip()
                if chosen and Path(chosen).exists():
                    image_paths.add(Path(chosen))
    
    return image_paths

def compute_embeddings(
    image_paths: List[Path],
    output_dir: Path,
    cache_file: Path,
    device: str = "mps",
    max_images: int = None
):
    """
    Compute CLIP embeddings for images.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Directory to save embeddings
        cache_file: File to track processed images
        device: Device to use (mps, cuda, cpu)
        max_images: Maximum images to process (for testing)
    """
    # Load model
    print(f"\nLoading OpenCLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    
    # Load cache of already processed images
    processed = set()
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed.add(data['image_path'])
        print(f"Found {len(processed)} already processed images in cache")
    
    # Filter out already processed
    to_process = [p for p in image_paths if str(p) not in processed]
    
    if max_images:
        to_process = to_process[:max_images]
        print(f"Test mode: Processing only {len(to_process)} images")
    
    if not to_process:
        print("All images already processed!")
        return
    
    print(f"\nProcessing {len(to_process)} images...")
    print(f"Saving embeddings to: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    failed = []
    with open(cache_file, 'a') as cache:
        for img_path in tqdm(to_process, desc="Extracting embeddings"):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                
                # Save embedding
                img_hash = get_image_hash(img_path)
                emb_path = output_dir / f"{img_hash}.npy"
                np.save(emb_path, embedding)
                
                # Update cache
                cache_entry = {
                    'image_path': str(img_path),
                    'embedding_file': str(emb_path),
                    'hash': img_hash
                }
                cache.write(json.dumps(cache_entry) + '\n')
                cache.flush()
                
            except Exception as e:
                failed.append((str(img_path), str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"✅ Processed: {len(to_process) - len(failed)}")
    print(f"❌ Failed:    {len(failed)}")
    print(f"📁 Saved to:  {output_dir}")
    print(f"📝 Cache:     {cache_file}")
    
    if failed:
        print(f"\n⚠️  Failed images ({len(failed)}):")
        for path, error in failed[:10]:
            print(f"  • {path}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from training images")
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum images to process (for testing)'
    )
    parser.add_argument(
        '--device',
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for processing'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path("data")
    output_dir = data_dir / "ai_data" / "embeddings"
    cache_dir = data_dir / "ai_data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "processed_images.jsonl"
    
    # Load training image paths
    print("=" * 60)
    print("CLIP EMBEDDING EXTRACTION")
    print("=" * 60)
    
    image_paths = load_training_image_paths(data_dir)
    print(f"\nFound {len(image_paths)} unique images in training logs")
    
    # Filter to existing images only
    existing = [p for p in image_paths if p.exists()]
    missing = len(image_paths) - len(existing)
    
    if missing > 0:
        print(f"⚠️  Warning: {missing} images from logs no longer exist")
    
    if not existing:
        print("❌ No images found to process!")
        return 1
    
    # Compute embeddings
    compute_embeddings(
        image_paths=existing,
        output_dir=output_dir,
        cache_file=cache_file,
        device=args.device,
        max_images=args.max_images
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

