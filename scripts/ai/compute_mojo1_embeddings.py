#!/usr/bin/env python3
"""
Compute CLIP embeddings for Mojo 1 images.
This is a simplified version that directly processes a directory.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

# Paths
MOJO1_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1")
EMBEDDINGS_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/data/ai_data/cache/embeddings")
CACHE_FILE = Path("/Users/eriksjaastad/projects/Eros Mate/data/ai_data/cache/processed_images.jsonl")

def get_image_hash(image_path: Path) -> str:
    """Generate a hash for the image path."""
    return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]

def main():
    print("="*60)
    print("MOJO 1 EMBEDDINGS COMPUTATION")
    print("="*60)
    print(f"Source: {MOJO1_DIR}")
    print(f"Output: {EMBEDDINGS_DIR}")
    print()
    
    # Find all PNGs
    print("📂 Scanning for images...")
    image_paths = list(MOJO1_DIR.rglob("*.png"))
    print(f"   Found {len(image_paths)} images")
    
    # Load already processed
    processed = set()
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed.add(data['image_path'])
    
    print(f"   Already processed: {len(processed)}")
    
    # Filter to unprocessed
    to_process = []
    for img_path in image_paths:
        rel_path = str(img_path.relative_to(Path("/Users/eriksjaastad/projects/Eros Mate")))
        if rel_path not in processed:
            to_process.append((img_path, rel_path))
    
    print(f"   To process: {len(to_process)}")
    
    if not to_process:
        print("\n✅ All images already have embeddings!")
        return
    
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🤖 Loading OpenCLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    print("   Model loaded")
    
    # Create output dir
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process images
    print(f"\n⚡ Processing {len(to_process)} images...")
    print()
    
    with open(CACHE_FILE, 'a') as cache_f:
        for img_path, rel_path in tqdm(to_process, desc="Computing embeddings"):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Compute embedding
                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                
                # Save embedding
                img_hash = get_image_hash(img_path)
                emb_file = EMBEDDINGS_DIR / f"{img_hash}.npy"
                np.save(emb_file, embedding)
                
                # Update cache
                cache_entry = {
                    'image_path': rel_path,
                    'embedding_file': str(emb_file.relative_to(Path("/Users/eriksjaastad/projects/Eros Mate"))),
                    'hash': img_hash
                }
                cache_f.write(json.dumps(cache_entry) + '\n')
                cache_f.flush()
                
            except Exception as e:
                print(f"\n⚠️  Error processing {img_path.name}: {e}")
                continue
    
    print()
    print("="*60)
    print("✅ EMBEDDINGS COMPUTATION COMPLETE")
    print("="*60)
    print(f"Processed: {len(to_process)} images")
    print(f"Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"Cache updated: {CACHE_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()

