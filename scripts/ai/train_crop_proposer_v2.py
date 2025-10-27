#!/usr/bin/env python3
"""
Train Crop Proposer Model v2 - Full Dataset

Purpose: Learn where users crop images based on 12,679 historical crop decisions.

Architecture:
- Input: 514-dim (512 CLIP embedding + 2 image dimensions normalized)
- Network: MLP (514 → 512 → 256 → 128 → 4)
- Output: 4 values (x1, y1, x2, y2) in normalized [0, 1] coordinates
- Loss: Smooth L1 Loss (less sensitive to outliers than MSE)

Usage:
    python scripts/ai/train_crop_proposer_v2.py --epochs 50
    
Output:
    - data/ai_data/models/crop_proposer_v2.pt
    - data/ai_data/models/crop_proposer_v2_metadata.json
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path("/Users/eriksjaastad/projects/Eros Mate")
CROP_LOG = PROJECT_ROOT / "data/training/select_crop_log.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data/ai_data/cache/processed_images.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data/ai_data/cache/embeddings"  # Fixed: was data/ai_data/embeddings
MODEL_DIR = PROJECT_ROOT / "data/ai_data/models"

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_embeddings_cache() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load embeddings cache with path and filename lookup."""
    print("📂 Loading embeddings cache...")
    cache = {}
    filename_cache = {}
    
    with EMBEDDINGS_CACHE.open('r') as f:
        for line in f:
            entry = json.loads(line)
            path = entry['image_path']
            hash_val = entry['hash']
            
            cache[path] = hash_val
            
            filename = Path(path).name
            if filename not in filename_cache:
                filename_cache[filename] = hash_val
            elif 'training data' not in path:
                filename_cache[filename] = hash_val
    
    print(f"   Loaded {len(cache):,} cached embeddings")
    print(f"   Filename index: {len(filename_cache):,} unique filenames")
    
    return cache, filename_cache


def normalize_path(path: str) -> str:
    """Normalize path to match embeddings format."""
    p = Path(path)
    
    if p.is_absolute():
        try:
            p = p.relative_to(PROJECT_ROOT)
        except ValueError:
            parts = p.parts
            for i, part in enumerate(parts):
                if part in ['mojo1', 'mojo2', 'jmlimages-random', 'tattersail-0918',
                           '1100', '1101_hailey', '1011', '1012', '1013',
                           'agent-1001', 'agent-1002', 'agent-1003',
                           'Kiara_Slender', 'Aiko_raw', 'Eleni_raw']:
                    return str(Path(*parts[i:]))
    
    return str(p)


def find_image_in_training_data(filename: str) -> Path:
    """Find image file in training data directories by searching recursively."""
    training_data_dir = PROJECT_ROOT / "training data"
    
    # Search recursively for the filename
    for img_path in training_data_dir.rglob(filename):
        if img_path.is_file():
            return img_path
    
    return None


def load_image_dimensions(filename: str) -> Tuple[int, int]:
    """Load actual image dimensions from disk."""
    img_path = find_image_in_training_data(filename)
    if not img_path:
        raise FileNotFoundError(f"Image not found: {filename}")
    
    with Image.open(img_path) as img:
        return img.size  # Returns (width, height)


def load_crop_data(cache: Dict, filename_cache: Dict):
    """Load crop training data, matching by filename and loading image dimensions."""
    print(f"\n📂 Loading crop data: {CROP_LOG}")
    
    crops = []
    skipped = 0
    skipped_no_embed = 0
    skipped_no_image = 0
    skipped_invalid_crop = 0
    loaded_dims_from_csv = 0
    loaded_dims_from_disk = 0
    
    with CROP_LOG.open('r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                path = row['chosen_path']
                
                # Extract just the filename for matching
                filename = Path(path).name
                
                # Check if we have an embedding for this filename
                if filename not in filename_cache:
                    skipped_no_embed += 1
                    skipped += 1
                    continue
                
                # Verify the embedding file exists
                hash_val = filename_cache[filename]
                emb_file = EMBEDDINGS_DIR / f"{hash_val}.npy"
                if not emb_file.exists():
                    skipped_no_embed += 1
                    skipped += 1
                    continue
                
                # Get the chosen index to find the right dimensions
                chosen_idx = int(row['chosen_index'])
                width_key = f'width_{chosen_idx}'
                height_key = f'height_{chosen_idx}'
                
                # Try to get dimensions from CSV
                width = float(row[width_key])
                height = float(row[height_key])
                
                # If dimensions are missing/invalid, load from actual image
                if width <= 0 or height <= 0:
                    try:
                        width, height = load_image_dimensions(filename)
                        loaded_dims_from_disk += 1
                    except (FileNotFoundError, OSError):
                        skipped_no_image += 1
                        skipped += 1
                        continue
                else:
                    loaded_dims_from_csv += 1
                
                x1 = float(row['crop_x1'])
                y1 = float(row['crop_y1'])
                x2 = float(row['crop_x2'])
                y2 = float(row['crop_y2'])
                
                # Normalize coordinates if they're in pixels
                if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                    # Convert from pixels to normalized [0, 1]
                    x1 = x1 / width
                    y1 = y1 / height
                    x2 = x2 / width
                    y2 = y2 / height
                
                # Validate normalized crop coordinates
                if x2 <= x1 or y2 <= y1:
                    skipped_invalid_crop += 1
                    skipped += 1
                    continue
                if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    skipped_invalid_crop += 1
                    skipped += 1
                    continue
                
                crop = {
                    'filename': filename,  # Store filename for matching
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': width,
                    'height': height
                }
                crops.append(crop)
            except (KeyError, ValueError):
                skipped += 1
                continue
    
    print(f"   ✅ Loaded: {len(crops):,} crop examples")
    print("   📊 Dimensions:")
    print(f"      • From CSV: {loaded_dims_from_csv:,}")
    print(f"      • From disk: {loaded_dims_from_disk:,}")
    print(f"   ⏭️  Skipped: {skipped:,} total")
    print(f"      • No embedding: {skipped_no_embed:,}")
    print(f"      • No image file: {skipped_no_image:,}")
    print(f"      • Invalid crop coords: {skipped_invalid_crop:,}")
    
    if not crops:
        raise ValueError("No crop data loaded!")
    
    return crops


class CropDataset(Dataset):
    """Dataset for crop proposer training."""
    
    def __init__(self, crops, cache, filename_cache):
        self.crops = crops
        self.cache = cache
        self.filename_cache = filename_cache
    
    def load_embedding(self, filename: str) -> np.ndarray:
        """Load embedding by filename."""
        if filename not in self.filename_cache:
            raise ValueError(f"No embedding for filename: {filename}")
        
        hash_val = self.filename_cache[filename]
        emb_file = EMBEDDINGS_DIR / f"{hash_val}.npy"
        
        if not emb_file.exists():
            raise ValueError(f"Embedding file missing: {emb_file}")
        
        return np.load(emb_file)
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop = self.crops[idx]
        
        # Load CLIP embedding (512-dim) by filename
        embedding = self.load_embedding(crop['filename'])
        embedding = torch.from_numpy(embedding).float()
        
        # Add image dimensions (normalized to [0, 1])
        # Typical range: 512-2048 pixels, so divide by 2048
        dims = torch.tensor([
            crop['width'] / 2048.0,
            crop['height'] / 2048.0
        ], dtype=torch.float32)
        
        # Concatenate: 512 + 2 = 514 dimensions
        features = torch.cat([embedding, dims])
        
        # Crop coordinates are now normalized to [0, 1]
        target = torch.tensor([
            crop['x1'],
            crop['y1'],
            crop['x2'],
            crop['y2']
        ], dtype=torch.float32)
        
        return features, target


class CropProposer(nn.Module):
    """
    MLP that predicts crop box from image embedding + dimensions.
    
    Architecture: 514 → 512 → 256 → 128 → 4
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
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, target in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(features)
        loss = criterion(predictions, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            target = target.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def compute_iou(pred, target):
    """Compute Intersection over Union for crop boxes."""
    # pred and target are (batch_size, 4) tensors with [x1, y1, x2, y2]
    
    # Intersection
    x1_i = torch.max(pred[:, 0], target[:, 0])
    y1_i = torch.max(pred[:, 1], target[:, 1])
    x2_i = torch.min(pred[:, 2], target[:, 2])
    y2_i = torch.min(pred[:, 3], target[:, 3])
    
    intersection = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)
    
    # Union
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_target - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def evaluate_iou(model, dataloader, device):
    """Evaluate IoU on validation set."""
    model.eval()
    total_iou = 0
    
    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            target = target.to(device)
            
            predictions = model(features)
            iou = compute_iou(predictions, target)
            
            total_iou += iou
    
    return total_iou / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train Crop Proposer v2')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CROP PROPOSER V2 TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Load data
    cache, filename_cache = load_embeddings_cache()
    crops = load_crop_data(cache, filename_cache)
    
    # Train/val split (85% / 15%)
    val_size = int(len(crops) * 0.15)
    train_crops = crops[val_size:]
    val_crops = crops[:val_size]
    
    print("\n📊 Train/Val split:")
    print(f"   Train: {len(train_crops):,} crops")
    print(f"   Val:   {len(val_crops):,} crops")
    
    # Create datasets
    train_dataset = CropDataset(train_crops, cache, filename_cache)
    val_dataset = CropDataset(val_crops, cache, filename_cache)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = CropProposer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()  # Less sensitive to outliers than MSE
    
    # Training loop
    print("\n🚀 Starting training...")
    print()
    
    best_iou = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        val_iou = evaluate_iou(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val IoU:    {val_iou:.4f} ⭐")
        
        # Save best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            
            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "crop_proposer_v2.pt")
            
            # Save metadata
            metadata = {
                'model_version': 'v2',
                'created': datetime.now(timezone.utc).isoformat(),
                'training_examples': len(train_crops),
                'validation_examples': len(val_crops),
                'best_epoch': best_epoch,
                'best_iou': best_iou,
                'best_val_loss': val_loss,
                'hyperparameters': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'loss_function': 'SmoothL1Loss'
                },
                'architecture': 'MLP (514→512→256→128→4) with Sigmoid',
                'input_features': '512 CLIP + 2 dimensions'
            }
            
            with (MODEL_DIR / "crop_proposer_v2_metadata.json").open('w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✅ Saved best model (IoU={best_iou:.4f})")
        
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best IoU: {best_iou:.4f}")
    print()
    print("📁 Model saved:")
    print(f"   {MODEL_DIR / 'crop_proposer_v2.pt'}")
    print(f"   {MODEL_DIR / 'crop_proposer_v2_metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

