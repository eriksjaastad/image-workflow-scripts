#!/usr/bin/env python3
"""
Train Ranking Model v3 with Full Dataset + Configurable Anomaly Weighting

Key improvements over v2:
1. Uses ALL 21,000+ training selections (v2 used ~10k)
2. Proper train/val split BEFORE any training
3. Configurable anomaly oversampling weight (5x, 10x, 20x)
4. Separate anomaly validation set for focused testing
5. Project boundary validation throughout

Usage:
    python scripts/ai/train_ranker_v3.py --anomaly-weight 10
    python scripts/ai/train_ranker_v3.py --anomaly-weight 20 --epochs 50
    
Output:
    - data/ai_data/models/ranker_v3_w{weight}.pt
    - data/ai_data/models/ranker_v3_w{weight}_metadata.json
"""

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path("/Users/eriksjaastad/projects/Eros Mate")
SELECTION_LOG = PROJECT_ROOT / "data/training/selection_only_log.csv"
ANOMALY_CSV = PROJECT_ROOT / "data/training/anomaly_cases.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data/ai_data/cache/processed_images.jsonl"
MODEL_DIR = PROJECT_ROOT / "data/ai_data/models"

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"


def get_project_id(path: str) -> str:
    """Extract project ID from file path."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith('mojo'):
            return part
        if part in ['eleni', 'aiko', 'dalia', 'kiara', 'jmlimages-random',
                    'tattersail-0918', '1100', '1101_hailey', '1011', '1012',
                    '1013', 'agent-1001', 'agent-1002', 'agent-1003',
                    'Kiara_Slender', 'Kiara_Average', 'Aiko_raw', 'Eleni_raw']:
            return part
        if part.endswith('_raw') or part.endswith('_final'):
            return part.replace('_raw', '').replace('_final', '')
    return 'unknown'


def validate_set_integrity(chosen_path: str, neg_paths: List[str]) -> None:
    """Verify all paths in a set are from the same project."""
    chosen_project = get_project_id(chosen_path)
    for neg_path in neg_paths:
        neg_project = get_project_id(neg_path)
        if neg_project != chosen_project:
            raise ValueError(
                f"🚨 MIXED PROJECTS DETECTED\n"
                f"  Winner: {chosen_path} (project: {chosen_project})\n"
                f"  Loser:  {neg_path} (project: {neg_project})\n"
            )


def normalize_path(path: str) -> str:
    """
    Normalize path to match embeddings cache format.
    
    Cache uses relative paths like: mojo2/_mixed/file.png
    CSV has absolute paths like: /Users/.../Eros Mate/mojo2/_mixed/file.png
    """
    p = Path(path)
    
    # Make relative to project root if absolute
    if p.is_absolute():
        try:
            p = p.relative_to(PROJECT_ROOT)
        except ValueError:
            # If not under project root, try to extract just the relevant parts
            parts = p.parts
            # Look for known project directories
            for i, part in enumerate(parts):
                if part in ['mojo1', 'mojo2', 'jmlimages-random', 'tattersail-0918',
                           '1100', '1101_hailey', '1011', '1012', '1013',
                           'agent-1001', 'agent-1002', 'agent-1003',
                           'Kiara_Slender', 'Aiko_raw', 'Eleni_raw']:
                    # Reconstruct path from this point
                    return str(Path(*parts[i:]))
    
    return str(p)


def load_embeddings_cache() -> Dict[str, str]:
    """
    Load embeddings cache with BOTH path-based and filename-based lookup.
    
    Returns dict with:
    - Full paths as keys
    - ALSO filenames as keys (for fuzzy matching)
    """
    print("📂 Loading embeddings cache...")
    cache = {}
    filename_cache = {}  # Backup lookup by filename only
    
    with EMBEDDINGS_CACHE.open('r') as f:
        for line in f:
            entry = json.loads(line)
            path = entry['image_path']
            hash_val = entry['hash']
            
            # Store by full path
            cache[path] = hash_val
            
            # Also store by filename for fuzzy matching
            filename = Path(path).name
            if filename not in filename_cache:
                filename_cache[filename] = hash_val
            else:
                # If duplicate filename, keep the one without 'training data' prefix
                if 'training data' not in path:
                    filename_cache[filename] = hash_val
    
    print(f"   Loaded {len(cache):,} cached embeddings")
    print(f"   Filename index: {len(filename_cache):,} unique filenames")
    
    # Merge filename cache into main cache for fallback lookup
    cache['__filename_cache__'] = filename_cache
    return cache


def load_anomaly_set() -> set:
    """Load set of (chosen_path, neg_path) tuples that are anomalies."""
    print("📂 Loading anomaly cases...")
    anomalies = set()
    
    with ANOMALY_CSV.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chosen = normalize_path(row['chosen_path'])
            # Replace single quotes with double quotes for JSON parsing
            rejected_str = row['rejected_paths'].replace("'", '"')
            rejected = json.loads(rejected_str)
            for neg_path in rejected:
                anomalies.add((chosen, normalize_path(neg_path)))
    
    print(f"   Loaded {len(anomalies):,} anomaly pairs")
    return anomalies


def load_training_data(embeddings_cache: Dict, anomaly_set: set) -> Tuple[List[Dict], List[Dict]]:
    """
    Load ALL selection data and split into normal vs anomaly cases.
    
    Returns:
        (normal_cases, anomaly_cases) as list of pairwise comparisons
    """
    print(f"\n📂 Loading training data: {SELECTION_LOG}")
    
    normal_pairs = []
    anomaly_pairs = []
    skipped = 0
    
    with SELECTION_LOG.open('r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            chosen_path = row['chosen_path']
            neg_paths_str = row['neg_paths']
            
            # Parse negative paths
            try:
                neg_paths = json.loads(neg_paths_str.replace('""', '"'))
            except Exception:
                skipped += 1
                continue
            
            # Validate project boundaries
            try:
                validate_set_integrity(chosen_path, neg_paths)
            except ValueError as e:
                print(f"\n{e}")
                raise
            
            # Normalize paths
            chosen_norm = normalize_path(chosen_path)
            neg_norms = [normalize_path(p) for p in neg_paths]
            
            # Check embeddings exist - try exact path first, then filename fallback
            filename_cache = embeddings_cache.get('__filename_cache__', {})
            embeddings_dir = PROJECT_ROOT / "data/ai_data/embeddings"
            
            def has_valid_embedding(path):
                """Check if path has embedding AND the file actually exists."""
                # Try exact path
                if path in embeddings_cache:
                    hash_val = embeddings_cache[path]
                    emb_file = embeddings_dir / f"{hash_val}.npy"
                    if emb_file.exists():
                        return True
                
                # Try filename fallback
                filename = Path(path).name
                if filename in filename_cache:
                    hash_val = filename_cache[filename]
                    emb_file = embeddings_dir / f"{hash_val}.npy"
                    if emb_file.exists():
                        return True
                
                return False
            
            if not has_valid_embedding(chosen_norm):
                skipped += 1
                continue
            
            missing = [p for p in neg_norms if not has_valid_embedding(p)]
            if missing:
                skipped += 1
                continue
            
            # Create pairwise comparisons
            for neg_norm in neg_norms:
                is_anomaly = (chosen_norm, neg_norm) in anomaly_set
                
                pair = {
                    'winner': chosen_norm,
                    'loser': neg_norm,
                    'is_anomaly': is_anomaly,
                    'project': get_project_id(chosen_path)
                }
                
                if is_anomaly:
                    anomaly_pairs.append(pair)
                else:
                    normal_pairs.append(pair)
    
    print(f"   Normal pairs: {len(normal_pairs):,}")
    total_pairs = len(normal_pairs) + len(anomaly_pairs)
    if total_pairs > 0:
        print(f"   Anomaly pairs: {len(anomaly_pairs):,} ({100*len(anomaly_pairs)/total_pairs:.1f}%)")
    else:
        print(f"   Anomaly pairs: {len(anomaly_pairs):,}")
    print(f"   Skipped: {skipped:,}")
    
    if not normal_pairs and not anomaly_pairs:
        raise ValueError("No training data loaded!")
    
    return normal_pairs, anomaly_pairs


class RankingDataset(Dataset):
    """Dataset with configurable anomaly oversampling."""
    
    def __init__(self, normal_pairs: List[Dict], anomaly_pairs: List[Dict],
                 embeddings_cache: Dict, anomaly_oversample_factor: int = 10):
        self.embeddings_cache = embeddings_cache
        self.embeddings_dir = PROJECT_ROOT / "data/ai_data/embeddings"
        
        # Build training pairs with oversampling
        self.pairs = []
        
        # Add normal pairs once
        self.pairs.extend(normal_pairs)
        
        # Add anomaly pairs multiple times
        for _ in range(anomaly_oversample_factor):
            self.pairs.extend(anomaly_pairs)
        
        # Shuffle
        random.shuffle(self.pairs)
        
        print("\n📊 Dataset created:")
        print(f"   Normal pairs: {len(normal_pairs):,}")
        print(f"   Anomaly pairs: {len(anomaly_pairs):,} × {anomaly_oversample_factor} = {len(anomaly_pairs)*anomaly_oversample_factor:,}")
        print(f"   Total: {len(self.pairs):,} pairs")
    
    def load_embedding(self, image_path: str) -> np.ndarray:
        """Load embedding from cache, with filename fallback if needed."""
        # Try exact path first
        if image_path in self.embeddings_cache:
            emb_hash = self.embeddings_cache[image_path]
        else:
            # Fallback to filename lookup
            filename = Path(image_path).name
            filename_cache = self.embeddings_cache.get('__filename_cache__', {})
            if filename not in filename_cache:
                raise ValueError(f"No embedding found for {image_path} (or filename {filename})")
            emb_hash = filename_cache[filename]
        
        # Hash is stored as just the hash value, load as {hash}.npy
        emb_file = self.embeddings_dir / f"{emb_hash}.npy"
        
        if not emb_file.exists():
            raise ValueError(f"Embedding file not found: {emb_file}")
        
        return np.load(emb_file)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        winner_emb = torch.from_numpy(self.load_embedding(pair['winner'])).float()
        loser_emb = torch.from_numpy(self.load_embedding(pair['loser'])).float()
        is_anomaly = torch.tensor(1.0 if pair['is_anomaly'] else 0.0)
        
        return winner_emb, loser_emb, is_anomaly


class RankingModel(nn.Module):
    """Simple MLP scorer: 512 → 256 → 64 → 1"""
    
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


class WeightedMarginRankingLoss(nn.Module):
    """Margin ranking loss with higher weight for anomaly cases."""
    
    def __init__(self, margin=0.5, anomaly_weight=10.0):
        super().__init__()
        self.margin = margin
        self.anomaly_weight = anomaly_weight
    
    def forward(self, winner_scores, loser_scores, is_anomaly):
        # Basic margin loss
        loss = torch.relu(self.margin - (winner_scores - loser_scores))
        
        # Weight anomalies higher
        weights = 1.0 + is_anomaly * (self.anomaly_weight - 1.0)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for winner_emb, loser_emb, is_anomaly in tqdm(dataloader, desc="Training"):
        winner_emb = winner_emb.to(device)
        loser_emb = loser_emb.to(device)
        is_anomaly = is_anomaly.to(device)
        
        optimizer.zero_grad()
        
        winner_scores = model(winner_emb)
        loser_scores = model(loser_emb)
        
        loss = criterion(winner_scores, loser_scores, is_anomaly)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (winner_scores > loser_scores).sum().item()
        total += len(winner_scores)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate with separate normal and anomaly metrics."""
    model.eval()
    total_loss = 0
    
    normal_correct = 0
    normal_total = 0
    anomaly_correct = 0
    anomaly_total = 0
    
    with torch.no_grad():
        for winner_emb, loser_emb, is_anomaly in dataloader:
            winner_emb = winner_emb.to(device)
            loser_emb = loser_emb.to(device)
            is_anomaly = is_anomaly.to(device)
            
            winner_scores = model(winner_emb)
            loser_scores = model(loser_emb)
            
            loss = criterion(winner_scores, loser_scores, is_anomaly)
            total_loss += loss.item()
            
            correct_mask = (winner_scores > loser_scores)
            anomaly_mask = (is_anomaly > 0.5)
            normal_mask = ~anomaly_mask
            
            normal_correct += (correct_mask & normal_mask).sum().item()
            normal_total += normal_mask.sum().item()
            
            anomaly_correct += (correct_mask & anomaly_mask).sum().item()
            anomaly_total += anomaly_mask.sum().item()
    
    overall_acc = (normal_correct + anomaly_correct) / (normal_total + anomaly_total) if (normal_total + anomaly_total) > 0 else 0
    normal_acc = normal_correct / normal_total if normal_total > 0 else 0
    anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
        'overall_acc': overall_acc,
        'normal_acc': normal_acc,
        'anomaly_acc': anomaly_acc,
        'normal_total': normal_total,
        'anomaly_total': anomaly_total
    }


def main():
    parser = argparse.ArgumentParser(description='Train Ranker v3')
    parser.add_argument('--anomaly-weight', type=float, default=10.0,
                       help='Oversampling factor for anomalies (default: 10)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("RANKING MODEL V3 TRAINING - FULL DATASET")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Anomaly weight: {args.anomaly_weight}x")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Load data
    embeddings_cache = load_embeddings_cache()
    anomaly_set = load_anomaly_set()
    normal_pairs, anomaly_pairs = load_training_data(embeddings_cache, anomaly_set)
    
    # Train/val split (stratified by anomaly status)
    # Hold out 20% of anomalies, 10% of normal for validation
    random.seed(42)  # Reproducible splits
    random.shuffle(normal_pairs)
    random.shuffle(anomaly_pairs)
    
    normal_val_size = int(len(normal_pairs) * 0.1)
    anomaly_val_size = int(len(anomaly_pairs) * 0.2)
    
    normal_train = normal_pairs[normal_val_size:]
    normal_val = normal_pairs[:normal_val_size]
    anomaly_train = anomaly_pairs[anomaly_val_size:]
    anomaly_val = anomaly_pairs[:anomaly_val_size]
    
    print("\n📊 Train/Val split:")
    print(f"   Train: {len(normal_train):,} normal + {len(anomaly_train):,} anomalies")
    print(f"   Val:   {len(normal_val):,} normal + {len(anomaly_val):,} anomalies")
    
    # Create datasets
    train_dataset = RankingDataset(normal_train, anomaly_train, embeddings_cache,
                                   anomaly_oversample_factor=int(args.anomaly_weight))
    val_dataset = RankingDataset(normal_val, anomaly_val, embeddings_cache,
                                 anomaly_oversample_factor=1)  # No oversampling in val
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = RankingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedMarginRankingLoss(margin=0.5, anomaly_weight=args.anomaly_weight)
    
    # Training loop
    print("\n🚀 Starting training...")
    print()
    
    best_anomaly_acc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, overall={val_metrics['overall_acc']:.4f}")
        print(f"         normal={val_metrics['normal_acc']:.4f} ({val_metrics['normal_total']} cases)")
        print(f"         anomaly={val_metrics['anomaly_acc']:.4f} ({val_metrics['anomaly_total']} cases) ⭐")
        
        # Save best model based on anomaly accuracy
        if val_metrics['anomaly_acc'] > best_anomaly_acc:
            best_anomaly_acc = val_metrics['anomaly_acc']
            best_epoch = epoch + 1
            
            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_name = f"ranker_v3_w{int(args.anomaly_weight)}"
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pt")
            
            # Save metadata
            metadata = {
                'model_version': 'v3',
                'created': datetime.now(timezone.utc).isoformat(),
                'training_pairs': len(normal_train) + len(anomaly_train),
                'validation_pairs': len(normal_val) + len(anomaly_val),
                'normal_train': len(normal_train),
                'anomaly_train': len(anomaly_train),
                'anomaly_oversample_factor': int(args.anomaly_weight),
                'best_epoch': best_epoch,
                'best_anomaly_accuracy': best_anomaly_acc,
                'best_normal_accuracy': val_metrics['normal_acc'],
                'best_overall_accuracy': val_metrics['overall_acc'],
                'hyperparameters': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'margin': 0.5,
                    'anomaly_weight': args.anomaly_weight
                },
                'architecture': 'MLP (512→256→64→1)',
                'loss_function': 'WeightedMarginRankingLoss'
            }
            
            with (MODEL_DIR / f"{model_name}_metadata.json").open('w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✅ Saved best model (anomaly_acc={best_anomaly_acc:.4f})")
        
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best anomaly accuracy: {best_anomaly_acc:.4f}")
    print()
    model_name = f"ranker_v3_w{int(args.anomaly_weight)}"
    print("📁 Model saved:")
    print(f"   {MODEL_DIR / model_name}.pt")
    print(f"   {MODEL_DIR / model_name}_metadata.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

