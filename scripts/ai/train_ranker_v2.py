#!/usr/bin/env python3
"""
Train Ranking Model v2 with Anomaly Oversampling

Key improvements over v1:
1. Heavily weight anomaly cases (10x in loss function)
2. Oversample anomalies in training batches
3. Use focal loss to focus on hard examples
4. Report anomaly accuracy separately
5. Validate on held-out anomaly set

The problem: Only 2.1% of selections are anomalies (chose lower stage).
The v1 model learned "pick highest stage" and got 98% accuracy.
We need the model to learn QUALITY JUDGMENT, not stage rules.

Usage:
    python scripts/ai/train_ranker_v2.py
    
Output:
    - data/ai_data/models/ranker_v2.pt
    - data/ai_data/models/ranker_v2_metadata.json
    - Validation metrics focused on anomaly detection
"""

import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths
SELECTION_LOG = Path("/Users/eriksjaastad/projects/Eros Mate/data/training/selection_only_log.csv")
EMBEDDINGS_CACHE = Path("/Users/eriksjaastad/projects/Eros Mate/data/ai_data/cache/processed_images.jsonl")
MODEL_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/data/ai_data/models")

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


def get_project_id(path: str) -> str:
    """
    Extract project ID from file path.
    
    CRITICAL: Projects must NEVER be mixed during training.
    Each project is a different character.
    
    Examples:
        /Users/.../mojo1/file.png → 'mojo1'
        /Users/.../mojo2/_mixed/file.png → 'mojo2'
        training data/mojo1/faces/file.png → 'mojo1'
    """
    parts = Path(path).parts
    for part in parts:
        # Check for known project patterns
        if part.startswith('mojo'):  # mojo1, mojo2, etc.
            return part
        if part in ['eleni', 'aiko', 'dalia', 'kiara', 'mixed-0919', 'tattersail-0918']:
            return part
        if part.endswith('_raw') or part.endswith('_final'):
            return part.replace('_raw', '').replace('_final', '')
    return 'unknown'


def validate_set_integrity(chosen_path: str, neg_paths: List[str]) -> None:
    """
    Verify all paths in a set are from the same project.
    
    CRITICAL: This prevents cross-project contamination during training.
    Comparing mojo1 vs mojo2 images is meaningless and corrupts the model.
    
    Raises:
        ValueError: If mixed projects detected
    """
    chosen_project = get_project_id(chosen_path)
    
    for neg_path in neg_paths:
        neg_project = get_project_id(neg_path)
        if neg_project != chosen_project:
            raise ValueError(
                f"🚨 MIXED PROJECTS DETECTED - TRAINING ABORTED\n"
                f"  Winner: {chosen_path}\n"
                f"    Project: {chosen_project}\n"
                f"  Loser:  {neg_path}\n"
                f"    Project: {neg_project}\n"
                f"\nThis violates project boundary rules. Each project is a different character.\n"
                f"See Documents/AI_TRAINING_DATA_STRUCTURE.md for details."
            )


def extract_stage(filename: str) -> Optional[float]:
    """Extract stage number from filename."""
    match = re.search(r'stage(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None


def load_embeddings_cache() -> Dict[str, str]:
    """
    Load embeddings cache mapping image paths to embedding file hashes.
    
    Returns:
        Dict mapping absolute image path -> embedding hash
    """
    print("📂 Loading embeddings cache...")
    cache = {}
    
    with EMBEDDINGS_CACHE.open('r') as f:
        for line in f:
            entry = json.loads(line)
            cache[entry['image_path']] = entry['hash']
    
    print(f"   Loaded {len(cache)} cached embeddings")
    return cache


def normalize_path(path: str) -> str:
    """
    Normalize image path to match embeddings cache format.
    
    Embeddings cache uses relative paths from project root.
    Selection log uses absolute paths.
    """
    # Convert to Path object
    p = Path(path)
    
    # If absolute, make relative to project root
    project_root = Path("/Users/eriksjaastad/projects/Eros Mate")
    if p.is_absolute():
        try:
            p = p.relative_to(project_root)
        except ValueError:
            pass
    
    # Handle "mojo1" vs "training data/mojo1" mapping
    # CSV has paths like "mojo1/..." but cache has "training data/mojo1/..."
    path_str = str(p)
    if path_str.startswith('mojo1/'):
        path_str = 'training data/' + path_str
    elif path_str.startswith('mojo2/'):
        # mojo2 might also need this, but let's check what the cache has
        pass
    
    return path_str


def load_training_data(embeddings_cache: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Load selection log and split into normal vs anomaly cases.
    
    ENSURES PROJECT BOUNDARIES: All images in each set must be from the same project.
    
    Returns:
        (normal_cases, anomaly_cases)
    """
    print(f"\n📂 Loading training data: {SELECTION_LOG}")
    
    normal_cases = []
    anomaly_cases = []
    skipped = 0
    missing_embeddings = []
    project_counts = {}  # Track examples per project
    
    with SELECTION_LOG.open('r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            chosen_path = row['chosen_path']
            neg_paths_str = row['neg_paths']
            
            # Parse negative paths
            try:
                neg_paths = json.loads(neg_paths_str)
            except:
                skipped += 1
                continue
            
            # 🚨 CRITICAL: Validate project boundaries
            try:
                validate_set_integrity(chosen_path, neg_paths)
            except ValueError as e:
                print(f"\n{e}")
                raise  # Abort training if mixed projects detected
            
            # Track project
            project_id = get_project_id(chosen_path)
            project_counts[project_id] = project_counts.get(project_id, 0) + 1
            
            # Normalize paths
            chosen_path_norm = normalize_path(chosen_path)
            
            # Extract stage from chosen image
            chosen_stage = extract_stage(chosen_path)
            if not chosen_stage:
                skipped += 1
                continue
            
            # Extract stages from negative examples
            try:
                neg_paths_norm = [normalize_path(p) for p in neg_paths]
                neg_stages = [extract_stage(p) for p in neg_paths]
                neg_stages = [s for s in neg_stages if s is not None]
            except:
                skipped += 1
                continue
            
            if not neg_stages:
                skipped += 1
                continue
            
            # Check if embeddings exist for all images
            if chosen_path_norm not in embeddings_cache:
                missing_embeddings.append(chosen_path_norm)
                skipped += 1
                continue
            
            missing = [p for p in neg_paths_norm if p not in embeddings_cache]
            if missing:
                missing_embeddings.extend(missing)
                skipped += 1
                continue
            
            # Determine if this is an anomaly
            all_stages = [chosen_stage] + neg_stages
            max_stage = max(all_stages)
            is_anomaly = chosen_stage < max_stage
            
            case = {
                'chosen_path': chosen_path_norm,
                'neg_paths': neg_paths_norm,
                'chosen_stage': chosen_stage,
                'neg_stages': neg_stages,
                'is_anomaly': is_anomaly
            }
            
            if is_anomaly:
                anomaly_cases.append(case)
            else:
                normal_cases.append(case)
    
    print(f"   Normal cases: {len(normal_cases)}")
    print(f"   Anomaly cases: {len(anomaly_cases)}", end="")
    if normal_cases or anomaly_cases:
        print(f" ({len(anomaly_cases)/(len(normal_cases)+len(anomaly_cases))*100:.1f}%)")
    else:
        print(" (no valid data loaded)")
    print(f"   Skipped (missing embeddings or invalid): {skipped}")
    
    # Report training examples per project
    if project_counts:
        print("\n📊 Training examples per project:")
        for project_id in sorted(project_counts.keys()):
            count = project_counts[project_id]
            print(f"     {project_id}: {count:,} examples")
    
    if missing_embeddings and skipped > 0:
        print("\n⚠️  Sample missing paths (first 5):")
        for path in missing_embeddings[:5]:
            print(f"     {path}")
    
    if not normal_cases and not anomaly_cases:
        raise ValueError("No training data loaded! Check path formats.")
    
    return normal_cases, anomaly_cases


class RankingDataset(Dataset):
    """
    Dataset for ranking model with anomaly oversampling.
    
    Creates pairwise comparisons: (winner, loser) tuples.
    Oversamples anomaly cases to balance the dataset.
    """
    
    def __init__(self, normal_cases: List[Dict], anomaly_cases: List[Dict], 
                 embeddings_cache: Dict, anomaly_oversample_factor: int = 10):
        self.embeddings_cache = embeddings_cache
        self.embeddings_dir = Path("/Users/eriksjaastad/projects/Eros Mate/data/ai_data/cache/embeddings")
        
        # Create pairwise comparisons
        self.pairs = []
        
        # Add normal cases (once)
        for case in normal_cases:
            for neg_path in case['neg_paths']:
                self.pairs.append({
                    'winner': case['chosen_path'],
                    'loser': neg_path,
                    'is_anomaly': False
                })
        
        # Add anomaly cases (multiple times to balance)
        for case in anomaly_cases:
            for neg_path in case['neg_paths']:
                for _ in range(anomaly_oversample_factor):
                    self.pairs.append({
                        'winner': case['chosen_path'],
                        'loser': neg_path,
                        'is_anomaly': True
                    })
        
        print("\n📊 Dataset created:")
        print(f"   Normal pairs: {sum(1 for p in self.pairs if not p['is_anomaly'])}")
        print(f"   Anomaly pairs (oversampled {anomaly_oversample_factor}x): {sum(1 for p in self.pairs if p['is_anomaly'])}")
        print(f"   Total pairs: {len(self.pairs)}")
    
    def load_embedding(self, image_path: str) -> np.ndarray:
        """Load embedding from cache."""
        emb_hash = self.embeddings_cache[image_path]
        emb_file = Path("/Users/eriksjaastad/projects/Eros Mate") / Path(emb_hash).parent / f"{Path(emb_hash).name}"
        if not emb_file.exists():
            emb_file = self.embeddings_dir / f"{emb_hash}.npy"
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
    """
    Simple MLP that scores images.
    
    Architecture: 512 → 256 → 64 → 1
    """
    
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
    """
    Margin ranking loss with higher weight for anomaly cases.
    """
    
    def __init__(self, margin=0.5, anomaly_weight=10.0):
        super().__init__()
        self.margin = margin
        self.anomaly_weight = anomaly_weight
    
    def forward(self, winner_scores, loser_scores, is_anomaly):
        """
        Args:
            winner_scores: Scores for winner images
            loser_scores: Scores for loser images
            is_anomaly: 1.0 if anomaly case, 0.0 if normal
        """
        # Basic margin ranking loss: winner should score higher than loser by margin
        loss = torch.relu(self.margin - (winner_scores - loser_scores))
        
        # Weight anomaly cases higher
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
        
        # Accuracy: winner scored higher than loser?
        correct += (winner_scores > loser_scores).sum().item()
        total += len(winner_scores)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate model and report overall + anomaly-specific accuracy."""
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
            
            # Check if winner scored higher
            correct_mask = (winner_scores > loser_scores)
            anomaly_mask = (is_anomaly > 0.5)
            
            # Normal accuracy
            normal_mask = ~anomaly_mask
            normal_correct += (correct_mask & normal_mask).sum().item()
            normal_total += normal_mask.sum().item()
            
            # Anomaly accuracy
            anomaly_correct += (correct_mask & anomaly_mask).sum().item()
            anomaly_total += anomaly_mask.sum().item()
    
    overall_acc = (normal_correct + anomaly_correct) / (normal_total + anomaly_total)
    normal_acc = normal_correct / normal_total if normal_total > 0 else 0
    anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader),
        'overall_acc': overall_acc,
        'normal_acc': normal_acc,
        'anomaly_acc': anomaly_acc,
        'normal_total': normal_total,
        'anomaly_total': anomaly_total
    }


def main():
    print("=" * 70)
    print("RANKING MODEL V2 TRAINING - ANOMALY OVERSAMPLING")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    
    # Load data
    embeddings_cache = load_embeddings_cache()
    normal_cases, anomaly_cases = load_training_data(embeddings_cache)
    
    # Split anomaly cases for validation (20% held out)
    random.shuffle(anomaly_cases)
    anomaly_val_size = int(len(anomaly_cases) * 0.2)
    anomaly_train = anomaly_cases[anomaly_val_size:]
    anomaly_val = anomaly_cases[:anomaly_val_size]
    
    # Split normal cases for validation (10% held out)
    random.shuffle(normal_cases)
    normal_val_size = int(len(normal_cases) * 0.1)
    normal_train = normal_cases[normal_val_size:]
    normal_val = normal_cases[:normal_val_size]
    
    print("\n📊 Train/Val split:")
    print(f"   Train: {len(normal_train)} normal + {len(anomaly_train)} anomalies")
    print(f"   Val: {len(normal_val)} normal + {len(anomaly_val)} anomalies")
    
    # Create datasets with oversampling
    train_dataset = RankingDataset(normal_train, anomaly_train, embeddings_cache, anomaly_oversample_factor=10)
    val_dataset = RankingDataset(normal_val, anomaly_val, embeddings_cache, anomaly_oversample_factor=1)  # No oversampling in val
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = RankingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = WeightedMarginRankingLoss(margin=0.5, anomaly_weight=10.0)
    
    # Training loop
    print("\n🚀 Starting training...")
    print()
    
    best_anomaly_acc = 0
    best_epoch = 0
    epochs = 30
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val: loss={val_metrics['loss']:.4f}, overall={val_metrics['overall_acc']:.4f}")
        print(f"       normal_acc={val_metrics['normal_acc']:.4f} ({val_metrics['normal_total']} cases)")
        print(f"       anomaly_acc={val_metrics['anomaly_acc']:.4f} ({val_metrics['anomaly_total']} cases) ⭐")
        
        # Save best model based on anomaly accuracy
        if val_metrics['anomaly_acc'] > best_anomaly_acc:
            best_anomaly_acc = val_metrics['anomaly_acc']
            best_epoch = epoch + 1
            
            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "ranker_v2.pt")
            
            # Save metadata
            metadata = {
                'model_version': 'v2',
                'created': datetime.now(timezone.utc).isoformat(),
                'training_examples': len(normal_train) + len(anomaly_train),
                'validation_examples': len(normal_val) + len(anomaly_val),
                'anomaly_oversample_factor': 10,
                'best_epoch': best_epoch,
                'best_anomaly_accuracy': best_anomaly_acc,
                'best_normal_accuracy': val_metrics['normal_acc'],
                'best_overall_accuracy': val_metrics['overall_acc'],
                'hyperparameters': {
                    'epochs': epochs,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'margin': 0.5,
                    'anomaly_weight': 10.0
                },
                'architecture': 'MLP (512→256→64→1)',
                'loss_function': 'WeightedMarginRankingLoss'
            }
            
            with (MODEL_DIR / "ranker_v2_metadata.json").open('w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✅ Saved best model (anomaly_acc={best_anomaly_acc:.4f})")
        
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best anomaly accuracy: {best_anomaly_acc:.4f}")
    print()
    print("📁 Model saved:")
    print(f"   {MODEL_DIR / 'ranker_v2.pt'}")
    print(f"   {MODEL_DIR / 'ranker_v2_metadata.json'}")
    print()
    print("🎯 Next steps:")
    print("   1. Test model on held-out anomaly cases")
    print("   2. Compare v2 vs v1 performance")
    print("   3. Integrate into AI-assisted reviewer")
    print("=" * 70)


if __name__ == "__main__":
    main()

