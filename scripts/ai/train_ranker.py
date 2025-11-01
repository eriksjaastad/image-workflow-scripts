#!/usr/bin/env python3
"""
Phase 2.3: Train Ranking Model
================================

Train AI to predict which image Erik will select from a group.

Uses:
- Input: CLIP embeddings (512-dim vectors)
- Training data: selection_only_log.csv (9,838 examples)
- Model: MLP neural network (512→256→64→1)
- Loss: Pairwise ranking loss (winner > losers)

Output: data/ai_data/models/ranker_v1.pt

USAGE:
------
  python scripts/ai/train_ranker.py

  --epochs 50       Number of training epochs (default: 30)
  --batch-size 32   Batch size (default: 16)
  --lr 0.001        Learning rate (default: 0.001)
  --val-split 0.1   Validation split (default: 0.1)
  --device mps      Device: mps, cuda, or cpu (default: auto)
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class ImageRankingDataset(Dataset):
    """Dataset for pairwise ranking: winner vs losers."""
    
    def __init__(self, examples: list[dict]):
        """
        examples: List of dicts with:
          - winner_emb: np.array (512,)
          - loser_embs: List[np.array] (each 512,)
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        winner = torch.from_numpy(ex['winner_emb']).float()
        
        # Randomly pick one loser for this training step
        loser_idx = np.random.randint(len(ex['loser_embs']))
        loser = torch.from_numpy(ex['loser_embs'][loser_idx]).float()
        
        return winner, loser


class RankingModel(nn.Module):
    """MLP that scores images (higher = better)."""
    
    def __init__(self, input_dim=512, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [256, 64]
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        
        # Output: single score
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: (batch, 512) embedding
        returns: (batch, 1) score
        """
        return self.net(x)


def load_embeddings_cache(cache_file: Path) -> dict[str, np.ndarray]:
    """Load embedding cache mapping image_path -> embedding."""
    cache = {}
    
    if not cache_file.exists():
        return cache
    
    with open(cache_file) as f:
        for line in f:
            data = json.loads(line)
            img_path = data['image_path']
            emb_file = Path(data['embedding_file'])
            
            if emb_file.exists():
                cache[img_path] = np.load(emb_file)
    
    return cache


def load_training_data(log_file: Path, embeddings: dict[str, np.ndarray]) -> list[dict]:
    """
    Load selection decisions and pair with embeddings.
    
    Returns list of examples:
      {
        'winner_emb': np.array (512,),
        'loser_embs': [np.array (512,), ...],
        'group_id': str
      }
    """
    examples = []
    skipped = 0
    skipped_reasons = {'missing_winner': 0, 'missing_losers': 0, 'parse_error': 0}
    
    
    with open(log_file) as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            winner_path = row['chosen_path'].strip()
            neg_paths_str = row['neg_paths'].strip()
            
            # Parse negative paths (JSON array)
            try:
                neg_paths = json.loads(neg_paths_str)
            except Exception:
                # Fallback: might be stored with escaped quotes
                try:
                    neg_paths = json.loads(neg_paths_str.replace('""', '"'))
                except Exception:
                    skipped += 1
                    skipped_reasons['parse_error'] += 1
                    continue
            
            # Try to match paths - embeddings use relative paths
            # Training log uses absolute paths, so extract just filename
            winner_name = Path(winner_path).name
            
            # Find embedding by filename match
            winner_emb = None
            for emb_path, emb in embeddings.items():
                if Path(emb_path).name == winner_name:
                    winner_emb = emb
                    break
            
            # Find loser embeddings
            loser_embs = []
            for neg_path in neg_paths:
                neg_name = Path(neg_path).name
                for emb_path, emb in embeddings.items():
                    if Path(emb_path).name == neg_name:
                        loser_embs.append(emb)
                        break
            
            # Skip if missing embeddings
            if winner_emb is None:
                skipped += 1
                skipped_reasons['missing_winner'] += 1
                continue
            
            if len(loser_embs) == 0:
                skipped += 1
                skipped_reasons['missing_losers'] += 1
                continue
            
            examples.append({
                'winner_emb': winner_emb,
                'loser_embs': loser_embs,
                'group_id': row.get('set_id', f"group_{len(examples)}")
            })
    
    return examples


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for winner, loser in dataloader:
        winner = winner.to(device)
        loser = loser.to(device)
        
        optimizer.zero_grad()
        
        # Score both
        winner_score = model(winner).squeeze()
        loser_score = model(loser).squeeze()
        
        # Margin ranking loss: winner should score higher
        loss = criterion(winner_score, loser_score, torch.ones_like(winner_score))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for winner, loser in dataloader:
            winner = winner.to(device)
            loser = loser.to(device)
            
            winner_score = model(winner).squeeze()
            loser_score = model(loser).squeeze()
            
            loss = criterion(winner_score, loser_score, torch.ones_like(winner_score))
            total_loss += loss.item()
            
            # Accuracy: did model rank winner higher?
            correct += (winner_score > loser_score).sum().item()
            total += winner.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train image ranking model")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--device', type=str, default='auto', help='Device: mps, cuda, cpu, auto')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    
    # Paths
    data_dir = Path("data")
    cache_file = data_dir / "ai_data" / "cache" / "processed_images.jsonl"
    log_file = data_dir / "training" / "selection_only_log.csv"
    model_dir = data_dir / "ai_data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    embeddings = load_embeddings_cache(cache_file)
    examples = load_training_data(log_file, embeddings)
    
    if len(examples) == 0:
        sys.exit(1)
    
    # Train/val split
    np.random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    
    # Create datasets
    train_dataset = ImageRankingDataset(train_examples)
    val_dataset = ImageRankingDataset(val_examples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = RankingModel().to(device)
    criterion = nn.MarginRankingLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            model_path = model_dir / "ranker_v1.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'train_examples': len(train_examples),
                'val_examples': len(val_examples),
            }, model_path)
            
    
    
    # Save metadata
    metadata = {
        'model_version': 'v1',
        'created': datetime.utcnow().isoformat() + 'Z',
        'training_examples': len(train_examples),
        'validation_examples': len(val_examples),
        'best_epoch': best_epoch,
        'best_val_accuracy': float(best_val_acc),
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'val_split': args.val_split,
        },
        'architecture': 'MLP (512→256→64→1)',
        'loss_function': 'MarginRankingLoss (margin=0.5)',
    }
    
    metadata_path = model_dir / "ranker_v1_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    


if __name__ == "__main__":
    main()

