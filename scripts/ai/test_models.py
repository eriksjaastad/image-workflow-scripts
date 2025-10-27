#!/usr/bin/env python3
"""
Test and Compare Ranking Models

Purpose: Evaluate ranker v2 and v3 performance on validation sets,
with special focus on anomaly cases.

Usage:
    python scripts/ai/test_models.py
    
Output:
    - Console output with detailed metrics
    - data/ai_data/logs/model_evaluation.json
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Paths
PROJECT_ROOT = Path("/Users/eriksjaastad/projects/Eros Mate")
SELECTION_LOG = PROJECT_ROOT / "data/training/selection_only_log.csv"
ANOMALY_CSV = PROJECT_ROOT / "data/training/anomaly_cases.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data/ai_data/cache/processed_images.jsonl"
MODEL_DIR = PROJECT_ROOT / "data/ai_data/models"
LOG_DIR = PROJECT_ROOT / "data/ai_data/logs"

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"


class RankingModel(nn.Module):
    """MLP that scores images (512 → 256 → 64 → 1)."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def load_embeddings_cache() -> Tuple[Dict, Dict]:
    """Load embeddings cache."""
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
    return cache, filename_cache


def normalize_path(path: str) -> str:
    """Normalize path."""
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


def load_anomaly_set() -> set:
    """Load anomaly cases."""
    print("📂 Loading anomaly cases...")
    anomalies = set()
    
    with ANOMALY_CSV.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chosen = normalize_path(row['chosen_path'])
            rejected_str = row['rejected_paths'].replace("'", '"')
            rejected = json.loads(rejected_str)
            for neg_path in rejected:
                anomalies.add((chosen, normalize_path(neg_path)))
    
    print(f"   Loaded {len(anomalies):,} anomaly pairs")
    return anomalies


def load_embedding(path: str, cache: Dict, filename_cache: Dict) -> np.ndarray:
    """Load embedding with filename fallback."""
    embeddings_dir = PROJECT_ROOT / "data/ai_data/embeddings"
    
    # Try exact path
    if path in cache:
        hash_val = cache[path]
    else:
        filename = Path(path).name
        if filename not in filename_cache:
            raise ValueError(f"No embedding for {path}")
        hash_val = filename_cache[filename]
    
    emb_file = embeddings_dir / f"{hash_val}.npy"
    if not emb_file.exists():
        raise ValueError(f"Embedding file missing: {emb_file}")
    
    return np.load(emb_file)


def load_validation_data(cache: Dict, filename_cache: Dict, anomaly_set: set):
    """Load validation sets (normal and anomaly)."""
    print("\n📂 Loading validation data...")
    
    normal_pairs = []
    anomaly_pairs = []
    skipped = 0
    
    with SELECTION_LOG.open('r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            chosen_path = row['chosen_path']
            neg_paths = json.loads(row['neg_paths'])
            
            chosen_norm = normalize_path(chosen_path)
            neg_norms = [normalize_path(p) for p in neg_paths]
            
            # Check embeddings exist
            try:
                chosen_emb = load_embedding(chosen_norm, cache, filename_cache)
            except ValueError:
                skipped += 1
                continue
            
            try:
                neg_embs = [load_embedding(p, cache, filename_cache) for p in neg_norms]
            except ValueError:
                skipped += 1
                continue
            
            # Create pairs
            for neg_norm, neg_emb in zip(neg_norms, neg_embs):
                is_anomaly = (chosen_norm, neg_norm) in anomaly_set
                
                pair = {
                    'winner': chosen_norm,
                    'loser': neg_norm,
                    'winner_emb': chosen_emb,
                    'loser_emb': neg_emb
                }
                
                if is_anomaly:
                    anomaly_pairs.append(pair)
                else:
                    normal_pairs.append(pair)
    
    # Take last 10% as validation
    val_size = int(len(normal_pairs) * 0.1)
    normal_val = normal_pairs[-val_size:]
    anomaly_val = anomaly_pairs[-int(len(anomaly_pairs) * 0.2):]  # 20% for anomalies since they're rare
    
    print(f"   Normal validation: {len(normal_val):,} pairs")
    print(f"   Anomaly validation: {len(anomaly_val):,} pairs")
    print(f"   Skipped: {skipped:,}")
    
    return normal_val, anomaly_val


def evaluate_model(model: nn.Module, pairs: List[Dict], desc: str) -> Dict:
    """Evaluate model on pairs."""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for pair in pairs:
            winner_emb = torch.from_numpy(pair['winner_emb']).float().to(device)
            loser_emb = torch.from_numpy(pair['loser_emb']).float().to(device)
            
            winner_score = model(winner_emb).item()
            loser_score = model(loser_emb).item()
            
            if winner_score > loser_score:
                correct += 1
    
    accuracy = correct / len(pairs) if pairs else 0
    return {
        'description': desc,
        'total_pairs': len(pairs),
        'correct': correct,
        'accuracy': accuracy
    }


def main():
    print("=" * 70)
    print("MODEL EVALUATION - Ranker v2 vs v3")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # Load data
    cache, filename_cache = load_embeddings_cache()
    anomaly_set = load_anomaly_set()
    normal_val, anomaly_val = load_validation_data(cache, filename_cache, anomaly_set)
    
    results = {}
    
    # Test ranker v2
    print("\n" + "=" * 70)
    print("TESTING RANKER V2")
    print("=" * 70)
    
    v2_path = MODEL_DIR / "ranker_v2.pt"
    if v2_path.exists():
        model_v2 = RankingModel().to(device)
        model_v2.load_state_dict(torch.load(v2_path, map_location=device))
        
        v2_normal = evaluate_model(model_v2, normal_val, "Ranker v2 - Normal cases")
        v2_anomaly = evaluate_model(model_v2, anomaly_val, "Ranker v2 - Anomaly cases")
        
        print("\n✅ Ranker v2 Results:")
        print(f"   Normal:  {v2_normal['correct']:,}/{v2_normal['total_pairs']:,} = {v2_normal['accuracy']:.1%}")
        print(f"   Anomaly: {v2_anomaly['correct']:,}/{v2_anomaly['total_pairs']:,} = {v2_anomaly['accuracy']:.1%} ⭐")
        
        results['ranker_v2'] = {
            'normal': v2_normal,
            'anomaly': v2_anomaly,
            'overall': {
                'accuracy': (v2_normal['correct'] + v2_anomaly['correct']) / (v2_normal['total_pairs'] + v2_anomaly['total_pairs'])
            }
        }
    else:
        print(f"⚠️  Ranker v2 not found: {v2_path}")
        results['ranker_v2'] = None
    
    # Test ranker v3
    print("\n" + "=" * 70)
    print("TESTING RANKER V3")
    print("=" * 70)
    
    v3_path = MODEL_DIR / "ranker_v3_w10.pt"
    if v3_path.exists():
        model_v3 = RankingModel().to(device)
        model_v3.load_state_dict(torch.load(v3_path, map_location=device))
        
        v3_normal = evaluate_model(model_v3, normal_val, "Ranker v3 - Normal cases")
        v3_anomaly = evaluate_model(model_v3, anomaly_val, "Ranker v3 - Anomaly cases")
        
        print("\n✅ Ranker v3 Results:")
        print(f"   Normal:  {v3_normal['correct']:,}/{v3_normal['total_pairs']:,} = {v3_normal['accuracy']:.1%}")
        print(f"   Anomaly: {v3_anomaly['correct']:,}/{v3_anomaly['total_pairs']:,} = {v3_anomaly['accuracy']:.1%} ⭐")
        
        results['ranker_v3'] = {
            'normal': v3_normal,
            'anomaly': v3_anomaly,
            'overall': {
                'accuracy': (v3_normal['correct'] + v3_anomaly['correct']) / (v3_normal['total_pairs'] + v3_anomaly['total_pairs'])
            }
        }
    else:
        print(f"⚠️  Ranker v3 not found: {v3_path}")
        results['ranker_v3'] = None
    
    # Comparison
    if results['ranker_v2'] and results['ranker_v3']:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        v2_anom = results['ranker_v2']['anomaly']['accuracy']
        v3_anom = results['ranker_v3']['anomaly']['accuracy']
        improvement = (v3_anom - v2_anom) * 100
        
        print("\n📊 Anomaly Accuracy:")
        print(f"   v2: {v2_anom:.1%}")
        print(f"   v3: {v3_anom:.1%}")
        print(f"   Improvement: {improvement:+.1f} percentage points")
        
        print("\n📊 Overall Accuracy:")
        print(f"   v2: {results['ranker_v2']['overall']['accuracy']:.1%}")
        print(f"   v3: {results['ranker_v3']['overall']['accuracy']:.1%}")
        
        if v3_anom > v2_anom:
            print("\n🎉 Ranker v3 is BETTER at quality judgment!")
        else:
            print("\n⚠️  Ranker v2 performed better on anomalies")
    
    # Save results
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOG_DIR / "model_evaluation.json"
    
    with results_path.open('w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

