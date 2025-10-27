#!/usr/bin/env python3
"""
Test: Did AI learn anomaly detection or just "pick highest stage"?

Checks cases where Erik chose a LOWER stage (overriding the usual rule).
Did the AI also pick the lower stage in those cases?
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

# Load model
model_path = Path("data/ai_data/models/ranker_v1.pt")
checkpoint = torch.load(model_path, map_location='cpu')

from train_ranker import RankingModel

model = RankingModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load embeddings
embeddings = {}
cache_file = Path("data/ai_data/cache/processed_images.jsonl")
print("[*] Loading embeddings...")
with open(cache_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        img_path = data['image_path']
        emb_file = Path(data['embedding_file'])
        if emb_file.exists():
            embeddings[img_path] = np.load(emb_file)

print(f"[*] Loaded {len(embeddings)} embeddings\n")

# Load training log
log_file = Path("data/training/selection_only_log.csv")

# Find cases where Erik chose LOWER stage
anomaly_cases = []

print("[*] Finding cases where you chose a LOWER stage...\n")

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        winner_path = row['chosen_path'].strip()
        neg_paths_str = row['neg_paths'].strip()
        
        try:
            neg_paths = json.loads(neg_paths_str.replace('""', '"'))
        except:
            continue
        
        # Extract stages
        winner_name = Path(winner_path).name
        
        # Parse stage from filename
        def get_stage_num(filename):
            if 'stage3' in filename:
                return 3
            elif 'stage2' in filename:
                return 2
            elif 'stage1.5' in filename:
                return 1.5
            elif 'stage1' in filename:
                return 1
            return 0
        
        winner_stage = get_stage_num(winner_name)
        
        # Check if any loser has HIGHER stage
        for neg_path in neg_paths:
            neg_name = Path(neg_path).name
            neg_stage = get_stage_num(neg_name)
            
            if neg_stage > winner_stage:
                # FOUND ONE! Erik chose lower stage
                # Find embeddings
                winner_emb = None
                loser_emb = None
                
                for emb_path, emb in embeddings.items():
                    if Path(emb_path).name == winner_name:
                        winner_emb = emb
                    if Path(emb_path).name == neg_name:
                        loser_emb = emb
                
                if winner_emb is not None and loser_emb is not None:
                    anomaly_cases.append({
                        'erik_choice': winner_name,
                        'erik_stage': winner_stage,
                        'rejected': neg_name,
                        'rejected_stage': neg_stage,
                        'winner_emb': winner_emb,
                        'loser_emb': loser_emb
                    })

print(f"[*] Found {len(anomaly_cases)} cases where you chose LOWER stage\n")
print("="*70)
print("TEST: Did AI also detect these anomalies?")
print("="*70)

if len(anomaly_cases) == 0:
    print("\n⚠️  No cases found where you chose a lower stage!")
    print("   (This might mean the pattern is even stronger than expected)")
else:
    # Test AI on these anomaly cases
    ai_correct = 0
    ai_wrong = 0
    
    print(f"\nTesting AI on {len(anomaly_cases)} anomaly cases:\n")
    
    for i, case in enumerate(anomaly_cases[:20]):  # Show first 20
        winner_emb = torch.from_numpy(case['winner_emb']).float().unsqueeze(0)
        loser_emb = torch.from_numpy(case['loser_emb']).float().unsqueeze(0)
        
        with torch.no_grad():
            winner_score = model(winner_emb).item()
            loser_score = model(loser_emb).item()
        
        ai_picked_winner = winner_score > loser_score
        
        if ai_picked_winner:
            ai_correct += 1
            result = "✅ CORRECT"
        else:
            ai_wrong += 1
            result = "❌ WRONG"
        
        print(f"{i+1:2d}. {result}")
        print(f"    Erik:     {case['erik_choice'][:50]:<50} (stage {case['erik_stage']}) ← YOUR CHOICE")
        print(f"    Rejected: {case['rejected'][:50]:<50} (stage {case['rejected_stage']}) ← Higher stage!")
        print(f"    AI scores: Your choice={winner_score:.3f}, Rejected={loser_score:.3f}")
        print()
    
    if len(anomaly_cases) > 20:
        print(f"... and {len(anomaly_cases) - 20} more cases")
    
    print("="*70)
    print("RESULTS:")
    print("="*70)
    print(f"AI correctly detected anomaly: {ai_correct}/{len(anomaly_cases)} ({ai_correct/len(anomaly_cases)*100:.1f}%)")
    print(f"AI failed to detect anomaly:   {ai_wrong}/{len(anomaly_cases)} ({ai_wrong/len(anomaly_cases)*100:.1f}%)")
    
    if ai_correct / len(anomaly_cases) > 0.5:
        print("\n✅ AI learned anomaly detection! Not just picking highest stage.")
    else:
        print("\n⚠️  AI might be overfitting to stage numbers.")

