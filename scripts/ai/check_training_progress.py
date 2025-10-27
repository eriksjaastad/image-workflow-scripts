#!/usr/bin/env python3
"""
Quick script to monitor training progress.
Reads the model file and shows current status.
"""

import json
from pathlib import Path

import torch

model_path = Path("data/ai_data/models/ranker_v1.pt")
metadata_path = Path("data/ai_data/models/ranker_v1_metadata.json")

if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    print("✅ Model exists!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Accuracy: {checkpoint['val_accuracy']:.2%}")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Train examples: {checkpoint['train_examples']}")
    print(f"   Val examples: {checkpoint['val_examples']}")
else:
    print("⏳ Training in progress... Model not saved yet.")
    print("   (Model saves after first epoch with good validation accuracy)")

if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    print("\n📊 Final Results:")
    print(f"   Best Epoch: {meta['best_epoch']}")
    print(f"   Best Val Accuracy: {meta['best_val_accuracy']:.2%}")
    print(f"   Training Time: {meta['created']}")

