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
    checkpoint = torch.load(model_path, map_location="cpu")
else:
    pass

if metadata_path.exists():
    with open(metadata_path) as f:
        meta = json.load(f)
