#!/usr/bin/env python3
"""
Live training monitor - shows progress every 5 seconds.
Press Ctrl+C to stop.
"""

import json
import time
from pathlib import Path

import torch

model_path = Path("data/ai_data/models/ranker_v1.pt")
metadata_path = Path("data/ai_data/models/ranker_v1_metadata.json")


last_epoch = 0
start_time = time.time()

try:
    while True:
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                epoch = checkpoint["epoch"]
                val_acc = checkpoint["val_accuracy"]
                val_loss = checkpoint["val_loss"]

                last_epoch = max(last_epoch, epoch)
            except Exception:
                pass
        elif elapsed % 10 == 0:  # Print every 10 seconds if no model yet
            pass

        # Check if training is complete
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)

            if meta["best_val_accuracy"] >= 0.70:
                pass
            else:
                pass

            break

        time.sleep(5)

except KeyboardInterrupt:
    pass
