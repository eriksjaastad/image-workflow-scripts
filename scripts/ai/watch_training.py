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

print("🤖 Training Monitor - Checking every 5 seconds...")
print("   Press Ctrl+C to stop\n")

last_epoch = 0
start_time = time.time()

try:
    while True:
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                epoch = checkpoint['epoch']
                val_acc = checkpoint['val_accuracy']
                val_loss = checkpoint['val_loss']
                
                if epoch > last_epoch:
                    print(f"⏱️  {mins:02d}:{secs:02d} | Epoch {epoch:2d}/30 | "
                          f"Val Acc: {val_acc:6.2%} | Val Loss: {val_loss:.4f}")
                    last_epoch = epoch
            except:
                pass
        else:
            if elapsed % 10 == 0:  # Print every 10 seconds if no model yet
                print(f"⏳ {mins:02d}:{secs:02d} | Loading data & starting first epoch...")
        
        # Check if training is complete
        if metadata_path.exists():
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            
            print("\n📊 Final Results:")
            print(f"   Best Epoch: {meta['best_epoch']}")
            print(f"   Best Val Accuracy: {meta['best_val_accuracy']:.2%}")
            print(f"   Training Examples: {meta['training_examples']}")
            print(f"   Validation Examples: {meta['validation_examples']}")
            
            if meta['best_val_accuracy'] >= 0.70:
                print("\n🎉 SUCCESS! Exceeded 70% target!")
            else:
                print("\n⚠️  Below 70% target - may need more data")
            
            print("\n💾 Model saved to: data/ai_data/models/ranker_v1.pt")
            break
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\n👋 Monitoring stopped. Training continues in background.")

