# AI Training Scripts

**READ-ONLY SCRIPTS** - Never modify images, YAML files, or any existing content.

## Safety Guarantees

### ✅ What These Scripts Do (SAFE)
- **Read** images to extract features
- **Create NEW** sidecar files (`.embedding.npy`, etc.)
- **Create NEW** files in `data/ai_data/`
- **Never modify** source images or companion files

### ❌ What These Scripts NEVER Do
- Modify existing images
- Modify YAML companion files
- Modify text content files
- Delete or move production files
- Automated operations on production content

## Scripts

### `compute_training_features.py`
**Purpose:** Extract CLIP embeddings from training images (read-only)

**Usage:**
```bash
# Process all training images
python scripts/ai/compute_training_features.py

# Process limited set (testing)
python scripts/ai/compute_training_features.py --max-images 100

# Force re-process cached images
python scripts/ai/compute_training_features.py --force
```

**What it does:**
1. Scans `data/training_data/` for images (read-only)
2. Extracts CLIP embeddings using Apple GPU
3. Saves embeddings to `data/ai_data/embeddings/{hash}.npy` (NEW files)
4. Caches processed files in `data/ai_data/cache/processed_files.jsonl`

**Performance:** ~5-10 images/sec on M4 Pro (MPS), ~2 hours for 14k images

**Output:** NEW files only (embeddings, cache)

---

### `train_crop_proposer.py` (coming next)
**Purpose:** Train crop suggestion model from your decisions

**Usage:**
```bash
python scripts/ai/train_crop_proposer.py
```

**What it does:**
1. Reads training logs (read-only)
2. Loads embeddings (read-only)
3. Trains PyTorch model
4. Saves model to `data/ai_data/models/crop_proposer_v1.pth` (NEW file)

---

## Directory Structure

```
data/ai_data/
├── embeddings/          # CLIP embeddings (.npy files)
│   └── {hash}.npy
├── models/              # Trained PyTorch models
│   └── crop_proposer_v1.pth
├── cache/               # Processing cache
│   └── processed_files.jsonl
└── logs/                # Training logs
    └── training_log.jsonl
```

## Sidecar Files (Removable Before Delivery)

All sidecar files are **NEW files** that can be deleted anytime:
- `.embedding.npy` - CLIP embeddings
- `.phash` - Perceptual hashes
- `.saliency.npy` - Saliency maps (optional)
- `.hands.json` - Hand keypoints (optional)

**Before project delivery:** Delete all sidecar files
```bash
# Remove all AI sidecar files (safe - NEW files only)
find content/ -name "*.embedding.npy" -delete
find content/ -name "*.phash" -delete
```

---

## Integration with Multi Crop Tool

**Flag:** `--ai-assist`

**Usage:**
```bash
python scripts/04_multi_crop_tool.py --ai-assist
```

**What happens:**
1. Tool loads image (normal)
2. AI draws suggested crop box (dotted green line)
3. You decide:
   - ✅ Press Enter → Accept suggestion (logs as "approved")
   - 🔧 Adjust handles → Modify suggestion (logs as "modified")
   - ❌ Mark delete → Reject suggestion (logs as "rejected")
4. AI learns from your corrections

**Safety:** Suggestions only - you still make final decision

---

## Continuous Learning

When `--ai-assist` is enabled:
- Logs your decisions to `data/ai_data/logs/suggestions.jsonl`
- Periodically retrain to improve suggestions
- Never automated - always requires your approval

---

## Performance Notes

**Apple M4 Pro with MPS:**
- Feature extraction: ~5-10 images/sec
- Model training: ~30 minutes for 14k examples
- Inference (suggestions): <100ms per image

**Memory Usage:**
- CLIP model: ~350MB GPU
- Feature cache: ~500MB for 14k images
- Trained model: ~10MB

