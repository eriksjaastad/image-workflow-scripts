# AI Training Guide

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** AI Training Team, Developers
**Estimated Reading Time:** 18 minutes

## Overview
This guide consolidates Phase 2 training documentation into a single how-to covering: environment setup, data preparation, feature extraction, model training, validation, and integration. It reflects the current implementation and fixes prior path inconsistencies.

## Quick Start
1) Activate environment
```bash
source .venv311/bin/activate
python --version  # 3.11+
```
2) Install dependencies
```bash
pip install torch torchvision torchaudio lightning accelerate open_clip_torch \
  opencv-python pillow numpy scipy pandas pyarrow tqdm
```
3) Verify Apple GPU
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
4) Compute embeddings (one-time per corpus)
```bash
python scripts/ai/compute_embeddings.py
```
5) Train ranker
```bash
python scripts/ai/train_ranker.py
# or: python scripts/ai/train_ranker_v2.py
# or: python scripts/ai/train_ranker_v3.py
```
6) Train crop proposer
```bash
python scripts/ai/train_crop_proposer.py
# or: python scripts/ai/train_crop_proposer_v2.py
```
7) Test models end-to-end
```bash
python scripts/ai/test_models.py
```

## Current Status
- Selections: 21,250
- Crops: 12,679
- Embeddings: 77,304
- Ranker v2 trained; v3 available (ensure path normalization)

## Critical Rules
- Projects never mix; pairs must be within the same project.
- Match embeddings by filename within project context.
- Validate each selection set before pair construction.

## Step-by-Step
### 1. Environment
- Python 3.11+, Apple Silicon with MPS
- Packages listed in Quick Start

### 2. Data Locations
- Selections: `data/training/selection_only_log.csv`
- Crops: `data/training/select_crop_log.csv`
- Embeddings cache/files: `data/ai_data/cache/` and `data/ai_data/embeddings/`

### 3. Feature Extraction
Run once per image corpus.
```bash
python scripts/ai/compute_embeddings.py
```
Verify cache JSONL and `.npy` files exist.

### 4. Ranker Training
Options:
- `scripts/ai/train_ranker.py` (baseline)
- `scripts/ai/train_ranker_v2.py` (validated subset)
- `scripts/ai/train_ranker_v3.py` (full dataset; fix path normalization first)

### 5. Crop Proposer Training
- `scripts/ai/train_crop_proposer.py` or `scripts/ai/train_crop_proposer_v2.py`
- Input: 12,679 crop examples

### 6. Testing
- `scripts/ai/test_models.py` — load ranker + cropper; test on sample groups

### 7. Integration
- Integrate models into `scripts/01_ai_assisted_reviewer.py` when ready.

## Best Practices & Lessons
- Validate data at collection time; add inline checks.
- Nightly validation: `scripts/ai/validate_training_data.py`.
- Keep embeddings cache and files in sync; verify after generation.

## Troubleshooting
- "No training data loaded" → check path normalization and filename+project matching.
- Missing embeddings → recompute via `compute_embeddings.py` and re-verify.
- Invalid crop dimensions → ensure logging uses actual image dimensions and normalization.

## Related Documents
- AI Training Reference: `Documents/ai/AI_TRAINING_REFERENCE.md`
- AI Project Plan: `Documents/AI_PROJECT_IMPLEMENTATION_PLAN.md`
- Data Structure Rules: `Documents/AI_TRAINING_DATA_STRUCTURE.md`
