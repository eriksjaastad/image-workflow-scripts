# AI Training Playbook
**Status:** Active


Last Updated: 2025-10-23
Audience: AI agents training/validating models for selection and crop.

## Philosophy
- Training data integrity > model speed. Validate before training.
- Prefer SQLite v3 decisions + sidecar flow for ground truth.
- Keep costs predictable; warn at ~10k tokens/day [[memory:9146206]].

## Datasets
- Selection decisions: from SQLite v3 (`data/training/ai_training_decisions/{project}.db`).
- Crop ground truth: `final_crop_coords` joined via `.decision` lifecycle.
- Embeddings cache: `data/ai_data/cache/` (CLIP or similar feature extraction).
- Legacy CSVs retained for backfill only.

## Integrity Checks (must-pass)
- Row counts per project; non-empty `group_id`, `project_id`.
- `user_action` ∈ {approve,crop,reject}; `image_width,height` > 0.
- Crop coords normalized [0,1], with x1<x2, y1<y2.
- No orphaned “crop” rows without `final_crop_coords` in final training set.
- Consistent filename-only entries; no absolute paths.

## Inspection Report Pattern (BEFORE bulk ops)
- Show first 50 rows with `repr()` values; counts by `user_action`, by project; invalid row list.
- Save to `data/daily_summaries/ai_training_inspection_<ts>.json`.

## Workflows

### 1) Feature Extraction
```bash
python3 scripts/ai/extract_embeddings.py --input mojo3/ --out data/ai_data/cache/
```
- Verify: presence of `processed_images.jsonl` and `.npy` files; sample 20.

### 2) Train Ranking Model
```bash
python3 scripts/ai/train_ranker_model.py --db data/training/ai_training_decisions/mojo3.db \
  --embeddings data/ai_data/cache/ --out data/ai_data/models/ranker_vX.pt
```
- Validate: selection accuracy on holdout; ensure it’s not “always highest stage”.

### 3) Train Crop Proposer
```bash
python3 scripts/ai/train_crop_model.py --db data/training/ai_training_decisions/mojo3.db \
  --out data/ai_data/models/crop_proposer_vX.pt
```
- Validate: IoU/center distance; analyze by stage.

### 4) Backfill (optional)
- Convert legacy CSVs to v3 SQLite; keep originals; write NEW `.db` files.

## Cost Controls
- Batch reads; avoid loading entire trees at once.
- For LLM-based labeling or audits, enforce 10k/day soft limit; pause if exceeded [[memory:9146206]].

## Deliverables
- Model file in `data/ai_data/models/` with `*_metadata.json` summarizing dataset sizes, metrics.
- A short report in `data/daily_summaries/` with integrity stats and validation.

## Links
- Schema Reference: `Documents/SCHEMA_REFERENCE.md`
- Technical KB (SQLite v3): `Documents/TECHNICAL_KNOWLEDGE_BASE.md`
- Training docs: `Documents/AI_TRAINING_DATA_STRUCTURE.md`
