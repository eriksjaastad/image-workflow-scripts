# AI Training Reference

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers, AI Training Team
**Estimated Reading Time:** 12 minutes

## Data Structure Specifications
- Selection log: `data/training/selection_only_log.csv`
- Crop log: `data/training/select_crop_log.csv`
- Embeddings: `data/ai_data/embeddings/*.npy`; cache metadata `data/ai_data/cache/processed_images.jsonl`

### Selection Log Schema
```csv
session_id,set_id,chosen_path,neg_paths,timestamp
```

### Crop Log Schema
```csv
session_id,directory,chosen_path,crop_x1,crop_y1,crop_x2,crop_y2,timestamp
```

## Project Boundary Rules
- Extract `project_id` from path segments (e.g., `mojo1`, `mojo2`).
- Validate set integrity: all images in a set must share the same `project_id`.
- Match embeddings by `(project_id, filename)` to handle path differences.

## APIs and Scripts
- Feature extraction: `scripts/ai/compute_embeddings.py`
- Data extraction: `scripts/ai/extract_project_training.py`
- Anomaly analysis: `scripts/ai/analyze_anomalies.py`
- Rankers: `scripts/ai/train_ranker.py`, `train_ranker_v2.py`, `train_ranker_v3.py`
- Crop proposers: `scripts/ai/train_crop_proposer.py`, `train_crop_proposer_v2.py`
- Validation: `scripts/ai/validate_training_data.py`
- Testing: `scripts/ai/test_models.py`

## Troubleshooting Matrix
- Missing embeddings → recompute; verify cache vs disk counts.
- Path mismatches → normalize and match by project+filename.
- Invalid crop coords → ensure dimensions logged and normalization correct.

## Known Issues
- Some historical docs referenced `scripts/ai_training/*`; use `scripts/ai/*` equivalents.
- Planned scripts (not yet implemented):
  - `scripts/ai/generate_automation_decisions.py`
  - `scripts/07_automation_reviewer.py`

## Related
- Training Guide: `Documents/ai/AI_TRAINING_GUIDE.md`
- Data Structure Rules: `Documents/AI_TRAINING_DATA_STRUCTURE.md`
- Implementation Plan: `Documents/AI_PROJECT_IMPLEMENTATION_PLAN.md`
