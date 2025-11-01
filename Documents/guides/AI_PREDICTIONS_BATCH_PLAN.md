# AI Predictions Batch Processing Plan

**Generated:** 2025-11-01
**Purpose:** Run Phase 1A AI predictions across all historical projects

## Dataset Inventory

### Zip Files in `/Volumes/T7Shield/Eros/original/`

| Project ID       | Zip File             | Image Count | Has Database? | Status               |
| ---------------- | -------------------- | ----------- | ------------- | -------------------- |
| 1010             | 1010.zip             | 2,912       | ❌            | ⏳ Needs predictions |
| 1011             | 1011.zip             | 4,666       | ✅            | ⏳ Needs predictions |
| 1012             | 1012.zip             | 5,680       | ✅            | ⏳ Needs predictions |
| 1013             | 1013.zip             | 2,882       | ✅            | ⏳ Needs predictions |
| 1100             | 1100.zip             | 8,904       | ✅            | ⏳ Needs predictions |
| 1101_Hailey      | 1101.zip             | 8,904       | ✅            | ⏳ Needs predictions |
| 1102             | 1102.zip             | 2,968       | ❌            | ⏳ Needs predictions |
| Aiko_raw         | Aiko_raw.zip         | 1,050       | ✅            | ⏳ Needs predictions |
| Patricia         | Average Patricia.zip | 255         | ❌            | ⏳ Needs predictions |
| Eleni            | Eleni_raw.zip        | 5,816       | ✅            | ⏳ Needs predictions |
| Kiara_Slender    | Slender Kiara.zip    | 5,796       | ✅            | ⏳ Needs predictions |
| agent-1001       | agent-1001.zip       | 2,063       | ✅            | ⏳ Needs predictions |
| agent-1002       | agent-1002.zip       | 2,057       | ✅            | ⏳ Needs predictions |
| agent-1003       | agent-1003.zip       | 2,053       | ✅            | ⏳ Needs predictions |
| dalia            | dalia.zip            | 98          | ❌            | ⏳ Needs predictions |
| jmlimages-random | jmlimages-random.zip | 26,690      | ✅            | ⏳ Needs predictions |
| mixed-0919       | mixed-0919.zip       | 852         | ❌            | ⏳ Needs predictions |
| mojo1            | mojo1.zip            | 38,366      | ✅            | ⏳ Needs predictions |
| mojo2            | mojo2.zip            | 35,870      | ✅            | ⏳ Needs predictions |
| mojo3            | mojo3/ (dir)         | 19,406      | ✅            | ✅ Already done      |
| tattersail-0918  | tattersail-0918.zip  | 13,928      | ✅            | ⏳ Needs predictions |

**Total images:** ~202,000 images across 21 projects
**Already processed:** mojo3 (19,406 images)
**Remaining:** ~182,000 images across 20 projects

## AI Models

**Ranker:** `data/ai_data/models/ranker_v4.pt` (or v3_w10 as fallback)

- Architecture: RankingModel (512 → 256 → 64 → 1)
- Purpose: Select best image from group
- Output: `ai_selected_index`, `ai_confidence`

**Crop Proposer:** `data/ai_data/models/crop_proposer_v3.pt` ⭐ **LATEST**

- Architecture: CropProposer (514 → 256 → 128 → 4)
- Input: CLIP embedding (512) + dimensions (2)
- Output: `ai_crop_coords` [x1, y1, x2, y2] normalized [0-1]
- Version: 3 (current production model)

**Embeddings:** CLIP ViT-B/32 (OpenCLIP)

- Cache: `data/ai_data/cache/processed_images.jsonl`
- Storage: `data/ai_data/embeddings/{hash}.npy`

## Batch Processing Strategy

### Phase 1A: AI Predictions Only

For each project:

1. Extract zip to temporary directory (if zip)
2. Run AI predictions (ranker + crop proposer)
3. Store results in `data/training/ai_training_decisions/{project_id}.db`
4. Clean up temporary directory
5. Log progress to `data/ai_data/batch_predictions_log.jsonl`

### Estimated Runtime

**Per-image processing time:** ~0.15-0.2 seconds

- CLIP embedding: ~0.1s
- Ranker inference: ~0.02s
- Crop proposer inference: ~0.02s
- Database write: ~0.01s

**Total estimated runtime:**

- 182,000 images × 0.2s = ~36,400 seconds
- **~10 hours** for full batch
- **Can be paused/resumed** between projects

### Projects by Size (recommended order)

**Small (< 3,000 images):** ~1 hour total

- dalia (98 images) - 20 seconds
- Patricia (255 images) - 1 minute
- mixed-0919 (852 images) - 3 minutes
- agent-1003 (2,053 images) - 7 minutes
- agent-1002 (2,057 images) - 7 minutes
- agent-1001 (2,063 images) - 7 minutes
- 1013 (2,882 images) - 10 minutes
- 1010 (2,912 images) - 10 minutes
- 1102 (2,968 images) - 10 minutes

**Medium (3,000-10,000 images):** ~4 hours total

- 1011 (4,666 images) - 15 minutes
- Eleni (5,816 images) - 20 minutes
- 1012 (5,680 images) - 19 minutes
- Kiara_Slender (5,796 images) - 19 minutes
- 1100 (8,904 images) - 30 minutes
- 1101_Hailey (8,904 images) - 30 minutes

**Large (10,000+ images):** ~5 hours total

- tattersail-0918 (13,928 images) - 47 minutes
- jmlimages-random (26,690 images) - 1.5 hours
- mojo2 (35,870 images) - 2 hours
- mojo1 (38,366 images) - 2.1 hours

## Batch Script Features

The batch script will:

- ✅ Process projects in size order (small → large)
- ✅ Extract zips to `/tmp/ai_predictions_batch/{project_id}/`
- ✅ Reuse existing embeddings cache when possible
- ✅ Log progress after each project
- ✅ Create/update databases automatically
- ✅ Clean up temporary files after each project
- ✅ Resume from last completed project if interrupted
- ✅ Provide detailed progress output
- ✅ Estimate remaining time
- ✅ Save summary report at completion

## Output Files

For each project:

- Database: `data/training/ai_training_decisions/{project_id}.db`
- Log entry: `data/ai_data/batch_predictions_log.jsonl`

Final report: `data/ai_data/batch_predictions_summary_TIMESTAMP.json`

## Safety Notes

**This is Phase 1A only:**

- ✅ Creates AI predictions (what AI would have suggested)
- ❌ Does NOT create user ground truth (Phase 1B needed for that)
- ✅ Read-only for source images
- ✅ Creates NEW databases/records only
- ✅ Never modifies existing files

**To get complete training data:**

1. Run this batch script (Phase 1A)
2. Run Phase 1B separately for each project with final images
3. Run Phase 2 to merge temp databases (if needed)

## Next Steps

1. Run batch script: `bash scripts/ai/run_batch_predictions.sh`
2. Monitor progress in terminal
3. Check summary report when complete
4. Validate databases: `python scripts/ai/validate_all_databases.py`
