# Backfill Training Data - Complete Guide

**Status:** Active (Updated 2025-10-31)
**Audience:** Developers, Operators
**Estimated Reading Time:** 12 minutes

## Overview

This guide describes the **3-phase backfill process** for reconstructing complete training data (AI predictions + user ground truth) from historical projects. The process uses physical images as the source of truth, extracting crop coordinates via template matching.

**Key Principle:** Physical images are ALWAYS the source of truth. We NEVER copy AI predictions as user ground truth.

## When to Use

- **Scenario 1:** Reconstruct lost data (e.g., database not created during actual work)
- **Scenario 2:** Fill gaps in existing databases (missing AI predictions or user data)
- **Scenario 3:** Backfill AI predictions for historical projects (to measure AI improvement over time)

## The 3-Phase Process

### Phase 1A: Generate AI Predictions

**Purpose:** Run current AI models on original images to recreate what the AI would have predicted.

**Input:**
- Original images directory (e.g., `/Volumes/T7Shield/Eros/original/mojo3`)
- Trained AI models (`ranker_v3_w10.pt`, `crop_proposer_v2.pt`)

**Output:**
- Temporary database with AI predictions: `{project_id}_backfill_temp.db`
- Fields populated: `ai_selected_index`, `ai_crop_coords`, `ai_confidence`
- User fields NULL (will be filled in Phase 1B)

**Command:**
```bash
cd /Users/eriksjaastad/projects/image-workflow
source .venv311/bin/activate

python3 scripts/ai/backfill_project_phase1a_ai_predictions.py \
  --project-id mojo3 \
  --original-dir /Volumes/T7Shield/Eros/original/mojo3 \
  --output-db data/training/ai_training_decisions/mojo3_backfill_temp.db
```

**Runtime:** ~15-20 minutes for 6,000 groups (depends on GPU/CPU)

**What It Does:**
1. Groups original images using standard grouping logic
2. Loads CLIP embeddings (from cache or computes on-the-fly)
3. Runs ranker model to select best image
4. Runs crop proposer to suggest crop coordinates
5. Stores all AI predictions in temp database

---

### Phase 1B: Extract User Ground Truth

**Purpose:** Find user's actual selections and crop coordinates from physical cropped images.

**Input:**
- Temp database from Phase 1A (with AI predictions)
- Final cropped images directory (e.g., `mojo3/`)

**Output:**
- Updated temp database with BOTH AI predictions and user ground truth
- Fields populated: `user_selected_index`, `user_action`, `final_crop_coords`
- Match fields calculated: `selection_match`, `crop_match`

**Command:**
```bash
python3 scripts/ai/backfill_project_phase1b_user_data.py \
  --temp-db data/training/ai_training_decisions/mojo3_backfill_temp.db \
  --final-dir mojo3/ \
  --dry-run  # Remove --dry-run to actually run
```

**Runtime:** ~20 minutes for 6,000 groups (template matching is CPU intensive)

**What It Does:**
1. For each group in temp database:
   - Searches final directory for images from that group (recursive)
   - If found: Extracts crop coordinates via OpenCV template matching
   - If not found: Marks as `user_action = 'reject'` (user rejected entire group)
2. Determines user action:
   - If `final_crop_coords ≈ [0, 0, 1, 1]` (within 2% tolerance) → `'approve'`
   - If coordinates differ significantly → `'crop'`
   - If image not in final directory → `'reject'`
3. Calculates accuracy metrics:
   - `selection_match`: Did AI pick same image as user?
   - `crop_match`: Was AI's crop within 5% of user's crop?

**Important:** For `'reject'` actions, `user_selected_index` is set to AI's selection (database schema requires 0-3, not NULL).

---

### Phase 2: Merge with Real Database

**Purpose:** Intelligently merge temp database into production database without overwriting existing data.

**Input:**
- Temp database (complete from Phase 1A + 1B)
- Real production database (may have partial data)

**Output:**
- Updated production database with merged data

**Command:**
```bash
# DRY RUN FIRST! (highly recommended)
python3 scripts/ai/backfill_project_phase2_compare.py \
  --temp-db data/training/ai_training_decisions/mojo3_backfill_temp.db \
  --real-db data/training/ai_training_decisions/mojo3.db \
  --dry-run

# After reviewing dry run, run for real:
python3 scripts/ai/backfill_project_phase2_compare.py \
  --temp-db data/training/ai_training_decisions/mojo3_backfill_temp.db \
  --real-db data/training/ai_training_decisions/mojo3.db
```

**Runtime:** < 1 minute (pure database operations, no image processing)

**Merge Rules:**
1. **Match records by image filename** (not group_id - different workflows use different ID formats)
2. **Add new records:** Groups in temp but not in real → insert into real database
3. **Fill NULL fields:** Real database has NULL values, temp has data → fill the NULLs
4. **Update coordinates if they differ beyond 1% tolerance:** Temp coordinates (from physical images) override real database
5. **NEVER overwrite:**
   - Timestamps (`timestamp`, `crop_timestamp`)
   - Existing AI predictions (unless NULL)
   - Existing user data (unless NULL or coordinates differ beyond tolerance)

**Safety:**
- Automatic backup created before merge
- Dry-run mode shows exactly what will change
- All timestamp data preserved from real database

---

## Critical Rules

### 1. Physical Images Are Source of Truth

**For User Data:**
- User crop coordinates extracted via template matching from physical images
- If coordinates in database differ from physical images → physical images win

**For AI Data:**
- AI predictions generated by running models on original images
- If real database has AI predictions → keep them (don't overwrite with backfill)
- Only fill NULL AI fields, never replace existing predictions

### 2. User Action Semantics

**Three possible values:**
- `'approve'`: User selected an image, no crop needed (coords ≈ [0, 0, 1, 1])
- `'crop'`: User selected an image and cropped it (coords differ from full image)
- `'reject'`: User rejected entire group (no image in final directory)

**There is NO 'skip'** - this was a mistake in earlier versions. Either user accepted something or rejected everything.

### 3. Coordinate Tolerance

**When comparing coordinates:**
- **2% tolerance** for determining approve vs crop (full image detection)
- **1% tolerance** for coordinate comparison in Phase 2 (matching existing data)
- **5% tolerance** for crop_match calculation (AI performance metric)

**Why tolerances matter:**
- Template matching isn't pixel-perfect
- Small variations (±1-2px) don't indicate different crops
- Real crops differ significantly (>5% change)

### 4. Database Schema Constraints

**user_selected_index:**
- MUST be 0-3 (database CHECK constraint)
- For `'reject'` actions: set to AI's selection (not NULL, not -1)
- This is a compromise - the field is required but meaningless for rejected groups

## Real Example: mojo3 Backfill (Oct 31, 2025)

**Starting point:**
- 1,762 records in real database (from recent work with AI system)
- ~6,000 images in `mojo3/` final directory (from months of work)
- Need: Reconstruct complete dataset with both AI predictions and user ground truth

**Phase 1A results:**
- Processed 6,468 groups from original images
- Generated AI predictions for all groups
- Runtime: ~15 minutes
- Output: temp database with 6,468 records (AI data only)

**Phase 1B results:**
- Found 4,657 images in final directory
- Extracted crop coordinates for all 4,657
- Identified 1,811 rejected groups (not in final)
- Breakdown:
  - 1,539 approve (full image kept)
  - 3,118 crop (actually cropped)
  - 1,811 reject (not in final)
- Runtime: ~20 minutes

**Phase 2 results:**
- Added 4,706 new records to real database
- Updated 4,954 existing records (filled NULL fields)
- Corrected 0 coordinates (all matched within 1% tolerance)
- Preserved all 1,762 original timestamps
- Runtime: < 1 minute
- **Final database: 6,468 total records**

**AI Performance Metrics:**
- Selection accuracy: 53.6% (AI picked same image as user)
- Crop accuracy: 4.3% (AI's crop within 5% of user's)
- This data enables model improvement!

## Troubleshooting

### "No images found in original directory"
- Check path is correct
- Verify images exist and have valid extensions (.png, .jpg, .jpeg)
- Original directory should be searched recursively

### "Template matching confidence too low"
- Image might have been edited after cropping
- Different resolution/compression
- Try lowering confidence threshold (default 0.8)

### "CHECK constraint failed: user_selected_index"
- For rejected groups, must set `user_selected_index` to 0-3
- Phase 1B now handles this automatically (sets to AI's selection)

### "NOT NULL constraint failed"
- Database schema requires certain fields
- Check that Phase 1A completed successfully before running Phase 1B
- Verify temp database has all required fields

## Validation

After completing all phases:

```bash
# Check record counts and action breakdown
sqlite3 data/training/ai_training_decisions/mojo3.db \
  "SELECT user_action, COUNT(*) FROM ai_decisions GROUP BY user_action;"

# Check AI performance metrics
sqlite3 data/training/ai_training_decisions/mojo3.db \
  "SELECT 
    COUNT(*) as total,
    AVG(selection_match)*100 as selection_accuracy_pct,
    AVG(crop_match)*100 as crop_accuracy_pct
  FROM ai_decisions 
  WHERE user_action != 'reject';"

# Verify no NULL crop coordinates for crop actions
sqlite3 data/training/ai_training_decisions/mojo3.db \
  "SELECT COUNT(*) FROM ai_decisions 
  WHERE user_action = 'crop' AND final_crop_coords IS NULL;"
  # Should return 0

# Verify all approve actions have full-image coordinates
sqlite3 data/training/ai_training_decisions/mojo3.db \
  "SELECT COUNT(*) FROM ai_decisions 
  WHERE user_action = 'approve' AND final_crop_coords != '[0.0, 0.0, 1.0, 1.0]';"
  # Should return 0 or very small number (tolerance edge cases)
```

## Safety Checklist

Before running Phase 2 (merge):

- [ ] Dry run completed and reviewed
- [ ] Backup of real database created
- [ ] Sample records look correct
- [ ] Record counts make sense
- [ ] No unexpected "Coords to CORRECT" in dry run
- [ ] User actions are approve/crop/reject (no 'skip')

## Files Created

**By Phase 1A:**
- `data/training/ai_training_decisions/{project_id}_backfill_temp.db`

**By Phase 2:**
- `data/training/ai_training_decisions/{project_id}.db` (updated)
- `data/training/ai_training_decisions/{project_id}.db.backup-TIMESTAMP` (automatic backup)

## Related Scripts

- `scripts/ai/backfill_project_phase1a_ai_predictions.py` - Phase 1A
- `scripts/ai/backfill_project_phase1b_user_data.py` - Phase 1B
- `scripts/ai/backfill_project_phase2_compare.py` - Phase 2
- `scripts/ai/validate_training_data.py` - Post-backfill validation

## Related Documentation

- `Documents/ai/AI_TRAINING_REFERENCE.md` - Database schema
- `Documents/ai/AI_TRAINING_GUIDE.md` - Training workflow
- `Documents/safety/CURSOR_AI_RULES.md` - File safety rules
