# Running Batch AI Crop Predictions

## Summary

**What this does:** Runs Phase 1A AI predictions (using **Crop Proposer v3**) across all 20 historical projects (~182,000 images, excluding mojo3 which is already done).

**Estimated time:** ~10 hours total for all projects

**Models used:**
- Ranker: `ranker_v4.pt` (latest)
- Crop Proposer: `crop_proposer_v3.pt` ⭐ (latest - as requested)

## Quick Start

```bash
cd /Users/eriksjaastad/projects/image-workflow
source .venv311/bin/activate

# Run the batch script
./scripts/ai/run_batch_predictions.sh
```

The script will:
1. Process projects from smallest to largest
2. Extract each zip to `/tmp/` temporarily
3. Run AI predictions (selection + crop coords)
4. Save results to `data/training/ai_training_decisions/{project_id}.db`
5. Clean up temp files
6. Log progress
7. **Can be stopped and resumed** - tracks completed projects

## Output

**Databases:** `data/training/ai_training_decisions/{project_id}.db`
- One per project
- Contains AI predictions only (Phase 1A)
- Ready for Phase 1B (user ground truth extraction)

**Logs:**
- Progress: `data/ai_data/batch_predictions_log.jsonl`
- Summary: `data/ai_data/batch_predictions_summary_TIMESTAMP.json`

## Monitoring Progress

The script shows:
- ✅ Current project being processed
- 📊 Progress: X/20 projects completed
- ⏱ Estimated time remaining
- 🎉 Final summary when done

## Pausing & Resuming

**To stop:** Press `Ctrl+C` (it will finish current project first)

**To resume:** Just run the script again - it automatically skips completed projects

**Progress file:** `data/ai_data/batch_predictions_progress.txt`

## Projects To Process (in order)

Small projects first (fast wins):
1. dalia (98 images) - 20 seconds
2. Patricia (255 images) - 1 minute
3. mixed-0919 (852 images) - 3 minutes
4. agent-1003, agent-1002, agent-1001 (2k images each) - 7 min each
5. 1013, 1010, 1102 (3k images each) - 10 min each

Medium projects:
6-11. 1011, 1012, Eleni, Kiara_Slender, 1100, 1101_Hailey (4k-9k images) - 15-30 min each

Large projects (save for last):
12. tattersail-0918 (14k images) - 47 minutes
13. jmlimages-random (27k images) - 1.5 hours
14. mojo2 (36k images) - 2 hours
15. mojo1 (38k images) - 2.1 hours

**Tip:** You can stop after the small projects to verify everything works, then continue with the big ones later.

## What This Is (Phase 1A Only)

**Phase 1A** = AI Predictions
- What the AI would have suggested for selection
- What the AI would have suggested for crop coordinates
- No user data yet - that comes from Phase 1B

**To get complete training data:**
1. ✅ Run this script (Phase 1A) - creates AI predictions
2. ⏳ Run Phase 1B separately - extracts user ground truth from final images
3. ⏳ Run Phase 2 - merges temp databases (if needed)

## Safety Notes

This is **READ-ONLY** for your zip files:
- ✅ Only reads from zips
- ✅ Extracts to `/tmp/` (deleted after each project)
- ✅ Creates NEW database files only
- ❌ Never modifies zip files
- ❌ Never modifies existing images

## Troubleshooting

**"Ranker model not found"**
- Check that `data/ai_data/models/ranker_v4.pt` exists
- If not, script will tell you what's missing

**"Crop proposer model not found"**
- Check that `data/ai_data/models/crop_proposer_v3.pt` exists

**"Failed to extract zip"**
- Zip file might be corrupted
- Check disk space on `/tmp/`

**Script stops mid-project**
- No problem! Just run again - it will resume from next project
- Partially completed project will be re-processed

**Low disk space warning**
- Largest temp extraction is mojo1 (38k images, ~20GB)
- Make sure you have at least 30GB free in `/tmp/`

## After Batch Completes

1. **Validate databases:**
   ```bash
   python scripts/ai/validate_all_databases.py
   ```

2. **Check the summary report:**
   ```bash
   cat data/ai_data/batch_predictions_summary_*.json
   ```

3. **Review the plan document:**
   - `data/ai_data/AI_PREDICTIONS_BATCH_PLAN.md`
   - Contains full inventory and details

## Questions?

Read the full documentation:
- `Documents/guides/BACKFILL_QUICK_START.md` - Complete backfill guide
- `Documents/ai/AI_TRAINING_GUIDE.md` - Training workflow
- `data/ai_data/AI_PREDICTIONS_BATCH_PLAN.md` - This batch plan

---

**Ready to run?**

```bash
./scripts/ai/run_batch_predictions.sh
```

Grab some coffee - the first few projects are fast, but the big ones take a while! ☕

