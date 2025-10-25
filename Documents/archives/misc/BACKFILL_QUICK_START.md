# Historical Backfill - Quick Start Guide

## 🎯 What We're Doing
Extracting v3 training data from old project archives by matching cropped images back to originals using computer vision.

## 🚨 Current Issue
**Aiko_raw project has BAD DATA:**
- Project date: Aug 29, 2025
- Files modified: Aug 27, 2025 (2 days earlier!)
- All 258 crops show full-frame `[0,0,1,1]` (no actual cropping)
- **Decision needed:** Delete Aiko_raw.db or keep it?

## 📝 How to Continue

### Step 1: Validate Next Project
```bash
# Pick a project from timesheet
grep "ProjectName" data/timesheet.csv

# Check if files were modified during project
stat -f "%Sm" -t "%Y-%m-%d" "training data/ProjectName_final/"*.png | sort -u
# ^ Should match the project date!
```

### Step 2: Run Backfill
```bash
# Dry-run first
python3 scripts/tools/backfill_v3_from_archives.py ProjectName

# If looks good, execute
python3 scripts/tools/backfill_v3_from_archives.py ProjectName --execute
```

### Step 3: Verify Results
```bash
# Check crop variety (should NOT all be [0,0,1,1])
sqlite3 data/training/ai_training_decisions/ProjectName.db \
  "SELECT final_crop_coords, COUNT(*) FROM ai_decisions WHERE user_action='crop' GROUP BY final_crop_coords;"
```

## 📂 Key Files
- **Script:** `scripts/tools/backfill_v3_from_archives.py`
- **Full handoff:** `Documents/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md`
- **Test data:** `training data/Aiko_raw/` and `training data/Aiko_raw_final/`
- **Questionable DB:** `data/training/ai_training_decisions/Aiko_raw.db`

## ❓ Ask Erik
1. Is Aiko_raw data valid despite timestamp mismatch?
2. Where are the actual cropped images stored in archives?
3. Which historical projects should we prioritize?

---
**Status:** Paused - need data validation before continuing

