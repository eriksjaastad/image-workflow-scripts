# Historical Backfill - Quick Start Guide

**⚠️ OBSOLETE ⚠️**

**This guide is outdated. See the current guide at:**
`Documents/guides/BACKFILL_QUICK_START.md`

**Current process (Oct 31, 2025):**
- Phase 1A: Generate AI predictions by running models on original images
- Phase 1B: Extract user ground truth from physical cropped images  
- Phase 2: Intelligently merge temp database with real database

**This old guide references:**
- CSV files (no longer used)
- `timesheet.csv` (no longer used for backfill)
- `backfill_v3_from_archives.py` (replaced by 3-phase scripts)

---

# ARCHIVED CONTENT BELOW (FOR REFERENCE ONLY)

**Status:** OBSOLETE
**Audience:** Developers

**Last Updated:** 2025-10-26


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
python3 scripts/tools/backfill_v3_from_archives.py ProjectName
```

### Step 3: Validate Results
```bash
sqlite3 data/training/ai_training_decisions/ProjectName.db \
  "SELECT final_crop_coords, COUNT(*) FROM ai_decisions WHERE user_action='crop' GROUP BY final_crop_coords;"
```

## Related
- `archives/sessions/2025-10-22/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md`
