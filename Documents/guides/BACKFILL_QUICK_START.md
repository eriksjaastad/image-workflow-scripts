# Historical Backfill - Quick Start Guide

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers, Operators
**Estimated Reading Time:** 6 minutes

## Goal
Extract v3 training data from historical project archives by matching cropped images back to originals (vision-based), with strict validation.

## Workflow

### 1) Validate Project
```bash
# Identify project
grep "ProjectName" data/timesheet.csv

# Verify file modification dates match project window
stat -f "%Sm" -t "%Y-%m-%d" "training data/ProjectName_final/"*.png | sort -u
```

### 2) Dry Run
```bash
python3 scripts/tools/backfill_v3_from_archives.py ProjectName
```
- Inspect console and any generated inspection report.

### 3) Execute
```bash
python3 scripts/tools/backfill_v3_from_archives.py ProjectName --execute
```
- Writes SQLite v3 DB under `data/training/ai_training_decisions/ProjectName.db`.

### 4) Verify Results
```bash
sqlite3 data/training/ai_training_decisions/ProjectName.db \
  "SELECT final_crop_coords, COUNT(*) FROM ai_decisions WHERE user_action='crop' GROUP BY final_crop_coords;"
```
- Expect diversity of `final_crop_coords` (not all `[0,0,1,1]`).

## Known Issues
- Timestamp mismatches (e.g., `Aiko_raw`) likely indicate invalid crops.
- Skip invalid projects or redo archives before ingest.

## Safety
- Read-only on archives; creates NEW DB files only.
- No in-place changes to archived zips or images.

## Related Documents
- `archives/sessions/2025-10-22/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md`
- `core/PROJECT_LIFECYCLE_SCRIPTS.md`
- `ai/AI_TRAINING_REFERENCE.md`

