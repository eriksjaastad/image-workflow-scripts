# Handoff Notes - Historical Backfill Project
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** 2025-10-22  
**From:** Claude (Sonnet 4.5)  
**Status:** In Progress - Data Validation Issue Found

---

## 🎯 PROJECT GOAL
Extract v3 training data from historical project archives by comparing original images to final cropped images, then populating SQLite databases with full decision records (selected image + crop coordinates).

---

## ✅ WHAT'S BEEN COMPLETED

### 1. Created Backfill Script
**File:** `/Users/eriksjaastad/projects/Eros Mate/scripts/tools/backfill_v3_from_archives.py`

**What it does:**
- Takes project name (e.g., "Aiko_raw") from timesheet
- Loads original images from `training data/{project_name}/`
- Loads final cropped images from `training data/{project_name}_final/`
- Groups originals using shared grouping logic (`find_consecutive_stage_groups`)
- Matches cropped images back to original groups by timestamp
- Extracts crop coordinates using OpenCV template matching (`cv2.matchTemplate`)
- Writes v3 SQLite database: `data/training/ai_training_decisions/{project_name}.db`

**How to run:**
```bash
# Dry-run (preview only)
python3 scripts/tools/backfill_v3_from_archives.py Aiko_raw

# Execute (write database)
python3 scripts/tools/backfill_v3_from_archives.py Aiko_raw --execute
```

### 2. Dependencies Installed
- `opencv-python` (4.12.0.88) - for template matching
- `numpy` (2.2.6) - required by OpenCV
- Installed with: `pip3 install opencv-python --break-system-packages`

### 3. Tested on Aiko_raw Project
**Results:**
- ✅ Script ran successfully
- ✅ Created database with 350 decision records
- ✅ Template matching worked (99.99% confidence)
- ⚠️ **DATA INTEGRITY ISSUE FOUND** (see below)

---

## 🚨 CRITICAL ISSUE DISCOVERED

### Problem: File Modification Dates Don't Match Project Dates

**Aiko_raw Project:**
- **Timesheet date:** August 29, 2025
- **Final images modified:** August 27, 2025 (ALL 258 files)
- **Original images modified:** August 27, 2025 (ALL 1,050 files)

**What this means:**
- Files were NOT modified during the project date
- They were already in "final" state 2 days before project started
- Likely copied/moved from another location (preserving old timestamps)
- **We don't know which images were actually selected vs rejected**

**Current database (Aiko_raw.db) contains:**
- 350 decision records
- 258 marked as "crop" (but all have `[0.0, 0.0, 1.0, 1.0]` = full frame)
- 92 marked as "approve"
- **This data is unreliable for AI training!**

---

## 🔍 WHAT NEEDS TO HAPPEN NEXT

### Option 1: Verify Data Integrity (RECOMMENDED)
Check if Aiko_raw archive is valid by asking Erik:
1. Were these files copied from elsewhere?
2. Are the modification dates expected to be Aug 27?
3. Should we use this project or skip it?

### Option 2: Find Better Historical Projects
Look for projects where modification dates actually match project dates:

```bash
# Check project dates from timesheet
grep "ProjectName" data/timesheet.csv

# Check file modification dates
stat -f "%Sm" -t "%Y-%m-%d" "training data/ProjectName_final/"*.png | sort -u

# Files should be modified ON or DURING the project date range
```

### Option 3: Add Validation to Script
Update `backfill_v3_from_archives.py` to validate dates BEFORE processing:
- Load project date range from timesheet
- Check final images' modification dates
- Warn if dates don't match
- Skip processing if dates are suspicious

**Add this to the `run()` method after loading images:**
```python
# Validate modification dates
final_mod_dates = set()
for img in cropped_images:
    mod_time = img.stat().st_mtime
    mod_date = datetime.fromtimestamp(mod_time).date()
    final_mod_dates.add(mod_date)

# Parse project date
project_date = datetime.strptime(metadata['start_date'], "%m/%d/%Y").date()

# Warn if no files modified on project date
if project_date not in final_mod_dates:
    print(f"[!] WARNING: No final images modified on project date {project_date}")
    print(f"[!] Final images modified on: {sorted(final_mod_dates)}")
    print(f"[!] This suggests files were copied/moved, not actually cropped during project")
    if not self.dry_run:
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            return False
```

---

## 📂 FILE LOCATIONS

### Key Files
- **Backfill script:** `/Users/eriksjaastad/projects/Eros Mate/scripts/tools/backfill_v3_from_archives.py`
- **Timesheet:** `/Users/eriksjaastad/projects/Eros Mate/data/timesheet.csv`
- **Current test data:**
  - Originals: `/Users/eriksjaastad/projects/Eros Mate/training data/Aiko_raw/`
  - Finals: `/Users/eriksjaastad/projects/Eros Mate/training data/Aiko_raw_final/`
- **Created database:** `/Users/eriksjaastad/projects/Eros Mate/data/training/ai_training_decisions/Aiko_raw.db` (156 KB)

### Shared Utilities Used
- `scripts/utils/companion_file_utils.py`:
  - `find_consecutive_stage_groups()` - groups images into pairs/triplets
  - `sort_image_files_by_timestamp_and_stage()` - sorts by timestamp
  - `extract_timestamp_from_filename()` - returns string like "20250709_131100"

---

## 🧪 HOW TO TEST

### Verify a project is good for backfill:
```bash
# 1. Get project info
grep "ProjectName" data/timesheet.csv
# Note the date (column A) and image counts (columns G, H)

# 2. Check file counts match
find "training data/ProjectName/" -name "*.png" | wc -l
find "training data/ProjectName_final/" -name "*.png" | wc -l

# 3. Check modification dates
stat -f "%Sm" -t "%Y-%m-%d" "training data/ProjectName_final/"*.png | sort -u
# Should match or be within project date range!

# 4. Run dry-run
python3 scripts/tools/backfill_v3_from_archives.py ProjectName

# 5. Check results look reasonable
# - Groups matched should be close to final image count
# - Crop confidence should be high (>0.8)
# - Crop coordinates should vary (not all [0,0,1,1])
```

### Verify database contents:
```bash
# Check action distribution
sqlite3 data/training/ai_training_decisions/ProjectName.db \
  "SELECT user_action, COUNT(*) FROM ai_decisions GROUP BY user_action;"

# Check crop variety
sqlite3 data/training/ai_training_decisions/ProjectName.db \
  "SELECT final_crop_coords, COUNT(*) FROM ai_decisions WHERE user_action='crop' GROUP BY final_crop_coords LIMIT 20;"

# If all crops are [0,0,1,1], something is wrong!
```

---

## 🤔 QUESTIONS FOR ERIK

1. **About Aiko_raw archive:**
   - Is it normal for files to be dated Aug 27 when project was Aug 29?
   - Should we keep the Aiko_raw.db or delete it (unreliable data)?

2. **About historical archives in general:**
   - Do all archived projects have this timestamp issue?
   - Are we looking at the right "final" directories?
   - Should we look for actual cropped files elsewhere?

3. **About workflow expectations:**
   - In old projects, did Desktop Multi-Crop actually modify files?
   - Or were files moved to a "finished" directory without modification?
   - Where are the ACTUAL cropped images stored?

---

## 💡 TECHNICAL NOTES

### Template Matching (OpenCV)
The script uses `cv2.matchTemplate()` with `cv2.TM_CCOEFF_NORMED` to find cropped regions:
- Works by finding where the cropped image appears in the original
- Returns confidence score (0-1, higher = better match)
- Currently getting 0.99+ confidence, which is suspiciously perfect
- This happens when cropped image = full original (no actual crop)

### Coordinate Format
Crop coordinates stored as JSON array: `[x1, y1, x2, y2]`
- Normalized to [0, 1] range
- `[0.0, 0.0, 1.0, 1.0]` = full image (no crop)
- Example actual crop: `[0.15, 0.1, 0.85, 0.95]`

### Database Schema
Matches current v3 format:
- `group_id`: Unique ID like "Aiko_raw_legacy_20250709_131100"
- `user_action`: 'crop' or 'approve'
- `final_crop_coords`: JSON array or NULL
- `user_selected_index`: Which image in group was chosen (0-3)
- Full schema in: `/Users/eriksjaastad/projects/Eros Mate/data/schema/ai_training_decisions_v3.sql`

---

## 🎯 NEXT STEPS (Priority Order)

1. **Ask Erik about Aiko_raw data** - Is this expected or a problem?
2. **Add date validation to script** - Warn before processing suspicious dates
3. **Find better historical project** - One with proper modification dates
4. **Test on good project** - Verify crop coordinates are actually varied
5. **Process remaining historical projects** - Once we confirm approach works
6. **Update dashboard** - New crop rate graphs should show historical data

---

## 📊 PROGRESS TRACKING

**Completed today:**
- ✅ AI Reviewer auto-selection fix
- ✅ Desktop Multi-Crop performance logging
- ✅ Dashboard crop rate chart
- ✅ Backfill script creation
- ✅ Aiko_raw test run

**Blocked on:**
- ⏸️ Data integrity validation (need Erik's input)
- ⏸️ Historical project selection (need valid archives)

**Estimated token usage this session:** ~55k tokens

---

## 🔧 QUICK REFERENCE COMMANDS

```bash
# List all projects in timesheet
cat data/timesheet.csv | grep -v "^$" | awk -F, '{print $5}' | grep -v "^$" | sort -u

# Check what historical data exists
ls -la "training data/"

# Quick image count
find "training data/ProjectName/" -name "*.png" | wc -l

# Run backfill
python3 scripts/tools/backfill_v3_from_archives.py ProjectName --execute

# Check database
sqlite3 data/training/ai_training_decisions/ProjectName.db "SELECT COUNT(*) FROM ai_decisions;"
```

---

**Ready for next model to continue from here!** 🚀

