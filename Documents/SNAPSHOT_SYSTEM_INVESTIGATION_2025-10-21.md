# Snapshot System Investigation & Project ID Fix
**Date:** October 21, 2025  
**Status:** ✅ Project ID Fixed | 📊 Snapshot System Understood

---

## ✅ **FIXED: Project ID Detection**

**Problem:** Was using directory names instead of project manifest files.

**Solution:** Created `get_current_project_id()` that reads from `data/projects/*.project.json` and finds the project with `finishedAt: null`.

**Changes Made:**
- `scripts/01_ai_assisted_reviewer.py`:
  - Replaced `detect_project_id(base_dir)` with `get_current_project_id()`
  - Now reads from project manifest system
  - Prints active project to console for verification
  
**Result:** AI-Assisted Reviewer will now correctly log training data with `project_id="mojo3"` (or whatever the current active project is).

---

## 📊 **How the Snapshot System Works**

### **What Triggers Snapshots:**

**1. Cron Job (Configured, but NOT Installed)**
```bash
# From scripts/setup_cron.sh:
# SUPPOSED to run daily at 2:15 AM:
# - extract_operation_events_v1.py
# - build_daily_aggregates_v1.py  
# - derive_sessions_from_ops_v1.py
```

**Current Status:**
```bash
$ crontab -l
# Only shows legacy cleanup job (2:00 AM daily)
# Snapshot pipeline cron job is NOT installed!
```

**To Install (if wanted):**
```bash
bash scripts/setup_cron.sh
```

---

**2. Manual Extraction (Current Method)**

These scripts **manually** extract data to snapshots:

```bash
# Extract project manifests (from data/projects/)
python scripts/data_pipeline/extract_projects_v1.py

# Extract operation events (from data/file_operations_logs/)
python scripts/data_pipeline/extract_operation_events_v1.py

# Extract progress snapshots (from data/crop_progress/, data/sorter_progress/)
python scripts/data_pipeline/extract_progress_snapshots_v1.py

# Extract timer sessions (from data/timer_data/)
python scripts/data_pipeline/extract_timer_sessions_v1.py

# Derive work sessions from operation events
python scripts/data_pipeline/derive_sessions_from_ops_v1.py

# Build daily aggregates for dashboard
python scripts/data_pipeline/build_daily_aggregates_v1.py
```

---

### **Where Raw Data Comes From:**

| Source | Written By | Extracted To Snapshot |
|--------|-----------|----------------------|
| `data/projects/*.project.json` | `00_start_project.py`, `07_finish_project.py` | `data/snapshot/projects_v1/projects.jsonl` |
| `data/file_operations_logs/*.csv` | `FileTracker` (all tools) | `data/snapshot/operation_events_v1/day=YYYYMMDD/events.jsonl` |
| `data/crop_progress/*.json` | `04_desktop_multi_crop.py` | `data/snapshot/progress_snapshots_v1/day=YYYYMMDD/snapshots.jsonl` |
| `data/sorter_progress/*.json` | `03_web_character_sorter.py` | `data/snapshot/progress_snapshots_v1/day=YYYYMMDD/snapshots.jsonl` |
| `data/timer_data/*.csv` | Dashboard/timer system | `data/snapshot/timer_sessions_v1/day=YYYYMMDD/sessions.jsonl` |

---

### **When to Run Extraction:**

**From `scripts/data_pipeline/README.md`:**

> **Automatically**: The dashboard reads from snapshots automatically. You don't need to run these unless:
> - You want to refresh snapshot data
> - You're doing data analysis with SQL queries
> - You need to backfill historical data
> - You're validating data integrity

**Recommendation:** Weekly or "as needed" basis.

---

## 🔍 **Current Snapshot Status**

### **Projects Snapshot:**
- **File:** `data/snapshot/projects_v1/projects.jsonl`
- **Last Updated:** Unknown (need to check file mtime)
- **Contains:** 19 projects (1011, 1012, ..., mojo1, mojo2)
- **Missing:** `mojo3` (created today, Oct 21)

**To Fix:**
```bash
python scripts/data_pipeline/extract_projects_v1.py
# This will add mojo3 to the projects snapshot
```

---

### **Operation Events:**
- **Directory:** `data/snapshot/operation_events_v1/`
- **Last Entry:** `day=20251016/` (October 16, 2025)
- **Gap:** Oct 17-21 (5 days of missing data)

**Raw logs still exist?**
```bash
ls -lh data/file_operations_logs/ | tail -10
# Check if logs from Oct 17-21 exist
```

**To Fix:**
```bash
python scripts/data_pipeline/extract_operation_events_v1.py
# Will process any new log files since Oct 16
```

---

### **Progress Snapshots:**
- **Directory:** `data/snapshot/progress_snapshots_v1/`
- **Last Entry:** `day=20251016/` (October 16, 2025)
- **Contains:** Crop and sorter progress from tools

**To Fix:**
```bash
python scripts/data_pipeline/extract_progress_snapshots_v1.py
# Will process any new progress files
```

---

## 🎯 **Erik's Context on This System**

**From Erik:**
> "That data pipeline... we came up with it sort of so that our dashboard could end up looking better. Because I wanted to be able to compare projects in our dashboard data."

**Purpose:**
1. **Project-based comparison** - Dashboard can show metrics per project
2. **Normalized data** - Raw logs → Clean JSON schemas
3. **Efficient queries** - Partitioned by day for fast access
4. **Historical analysis** - Can query trends over time

**Entry Point:**
- `00_start_project.py` - Kicks off project, starts timer, sets up manifest
- Creates `data/projects/{projectId}.project.json`
- All other tools read this to know the current project

---

## 🛠️ **What Was Just Fixed**

### **AI-Assisted Reviewer Now Uses Project Manifest:**

**Before (WRONG):**
```python
# Guessed from directory name
project_id = detect_project_id(Path("mojo3"))  # Returns "mojo3" from directory
```

**After (CORRECT):**
```python
# Reads from manifest
project_id = get_current_project_id()  # Returns "mojo3" from mojo3.project.json (finishedAt=null)
```

**Benefit:** 
- Always uses the correct project ID from the authoritative source
- Doesn't break if directory names don't match project IDs
- Respects the project lifecycle (active = no finish date)

---

## 📋 **Action Items**

### **Immediate (Optional):**

1. **Update mojo3 in snapshots:**
   ```bash
   python scripts/data_pipeline/extract_projects_v1.py
   ```

2. **Catch up operation events (Oct 17-21):**
   ```bash
   python scripts/data_pipeline/extract_operation_events_v1.py
   python scripts/data_pipeline/build_daily_aggregates_v1.py
   ```

3. **Install cron job (if you want automatic daily updates):**
   ```bash
   bash scripts/setup_cron.sh
   # Will run daily at 2:15 AM
   ```

---

### **Future Considerations:**

1. **Add crop_training_v2 schema to `data/schema/`:**
   - Create `data/schema/crop_training_v2.json`
   - Follow JSON Schema format like other schemas
   - Document relationship to `crop_training_data.csv`

2. **Decide on snapshot update frequency:**
   - Daily (via cron) - Automatic, always current
   - Weekly (manual) - Less overhead, still useful
   - As-needed (when doing analysis) - Minimal effort

---

## 🎉 **Summary**

**What You Discovered:**
- ✅ Sophisticated snapshot/data warehouse system exists
- ✅ JSON Schema definitions for all data types
- ✅ Partitioned event logs (35+ days of data)
- ✅ Project manifest system (19 projects tracked)

**What Got Fixed:**
- ✅ Project ID now read from manifest (not directory name)
- ✅ AI-Assisted Reviewer respects project lifecycle
- ✅ Training data will use correct project ID

**What You Learned:**
- Snapshots are extracted **manually** or via **cron** (not installed)
- Dashboard reads from snapshots for analysis
- Current data is Oct 16 (5 days behind)
- Mojo3 created today, not in snapshots yet

**What's Optional:**
- Running extraction scripts to update snapshots
- Installing cron job for automatic daily updates
- Adding crop schema to schema directory

---

**Bottom Line:** The system is working as designed (manual extraction), but snapshots haven't been updated since Oct 16. You can update them anytime by running the extraction scripts, or install the cron job for automatic daily updates.

