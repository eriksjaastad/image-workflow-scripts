# Technical Knowledge Base
## Key Learnings and Solutions for Image Processing Workflow

*This file contains technical solutions, common bugs, and patterns that work well for the image processing workflow.*

---

## 🚨 **CRITICAL: Data Operation Best Practices (October 2025)**

**⚠️ READ THIS FIRST - Always Create Inspection Reports Before Bulk Data Operations**

**Rule:** Before running ANY script that modifies bulk data (CSV files, training data, databases), ALWAYS create a detailed inspection report first.

**Why:** Provides confidence, catches corruption, prevents disasters. Example: When backfilling 7,193 crop dimensions, the inspection report revealed 5,431 corrupted rows with invalid timestamps that would have been incorrectly processed.

### **Inspection Report Must Include:**
1. **Row numbers** - CSV/database line numbers being affected
2. **Exact field values** - Use `repr()` to show `None` vs `'0'` vs `''` (empty string)
3. **Validation status** - Which rows pass/fail validation checks
4. **Sample data** - First 50+ rows in detail, plus summary table of all rows
5. **Counts by category** - How many rows in each project/status/type

### **Example Report Structure:**
```
====================================================================================================
OPERATION INSPECTION REPORT
====================================================================================================
Total rows to modify: 7193
Operation: Backfill dimensions
Target file: select_crop_log.csv
Report generated: 2025-10-21 16:17:38
====================================================================================================

BREAKDOWN:
  Mojo1 (Oct 8-11):  3978 rows ✅
  Mojo2 (Oct 16-19): 3215 rows ✅
  Corrupted/Invalid:    0 rows (5431 filtered out)

====================================================================================================

FIRST 50 ROWS (showing EXACT values):
----------------------------------------------------------------------------------------------------

Row 3:
  Filename:  20250705_230713_stage3_enhanced.png
  Timestamp: 2025-10-08 18:47:32 ✅
  Project:   mojo1
  Crop box:  x1=0.0, y1=66.0, x2=1787.0, y2=1853.0 ✅
  Dimensions (showing Python repr()):
    width_0  =      '0' → WILL BE BACKFILLED
    height_0 =      '0' → WILL BE BACKFILLED
    width_1  =     None → OK (not required)
    height_1 =     None → OK (not required)
```

### **Script Template:**
```python
#!/usr/bin/env python3
"""
Generate inspection report for [OPERATION]
Run this BEFORE running the actual operation script!
"""

def generate_report():
    # Read data
    with open(target_file) as f:
        rows = list(csv.DictReader(f))
    
    # Analyze which rows will be affected
    affected_rows = []
    for i, row in enumerate(rows, start=2):
        if should_process(row):  # Your validation logic
            affected_rows.append({
                'row': i,
                'field1': repr(row['field1']),  # Use repr()!
                'field2': repr(row['field2']),
                'valid': validate(row)
            })
    
    # Write detailed report
    with open(report_path, 'w') as f:
        f.write(f"Total affected: {len(affected_rows)}\n")
        
        # Show first 50 in DETAIL
        for r in affected_rows[:50]:
            f.write(f"\nRow {r['row']}:\n")
            f.write(f"  field1: {r['field1']} → valid={r['valid']}\n")
        
        # Summary table of ALL rows
        for r in affected_rows:
            f.write(f"{r['row']:<6} {r['field1']:<20}\n")
```

### **Confidence Gained:**
- ✅ See exact data types (`'0'` vs `None` vs `''`)
- ✅ Verify validation catches corruption
- ✅ Confirm row count matches expectations
- ✅ Review sample data before committing
- ✅ Catch edge cases (null timestamps, malformed data)

**Erik's Quote:** "Man, we should always build reports of data before we do anything. This gives me a lot of confidence."

---

## 🗄️ **AI Training Decisions v3 - SQLite System (October 2025)**

**TL;DR:** We replaced fragile CSV logging with robust SQLite databases. Per-project databases auto-create, validate on write, and track AI vs human decisions for ML training.

### **Why We Switched from CSV to SQLite**

**The Problem with CSV:**
- ❌ No validation (corrupt data possible)
- ❌ Slow writes (1-2 second lag per operation!)
- ❌ Concurrent access risks (file locking issues)
- ❌ Difficult to query (need custom parsers)
- ❌ No relationships (can't link AI recommendation to final crop)

**The SQLite Solution:**
- ✅ **ACID Compliant** - No data corruption possible
- ✅ **Instant Writes** - No lag, built-in transactions
- ✅ **Built-in Validation** - Constraints reject invalid data at write time
- ✅ **SQL Queries** - Easy analysis without custom code
- ✅ **Relationships** - Foreign keys, joins, views
- ✅ **Zero Setup** - Built into Python, works everywhere
- ✅ **File-Based** - One `.db` file per project (easy backup/archive)

### **Architecture: Two-Stage Logging**

**Stage 1: AI Reviewer (Selection)**
```python
# When user makes selection in AI Reviewer:
log_ai_decision(
    db_path=Path("data/training/ai_training_decisions/mojo3.db"),
    group_id="mojo3_group_20251021T234530Z_batch001_img002",
    project_id="mojo3",
    images=["img1.png", "img2.png", "img3.png"],
    ai_selected_index=1,       # AI picked image 2
    user_selected_index=2,     # USER picked image 3 (AI was wrong!)
    user_action="crop",        # Needs cropping
    image_width=3072,
    image_height=3072,
    ai_crop_coords=[0.1, 0.0, 0.9, 0.8],  # AI's crop proposal
    ai_confidence=0.87
)

# Creates .decision sidecar file for Desktop Multi-Crop:
# crop/img3.decision
{
    "group_id": "mojo3_group_20251021T234530Z_batch001_img002",
    "project_id": "mojo3",
    "needs_crop": true
}
```

**Stage 2: Desktop Multi-Crop (Cropping)**
```python
# When user completes crop in Desktop Multi-Crop:
# 1. Read .decision file → get group_id
decision_file = image_path.with_suffix('.decision')
with open(decision_file) as f:
    data = json.load(f)
    group_id = data['group_id']
    project_id = data['project_id']

# 2. Update database with final crop
update_decision_with_crop(
    db_path=Path(f"data/training/ai_training_decisions/{project_id}.db"),
    group_id=group_id,
    final_crop_coords=[0.2, 0.0, 0.7, 0.6]  # USER's actual crop
)

# 3. Delete .decision file
decision_file.unlink()
```

**Result:** Complete training record with AI recommendation + human correction!

### **Database Schema**

**Table: `ai_decisions`**

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | TEXT PRIMARY KEY | Unique identifier |
| `timestamp` | TEXT NOT NULL | ISO 8601 UTC |
| `project_id` | TEXT NOT NULL | Project name (e.g., "mojo3") |
| `images` | TEXT NOT NULL | JSON: `["img1.png", "img2.png", ...]` |
| `ai_selected_index` | INTEGER | Which image AI picked (0-3) |
| `ai_crop_coords` | TEXT | JSON: `[x1, y1, x2, y2]` normalized |
| `ai_confidence` | REAL | Model confidence (0.0-1.0) |
| `user_selected_index` | INTEGER NOT NULL | Which image user picked (0-3) |
| `user_action` | TEXT NOT NULL | `'approve'` \| `'crop'` \| `'reject'` |
| `final_crop_coords` | TEXT | JSON: `[x1, y1, x2, y2]` (filled later) |
| `crop_timestamp` | TEXT | ISO 8601 UTC when crop completed |
| `image_width` | INTEGER NOT NULL | Original width in pixels |
| `image_height` | INTEGER NOT NULL | Original height in pixels |
| `selection_match` | BOOLEAN | TRUE if AI picked same image as user |
| `crop_match` | BOOLEAN | TRUE if AI crop within 5% tolerance |

**Constraints:**
- `CHECK(user_action IN ('approve', 'crop', 'reject'))`
- `CHECK(ai_confidence IS NULL OR ai_confidence BETWEEN 0.0 AND 1.0)`
- `CHECK(image_width > 0 AND image_height > 0)`

**Indexes:**
- `idx_project_id` - Fast queries by project
- `idx_selection_match` - Filter by AI correctness
- `idx_crop_match` - Filter by crop quality

**Views:**
- `ai_performance` - Aggregated accuracy stats per project
- `incomplete_crops` - Images marked for crop but not yet done
- `ai_mistakes` - Decisions where AI was wrong (for training)

### **Per-Project Databases**

**Structure:**
```
data/training/ai_training_decisions/
├── mojo1.db              # Historical project
├── mojo2.db              # Historical project
├── mojo3.db              # Active project (auto-created!)
└── mojo4.db              # Future projects...
```

**Benefits:**
- ✅ Manageable size (~1-5MB per project vs one giant file)
- ✅ Easy to archive (copy `.db` file with finished project)
- ✅ Fast queries (smaller indexes, project isolation)
- ✅ Clean separation (bug in one doesn't affect others)

### **Auto-Initialization (Zero Setup!)**

**When you run AI Reviewer:**
```bash
python scripts/01_ai_assisted_reviewer.py mojo3/faces/
```

**What happens automatically:**
```
[*] Found active project: mojo3 (from mojo3.project.json)
[SQLite] Decision database ready: mojo3.db  ← Auto-created!
```

**Code:**
```python
# In AI Reviewer (automatic):
project_id = get_current_project_id()  # Reads manifest
db_path = init_decision_db(project_id)  # Creates if doesn't exist
```

**No manual setup required!**

### **Key Features**

**1. Auto-Calculated Match Flags**
```python
selection_match = (ai_selected_index == user_selected_index)
crop_match = all(abs(ai - user) < 0.05 for ai, user in zip(ai_crop, user_crop))
```

**Enables:**
- Training on mistakes (weight wrong examples higher)
- Progress tracking (is AI improving?)
- Analysis (which images fool the AI?)

**2. Crop Similarity Metrics**
```python
from scripts.utils.ai_training_decisions_v3 import calculate_crop_similarity

metrics = calculate_crop_similarity(
    ai_crop=[0.1, 0.0, 0.9, 0.8],
    user_crop=[0.2, 0.05, 0.75, 0.65]
)
# Returns: {'iou': 0.61, 'center_distance': 0.14, 'size_difference': 0.04}
```

**Useful for:**
- Analyzing crop proposals even when selection differs
- Understanding what AI learned about cropping
- Identifying systematic biases

**3. Data Validation**
```python
from scripts.utils.ai_training_decisions_v3 import validate_decision_db

errors = validate_decision_db(
    Path("data/training/ai_training_decisions/mojo3.db"),
    verbose=True
)

if errors:
    print("❌ Validation failed:")
    for err in errors:
        print(f"  - {err}")
else:
    print("✅ Database valid!")
```

**Checks:**
- Missing required fields
- Invalid coordinate ranges
- Invalid dimensions
- Incomplete crops (marked but not done)
- Orphaned entries

**4. Performance Stats**
```python
from scripts.utils.ai_training_decisions_v3 import get_ai_performance_stats

stats = get_ai_performance_stats(
    Path("data/training/ai_training_decisions/mojo3.db")
)

print(f"Selection Accuracy: {stats['selection_accuracy']:.1f}%")
print(f"Crop Accuracy: {stats['crop_accuracy']:.1f}%")
print(f"Total Decisions: {stats['total_decisions']}")
```

### **Integration Points**

**Scripts Using SQLite v3:**
1. `scripts/01_ai_assisted_reviewer.py` - Logs AI decisions + creates `.decision` files
2. `scripts/04_desktop_multi_crop.py` - Reads `.decision` files + updates with final crops
3. `scripts/ai/train_ranker_model.py` - Loads training data from SQLite
4. `scripts/ai/train_crop_model.py` - Loads crop training data from SQLite

### **Common Operations**

**Query AI Mistakes (for training):**
```sql
SELECT group_id, images, ai_selected_index, user_selected_index, ai_confidence
FROM ai_decisions
WHERE selection_match = 0
ORDER BY ai_confidence DESC;
```

**Find Incomplete Crops:**
```sql
SELECT * FROM incomplete_crops;
```

**Get Project Performance:**
```sql
SELECT * FROM ai_performance WHERE project_id = 'mojo3';
```

**Export to CSV for Analysis:**
```python
import sqlite3
import csv

conn = sqlite3.connect("data/training/ai_training_decisions/mojo3.db")
cursor = conn.execute("SELECT * FROM ai_decisions")

with open("mojo3_decisions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([desc[0] for desc in cursor.description])  # Header
    writer.writerows(cursor)
```

### **Performance Impact**

**Before (CSV logging):**
- ❌ 1-2 second lag per crop submit
- ❌ Risk of data corruption
- ❌ Slow bulk queries

**After (SQLite logging):**
- ✅ Instant crop operations (<10ms)
- ✅ ACID compliance (no corruption possible)
- ✅ Fast queries (indexed, optimized)

**Result:** Desktop Multi-Crop is now BLAZING FAST! ⚡

### **Testing**

**Unit Tests:** 14 tests in `scripts/tests/test_ai_training_decisions_v3.py`
- Database initialization
- Decision logging
- Crop updates
- Validation
- Performance stats
- Error handling

**Integration Tests:** 4 tests in `scripts/tests/test_ai_training_integration.py`
- Full workflow (AI Reviewer → Desktop Multi-Crop)
- `.decision` sidecar file lifecycle
- Missing file error handling
- Performance calculation

**Run Tests:**
```bash
pytest scripts/tests/test_ai_training_decisions_v3.py -v      # Unit tests
pytest scripts/tests/test_ai_training_integration.py -v       # Integration tests
```

### **Migration from Old Systems**

**Old CSV files are KEPT for historical data:**
- `data/training/select_crop_log.csv` - Legacy 19-column format
- `data/training/mojo1_crop_log.csv` - Historical Mojo1 data
- `data/training/mojo2_crop_log.csv` - Historical Mojo2 data

**NEW data goes to SQLite:**
- `data/training/ai_training_decisions/mojo3.db` - NEW system!

**Can backfill old data later (optional):**
- Read old CSVs
- Convert to SQLite format
- Populate historical databases

### **Files Reference**

**Core Utilities:**
- `scripts/utils/ai_training_decisions_v3.py` - Main library (580 lines)
- `data/schema/ai_training_decisions_v3.sql` - Database schema (150 lines)

**Documentation:**
- `Documents/AI_TRAINING_DECISIONS_V3_IMPLEMENTATION.md` - Complete spec (930 lines)
- `Documents/PHASE1_COMPLETE_SUMMARY.md` - Implementation summary

**Tests:**
- `scripts/tests/test_ai_training_decisions_v3.py` - Unit tests (460 lines)
- `scripts/tests/test_ai_training_integration.py` - Integration tests (280 lines)

### **Key Insight: The `.decision` Sidecar Pattern**

**Problem:** How to link AI Reviewer's initial decision with Desktop Multi-Crop's final crop when they're separate tools run at different times?

**Solution:** `.decision` sidecar files!

**Why Brilliant:**
- ✅ No hardcoded paths (just filename matching)
- ✅ Works across sessions (persistent)
- ✅ Self-documenting (JSON with group_id)
- ✅ Clean separation (AI Reviewer creates, Desktop Multi-Crop consumes)
- ✅ Automatic cleanup (deleted after successful update)

**Pattern:**
```
AI Reviewer:
  image.png → crop/
  group_id  → crop/image.decision

Desktop Multi-Crop:
  crop/image.png (load)
  crop/image.decision (read group_id)
  database (update row)
  crop/image.decision (delete)
  crop_cropped/image.png (save)
```

### **Future Enhancements**

**Planned:**
1. Build "AI Maturity Gauge" (Fetus → Newborn → Child → Adult progression)
2. Automated retraining when accuracy drops
3. Explainable AI (why did AI pick this image?)
4. Anomaly detection integration (flag disfigurements)
5. Dashboard integration (live performance tracking)

**Possible:**
- SQLite → PostgreSQL for multi-user scenarios
- Real-time training (update model as decisions come in)
- Confidence calibration (adjust confidence scores based on actual accuracy)

---

## 🎯 **Crop Training Data Schema Evolution (October 2025)**

**TL;DR:** We replaced a bloated 19-column CSV with a clean 8-column format. File moves no longer break training data!

### **Problem: Old Schema Was Bloated and Fragile**

The original `select_crop_log.csv` had **19 columns** with massive redundancy:

```csv
session_id, set_id, directory, image_count, chosen_index, chosen_path,
crop_x1, crop_y1, crop_x2, crop_y2, timestamp,
image_0_path, image_0_stage, width_0, height_0,
image_1_path, image_1_stage, width_1, height_1
```

**Issues:**
- ❌ Full paths stored → Break when files move
- ❌ No project tracking → Must deduce from timestamps/directories
- ❌ Redundant data → Multiple width/height columns for same image
- ❌ Irrelevant fields → session_id, set_id, chosen_index, etc.
- ❌ Directory storage → Meaningless (files move!)

### **Solution: New Minimal Schema (8 columns)**

```csv
timestamp,project_id,filename,crop_x1,crop_y1,crop_x2,crop_y2,width,height
```

**Example:**
```csv
2025-10-08T18:47:32Z,mojo1,20250705_230713_stage3_enhanced.png,0.0,0.0215,0.5820,0.6030,3072,3072
```

### **Benefits:**
1. ✅ **File-move resilient** - No paths, just filenames
2. ✅ **Project tracking built-in** - No timestamp deduction needed
3. ✅ **58% smaller** - 8 columns vs 19
4. ✅ **Faster processing** - Less data to parse
5. ✅ **Industry standard** - Similar to COCO/YOLO formats

### **Usage:**

```python
from scripts.utils.companion_file_utils import log_crop_decision

# NEW way (clean!)
log_crop_decision(
    project_id='mojo3',
    filename='20250820_065626_stage2_upscaled.png',
    crop_coords=(0.0, 0.0215, 0.5820, 0.6030),  # Normalized [0-1]
    width=3072,
    height=3072
)

# OLD way (deprecated - use for legacy code only)
log_select_crop_entry(
    session_id=..., set_id=..., directory=..., 
    image_paths=..., image_stages=..., image_sizes=..., 
    chosen_index=..., crop_norm=...
)
```

### **Validation:**

The new function enforces strict validation:
- ✅ Project ID must not be empty
- ✅ Filename must not contain paths (`/` or `\`)
- ✅ Crop coords must be normalized [0, 1] with x1 < x2, y1 < y2
- ✅ Dimensions must be positive integers
- ✅ Timestamp must be valid ISO 8601

**Raises `ValueError` immediately if validation fails!**

### **Files:**
- **New log:** `data/training/crop_training_data.csv` (new schema)
- **Legacy log:** `data/training/select_crop_log.csv` (old schema, 7,194 rows, kept for historical data)
- **Documentation:** `Documents/CROP_TRAINING_SCHEMA_V2.md`

**Status:** ✅ Implemented, documented, ready for production use  
**Migration:** TODO item to convert 7,194 legacy rows to new format

---

## 🚨 **CRITICAL: File Safety System (October 2025)**

**⚠️ File Integrity Protection**

**Achievement:** Multi-layer protection against accidental file modifications
**Impact:** Prevents data loss from accidental overwrites or corruption
**Date:** October 20, 2025

### **Quick Reference:**
- **Cursor Rules:** `.cursorrules` - AI must follow these rules
- **Audit Script:** `scripts/tools/audit_file_safety.py` - Scan for violations
- **Documentation:** `Documents/FILE_SAFETY_SYSTEM.md` - Complete guide
- **Run Audit:** `python scripts/tools/audit_file_safety.py`

### **Core Rules (NEVER VIOLATE):**
1. ✅ **ONLY** `04_desktop_multi_crop.py` may modify images
2. ✅ Move/delete operations allowed (via safe utilities)
3. ✅ Create NEW files in safe zones (`data/`, `sandbox/`)
4. ❌ NO modifications to existing production images/YAML/captions
5. ❌ NO overwrites without explicit justification

### **Safe Zones (NEW files OK):**
- `data/ai_data/`, `data/file_operations_logs/`, `data/daily_summaries/`, `sandbox/`

### **Protected Zones (NO modifications):**
- `mojo1/`, `mojo2/`, `selected/`, `crop/` - All production images

### **Before Committing Code:**
```bash
# Run safety audit
python scripts/tools/audit_file_safety.py

# Review flagged issues
# Verify they're in safe zones or crop tool
```

### **Philosophy:**
- **"Move, Don't Modify"** - Scripts move files, don't change contents
- **Read-Only by Default** - Treat production files as immutable
- **Data is Permanent** - Can't recover corrupted files

### **If You See Weird File Behavior:**
1. Check FileTracker logs: `grep "filename.png" data/file_operations_logs/*.log`
2. Look for unexpected modifications
3. Use git to recover: `git checkout filename`
4. Check macOS Trash: `~/.Trash/`

**See:** `Documents/FILE_SAFETY_SYSTEM.md` for complete documentation

---

## 🏗️ **Major Architectural Improvements (October 2025)**

### **Centralized Utility System**
**Achievement:** Created comprehensive `companion_file_utils.py` with shared functions
**Impact:** Eliminated code duplication across 6+ scripts
**Key Functions:**
- `find_all_companion_files()` - Wildcard companion file detection
- `move_file_with_all_companions()` - Safe file movement with companions

## Watchdog & Heartbeat (Sandbox Experiments)

Runner: `scripts/tools/reducer.py` (sandbox-only)

New CLI flags:
- `--max-runtime <sec>`: hard wall-clock timeout (default 900s). Triggers abort.
- `--stage-timeout <sec>`: per-phase time budget (reserved for future use in phases).
- `--progress-interval <sec>`: terminal progress cadence (default 10s).
- `--watchdog-threshold <sec>`: no-progress stall threshold (default 120s). Triggers abort if heartbeat doesn’t advance.
- `--no-stack-dump`: disable stack dump on abort.
- `--simulate-hang`: test hook that suppresses heartbeats to validate watchdog (E2E test uses this).

Behavior:
- Heartbeat tracks `files_scanned`, `groups_built`, `items_processed`, and updates timestamps.
- Watchdog monitors heartbeats and max runtime; on stall/timeout it emits `ABORT <run-id> reason=<...>` and writes sandbox-only error artifacts:
  - `sandbox/mojo2/logs/error_<run-id>.json` (reason, timers, last snapshot)
  - `sandbox/mojo2/logs/stack_<run-id>.txt` (if stack dump enabled)
- Clean shutdown ensures background threads stop; no FileTracker or global logs are written in harness runs.

Tests:
- `scripts/tests/test_watchdog.py`: unit test (stall and error report creation).
- `scripts/tests/test_runner_watchdog_e2e.py`: end-to-end simulated hang; expects abort and sandbox logs.
- `launch_browser()` - Centralized browser launching
- `generate_thumbnail()` - Optimized thumbnail generation
- `format_image_display_name()` - Consistent image name formatting
- `calculate_work_time_from_file_operations()` - Intelligent work time calculation

### **File-Operation-Based Timing System (Hour-Blocking)**
**Achievement:** Replaced ActivityTimer with simple, robust hour-blocking timing
**Method:** Count unique hour blocks (YYYY-MM-DD HH) where ANY file operation occurred
**Benefits:** 
- No subjective break detection thresholds
- Brutally honest: if files moved during an hour, that hour counts
- Works across midnight naturally
- Productivity variation shown via img/h metric
**Implementation:** `calculate_work_time_from_file_operations()` in `companion_file_utils.py`
**Formula:** Each unique hour block = 1 hour (3600 seconds)
**Tools Updated:** All file-heavy tools (image selector, character sorter, crop tools)
**Date:** October 15, 2025

### **Productivity Dashboard - Architecture & Patterns (October 2025)**
**Goal:** A fast, reliable dashboard that surfaces production throughput from local logs only.

Components:
- `scripts/dashboard/data_engine.py` (backend data assembler)
- `scripts/dashboard/productivity_dashboard.py` (Flask app, API + transform)
- `scripts/dashboard/dashboard_template.html` (Chart.js UI)
- `scripts/dashboard/project_metrics_aggregator.py` (per-project metrics)

Data contracts (API /api/data/<time_slice>):
- `metadata`: { generated_at, time_slice, lookback_days, scripts_found, projects_found, active_project, data_range }
- `activity_data`: aggregated ActivityTimer metrics (when present)
- `file_operations_data`: aggregated FileTracker metrics
- `historical_averages`: pre-aggregated for overlays (optional)
- `project_metrics`: map of projectId →
  - `totals`: { images_processed, operations_by_type }
  - `throughput`: { images_per_hour }
  - `timeseries`: { daily_files_processed: [[YYYY-MM-DD, count], ...] }
  - `startedAt`, `finishedAt`, `title`, `status`
- `project_kpi`: convenience payload for the selected project
- `timing_data`: per-display-tool timing with method source:
  - { work_time_seconds, work_time_minutes, files_processed, efficiency_score, timing_method: 'file_operations'|'activity_timer' }

Timing system and fallbacks:
- Prefer file-operation timing for file-heavy tools; fallback to ActivityTimer sums if file ops absent for that tool.
- `timing_data` exists even when ActivityTimer is missing; UI cards do not need extra flags.

Timestamp policy:
- Normalize ingested timestamps to naive `datetime` (drop tzinfo) to avoid mixed aware/naive comparisons.
- UI formats dates in local time for display; intraday banding uses the label values directly.

Intraday day-banding (15m/1h):
- Early approach used tick centers → misaligned bands when ticks auto-skipped.
- Final approach computes each bar group's left/right edges from adjacent tick spacing and draws bands from `leftEdge(firstIndexOfDay)` to `rightEdge(lastIndexOfDay)` with separator at the exact left edge. Result: bands flip exactly at midnight and align with bars for 15m/1h/D/W/M.

KPI and project selection:
- KPI shows per-project throughput if a project is selected and metrics exist.
- When “All Projects” or no metrics: KPI falls back to aggregation from `by_operation` totals (sparkline from daily sums) and computes images/hour using total files divided by summed `timing_data` minutes.
- Project markers: dashed start (blue) and end (red) lines with ISO tooltip labels.

Selection persistence:
- Persist operation/tool selections and project choice in `localStorage` under `dashboardSelections` and `dashboardProjectId`.
- On each data reload, UI restores selections before re-rendering charts; toggles remain stable across time-frame changes.

Hardening fixes (symptoms → fix):
- 500 "can't compare offset-naive and offset-aware datetimes" → normalize all timestamps at load.
- "'str' object cannot be interpreted as an integer" in companion metrics path → coerce `timestamp` fields to ISO strings before calling `get_file_operation_metrics`.
- Bands showing multi-day spans on 15m/1h → switch to bar-edge calculations, not tick centers.

Performance:
- Keep rendering ≤100ms by pre-aggregating server-side and minimizing DOM churn. Chart rebuilds respect restored selections to avoid extra work.

Testing additions:
- Engine tests for project metrics aggregation, mixed tz, no-finishedAt, and presence of `timing_data`.
- Core test for intraday slice alignment across midnight.


### **Desktop Tool Refactoring**
**Achievement:** Created `BaseDesktopImageTool` base class
**Impact:** Eliminated 200+ lines of duplicate code between desktop tools
**Benefits:** Consistent behavior, easier maintenance, shared improvements
**Tools Refactored:** `01_desktop_image_selector_crop.py`, `04_multi_crop_tool.py`

### **Project Organization Cleanup**
**Achievement:** Moved all files to proper directories
**Structure:**
- `Documents/` - All documentation and guides
- `data/` - All data files and models
- `scripts/tests/` - All test files
- Root directory - Only essential config files (.gitignore, .coverage, etc.)

### **Project Lifecycle Automation**
**Achievement:** Automated project start/finish with comprehensive scripts
**Scripts:**
- `00_start_project.py` - Initialize new projects with proper timestamps and metadata
- `00_finish_project.py` - Complete projects with ZIP creation and manifest updates
- `import_historical_projects.py` - Import historical projects from CSV timesheets

**Benefits:**
- No manual manifest editing
- Consistent timestamp formatting (ISO-8601 UTC with Z)
- Auto-count initial/final images
- Integrated with `prezip_stager.py` for delivery ZIPs
- Historical data backfill from timesheets

**Key Features:**
- Interactive and command-line modes
- Dry-run safety by default
- Manifest backup before updates
- Handles multi-day projects
- Special case handling (missing data, dual projects same day)

**Date:** October 15, 2025

---

## 🐛 **Common Bugs & Solutions**

### **Matplotlib Display Crashes**
**Problem:** Desktop image selector crop tool crashes when advancing to next triplet
**Root Cause:** Recreating matplotlib display on every triplet load causes backend conflicts
**Solution:** Only recreate display when number of images changes, reuse existing display otherwise
```python
# Only recreate display if number of images changed
if not hasattr(self, 'current_num_images') or self.current_num_images != num_images:
    self.setup_display(num_images)
    self.current_num_images = num_images
else:
    # Reuse existing display
```
**Date:** October 1, 2025

### **FileTracker Method Name Mismatch**
**Problem:** `'FileTracker' object has no attribute 'log_action'`
**Root Cause:** Method is called `log_operation`, not `log_action`
**Solution:** Use correct method name with proper parameters
```python
# Wrong:
self.tracker.log_action("crop", str(png_path))

# Correct:
self.tracker.log_operation("crop", source_dir=str(png_path.parent), dest_dir=str(png_path.parent))
```
**Date:** October 1, 2025

### **Aspect Ratio Auto-Adjustment Resetting Status**
**Problem:** When crop tool auto-adjusts for aspect ratio, it resets image status from KEEP back to DELETE
**Root Cause:** Aspect ratio adjustment triggers crop selection event again, calling select_image()
**Solution:** Check current status before auto-selecting
```python
# Only auto-select if currently marked for deletion
current_status = self.image_states[image_idx]['status']
if current_status == 'delete':
    self.select_image(image_idx)
else:
    # Preserve existing status
```
**Date:** October 1, 2025

### **ActivityTimer Integration Issues**
**Problem:** ActivityTimer causing crashes and complexity in file-heavy tools
**Root Cause:** ActivityTimer designed for scroll-heavy tools, not file operations
**Solution:** Replaced with file-operation-based timing system
```python
# Old approach (problematic):
activity_timer.mark_activity()
activity_timer.log_operation("crop", file_count=1)

# New approach (intelligent):
work_time = calculate_work_time_from_file_operations(file_operations)
```
**Date:** October 3, 2025

### **Search/Replace Failures During Refactoring**
**Problem:** Multiple search/replace operations failing due to whitespace variations
**Root Cause:** Exact string matching too strict for large refactoring operations
**Solution:** Use more precise edits, read exact lines before replacing
**Best Practice:** Break large refactoring into smaller, more targeted changes
**Date:** October 3, 2025

### **JavaScript Syntax Errors in Dashboard**
**Problem:** Extra closing braces causing JavaScript syntax errors
**Root Cause:** Manual editing introducing syntax errors
**Solution:** Always validate JavaScript syntax after edits
**Prevention:** Use proper indentation and bracket matching
**Date:** October 3, 2025

---

## 🎨 **UI/UX Patterns That Work**

### **Colorblind-Friendly Colors**
**Use:** Blue/Red instead of Green/Red for better accessibility
**Implementation:** 
- Blue = KEEP/Selected
- Red = DELETE/Unselected

### **Dynamic Layout Based on Content**
**Pattern:** Adjust UI layout based on actual data (2 vs 3 images)
**Implementation:**
- Detect number of items
- Adjust spacing and sizing accordingly
- Reuse existing display when possible

### **Centralized Error Display**
**Pattern:** Persistent, dismissible error bars instead of alert popups
**Implementation:**
```html
<div class="error-bar" id="error-bar" style="display: none;">
    <span id="error-message"></span>
    <button onclick="hideError()">×</button>
</div>
```
**Benefits:** Non-blocking, persistent, better UX

### **Intelligent Work Time Calculation**
**Pattern:** Calculate work time from file operations with break detection
**Implementation:**
```python
def calculate_work_time_from_file_operations(file_operations, break_threshold_minutes=5):
    # Only count time between operations if gap < threshold
    # Automatically detects breaks and excludes idle time
```
**Benefits:** More accurate than manual timers, automatic break detection

### **Wildcard Companion File Logic**
**Pattern:** Find all files with same base name as image
**Implementation:**
```python
def find_all_companion_files(image_path):
    base_name = image_path.stem
    return [f for f in parent_dir.iterdir() 
            if f.stem == base_name and f != image_path]
```
**Benefits:** Handles any file type, future-proof, consistent behavior

---

## 🔧 **Technical Patterns**

### **Base Class Inheritance Pattern**
**Pattern:** Create base classes for tools with shared functionality
**Implementation:**
```python
class BaseDesktopImageTool:
    def __init__(self, tool_name):
        # Shared initialization
    def setup_display(self, num_images):
        # Shared display logic
    def load_image_safely(self, image_path, subplot_idx):
        # Shared image loading
```

### **Centralized Utility Pattern**
**Pattern:** Move common functions to shared utility modules
**Benefits:** Single source of truth, easier maintenance, consistent behavior
**Implementation:** Create `companion_file_utils.py` with all shared functions

### **File-Operation Timing Pattern**
**Pattern:** Use file operations to calculate work time instead of manual timers
**Benefits:** More accurate, automatic break detection, no user interaction required
**Implementation:** Analyze FileTracker logs with intelligent gap detection

### **Triplet Detection Logic - SIMPLE IS BETTER**
**Pattern:** Group images by strictly increasing stage numbers using simple comparison
**Critical Rule:** Timestamps are ONLY for SORTING, stage numbers are for GROUPING
**Revolutionary Insight:** Simple solutions are often better than "robust" over-engineered ones

**The Problem with Over-Engineering:**
1. **Complex lookup tables:** Unnecessary complexity for simple comparisons
2. **Configuration parameters:** `consecutive_only`, `ordered_stages` - more things to get wrong
3. **Brittle design:** Breaks if `ordered_stages` doesn't match your data
4. **Hard to debug:** More moving parts to go wrong

**The Simple Solution That Actually Works:**
```python
def group_progressive(files, stage_of, min_group_size=2):
    """
    Group files into progressive stage runs.
    
    Args:
        files: list of file paths/objects sorted by timestamp (and then stage).
        stage_of: callable that extracts the float stage from a file.
        min_group_size: only emit groups >= this many files.
        
    Returns:
        list of groups (each group is a list of files).
    """
    groups = []
    n = len(files)
    i = 0

    while i < n:
        # start a new group anywhere
        cur_group = [files[i]]
        prev_stage = stage_of(files[i])
        i += 1

        # extend the run while stage strictly increases
        while i < n:
            s = stage_of(files[i])
            if s > prev_stage:  # strictly increasing, any jump allowed
                cur_group.append(files[i])
                prev_stage = s
                i += 1
            else:
                break  # stage repeated or decreased — end group

        if len(cur_group) >= min_group_size:
            groups.append(cur_group)

    return groups

# Usage:
files = sort_image_files_by_timestamp_and_stage([...])
groups = group_progressive(
    files,
    stage_of=lambda p: get_stage_number(detect_stage(p.name)),
    min_group_size=2,
)
```

**Why This Is Revolutionary:**

1. **Simplicity:** 15 lines of clear logic vs 50+ lines of complex lookup tables
2. **Self-Documenting:** `if s > prev_stage:` - crystal clear intent
3. **Robust:** Works with any stage numbering system, no configuration needed
4. **Future-Proof:** Automatically handles new stages (like `stage4_final`)
5. **No Configuration Errors:** No parameters to get wrong
6. **Handles All Cases:** 1→2, 1→3, 1.5→3, 2→3, 1→1.5→2→3 - all work naturally

**The Key Insight:**
Your workflow is simple: **"Group files where each stage is greater than the previous stage."** 

This code implements exactly that logic without any unnecessary complexity.

**Real-World Example:**
- Sorted files: `stage2_upscaled`, `stage2_upscaled`, `stage3_enhanced`, `stage2_upscaled`
- Logic: `stage2_upscaled` (2.0) → `stage2_upscaled` (2.0) → `stage3_enhanced` (3.0)
- Result: Groups `stage2_upscaled` → `stage2_upscaled` → `stage3_enhanced` (stops at next `stage2_upscaled`)

**Centralized Implementation:**
This logic is now in `companion_file_utils.py` as `find_consecutive_stage_groups()`, ensuring ALL tools use the same simple, robust algorithm.

**CRITICAL RULE — TIMESTAMPS ARE ONLY FOR SORTING**
Do not use timestamps for grouping boundaries or gap decisions. They are inherently unreliable for gap inference. We use timestamps strictly to sort files deterministically before grouping. Grouping itself is based ONLY on stage numbers and the nearest-up rule below.

**Nearest-Up Grouping (Definitive Spec):**
- Files are pre-sorted by `(timestamp, then stage)`.
- A run starts at any file; at each step, pick the smallest stage strictly greater than the previous stage within a lookahead window.
- Boundaries:
  - If a duplicate or non-increasing stage is encountered, the current run ends.
  - No time-gap boundaries in production. `time_gap_minutes` exists for rare analysis but defaults to `None` and should not be used in normal workflows.
- Defaults: `min_group_size=2`, `lookahead=50`, `time_gap_minutes=None`.
- Determinism: Sorting + nearest-up selection produces stable, predictable groups.

**Practical Examples:**
- Nearest-up with early stage3 present:
  - Files: `1`, `3` (00:10), `2` (00:20), `3` (00:30)
  - Group: `[1, 2, 3]` (the early `3` is ignored until `2` is found, then the later `3` completes the run)
- Duplicate stage splits runs:
  - Files: `1`, `1.5`, `2`, `2`, `3`
  - Groups: `[1, 1.5, 2]`, `[2, 3]`
- Large timestamp gaps are ignored for grouping (sorting only):
  - Files: `1` (00:00), `2` (00:10), `3` (01:00)
  - Group: `[1, 2, 3]` in production (no time-gap cutoffs)

**Critical Lessons Learned:**
1. **Simple, direct solutions are often better** than "robust" over-engineered ones
2. **Don't solve problems you don't have** - avoid unnecessary complexity
3. **Configuration parameters are liabilities** - more things to get wrong
4. **Self-documenting code is better** than complex algorithms with explanations
5. **Simplicity is the ultimate sophistication**

**Date:** October 3, 2025 (learned that simple solutions are often better than complex ones)

### **Critical Testing Lessons - Why Tests Were Failing Us**
**Problem:** Our tests were passing when they should have been failing, allowing bugs to persist for weeks
**Root Cause:** Tests were testing "does it run?" instead of "does it work correctly?"

**The Terrible Test Pattern (What NOT to do):**
```python
# BAD TEST - This passes even with completely broken logic!
assert any("stage1" in f for f in filenames), "Should have stage1 files"
assert len(groups) > 0, "Should detect at least some triplet groups"
```

**Why This Test Was Terrible:**
1. **Weak Assertions:** `any("stage1" in f for f in filenames)` - passes even with random grouping
2. **No Edge Case Testing:** Doesn't test same stages, backwards progression, or specific requirements
3. **No Validation of Grouping Logic:** Only checks that some files were found, not HOW they were grouped
4. **No Comprehensive Coverage:** Doesn't test all valid combinations (1→2, 1→3, 1.5→3, etc.)

**The Excellent Test Pattern (What TO do):**
```python
# GOOD TEST - Tests specific expected behavior
test_cases = [
    ("1→1.5", ["stage1_generated.png", "stage1.5_face_swapped.png"], 1, [2]),
    ("1→2", ["stage1_generated.png", "stage2_upscaled.png"], 1, [2]),
    ("1→3", ["stage1_generated.png", "stage3_enhanced.png"], 1, [2]),
    ("1.5→2", ["stage1.5_face_swapped.png", "stage2_upscaled.png"], 1, [2]),
    ("1.5→3", ["stage1.5_face_swapped.png", "stage3_enhanced.png"], 1, [2]),
    ("2→3", ["stage2_upscaled.png", "stage3_enhanced.png"], 1, [2]),
    # ... all combinations
]

for description, test_files, expected_groups, expected_sizes in test_cases:
    groups = find_consecutive_stage_groups(file_paths)
    assert len(groups) == expected_groups, f"{description}: Expected {expected_groups} groups, got {len(groups)}"
    actual_sizes = [len(group) for group in groups]
    assert actual_sizes == expected_sizes, f"{description}: Expected group sizes {expected_sizes}, got {actual_sizes}"
```

**Why This Test Is Excellent:**
1. **Tests ALL valid combinations:** Every possible consecutive stage progression
2. **Validates exact group counts:** Not just "some groups exist"
3. **Validates stage progression:** Ensures stages are actually consecutive and in order
4. **Tests edge cases:** Same stages (should NOT group), backwards progression (should break groups)
5. **Would catch bugs immediately:** Same stage grouping would fail the first test

**Critical Testing Principles:**
1. **Test specific expected behavior** - not just "does it run without crashing"
2. **Test edge cases** - same stages, backwards progression, invalid data
3. **Test all valid combinations** - don't assume only one pattern works
4. **Validate exact outputs** - group counts, group sizes, stage progressions
5. **Test would catch the bug** - if the test passes with broken logic, it's a bad test

**The Lesson:** A test that passes with broken logic is worse than no test at all - it gives false confidence and hides bugs for weeks.

### **The Final Testing Insight**
**Critical Discovery:** Comprehensive tests that validate specific behavior are essential

**What Makes Our Tests Excellent Now:**
1. **Tests ALL valid combinations:** 1→1.5, 1→2, 1→3, 1.5→2, 1.5→3, 2→3, 1→1.5→2, 1→1.5→3, 1→2→3, 1.5→2→3, 1→1.5→2→3
2. **Validates exact group counts:** Not just "some groups exist"
3. **Validates stage progression:** Ensures stages are actually consecutive and in order
4. **Tests edge cases:** Same stages (should NOT group), backwards progression (should break groups)
5. **Would catch bugs immediately:** Same stage grouping would fail the first test

**The Test That Would Have Caught the Bug:**
```python
def test_same_stage_not_grouped():
    """Test that same stages are NOT grouped together (this would catch the bug)"""
    test_files = [
        "20250705_214626_stage2_upscaled.png",
        "20250705_214953_stage2_upscaled.png",  # Same stage
        "20250705_215137_stage2_upscaled.png",  # Same stage
        "20250705_215319_stage2_upscaled.png",  # Same stage
    ]
    
    groups = find_consecutive_stage_groups(file_paths)
    
    # CRITICAL TEST: Same stages should NOT be grouped together
    # This test would have FAILED with the old broken logic!
    assert len(groups) == 0, f"Same stages should not be grouped, but got {len(groups)} groups"
```

**Why This Test Is Perfect:**
- **Specific:** Tests exact behavior (same stages should not group)
- **Clear failure:** Would immediately show the bug
- **Edge case:** Tests the exact scenario that was broken
- **Comprehensive:** Tests all valid combinations plus edge cases

**Date:** October 3, 2025 (learned that comprehensive tests prevent weeks of broken functionality)

### Testing Playbook (Triplet Grouping)

- Always pre-sort input with `sort_image_files_by_timestamp_and_stage(files)` before grouping.
- Validate exact behavior with strong assertions (group counts, sizes, and strict stage sequences).
- Covered cases in `scripts/tests/test_triplet_detection_logic.py`:
  - All valid consecutive combinations (1→1.5, 1→2, 1→3, etc.)
  - Same-stage NOT grouped
  - Strictly increasing order within a group
  - Backwards stage breaks the group
  - Nearest-up selection (e.g., 1,3,2,3-later → [1,2,3])
  - Duplicate stage ends run
  - Integration check against real data in `mojo1`

Production stance: Do not use time gaps in grouping tests. Timestamps are for sorting only.

### Centralized Sorting - Non-Negotiable Rule

- All human-facing tools MUST call `sort_image_files_by_timestamp_and_stage(files)` before any display or grouping.
- Determinism: Sorting is by (timestamp, then stage number, then filename) to produce stable order.
- Where to use: web image selector, desktop selector + crop, multi-crop tool, character sorter, viewers.
- Unit test added: `scripts/tests/test_sorting_determinism.py` to validate ordering on a known 4-file set.

### **Test Suite Maintenance**
**Pattern:** Always catalog changes made without corresponding test updates
**Implementation:** Use todo list to track changes that need test updates later
**Example:** "Oct 1: Changed desktop image selector crop tool title to show just image name instead of batch/progress info"

### **Subprocess Path Handling**
**Pattern:** Always use proper working directory and relative paths in subprocess calls
**Implementation:**
```python
result = subprocess.run([
    sys.executable, "script_name.py", args
], capture_output=True, text=True, cwd=Path(__file__).parent)
```

### **Matplotlib Backend Setup**
**Pattern:** Consistent backend setup across all matplotlib-based tools
**Implementation:**
```python
# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

try:
    matplotlib.use('Qt5Agg', force=True)
    backend_interactive = True
except Exception as e:
    matplotlib.use('Agg', force=True)
    backend_interactive = False
```

### **Progress Tracking & Session Management Patterns**
**Pattern:** Robust progress tracking with stable IDs and graceful error handling
**Critical Insight:** Progress files need to be stable, portable, and handle edge cases gracefully

**Key Components:**

1. **Stable ID Generation (Path-Portable):**
```python
def make_triplet_id(paths):
    """Create stable ID that works across Windows/POSIX systems."""
    # Use as_posix() for cross-platform compatibility
    s = "|".join(p.resolve().as_posix() for p in paths)
    return "t_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
```

2. **Normalized Progress Filenames:**
```python
# Avoid path drift and super-long filenames
abs_base = self.base_directory.resolve()
safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', abs_base.as_posix())[:200]
self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
```

3. **Immediate Persistence After Reconciliation:**
```python
def load_progress(self):
    # ... load existing data ...
    self.ensure_all_triplets_in_session_data()
    self.save_progress()  # CRITICAL: Persist immediately
    self.migrate_old_keys()  # One-time migration
```

4. **Graceful Error Handling:**
```python
def cleanup_completed_session(self):
    try:
        if self.progress_file.exists():
            self.progress_file.unlink()
    except PermissionError:
        print("[!] Could not remove progress file (locked); ignoring.")
    except Exception as e:
        print(f"[!] Error cleaning up progress file: {e}")
```

5. **Status Distinction & Helper Methods:**
```python
def mark_status(self, status):
    """Mark current triplet with specific status (completed/skipped)."""
    ct = self.get_current_triplet()
    if not ct: return
    
    d = self.session_data.setdefault('triplets', {})
    key = ct.id if ct.id in d else ct.display_name
    d.setdefault(key, {
        'display_name': ct.display_name,
        'files_processed': 0,
        'total_files': len(ct.paths),
    })
    d[key]['status'] = status
    self.save_progress()
```

6. **One-Time Migration for Backward Compatibility:**
```python
def migrate_old_keys(self):
    """Migrate old display_name keys to stable IDs."""
    trips = self.session_data.get('triplets', {})
    changed = False
    for t in self.triplets:
        if t.display_name in trips and t.id not in trips:
            trips[t.id] = trips.pop(t.display_name)
            changed = True
    if changed:
        self.save_progress()
```

**Why These Patterns Matter:**
1. **Cross-Platform Stability:** `as_posix()` prevents hash changes between Windows/POSIX
2. **Immediate Persistence:** Prevents data loss if tool crashes during reconciliation
3. **Graceful Degradation:** File locks don't crash the tool
4. **Status Tracking:** Distinguish between completed vs skipped items
5. **Backward Compatibility:** Migrate old progress files automatically
6. **Clean Filenames:** Avoid filesystem issues with special characters

**Critical Lessons:**
- **Always persist immediately** after data reconciliation
- **Use stable, content-derived IDs** instead of display names for keys
- **Handle file system edge cases** (locks, permissions, long names)
- **Plan for migration** when changing data structures
- **Distinguish between different completion states** (completed vs skipped)

**Date:** October 3, 2025 (learned from ChatGPT conversation about robust progress tracking)

### **Code Improvement Patterns - Systematic Enhancement**
**Pattern:** Apply systematic improvements to existing code based on external feedback
**Critical Insight:** External code reviews often identify patterns that internal developers miss

**The Systematic Improvement Process:**

1. **External Code Review:** Get fresh perspective from experienced developers
2. **Categorize Improvements:** Group suggestions by type (stability, portability, UX, robustness)
3. **Implement Systematically:** Apply all improvements of the same type together
4. **Document Patterns:** Capture the patterns for future reference

**Example: Progress Tracking Improvements (October 2025)**

**External Feedback Identified:**
- Persist immediately after reconciliation
- Normalize progress filenames
- Make IDs path-portable
- Distinguish skipped vs completed
- Add migration for old keys
- Handle file locks gracefully

**Systematic Implementation:**
```python
# 1. Stable ID Generation
def make_triplet_id(paths):
    s = "|".join(p.resolve().as_posix() for p in paths)
    return "t_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

# 2. Normalized Filenames
abs_base = self.base_directory.resolve()
safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', abs_base.as_posix())[:200]

# 3. Immediate Persistence
self.ensure_all_triplets_in_session_data()
self.save_progress()  # CRITICAL: Persist immediately

# 4. Status Distinction
def mark_status(self, status):
    # Mark as 'completed' or 'skipped'

# 5. Migration Support
def migrate_old_keys(self):
    # One-time migration for backward compatibility

# 6. Graceful Error Handling
except PermissionError:
    print("[!] Could not remove progress file (locked); ignoring.")
```

**Why This Pattern Works:**
1. **Fresh Perspective:** External reviewers see patterns internal developers miss
2. **Systematic Application:** All related improvements applied together
3. **Pattern Documentation:** Future developers can apply same patterns
4. **Comprehensive Coverage:** Addresses stability, portability, UX, and robustness

**When to Use This Pattern:**
- After major feature development
- When code has been in production for a while
- Before refactoring or major changes
- When external feedback is available

**Critical Success Factors:**
1. **Don't cherry-pick:** Apply all improvements of the same category
2. **Document the patterns:** Capture why each improvement matters
3. **Test thoroughly:** Systematic changes need comprehensive testing
4. **Update documentation:** Keep knowledge base current with new patterns

**Date:** October 3, 2025 (learned from systematic application of external code review feedback)

### Tool Behavior at a Glance

- Web Image Selector (`scripts/01_web_image_selector.py`):
  - Modern batch UI; exactly one selection per group; selected items move to `selected/`, others go to Trash by default (`send2trash`).
  - Requires `send2trash` unless `--hard-delete` is explicitly used (dangerous).
  - Uses centralized grouping; timestamps used only for sorting.
  - Moves companion files with images: YAML and/or caption files (when present).

- Desktop Image Selector + Crop (`scripts/01_desktop_image_selector_crop.py`):
  - Single-selection per triplet with immediate cropping; unselected files go to Trash.
  - Progress is persisted with stable, path-portable IDs; immediate persistence after reconciliation.
  - Uses the same centralized grouping and sorting rules as the web selector.
  - New flag: `--reset-progress` clears saved progress for the directory and rediscover groups from scratch.
  - Enter behavior: If no image is selected, Enter deletes all images in the current triplet and advances. If one image is selected, Enter crops it, deletes the others, and advances.
  - Moves companion files with images: YAML and/or caption files (when present).

### Glossary

- Group: Sequence of images with strictly increasing stage numbers (min size 2).
- Pair/Triplet: Group of size 2/3.
- Selected: The image chosen to keep for a group.
- Skipped: Leave files in place (web) or mark triplet as skipped (desktop).
- Trash/Delete: Non-selected images are sent to system Trash by default; hard delete is opt-in and risky.

### Troubleshooting (Quick List)

- “No groups found” → Ensure filenames have timestamps and stage tokens; confirm inputs are pre-sorted.
- “Unexpected grouping” → Check for duplicates or backward stage steps before expected completion.
- “Trash not available” → Install `send2trash` or run with `--hard-delete` (dangerous, avoid if unsure).
- Desktop crashes on navigation → Verify backend initialization and avoid recreating displays when count unchanged.

---

## 📝 **Workflow Principles**

### **During Work Sessions**
- Only fix bugs and make functional changes
- Log all changes in todo list for later test maintenance
- No tiny test fixes during active work

### **End of Day**
- Do cleanup and test fixes
- Commit changes
- Update documentation

### **File Safety**
- Never alter zip directory contents
- Always use send2trash for deletions
- Test file operations before implementing

---

## 🚨 **Critical Rules**

1. **Never alter zip directory contents** - only extract/copy from them
2. **Always activate virtual environment** before running scripts
3. **Only run scripts when testing or explicitly asked**
4. **Keep repository clean** - remove temporary files after use
5. **Always use PWD before creating directories/files**
6. **Never commit sensitive data to git** - always check .gitignore first

### **Git Safety Rules (October 2025)**

**Critical Discovery:** Sidecar files (*.decision) and project directories need explicit git protection.

**Required .gitignore Patterns:**
```
# Project directories (automatically added by 00_start_project.py)
mojo*/
selected/
crop/
crop_cropped/

# Sidecar and companion files
*.decision
*.yaml
*.caption

# Training data and logs
data/training/
data/ai_data/
data/file_operations_logs/

# Development files
__pycache__/
*.pyc
.DS_Store
```

**Why This Matters:**
1. Prevents accidental exposure of sensitive data
2. Protects AI training decisions and metadata
3. Keeps repository clean and focused

**Best Practices:**
1. Always check .gitignore before first commit in new project
2. Use `git rm -r --cached <directory>` to untrack accidentally committed files
3. Add new patterns to .gitignore BEFORE creating sensitive files
4. Run `git status` before commits to catch untracked files

**Automatic Protection:**
- `00_start_project.py` automatically adds new project directories to .gitignore
- Format: `{project_id}/` (e.g., "mojo3/")
- Prevents accidental commits of project data

---

## 📚 **Reference Links**

- **Style Guide:** `Documents/WEB_STYLE_GUIDE.md`
- **TODO List:** `Documents/CURRENT_TODO_LIST.md`
- **Test Suite:** `scripts/tests/test_runner.py`
- **File Tracker:** `scripts/file_tracker.py`
- **Activity Timer:** `scripts/utils/activity_timer.py`
- **Companion File Utils:** `scripts/utils/companion_file_utils.py`
- **Base Desktop Tool:** `scripts/utils/base_desktop_image_tool.py`

---

## 🧪 **Testing Patterns & Infrastructure**

### **Selenium Integration Testing (October 2025)**
**Achievement:** Complete Selenium test infrastructure for all web tools
**Impact:** Automated end-to-end verification of all Flask applications

**Infrastructure Components:**

1. **Base Selenium Test Class** (`test_base_selenium.py`):
```python
class BaseSeleniumTest(unittest.TestCase):
    """Base class with headless Chrome + Flask server management."""
    
    @classmethod
    def setUpClass(cls):
        # Set up Chrome driver once for all tests
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
    
    def setUp(self):
        # Create temp directory, start Flask server on free port
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server_port = find_free_port()
        app = self.get_flask_app()  # Subclass implements
        self.server_thread = threading.Thread(
            target=lambda: app.run(port=self.server_port),
            daemon=True
        )
        self.server_thread.start()
        self.wait_for_server()
```

2. **Smoke Tests for All Web Tools** (`test_web_tools_smoke.py`):
- Tests that each tool starts without errors
- Verifies page loads and displays content
- Checks that key UI elements are present
- Uses subprocess to launch actual Python scripts
- Runs in ~10 seconds for all 4 tools

3. **Key Features:**
- **Headless mode:** No browser windows pop up
- **Automatic port management:** Finds free ports automatically
- **Test isolation:** Each test gets temp directory + unique port
- **Clean teardown:** Servers terminated, pipes closed, no orphans
- **Real integration:** Actually launches your Flask apps as subprocesses

**Critical Lessons:**

1. **Coverage Limitation (Expected):**
Selenium tests that launch subprocesses don't show up in coverage reports. This is normal and fine:
- **Selenium tests verify:** Integration/functionality (does it work end-to-end?)
- **Unit tests verify:** Code coverage (are all code paths tested?)
- Both types of tests are important and complementary

2. **Subprocess Testing Pattern:**
```python
# Launch actual script as subprocess
self.process = subprocess.Popen(
    [sys.executable, str(script_path), args],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=project_root
)

# Wait for server to start
self.wait_for_server(port)

# Use Selenium to verify UI
self.driver.get(f"http://127.0.0.1:{port}")
self.assertIn("Expected Title", self.driver.title)

# Always clean up
def tearDown(self):
    if self.process:
        self.process.terminate()
        self.process.wait(timeout=5)
        # Close pipes to prevent ResourceWarning
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()
```

3. **Port Management:**
```python
def find_free_port() -> int:
    """Find a free port for the Flask server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
```

4. **Test Data Naming Conventions:**
For image grouping tests, use proper file naming:
```python
# Correct naming for grouping tests
descriptors = {1: "generated", 2: "upscaled", 3: "enhanced"}
for stage in [1, 2, 3]:
    filename = f"20250101_000000_stage{stage}_{descriptors[stage]}.png"
```

**Test Suite Organization:**

- `test_base_selenium.py` - Infrastructure base class
- `test_selenium_simple.py` - Infrastructure verification (3 tests)
- `test_web_tools_smoke.py` - Web tool smoke tests (4 tests)
- Total: 7 Selenium tests, all passing, ~10 second runtime

**Benefits:**
- Catch integration issues before production
- Verify tools actually start and work
- Test real browser interactions
- No manual testing needed for basic functionality

**When to Use:**
- Verifying Flask apps start correctly
- Testing UI elements are present
- Integration testing (multiple systems working together)
- Regression testing after major changes

**When NOT to Use:**
- Testing internal logic (use unit tests)
- Testing individual functions (use unit tests)
- Measuring code coverage (use unit tests)

---

### **Test Isolation Patterns**

**Critical Pattern:** Every test must run in complete isolation to prevent contamination.

**Implementation:**
```python
def setUp(self):
    # Create isolated temp directory
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_path = Path(self.temp_dir.name)
    
    # Set environment variable for test data root
    os.environ['EM_TEST_DATA_ROOT'] = str(self.temp_path)
    
    # Prepare test data in isolation
    self.prepare_test_data()

def tearDown(self):
    # Clean up environment variable
    if 'EM_TEST_DATA_ROOT' in os.environ:
        del os.environ['EM_TEST_DATA_ROOT']
    
    # Clean up temp directory
    if self.temp_dir:
        self.temp_dir.cleanup()
```

**Why This Matters:**
- Tests don't interfere with each other
- Tests don't pollute production data directories
- Tests are reproducible (same result every time)
- Can run tests in parallel safely

**Application Code Support:**
Production code should respect `EM_TEST_DATA_ROOT`:
```python
def get_data_directory():
    """Get data directory, respecting test environment."""
    if 'EM_TEST_DATA_ROOT' in os.environ:
        return Path(os.environ['EM_TEST_DATA_ROOT']) / 'data'
    return Path(__file__).parent.parent / 'data'
```

---

### **Flask App Testing Pattern**

**For simple unit tests (without browser):**
```python
def test_flask_route():
    app = create_app(test_data)
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
```

**For integration tests (with browser):**
Use BaseSeleniumTest pattern shown above - actually launch the app as a subprocess and test with real browser.

---

### **Headless Browser Configuration**

**Chrome Options for CI/CD:**
```python
chrome_options = ChromeOptions()
chrome_options.add_argument('--headless')          # No GUI
chrome_options.add_argument('--no-sandbox')        # Required for containers
chrome_options.add_argument('--disable-dev-shm-usage')  # Prevent crashes
chrome_options.add_argument('--disable-gpu')       # Not needed in headless
chrome_options.add_argument('--window-size=1920,1080')  # Set viewport
chrome_options.add_argument('--disable-extensions')     # Faster startup
chrome_options.add_argument('--disable-logging')        # Less noise
chrome_options.add_argument('--log-level=3')            # Errors only
```

**Why These Options:**
- `--headless`: No browser window (essential for automated testing)
- `--no-sandbox`: Required when running in Docker/CI environments
- `--disable-dev-shm-usage`: Prevents crashes when /dev/shm is too small
- Others: Performance and noise reduction

---

### **Coverage Report Interpretation**

**Expected Coverage Patterns:**
- **High coverage (>80%):** Utility functions, business logic, data processing
- **Medium coverage (40-80%):** Complex workflows, error handling paths
- **Low/Zero coverage (0%):** GUI tools, subprocess-launched apps, integration points

**Why Some Files Show 0% Coverage:**
1. **Desktop tools (tkinter):** Require GUI automation or headless X
2. **Web tools (Flask):** Routes not exercised by subprocess launches
3. **Integration tests:** Selenium launches subprocesses (separate Python process)

**This is NORMAL and EXPECTED.** Different test types serve different purposes:
- **Unit tests:** Code coverage, logic verification
- **Integration tests:** End-to-end functionality, system behavior
- **Smoke tests:** Does it start? Does it work basically?

---

### **Test Maintenance Workflow**

**During Active Development:**
1. Make functional changes
2. Log test impact in TODO list
3. Don't stop to fix tests immediately

**End of Day:**
1. Fix test failures caused by changes
2. Add new tests for new features
3. Update test data if schemas changed
4. Run full test suite before committing

**After Major Changes:**
1. Review test coverage report
2. Add tests for uncovered edge cases
3. Update test documentation
4. Consider integration tests if behavior changed

---

## 📚 **Documentation & Repository Management**

### **Document Consolidation Pattern**
**Achievement:** Reduced documentation from 39 to 19 files (51% reduction) while improving clarity
**Method:** Combine related documents with clear section headers and table of contents
**Impact:** Easier navigation, reduced decision paralysis, better searchability

**Examples of Effective Consolidation:**

1. **Case Studies** (2 files → 1):
   - Combined `professional_case_study_draft.md` + `image_workflow_case_study.md`
   - Result: `CASE_STUDIES.md` with clear section dividers
   - Benefit: Complete story in one place

2. **Dashboard Documentation** (3 files → 1):
   - Combined `DASHBOARD_README.md` + `DASHBOARD_QUICKSTART.md` + `DASHBOARD_SPECIFICATION.md`
   - Result: `DASHBOARD_GUIDE.md` with table of contents
   - Sections: Quick Start, Specification, API Reference, Troubleshooting

3. **Feature Specifications** (5 files → 1):
   - Combined all `SPEC_*.md` files
   - Result: `FEATURE_SPECIFICATIONS.md` with active/reference sections
   - Benefit: One source of truth for all specs

**Why This Pattern Works:**

1. **Cognitive Load Reduction:** One comprehensive guide beats three fragments
   - No decision paralysis ("which doc do I need?")
   - No context switching between files
   - Complete information in one read

2. **Better Search Experience:**
   - Search once, find everything related
   - Context preserved (related info nearby)
   - Natural reading flow from basic to advanced

3. **Easier Maintenance:**
   - Update one file instead of syncing three
   - Clear section boundaries prevent confusion
   - Table of contents acts as mini-index

4. **New Team Member Friendly:**
   - Fewer files to discover
   - Clear document purpose
   - Progressive complexity (basics first)

**When to Consolidate:**
- Documents about the same topic (dashboard, specs, case studies)
- Multiple "README" or "GUIDE" files for one system
- Short docs (<5K) that reference each other
- Docs with overlapping content

**When NOT to Consolidate:**
- Different audiences (developer vs user docs)
- Different lifecycle (active vs archived)
- Massive files (>50K) that would become unmanageable
- Truly independent topics

**Implementation Pattern:**
```markdown
# Consolidated Guide Title

**Comprehensive Documentation** - Last Updated: YYYY-MM-DD

---

# Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Guide](#detailed-guide)
3. [Reference](#reference)

---

# Quick Start

## 🚀 Get Started in 3 Steps

[Quick start content...]

---

# Detailed Guide

## Overview
[Comprehensive content...]

---

# Reference

## API Documentation
[Reference content...]
```

**Date:** October 16, 2025

### **Clear Naming Convention Pattern**
**Achievement:** Zero-ambiguity document naming system
**Rule:** Every document name must instantly communicate its purpose
**Impact:** Immediate comprehension, easy discovery, logical organization

**Naming Convention:**

**Prefix System:**
- `AI_*` - AI training, models, and automation
- `DASHBOARD_*` - Dashboard features, config, and specs
- `PROJECT_*` - Project lifecycle management
- `TOOL_*` - Specific tool documentation
- `AUTOMATION_*` - Workflow automation systems

**Examples of Good Names:**
- `AI_TRAINING_CROP_AND_RANKING.md` - Clear: AI training for crop/rank models
- `DASHBOARD_PRODUCTIVITY_TABLE_SPEC.md` - Clear: Dashboard feature spec
- `PROJECT_ALLOWLIST_SCHEMA.md` - Clear: Project-related schema
- `TOOL_MULTICROP_PROGRESS_TRACKING.md` - Clear: Tool-specific feature

**Examples of Bad Names (Replaced):**
- ❌ `PHASE2_QUICKSTART.md` - Vague: What is Phase 2?
  - ✅ `AI_TRAINING_PHASE2_QUICKSTART.md` - Clear: AI training guide

- ❌ `hand_foot_anomaly_scripts.md` - Unclear: Is this code or documentation?
  - ✅ `AI_ANOMALY_DETECTION_OPTIONS.md` - Clear: AI detection approaches

- ❌ `BASELINE_TEMPLATES_AND_README.md` - Vague: Baseline for what?
  - ✅ `DASHBOARD_BASELINE_TEMPLATES.md` - Clear: Dashboard baseline data

- ❌ `CENTRALIZED_TOOL_ORDER.md` - Abstract: Centralized where?
  - ✅ `DASHBOARD_TOOL_ORDER_CONFIG.md` - Clear: Dashboard config

**Why This Pattern Works:**

1. **Zero Cognitive Load:**
   - File name = exact purpose
   - No need to open file to know what it contains
   - Alphabetical sorting groups related docs

2. **Easy Discovery:**
   - New team member: "Where's the dashboard stuff?" → All files start with `DASHBOARD_`
   - Looking for AI docs? → All files start with `AI_`
   - Need project lifecycle info? → All files start with `PROJECT_`

3. **Scalability:**
   - Add 100 more docs → still organized
   - New categories → add new prefix
   - No reorganization needed

4. **Prevents Ambiguity:**
   - No generic names like "README" or "GUIDE"
   - No context-dependent names
   - Self-documenting directory listings

**Implementation Guidelines:**

1. **Choose the Right Prefix:**
   - What is the PRIMARY purpose?
   - What category does a user expect it in?
   - Is it general or specific?

2. **Be Specific, Not Generic:**
   - ❌ `AI_GUIDE.md` (too vague)
   - ✅ `AI_TRAINING_CROP_AND_RANKING.md` (specific)

3. **Use Underscores, Not Spaces:**
   - `AI_TRAINING_PHASE2` not `AI Training Phase2`
   - Consistent with code naming conventions

4. **Keep It Readable:**
   - Max 4-5 words after prefix
   - Use common abbreviations (SPEC, CONFIG, GUIDE)
   - Avoid unnecessary words

**Critical Rule:** If someone asks "what is Phase 2?", the name is bad. 
If the name requires explanation, it needs a better name.

**Date:** October 16, 2025

### **Unbuilt Feature Detection Pattern**
**Problem:** Design documents exist for features that were never implemented
**Impact:** Confusion, wasted time reading irrelevant docs, false expectations
**Solution:** Systematically identify and remove or archive unbuilt specs

**Detection Methods:**

1. **Code Search:**
```bash
# Check if feature exists in codebase
grep -r "feature_name" scripts/
```

2. **Git History:**
```bash
# Check if feature was ever implemented
git log --all --oneline | grep "feature_name"
```

3. **Reality Check:**
"If I've been working for months without this feature, do I actually need it?"

**Action Matrix:**

| Situation | Action | Reasoning |
|-----------|--------|-----------|
| Spec exists, no code, not needed | **DELETE** | Clutter, won't build it |
| Spec exists, no code, might build | **Move to experiments/** | Keep idea, mark as future |
| Spec exists, partially built | **Update or DELETE** | Document reality or remove confusion |
| Spec exists, fully built | **Keep** | Active documentation |

**Real Example:**
- **Found:** `TOOL_MULTICROP_PROGRESS_TRACKING.md` (6K detailed spec)
- **Checked:** No progress tracking code in `04_multi_crop_tool.py`
- **Checked:** Empty `scripts/crop_progress/` directory
- **Reality:** Tool works fine without it for months
- **Action:** DELETED (moved to Trash for safety)

**Why This Matters:**

1. **Reduces Confusion:** Readers don't waste time on features that don't exist
2. **Accurate Documentation:** Docs reflect reality, not aspirations
3. **Easier Maintenance:** Fewer files to keep updated
4. **Honest Communication:** New team members see what IS, not what WAS PLANNED

**Critical Rule:** Aspirational docs belong in `experiments/` or project planning tools, not main documentation directory.

**Date:** October 16, 2025

### **File Deletion Safety - macOS Trash Integration**
**Problem:** File deletion tool behavior may not match macOS Trash expectations
**Symptom:** Deleted files don't appear in Finder Trash
**Impact:** Potentially permanent deletion without recovery option
**Status:** Under investigation

**Critical Safety Issue:**

When using automated file deletion (via tool or script), files may be permanently removed without going to macOS Trash (`~/.Trash/`). This differs from Finder's behavior where deleted files are recoverable.

**Verified Safe Method:**
```bash
# Always use mv to Trash directory for safety
mv unwanted_file.md ~/.Trash/
```

**Why This Matters:**

1. **No Undo:** Permanent deletion is irreversible
2. **User Expectation:** Users expect Trash behavior
3. **Safety Net:** Trash provides recovery window
4. **Peace of Mind:** Can verify deletion before emptying Trash

**Best Practices:**

1. **Always Use Trash First:**
```bash
# Safe deletion pattern
mv document_to_remove.md ~/.Trash/
```

2. **Test with Dummy Files:**
```bash
# Create test file
echo "test" > /tmp/test_deletion.txt
mv /tmp/test_deletion.txt ~/.Trash/
# Verify it appears in Finder Trash
```

3. **Batch Deletion Pattern:**
```bash
# For multiple files
for file in *.old; do
    mv "$file" ~/.Trash/
done
```

4. **Keep Important Files in Version Control:**
- Git tracks all changes
- Can recover from any commit
- Provides audit trail

**Never Use:**
```bash
# DANGEROUS - Permanent deletion
rm -f important_file.md      # No recovery
rm -rf directory/            # Mass destruction
```

**Recovery Options If Files Are Permanently Deleted:**

1. **Git Repository:**
```bash
# Check if file was committed
git log --all --full-history -- "path/to/file"
git checkout <commit-hash> -- "path/to/file"
```

2. **Time Machine (if enabled):**
- Open Time Machine
- Navigate to timestamp before deletion
- Restore file

3. **No Version Control + No Backup = GONE**
- Emphasizes importance of using Trash

**Implementation in Scripts:**

```python
# Safe deletion in Python
import shutil
from pathlib import Path

def safe_delete(file_path):
    """Move file to Trash instead of permanent deletion."""
    trash_dir = Path.home() / ".Trash"
    destination = trash_dir / file_path.name
    
    # Handle name collisions
    counter = 1
    while destination.exists():
        destination = trash_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1
    
    shutil.move(str(file_path), str(destination))
    return destination
```

**Critical Rule:** When in doubt, use `mv` to `~/.Trash/`. Convenience is never worth the risk of permanent data loss.

**Date:** October 16, 2025

---

*Last Updated: October 16, 2025*
*This file should be updated whenever new technical solutions are discovered or patterns are established.*
