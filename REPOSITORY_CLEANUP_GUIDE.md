# Repository Cleanup Guide - Updated September 2025

## Purpose
Keep the repository clean and organized by removing temporary files while preserving essential documentation and productivity data.

## Current System Overview
- **Production Scripts**: 01-05 workflow tools with activity timer integration
- **Dashboard System**: Complete productivity analytics in `scripts/dashboard/`
- **Data Protection**: `scripts/timer_data/` and `scripts/file_operations_logs/` auto-ignored
- **Documentation**: Core workflow docs + productivity system docs

## File Categories & Cleanup Rules

### ✅ **KEEP - Essential Core Files**

#### Production Workflow Scripts
- `scripts/01_web_image_selector.py` - Primary workflow tool ✅
- `scripts/03_web_character_sorter.py` - Character sorting tool ✅
- `scripts/04_batch_crop_tool.py` - Batch cropping tool ✅
- `scripts/05_web_multi_directory_viewer.py` - Review tool ✅
- `scripts/util_activity_timer.py` - Shared timer utility ✅
- `scripts/util_triplet_deduplicator.py` - Triplet deduplication utility ✅
- `scripts/file_tracker.py` - Operation logging utility ✅
- All other `scripts/util_*.py` files - Workflow utilities ✅

#### Dashboard System (Complete - Keep All)
- `scripts/dashboard/` - **Entire directory is essential** ✅
  - `run_dashboard.py` - Launch script
  - `productivity_dashboard.py` - Main Flask app
  - `data_engine.py` - Data processing
  - `dashboard_template.html` - Web interface
  - `data_retention_policy.py` - Storage management
  - `Productivity Dashboard Specification.md` - Requirements doc
  - `script_updates.csv` - Update tracking

#### Core Documentation (Critical - Never Remove)
- `IMAGE_PROCESSING_WORKFLOW.md` - **Master workflow documentation** ✅
- `WEB_STYLE_GUIDE.md` - **Design system reference** ✅
- `CURRENT_TODO_LIST.md` - **Project status tracking** ✅
- `REPOSITORY_CLEANUP_GUIDE.md` - **This file** ✅
- `.gitignore` - **Repository configuration** ✅

#### Productivity System Documentation (Keep)
- `ACTIVITY_TIMER_SYSTEM_OVERVIEW.md` - Timer system architecture ✅
- `TIMER_WEB_INTERFACE_DEMO.md` - UI documentation ✅
- `image_workflow_case_study.md` - **Living document** - Update with dashboard insights ✅

#### Test Infrastructure (Keep)
- `scripts/tests/` - **Entire directory** ✅
  - `README.md` - Test documentation
  - `README_WEB_TESTS.md` - Web test guide
  - All test files for workflow scripts

### 🧹 **REMOVE - Temporary/Outdated Files**

#### Superseded Planning Documents
- `timer_dashboard_plan.md` - ❌ Remove (dashboard now complete)
- `clustering_optimization_workflow.md` - ❌ Remove if clustering work complete
- `IMAGE_PROCESSING_WORKFLOW_FLOWCHART.md` - ❌ Remove if superseded by main workflow doc

#### Old Experiment Documentation
- `image_dataset_optimization_and_cropping.md` - ❌ Remove if superseded
- Any `*_comparison.md` files from failed experiments

#### Recovery & One-Off Scripts (Project Root)
- `recover_images.py` - ❌ Remove after recovery complete
- Any `*_recovery.py`, `*_temp.py`, or `*_test.py` files in root
- `demo_*.py` files after demos complete

#### File Lists & Inventories
- `*_files.txt` (crop_files.txt, white_files.txt) - ❌ Remove after operation
- `cropped_to_replace.txt` - ❌ Remove after replacement complete
- Any `*_to_*.txt` inventory files

### 🔒 **PROTECTED - Auto-Ignored (Never Manually Remove)**
- `scripts/timer_data/` - **Productivity data** (ignored by git)
- `scripts/file_operations_logs/` - **Operation logs** (ignored by git)
- `quality_log_*.csv` - **Quality metrics** (ignored by git)
- `people_scores*.csv` - **Scoring data** (ignored by git)

## Cleanup Timing & Procedures

### After Major Operations
- **Dashboard Development**: Remove planning docs (`timer_dashboard_plan.md`)
- **Image Recovery**: Remove recovery scripts, file lists, comparison files
- **Workflow Updates**: Remove old numbered scripts, temp docs  
- **Clustering Experiments**: Remove failed attempt files, old analysis docs
- **Documentation Updates**: Remove superseded workflow files

### Monthly Review
- Check for files > 30 days old in "REMOVE" categories above
- Evaluate experiment documentation for continued relevance
- Remove outdated planning documents after project completion

### Quarterly Review
- Assess all markdown files in root directory
- Remove superseded case studies and optimization docs
- Clean up any accumulated temporary files

## Safe Cleanup Commands

### Identify Candidates for Removal
```bash
# List all markdown files for review
find . -maxdepth 1 -name "*.md" | grep -v -E "(IMAGE_PROCESSING_WORKFLOW|WEB_STYLE_GUIDE|CURRENT_TODO_LIST|REPOSITORY_CLEANUP_GUIDE|ACTIVITY_TIMER|TIMER_WEB_INTERFACE)"

# List temporary files
find . -maxdepth 1 -name "*_files.txt" -o -name "*_temp.py" -o -name "*_recovery.py"

# List old demo files
find . -maxdepth 1 -name "demo_*.py"
```

### Safe Removal (After Manual Review)
```bash
# Remove confirmed temporary files
rm -f timer_dashboard_plan.md  # Dashboard complete
rm -f *_files.txt cropped_to_replace.txt  # After operations complete
rm -f recover_images.py demo_*.py  # After demos/recovery complete

# Remove old experiment docs (ONLY after manual review)
# rm -f clustering_optimization_workflow.md  # If clustering work complete
# NOTE: image_workflow_case_study.md is PROTECTED - living document for dashboard insights
```

## Data Protection Verification

### Verify .gitignore Protection
```bash
# Confirm data directories are ignored
git check-ignore scripts/timer_data/
git check-ignore scripts/file_operations_logs/

# Should return the paths (meaning they're ignored)
```

### Check for Accidentally Tracked Data
```bash
# Ensure no data files are tracked
git ls-files | grep -E "(timer_data|file_operations_logs|quality_log|people_scores)"

# Should return empty (no data files tracked)
```

## Git Integration

### Cleanup Commit Pattern
```bash
# Stage specific files for removal
git rm timer_dashboard_plan.md
git rm clustering_optimization_workflow.md  # If confirmed obsolete

# Commit with clear message
git commit -m "🧹 Repository cleanup: remove completed planning docs

- Removed timer_dashboard_plan.md (dashboard system complete)
- Removed clustering_optimization_workflow.md (work complete)
- Keeping all essential workflow and system documentation"
```

## Critical Reminders

### ⚠️ **NEVER REMOVE**
- **Entire `scripts/dashboard/` directory** - Complete productivity system
- **Core workflow scripts** (01, 03, 04, 05) - Production tools
- **Master documentation** (IMAGE_PROCESSING_WORKFLOW.md, WEB_STYLE_GUIDE.md)
- **System documentation** (ACTIVITY_TIMER_SYSTEM_OVERVIEW.md, etc.)
- **Living case study** (image_workflow_case_study.md) - Update with dashboard insights
- **Test infrastructure** (`scripts/tests/` directory)

### ✅ **Safe to Remove After Verification**
- Planning documents for completed projects
- Temporary file lists and inventories
- Old experiment documentation
- Recovery scripts after recovery complete
- Demo files after demos complete

### 🔒 **Protected by .gitignore**
- All files in `scripts/timer_data/` (productivity data)
- All files in `scripts/file_operations_logs/` (operation logs)
- CSV files matching `quality_log_*.csv` and `people_scores*.csv`

## Usage Notes
This guide should be referenced before any cleanup operations. **Always manually review files before removal** - when in doubt, keep the file. The goal is maintaining a clean repository while preserving all essential workflow tools, documentation, and productivity systems.
