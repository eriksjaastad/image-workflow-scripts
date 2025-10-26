---
title: Repository Cleanup Guide
status: Current
audience: DEVELOPER
tags: [cleanup, policy, repository, protected-files]
last_updated: 2025-10-16
---

**Last Updated:** 2025-10-26


# Repository Cleanup Guide - Updated October 2025

## Purpose
Keep the repository clean and organized by removing temporary files while preserving essential documentation, productivity data, and AI training infrastructure.

## Current System Overview (October 2025)
- **Production Scripts**: Web-based workflow tools (01, 02, 04, 05)
- **Dashboard System**: Complete productivity analytics with snapshot-based data pipeline
- **Data Protection**: `data/snapshot/`, `data/file_operations_logs/` auto-ignored
- **Documentation**: 19 well-organized documents with clear naming conventions
- **AI Training**: Passive data collection infrastructure for future automation
- **Archive**: Historical scripts preserved in `scripts/archive/`

## File Categories & Cleanup Rules

### ✅ **KEEP - Essential Core Files**

#### Production Workflow Scripts
- `scripts/01_ai_assisted_reviewer.py` - Primary version selection tool ✅
- `scripts/02_web_character_sorter.py` - Character sorting tool ✅
- `scripts/04_multi_crop_tool.py` - Batch cropping tool ✅
- `scripts/05_web_multi_directory_viewer.py` - Review tool ✅
- `scripts/06_web_duplicate_finder.py` - Duplicate detection ✅

#### Project Lifecycle Scripts
- `scripts/00_start_project.py` - Initialize new projects ✅
- `scripts/00_finish_project.py` - Complete projects with ZIP creation ✅

#### Utility Infrastructure
- `scripts/utils/*.py` - **All utility modules** ✅
  - `companion_file_utils.py` - File operations with companions
  - `base_desktop_image_tool.py` - Base class for desktop tools
  - `file_tracker.py` - Operation logging utility
  - All other utility modules

#### Dashboard System (Complete - Keep All)
- `scripts/dashboard/` - **Entire directory is essential** ✅
  - `productivity_dashboard.py` - Main Flask app
  - `data_engine.py` - Data processing
  - `analytics.py` - Aggregation and transformation
  - `snapshot_loader.py` - Snapshot data loader
  - `project_metrics_aggregator.py` - Project-level metrics
  - `dashboard_template.html` - Web interface
  - `tools/` - Debug and utility scripts

#### Data Pipeline (Complete - Keep All)
- `scripts/data_pipeline/` - **Entire directory is essential** ✅
  - `extract_operation_events_v1.py` - Event extraction
  - `build_daily_aggregates_v1.py` - Daily aggregation
  - `derive_sessions_from_ops_v1.py` - Session derivation
  - `extract_timer_sessions_v1.py` - Legacy timer extraction
  - `extract_progress_snapshots_v1.py` - Progress snapshot extraction
  - `extract_projects_v1.py` - Project manifest extraction
  - `backfill_snapshots_v1.py` - Historical data migration

#### AI Training Infrastructure (Keep All)
- `scripts/ai/` - **AI training and automation** ✅
  - `training_snapshot.py` - Training data capture
  - `compute_training_features.py` - Feature extraction
  - `README.md` - AI system documentation

#### Tools & Utilities
- `scripts/tools/` - **Utility scripts** ✅
  - `prezip_stager.py` - Delivery ZIP creation
  - `scan_dir_state.py` - Directory state scanning
  - Other utility scripts

#### Core Documentation (Critical - Never Remove)
- `Documents/CURRENT_TODO_LIST.md` - **Project status tracking** ✅
- `Documents/REPOSITORY_CLEANUP_GUIDE.md` - **This file** ✅
- `Documents/TECHNICAL_KNOWLEDGE_BASE.md` - **Technical patterns** ✅
- `Documents/DASHBOARD_GUIDE.md` - **Dashboard documentation** ✅
- `Documents/CASE_STUDIES.md` - **Workflow case studies** ✅
- `Documents/FEATURE_SPECIFICATIONS.md` - **Active feature specs** ✅
- `Documents/WEB_STYLE_GUIDE.md` - **Design system reference** ✅
- All other `Documents/*.md` files with clear naming ✅

#### AI & Automation Documentation
- `Documents/AI_TRAINING_CROP_AND_RANKING.md` - Training plan ✅
- `Documents/AI_TRAINING_PHASE2_QUICKSTART.md` - Setup guide ✅
- `Documents/AI_ANOMALY_DETECTION_OPTIONS.md` - Detection approaches ✅
- `Documents/AUTOMATION_REVIEWER_SPEC.md` - Review UI spec ✅

#### Dashboard Configuration
- `Documents/DASHBOARD_*` files - All dashboard-related docs ✅

#### Project Lifecycle Documentation
- `Documents/PROJECT_*` files - All project-related docs ✅

#### Configuration Files
- `configs/metrics_config.json` - **Dashboard and pipeline config** ✅
- `.gitignore` - **Repository configuration** ✅
- `.coverage` - **Test coverage config** ✅

#### Test Infrastructure (Keep All)
- `scripts/tests/` - **Entire directory** ✅
  - All test files for workflow scripts
  - `WEB_TESTS_GUIDE.md` - Web test guide
  - Test fixtures and utilities

#### Experiments Directory
- `Documents/experiments/` - **Active experimental work** ✅
  - `automation_reduction_experiments.md` - Sandbox experiments

#### Archive (Historical Reference)
- `scripts/archive/` - **Archived scripts** ✅
  - Old versions preserved for reference
  - Can be reviewed quarterly for permanent removal

### 🧹 **REMOVE - Temporary/Outdated Files**

#### Temporary Validation Files
- `checks/` - ❌ Remove validation directories after data migration
- `scripts/checks/` - ❌ Remove temporary check artifacts
- Any `validation_*.json` files after verification complete

#### Old Session Data
- `memory/` directory - ❌ Remove (superseded by AI Journal system)
  - `decisions.md` - Old architectural decisions
  - `journal.md` - Old development journal
  - `state.json` - Old session state

#### Root Directory Clutter
- Any `*.md` files in root except `cursor_global_rules_kit.md` ❌
- `Data-dashboard-review.md` - ❌ Remove after review complete
- Any `*_SUMMARY.md` or `*_STATUS.md` temporary files ❌
- Any `READY_TO_TEST.md` or similar temporary markers ❌

#### Backup Files
- `*.backup` files - ❌ Remove after verification
- `*_old.py` files - ❌ Remove after migration complete
- Any `.bak` files - ❌ Remove

#### Temporary Debug Files
- `debug_*.py` in root or scripts/ - ❌ Remove after debugging
- `test_*.py` in root - ❌ Move to tests/ or remove
- Any `scratch_*.py` files - ❌ Remove

#### Obsolete Documentation
- Any docs referencing deleted features - ❌ Remove
- Duplicate documentation (check for content, not just filename) - ❌ Remove
- Planning docs for completed features - ❌ Remove or archive

### 🔒 **PROTECTED - Auto-Ignored (Never Manually Remove)**
- `data/snapshot/` - **Pre-aggregated data** (ignored by git)
- `data/file_operations_logs/` - **Operation logs** (ignored by git)
- `data/daily_summaries/` - **Legacy summaries** (ignored by git)
- `data/timer_data/` - **Legacy timer data** (ignored by git)
- `data/log_archives/` - **Archived logs** (ignored by git)
- `data/ai_data/` - **AI training data** (ignored by git)
- `data/training/` - **Training datasets** (ignored by git)
- `.venv311/` - **Virtual environment** (ignored by git)
- `__pycache__/` - **Python cache** (ignored by git)
- AI training sidecar files: `*.embedding.npy`, `*.phash`, `*.saliency.npy`, `*.hands.json` (ignored by git)

## Cleanup Timing & Procedures

### After Major Operations
- **Data Pipeline Refactoring**: Remove validation directories, temporary check files
- **Dashboard Development**: Remove temporary debug scripts, status markers
- **Documentation Consolidation**: Remove superseded individual docs
- **AI Training Setup**: Remove temporary feature extraction test files
- **Project Completion**: Remove project-specific temporary files

### Weekly Review
- Check root directory for temporary markdown files
- Remove debug scripts and validation artifacts
- Check for backup files (*.backup, *.bak)
- Remove any temporary status files

### Monthly Review
- Review `Documents/experiments/` for completed experiments
- Check `scripts/archive/` - consider permanent removal of very old scripts
- Review `memory/` directory candidates
- Check for temporary directories (checks/, validation/, etc.)

### Quarterly Review
- Assess all archived scripts for permanent removal
- Review all documentation for continued relevance
- Clean up any accumulated temporary files
- Update this guide with new patterns

## Safe Cleanup Commands

### Identify Candidates for Removal

```bash
# List markdown files in root (should only be cursor_global_rules_kit.md)
ls -1 *.md 2>/dev/null

# List temporary directories
find . -maxdepth 1 -type d -name "checks" -o -name "validation" -o -name "memory"

# List backup files
find . -name "*.backup" -o -name "*.bak" -o -name "*_old.py"

# List debug files
find . -maxdepth 1 -name "debug_*.py" -o -name "test_*.py"

# Check for temporary status files
ls -1 *_SUMMARY.md *_STATUS.md READY_*.md 2>/dev/null
```

### Safe Removal (After Manual Review)

```bash
# Move to Trash (safer than rm)
mv memory ~/.Trash/
mv checks ~/.Trash/
mv Data-dashboard-review.md ~/.Trash/

# Remove backup files after verification
find . -name "*.backup" -exec mv {} ~/.Trash/ \;

# Remove debug scripts after debugging complete
mv debug_*.py ~/.Trash/
```

## Document Naming Convention

### Current Standard (October 2025)
All documents follow a clear prefix system:

- `AI_*` - AI training, models, and automation
- `DASHBOARD_*` - Dashboard features, config, and specs
- `PROJECT_*` - Project lifecycle management
- `TOOL_*` - Specific tool documentation
- `AUTOMATION_*` - Workflow automation systems

**Rule:** Every document name must instantly communicate its purpose.

**Examples:**
- ✅ `AI_TRAINING_CROP_AND_RANKING.md` - Clear
- ✅ `../../dashboard/DASHBOARD_PRODUCTIVITY_TABLE_SPEC.md` - Clear
- ✅ `PROJECT_ALLOWLIST_SCHEMA.md` - Clear
- ❌ `PHASE2_QUICKSTART.md` - Vague (what is Phase 2?)

### Cleanup Impact
If a document name requires explanation, it needs:
1. Better naming (follow prefix convention)
2. Consolidation with related docs
3. Removal if obsolete

## Data Protection Verification

### Verify .gitignore Protection

```bash
# Confirm data directories are ignored
git check-ignore data/snapshot/
git check-ignore data/file_operations_logs/
git check-ignore data/ai_data/

# Should return the paths (meaning they're ignored)
```

### Check for Accidentally Tracked Data

```bash
# Ensure no sensitive data files are tracked
git ls-files | grep -E "(snapshot|file_operations_logs|timer_data|ai_data|training)"

# Should return empty (no data files tracked)

# Check for AI sidecar files
git ls-files | grep -E "\.(embedding\.npy|phash|saliency\.npy|hands\.json)$"

# Should return empty (sidecar files ignored)
```

## Git Integration

### Cleanup Commit Pattern

```bash
# Stage specific files for removal
git rm memory/decisions.md memory/journal.md memory/state.json
git rm -r checks/
git rm Data-dashboard-review.md

# Commit with clear message
git commit -m "🧹 Repository cleanup: remove temporary files

- Removed memory/ directory (superseded by AI Journal)
- Removed checks/ validation directories (migration complete)
- Removed temporary status files
- Keeping all essential workflow and documentation"
```

### File Deletion Safety

**CRITICAL:** Always use `mv` to `~/.Trash/` instead of `rm`:

```bash
# Safe deletion pattern
mv unwanted_file.md ~/.Trash/

# Batch deletion
for file in *.backup; do
    mv "$file" ~/.Trash/
done
```

**Why:** Files in Trash can be recovered. Permanent deletion (`rm`) is irreversible.

## Critical Reminders

### ⚠️ **NEVER REMOVE**
- **Entire `scripts/dashboard/` directory** - Complete productivity system
- **Entire `scripts/data_pipeline/` directory** - Data processing infrastructure
- **Entire `scripts/ai/` directory** - AI training system
- **Core workflow scripts** (01, 02, 04, 05) - Production tools
- **Entire `Documents/` directory** - All documentation (19 organized files)
- **configs/metrics_config.json** - Dashboard and pipeline configuration
- **Test infrastructure** (`scripts/tests/` directory)
- **Utility modules** (`scripts/utils/` directory)

### ✅ **Safe to Remove After Verification**
- `memory/` directory - Superseded by AI Journal system
- `checks/` directories - Temporary validation artifacts
- Root markdown files except `cursor_global_rules_kit.md`
- Backup files (*.backup, *.bak, *_old.py)
- Debug scripts after debugging complete
- Temporary status/summary files

### 🔒 **Protected by .gitignore**
- All files in `data/snapshot/` (pre-aggregated data)
- All files in `data/file_operations_logs/` (operation logs)
- All files in `data/ai_data/` (AI training data)
- All AI sidecar files (*.embedding.npy, *.phash, etc.)
- Virtual environment (`.venv311/`)
- Python cache (`__pycache__/`)

## Document Organization Pattern

### Consolidation (October 2025 Pattern)
When multiple documents cover the same topic:

1. **Combine into one comprehensive guide** with:
   - Clear table of contents
   - Distinct sections with headers
   - Progressive complexity (basics → advanced)

2. **Use clear naming** with prefixes:
   - Multiple dashboard docs → `../../dashboard/DASHBOARD_GUIDE.md`
   - Multiple specs → `FEATURE_SPECIFICATIONS.md`
   - Multiple case studies → `../../reference/CASE_STUDIES.md`

3. **Benefits:**
   - Easier to read one guide than three fragments
   - Better search experience (everything in one place)
   - Reduced decision paralysis
   - Easier maintenance

### When to Keep Separate
- Different audiences (developer vs user)
- Different lifecycle (active vs archived)
- Truly independent topics
- Very large files (>50K) that would become unmanageable

## Usage Notes

This guide should be referenced before any cleanup operations. **Always manually review files before removal** - when in doubt, keep the file and mark it for quarterly review.

The goal is maintaining a clean repository while preserving all essential workflow tools, documentation, productivity systems, and AI training infrastructure.

### Cleanup Checklist

Before major cleanup:
1. ✅ Read this entire guide
2. ✅ Identify candidates using safe commands above
3. ✅ Manual review of each file
4. ✅ Use `mv ~/.Trash/` (not `rm`)
5. ✅ Verify .gitignore protection
6. ✅ Test that nothing broke
7. ✅ Commit with clear message
8. ✅ Update this guide if new patterns emerge

---

*Last Updated: October 16, 2025*  
*Major updates: Document consolidation, AI training infrastructure, snapshot-based data pipeline*
