# Finish Project Integration with Prezip Stager

## Overview

**`00_finish_project.py`** is now a friendly wrapper around `prezip_stager.py` that provides an interactive interface for completing projects.

Instead of duplicating the manifest update logic, it delegates to prezip_stager which already handles:
- ✅ Directory state validation (FULL check)
- ✅ Allowlist-based file filtering
- ✅ ZIP file creation for client delivery
- ✅ Manifest updates (finishedAt, stager metrics)
- ✅ Backup creation before updating

---

## Workflow

### Before (Manual):
```bash
# Step 1: Manually edit manifest JSON
vi data/projects/mojo3.project.json
# Set finishedAt, finalImages, status...

# Step 2: Run prezip_stager separately
python scripts/tools/prezip_stager.py \
  --project-id mojo3 \
  --content-dir mojo3 \
  --output-zip exports/mojo3_final.zip \
  --commit --update-manifest
```

### After (Integrated):
```bash
# Single command with preview
python scripts/00_finish_project.py --project-id mojo3

# Commit when ready
python scripts/00_finish_project.py --project-id mojo3 --commit
```

---

## What `00_finish_project.py` Does

### 1. **Validation**
- Checks project manifest exists
- Loads project configuration
- Verifies content directory exists

### 2. **Auto-Count Final Images**
- Counts PNG files in content directory
- No manual counting needed!

### 3. **Calls Prezip Stager**
```python
subprocess.run([
    "python", "scripts/tools/prezip_stager.py",
    "--project-id", project_id,
    "--content-dir", content_dir,
    "--output-zip", f"exports/{project_id}_final.zip",
    "--commit",  # Only if not dry-run
    "--update-manifest"  # Only if not dry-run
])
```

### 4. **Reports Results**
- Shows stager output
- Displays final metrics
- Provides next steps

---

## Dry Run Mode (Default)

**Safety first!** The script defaults to dry-run mode:

```bash
# Preview what will happen (no changes)
python scripts/00_finish_project.py --project-id mojo3

# Output shows:
# - What files will be included
# - What will be excluded
# - ZIP size estimate
# - Manifest changes preview
```

**When you're ready:**
```bash
# Commit changes
python scripts/00_finish_project.py --project-id mojo3 --commit
```

---

## Interactive Mode

```bash
python scripts/00_finish_project.py
```

Provides friendly prompts:
1. Lists active projects
2. Asks which project to finish
3. Shows preview of changes
4. Asks if you want dry-run first (recommended!)
5. Runs prezip_stager
6. Shows results and next steps

---

## What Prezip Stager Does

When called by `00_finish_project.py`, prezip_stager:

### 1. Validates Directory State
```python
# Checks if directory is ready for archiving
state = scan_dir_state(content_dir, recent_mins=10)
if state != 'FULL':
    return error("Directory not ready - recent edits detected")
```

### 2. Filters Files with Allowlist
- Loads `data/projects/{project}_allowed_ext.json`
- Loads `data/projects/global_bans.json`
- Filters files based on:
  - Allowed extensions (png, yaml, txt, caption)
  - Banned patterns (dotfiles, internal files)
  - Client whitelist overrides

### 3. Creates ZIP
- Compresses files into `exports/{project}_final.zip`
- Uses efficient streaming (no staging directory)
- Supports multiple compression methods

### 4. Updates Manifest
```json
{
  "finishedAt": "2025-10-25T14:20:00Z",
  "status": "finished",
  "counts": {
    "finalImages": 5432
  },
  "metrics": {
    "stager": {
      "zip": "exports/mojo3_final.zip",
      "eligibleCount": 15487,
      "byExtIncluded": {
        "png": 5432,
        "yaml": 5432,
        "caption": 4200,
        "txt": 423
      },
      "excludedCounts": {
        "hidden": 2,
        "banned_ext": 15
      }
    }
  }
}
```

### 5. Creates Backup
- Saves `{project}.project.json.bak` before updating
- Safe rollback if needed

---

## Benefits of Integration

### ✅ **Single Command**
No need to remember prezip_stager arguments

### ✅ **Dry Run Safety**
Preview before committing

### ✅ **Auto-Count Images**
No manual counting required

### ✅ **Unified Workflow**
Start project → Process images → Finish project
All with simple commands

### ✅ **No Duplication**
Reuses existing prezip_stager logic
No maintaining duplicate code

### ✅ **Comprehensive Validation**
Directory state check prevents premature archiving

---

## Examples

### Finish Active Project
```bash
$ python scripts/00_finish_project.py --project-id mojo3
📊 Found 5,432 PNG images in /path/to/mojo3

🔍 DRY RUN: python scripts/tools/prezip_stager.py ...
======================================================================
Status: success
Eligible files: 15,487
ZIP size (estimated): 2.3 GB
Excluded: 17 files (2 hidden, 15 banned extensions)

✅ Preview successful! No changes made.

To finalize, run again with:
  python scripts/00_finish_project.py --project-id mojo3 --commit
```

### Commit Changes
```bash
$ python scripts/00_finish_project.py --project-id mojo3 --commit
📊 Found 5,432 PNG images in /path/to/mojo3

🚀 EXECUTING: python scripts/tools/prezip_stager.py ...
======================================================================
Creating ZIP: exports/mojo3_final.zip
Compressing 15,487 files...
✅ ZIP created: 2.3 GB
✅ Manifest updated: data/projects/mojo3.project.json

======================================================================
✅ SUCCESS!
======================================================================
Manifest updated:  data/projects/mojo3.project.json
Project ID:        mojo3
Finished At:       2025-10-25T14:20:00Z
Final Images:      5,432
Output ZIP:        exports/mojo3_final.zip
ZIP Contents:      15,487 files

🎯 Next steps:
   • Upload: exports/mojo3_final.zip
   • View dashboard for final metrics
======================================================================
```

---

## Command-Line Reference

### Finish Project (Dry Run)
```bash
python scripts/00_finish_project.py --project-id mojo3
```

### Finish Project (Commit)
```bash
python scripts/00_finish_project.py --project-id mojo3 --commit
```

### Force Overwrite
```bash
python scripts/00_finish_project.py --project-id mojo3 --commit --force
```

### Interactive Mode
```bash
python scripts/00_finish_project.py
```

---

## Architecture

```
00_finish_project.py (Friendly UI)
  ↓
  Validates project
  Counts images
  Builds command
  ↓
  Calls prezip_stager.py (Heavy lifting)
    ↓
    Validates directory state
    Filters files (allowlist + bans)
    Creates ZIP
    Updates manifest
    Creates backup
    ↓
    Returns results
  ↓
  Displays results
  Shows next steps
```

**Separation of Concerns:**
- `00_finish_project.py`: User experience, validation, orchestration
- `prezip_stager.py`: ZIP creation, file filtering, manifest updates

---

## Error Handling

### Directory Not Ready
```
❌ ERROR: Directory not FULL (state=PENDING)
Recent edits detected. Wait for stability or use --no-require-full
```

### Missing Allowlist
```
❌ ERROR: Allowlist not found: data/projects/mojo3_allowed_ext.json
Create allowlist first using inventory snapshot
```

### Project Already Finished
```
⚠️ WARNING: Project already finished at 2025-10-20T10:00:00Z
Use --force to overwrite
```

---

## Summary

**`00_finish_project.py` + `prezip_stager.py` = Complete Project Completion**

- Simple interface for users
- Comprehensive validation and processing
- Safe dry-run preview
- Automatic ZIP creation
- Manifest updates with metrics
- Single unified workflow

**Result**: Finishing projects is now as easy as starting them! 🎉

---

**Last Updated**: October 15, 2025  
**Status**: Integrated and tested  
**Related**: `PROJECT_LIFECYCLE_SCRIPTS.md`, `SPEC_PROJECT_CLI.md`

