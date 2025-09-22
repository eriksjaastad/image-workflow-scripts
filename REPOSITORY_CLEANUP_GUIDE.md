# Repository Cleanup Guide

## Purpose
Keep the repository clean and organized by removing temporary files after they've served their purpose.

## File Categories & Cleanup Rules

### ✅ **KEEP - Core Project Files**
- `scripts/` - All numbered workflow scripts (01-06) and utilities
- `IMAGE_PROCESSING_WORKFLOW.md` - Main workflow documentation 
- `clustering_optimization_workflow.md` - Active project planning document
- `.gitignore` - Repository configuration

### 🧹 **REMOVE - Temporary Files**

#### Recovery & One-Off Scripts
- `recover_images.py` - ❌ Remove after image recovery complete
- Any `*_recovery.py` or `*_temp.py` files

#### File Lists & Inventories  
- `*_files.txt` (crop_files.txt, white_files.txt) - ❌ Remove after operation
- `cropped_to_replace.txt` - ❌ Remove after replacement complete
- Any `*_to_*.txt` inventory files

#### Old Comparison & Analysis Files
- `face_grouping_comparison.md` - ❌ Remove if from failed experiments
- Outdated comparison documents from previous attempts

#### Old CSV Data
- `people_scores*.csv` - ❌ Remove if > 30 days old and not referenced
- `quality_log_level3_*.csv` - ❌ Remove old quality logs (keep latest only)
- Any CSV files from failed clustering attempts

### 📋 **EVALUATE - Context-Dependent**

#### Workflow Documentation  
- `IMAGE_PROCESSING_WORKFLOW_FLOWCHART.md` - Keep if current, remove if outdated
- `image_workflow_case_study.md` - Keep if referenced, remove if superseded

## Cleanup Timing

### After Major Operations
- **Image Recovery**: Remove recovery scripts, file lists, comparison files
- **Workflow Reorganization**: Remove old numbered scripts, temp docs  
- **Clustering Experiments**: Remove failed attempt files, old CSV data
- **Documentation Updates**: Remove superseded workflow files

### Monthly Review
- Check for files > 30 days old in categories above
- Remove outdated CSV logs and temporary data
- Archive or remove old experiment documentation

## Implementation Commands

```bash
# Remove common temporary file patterns
rm -f *_files.txt cropped_to_replace.txt recover_images.py

# Remove old CSV logs (keep newest)
ls -t quality_log_*.csv | tail -n +2 | xargs rm -f

# Remove old people scores if not needed
rm -f people_scores*.csv

# Clean up old comparison files
rm -f face_grouping_comparison.md
```

## Git Integration

Always commit cleanup as a separate commit:
```bash
git rm [files to remove]
git commit -m "Repository cleanup: removed temporary files after [operation]"
```

## Memory Reference
This guide should be referenced whenever temporary files accumulate. The goal is keeping only active, current, and reference files in the repository while removing artifacts from completed operations.
