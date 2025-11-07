# Tools Directory

Command-line utilities for maintenance, analysis, and workflow support.

## 📋 Planning & Analysis

- **`todo_agent_planner.py`** - Analyze TODO lists and estimate token costs per task
- **`analyze_content_directory.py`** - Analyze content directory structure and file counts
- **`analyze_crop_timing.py`** - Analyze crop timing patterns and performance
- **`analyze_human_patterns.py`** - Analyze human work patterns from logs
- **`calculate_project_timeline.py`** - Calculate project timelines and milestones

## 🔍 Auditing & Validation

- **`audit_crop_queue.py`** - Audit crop queue for consistency
- **`audit_file_safety.py`** - Safety audit of file operations
- **`audit_files_vs_db.py`** - Compare filesystem state against database records
- **`audit_orphan_decisions.py`** - Find orphaned AI decisions without source files
- **`backup_health_check.py`** - Verify backup integrity
- **`daily_validation_report.py`** - Generate daily validation summary

## 📊 Reporting

- **`report_file_ops_by_project.py`** - Report file operations grouped by project
- **`report_image_counts.py`** - Report image counts by directory
- **`generate_archive_cleanup_report.py`** - Identify archivable/deletable files
- **`scan_dir_state.py`** - Scan and report directory state
- **`system_diagnostic.py`** - Full system diagnostic report

## 🔧 Data Operations

- **`backfill_filetracker_from_sqlite.py`** - Backfill file tracker from SQLite database
- **`backfill_v3_from_archives.py`** - Backfill v3 data from archive sources
- **`enqueue_test_batch.py`** - Enqueue test batch for processing
- **`flatten_image_directories.py`** - Flatten nested image directory structures
- **`prezip_stager.py`** - Stage files for zip archival
- **`subset_builder.py`** - Build image subsets for testing

## 🗂️ File Management

- **`check_directory_counts.py`** - Check and report directory file counts
- **`inventory_allowed_ext.py`** - Inventory files by allowed extensions
- **`list_character_dirs.py`** - List character directories and metadata
- **`update_archived_file_references.py`** - Update references to archived files

## 🖼️ Image Processing

- **`face_grouper.py`** - Group images by detected faces
- **`similar_image_grouper.py`** - Group similar images using perceptual hashing
- **`generate_cropped_sidecars.py`** - Generate sidecar metadata for cropped images

## 🚨 Monitoring & Operations

- **`watchdog_runner.py`** - Run watchdog process monitor
- **`smoke_test_processor.py`** - Quick smoke test for processor pipeline
- **`handoff_check.py`** - Check project readiness for handoff

## 🔬 Analysis & Profiling

- **`prof.py`** - Performance profiler for scripts
- **`reducer.py`** - Data reducer for analysis
- **`snapshot.py`** - Create system state snapshots

## 🗄️ Archive Management

- **`zip_group_scanner.py`** - Scan ZIP archives and index contents
- **`zip_layout_inspector.py`** - Inspect ZIP archive internal structure

## 🛟 Recovery & Restoration

- **`recover_all_code.py`** - Recover code from backups
- **`recover_data_from_backup.py`** - Restore data from backup snapshots
- **`recover_from_cursor_history.py`** - Recover files from Cursor editor history

## 📝 Scaffolding

- **`scaffold_review_doc.py`** - Generate review document templates

---

## Usage Notes

All tools are designed to be run directly:

```bash
python scripts/tools/tool_name.py --help
```

Most tools support `--dry-run` or `--verbose` flags for safe testing.

For production use, always review the tool's help output first to understand required parameters and effects.
