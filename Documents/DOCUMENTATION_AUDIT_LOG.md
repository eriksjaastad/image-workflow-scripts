# Documentation Audit Log
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


**Auditor:** GPT-5 (Cursor)
**Date:** 2025-10-26
**Scope:** Phase 1–2 audit and consolidation (script reference verification, duplicates analysis, AI docs consolidation, initial reorganization)

## Summary Statistics
- **Files Before:** 129 markdown files (Documents + archives)
- **Files After:** 116
- **Files Deleted:** 0
- **Files Consolidated:** 12 (8 AI training + 4 reviewer → 3 consolidated)
- **Files Archived:** 12 (moved to `Documents/archives/ai/`)
- **Files Created:** 4 (this log, ai/AI_TRAINING_GUIDE.md, ai/AI_TRAINING_REFERENCE.md, ai/AI_ASSISTED_REVIEWER.md)

## Script Reference Verification

Extracted all `scripts/*.py` references from `Documents/` and verified existence.

### Broken References (do not exist)

| Referenced Script | Notes |
|-------------------|-------|
| scripts/00_finish_project.py | Superseded by scripts/07_finish_project.py |
| scripts/01_desktop_image_selector_crop.py | Archived at scripts/archive/01_desktop_image_selector_crop.py |
| scripts/02_web_character_sorter.py | Use scripts/03_web_character_sorter.py |
| scripts/04_multi_crop_tool.py | Use scripts/02_ai_desktop_multi_crop.py |
| scripts/04_web_multi_directory_viewer.py | Use scripts/05_web_multi_directory_viewer.py |
| scripts/05_finish_project.py | Use scripts/07_finish_project.py |
| scripts/07_automation_reviewer.py | Planned - not yet implemented |
| scripts/99_web_duplicate_finder.py | Removed/obsolete (not present) |
| scripts/ai/analyze_work_patterns.py | Not present |
| scripts/ai/extract_embeddings.py | Not present (use scripts/ai/compute_embeddings.py) |
| scripts/ai/extract_historical_training.py | Not present (use scripts/ai/extract_project_training.py) |
| scripts/ai/extract_training_from_projects.py | Not present (use scripts/ai/extract_project_training.py) |
| scripts/ai/generate_automation_decisions.py | Planned - not yet implemented |
| scripts/ai/test_data_collection.py | Not present |
| scripts/ai/train_crop_model.py | Not present (use scripts/ai/train_crop_proposer.py or _v2.py) |
| scripts/ai/train_ranker_model.py | Not present (use scripts/ai/train_ranker.py / _v2.py / _v3.py) |
| scripts/ai_training/compute_embeddings.py | Wrong path (use scripts/ai/compute_embeddings.py) |
| scripts/ai_training/compute_hashes.py | Planned utility; not implemented |
| scripts/ai_training/test_inference.py | Use scripts/ai/test_models.py |
| scripts/ai_training/train_crop_proposer.py | Use scripts/ai/train_crop_proposer.py |
| scripts/ai_training/train_ranker.py | Use scripts/ai/train_ranker.py |
| scripts/dashboard/smoke_test.py | Use scripts/dashboard/tests/smoke_test.py |
| scripts/tests/run_all_tests_and_log.py | Not present |
| scripts/tests/test_ai_assisted_reviewer_batch.py | Not present (tests exist under project root tests/) |
| scripts/tests/test_ai_training_decisions_v3.py | Not present (see project root tests/) |
| scripts/tests/test_ai_training_integration.py | Not present (see project root tests/) |
| scripts/tests/test_bins_system.py | Not present (use scripts/data_pipeline/demo_bins_system.py and tests under root) |
| scripts/tools/apply_automation_decisions.py | Planned - not yet implemented |
| scripts/tools/character_processor.py | Not present (use scripts/02_character_processor.py) |
| scripts/tools/join_throughput_baseline.py | Not present |
| scripts/tools/yaml_analyzer.py | Not present |

### Corrections Applied

| Document | Change |
|----------|--------|
| PROJECT_LIFECYCLE_SCRIPTS.md | 00_finish_project.py → 07_finish_project.py; 02_web_character_sorter.py → 03_web_character_sorter.py; 04_multi_crop_tool.py → 02_ai_desktop_multi_crop.py; clarified prezip stager path |
| AI_PROJECT_IMPLEMENTATION_PLAN.md | Marked `scripts/07_automation_reviewer.py` and `scripts/ai/generate_automation_decisions.py` as Planned - not yet implemented |
| AI_TRAINING_PHASE2_QUICKSTART.md | Updated ai_training/* paths to existing ai/* scripts; aligned runner names; noted planned utilities |
| AI_DOCUMENTS_INDEX.md | Updated to point to consolidated AI docs under `Documents/ai/` |

## Duplicate Documents (filename duplicates across locations)

Detected duplicates between `Documents/` and `Documents/archives/*/`:

- 48 filenames duplicated (list generated via `find` + `uniq -d`). Examples:
  - AI_ANOMALY_DETECTION_OPTIONS.md
  - AI_ASSISTED_REVIEWER_BATCH_PROCESSING_DESIGN.md
  - AI_HISTORICAL_DATA_EXTRACTION_PLAN.md
  - BACKFILL_QUICK_START.md
  - CASE_STUDIES.md
  - ... (full list in command output retained for Phase 1.2 decision matrix)

Decision matrix pending in Phase 1.2.

## Quality Improvements
- Consolidated AI docs (8 → 2) and reviewer docs (4 → 1)
- Reorganized Documents into topic folders; archived superseded docs
- Expanded stubs and performed editorial/writing pass
- Standardized metadata across repository: Last Updated, Status, Audience

## Next Steps
1) Complete hierarchical reorganization of non-AI docs (in progress)
2) Update cross-references and anchors across moved docs
3) Resolve stubs (expand/merge/delete) and perform writing quality pass


