# Documentation

**Last Updated:** 2025-10-26
**Audience:** Developers, Operators

## Daily Shortcuts
- Current TODOs: `core/CURRENT_TODO_LIST.md`
- Knowledge Base: `reference/TECHNICAL_KNOWLEDGE_BASE.md`

## Introduction
This knowledge base documents the end-to-end image workflow automation system: tools, data, AI training, dashboards, and safety policies.

> Policy: Whenever any document is edited, update its **Last Updated**, **Status**, and **Audience** metadata at the top.

## Quick Start
- New developer? Read `core/ARCHITECTURE_OVERVIEW.md` (5 min) and `core/PROJECT_LIFECYCLE_SCRIPTS.md` (5 min).
- Start a new project: `core/PROJECT_LIFECYCLE_SCRIPTS.md`.
- Train the AI: `ai/AI_TRAINING_GUIDE.md` (15–20 min).
- Understand the data schema: `data/SCHEMA_REFERENCE.md` (10–15 min).
- Run the dashboard: `dashboard/DASHBOARD_GUIDE.md` (5–10 min).

## Core Documentation
- `core/CURRENT_TODO_LIST.md`
- `core/ARCHITECTURE_OVERVIEW.md`
- `core/OPERATIONS_GUIDE.md`
- `core/DISASTER_RECOVERY_GUIDE.md`
- `core/PROJECT_LIFECYCLE_SCRIPTS.md`

## By Topic
### AI
- `ai/AI_TRAINING_GUIDE.md`
- `ai/AI_TRAINING_REFERENCE.md`
- `ai/AI_ASSISTED_REVIEWER.md`
- `ai/AI_DOCUMENTS_INDEX.md`

### Dashboard
- `dashboard/DASHBOARD_GUIDE.md`
- `dashboard/DASHBOARD_API.md`

### Data
- `data/SCHEMA_REFERENCE.md`
- `data/PROJECT_MANIFEST_GUIDE.md`
- `data/CROP_TRAINING_SCHEMA_V2.md` (archived reference)
- `data/GIT_DATA_TRACKING_POLICY.md` (archived reference)

### Guides
- `guides/AUTO_GROUPING_GUIDE.md`
- `guides/COMPANION_FILE_SYSTEM_GUIDE.md`
- `guides/INLINE_VALIDATION_GUIDE.md`
- `guides/BACKFILL_QUICK_START.md`
- `guides/WORK_TIME_CALCULATION_GUIDE.md` (archived)
- `guides/15_MINUTE_BINS_GUIDE.md` (archived)

### Safety
- `safety/FILE_SAFETY_SYSTEM.md`
- `safety/FILE_SAFETY_CHECKLIST.md` (planned)
- `safety/PROJECT_DELIVERABLES_POLICY.md` (archived)
- `safety/REPOSITORY_CLEANUP_GUIDE.md`

### Reference
- `reference/TECHNICAL_KNOWLEDGE_BASE.md`
- `reference/FEATURE_SPECIFICATIONS.md` (archived)
- `reference/WEB_STYLE_GUIDE.md`
- `reference/CASE_STUDIES.md`

### Testing
- `testing/TESTS_GUIDE.md`

## Archives
- `archives/ai/` — Superseded AI docs consolidated into the new AI section
- `archives/sessions/` — Session summaries and investigations
- `archives/implementations/` — Completed implementation summaries
- `archives/misc/` — Other historical docs

### Archive Policy
- Sessions (`archives/sessions/`): Keep 12 months; auto-delete older if not referenced by any non-archives doc.
- AI (`archives/ai/`): Keep 6 months after consolidation; delete if fully captured in `ai/AI_TRAINING_GUIDE.md`, `ai/AI_TRAINING_REFERENCE.md`, or `ai/AI_ASSISTED_REVIEWER.md` and not referenced elsewhere.
- Dashboard (`archives/dashboard/`): Keep 6 months; delete if not referenced.
- Implementations (`archives/implementations/`): Keep indefinitely; optionally compress >12 months.
- Misc (`archives/misc/`): Keep 3 months; delete if not referenced.

Reference rule:
- “Referenced” means linked from any document outside `archives/`. If referenced, keep.

Quarterly cleanup (review-first, no auto-deletes):
- Generate a candidate list of unreferenced docs older than their thresholds.
- Script: `scripts/tools/generate_archive_cleanup_report.py` → writes to `data/daily_summaries/`
- Review the list, then remove or move to Trash manually.

## Contribution Guidelines
- Keep docs concise, accurate, and audience-aware.
- Include metadata at the top (Last Updated, Status, Audience, Estimated Reading Time when relevant).
- Cross-link related docs; use section anchors where helpful.
- Archive superseded docs instead of deleting.

## Index
For a fuller list, see `ai/AI_DOCUMENTS_INDEX.md` and browse topical folders above.
