# Project Directory Breakdown

Compact map of key directories, what lives there, and how to treat them.

## Top-Level

- configs/
  - Purpose: Small JSON configs used by dashboard/pipeline (e.g., binning, metrics).
  - Notes: Read-only at runtime; safe to edit via Git.

- data/
  - Purpose: All local data artifacts (logs, snapshots, summaries, schema, projects).
  - Key subdirs:
    - aggregates/
      - archives/ — per-project 15‑minute bins (agg_15m.jsonl) for finished projects.
      - daily/ — daily aggregates across tools.
      - overall/ — overall aggregates.
    - ai_data/ — caches, embeddings, models, training snapshots.
    - crop_progress/ — JSON progress markers for cropping sessions.
    - daily_summaries/ — consolidated per‑day JSON summaries (post‑consolidation).
    - file_operations_logs/ — raw FileTracker JSONL logs (primary source for ops).
    - log_archives/ — compressed historical logs (*.log.gz).
    - projects/ — per‑project manifests (`*.project.json`) plus allowlists.
    - schema/ — JSON/SQL schema definitions for snapshots and SQLite.
    - snapshot/ — new snapshot pipelines (operation_events_v1, daily_aggregates_v1, etc.).
    - sorter_progress/ — progress JSON for sorter tools.
    - timer_data/ — legacy ActivityTimer daily JSONs (kept for reference).
    - training/ — training datasets/exports prepared for AI.
    - timesheet.csv — billing/time source for Billed vs Actual.
  - Safety: Write‑heavy but controlled. Do NOT modify production images here; treat logs/snapshots as append‑only.

- Documents/
  - Purpose: All docs, specs, ADRs, guides. Read first for workflow/decisions.
  - Highlights: TECHNICAL_KNOWLEDGE_BASE.md, DASHBOARD_GUIDE.md, FILE_SAFETY_SYSTEM.md,
    AI_* docs, DATA_* investigations, project lifecycle docs.

- schema/
  - Purpose: Root‑level schema helpers (rare); main schemas live in `data/schema/`.

- scripts/
  - Purpose: All executable code (tools, web apps, pipelines, tests).
  - Key subdirs/files:
    - 00_start_project.py / 07_finish_project.py — project lifecycle automation.
    - 01_ai_assisted_reviewer.py — fast selection UI (web).
    - 01_ai_assisted_reviewer.py — AI‑assisted reviewer UI (web).
    - 02_character_processor.py / 03_web_character_sorter.py — character workflows.
    - 02_ai_desktop_multi_crop.py — desktop crop tool (the only tool allowed to write crops).
    - ai/ — training, backfills, embedding compute, analyzers.
    - archive/ — retired desktop selector + legacy utilities/tests (kept for reference).
    - backup/ — backup helper scripts (sh + py).
    - cleanup_logs.py — daily consolidation into `data/daily_summaries/` (dry‑run by default).
    - dashboard/
      - productivity_dashboard.py / run_dashboard.py — Flask dashboard app.
      - analytics.py / data_engine.py — core data assembly and transforms.
      - project_metrics_aggregator.py — per‑project throughput, img/h, markers.
      - tests/ — dashboard unit/contract tests.
      - bins_reader.py / snapshot_loader.py — smart loaders for archives/snapshots.
    - data_pipeline/ — extract/derive/validate snapshot and bin artifacts.
    - tools/ — CLI utilities (inventory, stager, snapshot, watchdog, etc.).
    - utils/ — shared helpers (companion files, timers, etc.).
    - tests/ — full test suite (unit, integration, smoke). See `scripts/tests/run_all_tests_and_log.py` for bulk runner+CSV log.
  - Safety: Code edits go here. Respect file‑safety utilities when writing ops.

- sandbox/
  - Purpose: Temporary/testbed data and UI runs; safe to experiment.
  - Notes: Contents can be cleared; do not rely on permanence.

- crop/ and crop_cropped/
  - Purpose: Source crops and final cropped images.
  - Safety: Only `scripts/02_ai_desktop_multi_crop.py` is allowed to write/modify images here.

- selected/
  - Purpose: Images selected for keep during selection tools.
  - Safety: Treated as production images; do not overwrite in place.

- delete_staging/
  - Purpose: Staging for deletes (to macOS Trash or review) with companion files.
  - Safety: Deletes must go to Trash (send2trash or mv to ~/.Trash) — never hard delete by default.

- training data/ (space in name)
  - Purpose: Historical training datasets (mojo2, mojo2_final, etc.).
  - Notes: Treat as read‑mostly unless running a documented data job.

- mojo3/ (and similar project roots)
  - Purpose: Active project image roots (by project ID).
  - Safety: Production images; do not modify in place.

## File‑Safety Cheat Sheet (critical)

- Move/Copy/Delete only (and log via FileTracker): OK
- Create new files only in safe zones: `data/`, `sandbox/`, `data/daily_summaries/`, `data/file_operations_logs/`
- Modify image contents in place: NEVER (exception: desktop multi‑crop tool creates new cropped files)
- Handle companions together: YAML/caption move with PNG using companion utils

## Where Code Lives (quick index)

- Web apps: `scripts/dashboard/` (Flask), `scripts/01_ai_assisted_reviewer.py`, `scripts/01_ai_assisted_reviewer.py`
- Pipelines: `scripts/data_pipeline/` (snapshots, bins, validation)
- Training: `scripts/ai/` (rankers, crop proposer, backfills, embeddings)
- Utilities: `scripts/tools/` and `scripts/utils/`
- Tests: `scripts/tests/` and `scripts/dashboard/tests/`

## Data Sources (dashboard)

- Primary: `data/file_operations_logs/*.log` (+ `data/log_archives/*.gz`)
- Consolidated daily: `data/daily_summaries/*.json`
- Snapshots/bins: `data/snapshot/`, `data/aggregates/archives/`
- Project manifests: `data/projects/*.project.json`
- Timesheet: `data/timesheet.csv`

## See Also

- `Documents/FILE_SAFETY_SYSTEM.md` — strict rules for file operations
- `Documents/DASHBOARD_GUIDE.md` — architecture + API
- `Documents/TECHNICAL_KNOWLEDGE_BASE.md` — patterns and recent decisions


