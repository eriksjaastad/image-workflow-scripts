# Architecture Overview

**Status:** Active
**Audience:** Developers

Last Updated: 2025-10-23

## Purpose

One-page map of the system: where code lives, how data flows, and the non-negotiable safety rules.

## Repository Map (authoritative)

- `Documents/` — Specs, guides, policies (read: `../safety/FILE_SAFETY_SYSTEM.md`, `../dashboard/DASHBOARD_GUIDE.md`).
- `scripts/` — Executable code:
  - `dashboard/` — API + data engine + HTML template.
  - `data_pipeline/` — Extract → aggregate → snapshot jobs.
  - `ai/` — Training utilities and backfills.
  - `tools/`, `utils/` — Shared helpers (e.g., companion file ops, timing helpers).
  - `tests/` — Unit/integration/Selenium smoke tests.
- `data/` — Append-only logs, snapshots, models; schemas under `data/schema/`.
- `configs/` — Small JSON configs for dashboard/pipeline.
- `sandbox/` — Safe scratch/drafts.
- Production images (read-only): `__crop/`, `__cropped/`, `__selected/`, plus `mojo*/`.

## Core Dataflows

### 1) Production Image Workflow (AI-first)

- Inputs (e.g., `mojo3/`) → Selection/Review:
  - AI-Assisted Reviewer (`scripts/01_ai_assisted_reviewer.py`): moves selected to `__selected/` or `__crop/`/`__crop_auto/`; logs decisions to SQLite v3; creates `.decision` sidecars.
- Cropping: `scripts/02_ai_desktop_multi_crop.py` writes NEW crops into `__cropped/` (only tool allowed to write images), updates SQLite v3.
- Companion integrity: always move image + `.yaml` + `.caption` together.

### 2) AI Training Decisions v3 (SQLite + sidecars)

- AI Reviewer logs to per-project SQLite at `data/training/ai_training_decisions/{project}.db`.
- Emits `.decision` sidecar in `__crop/` linking group → later crop.
- Desktop Multi-Crop reads sidecar, updates final crop in DB, removes sidecar.
- Result: joined AI recommendation + human correction for model training.

### 3) Dashboard Dataflow

- Raw ops: `data/file_operations_logs/*.log` (forensics).
- Snapshots (primary): `data/snapshot/*` daily aggregates and derived sessions.
- API: `scripts/dashboard/productivity_dashboard.py` serves `/api/data` to `dashboard_template.html`.
- Contract: See “Data Layer & API Reference” in `Documents/DASHBOARD_GUIDE.md`.

## Safety Invariants (must hold)

- Move, don’t modify; production images are read-only.
- Only `02_ai_desktop_multi_crop.py` (and legacy `02_ai_desktop_multi_crop.py`) write cropped images (creates NEW files).
- Deletions go to macOS Trash (recoverable). Prefer `send2trash`.
- Always move companions together; never orphan metadata.
- All file ops are logged by FileTracker under `data/file_operations_logs/`.

## Schemas and Storage (pointers)

- AI decisions schema: `data/schema/ai_training_decisions_v3.sql`.
- Snapshot formats: `data/snapshot/*` (daily aggregates, derived sessions).
- Consolidated reference (planned): `Documents/SCHEMA_REFERENCE.md`.

## Operability Hooks

- Safety audit: `scripts/tools/audit_file_safety.py`.
- Pipeline: `scripts/data_pipeline/*.py` (extract → build aggregates).
- Dashboard tests: `scripts/tests/test_dashboard*.py`; Selenium smoke for web tools.

## Quick Links

- FILE SAFETY: `Documents/FILE_SAFETY_SYSTEM.md`
- DASHBOARD GUIDE: `Documents/DASHBOARD_GUIDE.md`
- TECH KB: `Documents/TECHNICAL_KNOWLEDGE_BASE.md`
