# Feature Specifications

**Active Feature Specs** - Last Updated: October 16, 2025

This document contains specifications for built and deployed features in the Eros Mate productivity toolkit.

---

# Table of Contents

1. [Prezip Stager](#prezip-stager) - ✅ ACTIVE (`scripts/tools/prezip_stager.py`)
2. [Scan Directory State](#scan-directory-state) - ✅ ACTIVE (`scripts/tools/scan_dir_state.py`)
3. [Project Metrics Aggregator](#project-metrics-aggregator) - ✅ ACTIVE (`scripts/dashboard/project_metrics_aggregator.py`)

---

# Prezip Stager

**Script:** `scripts/tools/prezip_stager.py`  
**Status:** ✅ ACTIVE (built and tested)

## Goal

Safely produce the client zip by copying only eligible deliverables into a temporary staging directory, validating invariants, and zipping the staging tree.

## CLI

```bash
python scripts/tools/prezip_stager.py \
  --project-id mojo1 \
  --content-dir /abs/path/to/mojo1 \
  --output-zip /abs/path/to/out/mojo1_final.zip \
  [--allow-unknown] [--commit]
```

- Default is dry-run (no writes). `--commit` performs copy + zip.
- Reads: `data/projects/<project_id>_allowed_ext.json` + global bans.

## Logic

1) Load allowlist (allowed extensions + overrides) and global banned types/patterns.
2) Scan `--content-dir` recursively; for each file:
   - Lowercase extension; skip directories.
   - If hidden (dotfile) → exclude + report.
   - If extension in banned set or matches banned patterns → exclude + report.
   - If extension not in allowed ∪ overrides → exclude + report.
   - Else include (eligible).
3) Companion integrity (optional): if client initially had companions, verify same‑stem companions included/excluded consistently.
4) Build differences report (stdout + JSON alongside zip path).
5) If `--commit`: copy eligible files to staging dir (mirroring structure) and create zip from staging.

## Outputs

- Dry-run summary: totals by extension, excluded counts by reason.
- JSON report: `output-zip + .report.json` with file lists and reasons.
- On commit: `*.zip` + delete staging on success.

## Invariants

- No hidden files in staging.
- No global-banned types/patterns.
- Optional: forbid empty directories in staging.

## Failure modes

- Fail closed if any disallowed/banned types encountered (unless `--allow-unknown`).
- Nonexistent allowlist file → instruct to run inventory first.

## Integration hooks

- Project manifest: set `finishedAt` after successful zip; write counts under `metrics`.
- Dashboard: emit a compact summary (eligible totals, excluded by reason) for audit.

## Future

- Per-client policy presets; signed artifact manifest; checksum verification.

---

# Scan Directory State

**Script:** `scripts/tools/scan_dir_state.py`  
**Status:** ✅ ACTIVE (built and tested)

## Goal

Report a directory's current processing state (EMPTY / PARTIAL / FULL) with counts and recency, without mutating anything. Used for orchestration and safe finish checks.

## CLI

```bash
python scripts/tools/scan_dir_state.py \
  --path /abs/path/to/content \
  [--recent-mins 10] [--json /abs/out/state.json]
```

- Default `--recent-mins 10` marks files modified within the last N minutes as recent.
- Exits 0 always; consumers inspect JSON.

## Logic

- Recursively scan `--path`, skipping directories.
- Count:
  - `totalFiles`, `totalBytes`
  - `byExtension` map (lowercased, no dot) → { files, bytes }
- Recency:
  - `latestMtimeUtc` (ISO-8601 UTC)
  - `recentFiles` = files with mtime within `recent-mins`
- Hidden files: count separately as `hiddenFiles`.
- State rules:
  - `EMPTY`  → `totalFiles == 0`
  - `FULL`   → `totalFiles > 0` AND `recentFiles == 0`
  - `PARTIAL`→ otherwise (recent activity or mixed state)

**Notes:**
- We do not try to judge "completeness"; `FULL` only means non-empty and not actively changing.
- Thresholds are configurable via `--recent-mins`.

## JSON Output (example)

```json
{
  "path": "/abs/mojo1",
  "scannedAt": "2025-10-06T16:30:00Z",
  "totalFiles": 2450,
  "totalBytes": 1234567890,
  "byExtension": {
    "png": {"files": 2350, "bytes": 1200000000},
    "yaml": {"files": 100, "bytes": 3456789}
  },
  "hiddenFiles": 3,
  "latestMtimeUtc": "2025-10-06T16:22:11Z",
  "recentWindowMins": 10,
  "recentFiles": 0,
  "state": "FULL"
}
```

## Failure handling

- If path missing/unreadable, return a JSON with `error` and `state:"EMPTY"`, `totalFiles:0`.

## Integration

- Project manifest can record last known state for `content/` and gate finishing on `state == FULL`.
- Pre-zip stager may require `state == FULL` to proceed (configurable).

---

# Project Metrics Aggregator

**Script:** `scripts/dashboard/project_metrics_aggregator.py`  
**Status:** ✅ ACTIVE (built and used by dashboard)

## Goal

Compute project-level metrics (end-to-end images/hour, step-level rates, ahead/behind vs baseline) by reading manifests, dir-state/step events, and optional baselines.

## Inputs

- Manifest: `data/projects/<id>.project.json`
- Optional events: `data/timer_data/projects/projects_events.jsonl` (daily progress entries)
- Optional baseline: computed from history (`data/timer_data/history/daily_throughput.csv`) or a per-project baseline snapshot.

## CLI

```bash
python scripts/dashboard/project_metrics_aggregator.py \
  --project-id mojo1 \
  [--baseline-json /abs/path/baseline.json] \
  [--out-json /abs/path/metrics.json]
```

## Outputs (JSON)

```json
{
  "projectId": "mojo1",
  "durationHours": 12.5,
  "imagesTotal": 2500,
  "imagesPerHour": 200.0,
  "stepRates": {
    "select_versions": {"hours": 3.0, "images": 1200, "iph": 400.0},
    "character_sort": {"hours": 4.5, "images": 900, "iph": 200.0}
  },
  "aheadBehind": {
    "baselineIph": 180.0,
    "p25": 150.0,
    "p75": 220.0,
    "status": "ahead"  
  }
}
```

## Computation

- End-to-end: if `startedAt` and `finishedAt` present, duration = diff; else estimate using events (sum hours_today).
- Images total: use `counts.initialImages` or sum over step `imagesProcessed` where applicable.
- Step rates: for each step with both times, compute duration and iph using `imagesProcessed` if set; else omit.
- Ahead/Behind: compare `imagesPerHour` to baseline and set status relative to p25/p75.

## Safety & Integrity

- Read-only; never mutates project files.
- Validates timestamps and missing fields; degrade gracefully if incomplete.

## Dashboard integration

- The JSON can be read by a dashboard to render:
  - Top KPI card (Images/hour)
  - Step breakdown table
  - Ahead/Behind chip using p25/p75 bands

---

# Unbuilt Specs (Reference Only)

The following specs were designed but not implemented. They are retained for future reference:

## Portrait Fast-Pass (MediaPipe Pose)

**Status:** ❌ NOT BUILT  
**Original spec:** `SPEC_PORTRAIT_FASTPASS.md` (deleted)

**Goal:** Identify head-and-shoulders portraits that need no crop, route to fast lane where highest stage is auto-preferred.

**Notes:** Spec exists but no corresponding script implemented. May be revisited for future optimization.

---

## Project Manifest CLI

**Status:** ❌ NOT BUILT (alternative implementation exists)  
**Original spec:** `SPEC_PROJECT_CLI.md` (deleted)

**Goal:** Unified CLI for project lifecycle management (start, finish, step boundaries).

**Notes:** Alternative implementation exists as `00_start_project.py` and `00_finish_project.py` which handle project lifecycle differently than originally specified. Original CLI design was not implemented.

---

**Last Updated:** October 16, 2025  
**Active Specs:** 3  
**Unbuilt Specs:** 2

