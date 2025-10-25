Current Project Dashboard
=========================

Real-time dashboard for the ACTIVE project with predictive completion, stage-level tracking, and pace analysis. Built on Flask with zero-config startup.

Location
--------
- App: `scripts/dashboard/current_project_dashboard.py`
- Template: `scripts/dashboard/templates/current_project.html`
- API: `GET /api/progress`
- Port: `8082`

Quick Start
-----------
```bash
python scripts/dashboard/current_project_dashboard.py
```

- Auto-launches your browser to `http://127.0.0.1:8082/`.
- Auto-detects the active project from `data/projects/*.project.json` where `finishedAt` is `null`.
- Auto-refreshes every 30 seconds (frontend) with no backend caching.

What You’ll See
---------------
- **Header**: Project title and start date.
- **Overall Progress**: End-to-end images completed (bottleneck across stages).
- **Pace**: Current images/hour vs historical baseline; color-coded AHEAD/ON TRACK/BEHIND/PAUSED.
- **Prediction**: Hours remaining and predicted finish date (stage bottleneck aware) + original baseline estimate.
- **Milestones**: 25/50/75/100% with target dates and remaining counts.
- **Stage Breakdown**: Three bars (Selection, Crop, Sort) with per-stage rates and counts.
- **Chart**: Actual cumulative vs baseline-predicted line.
- **All-Time**: Work hours and operation totals.

Data Sources
------------
- `data/projects/*.project.json` (project manifests)
  - Active project detection: `finishedAt is null`.
  - `counts.initialImages` used as the total target.

- `data/file_operations_logs/*.log` and `data/log_archives/*.gz` (detailed logs)
  - Loaded via `DashboardDataEngine` for recent-rate calculations.
  - File operations come from the universal `FileTracker`.

- `data/daily_summaries/*.json` (consolidated daily aggregates)
  - Merged by `ProjectMetricsAggregator` for per-day actual progress.
  - If a day has a summary, detailed entries from that day are de-duplicated out.

Core Components
---------------
- `ProjectMetricsAggregator`
  - Aggregates per-project totals, per-operation counts, destination breakdowns, and daily time-series.
  - Computes overall images/hour across project window and **historical baselines** from recently completed projects.
  - Baseline: trimmed mean images/hour of the last N finished projects (light outlier trimming).

- `DashboardDataEngine`
  - Loads detailed file op records and summaries.
  - Supplies recent records (last ~2 days) for computing session rates (last 2 hours window).

Stage Model
-----------
Three stages are tracked explicitly:
- **Selection**: `move` operations with `dest_dir == 'selected'`.
- **Crop**: `crop` operations.
- **Sort**: `move` operations with `dest_dir` starting with `character_group`.

End-to-End Progress
-------------------
- The overall completed count equals the bottleneck across stages:
  - `min(selection_done, crop_done, sort_done)`
  - This reflects the number of images that have passed all three stages.

Recent Rate (Last 2 Hours)
--------------------------
- Filters recent detailed records to the active project using:
  - Preferred: `paths.root` hint from the manifest (`source_dir`, `dest_dir`, or `working_dir` contains root path).
  - Fallback: time window between `startedAt` and `finishedAt`.
- Computes:
  - **Overall images/hour** (all ops counted as files/hour)
  - **Per-stage images/hour**:
    - Selection/hr from moves to `selected`
    - Crop/hr from `crop` ops
    - Sort/hr from moves to `character_group_*`

Prediction Logic
----------------
- Stage remaining:
  - `sel_remaining = total - selection_done`
  - `crop_remaining = total - crop_done`
  - `sort_remaining = total - sort_done`
- Per-stage time-to-finish (when rate > 0): `remaining / rate`
- **Hours Remaining** = max of per-stage times (bottleneck)
- **Predicted Finish** = now + hours_remaining
- **Original Estimate** = start + `total_images / baseline_images_per_hour` (historical baseline)

Milestones
----------
- 25%, 50%, 75%, 100% of `counts.initialImages`.
- Target date for each milestone uses the baseline daily rate (baseline IPH × 24).
- Panel shows remaining images and estimated time to next milestone at current pace.

Chart Details
-------------
- Library: Chart.js (CDN).
- X-axis: day; Y-axis: images completed (cumulative).
- Series:
  - Predicted (baseline) – dashed line.
  - Actual – cumulative from daily summaries.

API Contract
------------
`GET /api/progress` returns JSON:

```json
{
  "project": {
    "projectId": "mojo3",
    "title": "Mojo3",
    "startedAt": "2025-10-21T00:00:00Z",
    "totalImages": 19406,
    "processedImages": 3247,
    "percentComplete": 16.7
  },
  "pace": {
    "baselineImagesPerHour": 445.0,
    "currentImagesPerHour": 315.0,
    "status": "BEHIND"
  },
  "prediction": {
    "hoursRemaining": 9.2,
    "predictedFinishDate": "2025-11-02",
    "originalEstimateDate": "2025-11-01"
  },
  "stages": {
    "selection": {"done": 4800, "ratePerHour": 300.0, "remaining": 14606},
    "crop": {"done": 197, "ratePerHour": 0.0, "remaining": 19209},
    "sort": {"done": 3200, "ratePerHour": 280.0, "remaining": 16206}
  },
  "milestones": [...],
  "timeseries": {"actual": [["2025-10-21", 100], ...], "predicted": [["2025-10-21", 800], ...]},
  "allTime": {
    "workHours": 24.25,
    "operationsByType": {"move": 2100, "crop": 197, "delete": 850}
  }
}
```

Operational Notes
-----------------
- Read-only: The dashboard does not modify project files or logs.
- Auto-refresh: Frontend calls `/api/progress` every 30 seconds.
- Mobile-friendly UI.
- No server-side caching; each API call recomputes metrics against current logs/snapshots.

Tuning & Heuristics
-------------------
- Sort detection: `dest_dir` starts with `character_group`. If additional sort destinations are used, extend the predicate in `current_project_dashboard.py`.
- Baseline: uses the last N finished projects with non-zero IPH; light trimming if N ≥ 5.
- Stage bottleneck: overall completion and prediction are bottleneck-aware to reflect real end-to-end progress.

Troubleshooting
---------------
- No active project: Ensure one manifest in `data/projects` has `finishedAt: null` and `status: active`.
- Baseline is 0: Not enough completed projects with valid IPH.
- Current rate is 0: Likely idle in the last 2 hours; the UI shows PAUSED.
- Chart time scale: If your browser lacks the date adapter for Chart.js, we can add explicit adapter CDN include.

Security & Safety
-----------------
- Follows repository file safety rules: only reads manifests/logs/snapshots.
- No in-place file edits or moves.

Change Log (Dashboard)
----------------------
- v1: Initial release with stage-aware progress, bottleneck-based overall completion, and predictive finish.


Phase Registry & Stable Mode
----------------------------

Purpose: Make the dashboard resilient as tools evolve by defining phases and units once, then mapping tools to phases.

Phase definitions
- review (aka selection): work measured in sections or images.
- crop: work measured in images.
- sort: work measured in images.

Script → Phase mapping (current)
- Review/Selection:
  - `ai_assisted_reviewer` (units=sections, itemsPerUnit=3 images/section)
  - `image_version_selector` (01 web image selector) (units=images)
- Crop:
  - `desktop_multi_crop` (`02_desktop_multi_crop.py`) (units=images)
  - `02_ai_desktop_multi_crop` (units=images)
- Optional pipeline:
  - `character_processor` (automated; counts toward crop or pre-sort depending on op)
- Sort:
  - `character_sorter` (web character sorter)
  - Any move to directories starting with `_` (underscore) or `character_group_*` is treated as sort.

Units & conversions
- Review can log in sections/hour; the dashboard converts to images/hour via `imagesPerSection` (default 3).
- All crop/sort metrics stay image-based.

Baselines
- Size-weighted across completed projects using `counts.initialImages` from the manifest.
- Tool-specific (review, crop, sort) baselines are computed separately; used when current pace is idle.
- A baseline snapshot can be frozen at project start (Stable Mode) to keep targets consistent.

Prediction logic (phase-first)
1) Detect current phase (review → crop → sort).
2) Use current phase rate (last 60–120 minutes). If idle, fall back to phase baseline; last resort: inventory delta/day.
3) Convert hours to dates using 6 h/day.
4) Milestones target the active phase; later phases activate as prior phases complete.

Inventory reconciliation
- Read-only directory counts provide a ground-truth view:
  - initial, remaining in root, selected, crop queue (`crop/` + `crop_auto/`), cropped done (`crop_cropped/` + `crop_auto_cropped/`), deleted.
- The dashboard compares FileTracker-derived totals with inventory; if drift > threshold, it can surface a warning.

Stable Mode (optional)
- Ignores unregistered/experimental tools for baselines and pace until they’re mapped in the registry.
- Locks baselines at project start and uses only registered phases for predictions.

Extending the system
- To add a new tool:
  1) Map `script_id` → `{ phase, units, itemsPerUnit? }` in the registry.
  2) Ensure FileTracker logs `operation` and `dest_dir` so moves to `_...` or `character_group_*` are recognized for sort.
  3) For review tools that work in sections, log `notes: "image-only"` and `files_sample` where possible; the dashboard will convert sections→images via registry.



