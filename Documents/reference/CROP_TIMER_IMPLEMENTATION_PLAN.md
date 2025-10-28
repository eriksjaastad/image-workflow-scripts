## AI Desktop Crop Timer and Record-Only Workflow — Detailed Plan

Status: Draft (planning only)
Owner: Erik + AI Assistant
Last Updated: 2025-10-26

### Objectives
- Replace “instant crop” in the desktop multi-crop tool with a record-only mode that writes crop coordinates to durable storage without modifying images.
- Build a scheduler and timer that replays historical cropping behavior (bursty batches, micro/macro breaks, time-of-day shape) to crop from the recorded queue over multi-day schedules.
- Provide a front-end quality operator experience (clean CLI/UX, optional web dashboard with stats/graphs).
- Maintain strict file safety rules (only the crop tool writes image pixels; all other scripts write new files only in safe zones).

### Non-Negotiable Constraints (File Safety)
- NEVER modify production images in place outside the designated crop tool execution path.
- All recorded data must be written as NEW files under safe zones:
  - data/ai_data/
  - data/file_operations_logs/
  - data/daily_summaries/
  - sandbox/
- All operations must be audited via FileTracker.

### High-Level Flow
1) Desktop Multi-Crop (record-only mode)
   - Operator selects rectangles as usual; upon Submit, we persist coordinates only.
   - Persist to: SQLite v3 decisions DB (final_crop_coords), training CSV (minimal schema), and a queue file for later execution.
   - No image pixels are written; no file moves occur.

2) Schedule Builder
   - Read historical crop events from snapshot data and/or FileTracker logs.
   - Generate a 3-day schedule that mimics historical cadence and break patterns.
   - Persist schedule to JSON per day and emit a human-readable preview report.

3) Timer Runner
   - Load schedule + queue; at each batch window, pop N items and execute real crops using the existing crop tool’s safe path.
   - Respect micro/macro/lunch breaks and stop windows; log every operation.

4) Operator UX
   - Clean CLI with clear summaries; optional mini web dashboard (charts for today’s plan, progress, next breaks, throughput).

---

## Data Models and Files

### Training CSV (Minimal Schema — already implemented)
Path: data/training/crop_training_data.csv
Schema:
timestamp,project_id,filename,crop_x1,crop_y1,crop_x2,crop_y2,width,height
Notes:
- Coordinates are normalized [0,1]. We also store width/height for validation and rescaling.
- Normalization rationale: resilient to file moves and resizes; convert back to pixel space when executing.

### Decisions DB (SQLite v3 — already implemented)
Table: ai_decisions
Write: update_decision_with_crop(db_path, group_id, final_crop_coords)
Notes:
- final_crop_coords are normalized [0,1]; crop_timestamp stored.
- crop_match is calculated against ai_crop_coords where present.

### Crop Queue (NEW)
Path: data/ai_data/crop_queue/pending.jsonl (append-only)
One JSON per line:
{
  "created_at": "2025-10-26T12:03:55Z",
  "project_id": "mojo3",
  "group_id": "mojo3_group_20251021_234530",
  "filename": "20250818_053224_stage1_generated.png",
  "dir": "/absolute/path/to/crop/subdir",
  "width": 3072,
  "height": 3072,
  "crop_norm": [0.12, 0.08, 0.82, 0.73]
}
Processed archive: data/ai_data/crop_queue/processed.jsonl

### Daily Schedule (NEW)
Path: data/ai_data/crop_schedule/YYYYMMDD.json
Shape:
{
  "date": "2025-10-27",
  "timezone": "America/Los_Angeles",
  "blocks": [
    { "type": "work", "start": "09:12:00", "end": "09:34:00", "batch_size": 3, "batches": 7 },
    { "type": "break", "start": "09:34:00", "end": "09:41:00", "reason": "micro" },
    { "type": "work", "start": "09:41:00", "end": "10:07:00", "batch_size": 3, "batches": 9 },
    { "type": "break", "start": "12:05:00", "end": "12:42:00", "reason": "lunch" }
  ]
}

### Human-Readable Schedule Preview (NEW)
Path: data/daily_summaries/crop_schedule_preview_YYYY-MM-DD.md
Contents: List blocks, totals, ETA, warnings (e.g., queue shorter than plan).

### Configuration (NEW)
Path: configs/crop_timer_config.json
Keys:
- batch_size_default: 3
- micro_break_min_s: 180
- micro_break_max_s: 420
- burst_min_s: 720
- burst_max_s: 1500
- coffee_every_min: 60
- coffee_break_min_s: 600
- coffee_break_max_s: 1200
- lunch_window_local: ["12:00", "13:30"]
- lunch_break_min_s: 1800
- lunch_break_max_s: 3600
- end_of_day_window_local: ["17:00", "20:00"]
- timezone: "America/Los_Angeles"

---

## Normalization Rationale and Guarantees
- We store normalized coordinates in DB/CSV for resilience.
- Execution converts normalized → pixel using the image’s actual width/height at crop time.
- We record width/height at record time for validation; if image dimensions differ at execution, we rescale proportionally and flag a warning in logs.
- This matches how AI preloads normalized crops into the desktop tool (converted to pixels for display/work). It preserves fidelity because the final pixel coordinates are derived from normalized ground truth with the correct dimensions.

Example (conceptual only):
```python
def normalize(x1, y1, x2, y2, w, h):
    return (x1 / w, y1 / h, x2 / w, y2 / h)

def denormalize(nx1, ny1, nx2, ny2, w, h):
    return (round(nx1 * w), round(ny1 * h), round(nx2 * w), round(ny2 * h))
```

---

## CLI Interfaces (No implementation yet)

### 1) Desktop Multi-Crop — Record-Only Mode
Script: scripts/02_ai_desktop_multi_crop.py
New flag: --record-only
Behavior:
- On Submit, for each image with a crop:
  - Resolve project_id (get_current_project_id), group_id (from .decision), width/height.
  - Write normalized coords to DB (update_decision_with_crop) and training CSV (log_crop_decision).
  - Append JSON to data/ai_data/crop_queue/pending.jsonl
  - Log FileTracker operation: create → crop_queue
- No crop/save/move operations.

### 2) Build Schedule (NEW)
Script: scripts/tools/build_crop_schedule.py
Usage:
- python scripts/tools/build_crop_schedule.py --days 3 --start 2025-10-27T09:00:00 --tz America/Los_Angeles
Inputs:
- Historical events: data/snapshot/operation_events_v1/, fallback to data/file_operations_logs/
- Timer sessions (optional): data/timer_data/
Outputs:
- data/ai_data/crop_schedule/YYYYMMDD.json per day
- data/daily_summaries/crop_schedule_preview_YYYY-MM-DD.md
Notes:
- Emulate time-of-day rate shapes using non-homogeneous Poisson sampling or empirical inter-event time sampling per hour bucket.
- Insert micro/macro/lunch breaks based on config and historical idle patterns.

### 3) Timer Runner (NEW)
Script: scripts/tools/crop_timer_runner.py
Usage:
- python scripts/tools/crop_timer_runner.py --days 3 --confirm
Behavior:
- Load today’s schedule; at each work block, perform ‘batches’ of N items (batch_size) popping from pending.jsonl.
- For each item, call into the existing crop tool execution path to write pixels safely, then move to __cropped and archive queue entry to processed.jsonl.
- Logs:
  - FileTracker: operation=crop for each image, batch summaries
  - Console summaries: progress, ETA, next break

### 4) Inspector and Visualizer (Optional NEW)
- scripts/tools/inspect_crop_queue.py: Show queue health, duplicates, dimension mismatches.
- scripts/tools/visualize_crop_schedule.py: Render schedule timeline and expected throughput (ASCII or small web view).
- scripts/dashboard/crop_timer_ui.py: Minimal Flask web UI with charts (today’s plan, progress, upcoming breaks).

---

## Detailed To-Do List (Implementation Plan)

1) Record-Only Mode in Desktop Tool
- Add --record-only flag and pass to tool instance.
- On batch submit:
  - For each selected crop:
    - Read image size; compute normalized coords
    - Append queue JSON (safe zone), write training CSV, update decisions DB
    - FileTracker.log_operation("create", dest_dir="data/ai_data/crop_queue", file_count=K)
- Edge cases: Missing .decision (group_id unknown) → still log CSV/queue with group_id=None; warn.
- Unit tests: verify queue append, CSV row, DB update path is called; no image write.

2) Crop Queue Writer
- Ensure directory exists; append-only JSONL with robust error handling.
- Include project_id, group_id, filename, absolute dir, width, height, crop_norm, created_at.
- Validate: 0≤x1<x2≤1, 0≤y1<y2≤1; width/height>0.

3) Historical Cadence Extractor
- Load data/snapshot/operation_events_v1/ to extract historical crop timestamps.
- Compute per-hour rate profile by day-of-week.
- Extract session/idle windows from data/timer_data/ (if present) to calibrate break policy.
- Persist derived stats to data/dashboard_cache/pattern_analysis.json (for UI).

4) Schedule Builder
- Inputs: desired days, start time, timezone, queue length.
- Build day plan:
  - Sample bursts and inter-batch intervals from empirical distribution within hour bucket.
  - Insert micro/coffee/lunch breaks per config and historical idle.
  - Ensure total planned crops ≤ queue length with safety margin.
- Write schedule JSON and preview MD report.
- Validation: lint schedule (no overlaps, within day, monotonic times).

5) Timer Runner
- Load schedule and queue; at runtime:
  - Sleep/await to each block; for work blocks, execute batches of N:
    - Pop entry, validate file presence and dimensions
    - Denormalize coords to pixels; call safe crop executor
    - On success: move queue item → processed.jsonl; log crop
    - On failure: requeue with backoff or write to sandbox/failures.jsonl
- Respect breaks exactly; provide pause/resume commands.
- Summaries per block and end-of-day totals.

6) Safe Crop Executor Integration
- Provide a headless entrypoint that reuses the existing crop-and-move logic from the desktop tool (the ONLY image writer).
- Arguments: path, pixels x1 y1 x2 y2; ensure companion moves and FileTracker crop logging are identical to UI path.
- If headless path is not feasible, spawn the tool in a controlled non-interactive mode for single-image crop (still leveraging same codepath).

7) Operator UX and Telemetry
- CLI: rich text summaries, progress bar, ETA, upcoming break display.
- Optional: small Flask UI (charts). Reads pattern_analysis.json and live state.
- Metrics: per-hour throughput vs plan, variance, overruns, queue burn-down.

8) Safety, Auditing, and Recovery
- All file ops logged via FileTracker.
- Queue is append-only; processed archive maintained; retries tracked.
- Dry-run modes for schedule building and execution.
- Inspector to validate queue items before execution.

9) Documentation
- Update Documents/ with user guide and troubleshooting.
- Dashboard quick link to schedule preview and live status.

---

## File and Module Name Proposals

Scripts (NEW):
- scripts/tools/build_crop_schedule.py
- scripts/tools/crop_timer_runner.py
- scripts/tools/inspect_crop_queue.py
- scripts/tools/visualize_crop_schedule.py (optional)
- scripts/dashboard/crop_timer_ui.py (optional web UI)

Data/Config:
- data/ai_data/crop_queue/pending.jsonl
- data/ai_data/crop_queue/processed.jsonl
- data/ai_data/crop_schedule/YYYYMMDD.json
- data/daily_summaries/crop_schedule_preview_YYYY-MM-DD.md
- data/dashboard_cache/pattern_analysis.json
- configs/crop_timer_config.json

Docs:
- Documents/AI_DESKTOP_CROP_TIMER_PLAN.md (this file)
- Documents/guides/AI_CROP_TIMER_USER_GUIDE.md (TBD)

---

## Reasoning Behind Key Decisions

- Normalized Coordinates: Store once, use anywhere; resilient to file moves and minor resizes. Paired with width/height capture for validation/fidelity.
- Queue JSONL: Append-only, human-inspectable, easy to stream/process, safe-zone write.
- Separate Schedule vs Runner: Lets us pre-plan multi-day workloads, review/adjust, and then execute; clean separation of concerns.
- Historical Mimicry: Using empirical distributions of inter-event times per hour-of-day and day-of-week yields realistic bursts and pauses.
- Break Policy via Config: Easy to tune micro/macro/lunch/stop windows without code changes.
- Headless Executor Reuses Crop Tool: Guarantees we don’t violate file-safety rules, and keeps companion file handling identical.

---

## Validation and Testing Plan

- Unit tests:
  - Queue writer validation; schedule linter; denormalize math; DB update payloads.
- Integration tests:
  - Dry-run schedule build (no writes outside safe zones).
  - Timer runner dry-run (no pixel writes) to validate pacing and logging.
  - Small end-to-end in sandbox on dummy images (write allowed for crop tool only).
- Monitoring:
  - FileTracker logs; dashboard counters; processed.jsonl counts.

---

## Example Snippets (Illustrative Only)

Normalized write (conceptual):
```python
from datetime import datetime

def record_only_entry(image_path, crop_px, project_id, group_id):
    w, h = read_dimensions(image_path)
    x1, y1, x2, y2 = crop_px
    nx1, ny1, nx2, ny2 = x1/w, y1/h, x2/w, y2/h

    # CSV (minimal schema)
    log_crop_decision(project_id, image_path.name, (nx1, ny1, nx2, ny2), w, h)

    # DB
    update_decision_with_crop(db_path, group_id, [nx1, ny1, nx2, ny2])

    # Queue
    append_jsonl("data/ai_data/crop_queue/pending.jsonl", {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "project_id": project_id,
        "group_id": group_id,
        "filename": image_path.name,
        "dir": str(image_path.parent.resolve()),
        "width": w,
        "height": h,
        "crop_norm": [nx1, ny1, nx2, ny2],
    })
```

Execution (conceptual):
```python
def execute_crop(queue_item):
    w, h = queue_item["width"], queue_item["height"]
    nx1, ny1, nx2, ny2 = queue_item["crop_norm"]
    x1, y1, x2, y2 = round(nx1*w), round(ny1*h), round(nx2*w), round(ny2*h)
    safe_crop_and_move(image_path, (x1, y1, x2, y2))  # delegates to crop tool
```

---

## Milestones

M1: Record-only mode (queue + CSV + DB writes; no pixel writes)
M2: Historical cadence extractor and pattern cache
M3: Schedule builder and preview report
M4: Timer runner with headless crop executor
M5: Optional UI and visualizers
M6: Docs and operator training

---

## Risks and Mitigations
- Dimension Drift: If source image dimensions at execution differ from recorded width/height, we rescale from normalized and warn in logs; add inspector to preflight queue.
- Queue Starvation: Schedule demands > queue size. Builder caps planned items to available queue with margin; runner verifies before each block.
- Clock/Timezone: Normalize to project timezone; store UTC timestamps; schedule JSON uses local wall time with tz key.
- Safety Violations: All pixel writes must go through crop tool; enforce via code structure and review.

---

## Next Steps
1) Wire --record-only flag and the record pipeline in desktop tool (no image writes).
2) Implement queue append helper in safe zone with FileTracker logging.
3) Build schedule generator using snapshots + timer data; emit JSON + preview.
4) Implement timer runner with safe crop executor.
5) Add optional UI/visualizers.


