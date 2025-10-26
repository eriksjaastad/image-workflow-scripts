## Code Review: claude/improve-cropping-utility-011CUVyPBdu7xPiYowp39Lvi

Reviewer: GPT-5 (Cursor assistant)
Date: 2025-10-26

### Summary
The branch introduces a queue mode in the desktop crop tool and a queue processor to execute crops with human-like timing. Directionally aligned with our goals, but must be reconciled with our safety model and data model.

This review incorporates Erik’s latest requirements:
- This is our first exception tool, so it must be beyond production-safe.
- Multi-layer safety: built-in dry run, end-to-end validator, and checks on checks.
- DB is the source of truth for crop coordinates; avoid duplicating schema/data in separate ad-hoc files where possible.
- Queue path must live in a designated safe zone.

---

### Critical Issues (must fix)

1) Pixel writes outside the desktop crop tool path
```182:189:scripts/process_crop_queue.py
with Image.open(source_path) as img:
    x1, y1, x2, y2 = crop_rect
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(source_path)  # Save in place
```
- Problem: Writes image pixels directly in a separate script. Our rule requires all image writes to go through the desktop crop tool’s trusted path (crop_and_save) with companion handling and logging.
- Update per Erik’s exception policy: Even as an exception tool, we still centralize pixel writes to a single audited code path to minimize risk.
- Recommendation: Provide a headless entrypoint in the existing crop tool (same code path as UI) and call that from the queue runner.

2) Queue and timing files outside safe zone
```110:114:scripts/02_ai_desktop_multi_crop.py
queue_file = _Path(__file__).parent.parent / "data" / "crop_queue" / "crop_queue.jsonl"
```
- Move to: data/ai_data/crop_queue/ (safe zone). Do the same for timing files/logs.

3) Missing FileTracker logs for creates/moves/crops
- Enqueue (create), move to staging (move), execution (crop), and timing logs (create) should each call FileTracker with appropriate metadata.

4) Duplicate/parallel data paths for coords
- Current queue mode records pixel rectangles only and does not persist normalized coords to our canonical training systems.
- Per Erik: DB is the source of truth. We should update the decisions DB with final_crop_coords at record-time and avoid parallel ad-hoc “truth.”
- Also append to the minimal CSV (crop_training_data.csv) if we keep that for analysis.

---

### Design Updates (per Erik’s guidance)

- DB-as-source-of-truth for coordinates
  - When queueing in the desktop tool, compute normalized coords and immediately:
    - update_decision_with_crop(db_path, group_id, [nx1,ny1,nx2,ny2])
    - log_crop_decision(project_id, filename, (nx1,ny1,nx2,ny2), width, height)
  - Queue entry can include pixel coords for convenience, but the truth remains the DB.

- Multi-layer safety and dry-run validation
  - Add a --dry-run mode to the processor that fully simulates execution end-to-end and writes a validation report instead of pixels.
  - Add a preflight validator that:
    - Verifies all source files exist, companions present, image dims match DB expectations (or precisely rescale from normalized), destinations writable in safe dirs.
    - Confirms total planned crops match queue and that moves/crops would not overwrite existing files.
    - Emits a detailed report to data/daily_summaries/ and refuses to run if violations exist (unless --force is provided, which still should be gated).

- Human timing
  - Timing patterns file should live under data/ai_data/crop_queue/.
  - Keep timing separate from scheduling; allow both free-run timing (current approach) and a pre-built schedule executor.

- Minimal surface area for exceptions
  - Even with the exception, keep all pixel writes in one place (desktop tool’s crop path).
  - Queue manager remains append-only JSONL with atomic writes; status updates via temp+replace with locking are good.

---

### Specific Suggestions

1) scripts/process_crop_queue.py
- Replace in-place PIL save with a call into the desktop crop tool’s headless crop-and-save function (same code path used by the UI).
- Add --dry-run and a preflight stage: when set, generate a validation report and exit without modifying files.
- Log all ops with FileTracker (create, crop, move).
- Read normalized coords from DB when available; use pixel rects only as convenience/fallback. Prefer normalized→pixel conversion at execution time using current image dimensions.
- Move all paths (queue, timing) to data/ai_data/crop_queue/.

2) scripts/02_ai_desktop_multi_crop.py (queue mode)
- On submit:
  - Load group_id from .decision and project_id via manifest.
  - Compute normalized coords and update decisions DB + training CSV immediately.
  - Enqueue with: created_at, project_id, group_id, filename, abs dir, width, height, crop_norm, and optional pixel rect.
  - FileTracker.log_operation("create", dest_dir="data/ai_data/crop_queue", file_count=len(batch)).
- Consider not moving originals into __crop_queued at record-time to keep directory semantics simpler, or if moving, log via FileTracker and document.

3) scripts/utils/crop_queue.py
- Change base path expectation to data/ai_data/crop_queue/.
- Ensure all writes use atomic temp+replace and locking (already done for updates; good).

4) scripts/utils/ai_crop_utils.py
- Functions look good for normalized→pixel. Keep as-is.

---

### Nice-to-haves (later)
- Add a schedule builder tool and allow the processor to execute a supplied schedule in addition to “timing pattern” mode.
- Provide a small dashboard view for progress and validation summaries.

---

### Conclusion
Great direction. With the above fixes:
- We respect the “single trusted path” for pixel writes while allowing automation.
- We avoid duplicate sources of truth by writing normalized coords to the DB at record time.
- We keep all queue/timing artifacts in safe zones and audit everything via FileTracker.
- We add multi-layer dry-run validation to meet the “beyond production safe” bar.


