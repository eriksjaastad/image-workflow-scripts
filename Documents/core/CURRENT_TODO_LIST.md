# Current TODO List

**Status:** Active
**Audience:** Developers
**Policy:** This is the single authoritative TODO list. Do not create separate TODO docs; add sections here instead.

---

## 🎯 Active Tasks

### Data Integrity Backfill (High)

- [ ] **Backfill missing crop data caused by dimension-logging bug** [PRIORITY: HIGH]

  - **Scope:** Rows/images impacted when Desktop Multi-Crop logged dimensions as (0,0)
  - **Action:** Identify affected items → recompute from source images/sidecars → validate → write to SQLite v3 → snapshot
  - **Output:** Verified updates in `data/training/ai_training_decisions/*.db` and daily snapshot

### TODO Hygiene

- [ ] **Review and prune this TODO list** [PRIORITY: HIGH]
  - **Action:** Archive stale items to `Documents/archives/`, consolidate duplicates, re-order by priority

### Dashboard: Work Time Accuracy & Timer Integration (High)

- [x] **PHASE 1 (Oct 30, 2025): Timer Integration** [PRIORITY: HIGH] ⭐ **IN PROGRESS**

  - **Goal:** Use Focus Timer (`000_focus_timer.py`) as ground truth for work time, correlate with file ops for tool attribution
  - **Current Problem:** File-op-based timing undercounts due to gaps, batches, thinking time
  - **Focus Timer Data:** `~/focus_sessions.csv` format: `timestamp,minutes`
  - **Implementation (Option 3: Timer + Validation):**
    1. **Timer Data Loader** (`data_engine.py`)
       - Add `load_focus_timer_sessions()` method
       - Parse `~/focus_sessions.csv` → list of sessions with start_time + duration
       - Return: `[{start: datetime, end: datetime, duration_minutes: float}, ...]`
    2. **Timer-File Op Correlation** (`analytics.py`)
       - For each timer session, find all file ops during that time window
       - Count PNG files moved per tool during session
       - Distribute timer duration proportionally by file ops
       - Example: 2-hour session, 80 crops + 20 selector moves → 96 min crop, 24 min selector
    3. **Validation Metrics**
       - Show both timer hours AND file-op hours side-by-side
       - Flag anomalies: timer says 8h but only 10 file ops (forgot to stop?)
       - Dashboard display: "Timer: 8.5h | File Ops: 6.2h | Using: Timer"
    4. **Fallback Logic**
       - If timer data exists → Use timer (trust user's explicit intent)
       - If no timer data → Fall back to current file-op calculation
       - Per-day granularity: can mix timer days with non-timer days
  - **Files to Modify:**
    - `scripts/dashboard/data_engine.py` - Add timer loader
    - `scripts/dashboard/analytics.py` - Add correlation logic in `_build_tools_breakdown_for_project()`
    - `scripts/dashboard/productivity_dashboard.py` - Display both metrics
  - **Testing:**
    - Create test CSV with known sessions
    - Verify correlation math (proportional distribution)
    - Check fallback when CSV missing
  - **Status:** Implementing on `claude/dashboard-review-011CUY1Sx4vrcJ5aZTzjScLV`

- [ ] **PHASE 2 (Future): Pattern Recognition for Historical Data** [PRIORITY: MEDIUM]

  - **Goal:** Backfill accurate work time for days BEFORE focus timer existed using pattern detection
  - **Why:** We have tons of file operation logs from historical projects, can extract better time estimates

  - **Feature 2A: Scheduled Batch Session Detection** (AI Assisted Reviewer)
    - **Pattern:** Top-of-hour starts, 1-hour intervals, consistent tool usage
    - **Detection Algorithm:**
      ```
      1. Extract all session starts from file ops (first op after 15+ min gap)
      2. Check if ≥3 starts cluster near hour boundaries (:00 ±5 min)
      3. Calculate intervals between starts
      4. If avg_interval is 55-65 minutes → BATCH MODE DETECTED
      5. Credit full 1-hour intervals for each session
      ```
    - **Example Pattern:**
      ```
      9:03 AM  → Batch 1 starts (file ops begin)
      9:42 AM  → Last file op (earned 18 min break)
      10:01 AM → Batch 2 starts ON TIME
      10:35 AM → Last file op (earned 25 min break)
      11:00 AM → Batch 3 starts ON TIME
      Result: Credit 9:00-10:00 (1h), 10:00-11:00 (1h), 11:00-12:00 (1h) = 3 hours
      ```
    - **Validation:**
      - Must have file ops during credited interval (not just timer)
      - Tool must be consistent (same tool across sessions)
      - Min 3 sessions to establish pattern
    - **Dashboard Display:**
      ```
      ┌─────────────────────────────────────┐
      │ AI Assisted Reviewer                │
      ├─────────────────────────────────────┤
      │ Batch Processing Mode Detected ✓    │
      │ 7 scheduled sessions @ 1h each      │
      │ Credited: 7.0h | File Ops: 4.8h     │
      │ Efficiency Bonus: +2.2h             │
      │ Pattern: 9:00, 10:00, 11:00...      │
      └─────────────────────────────────────┘
      ```
    - **Why This Works:**
      - Rewards disciplined scheduling (starting on time)
      - Credits efficiency (fast completion = earned break)
      - Honest (requires actual file ops during interval)
      - Matches user's actual work commitment (1-hour blocks)

  - **Feature 2B: Burst Work Session Detection** (Multi-Crop Tool)
    - **Pattern:** Short intense bursts (20-40 min) separated by micro-breaks (3-10 min)
    - **Detection Algorithm:**
      ```
      1. Find all file op timestamps for multi-crop
      2. Group into bursts using gap detection:
         - Gap < 10 minutes → Same session (micro-break, part of work)
         - Gap 10-15 minutes → Ambiguous (check context)
         - Gap > 15 minutes → Separate session (real break)
      3. For each session, calculate:
         - Start: first op timestamp
         - End: last op timestamp
         - Images: PNG count
         - Bursts: number of sub-bursts within session
      4. Credit entire session duration (includes micro-breaks)
      ```
    - **Example Pattern:**
      ```
      9:15 AM → Crop burst 1: 100 images in 27 min
      9:42 AM → 3 min break (hands tired, shake out)
      9:45 AM → Crop burst 2: 50 images in 20 min
      10:05 AM → 5 min break (water, stretch)
      10:10 AM → Crop burst 3: 150 images in 32 min
      10:42 AM → 18 min break (real break, YouTube)

      Session 1: 9:15-10:42 = 87 minutes (includes micro-breaks)
      Total images: 300
      Speed progression: 3.7 → 2.5 → 4.7 img/min
      ```
    - **Micro-Break Threshold:** < 10 minutes = part of session (like rest between sets at gym)
    - **Why This Works:**
      - Physical work requires hand rest
      - Micro-breaks are PART of the work (fatigue management)
      - Like weightlifting: rest between sets is part of workout
      - Matches actual user workflow
    - **Speed Tracking (Bonus):**
      - Track images/minute per burst
      - Detect warmup (slow start, faster later)
      - Detect fatigue (slowing down after 200+ crops)
      - Suggest breaks when speed drops >30%

  - **Feature 2C: Speed & Efficiency Metrics**
    - **Per-Tool Speed Tracking:**
      ```
      Multi-Crop:
      - Images per minute (overall)
      - Images per minute per burst
      - Speed progression over session
      - Fatigue detection (speed drop)

      AI Assisted Reviewer:
      - Batches per hour
      - Batch completion time trend
      - Efficiency improvement (58 min → 42 min)

      Web Image Selector:
      - Selections per minute
      - Decision speed
      ```
    - **Efficiency Trends:**
      ```
      Dashboard shows:
      "Your batch time improved from 58 min to 42 min! (-27%)"
      "You're doing 4.5 crops/min today vs 3.2 yesterday (+40%)"
      "Speed dropped 30% after crop #200 - consider break?"
      ```
    - **Implementation:**
      - Calculate per-session metrics from file ops
      - Store in time-series for trending
      - Display in dashboard insights panel
    - **Goal:** Celebrate improvements, detect fatigue, optimize workflow

  - **Priority Logic (When Multiple Patterns Exist):**
    1. **Timer data exists** → Use timer (highest trust)
    2. **Batch mode detected** → Credit full intervals
    3. **Burst work detected** → Include micro-breaks
    4. **Fallback** → Current file-op calculation

  - **Files to Modify:**
    - `scripts/dashboard/analytics.py` - Add pattern detection functions
    - `scripts/dashboard/data_engine.py` - Add session grouping utilities

  - **Testing:**
    - Use historical file ops from mojo2, mojo3
    - Manually verify detected patterns match known work sessions
    - Compare: pattern-detected hours vs current file-op hours
    - Validate against billed hours (should be closer)

- [ ] **PHASE 3 (Future): Timer Overlay for Production Tools** [PRIORITY: LOW]

  - **Goal:** Ensure focus timer is always running during work by requiring timer start before tools launch
  - **User's Idea:** Loading screen overlay that forces timer start before tool is usable
  - **Implementation:**
    - **Shared Timer Manager Module:**
      - Create `scripts/utils/timer_manager.py`
      - Check if timer is running (look for recent focus_sessions.csv entry)
      - Show modal overlay if timer not running
      - "Start Timer to Continue" button
      - Launches `000_focus_timer.py` subprocess
      - Polls until timer starts, then dismisses overlay
    - **Integration Points:**
      - `01_web_image_selector.py` - Add timer check on launch
      - `03_web_character_sorter.py` - Add timer check on launch
      - `04_multi_crop_tool.py` - Add timer check on launch
      - `01_ai_assisted_reviewer.py` - Add timer check on launch
    - **UI Design:**
      ```
      ┌─────────────────────────────────────┐
      │                                     │
      │         ⏰ Start Work Timer?        │
      │                                     │
      │  Timer is not running. Start it to  │
      │  track your work time accurately.   │
      │                                     │
      │   [Start Timer & Continue]  [Skip]  │
      │                                     │
      └─────────────────────────────────────┘
      ```
    - **Configuration:**
      - Add `require_timer: bool` to config
      - Optional: can disable overlay if you really want
    - **Benefits:**
      - Never forget to start timer
      - Ensures accurate work time tracking
      - Gentle reminder without being annoying
    - **Concerns:**
      - Adds friction to workflow (extra click)
      - Timer subprocess management complexity
      - What if timer crashes?
    - **Alternatives:**
      - Auto-start timer in background (no overlay)
      - Post-hoc detection: warn at end of day if file ops but no timer
      - Dashboard reminder: "No timer data for today, remember to use it!"

- [ ] **Make "Actual vs Billed" hours reliable with batch-aware timing** [PRIORITY: HIGH]

  - **Problem:** AI-Assisted Reviewer processes large batches (e.g., 700) with sparse log timestamps, so hour counting via activity gaps undercounts. Multi-crop (3-up) logs more steadily and counts better.
  - **Current Implementation:** 15-minute bins counted when either (a) ≥7.5 min active with ≤5 min gaps, or (b) ≥30 files processed in-bin.
  - **Debug Tool:** `python scripts/dashboard/tools/debug_project_hours.py mojo3`
    - Recent sample (minutes, files) illustrates current behavior:
      - 2025-10-23: 195.0 min (6323 files)
      - 2025-10-22: 150.0 min (2525 files)
      - 2025-10-25: 105.0 min (26379 files)
      - 2025-10-24: 90.0 min (10963 files)
      - 2025-10-27: 45.0 min (539 files)
  - **Actions:**
    1. Tune thresholds (gap and in-bin file-count) for batch sessions; validate with debug script.
    2. COMPLETED: Focus timer integration (Phase 1 above)
    3. Document rules in dashboard README and expose thresholds via config.

### Documentation Cleanup

- [ ] **Remove "Last Updated" dates from all documents** [PRIORITY: LOW]
  - **Issue:** Redundant with git history (already tracks file changes automatically)
  - **Action:** Remove "Last Updated:" fields from all markdown docs in `Documents/`
  - **Reason:** Manual maintenance overhead for info that git provides for free
  - **Benefit:** Less cruft, one less thing to remember to update

### TOP PRIORITY: Artifact Groups (cross-group/misaligned images)

✅ Moved to Recently Completed (Oct 26, 2025)

### Phase 3: Two-Action Crop Flow (Reviewer)

- [ ] Add analytics view for “perfect crop” (final ≈ AI crop within 5%) in SQLite v3
- [ ] Optional migration: extend user_action enum to include approve_ai_suggestion and approve_ai_crop (Phase 3B)

### Phase 3: AI-Assisted Reviewer Testing

- [ ] **Write tests for AI Assisted Reviewer** [PRIORITY: MEDIUM]
  - **Issue:** Hotkey routing logic is complex and needs automated tests
  - **Coverage Needed:**
    - 1234 keys: Accept with AI crop → `__crop_auto/`, without AI crop → `__selected/`
    - ASDF keys: Always remove AI crop → `__selected/`
    - QWER keys: Manual crop → `__crop/`
  - **Reference:** Check `scripts/tests/` for existing test patterns
  - **Benefit:** Prevent regression when making changes to reviewer logic

<!-- Removed obsolete quick-start block; tool is already in active use -->

#### Model Integration (Optional - Already Have Great Models)

- [ ] Integrate Ranker v3 into AI-Assisted Reviewer
  - **Ranker v3 stats:** 94.4% anomaly accuracy, 98.1% overall
  - **Replace:** Rule-based logic with model predictions
  - **Add:** Confidence scores from model
- [ ] Integrate Crop Proposer v1 (if it completed training)
  - **Check:** Does `crop_proposer_v1.pt` exist?
  - **Add:** Crop suggestions to reviewer UI

### AI Reviewer: Batch Summary Delete Count (Bug)

✅ Moved to Recently Completed (Oct 26, 2025)

### Test Follow-ups (Dashboard + Utils)

- [ ] Migrate tests off legacy desktop selector shim and remove file
  - File: `scripts/01_desktop_image_selector_crop.py` (now a compat shim)
  - Action: Update tests to import archived path `scripts/archive/01_desktop_image_selector_crop.py` or remove usages
  - Then: Delete the shim file once tests no longer reference it
- [ ] Fix prompt extraction tests (scripts/tests/test_prompt_extraction.py) — 3 failures
- [ ] Satisfy file safety audit (scripts/tests/test_file_safety_audit.py)
- [ ] Crop overlay rounding to integers (scripts/tests/test_ai_assisted_reviewer_batch.py)
- [ ] Full pytest rerun and address any remaining stragglers

---

## 📅 Backlog

### Productivity Dashboard (run_dashboard.py)

- [ ] Build vs Actual hours not working
  - Verify data sources and aggregation; ensure FileTracker-derived bins or equivalent are wired
- [ ] Project Productivity Table not populating
  - Likely missing multi-tool tracking for web image selector (AI version vs others)
  - Add source(s) to capture web selector actions and include in table
- [ ] Multi-Crop tool stats suspect (shows 327 cropped)
  - Audit metric definition and source; reconcile with crop logs and final counts
- [ ] Input vs Output panel empty for Mojo3
  - Confirm inputs (incoming images) and outputs (final images) queries; fix project filters
- [ ] Files Processed by Project looks wrong (only shows yesterday)
  - Check time window, grouping, and event source; validate daily aggregation for Mojo3
- [ ] Queue system charts empty (expected: unused)
  - Hide or collapse queue panels when no queue data present

### Dashboard Phase 3 & 4 Improvements

Context

- Phase 3 (Sorting) and Phase 4 (Final Review) need proper tracking in the dashboard. Current dashboard detects Phase 3 but doesn’t track progress. Must support Erik’s workflow: `character_processor` auto-grouping, manual Finder drags into single-underscore category bins (`_ethnicity/`, `_hair_color/`, `_body_type/`), and iterative refinement before final review.

Phase 3: Sorting Phase Tracking

- **Goal**: Track progress as images are sorted from `__selected/` subdirectories into final category directories.
- **Key metric**: Recursive PNG count in `__selected/` going DOWN (work remaining)
- **Tasks**
  - [ ] Capture initial baseline when Phase 3 starts
    - When Phase 3 detected (no files in `__crop/`, `__crop_auto/`, but files in `__selected/`)
    - Store `phase3_initial_count = recursive count of __selected/**/*.png`
    - Save once per project (manifest or state file)
  - [ ] Show Phase 3 progress in dashboard UI
    - Remaining: current recursive count in `__selected/`
    - Initial: stored baseline
    - Completed: `initial - remaining`
    - Progress: `(completed / initial) * 100%`
    - Display: Remaining, Sorted, percentage, progress bar
  - [ ] Optional: Category breakdown (informational only)
    - List single-underscore directories (`_ethnicity/`, `_hair_color/`, `_body_type/`, etc.) and counts
    - Exclude `__character_group_*` (temporary workspace)

Phase 4: Final Review Phase

- **Goal**: Detect and track when sorted files are moved back to `content/` for final review before delivery.
- **Workflow**: Finish Phase 3 (`__selected/` empty/nearly), drag `_*/` bins back into `content/`, dashboard switches to Phase 4; use web sorter to review/crop/fix.
- **Tasks**
  - [ ] Phase 4 detection logic
    - Triggers when: recursive `content/` PNGs > 50 AND `__selected/` empty or <10 AND Phase 3 previously active
    - Phase label: “Phase 4: Final Review”
  - [ ] Phase 4 progress/tracking (first version)
    - Initial: capture recursive PNG count in `content/` when Phase 4 starts
    - Current: current recursive count in `content/`
    - Status: “In Review”
    - Design exploration (later):
      - Option A: Show only total count in `content/`
      - Option B: Track movement OUT of `content/` to final delivery
      - Option C: Manual “mark as complete”
    - Display: Phase 4 header with `Files in content/: N`, Status
  - [ ] Design question (later): What signals Phase 4 complete? (empty `content/` vs manual flag vs delivered path)

Dynamic Dashboard Header

- **Goal**: Show relevant info for the current phase; hide/collapse irrelevant sections.
- **Tasks**
  - [ ] Phase-specific header content
    - Phase 1 (Selection): content/ remaining, selection stats
    - Phase 2 (Cropping): crop progress, batch counts, rate stats (existing behavior)
    - Phase 3 (Sorting): `__selected/` remaining, sorting progress, category breakdown
    - Phase 4 (Final Review): `content/` count, review status, delivery readiness
  - [ ] Hide/collapse irrelevant metrics in Phase 3/4 (keep historical stats lower on page)
  - [ ] Phase indicator clarity: large current phase label, color highlight, clear progression (Selection → Cropping → Sorting → Final Review)

Technical Notes

- **Files to modify**: `scripts/dashboard/current_project_dashboard_v2.py`
  - `get_directory_status()` (recursive counts already fixed)
  - Phase detection logic (around lines ~1266-1274) and progress calculation
  - HTML/template rendering (around lines ~1287+)
- **State management**
  - Store phase baselines in project manifest (e.g., `data/projects/<project>.project.json`) under `"phaseProgress": {}` (preferred)
  - Alternative: separate `*.state.json` if needed
- **Testing**
  - Simulate by dragging directories between `__selected/`, `_*/`, and `content/`
  - Refresh dashboard to verify phase switches and progress calculations

Open Questions / Design Decisions

- Phase 3 baseline capture: auto on first detection vs manual “start phase” switch?
- Phase 4 completion signal: empty `content/` vs manual button vs delivery path observation?
- Category directory tracking: show counts only vs compute “sorted by category” stats?
- Progression charts: add Phase 3/4 sorting/review rate over time vs keep a simple progress bar?

Priority

- **Must Have**: Phase 3 progress tracking (`__selected/` going down); dynamic dashboard header
- **Should Have**: Phase 4 detection and basic tracking
- **Nice to Have**: Category breakdown; Phase 4 workflow detail
- **Future**: Phase-specific progression charts

Timeline

- Add to TODO now; implement after Mojo3 delivery. Use directory dragging to test phase transitions.

### Historical Crop Data Extraction (Experiment)

- [ ] **Extract crop coordinates from historical projects using image matching** [PRIORITY: MEDIUM]
  - **Goal:** Recover thousands of crop training examples by comparing original vs cropped images
  - **Method:**
    1. Use project manifests to identify date ranges
    2. Find cropped images (files within project date range in `_cropped/` or `_final/` directories)
    3. Find matching original images (same filename in original/raw directory)
    4. Use OpenCV template matching to find exact crop location
    5. Extract crop coordinates (x1, y1, x2, y2)
    6. Normalize coordinates (0.0-1.0 range)
    7. Write to training CSV or SQLite (decide which)
  - **Why This Will Work:**
    - No compression/resizing in workflow = exact pixel match
    - Template matching will find location with 99.9%+ confidence
    - Fast processing (seconds per image)
  - **Implementation:**
    - Proof of concept: Test on 10 image pairs first
    - Visual verification: Show matches overlaid
    - Batch processing: Process all historical projects
    - Validation: Manual review of sample matches
  - **Potential Value:**
    - Could recover 5,000-10,000 crop training examples
    - Dramatically improve Crop Proposer model
    - Learn from historical crop patterns
  - **Output Format:**
    - Training data (not decision tracking)
    - Either: Add to crop training CSV
    - Or: Store in separate "recovered_crops.csv"
    - Include: timestamp, project_id, filename, crop_coords, image_width, image_height
  - **Projects to Process:**
    - Mojo1, Mojo2 (and any other finished projects with original + cropped pairs)
  - **Status:** EXPERIMENT - Build proof of concept first, then scale if successful

### Documentation

- [ ] Document training data structure rules in `AI_TRAINING_DATA_STRUCTURE.md`
- [ ] Create troubleshooting guide for common training issues

### Dashboard Improvements

- [ ] **Reimagine/Simplify Productivity Dashboard** [PRIORITY: MEDIUM]

  - **Issue:** Too many graphs that aren't actually useful
  - **Keep:**
    - Build vs Actual (helpful, locked in)
    - Billing Efficiency Tracker (fine)
    - Productivity Table (pretty good)
    - Input vs Output (favorite graph)
  - **Reconsider/Remove:**
    - Project Comparison (confusing - operations > total images?)
    - Files Processed by Project (lines too small, hard to read with many projects)
    - Other graphs that don't provide clear insights
  - **Improvements Needed:**
    - Make it easier to view just last 2-3 projects together (not all at once)
    - Better default filters (e.g., only show recent projects by default)
    - Larger/clearer visualizations (especially for time-series)
    - Consider minimum thresholds for graphs (500+ files?) to avoid flatlined data
  - **Goal:** Less overwhelming, more focused on actionable insights

- [ ] Composition Metrics (2-up vs 3-up) [PRIORITY: HIGH]

  - **Goal:** Establish baseline composition per project (group size distribution) and kept rates (approve/crop) from the decisions DB; inform predictions for future projects.
  - **Analyzer Script:** Create `scripts/tools/analyze_composition.py` to compute per-project:
    - groups_by_size (e.g., {2: N, 3: N})
    - by_action counts (approve/crop/reject)
    - kept rates per group size and overall
  - **Historical Baseline:** Extend snapshot pipeline to persist composition metrics across all projects (daily snapshot or per-project summary).
  - **API:** Expose metrics via dashboard API for current and historical projects.
  - **UI:** Add dashboard cards/charts for composition and kept rates; compare Mojo1 vs Mojo3.
  - **Data Source:** SQLite v3 (`data/training/ai_training_decisions/*.db`) — DB is single source of truth.

- [ ] **Add AI Performance Stats to Dashboard** [PRIORITY: HIGH]
  - **Data Source:** SQLite v3 databases (`data/training/ai_training_decisions/*.db`)
  - **Metrics to Show:**
    - **Selection Accuracy:** % of times AI picked the same image as user
    - **Crop Accuracy:** % of times AI's crop was within 5% of user's final crop
    - **Trend Over Time:** Is AI getting better as it trains on more data?
    - **Per-Project Stats:** Compare AI performance across Mojo1, Mojo2, Mojo3, etc.
    - **Confidence Calibration:** Does high AI confidence = correct prediction?
  - **Visualizations:**
    - Line graph: Selection accuracy over time (by project)
    - Bar chart: AI correct vs user override (per project)
    - Gauge: Current AI accuracy (like a speedometer)
    - Table: Detailed breakdown (total decisions, correct, wrong, accuracy %)
  - **Why This Is Awesome:**
    - See AI improvement in real-time as you work!
    - Know when to trust AI's suggestions more
    - Celebrate milestones (50% accuracy → 70% → 85%!)
    - Identify which types of images AI struggles with
  - **Implementation:**
    - Query SQLite databases for `selection_match` and `crop_match` flags
    - Group by project and timestamp
    - Calculate rolling accuracy (e.g., last 100 decisions)
    - Display prominently on dashboard (maybe top section?)

### Automation

- [ ] Set up daily validation report (cron job or manual)
- [ ] Add email/Slack alerts when validation fails
- [ ] **Create git helper bash scripts** [PRIORITY: MEDIUM]
  - **Goal:** Make git operations quick and foolproof to avoid wasting time on simple tasks
  - **Scripts to Create:**
    - `git-status-quick.sh` - Show current branch, what's changed, if behind/ahead
    - `git-sync.sh` - Pull latest from origin, show what changed
    - `git-cleanup-branches.sh` - List merged branches, offer to delete them
    - `git-quick-checkout.sh` - Fast branch switching with auto-pull
  - **Requirements:**
    - Simple, clear output (no git jargon)
    - Safe by default (ask before destructive operations)
    - Work with existing quickpr function
  - **Location:** `scripts/tools/git/`

### Web Sorter AI Feedback (Low Volume)

- [ ] Capture delete actions in web character sorter as training feedback [PRIORITY: LOW]
  - Scope: When user deletes an image during Phase 3/4 review in `03_web_character_sorter.py`, log a lightweight training signal
  - Implementation options:
    1. Call `log_selection_only_entry(session_id,set_id,chosen_path,negative_paths)` with `chosen_path=''` and `negatives=[deleted_path]` to record a negative-only example
    2. Add a minimal `review_feedback` CSV (filename, reason=deleted_bad_crop, timestamp)
  - Constraints: No extra file writes in production image dirs; logs go to `data/training/` or decisions DB v3
  - Value: Likely low counts but useful for future “bad-crop” classifier or data hygiene analytics

### File Operations & Logging

- [ ] Investigate retro-logging Finder moves and background tracking [PRIORITY: LOW]
  - Goal: If files were moved via Finder (outside FileTracker), optionally backfill a lightweight log entry so dashboards remain accurate.
  - Explore:
    1. Simple retro-log by recent mtime window (prototype exists; evaluate usefulness and noise)
    2. Optional background watcher (FSEvents) that records minimal “move” metrics without altering files
  - Constraints: No content modifications; respect file safety rules; logs only in `data/file_operations_logs/`
  - Exit criteria: Decide keep/kill based on signal quality and overhead

### Backups & Delivery Automation

- [ ] Weekly full rollup to cloud (tar+upload with manifest) [PRIORITY: HIGH]
  - Source: `~/project-data-archives/image-workflow/`
  - Compress: one tar.zst per week + manifest (counts, sizes, sha256)
  - Upload: `gbackup:weekly-rollups/`
  - Retention: keep 12 weeks locally + cloud
- [ ] Auto-upload finished ZIP from 07_finish_project to Drive [PRIORITY: MEDIUM]
  - Hook: after successful finish with `--commit`
  - Target: `gbackup:deliveries/<projectId>/`
  - Flow: copy → check → (optional) delete local zip
  - Reuse rclone remote `gbackup` and daily cron log

---

## ✅ Recently Completed

**AI Desktop Multi-Crop UX (Oct 28, 2025)**

- [x] Remove performance timer and visual focus timer from AI Desktop Multi-Crop
- [x] Update progress title to show Batch X/Y with directory context

**AI-Assisted Reviewer Adoption (Oct 28, 2025)**

- [x] Validated on ~1,000 images across recent projects; `.decision` sidecars created and used downstream

**Artifact Groups & Two-Action Crop Flow (Oct 26, 2025)**

- [x] Artifact detection + warning flow in reviewer
- [x] Audit tool artifact candidate scaffolding
- [x] Snapshot extraction artifact field marking
- [x] Dashboard artifact panel (UI + backend)
- [x] aiCropAccepted two-action routing
- [x] Sidecar schema with `ai_route` field
- [x] AI Reviewer batch summary delete count bug fix
- [x] JSONL batch summary logger

**SQLite v3 Training System (Oct 22, 2025 - Night/Morning)**

- [x] **Design and implement SQLite-based training data system** ⭐ **GAME CHANGER!**
- [x] Create database schema with decision tracking + crop matching
- [x] Build utility functions (`scripts/utils/ai_training_decisions_v3.py`)
- [x] Write comprehensive tests (18 tests, all passing)
- [x] Integrate into AI Reviewer (log AI decisions + create `.decision` files)
- [x] Integrate into Desktop Multi-Crop (read `.decision` files + update with final crops)
- [x] Document complete system (`Documents/AI_TRAINING_DECISIONS_V3_IMPLEMENTATION.md`)
- [x] Add to Technical Knowledge Base (365 lines)
- [x] Fix Desktop Multi-Crop performance lag (plt.draw → draw_idle, 20-40x faster!)
- [x] Test full workflow (90 decisions logged successfully)

**AI Reviewer UX Improvements (Oct 22, 2025 - Morning)**

- [x] **Add "Remove Crop" toggle button to AI-selected images** ⭐ **TESTED AND WORKING!**
- [x] Add regular crop button to AI-selected images (for manual cropping)
- [x] Auto-launch browser when starting AI Reviewer
- [x] Document --batch-size and other flags in header
- [x] Remove confusing "Approve" buttons from all images

**Phase 2: AI Training (90% Complete)**

- [x] Extract training data from 15 historical projects (21,250 selections, 12,679 crops)
- [x] Compute embeddings for all training images (77,304 total)
- [x] Train Ranker v2 with project boundary validation
- [x] **Train Ranker v3 with anomaly oversampling** ⭐ **94.4% anomaly accuracy!**
- [x] Analyze anomaly cases to identify model training gaps (518 cases)
- [x] Fix Desktop Multi-Crop dimension logging bug
- [x] Re-compute missing mojo2 embeddings (17,834 new embeddings)
- [x] Create validation script (`scripts/ai/validate_training_data.py`)
- [x] Document lessons learned (`Documents/archives/ai/AI_DATA_COLLECTION_LESSONS_LEARNED.md`)

**Data Integrity (Just Completed - Oct 21, 2025 Morning)**

- [x] **Integrate inline validation into all data collection tools** ⭐ **DONE!**
- [x] Add dimension validation to `log_select_crop_entry()`
- [x] Add path validation to `log_selection_only_entry()`
- [x] Create test suite (`scripts/tests/test_inline_validation.py`)
- [x] Documentation (`Documents/archives/misc/INLINE_VALIDATION_GUIDE.md`)

**Crop Training Data Schema Evolution (Oct 21, 2025 Afternoon)**

- [x] **Design and implement NEW minimal crop training schema** ⭐ **8 columns instead of 19!**
- [x] Create `log_crop_decision()` function with strict validation
- [x] Update AI-Assisted Reviewer to use new schema
- [x] Document schema evolution and benefits (`Documents/archives/misc/CROP_TRAINING_SCHEMA_V2.md`)
- [x] Add to Technical Knowledge Base
- [ ] **BACKLOG: Migrate 7,194 legacy rows to new schema** (Optional - keep both for now)

---

## 🗑️ Cancelled

- [x] ~~Extract crop data from jmlimages-random~~ (No crop data exists - only selections)
- [x] ~~Extract crop data from tattersail-0918~~ (No crop data exists - only selections)
- [x] ~~Extract crop data from historical projects~~ (Crop coordinates never logged before data collection system)

---

## 📝 Notes

- **Data Collection Crisis:** Discovered 3 weeks after the fact that Desktop Multi-Crop was logging dimensions as (0,0) instead of actual values. Lost ~6,700 potential training examples. Need real-time validation to prevent this in the future.
- **Embeddings Issue:** Some embeddings exist in cache but files are missing from disk. Need verification step after embedding generation.
- **Historical Data Limitation:** Only projects processed AFTER data collection system was built have usable crop data.

### AI Automation (Imported 2025-10-26)

- Backlog triage needed; items below are placeholders—convert to scoped tasks when prioritized:
  - Add configurable safe-zone allowlist (read from `configs/` and used in validation)
  - Add retry with backoff for per-crop failures and partial progress resume
  - Queue manager maintenance CLI: `clear_completed` + timing/log rotation helpers
  - Pre-commit installer script for root-file policy hook in `scripts/tools/`
  - Makefile shortcuts: `make timing`, `make queue-test`, `make process-fast`
  - CLI to delete/restore a batch to `__delete_staging` using companion utils
  - Processor: enforce decisions DB linkage in preflight (fail with clear error when missing)
  - Docs: Queue quickstart + analyzer usage (place in `Documents/guides/`)
  - Docs: Commit communication standard snippet in `Documents/README.md`
  - Dashboard: queue stats panel (pending/processing/completed/failed)
  - Dashboard: processing time trends and batches-per-session charts
  - Tool: audit of queue vs filesystem and DB (orphan/consistency report)
