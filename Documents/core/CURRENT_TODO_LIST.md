# Current TODO List

**Status:** Active
**Audience:** Developers
**Policy:** This is the single authoritative TODO list. Do not create separate TODO docs; add sections here instead.

---

## 🎯 Active Tasks

### 🦖 Raptor Reliability Audit ✅ **COMPLETE!**

**Purpose:** Systematic reliability review of core workflow scripts to catch silent failures, improve error handling, and ensure data integrity.

**Process:** Run 3-phase Raptor review on each script individually (iterative approach).

**Status:** **ALL 8 PHASES COMPLETE!** 🎉

**Results:**

- **47 critical silent failure modes eliminated**
- **8 comprehensive review documents** created in `reviews/`
- **Zero linting errors** in all fixed files
- **Full audit trail** for every change

**Completed Reviews:**

- [x] **Phase 1: `scripts/00_start_project.py`** ✅

  - 5 fixes applied (Block Merge → Fixed)
  - Review: `reviews/raptor_review_20251031T203130Z.md`

- [x] **Phase 2: `scripts/file_tracker.py`** ✅

  - 5 fixes applied (Block Merge → Fixed)
  - Review: `reviews/raptor_review_20251031T215516Z.md`

- [x] **Phase 3: `scripts/01_ai_assisted_reviewer.py`** ✅

  - 4 fixes applied (Small fixes → Fixed)
  - Review: `reviews/raptor_review_20251031T225323Z.md`

- [x] **Phase 4: `scripts/02_ai_desktop_multi_crop.py`** ✅

  - 5 fixes applied (Needs Rework → Fixed)
  - Review: `reviews/raptor_review_20251031T233219Z.md`

- [x] **Phase 5: `scripts/02_character_processor.py`** ✅

  - 8 fixes applied (Block Merge → Fixed)
  - Review: `reviews/raptor_review_20251031T234831Z.md`

- [x] **Phase 6: `scripts/03_web_character_sorter.py`** ✅

  - 6 fixes applied (Block Merge → Fixed)
  - Review: `reviews/raptor_review_20251101T002627Z.md`

- [x] **Phase 7: `scripts/04_character_check.py`** ✅

  - 5 fixes applied (Needs Rework → Fixed)
  - Review: `reviews/raptor_review_20251101T010158Z.md`

- [x] **Phase 8: `scripts/07_finish_project.py`** ✅
  - 9 fixes applied (Needs Rework → Fixed)
  - Review: `reviews/raptor_review_20251101T012014Z.md`

**Optional Future Work:**

- [ ] **Phase 9: Backup Scripts** (DEFERRED - lower priority)
  - `scripts/backup/daily_backup.py`
  - `scripts/backup/daily_backup_simple.py`
  - Can be reviewed in future session if needed

**Future Automation:**

- [ ] **Automate Raptor review workflow using API endpoints**
  - **Goal:** Replace manual copy/paste workflow with automated API calls
  - **Resources:** Erik has ChatGPT and Claude API endpoints available
  - **Workflow to automate:**
    1. Phase 1: Call Claude Sonnet 4.5 API with Phase 1 prompt + target file
    2. Parse Phase 1 output, extract findings section
    3. Phase 2: Call GPT-5 Codex API with Phase 2 prompt + Phase 1 results
    4. Parse Phase 2 output, extract validation summary
    5. Phase 3: Call GPT-5 API with Phase 3 prompt + all prior results
    6. Generate complete review document automatically
    7. Optionally: Auto-apply diffs if confidence score meets threshold
  - **Benefits:**
    - Remove manual copy/paste (slowest part of workflow)
    - Enable batch processing of multiple scripts overnight
    - Consistent prompt execution (no human error in pasting)
    - Generate structured JSON output for dashboard integration
  - **Priority:** LOW (current manual workflow is working well)
  - **Estimated Effort:** 2-3 hours for API integration script

- [ ] **Once automation is set up: Deep Raptor review of utilities and tools** [PRIORITY: HIGH after automation]
  
  - **Goal:** Review ALL utilities in the critical path, recursively checking dependencies
  
  - **Phase 1: Map Critical Dependencies**
    1. Scan scripts 0-4 and 7 (core workflow, skip 5/6) for utility imports
    2. For each utility found, scan IT for utility imports (recursive)
    3. Build dependency tree showing entire call chain
    4. Generate ordered review list (deepest dependencies first)
  
  - **Phase 2: Automated Recursive Raptor Review**
    - Start with leaf utilities (no dependencies)
    - Work up the dependency tree
    - For each utility:
      - Run 3-phase Raptor review
      - Apply fixes
      - Commit
      - Move to next utility
    - Benefits of automation: Can run overnight, process 20-30 utilities unattended
  
  - **Target Scripts to Scan for Dependencies:**
    - ✅ `scripts/00_start_project.py`
    - ✅ `scripts/01_ai_assisted_reviewer.py`
    - ✅ `scripts/02_ai_desktop_multi_crop.py`
    - ✅ `scripts/02_character_processor.py`
    - ✅ `scripts/04_character_check.py`
    - ✅ `scripts/07_finish_project.py`
    - ❌ Skip: `scripts/03_web_character_sorter.py` (less critical)
    - ❌ Skip: `scripts/02_character_processor.py` variant (if exists)
  
  - **Expected Critical Utilities (to be confirmed by scan):**
    - `scripts/utils/activity_timer.py` (used by character_processor)
    - `scripts/utils/companion_file_utils.py` (likely used by many)
    - `scripts/utils/crop_queue.py` (used by crop tools)
    - `scripts/utils/ai_crop_utils.py` (used by AI tools)
    - `scripts/tools/prezip_stager.py` (used by finish_project) **CRITICAL**
    - `scripts/tools/scan_dir_state.py` (used by prezip_stager) **CRITICAL**
    - `scripts/data_pipeline/archive_project_bins.py` (used by finish_project)
    - Plus any utilities THOSE utilities depend on (recursive)
  
  - **Deferred (low priority - audit/reporting tools):**
    - `scripts/tools/audit_*.py` - Auditing tools (not in critical path)
    - `scripts/tools/analyze_*.py` - Analysis tools (not in critical path)
    - `scripts/tools/report_*.py` - Reporting tools (not in critical path)
    - `scripts/tools/backup_*.py` - Backup utilities (lower priority, can review separately)
    - `scripts/data_pipeline/*` - Most data pipeline scripts (not in core workflow)
  
  - **Estimated Scope:**
    - Core workflow scripts: 6 scripts (already done)
    - Critical utilities (estimated): 10-15 scripts
    - Total Raptor reviews: 10-15 more reviews
    - With automation: Can run overnight/over weekend
    - Manual time saved: ~8-12 hours vs doing it manually
  
  - **Why This Approach:**
    - We've hardened the top-level scripts (they log and surface utility errors)
    - But if utilities have silent failures, those still cause production issues
    - Recursive scan ensures we catch the ENTIRE chain (utility → sub-utility → sub-sub-utility)
    - Automation makes reviewing 10-15 utilities feasible (would be tedious manually)
    - Result: End-to-end reliability from workflow script down to deepest dependency

---

### AI Predictions Backfill (HIGH PRIORITY)

- [ ] **Add zip file support to AI prediction script** [PRIORITY: HIGH]

  - **Goal:** Modify `backfill_project_phase1a_ai_predictions.py` to accept zip files as input (in addition to directories)
  - **Why:** Original images for mojo1/mojo2 may be in zip archives on external drives
  - **Current Status:** Script only accepts `--original-dir` argument pointing to unzipped directory
  - **Required Changes:**
    1. Add `--zip-file` argument (mutually exclusive with `--original-dir`)
    2. Modify `group_original_images()` function to:
       - Accept `source_path` and `is_zip` boolean
       - Extract zip to temporary directory if `is_zip=True`
       - Process images from temp directory
       - Clean up temp directory after processing
    3. Add imports: `zipfile`, `tempfile`, `shutil`
  - **Implementation Code:**

    ```python
    # In group_original_images():
    if is_zip:
        # Extract zip to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="ai_predict_zip_"))
        print(f"Extracting zip to temporary directory: {temp_dir}")

        try:
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                # Only extract image files
                image_extensions = {".png", ".jpg", ".jpeg"}
                for file_info in zip_ref.filelist:
                    if Path(file_info.filename).suffix.lower() in image_extensions:
                        zip_ref.extract(file_info, temp_dir)

            working_dir = temp_dir
            cleanup_after = True
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise Exception(f"Failed to extract zip: {e}")
    else:
        working_dir = source_path
        cleanup_after = False

    # ... process images from working_dir ...

    # Finally, cleanup:
    if cleanup_after and temp_dir.exists():
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    ```

  - **Argument Parser Changes:**

    ```python
    parser.add_argument("--original-dir", required=False,
                       help="Directory containing original images")
    parser.add_argument("--zip-file", required=False,
                       help="Zip file containing original images")

    # Validation
    if not args.original_dir and not args.zip_file:
        parser.error("Must specify either --original-dir or --zip-file")
    if args.original_dir and args.zip_file:
        parser.error("Cannot specify both --original-dir and --zip-file")

    is_zip = bool(args.zip_file)
    source_path = Path(args.zip_file) if is_zip else Path(args.original_dir)
    ```

  - **Safety:** Read-only operation on zip files (never modify zip contents)
  - **Files to Modify:** `scripts/ai/backfill_project_phase1a_ai_predictions.py`

- [ ] **Backfill AI predictions for ALL historical projects (for dataset completeness)** [PRIORITY: HIGH]

  - **Current State (as of Oct 31, 2025):**
    - ✅ mojo3.db: 6,468 rows → **6,468 with AI predictions (100%)**
    - ❌ mojo1.db: 8,268 rows → **0 with AI predictions**
    - ❌ mojo2.db: 5,985 rows → **0 with AI predictions**
    - ❌ Aiko.db: 350 rows → **0 with AI predictions**
    - ❌ Eleni.db: 1,920 rows → **0 with AI predictions**
    - ❌ Kiara_Slender.db: 1,845 rows → **0 with AI predictions**
    - ❌ agent-1001/1002/1003.db: 2,058 rows → **0 with AI predictions**
    - ❌ jmlimages-random.db: 4,486 rows → **0 with AI predictions**
    - ❌ Others: 1011, 1012, 1013, 1100, 1101_Hailey, tattersail-0918
    - **Total Missing: ~35,000+ rows need AI predictions across all projects**
  - **⚠️ IMPORTANT - Training Data Consideration:**
    - **Current AI models were trained on:** mojo1 (5,244 selections) + mojo2 (4,594 selections) + others
    - **This means:** Backfilling predictions on mojo1/mojo2 = testing on training data (not true accuracy test)
    - **Purpose:** Dataset completeness, debugging, baseline comparison ONLY
    - **For TRUE accuracy:** Use separate test data (see "Train/Test Split Validation" task below)
  - **Why This Is Still Useful:**
    - Complete database structure across all projects
    - Debugging: Check if model memorized vs learned patterns
    - Baseline: Compare training accuracy vs test accuracy (shows overfitting)
    - Future retraining: Want ALL user data available for next model version
  - **Process:**

    1. Locate original images for mojo1 and mojo2 (may be in zips on T7Shield)
    2. Run Phase 1A script to generate AI predictions:

       ```bash
       python scripts/ai/backfill_project_phase1a_ai_predictions.py \
           --project-id mojo1 \
           --original-dir /Volumes/T7Shield/Eros/original/mojo1 \
           --output-db data/training/ai_training_decisions/mojo1_backfill_temp.db

       python scripts/ai/backfill_project_phase1a_ai_predictions.py \
           --project-id mojo2 \
           --original-dir /Volumes/T7Shield/Eros/original/mojo2 \
           --output-db data/training/ai_training_decisions/mojo2_backfill_temp.db
       ```

       (Or use `--zip-file` once zip support is added)

    3. Run Phase 1B to merge AI predictions with existing user ground truth
    4. Validate: Check selection_match and crop_match columns populated

  - **Expected Output:**
    - mojo1.db: 8,268 rows with BOTH user data AND AI predictions
    - mojo2.db: 5,985 rows with BOTH user data AND AI predictions
    - Can then analyze: "On mojo1, AI would have matched user 45% of the time"
  - **Dependencies:**
    - Requires zip file support (above task) if originals are in zip archives
    - Requires access to original image directories/zips
  - **Files to Run:**
    - `scripts/ai/backfill_project_phase1a_ai_predictions.py`
    - `scripts/ai/backfill_project_phase1b_user_data.py`

- [ ] **Implement proper train/test split validation for AI models** [PRIORITY: MEDIUM]

  - **Problem:** Current AI models were trained on mojo1+mojo2 (9,838 selections total)
  - **Current Situation:**
    - Training data: mojo1 (5,244), mojo2 (4,594), plus other projects
    - Test data: mojo3 has 6,468 rows but only 11 selections in training CSV (likely added later)
    - Result: mojo3 is effectively held-out data (good!) but very little data
  - **Goal:** Properly validate AI accuracy using train/test split or cross-validation
  - **Approaches to Consider:**

    **Option 1: Use mojo3 as held-out test set (RECOMMENDED)**

    - Train on: mojo1 + mojo2 + other projects (exclude mojo3)
    - Test on: mojo3 only (6,468 rows, never seen during training)
    - Pro: Clean separation, represents "future" data
    - Con: Only one test project (can't generalize to other styles)

    **Option 2: K-Fold Cross-Validation by Project**

    - Fold 1: Train on mojo2+others, test on mojo1
    - Fold 2: Train on mojo1+others, test on mojo2
    - Fold 3: Train on mojo1+mojo2, test on another project
    - Pro: Uses all data, measures generalization across projects
    - Con: Complex, time-consuming

    **Option 3: Random 80/20 split**

    - Randomly split ALL data 80% train, 20% test
    - Pro: Simple, maximizes training data
    - Con: Mixes projects, doesn't test generalization to new projects

  - **Recommendation:** Use Option 1 first (mojo3 as test set)

    - Current models likely already exclude mojo3 from training (only 11 rows in CSV)
    - Can measure TRUE accuracy on mojo3
    - Then use Option 2 for deeper validation if needed

  - **Implementation Steps:**

    1. **Verify current training data:** Check `selection_only_log.csv` - confirm mojo3 mostly excluded
    2. **Run backfill on mojo3:** Already done! (6,468 predictions exist)
    3. **Calculate test accuracy:**
       ```sql
       -- mojo3 selection accuracy (test set)
       SELECT AVG(selection_match)*100 FROM ai_decisions
       WHERE project_id='mojo3';
       ```
    4. **Compare with training accuracy:** Backfill mojo1/mojo2, measure accuracy
    5. **Document findings:** "Training accuracy: 65%, Test accuracy: 54%" (example)
    6. **Use for model improvement:** Identify where model fails on test data

  - **Files to Check:**

    - `data/training/selection_only_log.csv` - What data was used for training?
    - `scripts/ai/train_ranker_v3.py` - How was train/val split done?
    - `scripts/ai/train_crop_proposer_v2.py` - Crop model training split?

  - **Expected Outcome:**
    - Know TRUE accuracy of current models (on held-out test data)
    - Identify overfitting (if training accuracy >> test accuracy)
    - Guide future model improvements based on test performance

### Data Integrity Backfill (High)

- [x] **Backfill missing crop data caused by dimension-logging bug** [PRIORITY: HIGH] ✅ **COMPLETED Oct 31, 2025**

  - **Scope:** Rows/images impacted when Desktop Multi-Crop logged dimensions as (0,0)
  - **Action:** Identify affected items → recompute from source images/sidecars → validate → write to SQLite v3 → snapshot
  - **Output:** Verified updates in `data/training/ai_training_decisions/*.db` and daily snapshot

- [ ] **Generalize mojo3 backfill scripts for any project** [PRIORITY: MEDIUM]

  - **Current Scripts:**
    - `scripts/ai/backfill_mojo3_phase1a_ai_predictions.py` (AI predictions from originals)
    - `scripts/ai/backfill_mojo3_phase1b_user_data.py` (User selections/crops from finals)
  - **Action:**
    - Accept command-line arguments for any original directory and final directory
    - Rename scripts to remove "mojo3" prefix (e.g., `backfill_project_phase1a_ai_predictions.py`)
    - Update to work with any project structure, not just mojo3
  - **Benefit:** Reuse for mojo1, mojo2, or any future project backfills

- [x] **Update and clean up backfill documentation** [PRIORITY: HIGH] ✅ **COMPLETED Oct 31, 2025**

  - **Completed Actions:**
    - ✅ Created comprehensive new guide: `Documents/guides/BACKFILL_QUICK_START.md`
    - ✅ Documents new 3-phase process (1A: AI predictions, 1B: User ground truth, Phase 2: Merge)
    - ✅ Marked old docs as OBSOLETE with pointers to new process:
      - `Documents/archives/sessions/2025-10-22/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md`
      - `Documents/archives/misc/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md`
      - `Documents/archives/misc/BACKFILL_QUICK_START.md`
    - ✅ Removed all CSV/timesheet references from current documentation
    - ✅ Emphasized: Physical images are source of truth, never copy AI predictions as user data
    - ✅ Explained user_action semantics (approve/crop/reject, NO 'skip')
    - ✅ Documented coordinate tolerances (2% for approve detection, 1% for comparison, 5% for metrics)
    - ✅ Included real example: mojo3 backfill with actual numbers and results

- [ ] **Backfill AI predictions for historical projects** [PRIORITY: MEDIUM]
  - **Goal:** Generate AI predictions for mojo1 and mojo2 projects to expand training data and measure AI improvement over time
  - **Why This Is Valuable:**
    - We have thousands of user decisions (ground truth) in mojo1/mojo2 databases
    - We DON'T have AI predictions for those decisions (AI system didn't exist then)
    - Running current AI models on those historical images would create:
      - Selection match data: Did AI pick what user picked?
      - Crop match data: Was AI's crop close to user's crop?
      - Confidence calibration: How confident was AI when right vs wrong?
    - This would give us 10,000+ AI-vs-user comparisons for training analysis
  - **Process:**
    - Use Phase 1A approach: Run ranker + crop proposer on original images
    - Create temp database with AI predictions
    - Compare with real database (which has user ground truth)
    - Add AI prediction columns WITHOUT touching user data
    - Analyze: How would current AI have performed on historical projects?
  - **Benefits:**
    - Measure AI improvement: "On mojo1, AI would have been 45% accurate. On mojo3, it's 54%!"
    - More training data for model improvements
    - Identify patterns: What types of images does AI struggle with?
    - Better confidence calibration
  - **Projects to Process:**
    - mojo1 (if original images still accessible)
    - mojo2 (if original images still accessible)
  - **Status:** Add to backlog after mojo3 backfill complete

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

#### Dashboard Server Test Failing (New)

- [ ] Fix dashboard server test startup failure (investigate and resolve)

  - Observed:
    - `scripts/tests/test_runner.py` → "Dashboard Test" consistently fails after ~42–48s.
    - Recent runs still fail even with snapshot skipping and richer logging.
  - Repro command (saves full logs):
    ```bash
    cd /Users/eriksjaastad/projects/image-workflow
    source .venv311/bin/activate
    export DASHBOARD_SKIP_SNAPSHOTS=1
    python /Users/eriksjaastad/projects/image-workflow/scripts/tests/test_dashboard.py 2>&1 | tee /Users/eriksjaastad/projects/image-workflow/sandbox/dashboard_test_$(date -u +%Y%m%dT%H%M%SZ).log
    ```
  - What we already tried (did not resolve):
    - Added `--skip-snapshots` flag and `DASHBOARD_SKIP_SNAPSHOTS` env to `scripts/dashboard/run_dashboard.py` to bypass preprocessing.
    - Modified `scripts/tests/test_dashboard.py` to start server with `--debug --skip-snapshots`, increased startup wait, captured stdout/stderr on failure, terminated process to prevent cascading timeouts.
    - Replaced `requests` usage with stdlib `urllib` to remove external dependency and reduce early aborts.
    - Updated test runner to print both stdout and stderr on dashboard test failure when `-v` is used.
  - Suspicions / next steps:
    - Template resolution: `productivity_dashboard.py` calls `render_template("dashboard_template.html")` but no file exists; likely 500s on `/` even if server starts. Consider switching to `render_template_string(DASHBOARD_TEMPLATE)` or embedding the template file.
    - Verify Flask import path and environment activation (`.venv311`).
    - Confirm port availability on 5002 and that binding is successful on `127.0.0.1`.
    - After capturing the log above, paste the stderr excerpt here and fix accordingly.

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

- [ ] **Add backup status indicators to dashboard** [PRIORITY: MEDIUM]

  - **Goal:** Show backup health in dashboard header (like validation status)
  - **Data Source:** `~/project-data-archives/image-workflow/backup_status.json`
  - **Display:** Last backup date, status (success/failed/overdue), file count, size
  - **Colors:** Green for recent success, red for failed/overdue
  - **Integration:** Add to both current project dashboard and main productivity dashboard
  - **Alert Logic:** Flag if backup >48 hours old

- [x] **Create comprehensive backup testing and monitoring system** ⭐ **COMPLETED**

  - **Backup System Tests** (`scripts/tests/test_backup_system.py`):
    - ✅ Backup status file validation
    - ✅ Recent backup existence verification
    - ✅ Database discovery testing
    - ✅ Backup script execution testing
    - ✅ Cron job configuration verification
    - ✅ Backup verification logic testing
  - **Backup Health Monitoring** (`scripts/tools/backup_health_check.py`):
    - ✅ Runs every 6 hours via cron
    - ✅ Checks backup status file integrity
    - ✅ Verifies recent backups exist
    - ✅ Monitors backup log freshness
    - ✅ **LOUD ALERTS** for backup failures (macOS notifications)
  - **Cron Integration**: Health check added to `setup_cron.sh`
  - **Prevention**: Tests for tests - ensures backup monitoring itself works
  - **Paranoia Level**: Maximum - alerts on any backup system issues

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

- [ ] **Set up automated cloud backup using rclone** [PRIORITY: HIGH] ⭐ **TOMORROW**

  - **Goal:** Get off-site backups running before more data loss from silent failures

  - **EXISTING INFRASTRUCTURE (Already Built!):**

    - ✅ `scripts/backup/weekly_rollup.py` - Complete weekly backup script
    - ✅ Cron job slot: `10 2 * * 0` (Sunday 2:10 AM Eastern)
    - ✅ Local retention: 12 weeks in `~/project-data-archives/image-workflow/`
    - ✅ Daily backups: Already running, feeding into weekly rollup

  - **WEEKLY ROLLUP PROCESS:**

    1. **Find daily backups** from last 7 days (e.g., 2025-10-27 to 2025-11-02)
    2. **Compress** into `weekly_20251027_20251102.tar.zst`
    3. **Create manifest** with file counts, sizes, SHA256 hashes
    4. **Upload** to `gbackup:weekly-rollups/` via rclone
    5. **Clean up** old archives (keep 12 weeks locally + cloud)

  - **WHAT NEEDS TO BE DONE:**

    - ✅ Daily backups: Already working → `~/project-data-archives/image-workflow/YYYY-MM-DD/`
    - ✅ Weekly script: Already implemented and ready
    - ✅ Cron job: **UPDATED** in `setup_cron.sh` to use `weekly_rollup.py`
    - ⏳ **Configure rclone remote** named `gbackup` (user has done this before)
    - ⏳ **Test upload** to verify rclone works
    - ⏳ **Enable weekly cron job** (`bash scripts/setup_cron.sh`)

  - **NEXT STEPS (Tomorrow):**

    1. Configure/test rclone remote `gbackup`
    2. Run `bash scripts/setup_cron.sh` to enable weekly job
    3. Monitor first weekly backup (next Sunday 2:10 AM Eastern)
    4. Add backup status to dashboard

  - **TIME SENSITIVE:** Daily backup runs tonight at 2:10 AM Eastern!

    - Check tomorrow if it succeeded (look in `~/project-data-archives/image-workflow/`)
    - Fix any issues before enabling weekly rollup

  - **DATA INTEGRITY CONCERN:** Recent silent failures erased data - cloud backup critical now!

- [ ] Auto-upload finished ZIP from 07_finish_project to Drive [PRIORITY: MEDIUM]
  - Hook: after successful finish with `--commit`
  - Target: `gbackup:deliveries/<projectId>/`
  - Flow: copy → check → (optional) delete local zip
  - Reuse rclone remote `gbackup` and daily cron log

---

## ✅ Recently Completed

**Comprehensive Safety & Monitoring System (Oct 30, 2025)**

- [x] **Implement comprehensive error monitoring system** ⭐ **GAME CHANGER!**
  - Loud, immediate alerts for critical errors with macOS notifications
  - Real-time data quality validation (prevents silent corruption)
  - File operation error detection and logging
  - Daily automated validation reports with alerts
  - Enhanced FileTracker logging with failure detection
- [x] **GitHub Actions test runner** - Multiple daily test runs without blocking commits
- [x] **Comprehensive AI Reviewer hotkey testing** - All routing logic validated
- [x] **Cron-based daily validation** - Noon Eastern time automated checks

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
- [x] **Added database backups to daily backup system** - SQLite files now backed up to project-data-archives/databases/
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
