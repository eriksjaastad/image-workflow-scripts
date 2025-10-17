# Current TODO List

**Last Updated:** October 16, 2025

---

## ⚠️ **IMPORTANT WORKFLOW RULE**
**ONE TODO ITEM AT A TIME** - Complete one task, check in with Erik, get approval before moving to the next item. Never complete multiple TODO items in sequence without user input. This prevents issues and ensures quality control.

---

## 🤖 **AI CROP ASSISTANT - IN PROGRESS**

### **Timeline:**
- **Thursday (today):** Crop 1 hour → AI logs silently ⏰ Starts in 22 min
- **Friday:** Crop 2-3 hours → I train model after session
- **Saturday:** Crop with `--ai-assist` flag (AI suggestions ready!)
- **Sunday:** Start automation pipeline testing in sandbox
- **Monday:** Wrap up current work, prep for next project

### **Current Status:**
- ✅ Phase 1: Data collection infrastructure - COMPLETE
- ✅ Phase 2: Live training capture - COMPLETE (implemented today)
- ⏸️ Phase 3: Train crop model - WAITING (after Friday's cropping)
- ⏸️ Phase 4: AI suggestions - WAITING (ready for Saturday)

### **Pending Tasks:**
- [ ] User crops ~1 hour today (Thursday) - **STARTING SOON**
- [ ] User crops 2-3 hours Friday (~300-450 images total)
- [ ] Train crop proposer model (after Friday session, ~30 min)
- [ ] Implement `--ai-assist` flag in Multi Crop Tool
- [ ] Add logging for approved/modified/rejected suggestions
- [ ] Test AI suggestions on sandbox subset
- [ ] **NEW: Sunday automation pipeline start (sandbox only)**

### **Key Insight from Erik:**
> "Everything the AI learns from me is gonna turn around and apply to the sandbox"
- ✅ **YES!** AI trains on your production cropping decisions
- ✅ **Then applies/tests** all automation in sandbox ONLY
- ✅ **Safe separation**: Learn from production, experiment in sandbox

**Documentation:** `AI_CROP_ASSISTANT_READY.md`, `scripts/ai/README.md`

---

## 🔧 **CLEANUP & ORGANIZATION**

### **Script Organization & Naming**
- [ ] Retire `scripts/01_desktop_image_selector_crop.py` to archive directory
- [ ] Consider renaming `00_finish_project.py` to `07_finish_project.py` (or similar)
- [ ] Note: We're missing `03` in numbering - evaluate if we need consistent numbering
- [ ] Move `import_historical_projects.py` from scripts/ top level (one-time utility)
- [ ] Move or organize `cleanup_logs.py` (currently in scripts/ top level)

### **Cursor AI Configuration**
- [ ] Review and implement cursor global rules kit document (newly added)
- [ ] Goal: Prevent the 3-hour circular debugging frustration from happening again
- [ ] Document learnings about what went wrong with ChatGPT session

### **General Housekeeping**
- [ ] Review all top-level scripts/ files for proper organization
- [ ] Verify all numbered scripts (00-07) are in correct order
- [ ] Check if any other one-time utilities should be moved to tools/

---

## Delivery upload improvements (Google Drive)

- Add rclone upload workflow for large zips (single-file):
  - Command: `rclone copy <zip-file> gdrive:Deliveries/<ProjectName> --drive-chunk-size 256M --progress`
  - Note: Drive uploads are chunked; this improves stability and resumes.
- Optional later: expose "Upload to Drive" helper flag in `prezip_stager.py` (uses rclone if configured).
- Optional later (deferred): evaluate split/merge flow for true parallel uploads; only if we can re-merge to a single zip automatically on the receiving side (client should still download one file).

---

## 🔥 **HIGH PRIORITY**

### **Next: Project Comparison Visualizations**
- **Goal:** Compare productivity across projects (Mojo1 vs Mojo2 vs historical)
- **Status:** Table working, need charts for visual comparison
- **See:** "Dashboard Enhancements" section below for chart variations

### **Shared Stage-Grouping Utility (Counting + API)**
- **Goal:** Create a reusable grouping/counting function shared by selectors/sorters to compute stage groups over a directory.
- **Scope:**
  - Provide `group_by_consecutive_stages(files)` using centralized logic (timestamp+stage sorted; 1→1.5→2→3)
  - CLI `scripts/tools/count_stage_groups.py <dir>` prints totals:
    - pairs (1↔2, 1↔1.5, 1.5↔2), triplets (1↔2↔3, 1↔1.5↔2), quads (1↔1.5↔2↔3)
    - overall group count and distribution
  - Export API under `scripts/utils/companion_file_utils.py` for consistency with web selector logic
- **Why:** Single source of truth for grouping across desktop/web selector and automation runs; quick sanity counts before/after runs.
- **Safety:** Read-only; no moves/deletes. Unit tests for representative fixtures.

### **Sandbox Automation Orchestrator**
- **Create sandbox copy**: copy remaining images from `mojo1/` into `sandbox/` (read-only source, safe target)
- **Run thinning steps**: pHash near-dup → FAISS semantic → timestamp-window clusters (reports + `_review_dupes/` staging only)
- **Grading metrics**: counts kept/flagged, estimated review saved, thresholds summary
- **Multi-source grouping**: extend similar-image grouper to accept `selected/` + `crop/` in one pass
- **Sidecar visual cues**: generate `.cropped` sidecars and add badge in Character Sorter
- **One-shot script**: single CLI to run all steps with dry-run/stage modes

### **Sidecar crop flag + UI badge + multi-source grouping**
- **Add sidecar flag**: create `.cropped` sidecar files next to PNGs in `crop/` (same-stem). No image changes; travels with file during moves.
- **Character sorter badge**: detect `.cropped` companions and render a small "Cropped" badge on thumbnails.
- **Similar-image grouper support**: add optional multi-source input (e.g., `selected/` + `crop/`) and preserve/propagate sidecars.
- **CLI helper**: simple script/command to generate sidecars for existing `crop/` PNGs.
- **Safety**: read-only marker; never modifies image bytes.

### **Quick UI wins – Web Image Selector header**
- **Add separate counters**: show "Kept approved" vs "Kept to crop" in the header summary
- **Add progress bar**: small inline bar for remaining files (current batch + overall)
- **Accuracy rules**: file-count based; default-delete for untouched; per-image-skip treated consistently with server
- **Style**: follow `Documents/WEB_STYLE_GUIDE.md` (colors, spacing, typography)
- **Scope**: UI only; no behavioral changes to move/delete
- **Why**: faster feedback during long sessions; reduces mental load during tedious passes

### **Cross-timestamp duplicate set pruning (safe staging)**
- **Goal**: detect visually identical sets with same stage lineup (1, 1.5, 2, 3) but different timestamps; keep one set, stage others for review.
- **Approach**: per-stage CLIP embedding similarity within candidate clusters; require same stage names across compared sets; cross-check YAML similarity.
- **Pipeline**: build candidate pairs via similar-image grouper; verify stage-by-stage match; produce report.
- **Action**: move redundant whole-set (image + all companions) into `delete_review/` (not Trash) for manual confirmation.
- **Modes**: dry-run report, conservative thresholds, manual approve step.

---

## 🧠 **Automation Pipeline Plan (Planning Only)**

> **Goal:** Reduce Erik's manual review time by auto-picking winners in 2–4 near-duplicate sets and deciding if crops are needed.
> 
> **⚠️ CRITICAL: SANDBOX-ONLY TESTING**
> - ALL automation testing happens in `sandbox/` ONLY
> - NO production file operations until 100% confident (zero bugs, proven reliability)
> - Must extensively use and validate tools in sandbox before any production use
> - Bugs, issues, and refinements happen safely in sandbox environment

### **Full Automation Workflow (2-Step Decision Process)**

**Step 1: Image Selection**
- AI analyzes a group of near-duplicate images (e.g., 4 versions of same character)
- Selects the best image based on: stage number, anomaly checks, quality metrics
- Output: Chosen image + reasoning

**Step 2: Crop Decision**
- AI analyzes the chosen image
- Decides: Does this image need cropping?
  - **Yes** → Proposes crop area (coordinates + reasoning)
  - **No** → Marks as "crop not needed" (already well-framed)
- Output: Crop recommendation (if needed) + reasoning

**Focus:** AI anomalies (hands/feet issues, missing body parts, extra fingers) - NOT photography issues (blur, exposure)

### **A. Objectives & Success Criteria**
- **Primary objective**: 
  1. Auto-select best image per near-duplicate set
  2. Decide if selected image needs cropping
  3. If yes, propose crop area
  4. Present decisions for review before any file operations
- **Safety**: 
  - **SANDBOX-ONLY until proven 100% bug-free**
  - Non-destructive by default; mark decisions, never move files automatically
  - All operations reversible through review UI
  - Companion integrity: Always move image + `.yaml`/`.caption` and any same-stem companions together
  - Never alter image bytes
- **Testing requirements**:
  - **Must extensively test in sandbox before ANY production use**
  - Zero tolerance for bugs affecting production content
  - All edge cases, errors, and refinements handled in sandbox
  - Production deployment only after complete confidence in reliability
- **Measurable success**:
  - Manual time reduction ≥ 50% on duplicate thinning passes
  - Auto-pick agreement with Erik's choice ≥ 98% on calibrated sets
  - Crop need detection accuracy ≥ 95%
  - Crop proposal acceptance ≥ 90%
  - False-positive staging (wrongly moved a keeper) ≤ 1 per 1,000 images
  - Zero bugs in sandbox testing before production consideration

### **B. What Tools Do What (Human vs Automation)**
- **Human-in-the-loop tools (existing/updated)**
  - `scripts/01_web_image_selector.py`: stays primary for version selection; may receive a "AI pick" hint later (optional).
  - `scripts/02_web_character_sorter.py`: add optional badges (e.g., "cropped" sidecar indicator already planned) and a future "AI suspect" tag.
  - **NEW: `scripts/07_automation_reviewer.py` (HIGH PRIORITY)**: Web UI to review ALL AI automation decisions before applying them.
    - **⚠️ RUNS IN SANDBOX ONLY - No production use until extensively tested and bug-free**
    - **Purpose**: Review AI's 2-step decision process (image selection + crop decision)
    - **What it shows**:
      - All images in each group (like web image selector layout)
      - **Step 1 Result**: AI's chosen image highlighted with green border + "AI Pick" badge
      - **Step 2 Result**: 
        - If crop needed: Proposed crop drawn on image (dotted green rectangle)
        - If no crop needed: "No crop required" badge
      - Reasoning for BOTH decisions:
        - Selection: "Highest stage (3), passed anomaly checks, best quality"
        - Crop: "Body cut at waist, recommend crop to full frame" OR "Already well-framed, no crop needed"
      - Quick stats: Confidence scores, anomaly flags, stage info
    - **Actions**:
      - ✅ Approve (accept AI decision - marks for actual move)
      - ❌ Reject (keep all images - no action)
      - 🔧 Override (pick different image or adjust crop)
      - ⏭️ Skip (review later)
    - **Batch workflow**:
      - Process groups in batches (20-50 at a time)
      - Keyboard shortcuts: 1/2/3/4 = override pick, A = approve, R = reject, S = skip
      - Progress bar: "Reviewed 45/200 groups"
      - Submit batch → writes approved decisions to staging file
    - **Data flow (SANDBOX-ONLY)**:
      1. Automation runs in "mark only" mode → writes decisions to `sandbox/automation_decisions.jsonl`
         - Each decision includes: chosen image, crop needed (yes/no), crop coords (if needed), reasoning
      2. Reviewer loads decisions → shows UI for review in sandbox
      3. User approves/rejects/overrides → writes to `sandbox/approved_decisions.jsonl`
      4. Commit script reads approved decisions → executes actual file moves (sandbox only)
      5. **Extensive testing and validation in sandbox**
      6. **Only after 100% confidence and zero bugs → consider production deployment**
    - **Safety**: 
      - Automation NEVER moves files directly - always through review step
      - ALL testing in sandbox - production files never touched until proven
      - Full rollback capability at every stage
  - `scripts/06_web_duplicate_finder.py`: remains useful for visual comparisons on tough clusters.
- **Automation scripts (offline, reversible)**
  - `scripts/tools/compute_phash.py`: compute pHash; write to `data/ai_data/hashes/`.
  - `scripts/tools/compute_embeddings.py`: OpenCLIP ViT-B/32; write to `data/ai_data/embeddings/`.
  - `scripts/tools/detect_hands_feet.py`: MediaPipe Hands (+ optional Pose feet presence); output per-image presence/anomaly hints to `data/ai_data/hands/`.
  - `scripts/utils/phash_group_near_dupes.py`: group pairs/sets by Hamming distance (≤ 8–10).
  - `scripts/utils/clip_group_near_dupes.py`: group pairs/sets by CLIP cosine (≥ 0.95; tunable).
  - `scripts/utils/auto_pick_near_dupes.py`: merge groupers, apply stage-aware ranking + anomaly gates, generate report; optional staging to `_review_dupes/`.
  - `scripts/orchestrators/automation_pipeline.py`: one CLI to run the pipeline in dry-run/stage modes, with metrics summary.

### **C. Two-Step Decision Logic**

**Step 1: Image Selection (Stage-aware + Anomaly-aware)**
- Candidate groups = filename stage runs + pHash near-exacts + CLIP semantic near-dupes
- Rank within group:
  1) Highest stage number wins if it passes anomaly gate
  2) If tie, prefer fewer detected hands/feet in-frame; if present, require anomaly checks to pass
  3) Break ties by larger resolution, then latest mtime
- Anomaly gate (AI-specific):
  - Hands/feet presence signal (fast): presence = higher risk; down-rank unless scores are strong
  - Simple hand heuristics from keypoints: flag likely extra/fused fingers if geometry inconsistent
  - Optional "missing belly button" is future work; keep placeholder tag
- **Output**: Chosen image + confidence + reasoning

**Step 2: Crop Decision (for chosen image only)**
- Analyze chosen image for crop necessity:
  - **Needs crop if**: Body cut off (waist/legs), excess background, off-center framing, head too close to edge
  - **No crop if**: Already well-framed (full body visible, centered, good composition)
- If crop needed:
  - Calculate optimal crop box using saliency maps + keypoint detection
  - Constraints: Keep original aspect ratio, head preference, ≥1/3 body visible
  - Score: `α·SaliencyIn – β·AnomalyOverlap – γ·HeadCut – δ·JointCut`
- **Output**: 
  - Crop needed: Yes/No
  - If yes: Crop coordinates + confidence + reasoning
  - If no: Reasoning for no crop needed

References: `Documents/hand_foot_anomaly_scripts.md`

### **D. Thresholds (initial; calibrate on sample)**
- pHash Hamming: near-exact ≤ 8; lenient ≤ 10.
- CLIP cosine: ≥ 0.95 to group; raise to 0.97 if too broad.
- Ambiguity margin: if top two scores within 0.05 → send to human review.
- Hands/feet presence: treat as risk booster; always route borderline groups with hands/feet for review.

### **E. Milestones & Tasks (Planning → Calibration → Tools → Review UI)**
1) Planning & calibration set (no code changes)
   - Define 100–200 image calibration subset across multiple characters and stages.
   - Enumerate anomaly tags we care about first (hands/feet, belly button missing, facial stacking).
   - Decide reporting schema fields for auto-pick report (group_id, chosen, losers, scores, reasons).
2) Offline feature tools (standalone; safe)
   - Build: `compute_phash.py`, `compute_embeddings.py`, `detect_hands_feet.py`.
   - Run on calibration subset; store outputs under `data/ai_data/`.
3) Grouping + auto-pick (reports only)
   - Build: `phash_group_near_dupes.py`, `clip_group_near_dupes.py`, `auto_pick_near_dupes.py` (report only).
   - Validate groups and picks on calibration subset; tune thresholds.
4) Decision marking mode (NO file moves)
   - Automation writes decisions to JSONL (chosen image, crop coords, reasoning, confidence)
   - Output: `sandbox/automation_decisions.jsonl`
   - NO file operations - just marks choices
   - Includes sidecar markers: `.ai_chosen`, `.ai_crop_proposed` for easy visual review
5) **Review UI - Automation Reviewer (CRITICAL PATH)**
   - **Build: `scripts/07_automation_reviewer.py` (Flask web UI)**
   - **Features:**
     - Load decisions from `sandbox/automation_decisions.jsonl`
     - Display groups in web image selector layout
     - Highlight AI's choice with green border + badge
     - Draw proposed crop rectangle on image (canvas overlay)
     - Show reasoning tooltip (stage, confidence, anomaly flags)
     - Keyboard shortcuts: 1/2/3/4 (override), A (approve), R (reject), S (skip), Enter (next)
     - Batch processing: 20-50 groups at a time
     - Progress tracking: "Reviewed 45/200 groups (22.5%)"
   - **Output:** `sandbox/approved_decisions.jsonl` (only approved items)
   - **Safety:** No file moves in this tool - pure review
6) Commit script (executes approved moves)
   - **Build: `scripts/tools/apply_automation_decisions.py`**
   - Reads `sandbox/approved_decisions.jsonl`
   - Executes actual file moves with companion handling
   - Logs all operations via FileTracker
   - Dry-run mode by default
   - **Safety:** Only runs on explicitly approved decisions
7) Orchestrator & metrics
   - `automation_pipeline.py` to chain steps and emit a summary (precision/recall estimates on calibration set, time saved)
   - Three-phase workflow: 1) Mark decisions, 2) Review in UI, 3) Apply approved moves

### **F. Testing & Evaluation (SANDBOX-ONLY, before any production consideration)**

**Phase 1: Unit Testing**
- Golden fixtures for grouping/selection rules (small fixture dirs committed to `scripts/tests/fixtures/`)
- Dry-run as default; CI-style check that no file moves occur without `--stage`
- Companion integrity tests (image + sidecars preserved through moves)
- Calibration evaluation: report agreement with Erik's picks on the subset; adjust thresholds until ≥98%

**Phase 2: Sandbox Integration Testing (CRITICAL)**
- **ALL testing happens in `sandbox/` directory ONLY**
- Run full pipeline on sandbox copy of production data
- Test scenarios:
  - Image selection accuracy (compare AI picks to your actual choices)
  - Crop need detection (does it correctly identify images that need cropping?)
  - Crop proposal quality (are proposed crops acceptable?)
  - Edge cases: unusual poses, partial bodies, multiple characters
  - Error handling: missing files, corrupted data, unexpected inputs
- **Requirements for production consideration**:
  - ✅ Zero bugs discovered in sandbox testing
  - ✅ Extensive use (hundreds of groups reviewed)
  - ✅ Selection accuracy ≥98%
  - ✅ Crop decision accuracy ≥95%
  - ✅ All edge cases handled gracefully
  - ✅ Complete confidence in reliability
- **Timeline**: Stay in sandbox as long as needed - NO rush to production

**Phase 3: Production Deployment (Only after Phase 2 complete)**
- Only proceed after 100% confidence from sandbox testing
- Start with small batches (10-20 groups)
- Monitor closely for any issues
- Easy rollback to manual workflow if needed

### **G. Risks & Safeguards**
- CLIP over-grouping across pose changes → mitigate with higher cosine threshold + stage-name parity.
- Hands detection false positives in stylized cases → treat presence as a review trigger, not an auto-reject.
- Always stage to review, never hard-delete; centralized logging via FileTracker.

### **H. Deliverables**
- Auto-pick decisions file: `sandbox/automation_decisions.jsonl` (all AI choices with 2-step reasoning)
  - Format per decision:
    ```json
    {
      "group_id": "char1_20250101_120000",
      "images": ["img1.png", "img2.png", "img3.png", "img4.png"],
      "step1_selection": {
        "chosen": "img3.png",
        "reasoning": "Highest stage (3), passed anomaly checks, sharpest",
        "confidence": 0.95
      },
      "step2_crop": {
        "needs_crop": true,
        "reasoning": "Body cut at waist, excess top background",
        "crop_coords": [0.1, 0.15, 0.9, 0.95],
        "confidence": 0.87
      }
    }
    ```
- Automation Reviewer UI: `scripts/07_automation_reviewer.py` (web interface for sandbox review)
- Approved decisions file: `sandbox/approved_decisions.jsonl` (user-approved choices only)
- Commit script: `scripts/tools/apply_automation_decisions.py` (executes approved moves in sandbox)
- Metrics report: Selection accuracy, crop decision accuracy, time saved, approval rate, override patterns
- **Sandbox testing documentation**: Detailed log of all testing, bugs found/fixed, confidence assessment

### **I. References**
- `Documents/hand_foot_anomaly_scripts.md`
- `image_batch_culling_pipeline.md`, `similar_image_dedup_automation_plan.md`, `stage_aware_image_culling_workflow_v2.md`

> Note: This section is planning-only. No scripts will be created until we review and approve this plan.

---

## 🔧 **MEDIUM PRIORITY**

### 1. Desktop Selector Selection Toggle Bug
**Issue:** When one image is selected by hotkey and then another image's crop position is modified, the original image doesn't toggle back to delete.
- **Expected:** Only ONE image should be kept at a time
- **Impact:** UX issue during long image processing sessions
- **File:** `scripts/01_desktop_image_selector_crop.py`
- **Note:** Lower priority - not using desktop selector for a few days

### 2. Improve Test Coverage
- Add unsorted input tests (verify pre-sorting requirement)
- Add unknown stage handling tests
- Add min_group_size variation tests
- Add two-runs-in-sequence tests
- Add more edge case tests for companion_file_utils.py (permission errors, malformed data)
- Add cleanup functionality tests for check_companions.py

### 3. Create Missing Test Files
**Need tests for:**
- `04_multi_crop_tool.py`
- `06_web_duplicate_finder.py`
- `utils/character_processor.py`
- `utils/duplicate_checker.py`
- `utils/recursive_file_mover.py`
- `utils/similarity_viewer.py`
- `utils/triplet_deduplicator.py`
- `utils/triplet_mover.py`

### 4. Companion file deletion parity (standardize deletes)
**Issue:** Some delete paths still handle only `.yaml`. We should always delete image + ALL companion files using centralized utilities.

**Files & examples:**
- `scripts/utils/base_desktop_image_tool.py` — YAML-only delete in base method used by inheritors.
- `scripts/04_multi_crop_tool.py` — Calls base `safe_delete(png_path, yaml_path)` during delete flow.
- `scripts/utils/triplet_deduplicator.py` — Dry-run and removal reference only `.yaml`.
- `scripts/archive/04_batch_crop_tool.py` — Legacy; uses `safe_delete(png_path, yaml_path)`.

**Actions:**
- Replace base `safe_delete` implementation with centralized utility:
  - `safe_delete_image_and_yaml(png_path, hard_delete=False, tracker=self.tracker)` (already deletes ALL companions)
- Update `triplet_deduplicator.py` to:
  - Use `find_all_companion_files(png_file)` for dry-run printing and actual deletion
  - Use `safe_delete_paths([png] + companions, ...)` or `safe_delete_image_and_yaml`
- Ensure FileTracker logging preserved (source dir, file_count, files list)

**Notes:**
- `01_desktop_image_selector_crop.py` already delegates deletes to the centralized companion delete — no change needed.
- Add tests for caption-delete paths in `utils/triplet_deduplicator.py` after changes.

---

## 📚 **LOW PRIORITY / FUTURE**

### Code Conventions & Patterns Catalog
**Create:** `Documents/CONVENTIONS_REFERENCE.md`
- Analyze all scripts for reusable patterns
- Document Flask structure, CSS, JavaScript patterns
- Document matplotlib setup, event handling
- Create ready-to-use code templates
- Benefits: Consistency, maintainability, easier onboarding

### Scripts layout cleanup (planning)
- Clarify conventions: `scripts/tools/` = runnable CLIs/automation; `scripts/utils/` = reusable libraries only.
- Audit `scripts/tools/` for code that belongs in `scripts/utils/` and propose moves.
- Refactor imports after moves; replace `project_root` path hacks with `sys.path.insert` only where necessary.
- Add `scripts/README.md` summarizing directory purposes and import rules.
- Add tests covering `utils/recursive_file_mover.py` CLI behaviors.

### Create Local Homepage
Build custom homepage in Documents with links to all AI systems and tools

### Web Interface Template System Investigation
Evaluate if template would simplify web tool maintenance vs add complexity

### Experiment: Hand/Foot Anomaly Scripts
General line item to check out/test hand and foot anomaly scripts when time allows. Purpose: see if they catch any of my mistakes or produce signals our AI could use. Not urgent; exploratory—run on recent batches and jot a brief note on usefulness and potential integration.

---

## 🧩 Planned UX Enhancement (Web Image Selector)

### Feature: Right‑Side Crop Button + Crop Mode (Plan Only)
- **Context**: In `scripts/01_web_image_selector.py`, each row displays 2–4 images with a right‑side vertical control column. Current buttons are 1/2/3/4, Skip, ▼. Keyboard: 1/2/3/4 select; Q/W/E/R = select for crop. Batch processing applies moves/deletes at submit time.
- **Goal**: Add a single Crop button in the right control column (per row) to enable mouse‑only crop selection without crossing the screen.

- **Right‑side layout (per group)**:
  - `[1]`
  - `[2]`
  - `[3]` (if present)
  - `[4]` (if present)
  - `[Crop]`  ← toggles crop mode for the row (highlight when ON)
  - `[Skip]`
  - `[▼]`

- **Behavior**:
  - No selection yet:
    - Click Crop → `cropMode=true` (for this group). Next number click selects that index with `crop=true`; `cropMode` then turns OFF.
  - Selection already set (`selectedImage = k`):
    - Click Crop → toggles `state.crop` for the current selected image (k) on/off. Selection index remains k.
  - Number buttons:
    - If `cropMode=false` → select/deselect exactly as today (single keep; deselect returns to delete state).
    - If `cropMode=true` → select index with `crop=true`, then `cropMode=false`.
  - Keyboard Q/W/E/R unchanged; ENTER/↑ navigation unchanged.
  - State only; actual moves/deletes occur on "Process Current Batch" (unchanged).

- **Visuals**:
  - Crop button shows an active style when `cropMode=true`.
  - Selected card shows white "crop-selected" outline when `state.crop=true` (existing class).

- **Implementation TODOs (execute when tool not in use)**:
  1) Add `Crop` button to right control column in the template (same style class as row buttons).
  2) Frontend JS: track per‑group `cropMode`; update handlers:
     - Crop click toggles `cropMode` when no selection; toggles `state.crop` when selection exists.
     - Number click applies select/crop per rules; clears `cropMode` when used.
     - Update `updateButtonStates` to reflect Crop button highlight.
  3) Keep `/submit` server logic unchanged (it already honors `selectedIndex` + `crop`).
  4) QA: 2/3/4‑image rows; select→crop toggle; cropMode→number; batch submit; confirm FileTracker logs and visuals.

- **Safety**:
  - No changes to keyboard behavior or batch semantics.
  - No changes to character sorter.

### Feature: Work Timer Widget (like Multi‑Crop Tool)
- **Context**: `scripts/01_web_image_selector.py`—long sessions get tedious; a small on‑page work timer helps focus and pacing.
- **Goal**: Add a lightweight, on‑page work timer similar to the multi‑crop tool's timer to encourage timeboxed passes and reduce context switching.

- **UI**:
  - Fixed, subtle header widget (top bar, right side): "Work: 00:00 • Session: 00:00 • Efficiency hint (optional)".
  - Colors match style guide; unobtrusive; no blocking dialogs.

- **Behavior**:
  - Starts when page loads; pauses on prolonged inactivity (e.g., no key/click/scroll for ≥5 minutes) and resumes on action.
  - Optional micro‑nudge every 10–15 minutes (silent visual nudge, no modal).
  - Resets per server session (no persistence needed for v1).

- **Implementation TODOs**:
  1) Add header widget markup and styles in the selector template (reuse existing toolbar).
  2) JS timer module: track active vs total time; pause on inactivity threshold; resume on input.
  3) Hook existing activity events (click/keydown/scroll/mousemove throttle) already present in the page.
  4) Optional: surface the current batch progress alongside timer (non‑blocking).

- **Safety**:
  - Pure front‑end; no server or file‑operation changes.
  - Feature is display‑only; won't affect selection/crop submissions.

---

## 📊 **Dashboard Enhancements**

### 🎨 **Project Comparison Graph Experiments (Next - High Priority)**

**Goal:** Create 5 different chart variations to compare productivity across projects (Mojo1 vs Mojo2). We'll "Frankenstein together something out of all the ideas" to find the best visualization.

**Context:**
- We have detailed per-project, per-tool data (hours, days, images processed, efficiency)
- Need to answer: "Am I going as fast on Mojo2 as I was on Mojo1?"
- Focus on actionable comparisons, not just pretty charts
- User loved "Files processed by tools" chart - aim for similar usefulness

**Chart Variations to Build:**

#### **A. Stacked Bar Chart - Time Investment**
**Purpose:** Show total hours spent per tool, stacked by project
- **X-axis:** Projects (Mojo1, Mojo2)
- **Y-axis:** Hours (left axis)
- **Bars:** Stacked segments for each tool
- **Colors:** Distinct color per tool (consistent across all charts)
- **Tooltip:** Tool name, hours, percentage of total project hours
- **Why useful:** Quick visual of time distribution across tools per project

#### **B. Grouped Bar Chart - Images Processed**
**Purpose:** Compare raw image throughput per tool across projects
- **X-axis:** Projects (Mojo1, Mojo2)
- **Y-axis:** Images processed (left axis)
- **Bars:** Grouped bars per tool (side-by-side for easy comparison)
- **Colors:** Same tool colors as Chart A
- **Tooltip:** Tool name, images processed, project name
- **Why useful:** Direct comparison of productivity numbers

#### **C. Grouped Bar Chart - Efficiency (Images/Hour)**
**Purpose:** Compare efficiency metrics per tool across projects
- **X-axis:** Projects (Mojo1, Mojo2)
- **Y-axis:** Images per hour (left axis)
- **Bars:** Grouped bars per tool showing img/h rates
- **Colors:** Same tool colors as Charts A/B
- **Tooltip:** Tool name, img/h rate, project name
- **Baseline overlay:** Historical average line (if available)
- **Why useful:** Shows if we're getting faster or slower; identifies bottlenecks

#### **D. Two Separate Charts (Time vs Images)**
**Purpose:** Side-by-side comparison of time investment and output
- **Chart D1:** Time Investment (same as Chart A)
- **Chart D2:** Images Processed (same as Chart B)
- **Layout:** Two charts stacked vertically or side-by-side
- **Shared:** Same tool colors, aligned X-axes
- **Why useful:** Separates input (time) from output (images) for clearer analysis

#### **E. Dual Y-Axis Combo Chart (Time + Images)**
**Purpose:** Show relationship between time investment and output on one chart
- **X-axis:** Tools
- **Left Y-axis:** Hours (time bars)
- **Right Y-axis:** Images processed (line or secondary bars)
- **Bars:** Grouped by project (Mojo1 vs Mojo2 side-by-side per tool)
- **Time bars:** Color-coded to left Y-axis (e.g., blue tones)
- **Image bars/lines:** Color-coded to right Y-axis (e.g., green tones)
- **Labels:** Bar width matches tool name abbreviations
- **Tooltip:** Tool name, project, hours, images, img/h
- **Why useful:** Shows both dimensions (time + output) in a single view

**Implementation Details:**
- Use Chart.js (already in dashboard)
- Position table at TOP of dashboard (most important view)
- Position graphs BELOW table
- All charts use same 4-tool filter
- Consistent color scheme across all charts
- Responsive design (mobile-friendly)
- Legend toggles for each series
- Export-friendly (PNG/CSV options later)

**Success Criteria:**
- User can quickly answer: "Am I faster on Mojo2 than Mojo1?"
- Easy to identify bottlenecks (which tool is slowest?)
- Clear visual separation between time investment and output
- Charts complement the table (not redundant)

**Next Steps:**
1. Build all 5 chart variations
2. Review with Erik to see which visualizations are most useful
3. Combine best elements from different charts
4. Polish and integrate final design

### Planned (Lower Priority)
1. Historical average overlays (longer-term historical bands)
2. Script update correlation with productivity
3. Pie chart time distribution
4. CSV/JSON data export
5. GitHub integration for change tracking

### Testing
- Add test coverage for:
  - Project filtering (data_engine + API)
  - Day band computation across DST and month boundaries
  - Marker rendering presence when startedAt/finishedAt provided
  - Toggle persistence for tools/ops and averages

---

## 📝 **Documentation Updates Needed**
1. Add desktop hotkey reference to Knowledge Base (p [ ] \\ and A/S/D/F/B)
2. Document training log flags and schemas
3. Update backup system runbook when implemented
4. **END OF DAY: Cleanup Documents directory** (lots of explosion, some useless/bloated docs)

---

## 📊 **Sandbox Automation Experiments**

### Baseline metrics (subset runs)
- 5-minute shard on `mojo2_subset`: scan=0.05s, group=0.02s, select≈14.86s, moves≈14.86s; shard groups=126; ≈254 groups/min.
- 15-minute full sandbox baseline earlier: select/moves dominated (≈208s in 15-min window on shard).

### Next experiments (queued)
1) Deviation logging: count/list groups where `--quality-aware` picks a lower stage than baseline; produce examples for review.
2) Cheap pre-prune toggles (default off): `--phash-buckets`, `--phash-hamming` (if lib present); goal = fewer groups/pairs entering selection without changing winners.
3) Two-phase runs: dry-run plan writer + short commit pass to reduce I/O wall time.
4) Challenge subset calibration: validate that downgrade rate is higher than random subset at similar wall time.
5) Future idea: Hands/feet screening pass (non-destructive): detect presence, route these sets to fast review queue; helps estimate crop workload and reduce time in crop UI.

CLI notes:
- Reducer investigation shard (subset):
  - `python scripts/tools/reducer.py run --sandbox-root sandbox/mojo2_subset --dry-run --investigate --quiet --shards 8 --shard-index 0 --max-runtime 600`
- Quality-aware variant:
  - Add `--quality-aware` (tunable: `--qa-thumb-size`, `--qa-clip-threshold`)
- Subset builder:
  - Hash sample: `python scripts/tools/subset_builder.py --source sandbox/mojo2 --dest sandbox/mojo2_subset --fraction 0.25 --commit`
  - Challenge sample: `python scripts/tools/subset_builder.py --source sandbox/mojo2 --dest sandbox/mojo2_challenge --fraction 0.25 --challenge --commit`

Safety: All runs are sandbox-only and dry-run by default. New flags default off.
