# Current TODO List

**Last Updated:** October 15, 2025

---

## ✅ **COMPLETED TODAY (October 15, 2025)**

### **Dashboard Data Accuracy - Major Overhaul**
- ✅ Fixed inflated image counts (PNG-only counting, not companions)
- ✅ September data reduced from 20k-28k to accurate 10k-15k counts
- ✅ Backfilled September 2025 daily summaries with correct PNG counts
- ✅ Updated `cleanup_logs.py` to count PNGs only in daily summaries
- ✅ Updated `data_engine.py` and `analytics.py` for PNG-only aggregation

### **Project Visibility - All Archived Projects Now Visible**
- ✅ Fixed chart showing only 3 projects (now shows all 7 with data)
- ✅ Implemented date-based project matching for archived projects (daily summaries)
- ✅ Fixed word-boundary matching (dalia vs dalia_hannah false positive)
- ✅ Fixed mid-day project bug (date vs datetime comparison)
- ✅ All archived projects visible: tattersail, jmlimages, mixed-0919, dalia, 1101_Hailey

### **Table & Chart Consistency**
- ✅ Resolved table vs chart data mismatches (perfect consistency now)
- ✅ Added centralized `STANDARD_TOOL_ORDER` across all visualizations
- ✅ Standardized tool order everywhere: Desktop, Web Selector, Character Sorter, Multi Crop
- ✅ Added "Web Character Sorter" to table visibility by default
- ✅ Increased default lookback from 30 to 60 days

### **Dashboard UI Improvements**
- ✅ Added "Started" date column to Project Productivity Table
- ✅ Sorted table by newest projects first (was alphabetical)
- ✅ Added "Hide empty projects" checkbox to table and charts
- ✅ Added spacing between dashboard sections
- ✅ Trimmed leading blank days from timeseries charts

### **Work Time Calculation - Hour-Blocking**
- ✅ Replaced break detection algorithm with simple hour-blocking approach
- ✅ Hours now count unique hour blocks (YYYY-MM-DD HH) where files were moved
- ✅ No more fragile thresholds or subjective break detection
- ✅ Accurate, honest time tracking: matches timesheet expectations
- ✅ Updated `calculate_work_time_from_file_operations()` in `companion_file_utils.py`

### **Historical Projects Import**
- ✅ Created `import_historical_projects.py` to import projects from CSV timesheet
- ✅ Imported 18 historical projects (Aug-Oct 2025) with status "archived"
- ✅ All manifests created with proper timestamps, image counts, and status
- ✅ Handles multi-day projects, missing data, special cases automatically
- ✅ Dashboard now has full historical data

### **Project Lifecycle Automation**
- ✅ Created `00_start_project.py` for automated project initialization
- ✅ Created `00_finish_project.py` integrated with prezip_stager
- ✅ Both scripts support interactive and command-line modes
- ✅ Automatic timestamp generation (UTC with Z suffix)
- ✅ Automatic image counting from content directories

### **Legacy Support & Backward Compatibility**
- ✅ Added display name mappings for historical script names
- ✅ Fixed `image_version_selector`, `character_sorter`, `hybrid_grouper` recognition
- ✅ Ensured backward compatibility with old log formats
- ✅ Silenced warnings from deprecated `timer_data` files

### **Code Organization & Documentation**
- ✅ Moved test files to `scripts/dashboard/tests/`
- ✅ Moved dashboard docs to `Documents/` with DASHBOARD_ prefix
- ✅ Updated `.gitignore` to exclude `mojo2/` and `exports/` directories
- ✅ Removed utility scripts (create_sample_data, backfill_september)
- ✅ Created 12 comprehensive documentation files for all fixes
- ✅ Updated `AI_DOCUMENTS_INDEX.md` and `TECHNICAL_KNOWLEDGE_BASE.md`
- ✅ Git commit: 48 files changed, 9,409 insertions, 579 deletions

---

## 🔧 **CLEANUP & ORGANIZATION (Next Session)**

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

## ⚠️ **IMPORTANT WORKFLOW RULE**
**ONE TODO ITEM AT A TIME** - Complete one task, check in with Erik, get approval before moving to the next item. Never complete multiple TODO items in sequence without user input. This prevents issues and ensures quality control.

---

## 🔥 **HIGH PRIORITY**

### ✅ **Project Lifecycle & Throughput Tracking (COMPLETED Oct 15, 2025)**
- ✅ Created `00_start_project.py` and `00_finish_project.py` for automated project management
- ✅ Integrated `prezip_stager.py` into finish workflow
- ✅ Imported 16 historical projects from timesheet CSV
- ✅ Dashboard now shows accurate hours/days/images per project per tool
- ✅ Hour-blocking time calculation (simple, robust, honest)
- ✅ All project manifests standardized with ISO-8601 timestamps

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

### **Sandbox Automation Orchestrator (Tomorrow)**
- **Create sandbox copy**: copy remaining images from `mojo1/` into `sandbox/` (read-only source, safe target)
- **Run thinning steps**: pHash near-dup → FAISS semantic → timestamp-window clusters (reports + `_review_dupes/` staging only)
- **Grading metrics**: counts kept/flagged, estimated review saved, thresholds summary
- **Multi-source grouping**: extend similar-image grouper to accept `selected/` + `crop/` in one pass
- **Sidecar visual cues**: generate `.cropped` sidecars and add badge in Character Sorter
- **One-shot script**: single CLI to run all steps with dry-run/stage modes

### **Sidecar crop flag + UI badge + multi-source grouping**
- **Add sidecar flag**: create `.cropped` sidecar files next to PNGs in `crop/` (same-stem). No image changes; travels with file during moves.
- **Character sorter badge**: detect `.cropped` companions and render a small “Cropped” badge on thumbnails.
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

> Goal: Reduce Erik's manual review time by auto-picking winners in 2–4 near-duplicate sets and staging non-keepers safely for quick review. AI photography issues (blur, exposure) are irrelevant; we focus on AI anomalies (hands/feet, missing belly button, extra fingers, etc.).

### **A. Objectives & Success Criteria**
- **Primary objective**: Auto-select a best image per near-duplicate set with high precision; present only ambiguous sets for review.
- **Safety**: Non-destructive by default; move non-keepers to review folders with companions; never alter image bytes.
- **Companion integrity**: Always move image + `.yaml`/`.caption` and any same-stem companions together.
- **Measurable success**:
  - Manual time reduction ≥ 50% on duplicate thinning passes.
  - Auto-pick agreement with Erik’s choice ≥ 98% on calibrated sets.
  - False-positive staging (wrongly moved a keeper) ≤ 1 per 1,000 images.

### **B. What Tools Do What (Human vs Automation)**
- **Human-in-the-loop tools (existing/updated)**
  - `scripts/01_web_image_selector.py`: stays primary for version selection; may receive a “AI pick” hint later (optional).
  - `scripts/02_web_character_sorter.py`: add optional badges (e.g., “cropped” sidecar indicator already planned) and a future “AI suspect” tag.
  - New: `scripts/07_auto_pick_review.py` (planning only): web UI to review auto-pick reports, show chosen vs staged, approve/override by group.
  - `scripts/06_web_duplicate_finder.py`: remains useful for visual comparisons on tough clusters.
- **Automation scripts (offline, reversible)**
  - `scripts/tools/compute_phash.py`: compute pHash; write to `data/ai_data/hashes/`.
  - `scripts/tools/compute_embeddings.py`: OpenCLIP ViT-B/32; write to `data/ai_data/embeddings/`.
  - `scripts/tools/detect_hands_feet.py`: MediaPipe Hands (+ optional Pose feet presence); output per-image presence/anomaly hints to `data/ai_data/hands/`.
  - `scripts/utils/phash_group_near_dupes.py`: group pairs/sets by Hamming distance (≤ 8–10).
  - `scripts/utils/clip_group_near_dupes.py`: group pairs/sets by CLIP cosine (≥ 0.95; tunable).
  - `scripts/utils/auto_pick_near_dupes.py`: merge groupers, apply stage-aware ranking + anomaly gates, generate report; optional staging to `_review_dupes/`.
  - `scripts/orchestrators/automation_pipeline.py`: one CLI to run the pipeline in dry-run/stage modes, with metrics summary.

### **C. Selection Logic (Stage-aware + Anomaly-aware)**
- Candidate groups = filename stage runs + pHash near-exacts + CLIP semantic near-dupes.
- Rank within group:
  1) Highest stage number wins if it passes anomaly gate.
  2) If tie, prefer fewer detected hands/feet in-frame; if present, require anomaly checks to pass.
  3) Break ties by larger resolution, then latest mtime.
- Anomaly gate (AI-specific):
  - Hands/feet presence signal (fast): presence = higher risk; down-rank unless scores are strong.
  - Simple hand heuristics from keypoints: flag likely extra/fused fingers if geometry inconsistent (see doc below).
  - Optional “missing belly button” is future work; keep placeholder tag.

References: `Documents/hand_foot_anomaly_scripts.md` (approaches and two‑stage pipeline ideas).

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
4) Staging mode (non-keepers → `_review_dupes/`)
   - Extend `auto_pick_near_dupes.py` with `--stage` flag; default is dry-run.
   - Ensure companion handling via `move_file_with_all_companions`.
   - Metrics: counts kept/staged/ambiguous; false-positive sampling plan.
5) Review UI (human approval loop)
   - Plan `07_auto_pick_review.py` for tabular review of auto-pick decisions with thumbnails and approve/override.
   - Keep it optional; start with CSV/Markdown report + `06_web_duplicate_finder.py` for hard cases.
6) Orchestrator & metrics
   - `automation_pipeline.py` to chain steps and emit a summary (precision/recall estimates on calibration set, time saved).

### **F. Testing & Evaluation (before any wide run)**
- Golden fixtures for grouping/selection rules (small fixture dirs committed to `scripts/tests/fixtures/`).
- Dry-run as default; CI-style check that no file moves occur without `--stage`.
- Companion integrity tests (image + sidecars preserved through moves).
- Calibration evaluation: report agreement with Erik’s picks on the subset; adjust thresholds until ≥98%.

### **G. Risks & Safeguards**
- CLIP over-grouping across pose changes → mitigate with higher cosine threshold + stage-name parity.
- Hands detection false positives in stylized cases → treat presence as a review trigger, not an auto-reject.
- Always stage to review, never hard-delete; centralized logging via FileTracker.

### **H. Deliverables**
- Auto-pick report (CSV/Markdown) with reasons per decision.
- `_review_dupes/` staging structure mirroring sources (dry-run off).
- Optional review UI plan + wireframe before implementation.

### **I. References**
- `Documents/hand_foot_anomaly_scripts.md`
- `image_batch_culling_pipeline.md`, `similar_image_dedup_automation_plan.md`, `stage_aware_image_culling_workflow_v2.md`

> Note: This section is planning-only. No scripts will be created until we review and approve this plan.

### **J. Portrait Fast‑Track via MediaPipe Pose (Planning + Experiment)**
- **Hypothesis**: Images that are clearly head‑and‑shoulders portraits rarely need cropping; selecting the highest stage is typically sufficient.
- **Goal**: Automatically identify portrait (head+shoulders) images and route them into a fast lane, reducing intermixing with images that need crop decisions.
- **Detection approach (planning)**:
  - MediaPipe Pose + Face for head box + shoulder keypoints; simple heuristics: face area fraction, shoulder span band, hips mostly out of frame.
  - Fallback: face‑area‑only heuristic for speed; escalate ambiguous cases to full Pose.
- **Proposed pipeline (planning)**:
  1) Offline classify portrait vs non‑portrait (dry‑run → CSV of flags).
  2) Option A: Pre‑stage portraits into a `portraits_fastpass/` review dir.
  3) Option B: Tag portraits and surface a “Portrait fast pass” filter in the web selector.
  4) In fast pass, default selection rule = highest stage that passes anomaly gate; do not send to `crop/`.
- **Calibration plan (planning)**: Build a 100–200 image labeled subset to tune thresholds and confirm ≥98% agreement with Erik on portrait identification.
- **Safety**: Planning‑only; if approved, first build as dry‑run report with zero file moves.

### **K. Workflow Concepts — Pros/Cons and Throughput Modeling (Planning)**
- Compare workflows to maximize “quick wins” first without harming overall throughput:
  - Portrait fast‑pass first vs after near‑dupe thinning.
  - Stage‑aware auto‑pick before/after character sorting.
  - Routing by hands/feet presence early vs late.
- Define evaluation metrics: items/hour, percent auto‑finalized, review queue size, context‑switch cost.
- Run small A/B sessions and log timings to confirm which sequence actually yields faster end‑to‑end results (avoid “feels fast but is slower”).

## 🤖 **AI Training - Comprehensive Implementation Plan**

> 📖 **Master Reference:** `Documents/image_cropping_ranking_training_plan.md`  
> Full technical details, architecture, and Apple Silicon optimization strategies

### ✅ **Current Status: Basic Logging Complete**

**What We've Built:**
- ✅ Training logging built into 4 tools (no flags needed!)
- ✅ Web image selector → `data/training/selection_only_log.csv`
- ✅ Desktop image selector crop → `data/training/select_crop_log.csv`
- ✅ Multi-crop tool → `data/training/select_crop_log.csv`
- ✅ Web duplicate finder → `data/training/duplicate_detection_log.csv`

**Migration Path:**
- ✅ CSV logging provides foundation
- 🔄 Need to upgrade to JSONL schema with anomaly tags (see master doc)
- 🔄 Add `set_id`, `anomaly_tags`, detailed metadata
- 💡 Keep CSV logs running while we build JSONL alongside

---

## 📊 **CURRENT STATUS: Environment Ready, Week 1 Next**

**✅ Completed Today (Oct 5):** MPS + AI packages installed, all tests passing  
**🔜 Next Step:** Create `data/ai_data/` directory structure (Step 1)  
**⏸️ Paused:** Waiting to begin Week 1 implementation

---

## 🎯 **Training Philosophy: Apprentice Model**

The AI will learn like an apprentice who:
1. **Watches you work** (weeks/months) - learns patterns, no interference
2. **Makes suggestions** (when ready) - you accept/reject/modify to refine it  
3. **Works semi-autonomously** (distant future) - with your review

**Critical:** No automation until AI proves it understands your preferences through extensive observation.

---

## 📊 **PHASE 1: OBSERVATION MODE** (Current Focus - Weeks 1-3+)

**Goal:** Silent logging system that watches your existing workflow without changing anything.

### **Week 1: Foundation - Passive Data Collection**

**1.1 Set Up AI Data Structure** ✓ Safe, no breaking changes
```
data/ai_data/
  ├── embeddings/           # Image embeddings (computed offline)
  ├── hashes/              # Perceptual hashes
  ├── saliency/            # Saliency maps
  ├── hands/               # Hand detection data
  ├── observations/        # Your actual decisions (JSONL logs)
  │   └── sessions/
  └── models/              # Future trained models
```

**1.2 Create Offline Analysis Tools** ✓ Can run anytime, won't break workflow
- `tools/compute_embeddings.py` - Batch process images in `mojo/`, `crop/`, `selected/`
  - Model: OpenCLIP ViT-B/32 (best balance speed/accuracy for M4 Pro)
  - Output: `data/ai_data/embeddings/`
- `tools/compute_phash.py` - Generate perceptual hashes for duplicate detection
  - Output: `data/ai_data/hashes/`
- `tools/compute_hands_saliency.py` - Extract hand keypoints + saliency maps
  - MediaPipe Hands for keypoint detection
  - U²-Net via rembg for saliency (simplest implementation)
  - Output: `data/ai_data/hands/`, `data/ai_data/saliency/`
- **All run independently, no integration needed yet**

**1.3 Design Observation Schema** ✓ Planning only
```jsonl
{
  "timestamp": "2025-10-05T19:12:31Z",
  "session_id": "mojo_batch_001",
  "tool": "web_image_selector",
  "image_group": ["img1.png", "img2.png", "img3.png"],
  "action": "selected",
  "chosen": "img2.png",
  "rejected": ["img1.png", "img3.png"],
  "crop_box": null,  # or {x, y, w, h} if cropped
  "deleted": false,
  "notes": "optional metadata"
}
```

---

### **Week 2: Passive Integration - Watch Without Interfering**

**2.1 Add Silent Logging to Existing Tools** ⚠️ Needs careful integration
- Modify `01_web_image_selector.py` - log selections
- Modify `04_multi_crop_tool.py` - log crop decisions
- Modify `01_desktop_image_selector_crop.py` - log select+crop
- **Critical Requirements:**
  - ✅ Completely silent (no UI changes)
  - ✅ Optional (can disable with env var)
  - ✅ Zero performance impact
  - ✅ Fail-safe (errors don't break tools)
- **Feature Flag:** `export ENABLE_AI_LOGGING=true` to activate
- **Integration Strategy:** Add to ONE tool first, test thoroughly, then expand

**2.2 Build Observation Dashboard** ✓ Separate tool, safe
- Web viewer to see what AI has learned
- Show statistics: "Observed 1,247 decisions across 89 sessions"
- Visualize patterns without making recommendations
- Check data quality
- Location: `scripts/dashboard/ai_observation_viewer.py`

---

### **Week 3: Data Accumulation & Quality Check**

**3.1 Run Batch Feature Extraction**
- Process all images in `mojo/`, `crop/`, `selected/`
- Generate embeddings, hashes, saliency maps
- Build feature database
- Verify outputs for quality

**3.2 Validate Observation Data**
- Check logs are capturing decisions correctly
- Verify file paths resolve
- Ensure data quality for training
- Build validation script: `tools/validate_observations.py`

**3.3 Initial Pattern Analysis** (Optional)
- Simple statistics: "You prefer images with X"
- No recommendations, just showing what it sees
- Sanity check the AI is learning something meaningful
- Output: Human-readable report of patterns detected

---

## 🛠️ **Implementation Strategy - No Breaking Changes**

### **What Can We Do RIGHT NOW (Zero Risk)**

1. ✅ **Create `data/ai_data/` structure** - new directory, can't break anything
2. ✅ **Build standalone tools** (`compute_embeddings.py`, etc.) - run independently
3. ✅ **Write observation schema docs** - planning only
4. ✅ **MPS benchmark test** - verify M4 Pro performance with PyTorch

### **What Needs Careful Integration**

1. ⚠️ **Adding logging to existing tools** - need to test thoroughly
   - Add behind feature flag: `ENABLE_AI_LOGGING=true`
   - Extensive error handling
   - Test on small batch first
   - Integrate ONE tool at a time

### **What We Can Build Incrementally**

1. 🔄 **Observation dashboard** - separate web tool
2. 🔄 **Analysis scripts** - run offline on collected data
3. 🔄 **Model training** - when enough data accumulated

---

## 🗓️ **Implementation Roadmap (Week-by-Week)**

### **✅ COMPLETED: Environment Setup (Oct 5, 2025)**

**What We Built:**
- ✅ MPS benchmark test - verified M4 Pro + PyTorch working (1.76x speedup!)
- ✅ Installed AI packages into `.venv311`:
  - OpenCLIP 3.2.0 (CLIP embeddings)
  - imagehash 4.3.2 (perceptual hashing)
  - MediaPipe 0.10.21 (hand detection)
  - rembg 2.0.67 (saliency detection)
  - pandas 2.3.3 (data handling)
- ✅ Verified no conflicts with existing tools (163 tests passed)
- ✅ Web image selector and all utilities working perfectly

**NumPy Note:** Downgraded from 2.2.6 → 1.26.4 for MediaPipe compatibility. OpenCV and all existing scripts work fine with this version.

---

### **🔜 NEXT: Week 1 - Foundation (Passive Data Collection)**

**Priority Order (each item depends on previous):**

**1️⃣ Create AI Data Structure** ⚠️ **DO THIS FIRST**
- Create `/absolute/path/to/project-root/data/ai_data/` directory
- Subdirectories: `embeddings/`, `hashes/`, `saliency/`, `hands/`, `observations/sessions/`, `models/`
- Safe operation: just creates empty directories, can't break anything
- **Status:** Ready to implement

**2️⃣ Build Standalone Analysis Tools** ⚠️ **DO SECOND**
- Create `scripts/tools/compute_embeddings.py` (OpenCLIP ViT-B/32, use MPS)
- Create `scripts/tools/compute_phash.py` (perceptual hashing)
- Create `scripts/tools/compute_hands_saliency.py` (MediaPipe + U²-Net)
- All run independently on `mojo/`, `crop/`, `selected/` directories
- No integration with existing tools yet - completely safe
- **Status:** Ready to implement
- **Dependencies:** Need step 1 completed first (directories must exist)

**3️⃣ Test on Small Sample** ⚠️ **DO THIRD**
- Run tools on small sample (e.g., 10-20 images from character_group_2/)
- Verify outputs are correct
- Check file sizes and formats
- **Status:** Ready after steps 1-2
- **Dependencies:** Steps 1-2 must work first

**4️⃣ Design & Document Observation Schema**
- Finalize JSONL schema for logging
- Document anomaly tags
- Create example session files
- **Status:** Ready to implement (documentation only)
- **Dependencies:** None (can do anytime)

---

### **🔮 FUTURE: Week 2 - Passive Integration (Watch Mode)**

**NOT STARTED - Don't begin until Week 1 complete**

**5️⃣ Add Silent Logging to ONE Tool**
- Integrate observation logging into `01_web_image_selector.py` first
- Always-on by default (learns from everything you do)
- Emergency kill switch: `export DISABLE_AI_LOGGING=true`
- Extensive testing on small batch
- **Status:** Not started
- **Dependencies:** Must complete Week 1 first + verify schema works

**6️⃣ Build Observation Dashboard**
- Create `scripts/dashboard/ai_observation_viewer.py`
- Show statistics: "Observed X decisions across Y sessions"
- Visualize patterns without making recommendations
- **Status:** Not started
- **Dependencies:** Need logging data from step 5

**7️⃣ Expand to Other Tools**
- Add logging to `04_multi_crop_tool.py`
- Add logging to `01_desktop_image_selector_crop.py`
- Only if step 5 works perfectly
- **Status:** Not started
- **Dependencies:** Step 5 must be rock-solid first

---

### **🔮 FUTURE: Week 3 - Data Accumulation & Validation**

**NOT STARTED - Don't begin until Week 2 complete**

**8️⃣ Run Batch Feature Extraction**
- Process all existing images in mojo/, crop/, selected/
- Generate complete feature database
- **Status:** Not started

**9️⃣ Validate Observation Data**
- Build `scripts/tools/validate_observations.py`
- Check data quality, file paths, completeness
- **Status:** Not started

**🔟 Initial Pattern Analysis**
- Simple statistics showing what AI learned
- No recommendations, just pattern visualization
- **Status:** Not started

---

## 📋 **Model & Technology Choices**

**Documented for future reference:**

- **OpenCLIP ViT-B/32**: Best balance of speed/accuracy for M4 Pro (512D embeddings, fast inference)
- **U²-Net via rembg**: Simplest saliency implementation (Apache 2.0 license)
- **MediaPipe Hands**: Standard hand keypoint detection (Apache 2.0 license)
- **pHash**: Perceptual hashing for near-duplicate detection (simple, fast)
- **Skip YOLO for now**: Add in Phase 2 if needed for face/foot detection

---

## 📊 **PHASE 2: RECOMMENDATION MODE** (Future - Weeks 4+)

**Won't start until Phase 1 data collection is satisfactory**

### **Week 4+: Optional Recommendation Layer**

**4.1 Build Ranking Model v0**
- Create: `train/rank_bt.py` (Bradley-Terry or tiny MLP)
- Features: CLIP embedding + sharpness + subject size + anomaly counts
- Train on accumulated observation data
- Use MPS for Apple Silicon acceleration

**4.2 Build Crop Proposer v0**
- Create: `tools/propose_crop.py` (fixed-aspect, objective-based)
- Constraints: Keep original aspect ratio, head preference, ≥1/3 body
- Score: `α·SaliencyIn – β·AnomalyOverlap – γ·HeadCut – δ·JointCut`
- Output: Proposed crops with confidence scores

**4.3 Add Toggle-able Recommendation UI**
- Add optional recommendation layer to tools (can toggle on/off)
- "AI suggests: img2.png (confidence: 87%)" - you agree/disagree
- Log your corrections to improve model
- **Critical:** Must be completely optional and easy to disable

**4.4 Correction Logging**
- Log when you accept AI suggestions
- Log when you reject and choose differently
- Log when you modify AI crop proposals
- Use corrections to refine models incrementally

---

## 📊 **PHASE 3: SEMI-AUTOMATION** (Distant Future)

**Won't start until Phase 2 shows sustained high accuracy**

### **Requirements for Phase 3 Activation:**
- **Metric goals:** Top-1 agreement ≥99%, Crop acceptance ≥99%, Residual anomalies ≈0
- **Promotion criteria:** Sustained for N consecutive sets (e.g., 300)
- **Manual override:** Easy kill switch to return to Phase 2 or Phase 1

### **Batch Processing Features:**
- Batch processing with review queue
- High confidence = auto-process
- Low confidence = ask you first
- **Safety:** AI-assist flags residual anomalies for your review

### **Active Learning & Routing**
- Route sets with low confidence margin first
- Prioritize: Small ranking margin, crop near threshold, detector disagreement
- Optional: Integrate Cleanlab/ActiveLab for label QA

### **Optional Enhancements**
- Add tiny YOLO face/foot detection
- Improve "don't cut here" penalties (face/foot boxes)
- Refine crop proposals based on accumulated data

---

### 🔧 **Supporting Tasks (Ongoing)**

**A. Data Monitoring & Validation**
- Create verification script: Check logs being written
- Alert if logs stale (>X days without updates)
- Validate CSV/JSONL structure and data quality
- Daily/weekly automated checks

**B. Documentation**
- Document logging schemas in Technical Knowledge Base
- Create training data inventory
- Track model versions and approval rates

**C. Environment Setup**
- Python ≥3.11, PyTorch with MPS (Metal)
- Install: `torch`, `lightning`, `accelerate`, `open_clip_torch`, `opencv-python`, `imagehash`, `mediapipe`, `u2net`
- Verify Apple Silicon acceleration working

---

## 🔐 **Data Backup Plan**

**Goal:** Automated, reliable off-repo backups of CSV/log data.

**Scope:**
- `data/training/*.csv`
- `data/file_operations_logs/*.log`
- Optional: manifests with hashes/counts

**Tasks:**
1. Choose backend (S3, Backblaze B2, or Google Drive)
2. Create `scripts/backup/backup_training_data.py`
   - Package files into timestamped tar.gz
   - Write manifest with sha256 + row counts
   - Upload with lifecycle policy (keep 8 weekly)
3. Create `scripts/backup/restore_training_data.py`
4. Schedule weekly cron (Sun 02:00 local)
5. End-to-end test: backup → delete → restore → verify
6. Document in Knowledge Base

**Defaults:** Weekly backups, 8 weeks retention, server-side encryption

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
  - Example:
    ```
    def safe_delete(self, png_path: Path, yaml_path: Path):
        send2trash(str(png_path))
        if yaml_path.exists():
            send2trash(str(yaml_path))
    ```
  - Impact: Tools calling base `safe_delete` won’t remove `.caption` sidecars.

- `scripts/04_multi_crop_tool.py` — Calls base `safe_delete(png_path, yaml_path)` during delete flow.
  - Example: `self.safe_delete(png_path, yaml_path)`
  - Note: Fixing base method fixes this tool automatically.

- `scripts/utils/triplet_deduplicator.py` — Dry-run and removal reference only `.yaml`.
  - Example dry-run:
    ```
    yaml_file = png_file.parent / f"{png_file.stem}.yaml"
    if yaml_file.exists():
        print(f"    🗑️  {yaml_file.name}")
    ```
  - Expected: Use wildcard companions (YAML and/or caption) for print and removal.

- `scripts/archive/04_batch_crop_tool.py` — Legacy; uses `safe_delete(png_path, yaml_path)`.
  - Note: Mark as legacy; optional to update.

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

### 4. Code Conventions & Patterns Catalog
**Create:** `Documents/CONVENTIONS_REFERENCE.md`
- Analyze all scripts for reusable patterns
- Document Flask structure, CSS, JavaScript patterns
- Document matplotlib setup, event handling
- Create ready-to-use code templates
- Benefits: Consistency, maintainability, easier onboarding

### 5. Scripts layout cleanup (planning)
- Clarify conventions: `scripts/tools/` = runnable CLIs/automation; `scripts/utils/` = reusable libraries only.
- Audit `scripts/tools/` for code that belongs in `scripts/utils/` and propose moves.
- Refactor imports after moves; replace `project_root` path hacks with `sys.path.insert` only where necessary.
- Add `scripts/README.md` summarizing directory purposes and import rules.
- Add tests covering `utils/recursive_file_mover.py` CLI behaviors.

### 5. Create Local Homepage
Build custom homepage in Documents with links to all AI systems and tools

### 6. Web Interface Template System Investigation
Evaluate if template would simplify web tool maintenance vs add complexity

### 7. Experiment: Hand/Foot Anomaly Scripts
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
  - Selected card shows white “crop-selected” outline when `state.crop=true` (existing class).

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
- **Goal**: Add a lightweight, on‑page work timer similar to the multi‑crop tool’s timer to encourage timeboxed passes and reduce context switching.

- **UI**:
  - Fixed, subtle header widget (top bar, right side): “Work: 00:00 • Session: 00:00 • Efficiency hint (optional)”.
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
  - Feature is display‑only; won’t affect selection/crop submissions.

## 📊 **Dashboard Enhancements (In Progress + Planned)**

### In Progress
- Day-banding alignment on intraday (15min/1H)
  - Status: In progress; alternating bands render, local-day boundary alignment needs final tweak for 1H.
  - Considerations: Ticks vs dataset label spacing; ensure midnight boundaries map to tick indices reliably.
- Project start/end markers
  - Status: In progress; dashed blue/red lines render; labels and tooltips pending.
  - Considerations: Label overlap, accessibility contrast, and hover tooltips with ISO time.

### Completed (recent - Oct 15, 2025)
- ✅ Project Productivity Table with tool filtering
  - Tool column order: Desktop Image Selector Crop → Web Image Selector → Web Character Sorter → Multi Crop Tool
  - Checkboxes to show/hide columns (Desktop Image Selector Crop unchecked by default)
  - Per-tool breakdown: Hours, Days, Images processed, Selected/Cropped (for Image Selector tools)
  - Rounded hours (standard rounding)
  - Start/End image counts from project manifests
  - Only 4 approved tools displayed (filtered backend + frontend)
- ✅ Project selector in header (persisted); backend manifest load + project_id filtering
- ✅ Operation/tool toggles with persistence; average overlays per visible series
- ✅ Single-column layout; readable local timestamps; hourly crop KPI; header lookback/timeframe/refresh

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
- **Bars:** Stacked segments for each tool (Desktop Image Selector Crop, Web Image Selector, Web Character Sorter, Multi Crop Tool)
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
- **X-axis:** Tools (Desktop Image Selector Crop, Web Image Selector, etc.)
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
- All charts use same 4-tool filter (Desktop Image Selector Crop, Web Image Selector, Web Character Sorter, Multi Crop Tool)
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

---

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

---

## 🗓️ Tomorrow: Automation Reduction Experiments — Runtime Guards and Plan (Oct 12, 2025)

### Where we left off (Oct 11, 2025)
- Snapshot utility implemented: `scripts/tools/snapshot.py` (save/restore baseline under `sandbox/mojo2`).
- Reducer outputs constrained under sandbox: `selected/`, `delete/`, `crop/` now live in `sandbox/mojo2/`.
- Watchdog self-test passed (ABORT on simulated hang writes logs under `sandbox/mojo2/logs/`).
- Quiet logging added: set `COMPANION_UTILS_QUIET=1` or `--quiet` to suppress per-file DRY-RUN spam.
- Experiment 1 (B-conservative + dedupe-first 0.90) dry-run attempted on full sandbox; aborted by watchdog due to max runtime (heavy stdout and ~18k images). No crashes.

### Decision needed (morning)
- Choose default runtime caps:
  - Dry-run sanity: 20 min (1200s) default? [decide 20 vs 30]
  - Commit runs: 30 min (1800s) default? [confirm]
- Per-run group cap policy: compute from early estimate vs fixed `--limit` (e.g., 1–2k groups)?

### Tasks for tomorrow (runtime guards + ergonomics)
1) Add early-size estimator to reducer run:
   - Fast scan counts (files, groups estimate) before selection loop.
   - Estimate runtime (items_per_sec from rolling window) after first 200 groups; abort early if projected > cap.
   - Print summary and write an estimator line to `sandbox/mojo2/metrics/`.
2) Make `--quiet` default for `--dry-run`; keep opt-out `--no-quiet`.
3) Adaptive `--limit` default:
   - If not provided, set from estimator so projected runtime <= cap.
4) Progress-based soft abort:
   - If projected remaining time > remaining budget, finish current group and end gracefully (write metrics/report).
5) Truncated logging for DRY-RUN:
   - Print only periodic progress lines and top-N sample move intents.
6) Metrics/report completeness per experiment:
   - Input count, groups, winners, reduction %, runtime, errors list, sample audit placeholders.
   - Write `sandbox/mojo2/metrics/exp_<id>.jsonl` and `sandbox/mojo2/reports/exp_<id>.md`.
7) Orchestrator helper:
   - Small shell/python wrapper to iterate Experiments 1→10 with restore between runs and unique sub-ids for sweeps.

### Morning run plan
1) Restore baseline:
   - `python3 scripts/tools/snapshot.py restore --root sandbox/mojo2 --in sandbox/mojo2/.baseline.tar`
2) Experiment 1 — B-conservative + dedupe-first (0.90):
   - Dry-run sanity (quiet, runtime cap 20m, estimator on):
     - `python3 scripts/tools/reducer.py run --variant B --profile conservative --dedupe 0.90 --sandbox-root sandbox/mojo2 --dry-run --quiet --max-runtime 1200 --watchdog-threshold 120 --progress-interval 2`
   - Commit run (quiet, runtime cap 30m; ensure moves remain sandbox-local):
     - `python3 scripts/tools/reducer.py run --variant B --profile conservative --dedupe 0.90 --sandbox-root sandbox/mojo2 --commit --quiet --max-runtime 1800 --watchdog-threshold 120 --progress-interval 2`
   - Capture metrics/report; on any error, continue and log.
3) Restore baseline for next experiment before proceeding.

### Notes
- Aim for early fail-fast: prefer smaller, faster runs with clear metrics over 1h runs.
- Keep companions intact on all moves; never alter image bytes.
- All artifacts strictly under `sandbox/mojo2`.

---

## Automation Reduction Experiments — Progress (Oct 12)

- Instrumentation added: `scripts/tools/prof.py`, `--investigate` in `reducer.py` with stage timers and checkpoints.
- Deterministic sharding: `--shards/--shard-index` to keep runs bounded and reproducible.
- Deterministic subset builder: `scripts/tools/subset_builder.py` (hash-based), 25% subset created at `sandbox/mojo2_subset` (867 groups, 2599 images).
- Challenge subset mode: new `--challenge` flag (risk-scored sampling by sharpness/exposure); created at `sandbox/mojo2_challenge` (~884 groups, 2652 images at 25%).
- Quality-aware selector (toggle off by default): `--quality-aware` computes fast thumbnail sharpness/exposure and falls back to a lower stage if the top stage is blur/clipped; dry-run only so far.
- Async moves experiment: added `--async-moves` (off by default); discarded for instability/no speedup in short benches.

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
