# Current TODO List - September 24, 2025

## 📅 Update — October 5, 2025

### ✅ Completed today
- Desktop selector: 4-up single row layout with centered, tight gutters
- Copyable timestamps + hotkeys (1–4) and per-image skip (A/S/D/F), B=Skip All
- Enter behavior: no selection deletes all; selection crops one, deletes others
- Selector stability: no "ghost" handles; safe extent updates; transition guard for Skip
- Window sizing: fit-to-screen without overflow; Qt window resize
- Centralized sorter enforced in human-facing tools (desktop, web selector, multi-crop, character sorter)
- Web selector: 4th button + key; cap groups to 4 per row; pre-sort before grouping
- Shared delete utility: `safe_delete_image_and_yaml` used by web/desktop
- Tests: desktop behavior tests; sorting determinism test
- Knowledge base: nearest‑up spec, timestamps-only-for-sorting, centralized sorting rule

### 🔜 Next
- Sweep remaining utilities/viewers for centralized sorter (confirm no stragglers)
- Add a short section in KB for new desktop hotkeys (p [ ] \\ and A/S/D/F/B)
- Optional: regression runner to exercise desktop 4-up flows + copy/skip

## 🧠 AI Select+Crop Training Plan (authoritative source)

### Phase 1 — Logging (no workflow disruption)
1. Desktop: add `--log-training` flag; on Enter write one CSV row to `data/training/select_crop_log.csv`
   - Columns: session_id, set_id, directory, image_count, chosen_index, chosen_path,
     crop_x1,y1,x2,y2 (normalized), image_i_path/stage, width_i/height_i, timestamp
   - Fail-open (try/except); create directory if missing; append mode
2. Web selector: on batch finalize write selection-only rows to `data/training/selection_only_log.csv`
   - Columns: session_id, set_id, chosen_path, neg_paths(list/JSON), timestamp
3. KB: add short section documenting flags, file locations, schemas

### Phase 2 — Dataset builder
1. Script: `scripts/datasets/build_select_crop_dataset.py`
2. Inputs: the two CSVs; validate file existence
3. Split by set_id into train/val/test (80/10/10)
4. Outputs:
   - Ranking pairs (chosen, negative) JSONL
   - Crop samples (image, bbox) JSONL
   - (Optional) COCO JSON for bbox visualization

### Phase 3 — Training (two-head)
1. Script: `scripts/train/train_select_crop.py` (PyTorch+timm)
2. Backbone ViT-B/16; heads: ranking (embedding+MarginRankingLoss) + bbox (SmoothL1+IoU)
3. Loss: total = 1.0*L_rank + 2.0*L_bbox (tunable)
4. Optimizer: AdamW (backbone 3e-5..5e-5; heads 1e-4); cosine decay; 10–30 epochs; amp

### Phase 4 — Evaluation
1. Script: `scripts/eval/eval_select_crop.py` (Top-1, IoU@0.5, mean IoU, MAE)
2. Notebook: `notebooks/eval_select_crop.ipynb` with qualitative side-by-sides

### Phase 5 — Inference integration (desktop)
1. Flag `--ai-suggest` to enable
2. On batch: score images → suggest top-1; predict crop for top-1; draw suggested rect
3. Hotkeys: T toggle suggestions; Y accept suggested selection+crop (user can tweak)
4. No auto-submit; always user-controlled

### Environment & repos
- pip: torch torchvision timm albumentations scikit-learn pycocotools tqdm pyyaml rich
- Sorting rule: ALWAYS pre-sort with `sort_image_files_by_timestamp_and_stage`
- Normalize crop coords to [0..1]; split by set_id to avoid leakage

## 🔐 Data Backup Plan (new project)

Goal: Automated, reliable off-repo backups of important CSV/log data.

Scope (include):
- `data/training/*.csv` (select_crop_log.csv, selection_only_log.csv)
- `data/file_operations_logs/*.log`
- Optional: manifests (hashes/counts) for verification

Out of scope (exclude):
- All image assets and large binaries (already backed up elsewhere)

Tasks:
1. Choose backend (S3 or Backblaze B2; fallback: Google Drive)
2. Create backup script `scripts/backup/backup_training_data.py`
   - Packages files into a timestamped tar.gz (UTC)
   - Writes a manifest with sha256 + row counts
   - Uploads to backend with lifecycle policy (e.g., keep 8 weekly)
3. Add restore script `scripts/backup/restore_training_data.py` (id or latest)
4. Schedule weekly cron (Sun 02:00 local)
5. Run an end‑to‑end test: backup → delete temp copy → restore → verify manifest
6. Document in KB (runbook + env vars)

Defaults:
- Frequency: Weekly (daily is unnecessary for current volume)
- Retention: 8 weeks
- Encryption: Server‑side (S3 SSE) or key‑based client encryption (optional)


## ⚠️ **IMPORTANT WORKFLOW RULE**
**ONE TODO ITEM AT A TIME** - Complete one task, check in with Erik, get approval before moving to the next item. Never complete multiple TODO items in sequence without user input. This prevents issues and ensures quality control.

**Status:** 🎉 MASSIVE PRODUCTIVITY DASHBOARD SYSTEM COMPLETED! ✅

## 🚀 INCREDIBLE SESSION COMPLETED (September 24, 2025)

### ✅ **COMPLETE PRODUCTIVITY DASHBOARD SYSTEM BUILT**
- ✅ **Full Flask web dashboard** with Chart.js visualization of 100,099+ files processed
- ✅ **Activity timer integration** in 01_web_image_selector, 03_web_character_sorter, 04_batch_crop_tool
- ✅ **Live timer widgets** with idle detection, batch tracking, and operation logging
- ✅ **Data aggregation engine** processing ActivityTimer + FileTracker logs into beautiful charts
- ✅ **Smart data retention policy** preventing storage explosion (30 days detailed, 1 year summaries)
- ✅ **Auto-browser opening** and dark theme matching Erik's style guide
- ✅ **Real productivity insights** from actual workflow data (100k+ files visualized!)

### ✅ **INFRASTRUCTURE & DATA PROTECTION**
- ✅ **Complete dashboard directory** organized in `scripts/dashboard/` with all components
- ✅ **Data protection** via .gitignore (timer_data/ and file_operations_logs/ properly ignored)
- ✅ **Comprehensive documentation** with detailed journal entry
- ✅ **Production-ready system** with proper error handling and user experience

### ✅ **TECHNICAL ACHIEVEMENTS**
- ✅ **12 new dashboard files** created with 2,517+ lines of new code
- ✅ **Cross-script data aggregation** combining timer and file operation data
- ✅ **Interactive time controls** (15min/1hr/daily/weekly/monthly views)
- ✅ **Real-time stats** showing total files, active days, avg daily processing
- ✅ **Beautiful bar charts** for files by script and operations by type

## 🔄 Next Session Priorities

### **High Priority Tasks**
1. **AI-powered automatic cropping system** - Train AI to learn from manual cropping patterns
   - **Goal**: Automatically detect optimal crop areas to reduce manual cropping time
   - **Strategy**: Start collecting training data from manual crops, build ML model
   - **Impact**: Massive productivity gain - AI learns your cropping preferences over time
   - **Timeline**: Start ASAP - the sooner we begin, the sooner it learns and improves

### **Dashboard Enhancements (Low Priority)**
2. **Historical average overlays** - "Cloud" background showing historical trends
3. **Script update correlation** - Timeline markers showing when tools were updated
4. **Pie chart time distribution** - Visual breakdown of time spent per tool
5. **Data export capabilities** - CSV/JSON export for deeper analysis
6. **Comprehensive testing suite** - Complete test coverage for dashboard components

### **Advanced Analytics (Future)**
7. **GitHub integration for script change tracking** - Pull commit history and correlate with productivity
   - **Goal**: Show detailed change logs correlated with productivity improvements
   - **Method**: Integrate with GitHub API to pull commit messages and dates
   - **Benefit**: See exactly what changes led to productivity gains
   - **Timeline**: Nice-to-have for later - not priority now
   - **Example**: "Added base class refactoring" → productivity spike visible on dashboard

8. **Update markers system** - Hover descriptions correlating productivity with tool changes
9. **Performance trend analysis** - Long-term efficiency tracking and insights

## 📋 Current System Status

### **🚀 PRODUCTION-READY DASHBOARD**
The productivity dashboard is **100% functional** and ready for daily use:
```bash
source .venv311/bin/activate
python scripts/dashboard/run_dashboard.py
```

### **📊 What You Get Right Now**
- **Real data visualization** of your 100,099+ files processed
- **Interactive charts** showing daily/weekly productivity patterns
- **Live activity timers** in all 3 production workflow tools
- **Beautiful dark theme** matching your existing tools
- **Auto-browser opening** for seamless workflow integration

### **🎯 Current Workflow Tools (All Enhanced)**
- **01_web_image_selector.py** - Activity timer + live stats widget ✅
- **03_web_character_sorter.py** - Activity timer + operation tracking ✅  
- **04_batch_crop_tool.py** - Activity timer + batch completion metrics ✅
- **Dashboard system** - Complete analytics platform ✅

### **📁 Key Files Created**
- `scripts/dashboard/run_dashboard.py` - Launch the dashboard
- `scripts/dashboard/productivity_dashboard.py` - Main Flask application
- `scripts/dashboard/data_engine.py` - Data processing engine
- `scripts/dashboard/dashboard_template.html` - Beautiful web interface
- `scripts/dashboard/data_retention_policy.py` - Smart storage management
- `scripts/utils/activity_timer.py` - Shared timer utility

### **🔧 Technical Setup**
- **Virtual environment:** `.venv311` (Flask already installed)
- **Data protection:** `scripts/timer_data/` and `scripts/file_operations_logs/` ignored by git
- **Dashboard port:** 5001 (auto-opens browser)
- **All changes committed** with comprehensive documentation

## 🎯 Next Session Options

The dashboard system is **complete and production-ready**! Future sessions could focus on:
1. **Using the dashboard** in daily workflow and gathering insights
2. **Optional enhancements** from the pending list above
3. **New workflow tools** or **different productivity projects**

## ✅ **COMPLETED SESSION WORK (October 2, 2025)**

### **🔄 Desktop Tool Refactoring & Standardization**
- ✅ **Git Tracking Issue** - Fixed "too many active changes" by adding `mojo1/` to `.gitignore`
- ✅ **Remove Redundant Hotkeys** - Removed duplicate [1,2,3] hotkeys from main title bar
- ✅ **Fix Top Text Updates** - Fixed main title bar text updates with `draw_idle()`
- ✅ **Fix Single Keep Logic** - Ensured only ONE image can be kept at a time
- ✅ **Analyze desktop tools** - Analyzed DesktopImageSelector crop and BatchCropTool multi
- ✅ **Inventory common functions** - Identified shared functionality between tools
- ✅ **Identify differences** - Cataloged unique functionality in each tool
- ✅ **Create standardization plan** - Planned shared components and templates
- ✅ **Create base class** - Built `BaseDesktopImageTool` with shared functionality
- ✅ **Refactor desktop selector** - Updated to inherit from base class
- ✅ **Refactor batch crop tool** - Updated to inherit from base class
- ✅ **Update tests** - Ensured tests work with refactored code
- ✅ **Rename multi crop tool** - Renamed batch crop tool to multi crop tool
- ✅ **Add image names** - Added filenames to bottom labels in both tools
- ✅ **Fix crop tool selection** - Crop tool now auto-selects images
- ✅ **Optimize performance** - Used `draw_idle()` for faster UI updates
- ✅ **Cleanup backup files** - Removed all temporary and backup files

### **📊 Code Coverage System Setup**
- ✅ **Setup code coverage** - Installed coverage.py, configured, created test runner
- ✅ **Moved htmlcov** - Relocated coverage reports to `scripts/tests/htmlcov/`
- ✅ **Created coverage runner** - `scripts/tests/run_coverage.py` for easy coverage analysis

### **🚀 FILE-OPERATION-BASED TIMING SYSTEM (MAJOR ACCOMPLISHMENT!)**
- ✅ **Create intelligent work time calculation** - Built `calculate_work_time_from_file_operations()` with 5-minute break detection
- ✅ **Create comprehensive metrics function** - Built `get_file_operation_metrics()` for dashboard integration
- ✅ **Remove ActivityTimer from file-heavy tools** - Updated all 4 file-heavy tools to use file-operation timing:
  - ✅ `01_web_image_selector.py` - Removed timer, now uses FileTracker logs
  - ✅ `01_desktop_image_selector_crop.py` - Removed timer via base class
  - ✅ `02_web_character_sorter.py` - Removed timer, now uses FileTracker logs  
  - ✅ `04_multi_crop_tool.py` - Removed timer via base class
- ✅ **Keep ActivityTimer on scroll-heavy tools** - Preserved timer on browsing tools:
  - ✅ `05_web_multi_directory_viewer.py` - No changes needed
  - ✅ `06_web_duplicate_finder.py` - No changes needed
- ✅ **Update dashboard data engine** - Added intelligent timing method selection
- ✅ **Test new timing system** - Verified break detection and work time calculation
- ✅ **Update dashboard UI** - Added work time stat card and timing method indicator

**🎯 RESULT: More accurate productivity metrics with intelligent break detection!**

## 🚀 **CURRENT PRIORITIES**

### **🐛 Bug Fixes (High Priority)**
1. **Fix desktop image selector crop tool selection toggle** - When one image is selected by hotkey and then another image's crop position is modified, the original image doesn't toggle back to delete
   - **Issue**: Only one image should be kept at a time, but selection doesn't auto-toggle
   - **Impact**: UX issue during long image processing sessions
   - **Priority**: High - affects daily workflow

### **🔧 Technical Improvements**
2. ~~**Implement canonical stage ladder approach**~~ - **CANCELLED** - Would break existing workflow
   - **Decision**: Current system works well for 7,000+ image processing workflow
   - **Current behavior**: Groups `1→2→3` (skips 1.5) - this is what we want
   - **Canonical approach**: Would NOT group `1→2→3` (requires `1→1.5→2→3`) - too strict
   - **Conclusion**: Current "any increasing stages" approach is better for this workflow

2. **Improve test coverage and quality** - Current tests are good but could be more comprehensive
   - **Current test strengths**: Tests all valid combinations, edge cases, comprehensive coverage
   - **Test improvements needed**:
     - **Add unsorted input tests** - Verify pre-sorting requirement is enforced
     - **Add unknown stage handling** - Test behavior with unexpected stage numbers
     - **Add min_group_size variations** - Test different grouping thresholds
     - **Add two-runs-in-sequence tests** - Test multiple separate groups
     - **Add edge case tests** - Test same stages, backwards progression, gaps
   - **Test quality insights**:
     - Current tests use `tempfile.TemporaryDirectory()` - good practice
     - Current tests validate exact group counts and sizes - excellent
     - Need tests that would catch the "same stage grouping" bug we had
   - **Priority**: Medium - improve test robustness and coverage

### **📚 Future Analysis & Development**
4. **Create missing test files** - Need tests for: `02_face_grouper.py`, `04_multi_crop_tool.py`, `06_web_duplicate_finder.py`
5. **Create utility tests** - Need tests for: `utils/character_processor.py`, `utils/duplicate_checker.py`, `utils/recursive_file_mover.py`, `utils/similarity_viewer.py`, `utils/triplet_deduplicator.py`, `utils/triplet_mover.py`
6. **Analyze coverage report** - Review `scripts/tests/htmlcov/index.html` for coverage gaps
7. **Improve code coverage** - Focus on increasing test coverage for critical scripts
8. **Catalog conventions** - Analyze all scripts for reusable patterns and create reference docs
9. **Root directory cleanup** - Check files in root directory to see if they need to be cleaned out or moved to a special directory
10. **Create local homepage** - Build custom homepage in Documents directory with links to all AI systems and tools from Google homepage
11. **Investigate Claude memory configuration** - Look into enabling memory tool in Cursor/Claude settings to improve session continuity and knowledge retention

## 📚 MAJOR ANALYSIS PROJECT: Code Conventions & Patterns

### **🎯 Catalog Conventions (Main Task)**
**Analyze all scripts to identify reusable patterns and conventions**
- **Goal**: Create comprehensive reference for consistent development
- **Scope**: All scripts in `/scripts/` directory and utilities
- **Output**: `Documents/CONVENTIONS_REFERENCE.md` with templates and best practices
- **Benefits**: Consistency, efficiency, maintainability, easier onboarding, reduced bugs
- **Strategy**: Tackle after quick wins for momentum

### **🌐 Web Tool Conventions Analysis**
**Analyze web tools for common patterns:**
- **Scripts**: 01_web_image_selector, 03_web_character_sorter, 05_web_multi_directory_viewer, 06_web_duplicate_finder
- **Patterns to catalog**:
  - Flask structure and routing patterns
  - CSS classes and styling conventions
  - JavaScript functions and modules
  - UI components and layouts
  - Error handling patterns
  - File upload/processing workflows

### **🖥️ Desktop Tool Conventions Analysis**
**Analyze desktop tools for common patterns:**
- **Scripts**: 01_desktop_image_selector_crop, 04_batch_crop_tool_multi
- **Patterns to catalog**:
  - Matplotlib setup and event handling
  - File processing workflows
  - Progress tracking systems
  - UI layouts and interactions
  - Keyboard/mouse event handling

### **🔄 Cross-Platform Conventions Analysis**
**Identify patterns that work across both web and desktop:**
- **Shared utilities**: FileTracker and ActivityTimer usage patterns
- **Common workflows**: Progress tracking and session management
- **Infrastructure**: Error handling and logging patterns
- **File operations**: Directory scanning and image processing
- **Safety patterns**: File operations and data protection

### **📖 Create Conventions Reference**
**Build comprehensive documentation:**
- **Location**: `Documents/CONVENTIONS_REFERENCE.md`
- **Content**: Code templates and best practices
- **Organization**: Categorized by tool type and functionality
- **Templates**: Ready-to-use code snippets for common patterns
- **Examples**: Real code examples from existing tools

## 🎯 STRATEGIC APPROACH

### **📋 Task Categories & Priority**
1. **🚀 QUICK WINS** - Knock out first 3-4 items quickly for momentum
2. **📋 MAINTENANCE TASKS** - Test maintenance section (completed items for reference)
3. **🎯 WORKFLOW TASKS** - Photo processing workflow, identify and process directories
4. **📚 ANALYSIS TASKS** - Bigger projects (catalog conventions, analyze patterns, create reference docs)

### **⚡ Execution Strategy**
**When you return:**
1. **Start with Quick Wins** - Git issue, redundant text, title updates (build momentum)
2. **Tackle Single Keep Logic** - More complex but well-defined
3. **Move to Analysis Projects** - When we have more time for comprehensive work

**Key Insight**: The git tracking issue should definitely be first - that's probably just a simple configuration or path issue that we can solve in minutes!

## 🔍 Future Investigation Items

### **Web Interface Template System**
**Investigate template design for web tools consolidation:**
- **Scope**: 01_web_image_selector, 03_web_character_sorter, 05_web_multi_directory_viewer
- **Common elements**: Same header, color palette (WEB_STYLE_GUIDE.md), similar JavaScript patterns
- **Shared functionality**: Image clicking, delete buttons, crop operations, navigation
- **Goal**: Determine if template would simplify maintenance or add complexity
- **Benefits**: Easier updates, consistent UI, reduced code duplication
- **Considerations**: Balance between reusability and tool-specific customization

---

**🎉 INCREDIBLE SESSION COMPLETE!** You now have a professional-grade productivity analytics system tracking your 100k+ file processing workflow! 📊✨🚀
