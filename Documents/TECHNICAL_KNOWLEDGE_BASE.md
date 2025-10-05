# Technical Knowledge Base
## Key Learnings and Solutions for Image Processing Workflow

*This file contains technical solutions, common bugs, and patterns that work well for the image processing workflow.*

---

## 🏗️ **Major Architectural Improvements (October 2025)**

### **Centralized Utility System**
**Achievement:** Created comprehensive `companion_file_utils.py` with shared functions
**Impact:** Eliminated code duplication across 6+ scripts
**Key Functions:**
- `find_all_companion_files()` - Wildcard companion file detection
- `move_file_with_all_companions()` - Safe file movement with companions
- `launch_browser()` - Centralized browser launching
- `generate_thumbnail()` - Optimized thumbnail generation
- `format_image_display_name()` - Consistent image name formatting
- `calculate_work_time_from_file_operations()` - Intelligent work time calculation

### **File-Operation-Based Timing System**
**Achievement:** Replaced ActivityTimer with intelligent file-operation timing
**Benefits:** More accurate work time tracking, automatic break detection
**Implementation:** Analyzes FileTracker logs to calculate actual work time
**Tools Updated:** All file-heavy tools (image selector, character sorter, crop tools)

### **Desktop Tool Refactoring**
**Achievement:** Created `BaseDesktopImageTool` base class
**Impact:** Eliminated 200+ lines of duplicate code between desktop tools
**Benefits:** Consistent behavior, easier maintenance, shared improvements
**Tools Refactored:** `01_desktop_image_selector_crop.py`, `04_multi_crop_tool.py`

### **Project Organization Cleanup**
**Achievement:** Moved all files to proper directories
**Structure:**
- `Documents/` - All documentation and guides
- `data/` - All data files and models
- `scripts/tests/` - All test files
- Root directory - Only essential config files (.gitignore, .coverage, etc.)

---

## 🐛 **Common Bugs & Solutions**

### **Matplotlib Display Crashes**
**Problem:** Desktop image selector crop tool crashes when advancing to next triplet
**Root Cause:** Recreating matplotlib display on every triplet load causes backend conflicts
**Solution:** Only recreate display when number of images changes, reuse existing display otherwise
```python
# Only recreate display if number of images changed
if not hasattr(self, 'current_num_images') or self.current_num_images != num_images:
    self.setup_display(num_images)
    self.current_num_images = num_images
else:
    # Reuse existing display
```
**Date:** October 1, 2025

### **FileTracker Method Name Mismatch**
**Problem:** `'FileTracker' object has no attribute 'log_action'`
**Root Cause:** Method is called `log_operation`, not `log_action`
**Solution:** Use correct method name with proper parameters
```python
# Wrong:
self.tracker.log_action("crop", str(png_path))

# Correct:
self.tracker.log_operation("crop", source_dir=str(png_path.parent), dest_dir=str(png_path.parent))
```
**Date:** October 1, 2025

### **Aspect Ratio Auto-Adjustment Resetting Status**
**Problem:** When crop tool auto-adjusts for aspect ratio, it resets image status from KEEP back to DELETE
**Root Cause:** Aspect ratio adjustment triggers crop selection event again, calling select_image()
**Solution:** Check current status before auto-selecting
```python
# Only auto-select if currently marked for deletion
current_status = self.image_states[image_idx]['status']
if current_status == 'delete':
    self.select_image(image_idx)
else:
    # Preserve existing status
```
**Date:** October 1, 2025

### **ActivityTimer Integration Issues**
**Problem:** ActivityTimer causing crashes and complexity in file-heavy tools
**Root Cause:** ActivityTimer designed for scroll-heavy tools, not file operations
**Solution:** Replaced with file-operation-based timing system
```python
# Old approach (problematic):
activity_timer.mark_activity()
activity_timer.log_operation("crop", file_count=1)

# New approach (intelligent):
work_time = calculate_work_time_from_file_operations(file_operations)
```
**Date:** October 3, 2025

### **Search/Replace Failures During Refactoring**
**Problem:** Multiple search/replace operations failing due to whitespace variations
**Root Cause:** Exact string matching too strict for large refactoring operations
**Solution:** Use more precise edits, read exact lines before replacing
**Best Practice:** Break large refactoring into smaller, more targeted changes
**Date:** October 3, 2025

### **JavaScript Syntax Errors in Dashboard**
**Problem:** Extra closing braces causing JavaScript syntax errors
**Root Cause:** Manual editing introducing syntax errors
**Solution:** Always validate JavaScript syntax after edits
**Prevention:** Use proper indentation and bracket matching
**Date:** October 3, 2025

---

## 🎨 **UI/UX Patterns That Work**

### **Colorblind-Friendly Colors**
**Use:** Blue/Red instead of Green/Red for better accessibility
**Implementation:** 
- Blue = KEEP/Selected
- Red = DELETE/Unselected

### **Dynamic Layout Based on Content**
**Pattern:** Adjust UI layout based on actual data (2 vs 3 images)
**Implementation:**
- Detect number of items
- Adjust spacing and sizing accordingly
- Reuse existing display when possible

### **Centralized Error Display**
**Pattern:** Persistent, dismissible error bars instead of alert popups
**Implementation:**
```html
<div class="error-bar" id="error-bar" style="display: none;">
    <span id="error-message"></span>
    <button onclick="hideError()">×</button>
</div>
```
**Benefits:** Non-blocking, persistent, better UX

### **Intelligent Work Time Calculation**
**Pattern:** Calculate work time from file operations with break detection
**Implementation:**
```python
def calculate_work_time_from_file_operations(file_operations, break_threshold_minutes=5):
    # Only count time between operations if gap < threshold
    # Automatically detects breaks and excludes idle time
```
**Benefits:** More accurate than manual timers, automatic break detection

### **Wildcard Companion File Logic**
**Pattern:** Find all files with same base name as image
**Implementation:**
```python
def find_all_companion_files(image_path):
    base_name = image_path.stem
    return [f for f in parent_dir.iterdir() 
            if f.stem == base_name and f != image_path]
```
**Benefits:** Handles any file type, future-proof, consistent behavior

---

## 🔧 **Technical Patterns**

### **Base Class Inheritance Pattern**
**Pattern:** Create base classes for tools with shared functionality
**Implementation:**
```python
class BaseDesktopImageTool:
    def __init__(self, tool_name):
        # Shared initialization
    def setup_display(self, num_images):
        # Shared display logic
    def load_image_safely(self, image_path, subplot_idx):
        # Shared image loading
```

### **Centralized Utility Pattern**
**Pattern:** Move common functions to shared utility modules
**Benefits:** Single source of truth, easier maintenance, consistent behavior
**Implementation:** Create `companion_file_utils.py` with all shared functions

### **File-Operation Timing Pattern**
**Pattern:** Use file operations to calculate work time instead of manual timers
**Benefits:** More accurate, automatic break detection, no user interaction required
**Implementation:** Analyze FileTracker logs with intelligent gap detection

### **Triplet Detection Logic - SIMPLE IS BETTER**
**Pattern:** Group images by strictly increasing stage numbers using simple comparison
**Critical Rule:** Timestamps are ONLY for SORTING, stage numbers are for GROUPING
**Revolutionary Insight:** Simple solutions are often better than "robust" over-engineered ones

**The Problem with Over-Engineering:**
1. **Complex lookup tables:** Unnecessary complexity for simple comparisons
2. **Configuration parameters:** `consecutive_only`, `ordered_stages` - more things to get wrong
3. **Brittle design:** Breaks if `ordered_stages` doesn't match your data
4. **Hard to debug:** More moving parts to go wrong

**The Simple Solution That Actually Works:**
```python
def group_progressive(files, stage_of, min_group_size=2):
    """
    Group files into progressive stage runs.
    
    Args:
        files: list of file paths/objects sorted by timestamp (and then stage).
        stage_of: callable that extracts the float stage from a file.
        min_group_size: only emit groups >= this many files.
        
    Returns:
        list of groups (each group is a list of files).
    """
    groups = []
    n = len(files)
    i = 0

    while i < n:
        # start a new group anywhere
        cur_group = [files[i]]
        prev_stage = stage_of(files[i])
        i += 1

        # extend the run while stage strictly increases
        while i < n:
            s = stage_of(files[i])
            if s > prev_stage:  # strictly increasing, any jump allowed
                cur_group.append(files[i])
                prev_stage = s
                i += 1
            else:
                break  # stage repeated or decreased — end group

        if len(cur_group) >= min_group_size:
            groups.append(cur_group)

    return groups

# Usage:
files = sort_image_files_by_timestamp_and_stage([...])
groups = group_progressive(
    files,
    stage_of=lambda p: get_stage_number(detect_stage(p.name)),
    min_group_size=2,
)
```

**Why This Is Revolutionary:**

1. **Simplicity:** 15 lines of clear logic vs 50+ lines of complex lookup tables
2. **Self-Documenting:** `if s > prev_stage:` - crystal clear intent
3. **Robust:** Works with any stage numbering system, no configuration needed
4. **Future-Proof:** Automatically handles new stages (like `stage4_final`)
5. **No Configuration Errors:** No parameters to get wrong
6. **Handles All Cases:** 1→2, 1→3, 1.5→3, 2→3, 1→1.5→2→3 - all work naturally

**The Key Insight:**
Your workflow is simple: **"Group files where each stage is greater than the previous stage."** 

This code implements exactly that logic without any unnecessary complexity.

**Real-World Example:**
- Sorted files: `stage2_upscaled`, `stage2_upscaled`, `stage3_enhanced`, `stage2_upscaled`
- Logic: `stage2_upscaled` (2.0) → `stage2_upscaled` (2.0) → `stage3_enhanced` (3.0)
- Result: Groups `stage2_upscaled` → `stage2_upscaled` → `stage3_enhanced` (stops at next `stage2_upscaled`)

**Centralized Implementation:**
This logic is now in `companion_file_utils.py` as `find_consecutive_stage_groups()`, ensuring ALL tools use the same simple, robust algorithm.

**CRITICAL RULE — TIMESTAMPS ARE ONLY FOR SORTING**
Do not use timestamps for grouping boundaries or gap decisions. They are inherently unreliable for gap inference. We use timestamps strictly to sort files deterministically before grouping. Grouping itself is based ONLY on stage numbers and the nearest-up rule below.

**Nearest-Up Grouping (Definitive Spec):**
- Files are pre-sorted by `(timestamp, then stage)`.
- A run starts at any file; at each step, pick the smallest stage strictly greater than the previous stage within a lookahead window.
- Boundaries:
  - If a duplicate or non-increasing stage is encountered, the current run ends.
  - No time-gap boundaries in production. `time_gap_minutes` exists for rare analysis but defaults to `None` and should not be used in normal workflows.
- Defaults: `min_group_size=2`, `lookahead=50`, `time_gap_minutes=None`.
- Determinism: Sorting + nearest-up selection produces stable, predictable groups.

**Practical Examples:**
- Nearest-up with early stage3 present:
  - Files: `1`, `3` (00:10), `2` (00:20), `3` (00:30)
  - Group: `[1, 2, 3]` (the early `3` is ignored until `2` is found, then the later `3` completes the run)
- Duplicate stage splits runs:
  - Files: `1`, `1.5`, `2`, `2`, `3`
  - Groups: `[1, 1.5, 2]`, `[2, 3]`
- Large timestamp gaps are ignored for grouping (sorting only):
  - Files: `1` (00:00), `2` (00:10), `3` (01:00)
  - Group: `[1, 2, 3]` in production (no time-gap cutoffs)

**Critical Lessons Learned:**
1. **Simple, direct solutions are often better** than "robust" over-engineered ones
2. **Don't solve problems you don't have** - avoid unnecessary complexity
3. **Configuration parameters are liabilities** - more things to get wrong
4. **Self-documenting code is better** than complex algorithms with explanations
5. **Simplicity is the ultimate sophistication**

**Date:** October 3, 2025 (learned that simple solutions are often better than complex ones)

### **Critical Testing Lessons - Why Tests Were Failing Us**
**Problem:** Our tests were passing when they should have been failing, allowing bugs to persist for weeks
**Root Cause:** Tests were testing "does it run?" instead of "does it work correctly?"

**The Terrible Test Pattern (What NOT to do):**
```python
# BAD TEST - This passes even with completely broken logic!
assert any("stage1" in f for f in filenames), "Should have stage1 files"
assert len(groups) > 0, "Should detect at least some triplet groups"
```

**Why This Test Was Terrible:**
1. **Weak Assertions:** `any("stage1" in f for f in filenames)` - passes even with random grouping
2. **No Edge Case Testing:** Doesn't test same stages, backwards progression, or specific requirements
3. **No Validation of Grouping Logic:** Only checks that some files were found, not HOW they were grouped
4. **No Comprehensive Coverage:** Doesn't test all valid combinations (1→2, 1→3, 1.5→3, etc.)

**The Excellent Test Pattern (What TO do):**
```python
# GOOD TEST - Tests specific expected behavior
test_cases = [
    ("1→1.5", ["stage1_generated.png", "stage1.5_face_swapped.png"], 1, [2]),
    ("1→2", ["stage1_generated.png", "stage2_upscaled.png"], 1, [2]),
    ("1→3", ["stage1_generated.png", "stage3_enhanced.png"], 1, [2]),
    ("1.5→2", ["stage1.5_face_swapped.png", "stage2_upscaled.png"], 1, [2]),
    ("1.5→3", ["stage1.5_face_swapped.png", "stage3_enhanced.png"], 1, [2]),
    ("2→3", ["stage2_upscaled.png", "stage3_enhanced.png"], 1, [2]),
    # ... all combinations
]

for description, test_files, expected_groups, expected_sizes in test_cases:
    groups = find_consecutive_stage_groups(file_paths)
    assert len(groups) == expected_groups, f"{description}: Expected {expected_groups} groups, got {len(groups)}"
    actual_sizes = [len(group) for group in groups]
    assert actual_sizes == expected_sizes, f"{description}: Expected group sizes {expected_sizes}, got {actual_sizes}"
```

**Why This Test Is Excellent:**
1. **Tests ALL valid combinations:** Every possible consecutive stage progression
2. **Validates exact group counts:** Not just "some groups exist"
3. **Validates stage progression:** Ensures stages are actually consecutive and in order
4. **Tests edge cases:** Same stages (should NOT group), backwards progression (should break groups)
5. **Would catch bugs immediately:** Same stage grouping would fail the first test

**Critical Testing Principles:**
1. **Test specific expected behavior** - not just "does it run without crashing"
2. **Test edge cases** - same stages, backwards progression, invalid data
3. **Test all valid combinations** - don't assume only one pattern works
4. **Validate exact outputs** - group counts, group sizes, stage progressions
5. **Test would catch the bug** - if the test passes with broken logic, it's a bad test

**The Lesson:** A test that passes with broken logic is worse than no test at all - it gives false confidence and hides bugs for weeks.

### **The Final Testing Insight**
**Critical Discovery:** Comprehensive tests that validate specific behavior are essential

**What Makes Our Tests Excellent Now:**
1. **Tests ALL valid combinations:** 1→1.5, 1→2, 1→3, 1.5→2, 1.5→3, 2→3, 1→1.5→2, 1→1.5→3, 1→2→3, 1.5→2→3, 1→1.5→2→3
2. **Validates exact group counts:** Not just "some groups exist"
3. **Validates stage progression:** Ensures stages are actually consecutive and in order
4. **Tests edge cases:** Same stages (should NOT group), backwards progression (should break groups)
5. **Would catch bugs immediately:** Same stage grouping would fail the first test

**The Test That Would Have Caught the Bug:**
```python
def test_same_stage_not_grouped():
    """Test that same stages are NOT grouped together (this would catch the bug)"""
    test_files = [
        "20250705_214626_stage2_upscaled.png",
        "20250705_214953_stage2_upscaled.png",  # Same stage
        "20250705_215137_stage2_upscaled.png",  # Same stage
        "20250705_215319_stage2_upscaled.png",  # Same stage
    ]
    
    groups = find_consecutive_stage_groups(file_paths)
    
    # CRITICAL TEST: Same stages should NOT be grouped together
    # This test would have FAILED with the old broken logic!
    assert len(groups) == 0, f"Same stages should not be grouped, but got {len(groups)} groups"
```

**Why This Test Is Perfect:**
- **Specific:** Tests exact behavior (same stages should not group)
- **Clear failure:** Would immediately show the bug
- **Edge case:** Tests the exact scenario that was broken
- **Comprehensive:** Tests all valid combinations plus edge cases

**Date:** October 3, 2025 (learned that comprehensive tests prevent weeks of broken functionality)

### Testing Playbook (Triplet Grouping)

- Always pre-sort input with `sort_image_files_by_timestamp_and_stage(files)` before grouping.
- Validate exact behavior with strong assertions (group counts, sizes, and strict stage sequences).
- Covered cases in `scripts/tests/test_triplet_detection_logic.py`:
  - All valid consecutive combinations (1→1.5, 1→2, 1→3, etc.)
  - Same-stage NOT grouped
  - Strictly increasing order within a group
  - Backwards stage breaks the group
  - Nearest-up selection (e.g., 1,3,2,3-later → [1,2,3])
  - Duplicate stage ends run
  - Integration check against real data in `mojo1`

Production stance: Do not use time gaps in grouping tests. Timestamps are for sorting only.

### Centralized Sorting - Non-Negotiable Rule

- All human-facing tools MUST call `sort_image_files_by_timestamp_and_stage(files)` before any display or grouping.
- Determinism: Sorting is by (timestamp, then stage number, then filename) to produce stable order.
- Where to use: web image selector, desktop selector + crop, multi-crop tool, character sorter, viewers.
- Unit test added: `scripts/tests/test_sorting_determinism.py` to validate ordering on a known 4-file set.

### **Test Suite Maintenance**
**Pattern:** Always catalog changes made without corresponding test updates
**Implementation:** Use todo list to track changes that need test updates later
**Example:** "Oct 1: Changed desktop image selector crop tool title to show just image name instead of batch/progress info"

### **Subprocess Path Handling**
**Pattern:** Always use proper working directory and relative paths in subprocess calls
**Implementation:**
```python
result = subprocess.run([
    sys.executable, "script_name.py", args
], capture_output=True, text=True, cwd=Path(__file__).parent)
```

### **Matplotlib Backend Setup**
**Pattern:** Consistent backend setup across all matplotlib-based tools
**Implementation:**
```python
# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

try:
    matplotlib.use('Qt5Agg', force=True)
    backend_interactive = True
except Exception as e:
    matplotlib.use('Agg', force=True)
    backend_interactive = False
```

### **Progress Tracking & Session Management Patterns**
**Pattern:** Robust progress tracking with stable IDs and graceful error handling
**Critical Insight:** Progress files need to be stable, portable, and handle edge cases gracefully

**Key Components:**

1. **Stable ID Generation (Path-Portable):**
```python
def make_triplet_id(paths):
    """Create stable ID that works across Windows/POSIX systems."""
    # Use as_posix() for cross-platform compatibility
    s = "|".join(p.resolve().as_posix() for p in paths)
    return "t_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
```

2. **Normalized Progress Filenames:**
```python
# Avoid path drift and super-long filenames
abs_base = self.base_directory.resolve()
safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', abs_base.as_posix())[:200]
self.progress_file = self.progress_dir / f"{safe_name}_progress.json"
```

3. **Immediate Persistence After Reconciliation:**
```python
def load_progress(self):
    # ... load existing data ...
    self.ensure_all_triplets_in_session_data()
    self.save_progress()  # CRITICAL: Persist immediately
    self.migrate_old_keys()  # One-time migration
```

4. **Graceful Error Handling:**
```python
def cleanup_completed_session(self):
    try:
        if self.progress_file.exists():
            self.progress_file.unlink()
    except PermissionError:
        print("[!] Could not remove progress file (locked); ignoring.")
    except Exception as e:
        print(f"[!] Error cleaning up progress file: {e}")
```

5. **Status Distinction & Helper Methods:**
```python
def mark_status(self, status):
    """Mark current triplet with specific status (completed/skipped)."""
    ct = self.get_current_triplet()
    if not ct: return
    
    d = self.session_data.setdefault('triplets', {})
    key = ct.id if ct.id in d else ct.display_name
    d.setdefault(key, {
        'display_name': ct.display_name,
        'files_processed': 0,
        'total_files': len(ct.paths),
    })
    d[key]['status'] = status
    self.save_progress()
```

6. **One-Time Migration for Backward Compatibility:**
```python
def migrate_old_keys(self):
    """Migrate old display_name keys to stable IDs."""
    trips = self.session_data.get('triplets', {})
    changed = False
    for t in self.triplets:
        if t.display_name in trips and t.id not in trips:
            trips[t.id] = trips.pop(t.display_name)
            changed = True
    if changed:
        self.save_progress()
```

**Why These Patterns Matter:**
1. **Cross-Platform Stability:** `as_posix()` prevents hash changes between Windows/POSIX
2. **Immediate Persistence:** Prevents data loss if tool crashes during reconciliation
3. **Graceful Degradation:** File locks don't crash the tool
4. **Status Tracking:** Distinguish between completed vs skipped items
5. **Backward Compatibility:** Migrate old progress files automatically
6. **Clean Filenames:** Avoid filesystem issues with special characters

**Critical Lessons:**
- **Always persist immediately** after data reconciliation
- **Use stable, content-derived IDs** instead of display names for keys
- **Handle file system edge cases** (locks, permissions, long names)
- **Plan for migration** when changing data structures
- **Distinguish between different completion states** (completed vs skipped)

**Date:** October 3, 2025 (learned from ChatGPT conversation about robust progress tracking)

### **Code Improvement Patterns - Systematic Enhancement**
**Pattern:** Apply systematic improvements to existing code based on external feedback
**Critical Insight:** External code reviews often identify patterns that internal developers miss

**The Systematic Improvement Process:**

1. **External Code Review:** Get fresh perspective from experienced developers
2. **Categorize Improvements:** Group suggestions by type (stability, portability, UX, robustness)
3. **Implement Systematically:** Apply all improvements of the same type together
4. **Document Patterns:** Capture the patterns for future reference

**Example: Progress Tracking Improvements (October 2025)**

**External Feedback Identified:**
- Persist immediately after reconciliation
- Normalize progress filenames
- Make IDs path-portable
- Distinguish skipped vs completed
- Add migration for old keys
- Handle file locks gracefully

**Systematic Implementation:**
```python
# 1. Stable ID Generation
def make_triplet_id(paths):
    s = "|".join(p.resolve().as_posix() for p in paths)
    return "t_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

# 2. Normalized Filenames
abs_base = self.base_directory.resolve()
safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', abs_base.as_posix())[:200]

# 3. Immediate Persistence
self.ensure_all_triplets_in_session_data()
self.save_progress()  # CRITICAL: Persist immediately

# 4. Status Distinction
def mark_status(self, status):
    # Mark as 'completed' or 'skipped'

# 5. Migration Support
def migrate_old_keys(self):
    # One-time migration for backward compatibility

# 6. Graceful Error Handling
except PermissionError:
    print("[!] Could not remove progress file (locked); ignoring.")
```

**Why This Pattern Works:**
1. **Fresh Perspective:** External reviewers see patterns internal developers miss
2. **Systematic Application:** All related improvements applied together
3. **Pattern Documentation:** Future developers can apply same patterns
4. **Comprehensive Coverage:** Addresses stability, portability, UX, and robustness

**When to Use This Pattern:**
- After major feature development
- When code has been in production for a while
- Before refactoring or major changes
- When external feedback is available

**Critical Success Factors:**
1. **Don't cherry-pick:** Apply all improvements of the same category
2. **Document the patterns:** Capture why each improvement matters
3. **Test thoroughly:** Systematic changes need comprehensive testing
4. **Update documentation:** Keep knowledge base current with new patterns

**Date:** October 3, 2025 (learned from systematic application of external code review feedback)

### Tool Behavior at a Glance

- Web Image Selector (`scripts/01_web_image_selector.py`):
  - Modern batch UI; exactly one selection per group; selected items move to `selected/`, others go to Trash by default (`send2trash`).
  - Requires `send2trash` unless `--hard-delete` is explicitly used (dangerous).
  - Uses centralized grouping; timestamps used only for sorting.

- Desktop Image Selector + Crop (`scripts/01_desktop_image_selector_crop.py`):
  - Single-selection per triplet with immediate cropping; unselected files go to Trash.
  - Progress is persisted with stable, path-portable IDs; immediate persistence after reconciliation.
  - Uses the same centralized grouping and sorting rules as the web selector.
  - New flag: `--reset-progress` clears saved progress for the directory and rediscover groups from scratch.
  - Enter behavior: If no image is selected, Enter deletes all images in the current triplet and advances. If one image is selected, Enter crops it, deletes the others, and advances.

### Glossary

- Group: Sequence of images with strictly increasing stage numbers (min size 2).
- Pair/Triplet: Group of size 2/3.
- Selected: The image chosen to keep for a group.
- Skipped: Leave files in place (web) or mark triplet as skipped (desktop).
- Trash/Delete: Non-selected images are sent to system Trash by default; hard delete is opt-in and risky.

### Troubleshooting (Quick List)

- “No groups found” → Ensure filenames have timestamps and stage tokens; confirm inputs are pre-sorted.
- “Unexpected grouping” → Check for duplicates or backward stage steps before expected completion.
- “Trash not available” → Install `send2trash` or run with `--hard-delete` (dangerous, avoid if unsure).
- Desktop crashes on navigation → Verify backend initialization and avoid recreating displays when count unchanged.

---

## 📝 **Workflow Principles**

### **During Work Sessions**
- Only fix bugs and make functional changes
- Log all changes in todo list for later test maintenance
- No tiny test fixes during active work

### **End of Day**
- Do cleanup and test fixes
- Commit changes
- Update documentation

### **File Safety**
- Never alter zip directory contents
- Always use send2trash for deletions
- Test file operations before implementing

---

## 🚨 **Critical Rules**

1. **Never alter zip directory contents** - only extract/copy from them
2. **Always activate virtual environment** before running scripts
3. **Only run scripts when testing or explicitly asked**
4. **Keep repository clean** - remove temporary files after use
5. **Always use PWD before creating directories/files**

---

## 📚 **Reference Links**

- **Style Guide:** `Documents/WEB_STYLE_GUIDE.md`
- **TODO List:** `Documents/CURRENT_TODO_LIST.md`
- **Test Suite:** `scripts/tests/test_runner.py`
- **File Tracker:** `scripts/file_tracker.py`
- **Activity Timer:** `scripts/utils/activity_timer.py`
- **Companion File Utils:** `scripts/utils/companion_file_utils.py`
- **Base Desktop Tool:** `scripts/utils/base_desktop_image_tool.py`

---

## 🧪 **Testing Patterns & Infrastructure**

### **Selenium Integration Testing (October 2025)**
**Achievement:** Complete Selenium test infrastructure for all web tools
**Impact:** Automated end-to-end verification of all Flask applications

**Infrastructure Components:**

1. **Base Selenium Test Class** (`test_base_selenium.py`):
```python
class BaseSeleniumTest(unittest.TestCase):
    """Base class with headless Chrome + Flask server management."""
    
    @classmethod
    def setUpClass(cls):
        # Set up Chrome driver once for all tests
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
    
    def setUp(self):
        # Create temp directory, start Flask server on free port
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server_port = find_free_port()
        app = self.get_flask_app()  # Subclass implements
        self.server_thread = threading.Thread(
            target=lambda: app.run(port=self.server_port),
            daemon=True
        )
        self.server_thread.start()
        self.wait_for_server()
```

2. **Smoke Tests for All Web Tools** (`test_web_tools_smoke.py`):
- Tests that each tool starts without errors
- Verifies page loads and displays content
- Checks that key UI elements are present
- Uses subprocess to launch actual Python scripts
- Runs in ~10 seconds for all 4 tools

3. **Key Features:**
- **Headless mode:** No browser windows pop up
- **Automatic port management:** Finds free ports automatically
- **Test isolation:** Each test gets temp directory + unique port
- **Clean teardown:** Servers terminated, pipes closed, no orphans
- **Real integration:** Actually launches your Flask apps as subprocesses

**Critical Lessons:**

1. **Coverage Limitation (Expected):**
Selenium tests that launch subprocesses don't show up in coverage reports. This is normal and fine:
- **Selenium tests verify:** Integration/functionality (does it work end-to-end?)
- **Unit tests verify:** Code coverage (are all code paths tested?)
- Both types of tests are important and complementary

2. **Subprocess Testing Pattern:**
```python
# Launch actual script as subprocess
self.process = subprocess.Popen(
    [sys.executable, str(script_path), args],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=project_root
)

# Wait for server to start
self.wait_for_server(port)

# Use Selenium to verify UI
self.driver.get(f"http://127.0.0.1:{port}")
self.assertIn("Expected Title", self.driver.title)

# Always clean up
def tearDown(self):
    if self.process:
        self.process.terminate()
        self.process.wait(timeout=5)
        # Close pipes to prevent ResourceWarning
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()
```

3. **Port Management:**
```python
def find_free_port() -> int:
    """Find a free port for the Flask server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
```

4. **Test Data Naming Conventions:**
For image grouping tests, use proper file naming:
```python
# Correct naming for grouping tests
descriptors = {1: "generated", 2: "upscaled", 3: "enhanced"}
for stage in [1, 2, 3]:
    filename = f"20250101_000000_stage{stage}_{descriptors[stage]}.png"
```

**Test Suite Organization:**

- `test_base_selenium.py` - Infrastructure base class
- `test_selenium_simple.py` - Infrastructure verification (3 tests)
- `test_web_tools_smoke.py` - Web tool smoke tests (4 tests)
- Total: 7 Selenium tests, all passing, ~10 second runtime

**Benefits:**
- Catch integration issues before production
- Verify tools actually start and work
- Test real browser interactions
- No manual testing needed for basic functionality

**When to Use:**
- Verifying Flask apps start correctly
- Testing UI elements are present
- Integration testing (multiple systems working together)
- Regression testing after major changes

**When NOT to Use:**
- Testing internal logic (use unit tests)
- Testing individual functions (use unit tests)
- Measuring code coverage (use unit tests)

---

### **Test Isolation Patterns**

**Critical Pattern:** Every test must run in complete isolation to prevent contamination.

**Implementation:**
```python
def setUp(self):
    # Create isolated temp directory
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_path = Path(self.temp_dir.name)
    
    # Set environment variable for test data root
    os.environ['EM_TEST_DATA_ROOT'] = str(self.temp_path)
    
    # Prepare test data in isolation
    self.prepare_test_data()

def tearDown(self):
    # Clean up environment variable
    if 'EM_TEST_DATA_ROOT' in os.environ:
        del os.environ['EM_TEST_DATA_ROOT']
    
    # Clean up temp directory
    if self.temp_dir:
        self.temp_dir.cleanup()
```

**Why This Matters:**
- Tests don't interfere with each other
- Tests don't pollute production data directories
- Tests are reproducible (same result every time)
- Can run tests in parallel safely

**Application Code Support:**
Production code should respect `EM_TEST_DATA_ROOT`:
```python
def get_data_directory():
    """Get data directory, respecting test environment."""
    if 'EM_TEST_DATA_ROOT' in os.environ:
        return Path(os.environ['EM_TEST_DATA_ROOT']) / 'data'
    return Path(__file__).parent.parent / 'data'
```

---

### **Flask App Testing Pattern**

**For simple unit tests (without browser):**
```python
def test_flask_route():
    app = create_app(test_data)
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
```

**For integration tests (with browser):**
Use BaseSeleniumTest pattern shown above - actually launch the app as a subprocess and test with real browser.

---

### **Headless Browser Configuration**

**Chrome Options for CI/CD:**
```python
chrome_options = ChromeOptions()
chrome_options.add_argument('--headless')          # No GUI
chrome_options.add_argument('--no-sandbox')        # Required for containers
chrome_options.add_argument('--disable-dev-shm-usage')  # Prevent crashes
chrome_options.add_argument('--disable-gpu')       # Not needed in headless
chrome_options.add_argument('--window-size=1920,1080')  # Set viewport
chrome_options.add_argument('--disable-extensions')     # Faster startup
chrome_options.add_argument('--disable-logging')        # Less noise
chrome_options.add_argument('--log-level=3')            # Errors only
```

**Why These Options:**
- `--headless`: No browser window (essential for automated testing)
- `--no-sandbox`: Required when running in Docker/CI environments
- `--disable-dev-shm-usage`: Prevents crashes when /dev/shm is too small
- Others: Performance and noise reduction

---

### **Coverage Report Interpretation**

**Expected Coverage Patterns:**
- **High coverage (>80%):** Utility functions, business logic, data processing
- **Medium coverage (40-80%):** Complex workflows, error handling paths
- **Low/Zero coverage (0%):** GUI tools, subprocess-launched apps, integration points

**Why Some Files Show 0% Coverage:**
1. **Desktop tools (tkinter):** Require GUI automation or headless X
2. **Web tools (Flask):** Routes not exercised by subprocess launches
3. **Integration tests:** Selenium launches subprocesses (separate Python process)

**This is NORMAL and EXPECTED.** Different test types serve different purposes:
- **Unit tests:** Code coverage, logic verification
- **Integration tests:** End-to-end functionality, system behavior
- **Smoke tests:** Does it start? Does it work basically?

---

### **Test Maintenance Workflow**

**During Active Development:**
1. Make functional changes
2. Log test impact in TODO list
3. Don't stop to fix tests immediately

**End of Day:**
1. Fix test failures caused by changes
2. Add new tests for new features
3. Update test data if schemas changed
4. Run full test suite before committing

**After Major Changes:**
1. Review test coverage report
2. Add tests for uncovered edge cases
3. Update test documentation
4. Consider integration tests if behavior changed

---

*Last Updated: October 5, 2025*
*This file should be updated whenever new technical solutions are discovered or patterns are established.*
