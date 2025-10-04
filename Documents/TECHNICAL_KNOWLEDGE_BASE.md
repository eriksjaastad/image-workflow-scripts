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

*Last Updated: October 3, 2025*
*This file should be updated whenever new technical solutions are discovered or patterns are established.*
