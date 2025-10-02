# Technical Knowledge Base
## Key Learnings and Solutions for Image Processing Workflow

*This file contains technical solutions, common bugs, and patterns that work well for the image processing workflow.*

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

### **Clear Visual Feedback**
**Pattern:** Multiple visual indicators for status
**Implementation:**
- Status text at top of images (big, bold)
- Border colors (blue/red)
- Control text below images
- Console feedback

---

## 🔧 **Technical Patterns**

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

- **Style Guide:** `WEB_STYLE_GUIDE.md`
- **Test Suite:** `scripts/tests/test_runner.py`
- **File Tracker:** `scripts/file_tracker.py`
- **Activity Timer:** `scripts/utils/activity_timer.py`

---

*Last Updated: October 1, 2025*
*This file should be updated whenever new technical solutions are discovered or patterns are established.*
