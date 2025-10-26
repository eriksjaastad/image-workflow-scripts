# Companion File System - Complete Guide
**Status:** Active
**Audience:** Developers, Operators


## 📋 **Overview**

The companion file system ensures that when an image (PNG) is moved or deleted, ALL associated files (.yaml, .caption, .txt, etc.) are moved/deleted together automatically.

---

## 🎯 **Core Principles**

1. **Single Source of Truth:** All companion file operations use centralized utilities in `scripts/utils/companion_file_utils.py`
2. **Automatic Discovery:** Companion files are found by matching the PNG's stem (filename without extension)
3. **Safe by Default:** Deletions use system trash (recoverable) unless `hard_delete=True` is explicitly specified
4. **Always Together:** Images and their companions are NEVER separated

---

## 🔧 **Centralized Utilities**

### Core Functions (in `scripts/utils/companion_file_utils.py`):

```python
# Find all companion files for an image
companions = find_all_companion_files(png_path: Path) -> List[Path]
# Returns: [.yaml, .caption, .txt, etc.] with same stem

# Move image + all companions to destination
moved = move_file_with_all_companions(
    png_path: Path,
    dest_dir: Path,
    dry_run: bool = False
) -> List[Path]

# Delete image + all companions (to trash by default)
deleted = safe_delete_image_and_yaml(
    png_path: Path,
    hard_delete: bool = False,
    tracker: Optional[FileTracker] = None
) -> List[Path]

# Delete multiple files at once
deleted = safe_delete_paths(
    paths: List[Path],
    hard_delete: bool = False,
    tracker: Optional[FileTracker] = None
) -> List[Path]
```

---

## ✅ **Production Scripts Using Centralized Utilities**

### Web Tools:
- ✅ `scripts/01_ai_assisted_reviewer.py`
- ✅ `scripts/03_web_character_sorter.py`
- ✅ `scripts/05_web_multi_directory_viewer.py`
- ✅ `scripts/06_web_duplicate_finder.py`

### Desktop Tools:
- ✅ `scripts/utils/base_desktop_image_tool.py` (base class for all desktop tools)
- ✅ `scripts/02_ai_desktop_multi_crop.py` (uses base class)

### Utilities:
- ✅ `scripts/utils/triplet_mover.py`
- ✅ `scripts/utils/triplet_deduplicator.py` (fixed October 20, 2025)

### All scripts that inherit from `BaseDesktopImageTool`:
- Automatically get proper companion file handling through base class

---

## 📝 **What Counts as a Companion File?**

Any file with the same stem (base name) as the PNG:

**Example:**
```
20250101_120000_stage1_generated.png      ← Primary file
20250101_120000_stage1_generated.yaml     ← Companion
20250101_120000_stage1_generated.caption  ← Companion
20250101_120000_stage1_generated.txt      ← Companion
20250101_120000_stage1_generated.json     ← Companion
20250101_120000_stage1_generated.cropped  ← Companion (sidecar marker)
```

**How it works:**
- PNG stem: `20250101_120000_stage1_generated`
- Find all files: `20250101_120000_stage1_generated.*`
- Exclude PNG itself: Any extension except `.png`
- Result: All companion files

---

## 🔍 **How to Use in Your Script**

### Moving Files:
```python
from pathlib import Path
from utils.companion_file_utils import move_file_with_all_companions

# Move image + companions to destination
png_path = Path("source/image.png")
dest_dir = Path("destination/")

moved_files = move_file_with_all_companions(png_path, dest_dir, dry_run=False)
print(f"Moved {len(moved_files)} files: {[f.name for f in moved_files]}")
```

### Deleting Files:
```python
from pathlib import Path
from utils.companion_file_utils import safe_delete_image_and_yaml
from file_tracker import FileTracker

# Delete image + companions (to trash)
png_path = Path("directory/image.png")
tracker = FileTracker()

deleted_files = safe_delete_image_and_yaml(
    png_path,
    hard_delete=False,  # Use trash (recoverable)
    tracker=tracker      # Optional: log operation
)
print(f"Deleted {len(deleted_files)} files")
```

### Finding Companions (Read-only):
```python
from pathlib import Path
from utils.companion_file_utils import find_all_companion_files

# Just find companions without moving/deleting
png_path = Path("directory/image.png")
companions = find_all_companion_files(png_path)

print(f"Found {len(companions)} companions:")
for comp in companions:
    print(f"  - {comp.name}")
```

---

## 🧪 **Test Coverage**

### Test Files:
1. `scripts/tests/test_companion_file_utils.py`
   - TestFindAllCompanionFiles (4 tests)
   - TestMoveFileWithAllCompanions (2 tests)
   - TestSafeDelete (6 tests) ← **NEW!**

2. `scripts/tests/test_triplet_deduplicator.py` ← **NEW FILE!**
   - TestRemoveTripletFiles (5 tests)

### What's Tested:
- ✅ Finding single companion (.yaml)
- ✅ Finding multiple companions (.yaml + .caption)
- ✅ Finding no companions (image only)
- ✅ Moving image + companions
- ✅ Deleting image + companions
- ✅ Deleting multiple images at once
- ✅ Hard delete vs trash
- ✅ Missing companion handling
- ✅ Error handling (continues on failure)

### Run Tests:
```bash
# All companion file tests
python -m pytest scripts/tests/test_companion_file_utils.py -v

# Just deletion tests
python -m pytest scripts/tests/test_companion_file_utils.py::TestSafeDelete -v

# Triplet deduplicator tests
python -m pytest scripts/tests/test_triplet_deduplicator.py -v
```

---

## ⚠️ **Common Mistakes to Avoid**

### ❌ DON'T: Manually find and delete companions
```python
# BAD - Misses .caption and other companions
yaml_file = png_file.parent / f"{png_file.stem}.yaml"
send2trash(str(png_file))
send2trash(str(yaml_file))
```

### ✅ DO: Use centralized utility
```python
# GOOD - Finds and deletes ALL companions
safe_delete_image_and_yaml(png_file, hard_delete=False, tracker=tracker)
```

---

### ❌ DON'T: Use shutil.move directly
```python
# BAD - Only moves PNG, orphans companions
shutil.move(str(png_file), str(dest_dir / png_file.name))
```

### ✅ DO: Use centralized utility
```python
# GOOD - Moves image + all companions together
move_file_with_all_companions(png_file, dest_dir, dry_run=False)
```

---

### ❌ DON'T: Check for specific extensions
```python
# BAD - Misses future companion types
for ext in ['.yaml', '.caption']:
    comp = png_file.parent / f"{png_file.stem}{ext}"
    if comp.exists():
        # ...
```

### ✅ DO: Use centralized discovery
```python
# GOOD - Automatically finds all companions
companions = find_all_companion_files(png_file)
for comp in companions:
    # ...
```

---

## 📈 **Benefits**

1. **No Orphaned Files:** Companions always stay with their images
2. **Consistent Behavior:** Same logic everywhere in codebase
3. **Easy Maintenance:** Update once, fix everywhere
4. **Test Coverage:** Comprehensive tests prevent regressions
5. **Safe by Default:** Trash (recoverable) unless explicitly hard delete
6. **Logging Support:** Optional FileTracker integration for audit trail
7. **Dry-run Support:** Preview operations before committing

---

## 🔮 **Future-Proof**

### Adding New Companion Types:
**No code changes needed!** The wildcard matching automatically includes new companion file types.

**Example:**
If you start using `.metadata` files:
```
image.png
image.yaml       ← Already supported
image.caption    ← Already supported
image.metadata   ← Automatically supported! (same stem)
```

The centralized utilities will automatically discover and handle the new file type.

---

## 📚 **Related Documentation**

- `Documents/TECHNICAL_KNOWLEDGE_BASE.md` - Overall system architecture
- `scripts/utils/companion_file_utils.py` - Source code with detailed docstrings
- `scripts/tests/test_companion_file_utils.py` - Test examples and edge cases
- `AUTONOMOUS_WORK_SESSION_SUMMARY.md` - Recent updates (October 20, 2025)

---

## 🎯 **Quick Reference Card**

```python
# Import centralized utilities
from utils.companion_file_utils import (
    find_all_companion_files,
    move_file_with_all_companions,
    safe_delete_image_and_yaml,
    safe_delete_paths
)

# Find companions (read-only)
companions = find_all_companion_files(png_path)

# Move image + companions
moved = move_file_with_all_companions(png_path, dest_dir)

# Delete image + companions (trash)
deleted = safe_delete_image_and_yaml(png_path, hard_delete=False)

# Delete multiple files
deleted = safe_delete_paths([path1, path2], hard_delete=False)
```

**Remember:** Always use centralized utilities. Never manually handle companions!

---

**Last Updated:** October 20, 2025  
**Status:** Production-ready, fully tested  
**Maintainer:** Centralized in `scripts/utils/companion_file_utils.py`

