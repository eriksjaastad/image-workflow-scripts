# Test Project Guide: Running a Full Pipeline Test

**Last Updated:** 2025-11-03
**Status:** Active
**Audience:** Developers
**Estimated Reading Time:** 15 minutes

## Overview

This guide shows you how to run a complete test project through your entire image workflow pipeline using **sandbox mode** for safe testing with easy cleanup.

## 🧪 What is Sandbox Mode?

Sandbox mode provides a safe, isolated testing environment:
- ✅ Test data stored separately from production
- ✅ Project IDs prefixed with `TEST-` for easy identification
- ✅ Simple one-command cleanup when done
- ✅ All file operations tracked separately
- ✅ No risk of polluting production data

For details, see: [`SANDBOX_MODE_GUIDE.md`](SANDBOX_MODE_GUIDE.md)

## 📋 Prerequisites

### 1. Test Data Setup

Create a small test dataset in `sandbox/test_content/`:

```bash
# Create test directories
mkdir -p sandbox/test_content/batch1

# Copy a small set of test images (10-20 groups recommended)
# Example: Copy from an existing project or create test images
cp ../mojo3/*.png sandbox/test_content/batch1/
# OR generate test images programmatically
```

**Recommended:** 10-20 image groups (~30-60 PNG files total) for quick testing.

### 2. Virtual Environment

Activate your Python virtual environment:

```bash
source .venv311/bin/activate
```

## 🚀 Complete Pipeline Test

### Step 1: Start a Test Project

```bash
python scripts/00_start_project.py \
  --sandbox \
  --project-id TEST-pipeline-demo \
  --content-dir sandbox/test_content/batch1 \
  --title "Pipeline Test Demo"
```

**What this does:**
- Creates `data/projects/sandbox/TEST-pipeline-demo.project.json`
- Counts initial images
- Sets UTC timestamp
- Logs to `data/file_operations_logs/sandbox/`

**Expected output:**
```
===========================================================
🧪 SANDBOX MODE - Test data will be isolated
   Projects: data/projects/sandbox
   Logs: data/file_operations_logs/sandbox
===========================================================

✅ SUCCESS!
===========================================================
Manifest created: data/projects/sandbox/TEST-pipeline-demo.project.json
Project ID:       TEST-pipeline-demo
Initial Images:   45
Started At:       2025-11-03T12:30:00Z
```

### Step 2: AI-Assisted Image Review

Review and select images using the AI-assisted reviewer:

```bash
python scripts/01_ai_assisted_reviewer.py sandbox/test_content/batch1 \
  --batch-size 20 \
  --port 8081
```

**What this does:**
- Groups images by timestamp
- Shows AI recommendations (rule-based or ML model if available)
- Lets you approve, override, or reject selections
- Moves selected images to `__selected/`
- Moves rejects to `__delete_staging/`
- Logs all decisions for training data

**Actions:**
- Press `A` to approve AI recommendation
- Press `1-4` to override with specific image
- Press `R` to reject entire group
- Press `C` for manual crop (moves to `__crop/`)

**Expected result:** Selected images in `__selected/`, rejects in `__delete_staging/`

### Step 3: Character Sorter (Optional)

If you need to sort images by character/category:

```bash
python scripts/03_web_character_sorter.py __selected \
  --port 8082
```

**What this does:**
- Shows images for categorization
- Moves to character-specific subdirectories
- Useful for multi-character projects

**Expected result:** Images organized in `__selected/char1/`, `__selected/char2/`, etc.

### Step 4: Crop Tool (If Needed)

For images that need cropping:

```bash
python scripts/02_ai_desktop_multi_crop.py __crop
```

**What this does:**
- Shows images with crop UI
- AI suggests crop coordinates (if model available)
- Saves cropped versions
- Logs crop decisions for training

**Expected result:** Cropped images ready for delivery

### Step 5: Duplicate Finder

Check for duplicates before final delivery:

```bash
python scripts/06_web_duplicate_finder.py __selected \
  --port 8083
```

**What this does:**
- Finds visually similar images
- Helps identify duplicate work
- Shows side-by-side comparisons

**Expected result:** Duplicate groups identified and resolved

### Step 6: Finish Project

When all processing is complete:

```bash
# Preview what will happen (safe - doesn't modify anything)
python scripts/07_finish_project.py --project-id TEST-pipeline-demo

# If preview looks good, commit the changes
python scripts/07_finish_project.py --project-id TEST-pipeline-demo --commit
```

**What this does:**
- Counts final images
- Creates delivery ZIP file
- Updates project manifest with completion timestamp
- Marks project as "finished"

**Expected output:**
```
======================================================================
✅ SUCCESS!
======================================================================
Manifest updated:  data/projects/sandbox/TEST-pipeline-demo.project.json
Project ID:        TEST-pipeline-demo
Finished At:       2025-11-03T14:45:00Z
Final Images:      32
Output ZIP:        exports/TEST-pipeline-demo_final.zip
ZIP Contents:      98 files

🎯 Next steps:
   • Review: exports/TEST-pipeline-demo_final.zip
   • Clean up: python scripts/tools/cleanup_sandbox.py --force
======================================================================
```

## 🧹 Cleanup After Testing

### View Sandbox Data

See what test data exists:

```bash
python scripts/tools/cleanup_sandbox.py --list
```

**Output:**
```
📋 Sandbox Directories:

📁 data/projects/sandbox
   Files: 3
   Size: 0.12 MB
   Status: ✅ Marked

📁 data/file_operations_logs/sandbox
   Files: 5
   Size: 0.05 MB
   Status: ✅ Marked

Total: 8 files, 0.17 MB
```

### Clean Up Test Data

Remove all sandbox data:

```bash
# Preview what will be deleted (safe)
python scripts/tools/cleanup_sandbox.py --dry-run

# Delete sandbox data (asks for confirmation)
python scripts/tools/cleanup_sandbox.py

# Force delete without confirmation
python scripts/tools/cleanup_sandbox.py --force
```

**What gets deleted:**
- `data/projects/sandbox/` - All test project manifests
- `data/file_operations_logs/sandbox/` - All test operation logs
- Any other directories marked with `.sandbox_marker`

**What's preserved:**
- Production project manifests in `data/projects/`
- Production logs in `data/file_operations_logs/`
- All image files (we never delete images without explicit action)

## 📊 Complete Test Script

Here's a complete bash script for automated pipeline testing:

```bash
#!/usr/bin/env bash
# test_pipeline.sh - Complete pipeline test with sandbox mode

set -e  # Exit on error

PROJECT_ID="TEST-pipeline-$(date +%Y%m%d-%H%M%S)"
TEST_CONTENT="sandbox/test_content/batch1"

echo "🧪 Starting pipeline test: $PROJECT_ID"

# Step 1: Start project
echo "▶️  Step 1: Creating test project..."
python scripts/00_start_project.py \
  --sandbox \
  --project-id "$PROJECT_ID" \
  --content-dir "$TEST_CONTENT" \
  --title "Automated Pipeline Test"

# Step 2: Run AI reviewer (web UI - requires manual interaction)
echo "▶️  Step 2: Running AI-assisted reviewer..."
echo "   Manual step: Review images at http://localhost:8081"
echo "   Press Ctrl+C when done, then run this script again for next steps"
python scripts/01_ai_assisted_reviewer.py "$TEST_CONTENT" --port 8081

# Step 3: Character sorter (optional - comment out if not needed)
# echo "▶️  Step 3: Running character sorter..."
# python scripts/03_web_character_sorter.py __selected --port 8082

# Step 4: Duplicate finder
echo "▶️  Step 4: Running duplicate finder..."
python scripts/06_web_duplicate_finder.py __selected --port 8083

# Step 5: Finish project
echo "▶️  Step 5: Finishing project..."
python scripts/07_finish_project.py --project-id "$PROJECT_ID" --commit

echo "✅ Pipeline test complete!"
echo "📦 Output ZIP: exports/${PROJECT_ID}_final.zip"
echo ""
echo "🧹 To clean up test data:"
echo "   python scripts/tools/cleanup_sandbox.py --force"
```

## 🐛 Troubleshooting

### "Invalid project ID for sandbox mode"

**Problem:** Project ID doesn't start with `TEST-`

**Solution:** Always use `TEST-` prefix when using `--sandbox` flag:
```bash
# Wrong
--sandbox --project-id demo

# Right
--sandbox --project-id TEST-demo
```

### Port Already in Use

**Problem:** `Address already in use` error when starting web tools

**Solution:** Either kill the existing process or use a different port:
```bash
# Check what's using port 8081
lsof -i :8081

# Use a different port
python scripts/01_ai_assisted_reviewer.py ... --port 8082
```

### No Test Images Found

**Problem:** "No PNG images found" error

**Solution:** Verify test images exist:
```bash
ls -l sandbox/test_content/batch1/*.png | wc -l
```

If no images, copy some test data:
```bash
# Copy from existing project
cp ../mojo3/*.png sandbox/test_content/batch1/

# Or create test directory structure
mkdir -p sandbox/test_content/batch1
```

### Sandbox Cleanup Won't Delete

**Problem:** "Directory missing .sandbox_marker file"

**Solution:** This is a safety feature. The cleanup tool only deletes directories explicitly marked as sandbox. To fix:

```bash
# Verify it's a sandbox directory first!
ls -la data/projects/sandbox/

# If it's truly a sandbox dir, the marker should exist
# If not, SandboxConfig should have created it
# Re-run any sandbox command to create markers:
python scripts/examples/demo_with_sandbox.py
```

## 📈 Testing Best Practices

### 1. Test with Small Datasets

Start with 10-20 image groups (~30-60 files). This is enough to test the full pipeline without taking too long.

### 2. Test Each Tool Independently

Before running the full pipeline, test each tool individually:
- Image reviewer: Works with your image format?
- Character sorter: UI displays correctly?
- Crop tool: Coordinates save properly?
- Duplicate finder: Finds similar images?

### 3. Verify Output Quality

After each step, manually check:
- Images moved to correct directories
- File counts match expectations
- No images left behind
- Companion files moved with images

### 4. Check Logs

Review operation logs for errors:
```bash
# Sandbox logs
tail -f data/file_operations_logs/sandbox/file_operations.log

# Production logs (should be empty during sandbox testing)
tail -f data/file_operations_logs/file_operations.log
```

### 5. Clean Up Between Tests

Always clean up after each test to avoid confusion:
```bash
python scripts/tools/cleanup_sandbox.py --force
```

## 🎯 What to Test

### Critical Path Tests
- [ ] Start project creates valid manifest
- [ ] AI reviewer selects and moves images correctly
- [ ] Selected images appear in `__selected/`
- [ ] Rejected images appear in `__delete_staging/`
- [ ] Finish project creates ZIP file
- [ ] Finish project updates manifest correctly
- [ ] Sandbox cleanup removes all test data

### Edge Cases
- [ ] Empty image directory
- [ ] Single image group
- [ ] 100+ image groups (stress test)
- [ ] Images with special characters in filenames
- [ ] Missing companion files (YAML, captions)
- [ ] Duplicate image groups

### Integration Tests
- [ ] Full pipeline with all tools
- [ ] Multiple test projects simultaneously
- [ ] Sandbox + production projects side-by-side (verify isolation)
- [ ] Error recovery (what happens if tool crashes mid-process?)

## 📚 Related Documentation

- [`SANDBOX_MODE_GUIDE.md`](SANDBOX_MODE_GUIDE.md) - Detailed sandbox documentation
- [`../core/PROJECT_LIFECYCLE_SCRIPTS.md`](../core/PROJECT_LIFECYCLE_SCRIPTS.md) - Individual script documentation
- [`TESTS_GUIDE.md`](TESTS_GUIDE.md) - Automated test suite documentation
- [`../../.cursorrules`](../../.cursorrules) - File safety rules and conventions

## 💡 Tips for Efficient Testing

1. **Use tmux/screen** for long-running tests - prevents interruption if terminal closes
2. **Create test data once** - reuse the same test images for multiple runs
3. **Automate where possible** - use scripts for repetitive testing
4. **Test incrementally** - verify each step before moving to next
5. **Document failures** - note what broke and how you fixed it

## ✅ Success Criteria

Your pipeline test is successful if:

- ✅ All images processed (source directory empty)
- ✅ Selected images in correct output directories
- ✅ Project manifest updated with correct counts
- ✅ Delivery ZIP created and contains expected files
- ✅ All file operations logged
- ✅ No errors in operation logs
- ✅ Cleanup removes all sandbox data
- ✅ Production data unchanged

## 🚀 Next Steps

After successful pipeline testing:

1. **Run with larger datasets** - test scalability
2. **Add error injection** - test recovery mechanisms
3. **Benchmark performance** - measure throughput
4. **Automate testing** - create CI/CD pipeline
5. **Document findings** - update this guide with learnings

---

**Remember:** Always use sandbox mode for testing! It's designed to keep your production data safe while giving you confidence in your workflow.

