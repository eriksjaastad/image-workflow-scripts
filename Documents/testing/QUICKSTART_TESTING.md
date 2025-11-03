# Quick Start: Testing the Full Pipeline

**TL;DR:** Run a test project through all your tools safely using sandbox mode.

## 🚀 Quick Test (5 minutes)

```bash
# 1. Create test data directory with a few images
mkdir -p sandbox/test_content/batch1
cp ../mojo3/*.png sandbox/test_content/batch1/  # or any test images

# 2. Start test project
python scripts/00_start_project.py \
  --sandbox \
  --project-id TEST-demo \
  --content-dir sandbox/test_content/batch1

# 3. Run image reviewer
python scripts/01_ai_assisted_reviewer.py sandbox/test_content/batch1

# 4. Finish project
python scripts/07_finish_project.py --project-id TEST-demo --commit

# 5. Clean up everything
python scripts/tools/cleanup_sandbox.py --force
```

## 📖 Full Documentation

See [`TEST_PROJECT_GUIDE.md`](TEST_PROJECT_GUIDE.md) for:
- Complete pipeline walkthrough
- All workflow tools
- Troubleshooting tips
- Best practices

## 🧹 Clean Up Test Data

```bash
# See what test data exists
python scripts/tools/cleanup_sandbox.py --list

# Delete all test data
python scripts/tools/cleanup_sandbox.py --force
```

## 📚 Related Docs

- [`TEST_PROJECT_GUIDE.md`](TEST_PROJECT_GUIDE.md) - Complete pipeline testing guide
- [`SANDBOX_MODE_GUIDE.md`](SANDBOX_MODE_GUIDE.md) - Sandbox mode details
- [`../core/PROJECT_LIFECYCLE_SCRIPTS.md`](../core/PROJECT_LIFECYCLE_SCRIPTS.md) - Individual script docs

