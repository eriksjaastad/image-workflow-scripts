# Sandbox Mode Guide

**Last Updated:** 2025-11-03
**Status:** Active
**Audience:** Developers
**Estimated Reading Time:** 10 minutes

## Overview

Sandbox mode provides a safe, isolated testing environment for workflow scripts without polluting production data. When enabled, all project manifests, logs, and test data are stored in separate directories that can be safely deleted.

## Why Use Sandbox Mode?

- **Data Safety**: Test scripts without risking production project manifests or logs
- **Easy Cleanup**: Delete all test data with a single command
- **Isolated Testing**: Projects created in sandbox mode are completely separate from production
- **Development Safety**: Experiment with new features without fear of breaking production data

## Quick Start

### 1. Using Workflow Scripts with Sandbox Mode

Most workflow scripts support the `--sandbox` flag:

```bash
# Create a test project in sandbox mode
python scripts/00_start_project.py --sandbox --project-id TEST-demo --content-dir data/test_content/batch1/

# Project manifest will be created in:
# data/projects/sandbox/TEST-demo.project.json
```

### 2. Project ID Requirements

Sandbox mode enforces strict project ID validation:

- **Sandbox mode**: Project IDs MUST start with `TEST-` (case-sensitive)
- **Production mode**: Project IDs MUST NOT start with `TEST-`

```bash
# ✅ Valid in sandbox mode
--sandbox --project-id TEST-demo
--sandbox --project-id TEST-mojo3

# ❌ Invalid in sandbox mode (missing TEST- prefix)
--sandbox --project-id demo
--sandbox --project-id test-demo  # lowercase 'test' doesn't count!

# ✅ Valid in production mode
--project-id mojo3
--project-id spring-batch-2024

# ❌ Invalid in production mode (has TEST- prefix)
--project-id TEST-demo
```

### 3. Viewing Sandbox Data

List all sandbox directories and their sizes:

```bash
python scripts/tools/cleanup_sandbox.py --list
```

Output:
```
📋 Sandbox Directories:

📁 /path/to/data/projects/sandbox
   Files: 5
   Size: 0.15 MB
   Status: ✅ Marked

📁 /path/to/data/file_operations_logs/sandbox
   Files: 2
   Size: 0.01 MB
   Status: ✅ Marked

Total: 7 files, 0.16 MB
```

### 4. Cleaning Up Test Data

When you're done testing, clean up all sandbox data:

```bash
# See what would be deleted (safe - doesn't delete anything)
python scripts/tools/cleanup_sandbox.py --dry-run

# Interactive cleanup (asks for confirmation)
python scripts/tools/cleanup_sandbox.py

# Force cleanup without confirmation (use carefully!)
python scripts/tools/cleanup_sandbox.py --force
```

## For Developers: Using SandboxConfig in Your Scripts

### Basic Usage

```python
from scripts.utils.sandbox_mode import SandboxConfig

# Create sandbox configuration
sandbox = SandboxConfig(enabled=args.sandbox)

# Show banner if enabled
sandbox.print_banner()

# Use sandbox-aware paths
project_file = sandbox.projects_dir / f"{project_id}.project.json"
log_dir = sandbox.logs_dir
```

### Project ID Validation

```python
# Validate project ID before processing
if not sandbox.validate_project_id(project_id):
    mode = "sandbox" if sandbox.enabled else "production"
    prefix_msg = "must start with TEST-" if sandbox.enabled else "must NOT start with TEST-"
    print(f"Error: Invalid project ID for {mode} mode: {project_id} {prefix_msg}")
    sys.exit(1)
```

### Project ID Formatting

```python
# Auto-format project IDs (adds TEST- prefix if in sandbox mode)
formatted_id = sandbox.format_project_id("demo")
# In sandbox mode: "TEST-demo"
# In production mode: "demo"

# Already has prefix - left unchanged
formatted_id = sandbox.format_project_id("TEST-demo")
# Returns: "TEST-demo" (regardless of mode)
```

### FileTracker Integration

```python
from scripts.file_tracker import FileTracker
from scripts.utils.sandbox_mode import SandboxConfig

# Create sandbox config
sandbox = SandboxConfig(enabled=True)

# FileTracker automatically uses sandbox logs
tracker = FileTracker(
    script_name="my_script",
    sandbox_config=sandbox
)

# Logs will be written to:
# data/file_operations_logs/sandbox/file_operations.log
```

### Complete Example

```python
import argparse
from pathlib import Path
from scripts.utils.sandbox_mode import SandboxConfig
from scripts.file_tracker import FileTracker

def main():
    parser = argparse.ArgumentParser(description="My workflow script")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--sandbox", action="store_true",
                       help="Run in sandbox mode (requires TEST- prefix)")
    args = parser.parse_args()

    # Configure sandbox
    sandbox = SandboxConfig(enabled=args.sandbox)
    sandbox.print_banner()

    # Validate project ID
    if not sandbox.validate_project_id(args.project_id):
        print(f"Error: Project ID '{args.project_id}' invalid for sandbox mode")
        return 1

    # Use sandbox paths
    manifest_path = sandbox.projects_dir / f"{args.project_id}.project.json"

    # Create FileTracker with sandbox support
    tracker = FileTracker(
        script_name="my_script",
        sandbox_config=sandbox
    )

    # ... rest of script logic ...

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Sandbox Directory Structure

When sandbox mode is enabled, the following directories are used:

```
data/
├── projects/
│   ├── sandbox/           # Sandbox project manifests
│   │   ├── .sandbox_marker  # Safety marker (auto-created)
│   │   ├── TEST-demo.project.json
│   │   └── TEST-mojo3.project.json
│   └── mojo3.project.json   # Production manifests (separate)
│
└── file_operations_logs/
    ├── sandbox/           # Sandbox operation logs
    │   ├── .sandbox_marker  # Safety marker (auto-created)
    │   └── file_operations.log
    └── file_operations.log  # Production logs (separate)
```

## Safety Features

### Marker Files

Sandbox directories are automatically tagged with `.sandbox_marker` files:

- Created automatically when `SandboxConfig(enabled=True)` initializes directories
- Prevents accidental deletion of non-sandbox directories
- `cleanup_sandbox.py` verifies markers before deleting anything

If cleanup finds directories without markers:
```
⚠️  ERROR: Some directories are missing .sandbox_marker files:
   - data/projects/sandbox

These directories will NOT be deleted (safety check).
If you're sure these are sandbox directories, add .sandbox_marker files manually.
```

### Project ID Enforcement

The TEST- prefix requirement prevents accidentally:
- Creating test projects in production mode
- Creating production projects in sandbox mode
- Mixing test and production data

## Examples and Demos

A comprehensive demonstration script is available:

```bash
python scripts/examples/demo_with_sandbox.py
```

This demo shows:
1. SandboxConfig basics and configuration
2. File operations in sandbox mode
3. FileTracker integration
4. Cleanup process

## Common Workflows

### Test a New Feature

```bash
# 1. Create test project
python scripts/00_start_project.py --sandbox --project-id TEST-feature-test --content-dir data/test_content/batch1/

# 2. Run your feature with the test project
# ... run various workflow scripts ...

# 3. Review test results

# 4. Clean up when done
python scripts/tools/cleanup_sandbox.py --force
```

### Automated Testing

```bash
# Create test batch
python scripts/tools/enqueue_test_batch.py --sandbox --batch-size 10

# Run tests
./run_tests.sh --sandbox

# Clean up automatically
python scripts/tools/cleanup_sandbox.py --force
```

### Development Iteration

```bash
# Create test environment
python scripts/00_start_project.py --sandbox --project-id TEST-dev --content-dir data/test_content/batch1/

# Iterate on your code
# ... make changes ...
# ... test changes ...
# ... repeat ...

# Check what test data exists
python scripts/tools/cleanup_sandbox.py --list

# Clean up when switching tasks
python scripts/tools/cleanup_sandbox.py --force
```

## Backward Compatibility

Legacy sandbox functions are still available but deprecated:

```python
# Old way (still works)
from scripts.utils.sandbox_mode import test_sandbox, in_test_mode

with test_sandbox(enabled=True, cleanup=False) as sandbox_root:
    if in_test_mode():
        print(f"Running in sandbox: {sandbox_root}")

# New way (recommended)
from scripts.utils.sandbox_mode import SandboxConfig

sandbox = SandboxConfig(enabled=True)
print(f"Projects dir: {sandbox.projects_dir}")
```

## Troubleshooting

### "Invalid project ID for sandbox mode"

**Problem**: Project ID doesn't have TEST- prefix in sandbox mode

**Solution**: Add TEST- prefix to your project ID:
```bash
# Wrong
--sandbox --project-id demo

# Right
--sandbox --project-id TEST-demo
```

### "Directory missing .sandbox_marker file"

**Problem**: Cleanup refuses to delete a directory

**Solution**: This is a safety feature. Either:
1. Manually verify the directory is safe to delete
2. Create `.sandbox_marker` file if it's truly a sandbox directory
3. Delete the directory manually if you're certain

### Sandbox data persists after cleanup

**Problem**: Some files remain after cleanup

**Solution**: Check if directories without markers were skipped. Use `--dry-run` first:
```bash
python scripts/tools/cleanup_sandbox.py --dry-run
```

## Best Practices

1. **Always use TEST- prefix** for sandbox project IDs (case-sensitive)
2. **Clean up regularly** to avoid accumulating stale test data
3. **Use --dry-run first** when cleaning up to see what will be deleted
4. **Check --list** to see current sandbox data usage
5. **Never manually create .sandbox_marker files** in production directories
6. **Use sandbox mode for all development** and testing workflows

## Related Documentation

- `scripts/utils/sandbox_mode.py` - Source code and implementation details
- `scripts/tools/cleanup_sandbox.py` - Cleanup utility documentation
- `scripts/examples/demo_with_sandbox.py` - Working examples
- `testing/TESTS_GUIDE.md` - General testing guide
- `core/PROJECT_LIFECYCLE_SCRIPTS.md` - Workflow scripts overview

## Summary

Sandbox mode provides a safe, isolated testing environment with:
- Automatic directory isolation
- Project ID validation (TEST- prefix)
- Safety markers to prevent accidental deletion
- Easy cleanup utilities
- FileTracker integration

Use it for all development and testing to keep production data safe!
