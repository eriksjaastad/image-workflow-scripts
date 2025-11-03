#!/usr/bin/env python3
"""Demo: Using Sandbox Mode for Safe Testing

This example demonstrates how to use SandboxConfig to safely test
workflow operations without polluting production data.

Key Features Demonstrated:
- Creating a SandboxConfig instance
- Validating and formatting project IDs
- Using sandbox paths for file operations
- Integrating with FileTracker
- Cleaning up test data

Usage:
    python scripts/examples/demo_with_sandbox.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.file_tracker import FileTracker
from scripts.utils.sandbox_mode import SandboxConfig


def demo_sandbox_basics():
    """Demonstrate basic SandboxConfig usage."""
    print("=" * 60)
    print("DEMO 1: SandboxConfig Basics")
    print("=" * 60)

    # Create sandbox configuration
    sandbox = SandboxConfig(enabled=True)
    sandbox.print_banner()

    # Show configured paths
    print("\nConfigured paths:")
    print(f"  Projects: {sandbox.projects_dir}")
    print(f"  Logs: {sandbox.logs_dir}")

    # Demonstrate project ID validation
    print("\nProject ID validation:")
    test_ids = ["TEST-demo", "production-project", "test-lowercase"]
    for pid in test_ids:
        is_valid = sandbox.validate_project_id(pid)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {pid}: {status}")

    # Demonstrate project ID formatting
    print("\nProject ID formatting:")
    print(f"  format_project_id('demo') → {sandbox.format_project_id('demo')}")
    print(
        f"  format_project_id('TEST-demo') → {sandbox.format_project_id('TEST-demo')}"
    )


def demo_file_operations():
    """Demonstrate file operations in sandbox mode."""
    print("\n" + "=" * 60)
    print("DEMO 2: File Operations in Sandbox")
    print("=" * 60)

    sandbox = SandboxConfig(enabled=True)

    # Example: Create a test project manifest
    project_id = "TEST-demo-example"
    manifest_path = sandbox.projects_dir / f"{project_id}.project.json"

    print("\nCreating test manifest:")
    print(f"  Path: {manifest_path}")

    # Write a simple test manifest
    test_manifest = {
        "projectId": project_id,
        "status": "test",
        "note": "This is sandbox test data - safe to delete",
    }

    import json

    manifest_path.write_text(json.dumps(test_manifest, indent=2), encoding="utf-8")
    print(f"  ✅ Created: {manifest_path.name}")

    # Verify marker file exists
    marker_exists = SandboxConfig.has_marker_file(sandbox.projects_dir)
    print("\nSandbox marker file:")
    print(f"  Exists: {marker_exists}")
    if marker_exists:
        marker_path = sandbox.projects_dir / ".sandbox_marker"
        print(f"  Path: {marker_path}")


def demo_filetracker_integration():
    """Demonstrate FileTracker integration with sandbox."""
    print("\n" + "=" * 60)
    print("DEMO 3: FileTracker Integration")
    print("=" * 60)

    sandbox = SandboxConfig(enabled=True)

    # Create FileTracker with sandbox configuration
    tracker = FileTracker(script_name="demo_sandbox", sandbox_config=sandbox)

    print("\nFileTracker configured:")
    print(f"  Log file: {tracker.log_file}")
    print(f"  Sandbox mode: {tracker.sandbox}")

    # Log a test operation
    tracker.log_operation(
        operation="test",
        source_dir="demo_source",
        dest_dir="demo_dest",
        file_count=5,
        notes="This is a sandbox test operation",
    )

    print("\n✅ Logged test operation to sandbox logs")


def demo_cleanup():
    """Demonstrate cleanup process."""
    print("\n" + "=" * 60)
    print("DEMO 4: Cleanup Process")
    print("=" * 60)

    print("\nTo clean up all sandbox data, run:")
    print("  python scripts/tools/cleanup_sandbox.py --dry-run")
    print("  python scripts/tools/cleanup_sandbox.py --force")

    print("\nThe cleanup utility will:")
    print("  1. Verify .sandbox_marker files exist (safety check)")
    print("  2. Show file counts and sizes")
    print("  3. Request confirmation (unless --force)")
    print("  4. Delete only marked sandbox directories")


def main():
    """Run all demos."""
    print("\n🧪 SANDBOX MODE DEMONSTRATION")
    print("This script demonstrates safe testing patterns\n")

    try:
        demo_sandbox_basics()
        demo_file_operations()
        demo_filetracker_integration()
        demo_cleanup()

        print("\n" + "=" * 60)
        print("✅ Demo Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  1. Always use SandboxConfig(enabled=True) for testing")
        print("  2. Test project IDs must start with TEST- prefix")
        print("  3. FileTracker automatically uses sandbox logs")
        print("  4. .sandbox_marker files prevent accidental deletion")
        print("  5. Use cleanup_sandbox.py to remove test data safely")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
