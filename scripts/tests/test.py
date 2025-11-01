#!/usr/bin/env python3
"""
Test Runner Entry Point

This script provides a convenient way to run the image processing workflow tests
from the project root without navigating to the scripts/tests directory.

Usage:
    python scripts/tests/test.py                    # Run all tests
    python scripts/tests/test.py --safety-only      # Run only critical safety tests
    python scripts/tests/test.py --performance      # Include performance tests
    python scripts/tests/test.py --create-data      # Create test data first
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Change to project root directory (go up from scripts/tests/ to root)
    project_root = Path(__file__).parent.parent.parent

    # Call the actual test runner (we're already in scripts/tests/)
    test_runner_path = Path(__file__).parent / "test_runner.py"

    if not test_runner_path.exists():
        print(f"❌ Test runner not found at: {test_runner_path}")
        sys.exit(1)

    # Pass all arguments to the actual test runner
    cmd = [sys.executable, str(test_runner_path)] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n🚨 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
