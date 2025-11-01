#!/usr/bin/env python3
"""
Enhanced Code Coverage Test Runner
Runs all tests (unit + Selenium) with coverage analysis and generates a beautiful combined report.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
  python scripts/tests/run_coverage.py

FEATURES:
---------
- Runs unit tests with code coverage
- Runs Selenium integration tests
- Combines results into one beautiful HTML report
- Styled with dark theme from WEB_STYLE_GUIDE.md
- Shows all test results in one place!
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run enhanced coverage with Selenium results."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent

    print("=" * 70)
    print("🚀 Enhanced Test Coverage Runner")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Run unit tests with coverage")
    print("  2. Run Selenium integration tests")
    print("  3. Generate beautiful combined HTML report")
    print()

    # Run the enhanced coverage generator
    enhanced_script = (
        project_root / "scripts" / "tests" / "generate_enhanced_coverage.py"
    )

    result = subprocess.run(
        [sys.executable, str(enhanced_script)], cwd=project_root, check=False
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
