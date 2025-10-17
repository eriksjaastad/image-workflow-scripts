#!/usr/bin/env python3
"""
Run desktop selector behavior regression tests (headless).
Usage: python scripts/tests/run_desktop_regression.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    tests = [
        PROJECT_ROOT / "scripts/tests/test_desktop_selector_behavior.py",
    ]
    all_ok = True
    for t in tests:
        print(f"[*] Running {t}")
        proc = subprocess.run([sys.executable, str(t)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            all_ok = False
        else:
            # silent tests: just confirm success
            print("  ✅ Passed")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()


