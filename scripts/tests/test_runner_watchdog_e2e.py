#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_reducer_watchdog_simulated_hang():
    cmd = [
        sys.executable,
        str(ROOT / "scripts/tools/reducer.py"),
        "run",
        "--variant", "A",
        "--profile", "conservative",
        "--dry-run",
        "--sandbox-root", "sandbox/mojo2",
        "--simulate-hang",
        "--max-runtime", "3",
        "--watchdog-threshold", "1",
        "--progress-interval", "0.5",
        "--no-stack-dump",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Expect non-zero exit due to abort
    assert result.returncode != 0, f"Expected abort non-zero exit, got 0. stdout={result.stdout} stderr={result.stderr}"
    # Expect ABORT line in stdout or stderr
    combined = (result.stdout or "") + (result.stderr or "")
    assert "ABORT" in combined, f"No ABORT message. stdout={result.stdout} stderr={result.stderr}"
    print("✓ reducer watchdog simulated hang aborts as expected")


if __name__ == "__main__":
    test_reducer_watchdog_simulated_hang()
    print("OK")
