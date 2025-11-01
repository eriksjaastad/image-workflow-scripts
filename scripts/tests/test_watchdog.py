#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Ensure project root on sys.path for "scripts.*" imports when run directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.watchdog import Heartbeat, Watchdog


def test_heartbeat_progress_and_watchdog_timeout(tmp_path: Path = None):
    hb = Heartbeat()
    aborted = {"reason": None}

    def on_abort(reason: str):
        aborted["reason"] = reason

    run_id = "test_run"
    sandbox_root = Path("sandbox/mojo2")
    wd = Watchdog(
        hb,
        start_time_utc=time.time(),
        max_runtime_sec=2.0,
        stall_threshold_sec=1.0,
        poll_interval_sec=0.1,
        on_abort=on_abort,
        sandbox_root=sandbox_root,
        run_id=run_id,
        write_stack=False,
    )
    wd.start()

    # Progress should keep it alive for ~1s
    hb.update(files_scanned=1, notes="tick1")
    time.sleep(0.5)
    hb.update(files_scanned=2, notes="tick2")
    time.sleep(0.7)  # surpass stall threshold if no update

    # Now stop updating; expect abort due to stall
    # Give it time to detect
    time.sleep(1.2)
    wd.stop()

    assert aborted["reason"] in {"no_progress_stall", "max_runtime_exceeded"}
    # Error file should exist when sandbox is present
    err = sandbox_root / "logs" / f"error_{run_id}.json"
    assert err.exists()
    return True


if __name__ == "__main__":
    ok = test_heartbeat_progress_and_watchdog_timeout()
    print("OK" if ok else "FAIL")
