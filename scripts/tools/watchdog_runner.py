#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def kill_process_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        # Give it a brief grace period
        try:
            p.wait(timeout=2)
        except Exception:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass


def tail_heartbeat(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def run_with_watchdog(cmd: str, timeout_sec: float, heartbeat_file: Path, hb_threshold: float) -> dict:
    started_at = time.time()
    started_iso = utc_now_iso()
    log_paths = {}
    status = "OK"
    reason = None
    exit_code = None

    # Launch child in new process group so we can terminate the entire tree
    p = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_hb = tail_heartbeat(heartbeat_file) if heartbeat_file else 0.0
    time.time()

    # Stream output to temp files under the same directory as heartbeat
    out_dir = heartbeat_file.parent if heartbeat_file else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
        while True:
            # Poll for new output
            if p.stdout:
                line = p.stdout.readline()
                if line:
                    f_out.write(line)
                    f_out.flush()
                    time.time()
                else:
                    # No more stdout; break if process ended
                    if p.poll() is not None:
                        pass
            if p.stderr:
                line_err = p.stderr.readline()
                if line_err:
                    f_err.write(line_err)
                    f_err.flush()
                    time.time()

            # Check process completion
            if p.poll() is not None:
                exit_code = p.returncode
                break

            now = time.time()
            # Timeout enforcement
            if timeout_sec and now - started_at > timeout_sec:
                status = "ABORT"
                reason = "ABORT (timeout)"
                kill_process_group(p)
                exit_code = p.returncode
                break

            # Heartbeat stall detection
            if heartbeat_file:
                current_hb = tail_heartbeat(heartbeat_file)
                if current_hb > 0:
                    last_hb = max(last_hb, current_hb)
                if last_hb > 0 and (now - last_hb) > hb_threshold:
                    status = "ABORT"
                    reason = "ABORT (stalled)"
                    kill_process_group(p)
                    exit_code = p.returncode
                    break

            time.sleep(0.2)

    ended_at = time.time()
    ended_iso = utc_now_iso()
    duration = ended_at - started_at
    log_paths = {"stdout": str(stdout_path), "stderr": str(stderr_path)}

    result = {
        "status": status,
        "reason": reason,
        "exit_code": exit_code,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "duration_sec": round(duration, 3),
        "log_paths": log_paths,
        "cmd": cmd,
    }

    # Write result.json alongside heartbeat
    result_path = (heartbeat_file.parent if heartbeat_file else Path.cwd()) / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Watchdog runner for long tasks")
    ap.add_argument("--cmd", required=True, help="Shell command to execute")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--heartbeat-file", required=False, default="", help="Path to heartbeat file to monitor")
    ap.add_argument("--hb-threshold", type=float, default=120.0)
    args = ap.parse_args()

    hb_file = Path(args.heartbeat_file) if args.heartbeat_file else None
    if hb_file:
        hb_file.parent.mkdir(parents=True, exist_ok=True)
        # Touch initial heartbeat file if not exists
        try:
            hb_file.touch(exist_ok=True)
        except Exception:
            pass

    run_with_watchdog(args.cmd, args.timeout, hb_file, args.hb_threshold)


if __name__ == "__main__":
    main()


