"""Sandbox mode for safe testing without polluting production data.

Enables test mode that redirects file writes to timestamped sandbox directories
under data/test_runs/<timestamp> instead of production directories. This allows
testing workflow scripts with real operations without risk of corrupting:
- Project manifests (data/projects/)
- Operation logs (data/file_operations_logs/)
- Production image files

Usage:
    from scripts.utils.sandbox_mode import test_sandbox, in_test_mode

    # Context manager approach
    with test_sandbox(enabled=True, cleanup=False) as sandbox_root:
        # All writes should check in_test_mode() and use rebase_path()
        if in_test_mode():
            print(f"Running in sandbox: {sandbox_root}")

    # Check if currently in test mode
    if in_test_mode():
        # Adjust file paths accordingly
        pass

Note:
    This is a simple implementation using environment variables. Scripts must
    explicitly check in_test_mode() and use rebase_path() to redirect writes.
    Future enhancement: Integrate with FileTracker for automatic redirection.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def sandbox_root() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/test_runs") / ts


@contextmanager
def test_sandbox(enabled: bool, cleanup: bool = False):
    """Redirect writes under data/test_runs/<ts> and optionally clean up."""
    root = None
    try:
        if enabled:
            os.environ["APP_MODE"] = "TEST"
            root = sandbox_root()
            root.mkdir(parents=True, exist_ok=True)
            print(f"🧪 TEST MODE — sandbox: {root}")
        yield root  # may be None if disabled
    finally:
        if enabled and cleanup and root and root.exists():
            shutil.rmtree(root, ignore_errors=True)
            print(f"🧹 cleaned up sandbox: {root}")


def in_test_mode() -> bool:
    return os.environ.get("APP_MODE") == "TEST"


def rebase_path(write_path: Path, sandbox_base: Path | None) -> Path:
    """If in test mode, rebase an intended write path under the sandbox."""
    if sandbox_base is None or not in_test_mode():
        return write_path
    rel = write_path.as_posix().lstrip("/")
    return sandbox_base / rel


def safe_write_text(
    path: Path, content: str, *, dry_run: bool, sandbox_base: Path | None
):
    target = rebase_path(path, sandbox_base)
    if dry_run:
        print(f"[DRY-RUN] would write: {target} ({len(content)} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"📝 wrote: {target}")
