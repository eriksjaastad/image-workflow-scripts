"""Sandbox mode for safe testing without polluting production data.

Enables test mode that redirects file writes to isolated sandbox directories
instead of production directories. This allows testing workflow scripts with
real operations without risk of corrupting:
- Project manifests (data/projects/)
- Operation logs (data/file_operations_logs/)
- Production image files

Two approaches are provided:
1. SandboxConfig class - Recommended for new code
2. Legacy functions (test_sandbox, in_test_mode) - For backward compatibility

Usage (Recommended - SandboxConfig):
    from scripts.utils.sandbox_mode import SandboxConfig

    # Create sandbox configuration
    sandbox = SandboxConfig(enabled=args.sandbox)
    sandbox.print_banner()

    # Use sandbox paths
    project_file = sandbox.projects_dir / f"{project_id}.project.json"

    # Validate project IDs
    if not sandbox.validate_project_id(project_id):
        print("Error: Sandbox mode requires TEST- prefix")
        sys.exit(1)

Usage (Legacy - test_sandbox):
    from scripts.utils.sandbox_mode import test_sandbox, in_test_mode

    with test_sandbox(enabled=True, cleanup=False) as sandbox_root:
        if in_test_mode():
            print(f"Running in sandbox: {sandbox_root}")

Note:
    Scripts must explicitly use sandbox paths. FileTracker integration
    for automatic redirection is planned for Phase 4.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class SandboxConfig:
    """Centralized sandbox configuration for safe testing.

    Manages paths and validation for sandbox mode, ensuring test data
    stays isolated from production data.

    Attributes:
        enabled: Whether sandbox mode is active
        project_prefix: Required prefix for sandbox project IDs (default: "TEST-")
        data_root: Root data directory
        projects_dir: Directory for project manifests (sandbox or production)
        logs_dir: Directory for operation logs (sandbox or production)

    Example:
        sandbox = SandboxConfig(enabled=True)
        sandbox.print_banner()

        # Use sandbox paths
        manifest_path = sandbox.projects_dir / f"{project_id}.project.json"

        # Validate project ID
        if not sandbox.validate_project_id(project_id):
            raise ValueError("Project ID must start with TEST- in sandbox mode")
    """

    def __init__(self, enabled: bool = False):
        """Initialize sandbox configuration.

        Args:
            enabled: If True, use sandbox directories and require TEST- prefix
        """
        self.enabled = enabled
        self.project_prefix = "TEST-"

        # Calculate paths
        self.data_root = Path(__file__).resolve().parents[2] / "data"

        if enabled:
            # Sandbox mode: isolated directories
            self.projects_dir = self.data_root / "projects" / "sandbox"
            self.logs_dir = self.data_root / "file_operations_logs" / "sandbox"
        else:
            # Production mode: standard directories
            self.projects_dir = self.data_root / "projects"
            self.logs_dir = self.data_root / "file_operations_logs"

        # Create sandbox directories if needed
        if enabled:
            self.projects_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def validate_project_id(self, project_id: str) -> bool:
        """Validate that project ID matches sandbox mode requirements.

        In sandbox mode, project IDs must start with TEST- prefix.
        In production mode, project IDs must NOT start with TEST- prefix.

        Args:
            project_id: Project identifier to validate

        Returns:
            True if project ID is valid for current mode

        Example:
            sandbox = SandboxConfig(enabled=True)
            sandbox.validate_project_id("TEST-demo")  # True
            sandbox.validate_project_id("mojo3")      # False
        """
        if self.enabled:
            return project_id.startswith(self.project_prefix)
        return not project_id.startswith(self.project_prefix)

    def format_project_id(self, project_id: str) -> str:
        """Add TEST- prefix if sandbox mode and not already present.

        Args:
            project_id: Base project identifier

        Returns:
            Formatted project ID with TEST- prefix if in sandbox mode

        Example:
            sandbox = SandboxConfig(enabled=True)
            sandbox.format_project_id("demo")       # "TEST-demo"
            sandbox.format_project_id("TEST-demo")  # "TEST-demo"
        """
        if self.enabled and not project_id.startswith(self.project_prefix):
            return f"{self.project_prefix}{project_id}"
        return project_id

    def print_banner(self) -> None:
        """Print warning banner for sandbox mode.

        Displays prominent message showing sandbox is active and where
        test data will be stored. Only prints if sandbox is enabled.
        """
        if self.enabled:
            print("=" * 60)  # noqa: T201
            print("🧪 SANDBOX MODE - Test data will be isolated")  # noqa: T201
            print(f"   Projects: {self.projects_dir}")  # noqa: T201
            print(f"   Logs: {self.logs_dir}")  # noqa: T201
            print("=" * 60)  # noqa: T201


# -------------------- Legacy Functions (Backward Compatibility) --------------------

def sandbox_root() -> Path:
    """Generate timestamped sandbox root path.

    Legacy function. New code should use SandboxConfig instead.

    Returns:
        Path to timestamped sandbox directory under data/test_runs/
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/test_runs") / ts


@contextmanager
def test_sandbox(enabled: bool, cleanup: bool = False):  # noqa: PT028
    """Context manager for sandbox mode using environment variables.

    Legacy function. New code should use SandboxConfig instead.

    Args:
        enabled: If True, activate sandbox mode
        cleanup: If True, delete sandbox directory on exit

    Yields:
        Path to sandbox root (or None if disabled)

    Example:
        with test_sandbox(enabled=True, cleanup=False) as root:
            if in_test_mode():
                print(f"Sandbox: {root}")
    """
    root = None
    try:
        if enabled:
            os.environ["APP_MODE"] = "TEST"
            root = sandbox_root()
            root.mkdir(parents=True, exist_ok=True)
            print(f"🧪 TEST MODE — sandbox: {root}")  # noqa: T201
        yield root  # may be None if disabled
    finally:
        if enabled and cleanup and root and root.exists():
            shutil.rmtree(root, ignore_errors=True)
            print(f"🧹 cleaned up sandbox: {root}")  # noqa: T201


def in_test_mode() -> bool:
    """Check if currently in test mode via environment variable.

    Legacy function. New code should use SandboxConfig.enabled instead.

    Returns:
        True if APP_MODE environment variable is set to "TEST"
    """
    return os.environ.get("APP_MODE") == "TEST"


def rebase_path(write_path: Path, sandbox_base: Path | None) -> Path:
    """Rebase a write path under sandbox directory if in test mode.

    Legacy function. New code should use SandboxConfig paths directly.

    Args:
        write_path: Intended write path
        sandbox_base: Sandbox root directory (or None)

    Returns:
        Rebased path under sandbox, or original path if not in test mode
    """
    if sandbox_base is None or not in_test_mode():
        return write_path
    rel = write_path.as_posix().lstrip("/")
    return sandbox_base / rel


def safe_write_text(
    path: Path, content: str, *, dry_run: bool, sandbox_base: Path | None
):
    """Write text file with sandbox and dry-run support.

    Legacy function. New code should use SandboxConfig paths directly.

    Args:
        path: Target file path
        content: Content to write
        dry_run: If True, only print what would be written
        sandbox_base: Sandbox root directory (or None)
    """
    target = rebase_path(path, sandbox_base)
    if dry_run:
        print(f"[DRY-RUN] would write: {target} ({len(content)} bytes)")  # noqa: T201
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"📝 wrote: {target}")  # noqa: T201
