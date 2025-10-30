#!/usr/bin/env python3
"""
Enhanced Error Monitoring System
=================================

Comprehensive error monitoring to prevent silent failures.
Provides loud, immediate alerts for critical errors.
"""

import subprocess
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.companion_file_utils import Logger


class ErrorMonitor:
    """
    Enhanced error monitoring system that makes failures LOUD and VISIBLE.
    """

    def __init__(self, script_name: str = "unknown"):
        self.script_name = script_name
        self.logger = Logger(script_name, enable_colors=True)
        self.error_log_path = (
            PROJECT_ROOT / "data" / "error_logs" / f"{script_name}_errors.log"
        )

        # Ensure error logs directory exists
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    def critical_error(self, message: str, exception: Optional[Exception] = None):
        """
        Log a CRITICAL error that requires IMMEDIATE attention.
        Sends macOS notification and logs to file.
        DOES NOT EXIT - let caller decide what to do.
        """
        timestamp = datetime.now().isoformat()

        # Build error message
        error_msg = f"🚨 CRITICAL ERROR in {self.script_name} 🚨\n{message}"

        if exception:
            error_msg += f"\nException: {exception}"
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"

        # Log to console with maximum visibility
        self._loud_error_display(error_msg)

        # Log to file
        self._log_to_file("CRITICAL", error_msg, timestamp)

        # Send macOS notification
        self._send_macos_notification(
            "CRITICAL ERROR", f"{self.script_name}: {message}"
        )

        # DO NOT EXIT - let caller decide what to do!
        # Removed: sys.exit(1)

    def fatal_error(self, message: str, exception: Optional[Exception] = None):
        """
        Unrecoverable system error - log, notify, and exit.
        Use only for truly fatal errors that prevent the script from continuing.
        """
        self.critical_error(message, exception)  # Log and notify
        self.logger.error("Script terminating due to fatal error")
        sys.exit(1)

    def validation_error(self, message: str, context: Optional[dict] = None):
        """
        Log a validation error (data quality issue).
        Sends notification but doesn't exit.
        """
        timestamp = datetime.now().isoformat()

        error_msg = f"⚠️ VALIDATION ERROR: {message}"
        if context:
            error_msg += f"\nContext: {context}"

        # Log to console
        self.logger.error(error_msg)

        # Log to file
        self._log_to_file("VALIDATION", error_msg, timestamp)

        # Send notification
        self._send_macos_notification("Validation Error", message)

    def silent_failure_detected(
        self, operation: str, expected: str, actual: str = None
    ):
        """
        Specifically for detecting silent failures (the big problem we had).
        """
        message = f"Silent failure detected in {operation}!"
        message += f"\nExpected: {expected}"

        if actual:
            message += f"\nActual: {actual}"

        self.critical_error(message)

    def _loud_error_display(self, message: str):
        """Display error with maximum visual impact."""
        border = "!" * 80

        print(f"\n{border}", file=sys.stderr)
        print("🚨 CRITICAL SYSTEM ERROR 🚨", file=sys.stderr)
        print(border, file=sys.stderr)
        print(message, file=sys.stderr)
        print(border, file=sys.stderr)
        print("This error requires IMMEDIATE attention!", file=sys.stderr)
        print(border, file=sys.stderr)

    def _log_to_file(self, level: str, message: str, timestamp: str):
        """Log error to persistent file."""
        try:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {level}: {message}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            # If we can't log to file, at least print it
            print(f"Failed to log to file: {e}", file=sys.stderr)

    def _send_macos_notification(self, title: str, message: str):
        """Send macOS notification."""
        try:
            # Use osascript for macOS notifications
            script = f'display notification "{message}" with title "{title}" sound name "Basso"'
            subprocess.run(["osascript", "-e", script], check=False)
        except Exception as e:
            # If notification fails, don't let it crash the error handling
            print(f"Failed to send notification: {e}", file=sys.stderr)


# Global error monitor instance
_error_monitor = None


def get_error_monitor(script_name: str = None) -> ErrorMonitor:
    """Get or create global error monitor instance."""
    global _error_monitor

    if script_name and (
        _error_monitor is None or _error_monitor.script_name != script_name
    ):
        _error_monitor = ErrorMonitor(script_name)

    if _error_monitor is None:
        _error_monitor = ErrorMonitor()

    return _error_monitor


def monitor_errors(script_name: str = None):
    """
    Decorator to add comprehensive error monitoring to functions.
    Catches all exceptions, reports them loudly, then re-raises them.
    This preserves the original exit behavior while adding monitoring.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = get_error_monitor(script_name or func.__name__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Unhandled exception in {func.__name__}"
                monitor.critical_error(error_msg, e)
                # Re-raise the exception to preserve original exit behavior
                raise

        return wrapper

    return decorator


def validate_data_quality(
    operation: str, data: Any, validator: Callable[[Any], bool], error_msg: str
):
    """
    Validate data quality and raise critical error if invalid.

    Usage:
        validate_data_quality("crop dimensions", (width, height),
                             lambda d: d[0] > 0 and d[1] > 0,
                             "Invalid crop dimensions detected")
    """
    monitor = get_error_monitor()

    if not validator(data):
        monitor.validation_error(f"{operation}: {error_msg}", {"data": data})


# Quick access functions for common use
def critical_error(message: str, exception: Optional[Exception] = None):
    """Quick access to critical error reporting."""
    get_error_monitor().critical_error(message, exception)


def validation_error(message: str, context: Optional[dict] = None):
    """Quick access to validation error reporting."""
    get_error_monitor().validation_error(message, context)


def fatal_error(message: str, exception: Optional[Exception] = None):
    """Quick access to fatal error reporting."""
    get_error_monitor().fatal_error(message, exception)


def silent_failure_detected(operation: str, expected: str, actual: str = None):
    """Quick access to silent failure detection."""
    get_error_monitor().silent_failure_detected(operation, expected, actual)
