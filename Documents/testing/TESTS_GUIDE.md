# Tests Guide

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers
**Estimated Reading Time:** 5 minutes

## Quick Start
```bash
# From repo root
python3 scripts/tests/test.py --safety-only    # ~30s
python3 scripts/tests/test.py                  # full suite (includes performance)
```

## Pytest Usage
```bash
# All tests (verbose)
pytest -v

# Single file
pytest scripts/tests/test_dashboard.py -v

# Pattern
pytest scripts/tests/test_dashboard*.py -v

# Markers (if defined)
pytest -m "not slow" -v
```

## Runner Scripts
```bash
python3 scripts/tests/test_runner.py --safety-only
python3 scripts/tests/test_runner.py --performance
```

## Web Tools (Selenium Smoke)
- Infrastructure: `scripts/tests/test_base_selenium.py`
- Smoke: `scripts/tests/test_web_tools_smoke.py`
- Behavior:
  - Launch each Flask tool as a subprocess on a free port
  - Wait for readiness, open headless browser
  - Verify title/body and key elements
- Isolation: sets `EM_TEST_DATA_ROOT`; creates temp dirs; cleans up on teardown

## Dashboard Tests
```bash
# Core dashboard
pytest scripts/tests/test_dashboard*.py -v

# End-to-end runner watchdog
pytest scripts/tests/test_runner_watchdog_e2e.py -v

# Snapshot integrity
pytest scripts/tests/test_snapshot_data_integrity.py -v
```

## Tips
- Ensure Chrome/driver available (webdriver-manager auto-installs in tests).
- Close orphaned servers if you ran tools manually.
- Prefer smaller lookbacks for faster dashboard API local tests.

## Related Documents
- `dashboard/DASHBOARD_GUIDE.md`
- `dashboard/DASHBOARD_API.md`
- `reference/TECHNICAL_KNOWLEDGE_BASE.md`
