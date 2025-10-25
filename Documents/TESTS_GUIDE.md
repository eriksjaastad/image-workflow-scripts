# Tests Guide

Last Updated: 2025-10-23
Scope: How to run quick safety checks, the full suite, and Selenium smoke tests.

## Quick Start
```bash
# From repo root
python3 scripts/tests/test.py --safety-only    # ~30s
python3 scripts/tests/test.py                  # full suite (includes performance)
```

## From scripts/tests runner
```bash
python3 scripts/tests/test_runner.py --safety-only
python3 scripts/tests/test_runner.py --performance
```

## Web tools (Selenium smoke)
- Infrastructure: `scripts/tests/test_base_selenium.py`
- Smoke tests: `scripts/tests/test_web_tools_smoke.py`
- Behavior:
  - Launches each Flask tool as a subprocess on a free port
  - Waits for server readiness, opens headless Chrome
  - Verifies title/body and presence of key elements
- Isolation: sets `EM_TEST_DATA_ROOT` and creates temporary directories; cleans up on teardown

## Dashboard tests
```bash
# Core dashboard tests
python3 -m pytest scripts/tests/test_dashboard*.py -v

# End-to-end runner watchdog
python3 -m pytest scripts/tests/test_runner_watchdog_e2e.py -v

# Snapshot integrity
python3 -m pytest scripts/tests/test_snapshot_data_integrity.py -v
```

## Tips
- Ensure ChromeDriver is available (webdriver-manager auto-installs in tests)
- Close orphaned servers if you ran tools manually
- Prefer smaller lookbacks for faster dashboard API during local testing

## References
- `scripts/tests/README.md`
- `scripts/tests/test_base_selenium.py`
- `scripts/tests/test_web_tools_smoke.py`
- `Documents/DASHBOARD_GUIDE.md`
