# Dashboard API Reference

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers
**Estimated Reading Time:** 6 minutes

## Overview
The Dashboard API serves aggregated productivity metrics for the web dashboard. It currently exposes a single read endpoint for dashboard data and a health check.

## Base URL
- Default (FastAPI local dev): `http://127.0.0.1:8000`

## How to Run
- FastAPI server:
```bash
python scripts/dashboard/api.py  # uses uvicorn under __main__
# or explicitly
python -m uvicorn scripts.dashboard.api:app --host 0.0.0.0 --port 8000 --reload
```
- Legacy launcher (Flask UI + backend combined): see `dashboard/DASHBOARD_GUIDE.md`.

## Endpoints

### GET /api/data/{time_slice}
- Purpose: Return the full data payload the dashboard UI expects.
- Path params:
  - `time_slice` (string): one of `15min`, `1H`, `D`, `W`, `M`
- Query params:
  - `lookback_days` (int, default 30, range 1–365): window size
  - `project_id` (string, optional): filter by project; empty = all
- Response: JSON, matching the dashboard template contract
- Example:
```bash
curl "http://127.0.0.1:8000/api/data/D?lookback_days=7&project_id=mojo1"
```

### GET /health
- Purpose: Service health check
- Response: `{ "status": "healthy", "service": "productivity-dashboard-api" }`

## Response Contract (summary)
- Canonical labels: `metadata.baseline_labels[time_slice]` define x-axis buckets.
- Charts:
  - `charts.by_script` and `charts.by_operation` are time-aligned to baseline labels.
- Project comparisons: includes per-project IPH and baselines.
- For details of field names and layout, see `dashboard/DASHBOARD_GUIDE.md`.

## Operational Notes
- Analytics engine: `scripts/dashboard/analytics.py` via `DashboardAnalytics`.
- Server: FastAPI app in `scripts/dashboard/api.py`.
- Dev convenience: `--reload` for hot reload while working on analytics.

## Examples
```bash
# Daily, last 30 days
curl "http://127.0.0.1:8000/api/data/D?lookback_days=30"

# Hourly, last day
curl "http://127.0.0.1:8000/api/data/1H?lookback_days=1"
```

## Related Documents
- `dashboard/DASHBOARD_GUIDE.md` — End-to-end dashboard guide
- `core/OPERATIONS_GUIDE.md` — Running services and troubleshooting
- `data/SCHEMA_REFERENCE.md` — Data sources that back the dashboard
- `reference/TECHNICAL_KNOWLEDGE_BASE.md` — Patterns and utilities
