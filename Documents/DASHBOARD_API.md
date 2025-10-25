# Dashboard API Reference

Last Updated: 2025-10-23
Source of truth: `scripts/dashboard/productivity_dashboard.py`

## Base URL
- Default: `http://127.0.0.1:5001`

## Endpoints

### GET /api/data/<time_slice>
- Purpose: Chart/table payload for the dashboard UI
- Path param `time_slice` (string): `15min` | `1H` | `D` | `W` | `M`
- Query params:
  - `lookback_days` (int, default 60): size of window
  - `project_id` (string, optional): filter; empty or missing = all projects
- Response: JSON (aligned to baseline labels; padded with zeros)
- Example:
```bash
curl "http://127.0.0.1:5001/api/data/D?lookback_days=7&project_id=mojo1"
```

### GET /api/scripts
- Purpose: List of discovered scripts for filtering
- Response: `{ "scripts": ["01_web_image_selector", ...] }`

### GET /api/script_updates
- Purpose: Fetch script update markers for overlay
- Response: JSON array of records `[ { date, tool, description }, ... ]`

### POST /api/script_updates
- Purpose: Append a script update marker
- Body (JSON):
```json
{ "script": "01_web_image_selector", "description": "Added batch skipping", "date": "2025-09-12" }
```
- Response: `{ "status": "success" }`

### GET /api/debug
- Purpose: Raw data structure for debugging
- Response: JSON of raw assembled payload (uncached)

## Response Contract (summary)
- See full spec in `Documents/DASHBOARD_GUIDE.md` (Data Layer & API Reference).
- Notes:
  - `metadata.baseline_labels[time_slice]` provides canonical labels used for alignment.
  - `charts.by_script` and `charts.by_operation` arrays are sorted chronologically.
  - `project_comparisons` contains per-project IPH and baseline references.

## Operational Notes
- Launcher auto-refreshes snapshots before starting: `scripts/dashboard/run_dashboard.py`.
- Server class: `ProductivityDashboard` (Flask) with routes defined in `setup_routes()`.
- Disable cache in responses for fresh UI renders.

## Examples
```bash
# Daily, last 30 days
curl "http://127.0.0.1:5001/api/data/D?lookback_days=30"

# Hourly, last day
curl "http://127.0.0.1:5001/api/data/1H?lookback_days=1"
```

## Links
- `Documents/DASHBOARD_GUIDE.md`
- `scripts/dashboard/productivity_dashboard.py`
- `scripts/dashboard/run_dashboard.py`
