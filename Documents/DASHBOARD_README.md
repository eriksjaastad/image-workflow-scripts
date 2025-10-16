# Productivity Dashboard - Data Layer & API

## Overview

This data layer serves aggregated productivity metrics to the dashboard frontend via a simple FastAPI server. It transforms raw event logs (file operations, activity timer sessions) into chart-ready JSON without manual tweaking.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Dashboard Template (HTML + Chart.js)                   │
│  Expects: /api/data/{slice}?lookback_days=N&project_id= │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  api.py (FastAPI Server)                                │
│  Route: GET /api/data/{slice}                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  analytics.py (Aggregation & Transformation)            │
│  - Bucket by time slice (15min/1H/D/W/M)                │
│  - Align series to baseline labels (fill gaps w/ zeros) │
│  - Compute historical averages for "cloud" overlays     │
│  - Transform project metrics for comparison charts      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  data_engine.py (Raw Data Collection)                   │
│  - Load file operations logs                            │
│  - Load activity timer sessions                         │
│  - Discover scripts dynamically                         │
│  - Generate baseline labels per time slice              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  project_metrics_aggregator.py (Project Rollups)        │
│  - Per-project totals (images processed, ops by type)   │
│  - Per-project throughput (images/hour)                 │
│  - Per-tool metrics and baselines                       │
│  - Daily timeseries for sparklines                      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Raw Data Sources                                        │
│  - data/file_operations_logs/*.log                      │
│  - data/daily_summaries/*.json                          │
│  - data/timer_data/daily_*.json                         │
│  - data/projects/*.project.json                         │
└─────────────────────────────────────────────────────────┘
```

## Data Sources & Primitives

### File Operations Logs
**Location:** `data/file_operations_logs/*.log`, `data/daily_summaries/*.json`

**Fields:**
- `timestamp` (ISO string): When the operation occurred
- `script` (string): Tool name (e.g., "01_web_image_selector")
- `operation` (string): Type of operation (crop, delete, move, send_to_trash, skip)
- `file_count` (int): Number of files affected
- `source_dir`, `dest_dir` (strings): Directories involved
- `notes` (string): Additional context (may indicate "image-only" counting)

**Usage:** Primary source for file counts by tool and operation type. Used for timing calculations via intelligent break detection.

### Activity Timer Sessions
**Location:** `data/timer_data/daily_*.json`

**Fields:**
- `script` (string): Tool name
- `session_id` (string): Unique session identifier
- `start_time`, `end_time` (timestamps): Session boundaries
- `active_time` (seconds): User-active time (not idle)
- `total_time` (seconds): Wall-clock time
- `efficiency` (float): active_time / total_time
- `files_processed` (int): Files processed in this session
- `operations` (dict): Operation counts by type

**Usage:** Timing source for scroll-heavy tools (Multi Directory Viewer, Duplicate Finder). Provides efficiency metrics.

### Project Manifests
**Location:** `data/projects/*.project.json`

**Fields:**
- `projectId` (string): Unique project identifier
- `title` (string): Human-readable project name
- `status` (string): "active", "completed", etc.
- `startedAt`, `finishedAt` (ISO timestamps): Project timeline
- `paths.root` (string): Root path hint for filtering operations

**Usage:** Project-level filtering and timeline markers.

## API Endpoint

### `GET /api/data/{slice}`

**Path Parameters:**
- `slice` (required): Time slice granularity
  - `15min` - 15-minute intervals (intraday)
  - `1H` - Hourly intervals (intraday)
  - `D` - Daily aggregation
  - `W` - Weekly aggregation (Monday start)
  - `M` - Monthly aggregation (first of month)

**Query Parameters:**
- `lookback_days` (int, default=30): Number of days to look back (1-365)
- `project_id` (string, optional): Filter to specific project (empty = all projects)

**Response:** JSON matching dashboard template contract (see below)

**Example:**
```bash
curl "http://localhost:8000/api/data/D?lookback_days=7&project_id=mojo1"
```

## Response Contract

The API returns exactly the following JSON structure (all fields required, even if empty):

```json
{
  "metadata": {
    "generated_at": "2025-10-15T12:34:56.789Z",
    "time_slice": "D",
    "lookback_days": 30,
    "baseline_labels": {
      "15min": ["2025-10-01T00:00:00", "2025-10-01T00:15:00", ...],
      "1H": ["2025-10-01T00:00:00", "2025-10-01T01:00:00", ...],
      "D": ["2025-10-01", "2025-10-02", ...],
      "W": ["2025-09-30", "2025-10-07", ...],
      "M": ["2025-09-01", "2025-10-01", ...]
    },
    "active_project": "mojo1"
  },
  
  "projects": [
    { "projectId": "mojo1", "title": "Project Mojo", "status": "active" }
  ],
  
  "charts": {
    "by_script": {
      "Web Image Selector": {
        "dates": ["2025-10-01", "2025-10-02", ...],
        "counts": [120, 95, 0, 143, ...]
      },
      "Multi Crop Tool": { "dates": [...], "counts": [...] }
    },
    "by_operation": {
      "crop": { "dates": [...], "counts": [...] },
      "delete": { "dates": [...], "counts": [...] },
      "move": { "dates": [...], "counts": [...] },
      "send_to_trash": { "dates": [...], "counts": [...] },
      "skip": { "dates": [...], "counts": [...] }
    }
  },
  
  "timing_data": {
    "Web Image Selector": {
      "work_time_minutes": 127.3,
      "timing_method": "file_operations"
    },
    "Multi Crop Tool": {
      "work_time_minutes": 89.5,
      "timing_method": "file_operations"
    }
  },
  
  "project_comparisons": [
    {
      "projectId": "mojo1",
      "title": "Project Mojo",
      "iph": 45.2,
      "baseline_overall": 42.8,
      "tools": {
        "Web Image Selector": { "iph": 52.1, "baseline": 48.3 },
        "Multi Crop Tool": { "iph": 38.7, "baseline": 35.2 }
      }
    }
  ],
  
  "project_kpi": {
    "images_per_hour": 45.2,
    "images_processed": 1234
  },
  
  "project_metrics": {
    "mojo1": {
      "projectId": "mojo1",
      "title": "Project Mojo",
      "status": "active",
      "startedAt": "2025-10-01T08:00:00Z",
      "finishedAt": null,
      "totals": {
        "images_processed": 1234,
        "operations_by_type": {
          "crop": 456,
          "delete": 234,
          "move": 544
        }
      },
      "throughput": {
        "images_per_hour": 45.2
      },
      "timeseries": {
        "daily_files_processed": [
          ["2025-10-01", 120],
          ["2025-10-02", 95],
          ["2025-10-03", 143]
        ]
      },
      "baseline": {
        "overall_iph_baseline": 42.8,
        "per_tool": {
          "Web Image Selector": 48.3,
          "Multi Crop Tool": 35.2
        }
      },
      "tools": {
        "Web Image Selector": {
          "images_processed": 678,
          "images_per_hour": 52.1
        }
      }
    }
  },
  
  "project_markers": {
    "startedAt": "2025-10-01T08:00:00Z",
    "finishedAt": null
  }
}
```

## Label Generation Rules

### Alignment & Gaps
- All series are aligned to `baseline_labels[time_slice]`
- Gaps are filled with `0` (not `null`)
- Labels are sorted chronologically (ascending)
- Series counts must have same length as labels

### Time Slice Formats

**15min & 1H:** ISO timestamp (naive, no timezone)
- Example: `"2025-10-15T14:30:00"`
- Aligned to 15-minute or hourly boundaries

**D (Daily):** ISO date
- Example: `"2025-10-15"`

**W (Weekly):** ISO date of Monday (week start)
- Example: `"2025-10-14"` (Monday)
- Uses ISO week calculation

**M (Monthly):** First day of month
- Example: `"2025-10-01"`

## Setup & Usage

### Prerequisites
```bash
# Python 3.11+
# Install dependencies
pip install fastapi uvicorn pydantic
```

### Start the API Server
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
python3 scripts/dashboard/api.py

# Or with uvicorn directly:
uvicorn scripts.dashboard.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### Open the Dashboard
1. Start the API server (above)
2. Open `scripts/dashboard/dashboard_template.html` in a web browser
3. The dashboard will automatically connect to `http://localhost:8000/api/data/{slice}`

### Testing Endpoints
```bash
# Daily view, last 7 days
curl "http://localhost:8000/api/data/D?lookback_days=7"

# Hourly view, last 2 days, specific project
curl "http://localhost:8000/api/data/1H?lookback_days=2&project_id=mojo1"

# Health check
curl "http://localhost:8000/health"

# API docs (Swagger UI)
open http://localhost:8000/docs
```

## Performance & Caching

### Data Engine Caching
- `DashboardDataEngine` uses in-memory cache for processed data
- Cache invalidates automatically on file changes

### Project Metrics Caching
- `ProjectMetricsAggregator` uses mtime-based cache
- Only recomputes when source files change
- Lightweight for typical workloads (< 1s for 10k records)

### Label Generation
- Deterministic, no randomness or unordered sets
- Pre-computed for all time slices in single pass
- Aligned series use index-based lookups (O(1) per point)

## Averages "Cloud" Overlay

The dashboard template supports optional average overlays. These are computed client-side from the `counts` arrays using a simple algorithm:

```javascript
// For each series, compute non-zero average
const nonZero = counts.filter(v => v > 0);
const avg = nonZero.length ? sum(nonZero) / nonZero.length : 0;
const avgSeries = Array(counts.length).fill(avg);
```

This provides a lightweight historical context line without server-side complexity.

## Project Markers

**Start/Finish Markers** are derived from project manifest timestamps:
- `startedAt`: Blue vertical line at project start
- `finishedAt`: Red vertical line at project completion

These are rendered as Chart.js plugins (see template for implementation).

## Update Markers (Script Updates)

**Location:** `scripts/dashboard/script_updates.csv`

**Format:**
```csv
date,script,description
2025-10-12,01_web_image_selector,Added batch skipping
2025-10-14,04_multi_crop_tool,Optimized crop algorithm
```

**Usage:** Vertical markers on timeline charts to correlate performance changes with code updates.

## Timing Methods

The dashboard uses two timing methods depending on the tool:

### File Operations Timing
- Used for: Image Selector, Character Sorter, Crop Tool
- Method: Analyzes file operation timestamps with intelligent break detection
- Break threshold: 5 minutes of inactivity = break
- Excludes: Idle time, breaks, pauses

### Activity Timer Timing
- Used for: Multi Directory Viewer, Duplicate Finder (scroll-heavy tools)
- Method: Sums `active_time` from timer sessions
- Tracks: User interactions, keyboard/mouse activity
- Excludes: Idle periods, context switches

### Combined Reporting
Tools report their timing method in `timing_data`:
```json
{
  "Web Image Selector": {
    "work_time_minutes": 127.3,
    "timing_method": "file_operations"
  }
}
```

The dashboard uses this to display appropriate timing indicators.

## Troubleshooting

### No data appears in charts
1. Check that data files exist: `ls data/file_operations_logs/`
2. Verify date range: `lookback_days` may be too short
3. Check console for errors: Open browser DevTools
4. Test API directly: `curl http://localhost:8000/api/data/D`

### Charts show gaps
- Expected behavior! Gaps = no activity on those days
- Gaps are filled with `0` to maintain alignment

### Project filter returns no data
- Verify `projectId` exists: Check `data/projects/*.project.json`
- Ensure operations reference project paths in logs
- Try "All Projects" (empty `project_id`) to verify data exists

### Performance is slow
- Check data volume: `du -sh data/file_operations_logs/`
- Consider archiving old logs to `data/log_archives/` (gzip supported)
- Increase cache effectiveness by running multiple queries

## Development

### Run Tests
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
python3 -m pytest scripts/dashboard/tests/ -v
```

### Run Analytics Engine Standalone
```bash
python3 scripts/dashboard/analytics.py
# Generates: dashboard_analytics_sample.json
```

### Inspect Raw Data Engine Output
```bash
python3 scripts/dashboard/data_engine.py
# Generates: dashboard_data_sample.json
```

## Files

- `api.py` - FastAPI server (endpoint handler)
- `analytics.py` - Aggregation and transformation logic
- `data_engine.py` - Raw data collection and time slicing
- `project_metrics_aggregator.py` - Per-project rollups
- `dashboard_template.html` - Frontend UI (Chart.js)
- `script_updates.csv` - Update marker tracking
- `README.md` - This file

## Future Enhancements

- [ ] Add server-side historical average computation (currently client-side)
- [ ] Support custom date ranges (not just lookback_days)
- [ ] Add caching middleware for repeated queries
- [ ] Implement WebSocket for real-time updates
- [ ] Add export endpoints (CSV, JSON download)
- [ ] Optimize for larger datasets (pagination, streaming)

## License

Part of the Eros Mate productivity toolkit. Internal use only.


