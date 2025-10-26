# Productivity Dashboard - Complete Guide
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


**Comprehensive Documentation** - Last Updated: October 16, 2025

---

# Table of Contents

1. [Quick Start](#quick-start)
2. [Dashboard Specification](#dashboard-specification)
3. [Data Layer & API Reference](#data-layer--api-reference)
4. [Troubleshooting](#troubleshooting)

---

# Quick Start

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
pip install -r scripts/dashboard/requirements.txt
```

### 2. Start the API Server
Option A (recommended launcher):
```bash
python3 scripts/dashboard/run_dashboard.py --host 127.0.0.1 --port 5001
```
Option B (direct server):
```bash
python3 scripts/dashboard/productivity_dashboard.py --host 127.0.0.1 --port 5001 --data-dir .
```

You should see:
```
🚀 Productivity Dashboard starting at http://127.0.0.1:5001
🌐 Opening browser to http://127.0.0.1:5001 (if not in debug)
```

### 3. Open the Dashboard
Open `scripts/dashboard/dashboard_template.html` in your browser. The dashboard will automatically connect to the API and display your productivity metrics.

---

## 📊 Example JSON Responses

### Daily View (3-day example)

**Request:**
```bash
curl "http://127.0.0.1:5001/api/data/D?lookback_days=3"
```

**Response (simplified):**
```json
{
  "metadata": {
    "time_slice": "D",
    "lookback_days": 3,
    "baseline_labels": {
      "D": ["2025-10-13", "2025-10-14", "2025-10-15"]
    }
  },
  
  "charts": {
    "by_script": {
      "AI-Assisted Reviewer": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [120, 95, 143]
      },
      "Multi Crop Tool": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [80, 0, 92]
      },
      "Character Sorter": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [0, 65, 88]
      }
    },
    
    "by_operation": {
      "crop": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [50, 45, 62]
      },
      "delete": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [30, 25, 40]
      },
      "move": {
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "counts": [120, 90, 141]
      }
    }
  },
  
  "timing_data": {
    "AI-Assisted Reviewer": {
      "work_time_minutes": 127.3,
      "timing_method": "file_operations"
    },
    "Multi Crop Tool": {
      "work_time_minutes": 89.5,
      "timing_method": "file_operations"
    }
  }
}
```

### Hourly View (3-hour example)

**Request:**
```bash
curl "http://127.0.0.1:5001/api/data/1H?lookback_days=1"
```

**Response (3 hours shown):**
```json
{
  "metadata": {
    "time_slice": "1H",
    "baseline_labels": {
      "1H": ["2025-10-15T14:00:00", "2025-10-15T15:00:00", "2025-10-15T16:00:00"]
    }
  },
  
  "charts": {
    "by_operation": {
      "crop": {
        "dates": ["2025-10-15T14:00:00", "2025-10-15T15:00:00", "2025-10-15T16:00:00"],
        "counts": [25, 30, 22]
      },
      "delete": {
        "dates": ["2025-10-15T14:00:00", "2025-10-15T15:00:00", "2025-10-15T16:00:00"],
        "counts": [10, 0, 8]
      }
    }
  }
}
```

---

## 🧪 Testing

### Run Smoke Tests
```bash
python3 scripts/dashboard/smoke_test.py
```

Expected output:
```
🎉 ALL TESTS PASSED!

Next steps:
  1. Start the API server: python3 scripts/dashboard/productivity_dashboard.py
  2. Open dashboard: scripts/dashboard/dashboard_template.html
  3. Test endpoints: curl "http://127.0.0.1:5001/api/data/D?lookback_days=7"
```

### Run Unit Tests (requires pytest)
```bash
pip install pytest pytest-asyncio httpx
pytest scripts/tests/test_dashboard*.py -v
```

---

## 📈 Key Features

### Time Slices
- **15min**: 15-minute intervals (intraday analysis)
- **1H**: Hourly intervals (intraday analysis)
- **D**: Daily aggregation (default)
- **W**: Weekly aggregation (Monday start)
- **M**: Monthly aggregation (first of month)

### Metrics Tracked
- **By Script**: Files processed per tool (AI-Assisted Reviewer, Multi Crop Tool, Character Sorter)
- **By Operation**: Operations by type (crop, delete, move, send_to_trash, skip)
- **Timing**: Work time per tool (file operations or activity timer)
- **Projects**: Per-project throughput (images/hour), totals, timeseries

### Data Alignment
- All series aligned to canonical baseline labels
- Gaps filled with zeros (not nulls)
- Chronologically sorted (ascending)
- Deterministic label generation

---

# Dashboard Specification

## Charting Approach

**Primary visualization type:** Bar charts for all time-based breakdowns.

**Overlay:** A "cloud" (faded background shape) representing historical averages for the same time slice. This provides context without obscuring the current bar values.

**Update markers:** Visual cues (vertical line or icon) plotted on the chart timeline whenever a script is updated.

## Time Scales

### Intraday view
- Bars represent fixed time slices (configurable: 15 minutes or 1 hour)
- X-axis: clock time across the day
- Y-axis: either active minutes or files processed
- Overlay: average historical productivity for each time slice (the "cloud")

### Daily view
- Bars represent total daily productivity
- X-axis: calendar days
- Y-axis: totals (active minutes, files processed, or both)
- Overlay: average daily totals
- Update markers: script update events aligned to the day they occurred

### Weekly view
- Bars represent one week of work
- X-axis: week number or start date of the week
- Y-axis: total active minutes/files processed for the week
- Overlay: average weekly totals
- Update markers: vertical lines when updates happen inside that week

### Monthly view
- Bars represent one month of work
- X-axis: calendar months
- Y-axis: monthly totals
- Overlay: historical monthly average
- Update markers: script update events aligned to the month

### Pie chart
- Displays share of time per tool/script
- Typically shown for monthly or cumulative periods
- Useful to see which tool consumes the largest share of overall time

## Script Update Tracking

Maintain a simple log (CSV or JSONL) of updates:

```csv
date,tool,description
2025-09-12,web_selector,"Added batch skipping"
2025-09-20,clustering,"Optimized embeddings"
```

Each update should be shown on relevant charts as a vertical marker (line or icon).

When hovering or clicking, the description should display.

This makes it possible to visually correlate productivity shifts with tool changes.

---

# Data Layer & API Reference

## Overview

This data layer serves aggregated productivity metrics to the dashboard frontend via a Flask server. It transforms raw event logs (file operations, activity timer sessions) into chart-ready JSON without manual tweaking.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Dashboard Template (HTML + Chart.js)                   │
│  Expects: /api/data/<time_slice>?lookback_days=&project_id= │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  productivity_dashboard.py (Flask Server)               │
│  Route: GET /api/data/<time_slice>                      │
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
│  - Load snapshot aggregates                             │
│  - Load derived sessions                                │
│  - Discover scripts dynamically                         │
│  - Generate baseline labels per time slice              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  snapshot_loader.py (Snapshot Data Loader)              │
│  - Load daily aggregates from snapshot                  │
│  - Load derived sessions from snapshot                  │
│  - Optimized for performance                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Raw Data Sources                                        │
│  - data/file_operations_logs/*.log (RAW, archived)      │
│  - data/snapshot/daily_aggregates_v1/*.json (PRIMARY)   │
│  - data/snapshot/derived_sessions_v1/*.jsonl (PRIMARY)  │
│  - data/projects/*.project.json                         │
└─────────────────────────────────────────────────────────┘
```

## Data Sources & Primitives

### File Operations Logs (RAW - archived for detail)
**Location:** `data/file_operations_logs/*.log`

**Fields:**
- `timestamp` (ISO string): When the operation occurred
- `script` (string): Tool name (e.g., "01_web_image_selector")
- `operation` (string): Type of operation (crop, delete, move, send_to_trash, skip)
- `file_count` (int): Number of files affected
- `source_dir`, `dest_dir` (strings): Directories involved
- `notes` (string): Additional context

**Usage:** Archive/detail source. Dashboard prioritizes snapshot aggregates for speed.

### Daily Aggregates (PRIMARY - snapshot)
**Location:** `data/snapshot/daily_aggregates_v1/day=YYYYMMDD.json`

**Fields:**
- `day` (string): YYYYMMDD format
- `total_operations` (int): Total operations for the day
- `total_files` (int): Total files processed
- `by_script` (dict): Per-script aggregates with timestamps
- `by_operation` (dict): Per-operation aggregates

**Usage:** Primary data source for dashboard charts (fast, pre-aggregated).

### Derived Sessions (PRIMARY - snapshot)
**Location:** `data/snapshot/derived_sessions_v1/day=YYYYMMDD.jsonl`

**Fields:**
- `session_id` (string): Unique session identifier
- `script` (string): Tool name
- `start_ts_utc`, `end_ts_utc` (timestamps): Session boundaries
- `active_time_seconds` (int): Active working time
- `files_processed` (int): Files processed in session
- `operation_count` (int): Number of operations

**Usage:** Session-based timing and activity metrics.

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

### `GET /api/data/<time_slice>`

**Path Parameter:**
- `<time_slice>` (required): `15min` | `1H` | `D` | `W` | `M`

**Query Parameters:**
- `lookback_days` (int, default=30): Number of days to look back (1-365)
- `project_id` (string, optional): Filter to specific project (empty = all projects)

**Response:** JSON matching dashboard template contract (see below)

**Example:**
```bash
curl "http://127.0.0.1:5001/api/data/D?lookback_days=7&project_id=mojo1"
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
    "active_project": "mojo1",
    "session_source": "derived",
    "performance_mode": true
  },
  
  "projects": [
    { "projectId": "mojo1", "title": "Project Mojo", "status": "active" }
  ],
  
  "charts": {
    "by_script": {
      "AI-Assisted Reviewer": {
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
    "AI-Assisted Reviewer": {
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
        "AI-Assisted Reviewer": { "iph": 52.1, "baseline": 48.3 },
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
          "AI-Assisted Reviewer": 48.3,
          "Multi Crop Tool": 35.2
        }
      },
      "tools": {
        "AI-Assisted Reviewer": {
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

## Timing Methods

The dashboard uses two timing methods depending on the tool:

### File Operations Timing
- Used for: Image Selector, Character Sorter, Crop Tool
- Method: Analyzes file operation timestamps with intelligent break detection
- Break threshold: 5 minutes of inactivity = break
- Excludes: Idle time, breaks, pauses

### Activity Timer Timing (Legacy)
- Used for: Multi Directory Viewer, Duplicate Finder (scroll-heavy tools)
- Method: Sums `active_time` from timer sessions
- Tracks: User interactions, keyboard/mouse activity
- Excludes: Idle periods, context switches

### Combined Reporting
Tools report their timing method in `timing_data`:
```json
{
  "AI-Assisted Reviewer": {
    "work_time_minutes": 127.3,
    "timing_method": "file_operations"
  }
}
```

## Performance & Caching

### Data Engine Caching
- `DashboardDataEngine` uses in-memory cache for processed data
- Cache invalidates automatically on file changes

### Snapshot Loading
- `SnapshotLoader` reads pre-aggregated daily summaries
- Optimized for speed (10x faster than raw log parsing)
- Falls back to raw logs for missing days

### Label Generation
- Deterministic, no randomness or unordered sets
- Pre-computed for all time slices in single pass
- Aligned series use index-based lookups (O(1) per point)

---

# Troubleshooting

## No data appears in charts

**Problem:** Dashboard shows empty charts

**Solutions:**
1. Check that data files exist:
   ```bash
   ls data/file_operations_logs/
   ls data/snapshot/daily_aggregates_v1/
   ```
2. Increase lookback days: `?lookback_days=30`
3. Check console errors: Open browser DevTools (F12)
4. Test API directly: `curl http://localhost:8000/api/data?slice=D`

## Charts show gaps

- Expected behavior! Gaps = no activity on those days
- Gaps are filled with `0` to maintain alignment

## Project filter returns no data

- Verify `projectId` exists: Check `data/projects/*.project.json`
- Ensure operations reference project paths in logs
- Try "All Projects" (empty `project_id`) to verify data exists

## Performance is slow

**Problem:** API takes >5 seconds to respond

**Solutions:**
1. Archive old logs to `data/log_archives/` (gzip supported)
2. Reduce lookback_days: `?lookback_days=7`
3. Check data volume:
   ```bash
   du -sh data/file_operations_logs/
   du -sh data/snapshot/
   ```
4. Run daily snapshot scripts to ensure aggregates are up to date:
   ```bash
   python3 scripts/data_pipeline/extract_operation_events_v1.py
   python3 scripts/data_pipeline/build_daily_aggregates_v1.py
   ```

## API returns 500 error

**Problem:** Server crashes when requesting data

**Solutions:**
1. Check server logs for Python errors
2. Verify data file integrity:
   ```bash
   python3 scripts/dashboard/data_engine.py
   ```
3. Check for malformed JSON in logs

## Charts show wrong dates

**Problem:** Labels don't match expected dates

**Cause:** Timezone issues or DST transitions

**Solution:** All timestamps are naive (no timezone) for simplicity. Ensure consistency across all data sources.

---

## 📞 Development

### Run Tests
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
python3 -m pytest scripts/tests/test_dashboard*.py -v
python3 -m pytest scripts/tests/test_snapshot_data_integrity.py -v
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

### Debug Project Data
```bash
python3 scripts/dashboard/tools/debug_project_data.py mojo2
# Shows detailed data flow for a specific project
```

---

## Files

- `productivity_dashboard.py` - Flask server (main entry point)
- `analytics.py` - Aggregation and transformation logic
- `data_engine.py` - Raw data collection and time slicing
- `snapshot_loader.py` - Snapshot data loader (optimized)
- `dashboard_template.html` - Frontend UI (Chart.js)
- `../dashboard/DASHBOARD_GUIDE.md` - This file

---

## Future Enhancements

- [ ] Add server-side historical average computation (currently client-side)
- [ ] Support custom date ranges (not just lookback_days)
- [ ] Implement WebSocket for real-time updates
- [ ] Add export endpoints (CSV, JSON download)
- [ ] Optimize for larger datasets (pagination, streaming)

---

**Built with:** Python 3.11, Flask, Chart.js  
**Status:** ✅ Production-ready

