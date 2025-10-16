# Productivity Dashboard - Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
pip install -r scripts/dashboard/requirements.txt
```

### 2. Start the API Server
```bash
python3 scripts/dashboard/api.py
```

You should see:
```
🚀 Starting Productivity Dashboard API...
📂 Data directory: /Users/eriksjaastad/projects/Eros Mate
🌐 API docs: http://localhost:8000/docs
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Open the Dashboard
Open `scripts/dashboard/dashboard_template.html` in your browser. The dashboard will automatically connect to the API and display your productivity metrics.

---

## 📊 Example JSON Responses

### Daily View (3-day example)

**Request:**
```bash
curl "http://localhost:8000/api/data/D?lookback_days=3"
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
      "Web Image Selector": {
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
    "Web Image Selector": {
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
curl "http://localhost:8000/api/data/1H?lookback_days=1"
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
  1. Start the API server: python3 scripts/dashboard/api.py
  2. Open dashboard: scripts/dashboard/dashboard_template.html
  3. Test endpoints: curl http://localhost:8000/api/data/D
```

### Run Unit Tests (requires pytest)
```bash
pip install pytest pytest-asyncio httpx
pytest scripts/dashboard/test_analytics.py -v
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
- **By Script**: Files processed per tool (Web Image Selector, Multi Crop Tool, Character Sorter)
- **By Operation**: Operations by type (crop, delete, move, send_to_trash, skip)
- **Timing**: Work time per tool (file operations or activity timer)
- **Projects**: Per-project throughput (images/hour), totals, timeseries

### Data Alignment
- All series aligned to canonical baseline labels
- Gaps filled with zeros (not nulls)
- Chronologically sorted (ascending)
- Deterministic label generation

---

## 🔧 Troubleshooting

### No data in charts
**Problem:** Dashboard shows empty charts

**Solutions:**
1. Check that data files exist:
   ```bash
   ls data/file_operations_logs/
   ls data/timer_data/
   ```
2. Increase lookback days: `?lookback_days=30`
3. Check console errors: Open browser DevTools (F12)

### API returns 500 error
**Problem:** Server crashes when requesting data

**Solutions:**
1. Check server logs for Python errors
2. Verify data file integrity:
   ```bash
   python3 scripts/dashboard/data_engine.py
   ```
3. Check for malformed JSON in logs

### Performance is slow
**Problem:** API takes >5 seconds to respond

**Solutions:**
1. Archive old logs to `data/log_archives/` (gzip supported)
2. Reduce lookback_days: `?lookback_days=7`
3. Check data volume:
   ```bash
   du -sh data/file_operations_logs/
   ```

### Charts show wrong dates
**Problem:** Labels don't match expected dates

**Cause:** Timezone issues or DST transitions

**Solution:** All timestamps are naive (no timezone). Ensure consistency across all data sources.

---

## 📚 API Reference

### Endpoints

#### `GET /api/data/{slice}`

Get dashboard data for a specific time slice.

**Path Parameters:**
- `slice` (required): Time slice granularity (`15min`, `1H`, `D`, `W`, `M`)

**Query Parameters:**
- `lookback_days` (int, default=30): Number of days to look back (1-365)
- `project_id` (string, optional): Filter by project ID (empty = all projects)

**Response:** JSON matching dashboard contract

**Examples:**
```bash
# Daily view, last 7 days
curl "http://localhost:8000/api/data/D?lookback_days=7"

# Hourly view, last 2 days
curl "http://localhost:8000/api/data/1H?lookback_days=2"

# Weekly view, specific project
curl "http://localhost:8000/api/data/W?lookback_days=30&project_id=mojo1"
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "productivity-dashboard-api"
}
```

---

## 🎯 Next Steps

### Customize the Dashboard
1. Edit chart colors: Modify `chartColors` array in `dashboard_template.html`
2. Add new metrics: Extend `analytics.py` aggregation logic
3. Change default lookback: Update `currentLookbackDays` in template

### Add Script Updates Tracking
Create `scripts/dashboard/script_updates.csv`:
```csv
date,script,description
2025-10-12,01_web_image_selector,Added batch skipping
2025-10-14,04_multi_crop_tool,Optimized crop algorithm
```

Update markers will appear on timeline charts.

### Deploy to Production
1. Use a production ASGI server:
   ```bash
   gunicorn scripts.dashboard.api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```
2. Set up reverse proxy (nginx) for HTTPS
3. Configure CORS for specific origins (update `api.py`)

---

## 📞 Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Run smoke tests to diagnose problems
3. Review server logs for error details

---

**Built with:** Python 3.11, FastAPI, Chart.js
**Status:** ✅ Production-ready


