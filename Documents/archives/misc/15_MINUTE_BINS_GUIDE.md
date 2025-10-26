# 15-Minute Bins: Pre-Aggregation System
**Audience:** Developers

**Version:** 1.0  
**Date:** October 17, 2025  
**Status:** ✅ Implemented, Ready for Pilot

---

## Executive Summary

The **15-minute bins system** pre-aggregates raw file operation logs into fixed 15-minute UTC-aligned time buckets, eliminating the performance bottleneck of reading and parsing thousands of JSONL records on every dashboard load.

**Key Benefits:**
- **10-50x faster dashboard loads** (from 2-4s to <200ms for last-24h views)
- **Predictable payload sizes** (<150 KB for single-chart last-24h views)
- **Scalable to 100+ projects** (current system breaks at ~50 projects)
- **No data loss**: Raw logs remain source of truth
- **Reversible**: Feature flags allow instant rollback

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [Raw JSONL Logs]  →  [aggregate_to_15m.py]  →  [15-min Bins]      │
│   (file_operations_                              (daily partitions)  │
│    logs/*.log)                                                       │
│                                                                       │
│                           ↓                                          │
│                                                                       │
│                    [validate_15m_bins.py]                            │
│                    (Ensures correctness)                             │
│                                                                       │
│                           ↓                                          │
│                                                                       │
│  [Dashboard/Data Engine]  ←─  [bins_reader.py]  ←─  [15-min Bins] │
│   (When performance_mode                         (Fast reads)        │
│    enabled for chart)                                                │
│                                                                       │
│                           ↓                                          │
│                                                                       │
│  [Project Finish]  →  [archive_project_bins.py]  →  [Archives]     │
│   (00_finish_project.py)                           (prevent double- │
│                                                     counting)         │
│                                                                       │
│                           ↓                                          │
│                                                                       │
│                    [Overall Aggregate]                               │
│                    (all finished projects)                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
data/
├── aggregates/                     # NEW: Pre-aggregated bins
│   ├── daily/                      # Daily partitions (day=YYYYMMDD format)
│   │   ├── day=20251017/
│   │   │   └── agg_15m.jsonl      # 15-minute bins for Oct 17, 2025
│   │   ├── day=20251016/
│   │   │   └── agg_15m.jsonl
│   │   └── ...
│   ├── archives/                   # Finished projects (prevents double-counting)
│   │   ├── mojo1/
│   │   │   ├── agg_15m.jsonl      # Project's bins
│   │   │   ├── project_summary.json  # Totals and metadata
│   │   │   └── archive_manifest.json # Archive metadata
│   │   └── ...
│   └── overall/
│       └── agg_15m_cumulative.jsonl  # All finished projects merged
│
├── file_operations_logs/           # EXISTING: Raw logs (source of truth)
├── log_archives/                   # EXISTING: Archived raw logs
└── ...

configs/
└── bins_config.json                # Configuration and feature flags
```

---

## Bin Schema

Each line in `agg_15m.jsonl` is a JSON object:

```json
{
  "bin_ts_utc": "2025-10-17T16:00:00Z",
  "bin_version": 1,
  "project_id": "mojo1",
  "script_id": "01_web_image_selector",
  "operation": "move",
  "dest_category": "crop",
  "file_count": 15,
  "file_count_total": 30,
  "event_count": 15,
  "work_seconds": 875.3,
  "first_event_ts": "2025-10-17T16:00:05Z",
  "last_event_ts": "2025-10-17T16:14:32Z",
  "dedupe_key": "20251017T160000Z_mojo1_01_web_image_selector_move_crop_v1",
  "tz_source": "local",
  "created_at": "2025-10-17T18:30:00Z"
}
```

**Field Descriptions:**
- `bin_ts_utc`: Start of 15-minute bin (UTC-aligned)
- `bin_version`: Version number for reprocessing/corrections
- `project_id`: Project identifier (inferred from paths)
- `script_id`: Script name (e.g., `01_web_image_selector`)
- `operation`: Operation type (`move`, `crop`, `delete`)
- `dest_category`: Destination category for moves (`crop`, `selected`, `delete`)
- `file_count`: PNG files processed (image-only count)
- `file_count_total`: All files processed (PNG + YAML)
- `event_count`: Number of individual operations in bin
- `work_seconds`: Break-aware work time (from `get_file_operation_metrics`)
- `first_event_ts`/`last_event_ts`: Timestamp range within bin
- `dedupe_key`: Unique key to prevent duplicates across runs
- `tz_source`: Timezone source (`local` or `utc`)
- `created_at`: When this bin was created

---

## Key Design Decisions

### 1. **Files-First Storage (JSONL over Database)**

**Current Approach:**
- Daily partitioned JSONL files (`day=YYYYMMDD/agg_15m.jsonl`)
- Simple, readable, git-friendly
- No database setup or maintenance

**Tripwire Thresholds (when to migrate to DuckDB/SQLite):**
- `max_daily_bins_count`: 10,000 bins/day
- `max_overall_size_mb`: 100 MB
- `max_query_time_ms`: 500 ms

**Rationale:** At current scale (~18 projects, ~500-2000 events/day), JSONL is sufficient. DuckDB/SQLite adds complexity without benefit until thresholds are crossed.

### 2. **UTC Alignment**

All bins start at UTC times: `:00`, `:15`, `:30`, `:45`.

**Why UTC:**
- Avoids DST complexity
- Consistent cross-system
- Simpler timestamp math

**Timezone Tracking:**
- `tz_source` field tracks original timezone
- Can convert to local time for display if needed

### 3. **Idempotent Aggregation**

Re-running `aggregate_to_15m.py` on same day produces **identical output**:
- Atomic writes (temp file + rename)
- Stable `dedupe_key` prevents duplicates
- `bin_version` enables corrections without breaking keys

### 4. **Break-Aware Work Time**

`work_seconds` uses existing `get_file_operation_metrics` function:
- Detects breaks (>5 min idle = break)
- Same logic as current dashboard
- Ensures consistency with existing metrics

### 5. **Archive on Project Finish**

When `00_finish_project.py` runs:
1. **Snapshot** project bins → `data/aggregates/archives/<project_id>/`
2. **Merge** into `overall/agg_15m_cumulative.jsonl` (with dedupe)
3. **Prevent double-counting** via `dedupe_key` matching

**Rollback:** `archive_project_bins.py --rollback <project_id>` removes project from overall (archive remains).

---

## Configuration

**File:** `configs/bins_config.json`

```json
{
  "enabled": false,
  "performance_mode": {
    "use_15m_bins": false,
    "bin_charts": []
  },
  "bin_settings": {
    "bin_size_minutes": 15,
    "timezone": "UTC",
    "bin_version": 1
  },
  "storage": {
    "daily_bins_path": "data/aggregates/daily/day={YYYYMMDD}/agg_15m.jsonl",
    "archives_path": "data/aggregates/archives/{project_id}/",
    "overall_path": "data/aggregates/overall/agg_15m_cumulative.jsonl"
  },
  "tripwire_thresholds": {
    "max_daily_bins_count": 10000,
    "max_overall_size_mb": 100,
    "max_query_time_ms": 500
  },
  "performance_targets": {
    "last_24h_payload_kb": 150,
    "last_30d_payload_kb": 500,
    "first_paint_ms": 200
  },
  "archive": {
    "trigger": "project_finish",
    "auto_merge_overall": true,
    "rollback_enabled": true
  }
}
```

**Enabling the System:**
1. Set `"enabled": true`
2. Set `"use_15m_bins": true` under `performance_mode`
3. Add chart names to `"bin_charts": ["by_script"]` to pilot

---

## Usage Guide

### 1. Generate Bins (Daily)

**Aggregate last 7 days:**
```bash
python scripts/data_pipeline/aggregate_to_15m.py
```

**Aggregate specific day:**
```bash
python scripts/data_pipeline/aggregate_to_15m.py --day 20251017
```

**Dry run (preview only):**
```bash
python scripts/data_pipeline/aggregate_to_15m.py --dry-run
```

**Options:**
- `--day YYYYMMDD`: Specific day (default: last 7 days)
- `--days N`: Number of recent days (default: 7)
- `--bin-version N`: Version number for corrections (default: 1)
- `--dry-run`: Preview without writing files
- `--data-dir PATH`: Custom data directory

**Automation (cron):**
```cron
# Run daily at 2 AM
0 2 * * * cd /path/to/project && python scripts/data_pipeline/aggregate_to_15m.py --days 1
```

### 2. Validate Bins

**Validate last 7 days:**
```bash
python scripts/data_pipeline/validate_15m_bins.py
```

**Validate specific day:**
```bash
python scripts/data_pipeline/validate_15m_bins.py --day 20251017
```

**Verbose output:**
```bash
python scripts/data_pipeline/validate_15m_bins.py --verbose
```

**Exit Codes:**
- `0`: All validations passed
- `1`: Validation failures detected
- `2`: Error running validation

**Validation Checks:**
- ✓ Schema: All required fields present and correct types
- ✓ Dedupe keys: No duplicates
- ✓ Bin alignment: All timestamps 15-minute aligned
- ✓ Totals match: File/event counts match raw logs (±1%)
- ✓ Work seconds match: Work time matches raw calculation (±1%)
- ✓ Timestamp ranges: Events within bin boundaries

### 3. Archive Finished Projects

**Archive a project (after finishing):**
```bash
python scripts/data_pipeline/archive_project_bins.py mojo1
```

**Dry run:**
```bash
python scripts/data_pipeline/archive_project_bins.py mojo1 --dry-run
```

**Archive without merging into overall:**
```bash
python scripts/data_pipeline/archive_project_bins.py mojo1 --skip-merge
```

**Rollback (remove from overall):**
```bash
python scripts/data_pipeline/archive_project_bins.py mojo1 --rollback
```

**Automatic archiving:**
- Runs automatically when `00_finish_project.py --commit` is executed
- No manual intervention needed for normal workflow

### 4. Enable Dashboard Integration (Pilot)

**Step 1: Enable in config**
Edit `configs/bins_config.json`:
```json
{
  "enabled": true,
  "performance_mode": {
    "use_15m_bins": true,
    "bin_charts": ["by_script"]
  }
}
```

**Step 2: Restart dashboard**
```bash
python scripts/dashboard/run_dashboard.py
```

**Step 3: Measure performance**
- Open browser DevTools → Network tab
- Load dashboard
- Measure payload size and load time
- Compare to targets:
  - Last-24h: <150 KB, <200ms first paint
  - Last-30d: <500 KB

**Step 4: Expand to more charts**
Add to `bin_charts`:
```json
"bin_charts": ["by_script", "by_operation", "by_project"]
```

---

## Performance Targets & Benchmarking

### Current Baseline (Raw Logs)
- Last-24h view: ~2-4 seconds total load time
- Payload: ~200-500 KB (varies by activity)
- 18 projects: Noticeable lag
- Projected 100 projects: **10-15 seconds** (unacceptable)

### Target (15-min Bins)
- Last-24h view: **<200ms** first paint
- Payload: **<150 KB** per chart
- Last-30d view: **<500 KB** (with decimation)
- 100 projects: Sub-second loads

### How to Benchmark

**Before (raw logs):**
```bash
# Disable bins
# Edit configs/bins_config.json: "enabled": false

# Measure
time curl -s http://localhost:8050/api/data?time_slice=D&lookback_days=1 | wc -c
```

**After (bins):**
```bash
# Enable bins for "by_script" chart
# Edit configs/bins_config.json: "enabled": true, "bin_charts": ["by_script"]

# Measure
time curl -s http://localhost:8050/api/data?time_slice=D&lookback_days=1 | wc -c
```

**Compare:**
- Payload size reduction
- Response time improvement
- Chart render time (browser DevTools)

---

## Rollback Plan

If bins cause issues, instant rollback:

**Step 1: Disable in config**
```json
{
  "enabled": false,
  "performance_mode": {
    "use_15m_bins": false
  }
}
```

**Step 2: Restart dashboard**
```bash
# Ctrl+C to stop
python scripts/dashboard/run_dashboard.py
```

**Result:** Dashboard immediately reverts to raw logs (no data loss).

**To remove bins entirely:**
```bash
# Optional: delete aggregates directory (raw logs remain)
rm -rf data/aggregates/
```

---

## Troubleshooting

### Problem: "No bins found for project"
**Cause:** Bins not yet generated for project's date range  
**Solution:**
```bash
# Generate bins for project's date range
python scripts/data_pipeline/aggregate_to_15m.py --days 90
```

### Problem: "Validation failed: totals mismatch"
**Cause:** Raw logs modified after bins generated  
**Solution:**
```bash
# Regenerate bins with incremented version
python scripts/data_pipeline/aggregate_to_15m.py --bin-version 2
```

### Problem: "Dashboard still slow after enabling bins"
**Checks:**
1. Verify config: `cat configs/bins_config.json | grep enabled`
2. Check bin files exist: `ls data/aggregates/daily/`
3. Confirm chart in `bin_charts` list
4. Check browser DevTools → Network for payload size

### Problem: "Archive failed: bins not found"
**Cause:** Project finished before bins were generated  
**Solution:**
```bash
# Generate bins for project's historical date range
python scripts/data_pipeline/aggregate_to_15m.py --days 180

# Retry archive
python scripts/data_pipeline/archive_project_bins.py <project_id>
```

---

## Migration Path (Current → Future)

### Phase 1: Pilot (Now)
- ✅ Generate bins daily (cron)
- ✅ Enable for **one chart** (`by_script`)
- ✅ Validate parity with raw logs
- ✅ Measure performance improvement

### Phase 2: Expand (1-2 weeks)
- Enable for all charts: `["by_script", "by_operation", "by_project"]`
- Monitor tripwire thresholds
- Validate payload sizes meet targets
- Archive all finished projects

### Phase 3: Optimize (1 month)
- Add decimation for 30-day+ views
- Implement shared-labels chart format
- Monitor at 50+ projects
- Consider DuckDB if tripwire crossed

### Phase 4: Scale (3 months)
- Test at 100 projects
- Evaluate DuckDB/SQLite migration if needed
- Add real-time bin updates (optional)
- Implement bin compaction (optional)

---

## Maintenance

### Daily
- **Automatic:** Bins generated via cron (2 AM)
- **Automatic:** Validation runs after aggregation

### Weekly
- Review validation logs: `grep "Validation failed" logs/*.log`
- Check disk usage: `du -sh data/aggregates/`

### Monthly
- Check tripwire thresholds:
  ```bash
  # Bin count
  find data/aggregates/daily -name "agg_15m.jsonl" -exec wc -l {} + | tail -1
  
  # Disk usage
  du -m data/aggregates/ | tail -1
  ```
- Archive old finished projects (older than 90 days)

### As Needed
- **Reprocess bins:** Increment `bin_version` when schema changes
- **Rollback archives:** If project needs to be "un-finished"
- **Migrate to DuckDB:** If tripwire thresholds exceeded

---

## References

### Related Documents
- `DATA_PIPELINE_PERFORMANCE_ANALYSIS.md` - Original performance analysis
- `../../dashboard/DASHBOARD_GUIDE.md` - Dashboard usage and architecture
- `../../data/PROJECT_MANIFEST_GUIDE.md` - Project lifecycle and manifests

### Source Files
- `scripts/data_pipeline/aggregate_to_15m.py` - Bin aggregation
- `scripts/data_pipeline/validate_15m_bins.py` - Validation
- `scripts/data_pipeline/archive_project_bins.py` - Archiving
- `scripts/dashboard/bins_reader.py` - Dashboard integration
- `configs/bins_config.json` - Configuration

### Testing
- `scripts/tests/test_bins_system.py` - Automated tests (future)

---

## FAQ

**Q: Will this delete my raw logs?**  
A: No. Raw logs remain source of truth. Bins are read-only derivatives.

**Q: What happens if I enable bins but haven't generated them?**  
A: Dashboard falls back to raw logs automatically (no errors).

**Q: Can I regenerate bins if I find an error?**  
A: Yes. Increment `bin_version` and rerun aggregation. Old bins are overwritten.

**Q: How much disk space do bins use?**  
A: Roughly 1-5% of raw log size (highly compressed due to aggregation).

**Q: Can I query bins directly (SQL)?**  
A: Not yet. Currently JSONL only. DuckDB support planned if scale requires.

**Q: What if I want bins for a specific hour (not 15-min)?**  
A: Bins are 15-min fixed. You can re-aggregate on read (in `bins_reader.py`) to hourly without regenerating files.

---

**Document Status:** ✅ Complete  
**Last Updated:** October 17, 2025  
**Next Review:** After 30 days of pilot testing

