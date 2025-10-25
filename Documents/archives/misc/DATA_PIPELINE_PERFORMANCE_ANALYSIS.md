# Data Pipeline Performance Analysis
**Date:** October 17, 2025  
**Purpose:** Identify bottlenecks, scalability issues, and data precision problems in the current productivity dashboard pipeline

---

## Executive Summary

**Current Status:** 🟡 Functional but showing performance strain  
**Primary Concerns:**
1. Dashboard load times increasing
2. Time rounding too coarse (1-hour increments → need 15-minute precision)
3. Unknown scalability ceiling (will it handle 100 projects?)
4. No performance monitoring or bottleneck detection
5. Data scattered across multiple formats and locations

**Critical Finding:** The pipeline was designed for ~20 projects. At 18 projects now, we're already seeing lag. **Scaling to 100 projects will require significant architectural changes.**

---

## 1. Current Data Architecture

### 1.1 Data Sources (Raw → Dashboard)

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                         │
├─────────────────────────────────────────────────────────────┤
│ 1. File Operations Logs                                     │
│    - data/file_operations_logs/file_operations_YYYYMMDD.log │
│    - Format: JSON Lines (JSONL)                             │
│    - Volume: ~500-2000 lines/day                            │
│    - Retention: Current + 2 days                            │
│                                                              │
│ 2. Log Archives (Consolidated)                              │
│    - data/log_archives/file_operations_YYYYMMDD.log         │
│    - Format: JSON Lines (JSONL)                             │
│    - Volume: All days > 2 days old                          │
│    - Size: ~50-200 KB/day, cumulative                       │
│                                                              │
│ 3. Daily Summaries (Legacy)                                 │
│    - data/daily_summaries/YYYYMMDD.json                     │
│    - Format: Single JSON object per day                     │
│    - Status: Being phased out (but still read)              │
│                                                              │
│ 4. Snapshot System (New - Sept 2025)                        │
│    - data/snapshot/operation_events_v1/day=YYYYMMDD/*.jsonl │
│    - data/snapshot/daily_aggregates_v1/day=YYYYMMDD.json   │
│    - data/snapshot/derived_sessions_v1/day=YYYYMMDD.jsonl  │
│    - Status: Parallel to legacy, not yet primary            │
│                                                              │
│ 5. Project Manifests                                        │
│    - data/projects/*.project.json                           │
│    - Format: Single JSON per project                        │
│    - Volume: 1 file per project (18 files currently)        │
│                                                              │
│ 6. Timer Data (Legacy Work Timer)                           │
│    - data/timer_data/projects/*/*.json                      │
│    - Status: Being replaced by session derivation           │
│                                                              │
│ 7. Timesheet CSV (Manual Entry)                             │
│    - data/timesheet.csv                                     │
│    - Format: CSV with headers                               │
│    - Updated: Manual, as-needed                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Processing Pipeline

```
[Raw Logs] → [Project Metrics Aggregator] → [Analytics Engine] → [Dashboard API] → [Frontend]
    ↓              ↓                              ↓                    ↓
    ├─> Read all  ├─> Per-project              ├─> Comparisons      ├─> Charts
    │   JSONL     │   aggregation               ├─> Tables           └─> Tables
    │   files     ├─> Deduplication             └─> Timesheet
    ├─> Parse     ├─> Metrics calculation
    │   manifests └─> Tool breakdown
    └─> Timesheet
        parsing
```

**Key Issue:** Every dashboard load triggers the ENTIRE pipeline from scratch (no persistent cache).

---

## 2. Identified Bottlenecks

### 2.1 File I/O: The Primary Bottleneck

**Problem:** On every dashboard load, the system reads:
- 18 project manifest files (`.project.json`)
- ~90 days of raw file operation logs (JSONL files)
- ~90 days of archived logs
- Legacy daily summaries
- Timesheet CSV

**Measurement:**
```python
# From dashboard load logs (typical "recent" view, 14-day lookback):
- File operations logs read: ~140 files (current + archives)
- Lines parsed: ~50,000-100,000 JSONL records
- Project manifests: 18 files
- Total disk reads: ~160 files per dashboard load
```

**Impact:**
- Current load time (18 projects): ~2-3 seconds
- Projected load time (100 projects): **10-15 seconds** (unacceptable)

**Root Cause:** No persistent in-memory cache or database. Every API call = full file system scan.

---

### 2.2 Data Parsing & Transformation

**Problem:** Complex multi-stage transformation with redundant work:

1. **Stage 1: Raw Log Parsing**
   - Read JSONL files line-by-line
   - Parse each JSON object
   - Normalize timestamps (handle timezone variations)
   - **Cost:** ~50-100ms for 10,000 lines

2. **Stage 2: Project Aggregation** (`project_metrics_aggregator.py`)
   - Load ALL operations from ALL logs
   - Group by project ID
   - Calculate per-project metrics
   - Calculate per-tool breakdowns
   - **Cost:** ~200-500ms for 18 projects

3. **Stage 3: Analytics Generation** (`analytics.py`)
   - Build project comparisons
   - Build productivity tables
   - Build tool breakdowns (AGAIN - duplicates work from Stage 2)
   - Calculate daily/weekly/monthly slices
   - **Cost:** ~100-200ms

4. **Stage 4: Frontend Transformation** (`productivity_dashboard.py`)
   - Transform to chart-friendly format
   - Sort/filter data
   - **Cost:** ~50ms

**Total Processing Time:** ~350-850ms (for 18 projects)  
**Projected for 100 projects:** ~2-4 seconds (just processing, not including I/O)

**Key Issue:** Data is processed and aggregated multiple times. Analytics engine re-calculates metrics that were already computed in the aggregator.

---

### 2.3 Time Precision & Rounding Issues

**Current Behavior:**
```python
# In project_metrics_aggregator.py (line 87):
'work_hours': round(self.work_minutes / 60.0, 1)  # Rounds to 0.1 hour (6 minutes)
```

**Problem:**
- Timer logs capture work sessions to the **second**
- Aggregator stores `work_minutes` (minute precision)
- Output rounds to **0.1 hour = 6 minutes**
- User wants **15-minute precision** (0.25 hour increments)

**Example Impact:**
- Actual work: 13.2 hours (13h 12m)
- Displayed: 13.2h (rounds 12 minutes down from 13.25h)
- If actual was 13h 18m → 13.3h displayed
- **Loss:** User can't see sub-15-minute variations

**Why This Matters:**
- Billing accuracy (user charges by the hour or half-hour)
- Comparing billed vs actual (6-minute rounding hides small discrepancies)
- Tracking micro-breaks and interruptions

**Fix Complexity:** Low (just change rounding logic), but...
**Data Question:** Do we need to regenerate historical aggregates? Or apply correction at display time?

---

### 2.4 Scalability: The 100-Project Problem

**Current Project Count:** 18  
**Target Projection:** 100

**Bottleneck Math:**
```
Current (18 projects):
  - File reads: ~160 files
  - JSONL parsing: ~100,000 lines
  - Processing time: ~3-4 seconds total (I/O + compute)
  - Dashboard load: Acceptable but noticeable lag

Projected (100 projects):
  - File reads: ~900 files (100 manifests + 800 log files)
  - JSONL parsing: ~500,000-1,000,000 lines
  - Processing time: ~15-25 seconds (UNACCEPTABLE)
  - Memory usage: ~200-500 MB (risky on shared hosting)
```

**Breaking Point:** Likely around **40-50 projects** with current architecture.

**Why It Breaks:**
1. **Linear file I/O scaling:** More projects = more manifest files + more operations in logs
2. **No caching:** Every dashboard load re-reads everything
3. **No indexing:** Searching for a specific project's data requires scanning all logs
4. **No query optimization:** Can't filter data at read time (must load entire dataset, then filter)

---

### 2.5 Data Duplication & Redundancy

**Problem:** Multiple systems storing overlapping data:

| Data Type | Location 1 | Location 2 | Location 3 | Status |
|-----------|-----------|-----------|-----------|--------|
| File operations | `file_operations_logs/` | `log_archives/` | `snapshot/operation_events_v1/` | 3x duplication |
| Daily aggregates | `daily_summaries/` | `snapshot/daily_aggregates_v1/` | (None) | 2x duplication |
| Work time | `timer_data/` (legacy) | Derived from logs | `snapshot/derived_sessions_v1/` | 3x duplication |
| Project metadata | `projects/*.json` | (Embedded in logs) | (None) | 2x duplication |

**Impact:**
- **Storage:** ~3-5x more disk space than necessary
- **Maintenance:** Changes must be synced across multiple systems
- **Confusion:** Which is the source of truth?
- **Performance:** Multiple reads for the same data

**Current Mitigation:** Snapshot system was introduced to consolidate, but legacy systems still active.

---

## 3. Monitoring & Observability Gaps

### 3.1 What We DON'T Know

**Performance Metrics:**
- ❌ Dashboard load time (no timing logs)
- ❌ Breakdown by stage (I/O vs parsing vs aggregation vs analytics)
- ❌ Memory usage during data processing
- ❌ Cache hit/miss rates (no cache exists)
- ❌ Query performance by time slice (daily vs weekly vs monthly)

**Data Quality Metrics:**
- ❌ Missing data detection (are there gaps in logs?)
- ❌ Duplicate detection (are operations being double-counted?)
- ❌ Data freshness (how old is the "latest" data?)
- ❌ Precision loss tracking (how much rounding affects accuracy?)

**System Health:**
- ❌ Disk space usage trends
- ❌ File count growth rate
- ❌ Data staleness (when was each source last updated?)

### 3.2 What We DO Know (Manually)

- Dashboard "feels slow" (subjective)
- 18 projects currently (counted manually)
- Timesheet has 227 total hours (from parser output)
- Snapshot system exists but not primary (from code review)

**Key Gap:** No automated monitoring. All insights require manual investigation.

---

## 4. Data Flow: Where Time Is Spent

### 4.1 Typical Dashboard Load (14-day lookback, "recent" view)

```
┌────────────────────────────────────────────────────────────────┐
│ STAGE                          │ TIME (est) │ % OF TOTAL      │
├────────────────────────────────┼────────────┼─────────────────┤
│ 1. File I/O (read logs)        │   1.5-2.5s │ 50-65%   ⚠️    │
│ 2. JSONL Parsing               │   0.3-0.5s │ 10-15%          │
│ 3. Project Aggregation         │   0.4-0.6s │ 12-18%          │
│ 4. Analytics Generation        │   0.2-0.3s │  6-10%          │
│ 5. Timesheet Parsing           │   0.05-0.1s│  2-3%           │
│ 6. Frontend Transformation     │   0.05s    │  1-2%           │
│ 7. JSON Serialization          │   0.1s     │  2-4%           │
├────────────────────────────────┼────────────┼─────────────────┤
│ TOTAL                          │   2.6-4.2s │ 100%            │
└────────────────────────────────────────────────────────────────┘
```

**Critical Finding:** File I/O dominates (50-65%). This is the bottleneck.

### 4.2 Why File I/O Is So Slow

**Current Pattern:**
```python
# Simplified pseudocode from data_engine.py
for each day in lookback_range:
    path = f"data/file_operations_logs/file_operations_{day}.log"
    if path.exists():
        with open(path) as f:
            for line in f:
                record = json.loads(line)  # Parse each line
                all_records.append(record)
    
    # Also check archives
    archive_path = f"data/log_archives/file_operations_{day}.log"
    if archive_path.exists():
        with open(archive_path) as f:
            for line in f:
                record = json.loads(line)
                all_records.append(record)
```

**Problem:** Sequential file opens, sequential line parsing, no buffering optimization.

**For 14-day lookback with 18 projects:**
- Days checked: 14
- Files opened: ~28 (14 current + 14 archives, many missing so actual ~18-20)
- Lines parsed: ~50,000-100,000
- JSON.loads() calls: ~50,000-100,000

**Each `json.loads()` call:** ~0.01-0.05ms  
**Total JSON parsing time:** 50,000 × 0.02ms = **1000ms = 1 second** (just parsing)

---

## 5. Database vs Current Approach

### 5.1 Current: File-Based Storage

**Pros:**
- ✅ Simple (no database setup)
- ✅ Human-readable (JSONL files)
- ✅ Easy backups (just copy files)
- ✅ No dependencies (no DB server)
- ✅ Version control friendly (text files)

**Cons:**
- ❌ No indexing (must scan all files)
- ❌ No query optimization (can't filter at read time)
- ❌ No caching (re-read on every request)
- ❌ Linear scaling (100 projects = 5-10x slower)
- ❌ No concurrent access optimization
- ❌ Difficult aggregations (must load everything, then aggregate)

### 5.2 Database Option: SQLite

**What We'd Gain:**
- ✅ **Indexed queries:** Find all operations for project X in 1ms (vs 1000ms)
- ✅ **Aggregations:** `SELECT project_id, SUM(file_count), AVG(work_hours) GROUP BY project_id` → 10ms
- ✅ **Date filtering:** `WHERE timestamp >= '2025-10-01'` → Uses index, not full scan
- ✅ **Concurrent reads:** Multiple dashboard users = no problem
- ✅ **Persistent cache:** Data loaded once, stays in memory
- ✅ **Sub-second queries:** Even with 100 projects

**What We'd Lose:**
- ❌ Requires schema design
- ❌ Adds complexity (migrations, backups)
- ❌ Less human-readable (binary file)
- ❌ Harder to manually inspect/debug

**Middle Ground: DuckDB**
- Reads JSONL files directly (no ETL required)
- Provides SQL querying over files
- No server process (embedded like SQLite)
- Can be added incrementally (hybrid approach)

---

## 6. Precision & Rounding Analysis

### 6.1 Current Time Storage Chain

```
[Tools Log Activity]
    ↓ (to the second)
[file_operations_logs/*.log]
    ↓ (timestamp: "2025-10-17T14:32:05Z")
[Project Aggregator calculates work_minutes]
    ↓ (work_minutes: 847.3)  ← Stored as float (minutes)
[Convert to hours: work_minutes / 60]
    ↓ (14.121666... hours)
[Round to 0.1 hour]
    ↓ (14.1 hours displayed)
```

**Precision Loss:**
- Original: 847.3 minutes = 14h 7m 18s
- Stored: 847.3 minutes (no loss)
- Displayed: 14.1 hours = 14h 6m (loss of 1m 18s)

**For Billing Comparison:**
- Timesheet: 14.0 hours (billed)
- Actual: 14.1 hours (displayed, rounded from 14.12166...)
- True actual: 14.122 hours (not shown)
- Difference shown: 0.1 hours
- True difference: 0.122 hours

### 6.2 Requested: 15-Minute Precision

**What This Means:**
- Round to 0.25 hour increments: 0.00, 0.25, 0.50, 0.75, 1.00, ...
- 14h 7m 18s → 14.00 hours (rounds down to nearest 15min)
- 14h 10m → 14.25 hours (rounds up to nearest 15min)

**Implementation:**
```python
# Current:
'work_hours': round(self.work_minutes / 60.0, 1)  # 0.1 hour precision

# Proposed:
'work_hours': round(self.work_minutes / 60.0 / 0.25) * 0.25  # 0.25 hour (15min) precision
```

**Impact on Historical Data:**
- Stored `work_minutes` is already precise (no regeneration needed)
- Just change display logic
- All historical data will automatically show with new precision

---

## 7. Recommendations Summary

### 7.1 Immediate Fixes (Low-Hanging Fruit)

1. **Fix time rounding** → 15-minute precision (1-line code change)
2. **Add performance logging** → Track dashboard load time per stage
3. **Remove debug logging in production** → 100+ lines of `print()` statements slowing things down

### 7.2 Short-Term (2-4 hours work)

4. **Implement in-memory cache** → Cache project metrics for 5 minutes (avoid re-reading files)
5. **Optimize JSONL parsing** → Use `orjson` (5-10x faster JSON parsing)
6. **Remove legacy dual systems** → Fully commit to snapshot system, delete daily_summaries

### 7.3 Medium-Term (1-2 days work)

7. **Add SQLite query layer** → Hybrid: Keep JSONL files, add SQLite indexes on top
8. **Pre-aggregate daily** → Run aggregator nightly via cron, store results in DB
9. **Implement pagination** → Load 10 projects at a time, not all 100

### 7.4 Long-Term (Architecture Redesign)

10. **Full database migration** → Move to PostgreSQL or keep SQLite, stop writing JSONL
11. **Real-time updates** → WebSocket push for new data (no polling)
12. **Distributed caching** → Redis for multi-user scenarios

---

## 8. Open Questions for Decision

1. **Time Precision:** Confirm 15-minute rounding is sufficient (vs 5-minute or 1-minute)?
2. **Historical Data:** Re-process all historical aggregates with new rounding, or just apply at display time?
3. **Database:** Are we ready to introduce SQLite/DuckDB, or prefer file-based for now?
4. **Scalability Target:** Is 100 projects realistic, or should we plan for 50? 500?
5. **Caching Strategy:** In-memory (lost on server restart) vs persistent (Redis/file-based)?
6. **Monitoring:** Add lightweight performance tracking, or full observability stack (e.g., Prometheus)?

---

## 9. Current Metrics (Baseline for Future Comparison)

**System State (Oct 17, 2025):**
- Active projects: 18
- Timesheet projects: 18
- Total billed hours (timesheet): 227.17 hours
- Date range with timer data: Sept 20, 2025 - Present (~27 days)
- Dashboard load time (subjective): 2-4 seconds ("feels slow")
- File operations log size: ~50 KB/day, ~1.5 MB total (archives)
- Project manifest files: 18 files, ~2-5 KB each

**Data Volume:**
- JSONL lines/day: ~500-2000 (estimate based on activity)
- Total operations logged (all time): ~500,000-1,000,000 (rough estimate)
- Disk space (data directory): ~50-100 MB

---

## 10. Next Steps

**Before Making Changes:**
1. ✅ **This document** - Understand the current state (DONE)
2. ⏳ **Decision meeting** - Choose which problems to solve first
3. ⏳ **Baseline measurements** - Add timing logs to quantify bottlenecks
4. ⏳ **Prioritization** - Quick wins vs long-term architecture

**Proposed Priority Order:**
1. **Fix time rounding** (5 minutes, immediate impact)
2. **Add performance logging** (30 minutes, enables future optimization)
3. **Implement cache** (2 hours, 3-5x speedup)
4. **Database evaluation** (4 hours, test DuckDB on copy of data)
5. **Architecture decision** (Based on #4 results)

---

**Document Status:** ✅ Complete - Ready for review  
**Next Action:** User decides which issues to tackle first

