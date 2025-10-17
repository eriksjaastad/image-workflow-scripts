# Data Pipeline Scripts

Infrastructure scripts for the snapshot-based data pipeline. These are **not** run daily - they're for data extraction, validation, and analysis.

## 📂 Directory Structure

This directory contains backend data processing scripts that support the dashboard and analytics.

## 📊 Extraction Scripts

Run these to extract and normalize raw data into snapshots:

```bash
# Extract operation events from logs
python scripts/data_pipeline/extract_operation_events_v1.py

# Extract legacy timer sessions
python scripts/data_pipeline/extract_timer_sessions_v1.py

# Extract progress snapshots
python scripts/data_pipeline/extract_progress_snapshots_v1.py

# Extract project manifests
python scripts/data_pipeline/extract_projects_v1.py

# Derive sessions from operation events
python scripts/data_pipeline/derive_sessions_from_ops_v1.py

# Build daily aggregates for dashboard
python scripts/data_pipeline/build_daily_aggregates_v1.py
```

## 🔍 Validation & Analysis

```bash
# Validate snapshot integrity
python scripts/data_pipeline/validate_snapshots_v1.py

# Query snapshots with SQL (requires: pip install duckdb)
python scripts/data_pipeline/query_snapshots_duckdb.py

# Backfill historical data
python scripts/data_pipeline/backfill_snapshots_v1.py --days 90
```

## 🚀 Optional Upgrades

```bash
# Convert to Parquet format (requires: pip install pyarrow pandas)
python scripts/data_pipeline/convert_to_parquet_v1.py --all
```

## 🎯 When to Run These

**Automatically**: The dashboard reads from snapshots automatically. You don't need to run these unless:

- You want to refresh snapshot data
- You're doing data analysis with SQL queries
- You need to backfill historical data
- You're validating data integrity

**Daily extraction** (optional):
```bash
# Add to cron if you want automatic daily updates
python scripts/data_pipeline/extract_operation_events_v1.py
python scripts/data_pipeline/derive_sessions_from_ops_v1.py
python scripts/data_pipeline/build_daily_aggregates_v1.py
```

## 📚 Documentation

See `Documents/DATA_DEEP_DIVE_COMPLETE.md` for comprehensive documentation.

## ⚡ Quick Reference

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `extract_operation_events_v1.py` | Normalize raw file operations | Weekly or as needed |
| `derive_sessions_from_ops_v1.py` | Create sessions from events | After extracting events |
| `build_daily_aggregates_v1.py` | Pre-aggregate for dashboard | After extracting events |
| `validate_snapshots_v1.py` | Check data integrity | After any extraction |
| `query_snapshots_duckdb.py` | SQL queries on snapshots | Anytime for analysis |
| `backfill_snapshots_v1.py` | Process historical data | One-time or monthly |

**Location of snapshots**: `snapshot/` directory in project root
**Location of schemas**: `schema/` directory in project root

