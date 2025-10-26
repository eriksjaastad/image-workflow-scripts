# Data Consolidation System

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers
**Estimated Reading Time:** 4 minutes

## Purpose
Describe how dashboard data is consolidated from daily summaries and detailed logs for tests and local development.

## Data Sources
- Daily summaries: `data/daily_summaries/`
- File operation logs: `data/file_operations_logs/` and `data/log_archives/`

## Consolidation Logic
- Prefer daily aggregates when present to speed reads.
- Fill gaps with per-operation logs.
- Normalize timestamps to UTC Z and align to baseline labels (`15min`, `1H`, `D`, `W`, `M`).
- Produce merged tables for:
  - Images per hour (IPH)
  - Operation counts over time
  - Project comparisons

## Testing Considerations
- Provide small fixture windows (e.g., 7–30 days) for fast runs.
- Ensure deterministic ordering for unit tests.
- Validate that merged series length matches baseline labels.

## Operational Notes
- Consolidation is executed inside the dashboard analytics layer.
- For local dev, rebuild aggregates as needed to reflect the latest logs.

## Related Documents
- `dashboard/DASHBOARD_GUIDE.md`
- `dashboard/DASHBOARD_API.md`
- `reference/TECHNICAL_KNOWLEDGE_BASE.md`


