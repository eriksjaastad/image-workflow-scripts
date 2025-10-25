# Schema Reference

Last Updated: 2025-10-23
Scope: SQLite (AI decisions v3) and snapshot JSON schemas used by the dashboard.

## SQLite: AI Training Decisions v3
- File: `data/training/ai_training_decisions/{project}.db`
- DDL: `data/schema/ai_training_decisions_v3.sql`

### Tables
- `ai_decisions` (PRIMARY)
  - Keys: `group_id` (TEXT PRIMARY KEY), `timestamp` (TEXT), `project_id` (TEXT)
  - Group data: `images` (TEXT JSON array)
  - AI: `ai_selected_index` (INT), `ai_crop_coords` (TEXT JSON), `ai_confidence` (REAL)
  - User: `user_selected_index` (INT), `user_action` (TEXT: approve|crop|reject)
  - Final crop: `final_crop_coords` (TEXT JSON), `crop_timestamp` (TEXT)
  - Dimensions: `image_width` (INT), `image_height` (INT)
  - Flags: `selection_match` (BOOL), `crop_match` (BOOL)

### Indexes
- `idx_project_id`, `idx_timestamp`, `idx_selection_match`, `idx_crop_match`, `idx_user_action`, `idx_batch_number`

### Views
- `ai_performance` → selection/crop accuracy by project
- `incomplete_crops` → user_action=crop and no final crop yet
- `ai_mistakes` → rows where selection_match=0 or crop_match=0

### Example Queries
```sql
-- Mistakes sorted by confidence
SELECT group_id, project_id, ai_confidence
FROM ai_decisions
WHERE selection_match = 0
ORDER BY ai_confidence DESC
LIMIT 50;

-- Incomplete crops
SELECT * FROM incomplete_crops LIMIT 100;

-- Project performance
SELECT * FROM ai_performance WHERE project_id = 'mojo3';
```

## Snapshot Schemas (JSON)
Location: `data/schema/*.json`

### Operation Event v1 (`operation_event_v1.json`)
- Core event from FileTracker logs → normalized.
- Required: `event_id`, `script_id`, `ts_utc`, `event_type`, `day`
- Optional: `operation`, `source_dir`, `dest_dir`, `file_count`, `notes`, `files_sample`, `extra`

### Derived Session v1 (`derived_session_v1.json`)
- Sessions derived from operation events (bounded gaps, active time).
- Required: `source`, `script_id`, `session_id`, `start_ts_utc`, `end_ts_utc`, `active_seconds`, `files_processed`, `params`, `day`
- Params: `gap_min`, `max_gap_contrib`

### Daily Aggregate v1 (`daily_aggregate_v1.json`)
- Pre-aggregated daily statistics for dashboard.
- Required: `by_script`, `by_operation`, `projects_touched`, `total_files_processed`, `total_events`

### Project v1 (`project_v1.json`)
- Project manifest schema for dashboard overlays.
- Required: `project_id`, `title`, `status`, `schema_version`

## Contracts and Labels
- API responses align series to `metadata.baseline_labels[time_slice]`.
- Gaps are padded with zeros; labels are chronological.

## Pointers
- Dashboard API: `Documents/DASHBOARD_API.md`
- Dashboard Guide (full response contract): `Documents/DASHBOARD_GUIDE.md`
- SQLite DDL: `data/schema/ai_training_decisions_v3.sql`
