# Backup Runbook (Local + Cloud)
**Status:** Active
**Audience:** Developers


Last updated: 2025-10-25

## What we back up (sources)
- data/file_operations_logs/ (raw JSON logs, daily + rolling)
- data/log_archives/ (compressed daily logs, if present)
- data/snapshot/** (operation_events_v1, daily_aggregates_v1, derived_sessions_v1)
- data/ai_data/backups/** (SQLite backup sets, migrations, etc.)
- Any live *.db under data/** (safety snapshot)

## Where backups go (local)
- Root: `~/project-data-archives/image-workflow/`
- Daily folder: `~/project-data-archives/image-workflow/YYYY-MM-DD/`
- Weekly rollups: `~/project-data-archives/image-workflow/weekly/weekly_YYYYMMDD_YYYYMMDD.tar.zst`
- Logs: `~/backup.log`

## Where backups go (cloud)
- rclone remote: `gbackup` (type: drive), restricted via `root_folder_id` to a single Google Drive folder
- Daily upload: `gbackup:YYYY-MM-DD/`
- Weekly rollups: `gbackup:weekly-rollups/`

## One-time setup (already done)
- Daily local backup script: `scripts/backup/daily_backup.py`
- Weekly rollup script: `scripts/backup/weekly_rollup.py`
- Cron installer for local backups: `scripts/backup/setup_cron_backup.sh`
- rclone remote `gbackup` configured and scoped to one Drive folder

## Schedules
- 02:10 local daily backup → `~/project-data-archives/image-workflow/YYYY-MM-DD/`
- 03:00 daily cloud upload (copy → check → delete that day’s local folder)
- 03:45 Sunday weekly rollup (packs last 7 day folders, uploads to Drive, keeps 12 local)

List current cron jobs:
```
crontab -l
```
Remove backup lines if needed:
```
crontab -l | grep -v -e 'gbackup:' -e 'weekly_rollup.py' | crontab -
```

## Manual operations
Daily local backup (now):
```
source .venv311/bin/activate 2>/dev/null || true
python scripts/backup/daily_backup.py --dest ~/project-data-archives/image-workflow
```
Upload today (copy → check → delete local):
```
DAY="$(date +%F)"
rclone copy ~/project-data-archives/image-workflow/"$DAY"/ gbackup:"$DAY"/ --log-file=~/backup.log --log-level INFO
rclone check ~/project-data-archives/image-workflow/"$DAY"/ gbackup:"$DAY"/ --one-way --size-only --log-file=~/backup.log --log-level INFO
rm -rf ~/project-data-archives/image-workflow/"$DAY"
```
Weekly rollup (now):
```
python scripts/backup/weekly_rollup.py >> ~/backup.log 2>&1
```

## Snapshots (rebuild from raw logs)
Rebuild snapshots safely any time (no data loss):
```
python scripts/data_pipeline/extract_operation_events_v1.py
python scripts/data_pipeline/build_daily_aggregates_v1.py
python scripts/data_pipeline/derive_sessions_from_ops_v1.py
```
Snapshot layout:
- operation_events_v1: `data/snapshot/operation_events_v1/day=YYYYMMDD/events.jsonl`
- daily_aggregates_v1: `data/snapshot/daily_aggregates_v1/day=YYYYMMDD/aggregate.json`
- derived_sessions_v1: `data/snapshot/derived_sessions_v1/day=YYYYMMDD/`

## Verification
Daily verify (pick a day):
```
DAY="$(date +%F)"
rclone check ~/project-data-archives/image-workflow/"$DAY"/ gbackup:"$DAY"/ --one-way --size-only --log-file=~/backup.log --log-level INFO
```
Remote sanity:
```
rclone lsd gbackup:
rclone about gbackup:
```

## Restore
From weekly rollup:
```
# Download weekly_YYYYMMDD_YYYYMMDD.tar.zst from gbackup:weekly-rollups/
zstd -d weekly_YYYYMMDD_YYYYMMDD.tar.zst
mkdir -p ~/restore && tar -xf weekly_YYYYMMDD_YYYYMMDD.tar -C ~/restore
```
From a daily folder:
```
rclone copy gbackup:"YYYY-MM-DD"/ ~/restore/YYYY-MM-DD/
```

## Safety & retention
- Scripts only copy and record manifests; no in-place file modifications
- Daily upload deletes the local day folder only after `rclone check` passes
- Weekly rollup keeps 12 local archives; cloud retains all unless pruned

## Disaster recovery notes
- If `data/log_archives/` is missing, older compressed daily logs were likely removed outside of git/Trash (unlink). We can still rebuild snapshots for any days covered by raw logs in `data/file_operations_logs/`.
- To search for missing archives system-wide (manual):
```
mdfind -name 'file_operations_*.log.gz'
```
- Recreate daily snapshots from surviving raw logs (see Snapshots section).

## File locations (quick reference)
- Local daily: `~/project-data-archives/image-workflow/YYYY-MM-DD/`
- Local weekly: `~/project-data-archives/image-workflow/weekly/`
- Cloud daily: `gbackup:YYYY-MM-DD/`
- Cloud weekly: `gbackup:weekly-rollups/`
- Logs: `~/backup.log`
