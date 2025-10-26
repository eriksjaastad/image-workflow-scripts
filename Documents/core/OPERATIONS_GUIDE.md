# Operations Guide
**Status:** Active


Last Updated: 2025-10-23
Audience: AI agents (and Erik) running daily workflows.

## Principles
- Safety first: follow `Documents/FILE_SAFETY_SYSTEM.md`.
- Move, don’t modify; only the AI desktop multi-crop tool writes new images.
- Keep snapshots fresh before dashboard sessions.
- Cleanly start/stop servers; avoid orphaned processes.

## Daily Runs

### 1) Update Snapshots (fast, idempotent)
```bash
python3 scripts/data_pipeline/extract_operation_events_v1.py
python3 scripts/data_pipeline/build_daily_aggregates_v1.py
```

### 2) Launch Dashboard
Option A (launcher):
```bash
python3 scripts/dashboard/run_dashboard.py --host 127.0.0.1 --port 5001
```
Option B (direct server):
```bash
cd scripts/dashboard
python3 productivity_dashboard.py --host 127.0.0.1 --port 5001 --data-dir ../../
```
Then open: `scripts/dashboard/dashboard_template.html`.

### 3) Work Sessions (AI-first)
- Selection/Review: `python3 scripts/01_ai_assisted_reviewer.py`
- Final Cropping: `python3 scripts/02_ai_desktop_multi_crop.py`
- Character Sorter: `python3 scripts/03_web_character_sorter.py`
- Multi-Directory Viewer: `python3 scripts/04_web_multi_directory_viewer.py`
- Finish Project: `python3 scripts/05_finish_project.py`

Always ensure:
- Companions moved together.
- Deletions go to Trash (`send2trash`).
- FileTracker logging present.

## Checkpoints
- After tool runs, confirm FileTracker logs:
```bash
grep -E '"operation":\s*"(move|delete|crop)"' data/file_operations_logs/*.log | tail -n 20
```
- Verify snapshots updated for the day:
```bash
ls data/snapshot/daily_aggregates_v1 | tail -n 3
```
- Dashboard health:
```bash
curl "http://127.0.0.1:5001/api/data/D?lookback_days=7"
```

## Failure Modes & Actions
- Dashboard 500 errors:
  - Check server logs; run `python3 scripts/dashboard/data_engine.py` and `python3 scripts/dashboard/analytics.py` to emit sample JSON.
- Missing data on charts:
  - Ensure snapshots exist; rerun snapshot scripts; reduce lookback.
- Timezone or label misalignments:
  - Confirm naive timestamps; rely on server baseline labels.
- File safety alert (audit failures):
  - Run `python3 scripts/tools/audit_file_safety.py` and fix flagged writes outside safe zones.
- Deletions not in Trash:
  - Use `send2trash` or `mv ~/.Trash/` and validate in Finder.

## Bulk Data Operations Protocol
- Always generate an inspection report first (counts, sample rows with `repr()`, validation status). See `Documents/TECHNICAL_KNOWLEDGE_BASE.md`.

## Clean Shutdown
- Stop Flask server with Ctrl+C; ensure subprocesses/threads terminate and no ports are held.
- Avoid long-running background jobs without explicit purpose.

## References
- Architecture: `Documents/ARCHITECTURE_OVERVIEW.md`
- Safety: `Documents/FILE_SAFETY_SYSTEM.md`
- Dashboard: `Documents/DASHBOARD_GUIDE.md`
- Tech KB: `Documents/TECHNICAL_KNOWLEDGE_BASE.md`
