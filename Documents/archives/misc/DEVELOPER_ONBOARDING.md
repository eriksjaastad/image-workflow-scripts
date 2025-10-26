# Developer Onboarding (AI-Agent Focused)
**Status:** Active


Last Updated: 2025-10-23
Audience: AI assistants starting a new chat session for this repository.

## 0) Ground Rules (must read)
- Never modify production images or companions. Only `scripts/02_ai_desktop_multi_crop.py` (and legacy `02_ai_desktop_multi_crop.py`) write NEW crops.
- Writes allowed only to `Documents/`, `data/` (append-only snapshots/JSON/CSV), and `sandbox/`.
- Use macOS Trash for deletions (prefer `send2trash`); do not hard-delete.
- Log all file operations via FileTracker.
- Follow `Documents/FILE_SAFETY_SYSTEM.md`.

## 1) 10-Minute Orientation
1. Read:
   - `Documents/ARCHITECTURE_OVERVIEW.md`
   - `Documents/FILE_SAFETY_SYSTEM.md`
   - `Documents/DASHBOARD_GUIDE.md` (API contract section)
   - `Documents/TECHNICAL_KNOWLEDGE_BASE.md` (SQLite v3 decisions, tests)
2. Confirm repo map and safe zones align with your planned actions.

## 2) Operational Identity (what kind of “developer” you are)
- You are an AI agent operating via chat. Treat each session as a fresh onboarding:
  - Validate assumptions against docs and code.
  - Prefer targeted reads (≤25 files) and semantic searches over bulk reads.
  - Respect cost gates (see Section 6).

## 3) Common Flows
- Dashboard:
  - Start API: `python3 scripts/dashboard/productivity_dashboard.py`
  - Open UI: `scripts/dashboard/dashboard_template.html`
  - Test endpoint: `curl "http://localhost:8000/api/data?slice=D&lookback_days=7"`
- Data pipeline (snapshots up-to-date):
  - `python3 scripts/data_pipeline/extract_operation_events_v1.py`
  - `python3 scripts/data_pipeline/build_daily_aggregates_v1.py`
- AI training decisions v3:
  - Reviewer logs to `data/training/ai_training_decisions/{project}.db`
  - `.decision` sidecars link reviewer → crop tool → DB update

## 4) Safety Checklist Before Any Write
- Is the write a move/delete/create-new in safe zones?
- Are companions handled together? (`find_all_companion_files`, `move_file_with_all_companions`)
- Are we creating NEW files (not overwriting)?
- Will deletions hit Trash (recoverable)?
- Are we logging via FileTracker?
- If bulk data change: prepare an inspection report first (counts, repr values, validation) per TECH KB.

## 5) Run/Stop Discipline (local tools)
- When launching servers or long-running tools, ensure clean shutdown to avoid orphans and port conflicts.
- Close pipes/terminate subprocesses in tests.

## 6) Token/Cost Gates
- Prefer focused reads. Warn if daily operations risk exceeding ~10k tokens; pause large sweeps or batch them. [[memory:9146206]]

## 7) First PR Path (for human or agent)
- Pick a doc-only change or a dashboard-only small fix.
- Add/update docs under `Documents/`.
- If touching code, run unit tests for the touched area.
- Add links back to authoritative docs (safety, API contract).

## 8) Quick Links
- Architecture: `Documents/ARCHITECTURE_OVERVIEW.md`
- Safety: `Documents/FILE_SAFETY_SYSTEM.md`
- Dashboard API: `Documents/DASHBOARD_GUIDE.md`
- Technical KB: `Documents/TECHNICAL_KNOWLEDGE_BASE.md`
