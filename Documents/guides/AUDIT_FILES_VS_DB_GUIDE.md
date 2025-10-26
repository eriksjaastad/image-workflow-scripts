---
title: Audit Files vs Decisions DB
status: Current
audience: Operators, Developers
version: 2025-10-26
---

**Last Updated:** 2025-10-26

## Purpose
Quickly verify that files on disk match expectations from the decisions DB for a given project.

## Command
```bash
python scripts/tools/audit_files_vs_db.py --project mojo3 --write-report
```

## What it does
- Reads the decisions DB at `data/training/ai_training_decisions/{project}.db`.
- Builds the expected "kept" set from decisions (approve/crop selected filename).
- Scans common workflow directories on disk (`crop`, `crop_auto`, `selected`, `__*`, `mojo*`).
- Compares expected vs found and reports:
  - kept_found, kept_missing
  - rejects_found_anywhere (informational)
  - duplicates (same filename in multiple places)
- Writes JSON + markdown reports to `data/daily_summaries/`.

## Safety
- Read-only scan of production files
- Reports are appended to safe directory

## Notes
- Filenames are matched by basename only; if there are multiple copies, they are flagged as duplicates.
- If you keep non-standard directories, pass them via `--dirs` to include in the scan.


