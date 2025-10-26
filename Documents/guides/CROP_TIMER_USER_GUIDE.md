---
title: Crop Timer User Guide
status: Draft
audience: Operators, Developers
version: 2025-10-26
---

**Last Updated:** 2025-10-26

## Purpose
Operate the crop timer safely with dry-run validation and human-like scheduling.

## Start Here
1) Record crops in the desktop tool using queue/record-only mode (once merged).
2) Build or load a schedule.
3) Run the timer with dry-run first, then execute.

## Commands (tentative)
- Build schedule:
  - `python scripts/tools/build_crop_schedule.py --days 3 --start 2025-10-27T09:00:00`
- Dry-run processor (no writes):
  - `python scripts/process_crop_queue.py --preview --no-breaks --limit 2`
- Execute with timing:
  - `python scripts/process_crop_queue.py --speed 1.0`

## Safety Checklist
- Queue and timing files live under `data/ai_data/crop_queue/`
- Dry-run report created and reviewed
- FileTracker logs enabled
- Pixel writes only via the desktop crop tool headless path

## Troubleshooting
- Empty queue: confirm record-only mode ran and wrote DB/CSV.
- Dimension mismatch: the processor will rescale from normalized and warn.


