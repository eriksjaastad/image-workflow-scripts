# Spec: scan_dir_state

Goal: Report a directory’s current processing state (EMPTY / PARTIAL / FULL) with counts and recency, without mutating anything. Used for orchestration and safe finish checks.

---

## CLI
```
python scripts/tools/scan_dir_state.py \
  --path /abs/path/to/content \
  [--recent-mins 10] [--json /abs/out/state.json]
```

- Default `--recent-mins 10` marks files modified within the last N minutes as recent.
- Exits 0 always; consumers inspect JSON.

## Logic
- Recursively scan `--path`, skipping directories.
- Count:
  - `totalFiles`, `totalBytes`
  - `byExtension` map (lowercased, no dot) → { files, bytes }
- Recency:
  - `latestMtimeUtc` (ISO-8601 UTC)
  - `recentFiles` = files with mtime within `recent-mins`
- Hidden files: count separately as `hiddenFiles`.
- State rules:
  - `EMPTY`  → `totalFiles == 0`
  - `FULL`   → `totalFiles > 0` AND `recentFiles == 0`
  - `PARTIAL`→ otherwise (recent activity or mixed state)

Notes:
- We do not try to judge “completeness”; `FULL` only means non-empty and not actively changing.
- Thresholds are configurable via `--recent-mins`.

## JSON Output (example)
```json
{
  "path": "/abs/mojo1",
  "scannedAt": "2025-10-06T16:30:00Z",
  "totalFiles": 2450,
  "totalBytes": 1234567890,
  "byExtension": {
    "png": {"files": 2350, "bytes": 1200000000},
    "yaml": {"files": 100, "bytes": 3456789}
  },
  "hiddenFiles": 3,
  "latestMtimeUtc": "2025-10-06T16:22:11Z",
  "recentWindowMins": 10,
  "recentFiles": 0,
  "state": "FULL"
}
```

## Failure handling
- If path missing/unreadable, return a JSON with `error` and `state:"EMPTY"`, `totalFiles:0`.

## Integration
- Project manifest can record last known state for `content/` and gate finishing on `state == FULL`.
- Pre-zip stager may require `state == FULL` to proceed (configurable).
