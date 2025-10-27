# Error Handling and Alerts
**Status:** Active
**Audience:** Developers, Operators

## Overview
A centralized error bus provides a single source of truth for runtime errors and warnings across tools:
- AI Reviewer (web)
- Web Character Sorter (web)
- Desktop Multi-Crop (desktop GUI)

Errors are persisted to `data/log_archives/errors.jsonl` and can be surfaced in each UI as a sticky top banner or overlay.

## Utilities
Location: `scripts/utils/error_bus.py`

```python
from scripts.utils.error_bus import record_error, load_recent_errors

# Record an error/warning
record_error(
    tool="ai_reviewer",              # e.g., ai_reviewer | web_sorter | desktop_multi_crop
    message="Failed to write .decision sidecar",
    level="error",                   # error | warning
    context={"group_id": gid}
)

# Load recent errors (for UI banners)
recent = load_recent_errors(limit=5, tools=["ai_reviewer"])  # returns list of dicts
```

## File Format (JSONL)
One JSON object per line:
```json
{"ts":"2025-10-27T02:05:51.120Z","tool":"ai_reviewer","level":"error","message":"...","context":{"group_id":"..."}}
```

## Integration Patterns

### Web Apps (Flask)
- Add a lightweight `/errors` endpoint that returns `load_recent_errors(limit=5, tools=[this_tool])`.
- In the template, render a sticky banner at the top; poll the endpoint every few seconds or include errors in normal API responses.

### Desktop (Matplotlib/GUI)
- On a timer or after operations, call `load_recent_errors(limit=5, tools=["desktop_multi_crop"])`.
- Draw a red text overlay on the figure if new errors exist; add a small dismiss control if desired.

## What to Record
- Companion policy violations (single-file image move attempt)
- .decision sidecar write failures
- SQLite logging failures
- Artifact auto-split required conditions
- Any unexpected exceptions that impact user flow or data integrity

## Notes
- The error bus is best-effort: failures to write should not crash callers.
- Keep messages concise; put details in `context`.
- UIs should avoid spamming users: show last few unique errors and debounce repeat alerts.

## Related
- `scripts/utils/companion_file_utils.py` — companion policy enforcement and utilities
- `Documents/guides/COMPANION_FILE_SYSTEM_GUIDE.md` — companion system guide
