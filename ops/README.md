# Operations Directory

This directory contains operational files for system monitoring and health checks.

## Heartbeat System

The `heartbeat.json` file is a **deadman switch** that proves the image workflow system is operational.

### How It Works

1. **heartbeat.json** - Contains a timestamp showing when the system last reported as healthy
2. **GitHub Actions Monitor** - `.github/workflows/deadman.yml` checks this file every 30 minutes
3. **Alert on Stale** - If the heartbeat is >60 minutes old, an alert is triggered

### Updating the Heartbeat

Update the heartbeat file using:

```bash
python scripts/tools/update_heartbeat.py
```

### Automated Updates

Add to your crontab to update every 30 minutes:

```bash
*/30 * * * * cd /path/to/image-workflow-scripts && python scripts/tools/update_heartbeat.py
```

Or add to your pre-commit hook (`.git/hooks/pre-commit`):

```bash
#!/bin/bash
python scripts/tools/update_heartbeat.py --notes "Commit: $(git log -1 --oneline)"
```

### Heartbeat File Format

```json
{
  "last_ok": "2025-11-03T02:17:29Z",
  "system": "image-workflow-scripts",
  "version": "1.0",
  "notes": "Optional status message"
}
```

### Troubleshooting

If the deadman switch alerts:

1. Check if the system is actually running
2. Update the heartbeat manually: `python scripts/tools/update_heartbeat.py`
3. Verify cron job is configured (if using automated updates)
4. Check GitHub Actions logs for more details

### Monitoring Thresholds

- **Check interval**: Every 30 minutes (configurable in `.github/workflows/deadman.yml`)
- **Stale threshold**: 60 minutes (configurable via `HEARTBEAT_MAX_MINUTES` env var)
