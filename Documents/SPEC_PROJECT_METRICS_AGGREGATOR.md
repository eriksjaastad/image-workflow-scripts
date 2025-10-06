# Spec: Project Metrics Aggregator

Goal: Compute project-level metrics (end-to-end images/hour, step-level rates, ahead/behind vs baseline) by reading manifests, dir-state/step events, and optional baselines.

---

## Inputs
- Manifest: `data/projects/<id>.project.json`
- Optional events: `data/timer_data/projects/projects_events.jsonl` (daily progress entries)
- Optional baseline: computed from history (`data/timer_data/history/daily_throughput.csv`) or a per-project baseline snapshot.

## CLI
```
python scripts/dashboard/project_metrics_aggregator.py \
  --project-id mojo1 \
  [--baseline-json /abs/path/baseline.json] \
  [--out-json /abs/path/metrics.json]
```

## Outputs (JSON)
```json
{
  "projectId": "mojo1",
  "durationHours": 12.5,
  "imagesTotal": 2500,
  "imagesPerHour": 200.0,
  "stepRates": {
    "select_versions": {"hours": 3.0, "images": 1200, "iph": 400.0},
    "character_sort": {"hours": 4.5, "images": 900, "iph": 200.0}
  },
  "aheadBehind": {
    "baselineIph": 180.0,
    "p25": 150.0,
    "p75": 220.0,
    "status": "ahead"  
  }
}
```

## Computation
- End-to-end: if `startedAt` and `finishedAt` present, duration = diff; else estimate using events (sum hours_today).
- Images total: use `counts.initialImages` or sum over step `imagesProcessed` where applicable.
- Step rates: for each step with both times, compute duration and iph using `imagesProcessed` if set; else omit.
- Ahead/Behind: compare `imagesPerHour` to baseline and set status relative to p25/p75.

## Safety & Integrity
- Read-only; never mutates project files.
- Validates timestamps and missing fields; degrade gracefully if incomplete.

## Dashboard integration
- The JSON can be read by a dashboard to render:
  - Top KPI card (Images/hour)
  - Step breakdown table
  - Ahead/Behind chip using p25/p75 bands
