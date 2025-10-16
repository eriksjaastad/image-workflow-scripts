# Project Manifest Guide

This guide explains what to fill in `data/projects/<projectId>.project.json` at the start, during, and at the end of a project. It also clarifies how `metrics.stager` is populated and how to capture images/hour.

## At Project Start (2 minutes)
- projectId: short id (e.g., `mojo2`).
- title: human-readable (e.g., `Mojo 2`).
- status: `active`.
- createdAt, startedAt: ISO UTC (Z-suffixed) or leave existing if continuing.
- paths.root: relative path to the project content (e.g., `../../mojo2`).
- counts.initialImages: set to initial PNG image count you intend to process.
- steps[]: keep names; optionally set first step's startedAt when you begin.

## During the Project
- For each step you actually run:
  - step.startedAt at the beginning of that step.
  - step.finishedAt and step.imagesProcessed when done.
- Leave metrics.stager untouched (it is populated by `prezip_stager`).

## At Project End (2 minutes)
- finishedAt: ISO UTC completion time.
- counts.finalImages: final count of PNG images that constitute the deliverable.
- metrics.imagesPerHourEndToEnd (optional but recommended):
  - hours = (finishedAt − startedAt) in hours
  - images/hour = counts.initialImages ÷ hours
  - Note: The dashboard also computes throughput independently from FileTracker logs; this field is explicit documentation in the manifest.

## metrics.stager (Filled by prezip_stager)
You do not fill these manually. Running the stager (dry-run/commit) writes:
- stager.zip: output zip path (on commit)
- stager.eligibleCount: total included files in staging (all allowed types)
- stager.byExtIncluded: counts per extension included (e.g., png/yaml/caption/txt)
- stager.excludedCounts: counts excluded by reason (hidden, banned_name, etc.)
- stager.incomingByExt: per-extension counts scanned from content

Run example:
```
python scripts/tools/prezip_stager.py \
  --project-id mojo2 \
  --content-dir /abs/path/to/mojo2 \
  --output-zip /abs/path/to/exports/mojo2_final.zip \
  --commit
```

## Tips to Ensure Dashboard Metrics Work
- Ensure `startedAt` (and later `finishedAt`) are set.
- Use tools that log via FileTracker (already integrated) so the dashboard can compute rates.
- If a project spans multiple working days, you can still finish it later; throughput will use the full duration.

## Minimal Checklist
- Start: set ids/times, paths.root, counts.initialImages.
- During: set step times + imagesProcessed where known.
- End: set finishedAt, counts.finalImages, (optionally) metrics.imagesPerHourEndToEnd.


