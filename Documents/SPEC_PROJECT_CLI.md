# Spec: Project Manifest CLI

Goal: Provide simple, explicit commands to start a project, mark step boundaries, and finish a project by updating `data/projects/<id>.project.json` safely.
## Location & Naming
- Manifest lives in repo under: `data/projects/<projectId>.project.json`
- Rationale: keep outside content/output so it never ships in deliverables and survives content cleanups.
- Example found in repo: `data/projects/mojo1.project.json`

## Required Fields
- `schemaVersion`: integer (current: 1)
- `projectId`: string (e.g., "mojo1")
- `status`: `active|finished|archived`
- `createdAt`, `startedAt`, `finishedAt`: ISO-8601 UTC
- `paths`: `root`, optional `selectedDir`, `cropDir`, etc. (relative paths ok)
- `counts`: `initialImages`, `finalImages`
- `steps[]`: `{ name, startedAt, finishedAt, imagesProcessed }`
- `metrics`: `imagesPerHourEndToEnd`, `stepRates`
- `removeFileOnFinish`: bool (if true, CLI may archive/delete manifest on finish after summarization)


---

## Commands

### project_start
```
python scripts/tools/project_cli.py start \
  --project-id mojo1 \
  --content-dir /abs/path/to/mojo1 \
  [--started-at 2025-10-06T16:15:00Z]
```
- Creates manifest if missing, or updates `startedAt` if present and unset.
- Initializes counts (optionally estimates `initialImages` by scanning `content-dir`).
- Writes allowlist snapshot file if not present (see ALLOWLIST_SCHEMA).

### project_step_start
```
python scripts/tools/project_cli.py step-start \
  --project-id mojo1 \
  --name select_versions \
  [--ts 2025-10-06T16:25:00Z]
```
- Finds/creates step by name, sets `startedAt` if unset.

### project_step_finish
```
python scripts/tools/project_cli.py step-finish \
  --project-id mojo1 \
  --name select_versions \
  [--images-processed 1200] \
  [--ts 2025-10-06T17:10:00Z]
```
- Sets `finishedAt` and `imagesProcessed` on the step (idempotent).

### project_finish
```
python scripts/tools/project_cli.py finish \
  --project-id mojo1 \
  [--finished-at 2025-10-06T20:00:00Z]
```
- Sets `finishedAt`.
- Optionally computes metrics: end-to-end images/hour from `counts.initialImages` and total duration; step rates from step durations and imagesProcessed if provided.

---

## Safety
- CLI never writes into content directories.
- Always backs up the prior manifest (e.g., `*.bak`) before writing.
- Validates timestamps are ISO-8601 UTC.
- Refuses to overwrite `finishedAt` unless `--force` is provided.

---

## Manifest fields touched
- `startedAt`, `finishedAt`
- `counts.initialImages`, `counts.finalImages` (optional)
- `steps[]` entries `{ name, startedAt, finishedAt, imagesProcessed }`
- `metrics` (compute on finish if requested)

---

## Integration
- `project_start` may call allowlist inventory.
- `project_finish` can optionally require `scan_dir_state --state FULL` and a successful pre-zip stager run before setting `finishedAt` (configurable flags).
