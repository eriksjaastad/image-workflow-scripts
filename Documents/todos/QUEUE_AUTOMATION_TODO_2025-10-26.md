## Queue Automation - Actionable TODOs (2025-10-26)

Branch owners:
- Me: `todo/chatgpt-queue-automation-2025-10-26`
- Claude: `todo/claude-queue-automation-2025-10-26`

### Me (ChatGPT)
- [completed] Create `scripts/tools/enqueue_test_batch.py` to queue N center-crops from a folder
- [completed] Add `scripts/tools/smoke_test_processor.py` to run 1-batch end-to-end
- [completed] Finalize analyzer API: `analyze_human_patterns.py` returns data and writes once
- [in_progress] Add configurable safe-zone allowlist (read from `configs/` and used in validation)
- [pending] Add retry with backoff for per-crop failures and partial progress resume
- [pending] Queue manager maintenance CLI: `clear_completed` + timing/log rotation helpers
- [pending] Pre-commit installer script for root-file policy hook in `scripts/tools/`
- [pending] Makefile shortcuts: `make timing`, `make queue-test`, `make process-fast`
- [pending] CLI to delete/restore a batch to `__delete_staging` using companion utils

### Claude
- [pending] Processor: enforce decisions DB linkage in preflight (fail with clear error when missing)
- [pending] Docs: Queue quickstart + analyzer usage (place in `Documents/guides/`)
- [pending] Docs: Commit communication standard snippet in `Documents/README.md`
- [pending] Dashboard: queue stats panel (pending/processing/completed/failed)
- [pending] Dashboard: processing time trends and batches-per-session charts
- [pending] Tool: audit of queue vs filesystem and DB (orphan/consistency report)

Notes:
- CI is now quiet on pushes, blocking on PRs (concurrency + path filters).
- Test helpers are available to enqueue and process quickly.


