## Queue-Based Cropping System Review (2025-10-26)

Branch: `claude/improve-cropping-utility-011CUVyPBdu7xPiYowp39Lvi`

Reviewed commits at time of review:
- a7d3a9d66 – Safe zone migration + trusted path implementation
- 95e691316 – DB integration + FileTracker + preflight validation
- 6898d5c2e – Built-in validation with interactive confirmation

HEAD at review time: `0e87ef576` (docs/tools additions) – code under review includes latest changes.

---

### Issue #1: Pixel Writes Outside Trusted Path
Status: ✅ VERIFIED

Review of `headless_crop()` (`scripts/utils/ai_crop_utils.py`):
- Centralizes the crop/save/move flow and validates PIL presence, file existence, rect bounds, and image bounds before writing.
- Uses the same operational pattern as the desktop tool: crop → save in place → move with companions.
- Critical fix needed: import path should be package-relative. Change `from companion_file_utils import ...` to `from .companion_file_utils import ...` (or fully-qualified package import) to avoid runtime import errors.
- Policy note: Ensure this trusted path is explicitly allowed per repo safety policy.

Verdict: Functionally correct centralization; fix import path; document policy allowance.

### Issue #2: Queue Files Outside Safe Zone
Status: ✅ VERIFIED

Findings:
- Default queue path moved to safe zone: `data/ai_data/crop_queue/crop_queue.jsonl`.
- Timing artifacts (`timing_log.csv`, `timing_patterns.json`) are colocated in the same safe zone.
- Minor: Update processor docstring mentioning legacy `data/crop_queue/` path.

### Issue #3: Missing FileTracker Logs
Status: ❌ NOT FIXED

Findings:
- `FileTracker.log_operation` signature expects `notes=`; several call sites pass `metadata=` which will raise `TypeError` and break flows.
  - `scripts/process_crop_queue.py`: crop log uses `metadata=`.
  - `scripts/02_ai_desktop_multi_crop.py`: queued move log uses `metadata=`.
  - `scripts/utils/crop_queue.py`: enqueue log uses `metadata=`.

Required fix: Replace `metadata=` with `notes=` in all FileTracker calls.

### Issue #4: DB Not Source of Truth
Status: ⚠️ PARTIAL

Findings:
- Queue mode computes normalized coords and attempts to write them to the decisions DB via v3 API at queue-time.
- DB update is best-effort (exceptions are swallowed); processor does not enforce/repair DB writes. If `.decision`/`group_id`/schema is missing, DB won’t be updated and the queue becomes de facto source.

Recommendations:
- Make DB update authoritative (fail queueing if DB update fails) or perform/update in the processor with retries.
- Preflight should verify DB row existence (by group_id) and valid `final_crop_coords` intent before processing.

### Issue #5: Missing Validation
Status: ⚠️ PARTIAL

Findings:
- Preflight always runs and interactive confirmation is required unless `--yes`.
- Checks include existence, rect shape and monotonicity, dest dir type/creation, overwrite warning, loose safe-zone heuristic.

Gaps to close:
- Enforce allowed destinations (make non-safe destinations a hard error, not just a warning), aligned with repo safety rules.
- Validate normalized coords in queue are within [0,1]; cross-check pixel vs normalized with recorded image dimensions.
- Optionally assert decisions DB presence/linkage per crop.

---

## Per-File Notes

- `scripts/utils/ai_crop_utils.py`
  - Fix import to `from .companion_file_utils import move_file_with_all_companions`.
  - Otherwise aligns with desktop crop path and validations.

- `scripts/02_ai_desktop_multi_crop.py`
  - Queue mode: normalized coords computed; both pixel and normalized stored in queue; DB update attempted; originals moved to `__crop_queued` with companions.
  - Replace `metadata=` with `notes=` in FileTracker call.
  - Consider documenting `__crop_queued`/central `__cropped` as allowed destinations (or switch to designated safe directories).

- `scripts/process_crop_queue.py`
  - Uses `headless_crop()` (good); crop logging present but uses `metadata=` instead of `notes=` (fix).
  - Strengthen preflight safe-zone enforcement and add normalized/pixel consistency checks; update docstring path.

- `scripts/utils/crop_queue.py`
  - Safe-zone default confirmed.
  - Replace `metadata=` with `notes=` in FileTracker call.
  - Race conditions:
    - Batch ID generation scans without exclusive lock; two writers can collide within the same second.
    - `update_batch_status()` performs read-modify-write without exclusive lock across the entire cycle; concurrent updates can lose changes.
  - Mitigations: Use `LOCK_EX` on the queue file for read-modify-write, or adopt per-batch files or SQLite for transactional updates. Consider UUIDs for IDs.

---

## Priority Fixes (Actionable)

1) Critical
- Replace all `metadata=` with `notes=` in `FileTracker.log_operation` calls.
- Fix `headless_crop` import to package-relative.
- Add exclusive locking over read-modify-write in `update_batch_status()`; avoid `_generate_batch_id()` races (exclusive lock or UUIDs).

2) High
- Tighten preflight: hard-error on non-allowed destinations; validate normalized ranges; check normalized↔pixel consistency; optionally verify DB linkage.
- Decide/document that `headless_crop()` is an approved trusted write path.

3) Medium
- Update docstrings referencing legacy `data/crop_queue`.
- Consider processor-side DB write to guarantee DB as source of truth.

---

## Answers to Questions

1) Is `headless_crop()` safe to centralize pixel writes? Yes, with the import fix and explicit policy allowance.
2) Is the DB integration correct? Partially; normalized coords are written best-effort. Make it authoritative.
3) Is validation beyond production safe? Not yet; strengthen safe-zone enforcement and consistency checks.
4) Any queue manager race conditions? Yes, in ID generation and status updates; add locking or use a transactional store.
5) Interactive confirmation UX? Good; consider optional JSON preflight report output for audits.
6) Biggest remaining risk? Broken FileTracker calls (`metadata` vs `notes`) causing runtime errors; and queue file race conditions.


