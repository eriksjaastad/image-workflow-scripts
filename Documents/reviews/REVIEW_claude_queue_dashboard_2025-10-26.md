## Code Review – Queue Dashboard Branch (2025-10-26)

Branch: `claude/queue-dashboard-011CUVyPBdu7xPiYowp39Lvi`

Reviewed commits (latest first):
- 1b145c79c – add queue stats panel and charts to dashboard UI
- 34040d1e8 – add queue stats to dashboard API
- cd2be08d2 – add queue consistency audit tool
- 6d0e558ae – enforce decisions DB linkage in processor preflight validation
- e08d151d6 – docs: quickstart & analyzer usage
- dfcfcd5b0 – docs: commit communication standard

Files reviewed:
- `scripts/dashboard/analytics.py`
- `scripts/dashboard/queue_reader.py`
- `scripts/dashboard/dashboard_template.html` (spot-checked)
- `scripts/process_crop_queue.py`
- `scripts/tools/audit_crop_queue.py`
- `Documents/guides/QUEUE_QUICKSTART_AND_ANALYZER_GUIDE.md` (spot-checked)

---

### High‑Priority Issues

1) DB table name mismatch (breaks validation)
- Where: `scripts/process_crop_queue.py` and `scripts/tools/audit_crop_queue.py`
- Problem: Both query `SELECT id FROM decisions ...`, but the v3 DB uses table `ai_decisions` (see `utils/ai_training_decisions_v3.py`).
- Impact: Preflight DB linkage check will always fail; the audit tool will report false negatives.
- Fix:
  - Change to: `SELECT id FROM ai_decisions WHERE group_id = ?`
  - Same change in both files.

2) Safe‑zone validation regression vs main
- Where: `scripts/process_crop_queue.py` (preflight `is_safe`)
- Problem: Uses substring match (`any(s in str(dest_dir) ...)`), which we already fixed on main to use `Path.parts` with consecutive part checks (e.g., `data/ai_data`).
- Impact: False positives/negatives for safe‑zone validation.
- Fix: Port the improved multi‑part safe‑zone check (the `b6c2a99` approach) into this branch.

3) Dashboard analytics writes sample output to repo root
- Where: `scripts/dashboard/analytics.py` (main() writes `dashboard_analytics_sample.json` to project root)
- Problem: Violates root‑file hygiene policy; root should stay clean.
- Fix: Write to `data/dashboard_cache/` or `sandbox/` instead, e.g., `data/dashboard_cache/dashboard_analytics_sample.json`.

---

### Medium Priority

4) Selector/crop classification via dest_dir substring
- Where: `analytics.py` `_build_tools_breakdown_for_project`
- Current: Checks tokens in `dest_dir` to split selected vs cropped counts.
- Risk: Brittle to naming changes.
- Improvement: Use `utils.standard_paths` to identify standard destinations or a centralized mapping function.

5) Verbose debug prints in analytics
- Where: `analytics.py` billed vs actual functions
- Suggestion: Gate under `DEBUG` flag or reduce noise; currently quite chatty.

---

### Low Priority / Nice‑to‑Have

6) Cross‑platform date formatting in analyzer
- Where: Daily timeseries format uses `strftime('%-m/%-d/%Y')`, which is macOS/Linux‑specific.
- Suggestion: Handle Windows (`%#m/%#d/%Y`) or normalize timesheet dates earlier.

7) Queue reader robustness
- `queue_reader.py` is good; consider guarding against malformed CSV rows (already catches exceptions) and adding simple unit tests for parsers.

---

### Summary of Status per Area

- Queue stats in dashboard: ✅ Verified
- Queue data reader: ✅ Verified
- Processor preflight – DB linkage: ❌ Not fixed (wrong table name)
- Safe‑zone validation: ⚠️ Partial (port the main branch fix)
- Audit tool: ❌ Not fixed (wrong table name)
- Docs: ✅ Verified (helpful quickstart)

---

### Actionable Diffs (sketch)

1) Use correct table in both files:
- In `scripts/process_crop_queue.py` and `scripts/tools/audit_crop_queue.py`:
  ```sql
  SELECT id FROM ai_decisions WHERE group_id = ?
  ```

2) Port safe‑zone parts check (example skeleton):
  ```python
  dest_parts = dest_directory.parts
  def has_pair(parts):
      return any(i < len(parts)-1 and parts[i]=="data" and parts[i+1]=="ai_data" for i in range(len(parts)))
  is_safe = (
      '__cropped' in dest_parts or '__crop_queued' in dest_parts or '__final' in dest_parts or '__temp' in dest_parts or has_pair(dest_parts)
  )
  ```

3) Move analytics sample output under cache dir:
  ```python
  output_file = project_root / "data" / "dashboard_cache" / "dashboard_analytics_sample.json"
  output_file.parent.mkdir(parents=True, exist_ok=True)
  ```

---

### Final Verdict

- This is very close. Fix the DB table name, port the safe‑zone check from main, and relocate the analytics sample output. After that, I’d consider it merge‑ready.


