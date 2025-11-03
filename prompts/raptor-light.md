# RAPTOR‑LITE

Cheap, single‑pass reliability review designed to cut token usage by **70–80%** while keeping the important safety fixes.

---

## Purpose
Eliminate silent failures and surface operator‑visible logging **without** multi‑model, multi‑phase reviews.

## When to Use
- You’re reviewing a single file (e.g., `scripts/00_start_project.py`).
- You want one high‑signal patch, then local tests decide the rest.
- You’re in “cheap mode” for the month.

---

## One‑Pass Prompt (paste this as your ask)
```
RAPTOR-LITE (one pass, cheap)
File: <path/to/file.py>
Goal: eliminate silent failures and add operator-visible logging.

Return ONLY:
1) Unified diff (≤120 changed lines) touching only this file.
   - Replace bare excepts with specific exceptions (ImportError, OSError, PermissionError) as appropriate
   - Read-after-write verify critical outputs; early-return on mismatch/JSON errors
   - Fail fast on allowlist/manifest writes; log inventory/paths at INFO; errors at ERROR
   - For scanning loops: warn if dir missing; count & warn on unreadables; log details at DEBUG
2) 5-bullet test plan (no code).

Strict rules:
- No prose beyond those two sections.
- No second model, no MAX mode, no “verification” pass.
- If you need assumptions, state them in ≤3 bullets at the end.
```
*(Make “≤120 lines” whatever ceiling you prefer.)*

---

## Workflow (3 steps)
1. **Run Haiku 4.5** with the one‑pass prompt above.
2. **Apply the diff** locally → run `ruff --fix`, `black`, `pytest -q`.
3. **Escalate once** to Sonnet 4.5 *only if* tests fail for non‑trivial reasons. Ask for **diff‑only** and stop.

---

## Burn Guardrails
- **One model per session.** Do *not* “verify” the same file with a second model.
- **Diff‑only outputs.** Ask for a unified diff and a **test plan**, not full test code.
- **No MAX mode** for reviews.
- **Batch changes**: fix multiple small issues in one patch rather than chatty micro‑asks.
- **Cap output size**: enforce a maximum changed‑line count.
- **No repo‑wide context**: provide file path + ~80 lines around changed blocks *only if requested*.

---

## Diff Requirements (what “good” looks like)
- Replace `except Exception: pass` with specific exceptions; log at `WARNING` or `ERROR` with context.
- Add **read‑after‑write** verification for critical artifacts (e.g., manifests, config JSON).
- Fail fast on writes that others depend on (e.g., allowlists). Don’t defer failures.
- For directory scans:
  - Warn if directory missing.
  - Count unreadable files; warn with total; log details at DEBUG.
- Avoid noisy prints; prefer `logging` with module logger:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```

---

## Test Plan Template (5 bullets, no code)
- **Import failure path** triggers a `WARNING` when optional deps (e.g., `FileTracker`) are unavailable.
- **Audit logging failure** logs `ERROR` yet the primary operation returns success.
- **Critical write failure** (e.g., allowlist/manifest) returns structured error immediately.
- **Read‑after‑write** catches corruption/mismatch and returns error.
- **Scan unreadables**: when encountering unreadable files, emits `WARNING` summarizing skipped count.

---

## Cursor Snippet (optional; paste into your snippets)
```
Name: RAPTOR-LITE one-pass
Trigger: raptor
Body:
RAPTOR-LITE (one pass, cheap)
File: ${1:path/to/file.py}
Goal: eliminate silent failures and add operator-visible logging.

Return ONLY:
1) Unified diff (≤120 changed lines) touching only this file.
   - Replace bare excepts with specific exceptions (ImportError, OSError, PermissionError) as appropriate
   - Read-after-write verify critical outputs; early-return on mismatch/JSON errors
   - Fail fast on allowlist/manifest writes; log inventory/paths at INFO; errors at ERROR
   - For scanning loops: warn if dir missing; count & warn on unreadables; log details at DEBUG
2) 5-bullet test plan (no code).

Strict rules:
- No prose beyond those two sections.
- No second model, no MAX mode, no “verification” pass.
- If you need assumptions, state them in ≤3 bullets at the end.
```
---

## Minimal Prompt Variants (for tiny edits)
- **Regex edit**: “Return a single sed/Perl one‑liner to replace `print(...)` with `logger.info(...)` in <file>.”
- **Batch micro‑fixes**: “One diff that (1) adds `--min-faces`, (2) guards `cv2.imread` None, (3) logs summary counts.”
- **Sanity check**: “Is there a library way to do X in ≤5 lines? One-liner example only.”

---

## Metrics to Track (manual)
- Avg changed lines per patch (target: 40–120).
- Warnings/Errors emitted after a run (expect ≥1 when something goes wrong).
- % of runs requiring Sonnet escalation (target: <20%).

---

*That’s it. One pass, one patch, local tests decide.*
