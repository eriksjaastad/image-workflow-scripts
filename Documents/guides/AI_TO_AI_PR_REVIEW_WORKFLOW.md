## AI↔AI PR Review Workflow (Draft)

Purpose: Eliminate confusion in handoffs between assistants by using one repeatable flow, one place for review docs, and one checklist both sides follow every time.

### 1) Branching & Naming
- Feature branches use: `feature/<area>/<short-desc>-<YYYY-MM-DD>` or task-specific: `todo/<owner>-<area>-<YYYY-MM-DD>`
- Examples:
  - `todo/chatgpt-queue-automation-2025-10-26`
  - `feature/char-sorter/orphan-decisions-2025-10-26`

### 2) PR Creation (Author)
Always create the PR first, then share the PR URL.
- Base: `main`
- Head: your feature/todo branch
- Title: action-oriented, e.g., `feat(character-sorter): preview + stage orphan .decision files`
- Body: use the repository PR template (auto-included) and fill all fields.

Required PR fields to include (not optional):
- Branch, PR URL, commit SHAs (short + full), files touched
- How to verify (exact commands/URLs)
- Risks/safety notes

### 3) Review Doc Location (Reviewer)
- Review docs go into the REVIEWEE’S BRANCH (not main):
  - `Documents/reviews/REVIEW_<short-branch-or-topic>_<YYYY-MM-DD>.md`
- The review doc must include:
  - Summary: ✅ verified / ⚠️ partial / ❌ not fixed
  - Precise file/line references (with code snippets if needed)
  - Actionable diffs (sketch or exact)
  - Priority order
  - Final verdict (merge, fix-then-merge, or block)

### 4) Handoff Bundle (What to Paste in Chat)
- For authors handing to reviewers:
  - PR: <PR URL>
  - Branch: <branch>
  - Head commit: <short> (<full>)
  - Files: <paths>
  - Verify: exact commands/URLs
- For reviewers handing feedback back:
  - Review doc path in the REVIEWEE’S BRANCH
  - Summary bullets of critical issues
  - Must‑fix list with file/line pointers
  - Optional: follow-up tasks and owners

### 5) Revision Loop
1) Author pushes fixes to the same branch (PR auto-updates)
2) Reviewer updates the same review doc with “round 2” notes or marks items resolved
3) Repeat until all must-fix items are resolved
4) Reviewer posts “Merge ✅” at top of the review doc

### 6) Merge Protocol
- Author merges only after Reviewer marks “Merge ✅” in the review doc
- If safety-critical code paths changed (file ops, DB writes), author must include a brief test transcript in PR comments (commands + outcomes)

### 7) Anti‑Pitfalls
- Never share a PR creation link; share the actual PR URL
- Never place review docs on main; always put review docs in the reviewee’s branch
- Don’t drop artifacts in repo root; use `Documents/`, `scripts/`, `data/**`, or `sandbox/`
- Use module mode for scripts to avoid import path issues: `python -m scripts.path.to.module`

### 8) Checklists

Author Checklist (before handing off):
- [ ] Branch pushed, PR opened against main
- [ ] PR body filled (branch, SHAs, files, verify steps, risks)
- [ ] Safety notes included for file operations/DB changes
- [ ] Commands verified locally (copy-paste ready)

Reviewer Checklist:
- [ ] Checked PR files changed
- [ ] Ran read‑only verification or smoke tests (if applicable)
- [ ] Wrote review doc in REVIEWEE’S BRANCH under `Documents/reviews/`
- [ ] Marked issues as ✅/⚠️/❌ with exact file/line
- [ ] Included actionable diffs/sketches
- [ ] Marked “Merge ✅” or “Fix and re‑review”

### 9) Mini Templates

Review Doc Title:
```
Documents/reviews/REVIEW_<topic>_<YYYY-MM-DD>.md
```

Review Doc Skeleton:
```
## Review – <topic> (<YYYY-MM-DD>)

Branch: <branch>
PR: <url>
Head: <short> (<full>)

### Summary
- <one-liners>

### Findings
1) <area> – ✅/⚠️/❌
   - File:line – note
   - Fix:

### Actionable Diffs (sketch)
```

### 10) Commands (Examples)
```
python -m scripts.tools.enqueue_test_batch --dir mojo3 --limit 10 --dest __cropped
python -m scripts.process_crop_queue --yes --fast
```

---

Status: Draft (delete once the process is stabilized and migrated into Cursor rules + PR template).


