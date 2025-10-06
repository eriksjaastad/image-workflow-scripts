# Fast-Volume Near-Duplicate Thinning & Workflow Automation Plan

This plan captures practical steps to reduce redundant images safely, speed up review, and keep everything reversible. It reflects ideas from `image_batch_culling_pipeline.md` adapted to our current tools.

---

## Guiding Principles
- Non-destructive by default: stage to review folders, use sidecars, never edit image bytes.
- Companion-aware moves: image + all same-stem files travel together.
- Reversible: reports first → staging dir → manual approve.

---

## Quick Wins (Low Risk, High Impact)

### 1) Sidecar crop flag + UI badge
- Add `.cropped` sidecar files next to every PNG under `crop/` (same stem). These sidecars will move with the image and can be used to display a badge.
- Command (one-time/periodic):
```bash
find "/Users/eriksjaastad/projects/Eros Mate/crop" -type f -name "*.png" -print0 | xargs -0 -I{} bash -c ': > "${1%.*}.cropped"' -- {}
```
- Character Sorter: detect `.cropped` companions and show a small “Cropped” badge on thumbnails.
- Similar-image grouper: preserve sidecars on moves (already supported via companion utilities).

### 2) External dedup pass (Czkawka GUI)
- Run Czkawka across `mojo1/` (exact + similar). Export JSON/CSV groups.
- Tiny script then moves group non-keepers to `_review_dupes/` using a rule (e.g., keep largest resolution or newest edit).
- Benefits: big surface-area reduction before our web tools, fully reversible.

### 3) pHash near-duplicate finder (in-repo)
- Build a simple `scripts/utils/phash_near_dupes.py` using `imagehash`.
- Outputs CSV with groups above a similarity threshold (tunable).
- Rule-based staging of non-keepers to `_review_dupes/` (no Trash).
- Use in addition to, or instead of, Czkawka where CLI/automation is preferred.

### 4) Time-window duplicate catch (timestamp-based)
- Script clusters images by filename timestamp within ±N seconds and identical stage names.
- Surfaces likely burst/near-duplicate sets for quick thinning.
- Non-destructive: write a report and optional `_review_dupes/` staging.

---

## Medium Next Steps

### A) CLIP + FAISS embeddings index
- Build a reusable FAISS index for `mojo1/`, `selected/`, `crop/`.
- For each image, fetch k-NN and group pairs above cosine-similarity threshold (e.g., ≥0.92).
- Export CSV groups; optionally stage low-scoring variants to `_review_dupes/`.

### B) Similar-Image Grouper: multi-source input
- Extend grouper to accept multiple source roots (`selected/` + `crop/`) in one run.
- Continue moving image + companions (including `.cropped` sidecar) into cluster folders.

### C) Character Sorter: crop badge
- Read `.cropped` sidecar presence and render a small badge on each thumbnail.
- No behavior change; purely visual guidance to reduce re-checking.

---

## Safe Cross-Timestamp Duplicate-Set Pruning (Design)

Goal: When two sets share the same stage lineup (e.g., 1 → 1.5 → 2 → 3) but different timestamps and are visually identical, keep one whole set and stage the other to `delete_review/`.

Proposed pipeline:
1) Candidate generation
   - Use FAISS/CLIP or the similar-image grouper to find high-similarity candidates across timestamps.
   - Require exact stage-name parity across compared sets.
2) Verification
   - Stage-by-stage cosine similarity on CLIP embeddings; all stages must exceed threshold.
   - YAML cross-check (optional): same/similar metadata content.
3) Report + staging
   - Produce a human-readable report (CSV/Markdown) listing matched sets.
   - Move the redundant full set (image + companions) into `delete_review/`.
4) Approval step
   - Manual review of `delete_review/` before any permanent deletion.

Modes & safety:
- Dry-run (report only) → Stage mode (move to `delete_review/`) → Manual approval.
- Conservative thresholds to minimize false positives.

---

## How This Maps To Our Current Stack
- We already have:
  - Ergonomic web selector + character sorter (grouped review flows).
  - Similar-image grouper using CLIP + HDBSCAN (moves image + companions).
  - Strong companion handling: sidecars travel with images.
- Gaps addressed by this plan:
  - No pHash/FAISS-based global dedup pipeline yet.
  - No multi-source grouper ingestion.
  - No cross-timestamp set-level pruning.

---

## Suggested Order of Operations
1) Sidecar crop flags + Character Sorter badge (UI-only, no data risk).
2) External Czkawka pass or pHash script → `_review_dupes/` staging.
3) FAISS index build + CSV of semantic near-dupes.
4) Cross-timestamp duplicate-set pruning (dry-run → stage → manual approve).

---

## Risks & Safeguards
- False positives in near-dupe detection → use high thresholds, stage to review, never hard-delete automatically.
- Companion integrity → always move image + all same-stem files together.
- Performance → batch and cache embeddings; incremental updates for FAISS.

---

## Handy Commands

Create `.cropped` sidecars for all PNGs in `crop/`:
```bash
find "/Users/eriksjaastad/projects/Eros Mate/crop" -type f -name "*.png" -print0 | xargs -0 -I{} bash -c ': > "${1%.*}.cropped"' -- {}
```

Two-directory duplicate filename check (PNGs only):
```bash
python "/Users/eriksjaastad/projects/Eros Mate/scripts/utils/duplicate_checker.py" \
  "/Users/eriksjaastad/projects/Eros Mate/mojo1" \
  "/Users/eriksjaastad/projects/Eros Mate/selected" --extensions png
```

---

## Review Notes
- Keep everything reversible and staged until we’re confident.
- Start small (one folder), validate results, then scale up.
- Use the reports to prioritize where the biggest wins are.
