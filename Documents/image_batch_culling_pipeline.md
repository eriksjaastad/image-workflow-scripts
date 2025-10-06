# Fast-Volume Image Culling & Cropping Pipeline (Reference)

This document sketches a safe, reversible pipeline for handling very large image sets (including AI-generated) with minimal manual effort.

---

## 1. Lightning-Fast Culling (No Edits Yet)

- **Photo Mechanic**, **FastRawViewer**, or **Aftershoot** — pro apps optimized for lightning-speed tagging/rating/keep-toss before editing.
- These tools handle tens of thousands of images without slowing down.

## 2. Deduplicate & “Near-Duplicate” Thinning

### Tools
- **imagededup** Python library (PHash/aHash/dHash) — groups duplicates even with different timestamps; exports JSON/CSV.
- **Czkawka** (Rust desktop app) — finds exact & similar images very quickly; GUI with export.
- **CLIP + FAISS** embeddings — stronger “semantic” grouping across crops/edits/burst shots.
- **FiftyOne** — turnkey deduplication and uniqueness flow for embedding-based comparisons.

### Workflow
1. Run pHash/CLIP grouping ignoring filenames & time drift.
2. Export groups to CSV or JSON.
3. Move all but one “keeper” to a `_review_dupes` folder for manual check.

This collapses duplicate sets confidently without destruction.

## 3. Auto-Propose Crops (Never Destructive)

- **smartcrop.js** (saliency-based) proposes fixed-aspect crops automatically.
- In Adobe Lightroom Classic/CC: copy a crop setting, paste to a batch, then spot-fix outliers.

All proposals can be sidecar files (JSON/XMP) — you review before applying.

## 4. Batch Apply, Keep Sidecars

- Write changes to **XMP** sidecars or virtual copies.
- Use **ExifTool** for safe metadata moves/renames and to cluster by capture time (±2–5s to group bursts).

## 5. Minimal “Starter” Automations

- **Tier 1 (10 min):** run **Czkawka** on your root folder → export JSON/CSV → batch-move duplicates except best into `_review_dupes/`.
- **Tier 2:** use **imagededup** PHash → CSV of groups; keep largest-resolution or newest edit per group.
- **Tier 3:** compute **CLIP+FAISS** embeddings for “similar sets” → only review pairs above a cosine-similarity threshold (e.g., ≥0.92).
- **Optional crop pass:** run **smartcrop.js** to generate *suggested* 4:5 or 16:9 crops into sidecar JSON; apply only where confidence is high, then review the rest.

## Why This Works

- **Duplicate files w/ different timestamps:** pHash/CLIP grouping ignores filenames & time drift.
- **Huge volume without destruction:** steps generate reports/sidecars first; deletes/moves last.
- **Speed:** you hand-touch only ~10–20% of images (the exceptions), not every image.

---

### Next Steps for Your Team

- Try one tier at a time and feed the results to Claude for deeper automation ideas.
- Build scripts to automate the repetitive parts (hashing, CSV grouping, sidecar generation).

