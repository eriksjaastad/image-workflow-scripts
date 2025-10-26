# Offline Training Plan — Best-Photo Ranking, Fixed-Aspect Cropping, and Anomaly Checks (Mac, compute‑light)
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


**Device target:** MacBook Pro (16‑inch, Nov 2024), **Apple M4 Pro**, **24 GB RAM**, macOS **15.6.1**  
**Workflow order:** 1) **Rank best photo** in each 2–4 image set → 2) **Propose a crop** that removes anomalies **without changing aspect ratio** → 3) **Flag residual anomalies**. You review/approve at every step until the system hits ~100% agreement, then (optionally) batch with review.

---

## Executive Summary

**Problem.** You curate large batches of AI‑generated person images. For each small set (2–4 similar images), you pick the best, crop to remove anomalies (hands/feet/limb issues) while **keeping the original aspect ratio**, and reject bad images. You want the system to **watch** first, then **recommend**, and eventually **bulk‑propose** with a human review queue—entirely **offline** on an Apple Silicon Mac.

**Approach (compute‑light, offline‑first):**
- **Representations & duplicates:** Use **OpenCLIP** (e.g., ViT‑B/32) embeddings for semantic similarity/ranking features; add **pHash** for near‑exact duplicates.
- **Anomaly cues (hands/feet/limbs):** **MediaPipe Hands** keypoints for hand geometry sanity checks; add a tiny **YOLO face/foot** detector; **U²‑Net saliency** mask to preserve subject in crops and penalize anomalous regions.
- **Cropping (fixed aspect):** Optimize a **window objective** (saliency in − anomaly overlap − head/joint penalties) under **exact original aspect ratio** + your rules (keep head when possible; retain ≥⅓ body for full‑body shots when feasible).
- **Ranking model:** Learn **your** preferences via **pairwise comparisons** (winner vs. losers) using CLIP features + simple heuristics (sharpness, subject size, anomaly counts). Start with **Bradley‑Terry** or a tiny MLP and update incrementally from your approvals.
- **Active learning:** Prioritize **uncertain** sets (close scores or detector disagreement) for your attention first to converge faster.
- **Apple Silicon acceleration:** Use **PyTorch MPS** (Metal) for any training; most value comes from pretrained models + your ongoing approvals, so compute is modest.

**Data you have vs. need:**
- **Have:** PNGs with companion metadata (YAML prompts and/or caption files); your approve/crop/delete actions (implicit supervision).
- **Need to log:** set membership, chosen winner, final crop box (orig coords), anomaly tags (checkboxes) and deletions.

**Success & promotion to batch:**
- Track: **Top‑1 agreement** on new sets, **crop acceptance** without edits, and **residual anomaly** rate post‑crop.  
- Once sustained ~**100% agreement** and near‑zero residual anomalies for N consecutive sets (e.g., 300), enable **batch propose** with **post‑batch review**.

**Why this fits an M4 Pro:** Components are small (MediaPipe CPU, saliency net, nano detectors, CLIP‑B/32) and rely primarily on **pretrained** models and **incremental** learning from your choices; all run comfortably offline on Apple Silicon.

### Key References (light reading)
- **Apple Silicon acceleration:** PyTorch MPS/Metal backend; HF Accelerate; Lightning on MPS.
- **Embeddings & similarity:** OpenAI CLIP; **OpenCLIP** (LAION); cosine similarity for ranking & near‑dup detection.
- **Saliency:** **U²‑Net** salient object detection.
- **Hands/pose:** **MediaPipe Hands** (21 keypoints per hand).
- **Face/foot detectors:** Small/“nano” **YOLO** variants for boxes.
- **Preference learning:** **Bradley‑Terry** / **Plackett‑Luce** models; pairwise margin losses; active preference querying.
- **Near‑duplicate detection:** **pHash** / perceptual hashing.
- **Active labeling:** **Cleanlab**/**ActiveLab** concepts for prioritizing ambiguous cases.

> If you want the exact URLs later, I can add a bibliography section. All choices are standard, stable, open‑source friendly, and known to run on‑device.

---

## 1) Architecture Overview (Offline‑First)

**Pipelines**
1. **Watch & Log (no ML decisions)** → capture labels from your normal workflow.
2. **Recommend (human‑in‑the‑loop)** → rank + propose crop; you approve/edit.
3. **Batch Propose** (after convergence) → propose for hundreds; queue a fast review pass.

**Core components**
- **Embeddings:** OpenCLIP ViT‑B/32 (or tiny ViT) → 512–768‑D vectors.
- **Hashes:** pHash (Hamming distance) for near‑exact duplicates.
- **Detectors:** MediaPipe Hands (keypoints), tiny YOLO for face/foot boxes.
- **Saliency:** U²‑Net full‑image saliency mask.
- **Crop Optimizer:** Fixed‑aspect window search maximizing (saliency in) and penalizing anomaly/head/joint cuts.
- **Preference Model:** Bradley‑Terry/PL or small MLP fit to your pairwise choices.
- **Active Learning Router:** Surfaces sets with small margins or detector disagreement.

**Devices**
- Inference mostly **CPU**; embedding and training **MPS (GPU)** for speed where helpful.

---

## 2) Data & Labeling that Match Your Current Flow

### Minimal logging (zero extra work)
When you approve/pick/crop/delete via your existing UI, append to `labels/sessions/<date>__session.jsonl`:
```json
{"ts":"2025-10-05T19:12:31Z","set_id":"abc123",
  "images":["img_001.png","img_002.png","img_003.png"],
  "winner":"img_002.png","needs_crop":true,"delete":false,
  "crop_box":{"x":123,"y":210,"w":1024,"h":1536},   // only if you cropped
  "anomaly_tags":["extra_fingers","malformed_phone_grip"]}
```
> `crop_box` uses **original image coordinates** and **keeps the original aspect ratio**.

### Quick anomaly checkboxes
On “send to crop” or “delete”, show small toggles:  
`extra_fingers • warped_wrist • limb_misaligned • long_hand • long_foot • wrong_toes_count • foreshortened_upper_arm • malformed_phone_grip`

### Pairwise labels
When you pick a winner from a 2–4 image set, log **(winner, loser)** pairs for each loser (implicit from your action).

### Recommended data split
Use **group‑wise splits** to avoid leakage: group by prompt/person‑cluster; keep groups intact within train/val/test. Also keep a small **rolling recent** validation slice to monitor real‑time agreement.

---

## 3) Cropping Policy (Encoded from your rules)

- **Aspect ratio:** **Never change**; all crops must match the **original** image aspect.
- **Head preference:** Keep the **head** when possible; avoid cutting eyes/forehead unless anomalies force it.
- **Full‑body shots:** Prefer crops that preserve **≥ 1/3** of body height where feasible.
- **Joints:** Penalize crops that cut joints (shoulders, knees, wrists, ankles) when avoidable.
- **Goal:** **Remove all anomalies** in-frame while **minimizing lost content**.

**Objective (per candidate window):**  
`Score = α·SaliencyInside – β·AnomalyOverlap – γ·HeadCutPenalty – δ·JointCutPenalty`  
Search via **multi‑start hill‑climb** or **multi‑scale sliding windows** + NMS (fast on CPU/GPU).

---

## 4) Ranking (“Best Photo”) Strategy

- **Features:** CLIP embedding (L2‑normed) + heuristics (Laplacian sharpness, subject size from saliency, anomaly counts/flags).  
- **Labels:** Pairwise choices from your sets.  
- **Models:** Start with **Bradley‑Terry** or **Plackett‑Luce**; if needed, a tiny **Siamese MLP** with pairwise margin loss.  
- **Uncertainty:** Margin between top‑1 and runner‑up scores; route low‑margin sets to you first.  
- **Goal metric:** **Top‑1 agreement** with your pick ≈ **100%** on recent sets.

---

## 5) Concrete Next Steps (small bites)

### Week 1 — Logging & Sidecars
1. **Add logging** to your current web/multi‑crop tools (`session.jsonl` schema above).
2. **Embeddings & Hashes:**
   - Compute **OpenCLIP** embeddings for all images → `sidecar/embeddings.parquet`.
   - Compute **pHash** → `sidecar/hashes.parquet`; dedup with Hamming ≤ 8 and cosine ≥ 0.95.
3. **Detectors & Saliency:**
   - Run **MediaPipe Hands** → `hands/*.json` keypoints and derived geometry.
   - Run **U²‑Net** saliency → `saliency/*.npy` float masks.

### Week 2 — First Recommendations
4. **Crop Proposer v0** (fixed‑aspect, objective above). Output `crops/*.json` + 2 fallback proposals.  
5. **Ranking v0** (Bradley‑Terry over CLIP features + heuristics). Start showing: **Proposed winner** + **Proposed crop** + **confidence**.  
6. **Review UI:** Buttons → `[Approve] [Edit crop] [Pick different] [Delete + anomaly tags]` → everything logs back to `session.jsonl`.

### Week 3 — Learn Faster
7. **Uncertainty routing** (margin/disagreement).  
8. **Add tiny YOLO face/foot** to improve “don’t cut here” penalties.  
9. **Batch propose** toggle: enable only after sustained ~**100%** approval for N sets (e.g., 300). Batch → **post‑batch review queue** with AI‑assist to re‑flag any residual anomalies.

**Optional:** Integrate **Cleanlab/ActiveLab** to focus labeling where it helps most; try **Lightweight SAM** only if saliency struggles on edge cases.

---

## 6) TRAINING.md (drop‑in)

> Recommended path: `ai_training/TRAINING.md` (you can change the folder).

```markdown
# TRAINING.md — Ranking, Fixed-Aspect Cropping, and Anomaly Checks (Mac, Offline-First)

## 0) Environment (Apple Silicon)

- macOS 15.x, Apple **M4 Pro** (24 GB)
- Python ≥ 3.11

### Install
```bash
# Core (PyTorch with Metal/MPS)
pip install torch torchvision torchaudio
pip install lightning accelerate

# Models & image ops
pip install open_clip_torch
pip install opencv-python pillow numpy scipy
pip install imagehash                # pHash
pip install mediapipe                # Hands (CPU/TFLite)
pip install u2net                    # or vendor U^2-Net code/weights
pip install cleanlab                 # optional (active labeling/QA)
```

> Use `device = "mps"` in PyTorch for Apple GPU acceleration.

## 1) Data Layout

```
data/
  raw/
    set_<id>/
      img_001.png
      img_002.png
      meta.yaml
  sidecar/
    embeddings.parquet
    hashes.parquet
    saliency/img_001.npy
    hands/img_001.keypoints.json
    det/img_001.facefoot.json
  labels/
    sessions/2025-10-05__session.jsonl
    crops/img_001.crop.json
  exports/
    approved/
    to_crop/
    cropped/
    review_queue/
```

### `session.jsonl` schema
```json
{"ts":"2025-10-05T19:12:31Z","set_id":"abc123","images":["img_001.png","img_002.png"],
  "winner":"img_002.png","needs_crop":true,"delete":false,
  "crop_box":{"x":123,"y":210,"w":1024,"h":1536},
  "anomaly_tags":["extra_fingers","malformed_phone_grip"]}
```

## 2) Embeddings & Near‑Duplicate Filtering

1) **OpenCLIP embeddings** (ViT‑B/32 by default) → `embeddings.parquet`  
2) **pHash** → `hashes.parquet`  
3) Dedup rule: Hamming ≤ 8 **and** cosine sim ≥ 0.95

## 3) Anomaly Cues

- **MediaPipe Hands** → 21 keypoints/hand; derive digit counts, ratios, bent‑angle outliers, phone‑grip flags.  
- **U²‑Net saliency** → focus on subject silhouette (save as `.npy`).  
- **(Optional) tiny YOLO** → face/foot boxes to define “don’t cut” regions.

## 4) Crop Proposer (fixed aspect)

- **Seeds:** saliency centroid; face box aligned to rule‑of‑thirds; prior accepted crop stats.  
- **Score:** `α·SaliencyIn – β·AnomalyOverlap – γ·HeadCut – δ·JointCut`.  
- **Constraints:** keep **original aspect**; for full‑body images prefer ≥ **1/3 body**.

CLI example:
```bash
python tools/propose_crop.py --img data/raw/.../img.png   --sal data/sidecar/saliency/img.npy   --hands data/sidecar/hands/img.keypoints.json   --aspect from_original   --out data/labels/crops/img.crop.json
```

## 5) Ranking Model (Pairwise Preferences)

- **Features:** CLIP embedding + sharpness + subject size + anomaly counts.  
- **Model:** Bradley‑Terry/Plackett‑Luce or tiny MLP (pairwise loss).  
- **Train:** incremental after each session; device `mps`.

CLI example:
```bash
python train/rank_bt.py   --sessions data/labels/sessions/*.jsonl   --embeddings data/sidecar/embeddings.parquet   --out models/rank_bt.pt --device mps
```

## 6) Inference Flow

1. Dedup set → score with ranking model → **proposed winner**.  
2. Run **crop proposer** → show **Proposed crop** + 2 fallbacks.  
3. UI: `[Approve] [Edit crop] [Pick different] [Delete + anomaly tags]` → log to `session.jsonl`.

## 7) Active Learning & Routing

- **Uncertainty:** small ranking margin, crop score near threshold, or detector disagreement.  
- Route uncertain sets **first** for faster learning. (Optional: `cleanlab` for label QA.)

## 8) Metrics & Batch Promotion

- **Top‑1 agreement** (winner) ≥ **99–100%** on recent sets.  
- **Crop acceptance** without edits ≥ **99–100%**.  
- **Residual anomaly** post‑crop ≈ **0**.  
- When stable for N sets (e.g., 300), enable **batch propose** + **post‑batch review queue**.

## 9) Reproducibility

- Pin `requirements.txt` and save model cards (data timestamp, features, metrics, approval rates).
```

---

## 7) Implementation Notes (Mac‑Friendly)

- Most steps are **CPU‑friendly**; enable `mps` for embedding extraction and tiny model training.  
- Keep all tools **offline**. Optional cloud APIs are not required and omitted here.  
- Scripts to scaffold first (all tiny):
  - `tools/compute_embeddings.py`
  - `tools/compute_phash.py`
  - `tools/run_hands_saliency.py`
  - `tools/propose_crop.py`
  - `train/rank_bt.py`

> If helpful, I can generate minimal working versions of these scripts next so you can start logging today.

---

## Appendix — Notes on References (no external dependencies required)

- **CLIP/OpenCLIP:** robust embeddings for similarity & ranking; many compact backbones.  
- **MediaPipe Hands:** real‑time 21‑keypoint hand tracking on CPU/TFLite.  
- **U²‑Net:** light saliency nets that segment salient objects well.  
- **YOLO (nano/small):** fast face/foot boxes; keep lightweight.  
- **Bradley‑Terry/PL:** classic pairwise ranking models; stable and convex to fit.  
- **pHash:** trivial to compute; reliable for near‑dup detection.

