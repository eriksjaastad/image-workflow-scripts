---
title: Image Dataset Optimization & Assisted Editing
status: Archived (clustering sections); Current (assist-first cropping)
audience: HISTORICAL, DEVELOPER
tags: [clustering, archived, assist-first-cropping]
---

# Image Dataset Optimization & Assisted Editing

This document describes advanced clustering optimization and AI-assisted cropping workflows. Clustering optimization sections are archived; assist-first cropping sections remain relevant.

---

## ⚠️ CURRENT STATUS (October 2025)

### Archived Components
- `02_face_grouper.py` - Moved to `scripts/archive/` (ineffective for AI-generated artwork)
- Face clustering optimization - Not suitable for AI-generated images

### Active Components
- `tool_similar_image_grouper.py` - Color-based image grouping (active)
- `tool_face_grouper.py` - General face grouping (active)
- Manual cropping tools - `01_web_image_selector.py`, `04_multi_crop_tool.py`

### Recommendation
This document describes advanced clustering workflows that may not be suitable for the current AI-generated artwork processing pipeline. Consider archiving or updating based on current needs.

---

## [ARCHIVED] Automated Clustering Optimization Across Multiple Image Directories

Note: This section describes clustering optimization that is not applicable to current AI-generated artwork workflow.

## 0. Cache embeddings (once per directory)

Run the grouper once per dataset to save embeddings, so optimization is fast:

```bash
for d in data/*; do
  python scripts/02_face_grouper.py     --images "$d"     --out "$d/.fg"     --save-embeddings --dry-run
done
```

Each directory now has:
- `X.npy` (embeddings)
- `paths.txt` (file paths)

---

### 1. Parameter sweep (bell-curve search)

For each directory, run ~200–300 trials across **Agglomerative** (your flow) and **Leiden** (graph-based).

**Agglomerative params:**
- `min_cluster_size`: {10, 12, 16, 20, 24, 28}  
- `start_threshold`: {auto, 0.28, 0.26, 0.24, 0.22, 0.20}  
- `merge_dist`: {0.14, 0.16, 0.18, 0.20}  
- `assign_threshold`: {0.16, 0.18, 0.20, 0.22}  
- `second_pass_start`: {0.34, 0.32, 0.30, 0.28}  

**Leiden params:**
- `resolution`: [0.6, 2.4] (continuous)  
- `temperature τ`: {0.8, 1.0, 1.2}  

Run optimizer (random sweep):

```bash
for d in data/*; do
  python optimize_face_grouper.py     --images "$d"     --cache-out "$d/.fg"     --search random --trials 240
done
```

Outputs for each directory:
- `.fg/optimize_results/runs.csv` (all trials + metrics)
- `.fg/optimize_results/rank_*` (top configs with previews and maps)

---

### 2. Within-directory consensus and stability

Instead of trusting one metric:

- **Consensus (EAC):** Take top ~25% runs by score. Build a **co-association matrix** (how often pairs cluster together), cut it to form a **consensus partition**.  
- **Stability:** For each run, compute **ARI vs consensus** and **bootstrap ARI** (subsampled stability).  
- Runs that are unstable *and* disagree with consensus are down-ranked.

This provides a **label-free truth proxy** per directory.

---

### 3. Cross-directory robustness

Aggregate results from all directories:

1. Normalize metrics per dataset (z-scores).  
2. Compute a composite score:

```
score =
  1.20 * silhouette_cosine
+ 0.60 * (-DaviesBouldin)
+ 0.60 * log1p(CalinskiHarabasz)
+ 1.00 * stability_ARI
+ 0.50 * coverage
+ 0.50 * modularity/DBCV
+ 0.40 * (-log1p(singleton_rate))
- 5.00 * penalty (coverage < 0.45 or n_clusters <= 1)
```

3. Aggregate by **parameter signature** across datasets:
   - `avg_rank`: mean rank across datasets
   - `worst_z`: worst-case score across datasets
   - `mean_z`: average score
   - `trials`: count of times tested

**Pick:**
- **Robust global winner** (good avg, strong worst-case)
- **Per-directory winners** (if local tuning is allowed)

---

### 4. Optional human evaluation (light but powerful)

- From top runs, auto-select ~150 **ambiguous pairs**:
  - Borderline merges (near merge threshold)
  - Cross-cluster nearest neighbors
  - Assignments close to threshold

- Ask “Same character?” → yes/no/unsure.  
- Compute **pairwise precision/recall** and **Matthews corr.** for each run.  
- Re-weight composite score using this signal (logistic regression or Bradley–Terry).  

This converts your “shade of gray” intuition into measurable accuracy.

---

### 5. Reporting

For each directory and globally:

- **Leaderboard:** params, score, k, coverage, stability, silhouette, modularity.  
- **Consensus heatmap:** co-association matrix visualization.  
- **Silhouette violin:** cluster cohesion per group.  
- **Pareto plot:** stability vs silhouette (coverage as point size).  
- **Parameter dependence plots:** how score changes with knobs.  
- **Recommendation card:** final params (global + per-directory), expected metrics, trade-offs.

---

### 6. Aggregator script

After sweeps, run:

```bash
python aggregate_across_dirs.py
```

This produces:
- `_meta/all_runs.csv` (all runs across all dirs)
- `_meta/per_dataset_winners.csv` (best per dataset)
- `_meta/robust_signature_leaderboard.csv` (best global config)

**Console output:**
- Robust global winner (params + stats)
- Per-dataset winners (params + stats)

---

## Why this works

- **Best results:** broad sweep + stability + consensus avoids cherry-picking one lucky run.  
- **Best analysis:** per-directory and cross-directory ranking with transparent metrics.  
- **Human-in-the-loop (optional):** converts subjective accuracy checks into measurable, reusable weights.  

This gives you both the *bell curve of tests* and the *robust, data-driven pick* for each dataset and globally.

---

## 7. Example Aggregator Script

```python
#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path("data")  # parent containing your directories

def composite(row):
    sil = np.nan_to_num(row.sil, nan=-1)
    dbi = np.nan_to_num(row.dbi, nan=10)
    chi = np.nan_to_num(row.chi, nan=0)
    cov = np.nan_to_num(row.coverage, nan=0)
    sing = np.nan_to_num(row.singleton_rate, nan=1)
    stab = np.nan_to_num(row.stability_ari, nan=0)
    mod = np.nan_to_num(row.modularity, nan=0)
    score = (1.2*sil + 0.6*(-dbi) + 0.6*np.log1p(max(chi,0))
             + 1.0*stab + 0.5*cov + 0.5*mod + 0.4*(-np.log1p(max(sing,1e-6))))
    if (cov < 0.45) or (row.n_clusters <= 1):
        score -= 5.0
    return score

rows = []
for d in sorted([p for p in ROOT.iterdir() if (p/".fg/optimize_results/runs.csv").exists()]):
    df = pd.read_csv(d/".fg/optimize_results/runs.csv")
    df["dataset"] = d.name
    need = ["sil","dbi","chi","coverage","singleton_rate","stability_ari","modularity","n_clusters"]
    for c in need:
        if c not in df.columns: df[c] = np.nan
    df["score_raw"] = df.apply(composite, axis=1)
    for c in ["sil","dbi","chi","coverage","singleton_rate","stability_ari","modularity","score_raw"]:
        z = (df[c] - df[c].mean()) / (df[c].std(ddof=0) + 1e-9)
        df[c + "_z"] = z.replace([np.inf,-np.inf], 0)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
per_ds = (all_df.sort_values(["dataset","score_raw_z"], ascending=[True,False])
                .groupby("dataset").head(1)
                .reset_index(drop=True))

param_cols = [c for c in all_df.columns if c in
              ["min_cluster_size","start_threshold","merge_dist","assign_threshold","second_pass_start",
               "resolution","temp","map_topk","map_threshold","map_scope"]]
all_df["signature"] = all_df[param_cols].astype(str).agg("|".join, axis=1)

all_df["rank_in_ds"] = all_df.groupby("dataset")["score_raw_z"].rank(ascending=False, method="min")
sig_stats = (all_df.groupby("signature")
                  .agg(avg_rank=("rank_in_ds","mean"),
                       worst_z=("score_raw_z","min"),
                       mean_z=("score_raw_z","mean"),
                       trials=("signature","size"))
                  .reset_index()
                  .sort_values(["avg_rank","worst_z","mean_z"], ascending=[True,False,False]))

robust = sig_stats.head(1)

out = ROOT / "_meta"
out.mkdir(exist_ok=True)
all_df.to_csv(out/"all_runs.csv", index=False)
per_ds.to_csv(out/"per_dataset_winners.csv", index=False)
sig_stats.to_csv(out/"robust_signature_leaderboard.csv", index=False)

print("\nRobust winner (signature):")
print(robust.to_string(index=False))
print("\nPer-dataset winners:")
print(per_ds[["dataset","score_raw","score_raw_z","n_clusters"] + param_cols].to_string(index=False))
print(f"\nSaved: {out/'all_runs.csv'}")
```

---

## Part B — Assist‑First Cropping (Same‑Ratio) & Speed Review

This add‑on turns your existing “crop three at once” tool and triage flow into a **one‑tap‑most‑of‑the‑time** workflow—without going full auto. It keeps the **original aspect ratio**, helps you hide AI oddities (extra fingers/toes, phantom limbs), and learns from every Accept/Override.

## B1. What you see (operator flow)
- For each image (or trio), the tool shows **1 best crop + 2 alternates**, all **exactly the same ratio** as the original.
- The **best crop is pre‑selected**. One tap = **Accept**. If it looks off, tap an alternate or **Nudge** (drag) and Accept.
- Optional small overlay labels: *“odd fingers”*, *“stray limb”*, *“awkward edge”*—what the suggester tried to avoid.
- Nothing moves without your OK. Later, you can enable “auto‑accept when super obvious.”

**Buttons:** `Accept` · `Alternate 1` · `Alternate 2` · `Nudge` · `Next`

## B2. How it decides (plain English)
1) **Anchor the subject.** Keep the face nicely framed (or torso if no face). Don’t crop eyes, don’t squash headroom.
2) **Find “don’t‑show” zones.** Look for hand/foot shapes and limb lines that don’t make sense—these become **keep‑out areas** near the frame edges.
3) **Slide a same‑ratio window** at a few zoom levels (e.g., 95%, 85%, 75%) and pick windows that **keep the anchor** while **avoiding keep‑out zones**.
4) **Order by confidence.** Clean, anchor‑true, artifact‑free crops come first; trickier ones come later in your queue.

> Works even when faces are missing: it falls back to torso/center‑mass as the anchor.

## B3. Fits your trio tool
- The suggester writes a sidecar file per image: `image.jpg.crop_suggestions.json` with:
  ```json
  {
    "ratio": "original",
    "candidates": [
      {"box": [x1, y1, x2, y2], "note": "best"},
      {"box": [x1, y1, x2, y2], "note": "alt1"},
      {"box": [x1, y1, x2, y2], "note": "alt2"}
    ],
    "keep_out": [{"shape": "hand", "region": [x1,y1,x2,y2]}]
  }
  ```
- Your UI loads this file, pre‑selects **best**, and shows `alt1/alt2` buttons. If absent, it uses your current default crop.

## B4. What to log (so it learns your taste)
For each image:
- `choice`: `best` | `alt1` | `alt2` | `nudged`
- `nudged_box`: final box if you dragged
- `reason_tags` (optional, quick taps): `odd_eyes`, `extra_fingers`, `toes`, `limb_overlap`, `glare`, `awkward_edge`, `background_distraction`
- `time_ms`: how long you spent

These tiny signals are enough to steer future suggestions toward your style.

## B5. Success readouts (to know it’s working)
- **Crop Accept Rate** = accepted without nudge (target 80–90% after a few sessions)
- **Avg Actions/Image** = taps or drags before Accept (should fall over time)
- **Queue Mix** = % easy vs hard images shown
- **Time Saved / 100 images** (your real‑world win)

## B6. Non‑face matching (body/age) support
- When faces are weak or hidden, similarity scoring blends **body‑type cues** and general appearance so the “Same person?” queue stays useful. You still approve every decision; suggestions simply surface the likely matches first.

## B7. How it improves without heavy training
- Every **Accept** is a positive lesson; every **Alternate/Nudge** is a gentle correction.
- The suggester keeps a small, local memory per dataset so results get better across sessions—even if you never run a big training job.

---

## Part C — Putting it all together (2‑hour day target)

1) **Pre‑sort trios by “obviousness.”** Clear winners first; top pick pre‑selected. One tap moves you along.
2) **Run the “Same person?” queue** (face → body/age). You only see borderline pairs.
3) **Use the Assist‑First Cropper** (same‑ratio) with one‑tap accept most of the time.
4) **Batch metrics at the end:** Accept Rate, Ask Rate, Time Saved. If any metric stalls, widen/adjust thresholds slightly.

> You remain in **always‑ask mode** until you decide to enable auto‑accept for “very obvious” cases.

---

## Appendix — Minimal file formats (so tools interoperate)

## A1. Crop suggestions sidecar (`*.crop_suggestions.json`)
```json
{
  "version": 1,
  "ratio": "original",
  "candidates": [
    {"id": "best", "box": [x1, y1, x2, y2], "note": "anchor+clean"},
    {"id": "alt1", "box": [x1, y1, x2, y2], "note": "avoid hand edge"},
    {"id": "alt2", "box": [x1, y1, x2, y2], "note": "avoid toes"}
  ],
  "keep_out": [
    {"shape": "hand", "region": [x1, y1, x2, y2]}
  ]
}
```

## A2. Crop decision log (`crop_decisions.csv`)
```
image,choice,nudged_box,reason_tags,time_ms
foo.jpg,best,,,
bar.jpg,nudged,"[34,60,910,1360]","awkward_edge;glare",4200
```

## A3. Trio winner log (`trio_choices.csv`)
```
group_id,winner,loser1,loser2,tags,time_ms
123,im_001.jpg,im_002.jpg,im_003.jpg,extra_fingers,1800
```

These tiny, human‑readable files let your existing tools slot together and improve over time.

---

# Current Workflow (October 2025)

## **Active Image Processing Pipeline:**

### **1. Image Selection & Cropping:**
- `01_web_image_selector.py` - Web-based image selection with cropping
- `01_desktop_image_selector_crop.py` - Desktop image selection with cropping
- `04_multi_crop_tool.py` - Multi-image batch cropping tool

### **2. Character Sorting:**
- `02_web_character_sorter.py` - Web-based character sorting by groups

### **3. Image Grouping (Alternative to Clustering):**
- `tool_similar_image_grouper.py` - Color-based image grouping
- `tool_face_grouper.py` - General face grouping

### **4. File Management:**
- `05_web_multi_directory_viewer.py` - Multi-directory file viewing
- `06_web_duplicate_finder.py` - Duplicate file detection

## **Key Changes from Original Design:**
1. **Face clustering abandoned** - Ineffective for AI-generated artwork
2. **Manual workflow preferred** - More reliable than automated clustering
3. **File-operation timing** - Intelligent work time calculation
4. **Centralized utilities** - Shared functions across all tools

---

*Last Updated: October 3, 2025*
*This document has been updated to reflect the current state of the image processing workflow.*
