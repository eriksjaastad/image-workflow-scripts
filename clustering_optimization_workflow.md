# Automated Clustering Optimization Across Multiple Image Directories

This document describes a **single, rigorous workflow** to: 1. Run a
broad bell-curve of clustering parameter sweeps on **each dataset
directory** (\~1000 images each). 2. Generate **per-directory consensus
partitions** and stability measures (to avoid guessing). 3.
Cross-reference runs across **multiple directories** to find robust,
generalizable parameter settings. 4. Optionally add a **small human
evaluation layer** to calibrate metrics with "ground-truth-ish"
feedback.

------------------------------------------------------------------------

## 0. Cache embeddings (once per directory)

Run the grouper once per dataset to save embeddings, so optimization is
fast:

``` bash
for d in data/*; do
  python scripts/02_face_grouper.py     --images "$d"     --out "$d/.fg"     --save-embeddings --dry-run
done
```

Each directory now has: - `X.npy` (embeddings) - `paths.txt` (file
paths)

------------------------------------------------------------------------

## 1. Parameter sweep (bell-curve search)

For each directory, run \~200--300 trials across **Agglomerative** (your
flow) and **Leiden** (graph-based).

**Agglomerative params:** - `min_cluster_size`: {10, 12, 16, 20, 24,
28}\
- `start_threshold`: {auto, 0.28, 0.26, 0.24, 0.22, 0.20}\
- `merge_dist`: {0.14, 0.16, 0.18, 0.20}\
- `assign_threshold`: {0.16, 0.18, 0.20, 0.22}\
- `second_pass_start`: {0.34, 0.32, 0.30, 0.28}

**Leiden params:** - `resolution`: \[0.6, 2.4\] (continuous)\
- `temperature τ`: {0.8, 1.0, 1.2}

Run optimizer (random sweep):

``` bash
for d in data/*; do
  python optimize_face_grouper.py     --images "$d"     --cache-out "$d/.fg"     --search random --trials 240
done
```

Outputs for each directory: - `.fg/optimize_results/runs.csv` (all
trials + metrics) - `.fg/optimize_results/rank_*` (top configs with
previews and maps)

------------------------------------------------------------------------

## 2. Within-directory consensus and stability

Instead of trusting one metric:

-   **Consensus (EAC):** Take top \~25% runs by score. Build a
    **co-association matrix** (how often pairs cluster together), cut it
    to form a **consensus partition**.\
-   **Stability:** For each run, compute **ARI vs consensus** and
    **bootstrap ARI** (subsampled stability).\
-   Runs that are unstable *and* disagree with consensus are
    down-ranked.

This provides a **label-free truth proxy** per directory.

------------------------------------------------------------------------

## 3. Cross-directory robustness

Aggregate results from all directories:

1.  Normalize metrics per dataset (z-scores).\
2.  Compute a composite score:

```{=html}
<!-- -->
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

3.  Aggregate by **parameter signature** across datasets:
    -   `avg_rank`: mean rank across datasets
    -   `worst_z`: worst-case score across datasets
    -   `mean_z`: average score
    -   `trials`: count of times tested

**Pick:** - **Robust global winner** (good avg, strong worst-case) -
**Per-directory winners** (if local tuning is allowed)

------------------------------------------------------------------------

## 4. Optional human evaluation (light but powerful)

-   From top runs, auto-select \~150 **ambiguous pairs**:
    -   Borderline merges (near merge threshold)
    -   Cross-cluster nearest neighbors
    -   Assignments close to threshold
-   Ask "Same character?" → yes/no/unsure.\
-   Compute **pairwise precision/recall** and **Matthews corr.** for
    each run.\
-   Re-weight composite score using this signal (logistic regression or
    Bradley--Terry).

This converts your "shade of gray" intuition into measurable accuracy.

------------------------------------------------------------------------

## 5. Reporting

For each directory and globally:

-   **Leaderboard:** params, score, k, coverage, stability, silhouette,
    modularity.\
-   **Consensus heatmap:** co-association matrix visualization.\
-   **Silhouette violin:** cluster cohesion per group.\
-   **Pareto plot:** stability vs silhouette (coverage as point size).\
-   **Parameter dependence plots:** how score changes with knobs.\
-   **Recommendation card:** final params (global + per-directory),
    expected metrics, trade-offs.

------------------------------------------------------------------------

## 6. Aggregator script

After sweeps, run:

``` bash
python aggregate_across_dirs.py
```

This produces: - `_meta/all_runs.csv` (all runs across all dirs) -
`_meta/per_dataset_winners.csv` (best per dataset) -
`_meta/robust_signature_leaderboard.csv` (best global config)

**Console output:** - Robust global winner (params + stats) -
Per-dataset winners (params + stats)

------------------------------------------------------------------------

## Why this works

-   **Best results:** broad sweep + stability + consensus avoids
    cherry-picking one lucky run.\
-   **Best analysis:** per-directory and cross-directory ranking with
    transparent metrics.\
-   **Human-in-the-loop (optional):** converts subjective accuracy
    checks into measurable, reusable weights.

This gives you both the *bell curve of tests* and the *robust,
data-driven pick* for each dataset and globally.

------------------------------------------------------------------------

## 7. Example Aggregator Script

``` python
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

------------------------------------------------------------------------
