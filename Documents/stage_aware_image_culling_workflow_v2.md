# Stage-Aware Image Culling Workflow (Version 2)

This document outlines a **refined, stage-aware workflow** for selecting the best image among near-duplicates with minimal manual effort. It blends traditional culling practices with automated scoring and human review.

---

## 1. Goals & Context

- You have sets of near-identical images captured or generated at different **stages** (Stage 1, 1.5, 2, 3…).  
- Within each set, you generally want the **highest stage number** (latest version) **unless** it has a visible defect.  
- Defects may include missing details (e.g., a missing belly button), bizarre artifacts (multiple faces stacked), or subtle anomalies.  
- You want to thin the herd safely and review only the exceptions, not every single image.

---

## 2. Core Workflow Overview

| Pass | Purpose | Operation | Output |
|---|---|---|---|
| **Pass 0: Import & Cluster** | Group potential variants | Cluster images by time & content similarity (hashing/embedding) | Each image gets a `group_id` |
| **Pass 1: Defect Filter** | Flag obviously bad images | Apply simple quality metrics (blur, exposure, artifact scoring) | Images marked `defect=True/False` |
| **Pass 2: Stage Sorting & Candidate Pick** | Pick best candidate per group | Within each `group_id`, sort by `stage_number DESC`, then by `defect=False` | “Default keeper” list + “Review list” |
| **Pass 3: Human Review of Ambiguities** | Handle exceptions | Show side-by-side comparisons for groups where the top stage had a defect or ambiguous score | Manual override / final selection |
| **Pass 4: Final Consistency Sweep** | Ensure global quality | Quick scan of all chosen images for outliers | Final clean set |

---

## 3. Pass Details

### Pass 0 — Import & Cluster
- Use **perceptual hash** or **CLIP embeddings** with a small time window to group images that are likely the same “family.”
- Each group forms a candidate set for stage-based selection.

### Pass 1 — Defect Filter
- Compute basic metrics: sharpness, histogram clipping, noise level, color shifts.
- Optionally run a lightweight anomaly detector trained on your own “bad” examples.
- Mark `defect=True` if any threshold exceeded.

### Pass 2 — Stage Sorting & Candidate Pick
- Sort each group by `stage_number DESC` (Stage 3 → Stage 2 → Stage 1).
- Choose the first non-defective image as the provisional keeper.
- If all high-stage images have defects, pick the next best stage automatically but flag the group for manual review.

### Pass 3 — Human Review of Ambiguities
- For flagged groups, display all variants side-by-side with stage numbers and defect scores.
- Let the user override the pick (e.g., pick Stage 1 if Stage 3 and Stage 2 are both defective).

### Pass 4 — Final Consistency Sweep
- Once all groups have a provisional keeper, do a fast visual sweep for anomalies that slipped through (weird crops, missing features, etc.).

---

## 4. Scoring & Weights (Optional Future Extension)

You can treat each image’s “goodness” as a **weighted score**:

```
Score = (StageWeight * StageNumber) 
      + (QualityWeight * QualityMetric) 
      - (DefectPenalty * DefectFlags)
```

Example weights:
- StageWeight = 2.0
- QualityWeight = 1.0
- DefectPenalty = 5.0

The image with the highest score in each group becomes the provisional keeper.

This allows you to automatically rank images by “best balance of highest stage + fewest defects” while still letting a human check the edge cases.

---

## 5. Why This Works

- **Stage-aware:** honors your desire to pick the highest stage first.
- **Defect-aware:** automatically demotes images with obvious anomalies.
- **Low-touch:** you only hand-review groups where automation is uncertain.
- **Extensible:** you can later plug in smarter detectors (missing body parts, artifact detection) as you build them.

---

## 6. Next Steps

- Test the workflow on a small subset to calibrate thresholds.
- Feed results to Claude (or any AI helper) to brainstorm detection features you can automate.
- Over time, adjust StageWeight, QualityWeight, and DefectPenalty to match your preferences.

