# Possible Scripts for Detecting Hand & Foot Anomalies in AI Images
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


This document outlines several production‑quality approaches you can try for automatically flagging problematic hands and feet in photo‑realistic AI images. Each entry includes a short description, pros, cons, and notes.

---

## 1. **MediaPipe Hands + MediaPipe Pose (Google)**

**What it does:**  
Detects 21 3D landmarks per hand and 33 2D+“world” landmarks for the body including ankles/feet.

**Pros:**  
- Very fast (real‑time on CPU).  
- Well maintained, simple Python API.  
- Works well on photo‑realistic images.  
- Easy to integrate into batch scripts.

**Cons:**  
- Limited direct “anomaly” score — you must build your own rules (finger counts, angle checks).  
- Sometimes confuses overlapping hands or occluded fingers.  
- Feet coverage less detailed than hands.

**Use case:**  
Quick first‑pass anomaly screening; output keypoints and build scoring heuristics.

---

## 2. **OpenPose (Carnegie Mellon)**

**What it does:**  
Detects body + hand + foot keypoints (up to 135 landmarks) in one pass.

**Pros:**  
- Single engine for body/hands/feet.  
- Mature, widely cited.  
- Good for multi‑person scenes.

**Cons:**  
- Heavier than MediaPipe; GPU recommended.  
- More setup (Caffe or PyTorch).  
- Still mainly 2D (no real 3D world coords).

**Use case:**  
When you want one detector for everything and can afford more compute.

---

## 3. **FrankMocap (SMPL‑X Fitting)**

**What it does:**  
Takes an image, fits a full 3D body mesh (including articulated hands and feet) using the SMPL‑X model.

**Pros:**  
- Outputs full 3D joint angles, bone lengths, interpenetration flags.  
- Captures “impossible” kinematics that 2D keypoints can’t.  
- Python package, easy batch mode.

**Cons:**  
- Slower than keypoint‑only detectors.  
- Requires PyTorch and GPU for speed.  
- Fit quality can vary with occlusion or artistic textures.

**Use case:**  
Second‑stage verification for borderline images flagged by keypoint heuristics.

---

## 4. **SMPLify‑X / Other SMPL‑X Fitting Variants**

**What it does:**  
Optimizes SMPL‑X parameters to match detected keypoints for extremely accurate 3D pose + shape.

**Pros:**  
- Most accurate 3D body/hand/foot estimate available.  
- Lets you enforce hard joint‑angle or bone‑length constraints.  
- Strong research community support.

**Cons:**  
- Even slower than FrankMocap.  
- More dependencies, heavier optimization.  
- Not “one‑click” — better as an offline QA tool.

**Use case:**  
Highest‑precision offline anomaly checking or dataset curation.

---

## 5. **Research‑Grade “Diffusion Hand/Foot Fixers”**

**What it does:**  
Experimental detectors specifically trained on AI‑generated hand anomalies (extra fingers, fused toes).

**Pros:**  
- Tuned for diffusion artifacts.  
- Can sometimes flag subtle issues keypoints miss.

**Cons:**  
- Not production‑ready; fragile setups.  
- Limited support and documentation.  
- Hard to batch at scale.

**Use case:**  
Optional side experiment if you want to explore bleeding‑edge research.

---

## Implementation Notes & Tips

- **Two‑Stage Pipeline Works Best:** fast keypoints first, heavy 3D fit second.  
- **Build Transparent Scores:** count fingers/toes, check ratios, joint angles; log reasons.  
- **Batch Processing:** All of the above have Python bindings; you can write a CLI that outputs CSV + overlays.  
- **AI Photo‑Realism Helps:** these models perform closer to spec on realistic images than on stylized art.  
- **Ground‑Truth Calibration:** label a small set of your images manually to set thresholds and evaluate precision/recall.

---

### Quick Test Plan (Optional)

1. Select 50–100 of your AI images.  
2. Run MediaPipe Hands + Pose to output landmarks.  
3. Implement simple rules (finger count, angle bounds).  
4. For borderline scores, run FrankMocap.  
5. Compare results to your own manual flags.

This will give you a clear picture of what the pipeline “sees” versus what you catch manually.

