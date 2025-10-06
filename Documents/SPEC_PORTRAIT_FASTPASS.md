# Spec: Portrait Fast‑Pass (MediaPipe Pose)

Goal: Identify head‑and‑shoulders portraits that generally need no crop, route them to a fast lane where the highest stage is auto‑preferred (still reviewable).

---

## Signals
- Face box (MediaPipe Face or Pose face landmark bounding box)
- Shoulder keypoints: `LEFT_SHOULDER`, `RIGHT_SHOULDER`
- Optional torso estimate: shoulder → hip line; require hips largely out of frame for tight portraits

## Heuristics (initial)
- Face area fraction ≥ 8–12% of image area (tune)
- Shoulder span fraction w.r.t. image width ∈ [0.25, 0.60] (tune)
- Hips visibility: absent or near bottom crop (suggests head+shoulders)
- Centering: face center near image center (e.g., within 20% of width/height center)

Ambiguity: if borderline (within tolerance bands) → mark as `portrait_maybe` and send to normal flow.

---

## Routing
- Tag images as `portrait_true` or `portrait_maybe` in a CSV/JSON side report (no file moves initially).
- Option A: Pre‑stage `portrait_true` into `portraits_fastpass/` (companion‑safe move) for quick pass.
- Option B: Keep in place; the web selector shows a “Portrait” badge and enables a fast‑lane filter.

Default decision rule in fast‑pass: pick the highest stage number that passes anomaly gates; do not send to `crop/`.

---

## Calibration
- Create a labeled subset (100–200 images across characters/stages) with human tags: portrait / not portrait.
- Grid search thresholds (face fraction, shoulder span, centering) to achieve ≥98% precision with acceptable recall.
- Record confusion cases to refine rules (e.g., tilted shoulders, occlusions).

---

## Output
- CSV: `image_path,portrait_tag,face_fraction,shoulder_span,center_offset`
- JSON (optional): array with scores per image

---

## Safety
- Analysis-only first (no moves). Switch to staging only after thresholds stabilize.
- Companion-safe operations if staging is enabled.
