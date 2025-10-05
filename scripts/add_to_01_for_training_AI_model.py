"""
AI Training Plan: Select + Crop Model (Desktop Selector Workflow)
===============================================================

This document lays out a concrete, step-by-step plan to train a model that
learns your selection and cropping preferences from the desktop selector tool.

Phases
------
1) Logging — capture supervision from real work (selection + crop)
2) Dataset building — create ranking and crop regression samples
3) Training — two-head model (ranking + bbox)
4) Evaluation — Top-1, IoU, MAE
5) Inference integration — suggest selection + crop in the desktop tool

Phase 1 — Logging (no workflow disruption)
-----------------------------------------
Desktop selector (authoritative labels):
- On Enter:
  - If one image is selected with a crop: append a row to
    data/training/select_crop_log.csv
  - If delete-all: optionally log a negative-only record
- Columns (one row per submission):
  - session_id: stable session identifier
  - set_id: stable group id (triplet id)
  - directory: base directory
  - image_count: 2–4
  - chosen_index: -1 if delete-all else 0..N-1
  - chosen_path: full path (empty if delete-all)
  - crop_x1, crop_y1, crop_x2, crop_y2: normalized [0..1] (empty if delete-all)
  - image_i_path for i in 0..N-1 (full paths)
  - image_i_stage for i in 0..N-1 (stage2_upscaled, etc.)
  - width_i, height_i for i in 0..N-1
  - timestamp: ISO8601

Web selector (optional negatives):
- On batch finalize, for each group moved to selected/:
  - Log the chosen image path and the other images from the group as negatives.
- CSV path: data/training/selection_only_log.csv
- Columns: session_id, set_id, chosen_path, neg_paths(list/JSON), timestamp

Notes:
- Timestamps are ONLY for sorting; logs/datasets must pre-sort via the centralized sorter
- Normalize crop coords to [0..1] to be resolution-agnostic

Phase 2 — Dataset builder
-------------------------
Create scripts/datasets/build_select_crop_dataset.py that:
1) Reads select_crop_log.csv and selection_only_log.csv
2) Validates files exist; drops broken entries
3) Splits by set_id into train/val/test (e.g., 80/10/10)
4) Emits two artifacts:
   - A) Pairwise ranking samples: (anchor=chosen, negative=one other) with label=1
   - B) Crop regression samples: (image, bbox normalized)
5) Optional: generate COCO-style JSON for the crop head (bbox in absolute px and image size)
6) Write a manifest JSONL for fast PyTorch/TF loading (one JSON per sample)

Phase 3 — Model & training
--------------------------
Architecture (PyTorch, timm):
- Backbone: ViT-B/16 (or ConvNeXt-T) pretrained (ImageNet-1k)
- Shared encoder → two heads:
  1) Ranking head (embedding of dimension D, e.g., 256). Loss: MarginRankingLoss
     Use pair sampling from each group: (chosen, negative)
  2) Crop head (bbox regression): 4 outputs (x1,y1,x2,y2 normalized). Loss: SmoothL1 + IoU loss
- Total loss = ranking_loss_weight * L_rank + bbox_loss_weight * L_bbox
  Start with 1.0 : 2.0 weighting (crop a bit heavier) and tune

Data pipeline:
- Augmentations (Albumentations): random brightness/contrast, small color jitter, flips
- IMPORTANT: Keep crop labels consistent if geometric transforms are applied
- Normalize to backbone’s expected mean/std; resize preserving aspect ratio

Training schedule:
- Optimizer: AdamW, lr ~3e-5–5e-5 for backbone, 1e-4 for heads
- Cosine decay or step LR; epochs 10–30 depending on dataset size
- Mixed precision (amp) enabled

Metrics:
- Selection: Top-1 accuracy over groups; MRR as secondary
- Crop: IoU@0.5, mean IoU, MAE in pixels (after denorm)

Phase 4 — Evaluation
--------------------
Add scripts/notebooks:
- scripts/eval/eval_select_crop.py → prints Top-1, IoU metrics on val/test
- notebooks/eval_select_crop.ipynb → qualitative plots (GT vs predicted crops)

Phase 5 — Inference integration (desktop tool)
----------------------------------------------
1) Load trained weights once on startup (lazy load with a flag: --ai-suggest)
2) When showing a batch (2–4 images):
   - Produce a ranking score for each image, pick top-1 as suggestion
   - Produce a crop for the top-1 image; draw suggested rectangle in white
   - Hotkeys:
     - T → toggle AI suggestion on/off
     - Y → accept suggestion (auto-select suggested image + set crop)
3) UI affordance:
   - Show “AI suggested: stage2_upscaled (92%)” small text under the image title
4) Safety:
   - Accepting suggestion never auto-submits; you can still tweak the crop

Environment & commands
----------------------
Create a dedicated env (or use .venv311):
  pip install torch torchvision timm albumentations scikit-learn pycocotools tqdm pyyaml rich

Repository additions
--------------------
- scripts/datasets/build_select_crop_dataset.py     # builds manifests + COCO json
- scripts/train/train_select_crop.py                # training loop (multi-head)
- scripts/eval/eval_select_crop.py                  # metrics + reports
- notebooks/eval_select_crop.ipynb                  # qualitative review

Pseudocode (dataset builder)
----------------------------
for row in select_crop_log.csv:
    if row.chosen_index >= 0:
        chosen = row.image_[chosen_index]
        for j in images_except(chosen):
            yield ranking_sample(chosen, j)
        yield crop_sample(chosen, bbox=row.crop)

Pseudocode (training loop)
--------------------------
for batch in loader:
    # ranking branch
    f_pos = enc(pos)
    f_neg = enc(neg)
    l_rank = margin_ranking_loss(f_pos, f_neg)
    # bbox branch
    pred_bbox = head_bbox(enc(img))
    l_bbox = smooth_l1(pred_bbox, gt_bbox) + iou_loss(pred_bbox, gt_bbox)
    total = w1*l_rank + w2*l_bbox
    total.backward(); optimizer.step()

Rollout plan
------------
1) Implement logging in desktop selector (CSV write on Enter)
2) Build dataset from a day of work; run a tiny training to validate pipeline
3) Expand dataset with more sessions; increase epochs
4) Integrate inference (behind --ai-suggest flag)
5) Iterate on weights/augmentations based on IoU and Top-1 results

Notes
-----
- Timestamps are ONLY for sorting; dataset builder must pre-sort via the centralized sorter
- Always normalize crop coords to [0..1] to be resolution-agnostic
- Keep train/val/test split by set_id to avoid leakage
"""