# Dashboard Ideas - Vision Board

This document collects dashboard visualization ideas without immediately building them. Think of it as a daydreaming space for data insights we might want to see someday.

## Image Groups Visualization

**The Problem:** Traditional metrics like "10,000 images processed" are misleading when doing best-of-N selection.

**Better Metrics:**
```
📊 Project: mojo3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source Images:     12,450
Image Groups:       4,150 (avg 3.0 per group)
                    ├─ Pairs: 0 (0%)
                    ├─ Triplets: 4,150 (100%)
                    └─ Quads: 0 (0%)

Selection Phase:
  Selected:         4,150 (33% of source, 100% of groups)
  Skipped:          8,300 (67% of source)
  Work time:        ~18 hours (estimated)

Crop Phase:
  Cropped:          1,850 (45% of selected)
  Queued:           0
  Remaining:        2,300 (55%)
  Work time:        ~8 hours (in progress)

Training Data:
  Selection logs:   4,150 ✓
  Crop logs:        1,850 ✓
  CLIP embeddings:  4,150 ✓
```

**Why This Matters:**
- Project A: 9,000 images (3,000 groups of 3) → 3,000 selected → 33% selection rate
- Project B: 9,000 images (9,000 singles) → 9,000 selected → 100% selection rate
- Same input count, wildly different workload and output!

**Group count is the real workload metric.**

## Cross-Project Historical Stats

Track patterns across all projects to understand efficiency and identify anomalies:

```
📈 Cross-Project Stats
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total groups processed:    47,250
Avg group size:            2.8 images/group
Selection efficiency:      38% (selected/source)
Crop rate:                 42% (cropped/selected)
Total work hours:          ~340 hours
Avg time per group:        26 seconds
```

**Potential Insights:**
- Are we getting faster over time?
- Do certain projects have unusually high/low selection rates?
- Which projects took way longer than expected?
- Is group size correlated with work speed?

## Work Time Tracking Reality

**Current Problem:** File timestamps don't reflect actual work time when batch processing hundreds of images at once.

**Ideas to Explore:**
1. Session-based timing (start/stop markers)
2. Batch-level timing (time between batch submits)
3. Idle detection (exclude gaps >5 minutes)
4. Manual session tracking (CLI prompts)

**Ideal Dashboard View:**
```
Work Session Summary:
  Started:          09:00 AM
  Ended:            02:30 PM
  Elapsed:          5.5 hours
  Active work:      4.2 hours (76%)
  Breaks/idle:      1.3 hours (24%)
  Batches:          42
  Images:           714
  Avg per batch:    6 minutes
```

## Training Data Health

Visualize completeness and quality of training data:

```
Training Data Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project          Groups  Selections  Crops  Embeddings  Status
────────────────────────────────────────────────────────────
mojo1             8,450      8,450   3,200      8,450   ✓ Complete
mojo2             6,750      6,750   2,800      6,750   ✓ Complete
mojo3             4,150      4,150   1,850      4,150   ⚠ In Progress
project_x         2,100      2,100       0          0   ⚠ Missing crops
project_y         1,500          0       0          0   ⚠ No data
────────────────────────────────────────────────────────────
Total            22,950     21,450   7,850     19,350
```

## Project Comparison

Side-by-side comparison to identify outliers:

```
Project Efficiency Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  mojo1    mojo2    mojo3    Average
Groups:           8,450    6,750    4,150
Selection rate:     42%      38%      33%      38%
Crop rate:          38%      41%      45%      41%
Time per group:     28s      24s      22s      25s
Total hours:        66h      45h      24h
```

**Questions This Could Answer:**
- Is mojo3's low selection rate normal or a problem?
- Why is mojo3 faster per-group than mojo1?
- Which project workflow was most efficient?

## Bottleneck Analysis

Identify where time is being spent:

```
Time Breakdown: mojo3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase            Hours    % Total    Notes
────────────────────────────────────────
Selection         18h        43%     Best-of-3 decisions
Cropping (UI)     12h        29%     Manual crop rectangles
Crop queue         8h        19%     Batch processing queued crops
AI training        4h         9%     Model training/validation
────────────────────────────────────
Total             42h       100%
```

**Optimization Ideas:**
- Can we reduce selection time with better preloading?
- Is crop queue underutilized?
- Could AI suggestions speed up manual cropping?

## Data Gaps Dashboard

Quick visual of what's missing:

```
🔍 Data Completeness Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All projects have manifests
✓ All selected images have .yaml companions
⚠ 3 projects missing crop decisions
⚠ 1 project has no training data logged
✗ 5 projects missing time tracking data
✗ Historical archive has no timestamps
```

---

## How to Use This Document

1. **Add ideas** whenever dashboard inspiration strikes
2. **Don't build yet** - just capture the vision
3. **Review periodically** to see if an idea has become useful
4. **Prioritize** based on actual pain points, not "nice to have"

When an idea graduates to implementation, move it to a proper spec in `Documents/dashboard/`.
