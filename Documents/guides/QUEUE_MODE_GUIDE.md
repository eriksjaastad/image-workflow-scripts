# Crop Queue Mode - Usage Guide

## Overview

Queue Mode decouples the **interactive cropping** (setting coordinates) from the **image processing** (loading, cropping, saving). This allows you to crop as fast as you can think, without waiting for image I/O operations!

## Benefits

- **2-3x faster** interactive workflow - no waiting for image processing
- **Process later** with realistic human timing patterns (optional)
- **Resilient** - if processor crashes, queue persists
- **Complete timing data** for analysis

## Workflow

### Step 1: Queue Your Crops (FAST!)

Use the crop tool in queue mode:

```bash
python scripts/02_ai_desktop_multi_crop.py __crop_auto/ --queue-mode
```

This will:

- Let you set crop coordinates at full speed (no I/O delays!)
- Write crop operations to `data/crop_queue/crop_queue.jsonl`
- Move processed images to `__crop_queued/` directory
- Track your progress normally

**Speed improvement:** Instead of ~0.3-16s per batch (including I/O), you'll be limited only by how fast you can draw rectangles and hit Enter!

### Step 2: Process The Queue (Later)

When you're ready, process the queued crops with realistic human timing:

```bash
# Process with human-like timing (looks natural!)
python scripts/process_crop_queue.py

# Process faster (2x speed)
python scripts/process_crop_queue.py --speed 2.0

# Process without breaks
python scripts/process_crop_queue.py --no-breaks

# Process as fast as possible (no delays)
python scripts/process_crop_queue.py --fast

# Preview what will be processed
python scripts/process_crop_queue.py --preview

# Process only 10 batches
python scripts/process_crop_queue.py --limit 10
```

## Human Timing Simulation

The processor simulates your actual cropping patterns based on historical data:

### Timing Characteristics

- **Base speed:** 0.31s median between batches (your typical speed)
- **Variability:** Some batches instant (0.11s), some with "thinking pauses" (16-25s)
- **Warm-up:** Slower at session start (~100% baseline speed)
- **Peak performance:** 124-130% speed in middle of session
- **Slight fatigue:** 115% speed toward end of session
- **Random jitter:** ±6s to look natural

### Break Patterns

Based on your historical work patterns:

- **Short breaks (65%):** 10-30 minutes
- **Medium breaks (20%):** 30-120 minutes
- **Long breaks (15%):** 2-8 hours

Breaks occur after approximately 200 crops (with ±20% randomness).

## Your Statistics

Based on analysis of 7,194 historical crops:

```
Session lengths:     32 min median, 44 min average
Crops per session:   117 median, 206 average
Crops per hour:      259 median, 200-400 sustained
Time between crops:  0.31s median (batch submit time)
Break duration:      20.5 min median
```

## Project Timeline (3,367 images)

### Normal Mode

- Pure cropping time: **13.0 hours**
- With realistic breaks: **17.3 hours**
- At 6 hrs/day: **2.9 days**

### Queue Mode (50% faster UI)

- Pure cropping time: **8.7 hours** (save 4.3 hours!)
- Interactive work: Can complete in **~2 days** at 6 hrs/day

## Queue File Management

### Check Queue Status

```bash
# View queue statistics
python -c "
from scripts.utils.crop_queue import CropQueueManager
from pathlib import Path
qm = CropQueueManager(Path('data/crop_queue/crop_queue.jsonl'))
print(qm.get_queue_stats())
"
```

### Clear Completed Batches

```bash
python -c "
from scripts.utils.crop_queue import CropQueueManager
from pathlib import Path
qm = CropQueueManager(Path('data/crop_queue/crop_queue.jsonl'))
removed = qm.clear_completed()
print(f'Removed {removed} completed batches')
"
```

## Files & Directories

```
data/crop_queue/
  ├── crop_queue.jsonl           # Main queue file (JSONL format)
  ├── timing_log.csv             # Processing timing data
  ├── timing_patterns.json       # Your historical timing patterns
  └── QUEUE_FORMAT.md            # Technical format documentation

__crop_queued/                   # Staged images waiting to be processed
__cropped/                       # Final cropped output (processed)
```

## Tips

1. **Queue everything in one session** - work as fast as you want, process later
2. **Use preview mode first** to verify queue contents before processing
3. **Speed multiplier** lets you compress work time (e.g., 2.0x = half the time)
4. **Keep breaks enabled** for most realistic timing (unless you're in a hurry)
5. **Check timing_log.csv** to analyze actual processing performance

## Troubleshooting

**Queue not processing:**

- Check that images exist in `__crop_queued/` directory
- Verify queue file at `data/crop_queue/crop_queue.jsonl`
- Try preview mode first: `--preview`

**Want to start over:**

```bash
rm data/crop_queue/crop_queue.jsonl
# Move files back from __crop_queued/ to original location if needed
```

## Advanced: Custom Timing Patterns

You can edit `data/crop_queue/timing_patterns.json` to customize timing simulation:

```json
{
  "percentiles": {
    "p50": 0.31, // Median time between batches
    "p75": 16.25 // 75th percentile (thinking pauses)
  },
  "mean": 7.86,
  "stddev": 12.12, // Variability
  "session_avg_minutes": 44.0,
  "session_avg_crops": 199.8,
  "break_median_minutes": 20.5
}
```

---

**Have fun cropping faster than ever! 🚀**
