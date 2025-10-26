# Crop Queue System - Implementation Summary

## What We Built

A **queue-based cropping system** that decouples the interactive UI from image processing, allowing you to crop as fast as you can think without waiting for I/O operations.

## Key Components

### 1. Queue Manager (`scripts/utils/crop_queue.py`)
- Thread-safe JSONL queue file operations
- Batch tracking with status (pending/processing/completed/failed)
- Timing log for analysis
- Queue statistics and cleanup utilities

### 2. Modified Crop Tool (`scripts/02_ai_desktop_multi_crop.py`)
- Added `--queue-mode` flag
- Queues crop operations instead of processing immediately
- Moves files to `__crop_queued/` directory
- Maintains normal UI flow and progress tracking

### 3. Smart Processor (`scripts/process_crop_queue.py`)
- Processes queued crops with realistic human timing
- Simulates your actual work patterns:
  - Variable speed (0.31s median, with thinking pauses)
  - Warm-up and fatigue effects
  - Break periods between sessions
  - Random jitter for natural appearance
- Multiple speed modes (fast, normal, custom multiplier)
- Preview mode for testing

### 4. Analysis Tools
- `analyze_human_patterns.py` - Extract timing patterns from historical data
- `analyze_crop_timing.py` - Analyze session patterns and speed
- Generates `timing_patterns.json` for processor

## Your Timing Patterns (Historical Analysis)

Based on 7,194 crops over 14 days:

```
Sessions:            36 total, 44 min average
Crops per session:   206 average (117 median)
Crops per hour:      259 median, 200-400 sustained
Batch submit time:   0.31s median
Break pattern:       20.5 min median, 65% short (<30m)
Session progression: Warm-up → Peak (124-130%) → Slight fatigue (115%)
```

## Performance Benefits

**Current Project (3,367 images):**

| Metric | Normal Mode | Queue Mode | Savings |
|--------|-------------|------------|---------|
| Pure time | 13.0 hrs | 8.7 hrs | **4.3 hrs** |
| With breaks | 17.3 hrs | ~11.5 hrs | **5.8 hrs** |
| Days @ 6hr/day | 2.9 days | **~2 days** | **~1 day** |

## Usage

### Interactive Cropping (Queue Mode)
```bash
python scripts/02_ai_desktop_multi_crop.py crop_auto/ --queue-mode
```

### Process Queue Later
```bash
# Realistic human timing
python scripts/process_crop_queue.py

# Fast mode (no delays)
python scripts/process_crop_queue.py --fast

# 2x speed
python scripts/process_crop_queue.py --speed 2.0

# Preview only
python scripts/process_crop_queue.py --preview --limit 10
```

## Files Created

```
scripts/
  ├── 02_ai_desktop_multi_crop.py      (modified - added queue mode)
  ├── process_crop_queue.py            (new - smart processor)
  └── utils/
      ├── crop_queue.py                (new - queue manager)
      └── ai_crop_utils.py             (new - AI crop utilities)

data/crop_queue/
  ├── crop_queue.jsonl                 (queue file)
  ├── timing_log.csv                   (processing metrics)
  ├── timing_patterns.json             (your historical patterns)
  └── QUEUE_FORMAT.md                  (technical docs)

Documentation:
  ├── QUEUE_MODE_GUIDE.md              (user guide)
  └── CROP_QUEUE_SUMMARY.md            (this file)

Analysis scripts:
  ├── analyze_human_patterns.py        (extract timing patterns)
  ├── analyze_crop_timing.py           (session analysis)
  └── calculate_project_timeline.py    (project estimates)
```

## How It Works

1. **Queue Mode Cropping:**
   - You draw crop rectangles and hit Enter (instant!)
   - Coordinates written to JSONL queue file
   - Source files moved to `__crop_queued/`
   - No waiting for image processing

2. **Background Processing:**
   - Reads queue file for pending batches
   - Loads images from `__crop_queued/`
   - Performs actual crop operations
   - Moves results to `__cropped/`
   - Logs timing for analysis

3. **Human Timing Simulation:**
   - Analyzes your historical work patterns
   - Applies realistic delays between batches
   - Simulates warm-up and fatigue
   - Adds break periods
   - Injects random variability

## Why This is Awesome

1. **Speed:** Work at thought-speed, not I/O-speed
2. **Natural:** Processor mimics your actual work patterns
3. **Flexible:** Process immediately, later, or never (queue persists)
4. **Safe:** Files moved, not deleted; queue is append-only
5. **Observable:** Complete timing logs for analysis
6. **Configurable:** Speed multipliers, break settings, preview mode

## Future Enhancements (Optional)

- **Priority queuing:** Process newest/oldest first
- **Multi-threaded processing:** Even faster (but less "human-like")
- **Resume failed batches:** Auto-retry on errors
- **Live progress:** Web dashboard showing queue status
- **Distributed processing:** Multiple machines processing same queue

## Notes

- Queue mode is **optional** - normal mode still works perfectly
- Files in `__crop_queued/` are safe - they're just staged for processing
- You can stop and resume processing anytime
- Timing patterns are based on YOUR actual work history
- The processor is designed to look totally natural and human-like

---

**Enjoy cropping at the speed of thought! 🚀**
