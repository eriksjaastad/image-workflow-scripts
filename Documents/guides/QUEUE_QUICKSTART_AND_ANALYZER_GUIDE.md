# Queue System Quickstart & Analyzer Usage

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers, Operators
**Estimated Reading Time:** 10 minutes

## 5-Minute Quickstart

### 1. Queue Crops (Fast Interactive Mode)

```bash
python scripts/02_ai_desktop_multi_crop.py crop_auto/ --queue-mode
```

- Set crop coordinates as fast as you can think
- No waiting for I/O operations
- Files moved to `__crop_queued/` for processing later

### 2. Preview What's Queued

```bash
python scripts/process_crop_queue.py --preview
```

### 3. Process the Queue

```bash
# With realistic human timing (default)
python scripts/process_crop_queue.py

# As fast as possible
python scripts/process_crop_queue.py --fast

# Custom speed (2x = half the time)
python scripts/process_crop_queue.py --speed 2.0
```

Done! See `QUEUE_MODE_GUIDE.md` for full usage documentation.

---

## Analyzer Tools

The queue system includes analyzer tools to extract your actual work patterns from historical crop data and use them for realistic timing simulation.

### Overview

Two analyzer tools work together:

1. **`analyze_human_patterns.py`** - Extracts timing statistics from historical crop sessions
2. **`analyze_crop_timing.py`** - Analyzes session characteristics and work patterns

Both read from the FileTracker logs to understand your real cropping behavior.

### Running the Analyzers

#### Extract Timing Patterns

```bash
python scripts/tools/analyze_human_patterns.py
```

**What it does:**
- Scans FileTracker logs for crop operations
- Groups crops into sessions (breaks >5min apart)
- Calculates timing statistics (median, percentiles, variability)
- Identifies break patterns
- Writes results to `data/ai_data/crop_queue/timing_patterns.json`

**Output:** JSON file with your timing characteristics

#### Analyze Session Patterns

```bash
python scripts/tools/analyze_crop_timing.py
```

**What it does:**
- Analyzes how your speed changes throughout sessions
- Identifies warm-up, peak performance, and fatigue patterns
- Calculates crops-per-hour rates
- Shows session length distribution
- Writes detailed analysis to `data/ai_data/crop_queue/timing_analysis.json`

**Output:** Detailed session analysis with speed curves

### Understanding the Output

#### timing_patterns.json

```json
{
  "percentiles": {
    "p01": 0.11,       // Very fast batches
    "p50": 0.31,       // Median (your typical speed)
    "p75": 16.25,      // "Thinking pauses"
    "p99": 25.13       // Longest pauses
  },
  "mean": 7.86,
  "stddev": 12.12,     // Timing variability
  "session_avg_minutes": 44.0,
  "session_avg_crops": 199.8,
  "break_median_minutes": 20.5,
  "break_patterns": {
    "short": 0.65,     // 65% of breaks are short (<30m)
    "medium": 0.20,    // 20% are medium (30m-2h)
    "long": 0.15       // 15% are long (2h+)
  }
}
```

**Key metrics:**
- **p50 (median)**: Your normal speed between batches
- **p75**: Includes your "thinking pauses" when deciding on crop
- **stddev**: How variable your timing is (higher = more variation)
- **session_avg_crops**: Typical session length before break

#### timing_analysis.json

```json
{
  "session_quarters": {
    "q1_speed": 306,   // Crops/hr in first quarter (warm-up)
    "q2_speed": 380,   // Second quarter (getting into flow)
    "q3_speed": 396,   // Third quarter (peak performance)
    "q4_speed": 350    // Fourth quarter (slight fatigue)
  },
  "total_sessions": 36,
  "total_crops": 7194,
  "median_session_minutes": 32,
  "sustained_rate_low": 200,
  "sustained_rate_high": 400
}
```

**Key insights:**
- **Session quarters**: Shows how speed changes throughout a session
- **Sustained rates**: Your realistic long-term speeds

### Using Analyzer Output

The processor (`scripts/process_crop_queue.py`) automatically reads `timing_patterns.json` and simulates your work patterns:

1. **Base speed**: Uses p50 (median) as baseline
2. **Variability**: Adds random delays based on p75, p99 for natural "thinking pauses"
3. **Session progression**: Simulates warm-up → peak → fatigue
4. **Breaks**: Inserts realistic break periods based on break_patterns

### Customizing Timing Patterns

You can manually edit `timing_patterns.json` to adjust simulation:

```bash
# Example: Make processing faster across the board
# Edit timing_patterns.json and reduce all percentile values by 50%
{
  "percentiles": {
    "p50": 0.15,   // Was 0.31, now 2x faster
    "p75": 8.12    // Was 16.25, now 2x faster
  }
}
```

Or use the speed multiplier when processing:

```bash
# 2x speed without editing files
python scripts/process_crop_queue.py --speed 2.0
```

### When to Re-Run Analyzers

Run the analyzers again when:

1. **After major workflow changes** - If you change how you crop (new UI, different tools)
2. **After significant practice** - Your speed might improve over time
3. **Different project types** - Complex crops vs simple crops have different timing
4. **Want updated statistics** - Periodically refresh to stay current

### Advanced: Filtering Analyzer Input

Both analyzers support optional filtering:

```bash
# Only analyze recent data (last 30 days)
python scripts/tools/analyze_human_patterns.py --days 30

# Only analyze specific project
python scripts/tools/analyze_human_patterns.py --project mojo3

# Output to custom location
python scripts/tools/analyze_human_patterns.py --output custom_timing.json
```

See `python scripts/tools/analyze_human_patterns.py --help` for all options.

### Troubleshooting Analyzers

**Problem: "No crop operations found"**
- Check that FileTracker logs exist in `data/file_operations_logs/`
- Verify logs contain crop operations (operation_type='crop')
- Try expanding date range: `--days 90`

**Problem: "Not enough data for reliable analysis"**
- Need at least ~100 crops for basic analysis
- Ideal: 1000+ crops for accurate patterns
- Consider using default patterns if insufficient data

**Problem: Timing seems unrealistic**
- Check for outliers in source data (very long pauses)
- Review FileTracker logs for data quality issues
- Consider manually editing timing_patterns.json to remove outliers

### Analyzer Output Locations

```
data/ai_data/crop_queue/
├── timing_patterns.json       # Used by processor (required)
├── timing_analysis.json       # Detailed analysis (optional)
└── timing_log.csv            # Processing run history
```

### Integration with Queue Processor

The processor automatically:

1. Loads `timing_patterns.json` on startup
2. Uses patterns to simulate realistic timing
3. Logs actual processing times to `timing_log.csv`
4. Allows real-time comparison of simulated vs actual timing

You can verify simulation accuracy by comparing:
- **Source**: `timing_patterns.json` (your historical patterns)
- **Actual**: `timing_log.csv` (processor execution log)

---

## Quick Reference Commands

```bash
# Analyze timing patterns
python scripts/tools/analyze_human_patterns.py

# Analyze session characteristics
python scripts/tools/analyze_crop_timing.py

# Queue crops
python scripts/02_ai_desktop_multi_crop.py crop_auto/ --queue-mode

# Preview queue
python scripts/process_crop_queue.py --preview

# Process with realistic timing
python scripts/process_crop_queue.py

# Process fast
python scripts/process_crop_queue.py --fast

# Process with custom speed
python scripts/process_crop_queue.py --speed 2.0
```

---

## See Also

- `QUEUE_MODE_GUIDE.md` - Complete queue mode usage documentation
- `CROP_QUEUE_SUMMARY.md` - Implementation summary and technical details
- `.cursorrules` - Commit communication standards for collaboration
- `scripts/tools/audit_files_vs_db.py` - Queue consistency auditing
