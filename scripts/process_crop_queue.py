#!/usr/bin/env python3
"""
Crop Queue Processor - Process queued crops with realistic human timing patterns.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Process the crop queue with human-like timing:
  python scripts/process_crop_queue.py

Options:
  --fast              # Process as fast as possible (no delays)
  --speed MULTIPLIER  # Speed multiplier (e.g., 2.0 = 2x faster, 0.5 = half speed)
  --no-breaks         # Skip break periods between sessions
  --preview           # Show what would be processed without actually doing it
  --limit N           # Process only N batches then stop

The processor simulates realistic human timing patterns based on your historical data:
- Variable time between batches (0.31s median, but with realistic pauses)
- Warm-up period at session start (slower initially)
- Slight fatigue toward end of session (minor slowdown)
- Break periods between sessions (median 20.5 minutes)
- Random jitter to look natural

Timing data is loaded from: data/crop_queue/timing_patterns.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.crop_queue import CropQueueManager
from scripts.utils.ai_crop_utils import headless_crop
from scripts.file_tracker import FileTracker


class HumanTimingSimulator:
    """Simulates realistic human cropping timing patterns."""

    def __init__(self, timing_data: Dict, speed_multiplier: float = 1.0, enable_breaks: bool = True):
        self.timing_data = timing_data
        self.speed_multiplier = speed_multiplier
        self.enable_breaks = enable_breaks

        # Extract key timing parameters
        self.median_batch_time = timing_data.get('percentiles', {}).get('p50', 0.31)
        self.p25_batch_time = timing_data.get('percentiles', {}).get('p25', 0.11)
        self.p75_batch_time = timing_data.get('percentiles', {}).get('p75', 16.25)
        self.stddev = timing_data.get('stddev', 12.12)
        self.session_avg_crops = timing_data.get('session_avg_crops', 200)
        self.break_median_minutes = timing_data.get('break_median_minutes', 20.5)

        # Session state
        self.crops_in_session = 0
        self.session_start_time = None

    def start_session(self):
        """Mark the start of a new session."""
        self.session_start_time = time.time()
        self.crops_in_session = 0

    def should_take_break(self) -> bool:
        """Determine if we should take a break based on session length."""
        if not self.enable_breaks:
            return False

        # Take a break after approximately session_avg_crops
        # Add some randomness (±20%)
        crops_threshold = self.session_avg_crops * random.uniform(0.8, 1.2)
        return self.crops_in_session >= crops_threshold

    def get_break_duration(self) -> float:
        """Get break duration in seconds with realistic variation."""
        # Short breaks are more common, but occasionally longer breaks
        roll = random.random()

        if roll < 0.65:  # 65% short breaks (< 30 min)
            duration_min = random.uniform(10, 30)
        elif roll < 0.85:  # 20% medium breaks (30-120 min)
            duration_min = random.uniform(30, 120)
        else:  # 15% long breaks (> 120 min)
            duration_min = random.uniform(120, 480)  # Up to 8 hours

        return duration_min * 60 / self.speed_multiplier

    def get_next_batch_delay(self) -> float:
        """Get delay before processing next batch, simulating human timing."""
        # Determine session quarter for fatigue/warmup effects
        progress = min(1.0, self.crops_in_session / self.session_avg_crops)

        # Warmup and fatigue factors based on historical data:
        # Quarter 1: 100% speed, Quarter 2-3: 124-130% speed, Quarter 4: 115% speed
        if progress < 0.25:
            speed_factor = 1.0  # Warmup period
        elif progress < 0.75:
            speed_factor = 1.27  # Peak performance
        else:
            speed_factor = 1.15  # Slight fatigue

        # Most batches are fast (median 0.31s), but occasionally pause to "think"
        roll = random.random()

        if roll < 0.5:  # 50% very fast (< median)
            base_delay = random.uniform(self.p25_batch_time, self.median_batch_time)
        elif roll < 0.85:  # 35% normal (median to 75th percentile)
            base_delay = random.uniform(self.median_batch_time, self.p75_batch_time)
        else:  # 15% thinking pause (75th percentile+)
            base_delay = random.uniform(self.p75_batch_time, self.p75_batch_time * 2)

        # Apply speed factor (faster during peak, slower during warmup/fatigue)
        base_delay = base_delay / speed_factor

        # Add small random jitter (±stddev/4)
        jitter = random.uniform(-self.stddev / 4, self.stddev / 4)
        final_delay = max(0.05, base_delay + jitter)  # Never less than 50ms

        # Apply speed multiplier
        final_delay = final_delay / self.speed_multiplier

        return final_delay


class CropQueueProcessor:
    """Processes the crop queue with realistic timing simulation."""

    def __init__(
        self,
        queue_manager: CropQueueManager,
        timing_simulator: Optional[HumanTimingSimulator] = None,
        preview_mode: bool = False
    ):
        self.queue_manager = queue_manager
        self.timing_simulator = timing_simulator
        self.preview_mode = preview_mode
        self.tracker = FileTracker("crop_queue_processor")

    def process_batch(self, batch: Dict) -> bool:
        """
        Process a single batch of crops.

        Returns:
            True if successful, False if failed
        """
        batch_id = batch['batch_id']
        crops = batch['crops']

        if self.preview_mode:
            print(f"[Preview] Would process batch {batch_id} with {len(crops)} crops")
            return True

        # Update status to processing
        self.queue_manager.update_batch_status(batch_id, 'processing')

        start_time = time.time()
        timestamp_started = datetime.now(timezone.utc).isoformat()

        try:
            # Process each crop in the batch using the trusted crop path
            for crop in crops:
                source_path = Path(crop['source_path'])
                crop_rect = tuple(crop['crop_rect'])  # Convert list to tuple
                dest_directory = Path(crop['dest_directory'])

                # Use headless_crop (trusted path - same code as desktop tool)
                moved_files = headless_crop(source_path, crop_rect, dest_directory)

                # Log crop operation
                self.tracker.log_operation(
                    "crop",
                    source_dir=str(source_path.parent),
                    dest_dir=str(dest_directory),
                    file_count=len(moved_files),
                    notes=f"batch={batch_id}, crop={crop_rect}"
                )

            # Calculate processing time
            end_time = time.time()
            processing_time_ms = int((end_time - start_time) * 1000)
            timestamp_completed = datetime.now(timezone.utc).isoformat()

            # Update status to completed
            self.queue_manager.update_batch_status(batch_id, 'completed', processing_time_ms)

            # Log timing
            self.queue_manager.log_timing(
                batch_id=batch_id,
                timestamp_queued=batch['timestamp_queued'],
                timestamp_started=timestamp_started,
                timestamp_completed=timestamp_completed,
                crops_in_batch=len(crops),
                total_time_ms=processing_time_ms,
                project_id=batch['project_id'],
                session_id=batch['session_id']
            )

            print(f"✓ Processed batch {batch_id} ({len(crops)} crops, {processing_time_ms}ms)")
            return True

        except Exception as e:
            print(f"✗ Failed to process batch {batch_id}: {e}")
            self.queue_manager.update_batch_status(batch_id, 'failed', error=str(e))
            return False

    def preflight_validation(self, batches: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Perform preflight validation on all batches.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        warnings = []

        print(f"\n{'='*80}")
        print(f"PREFLIGHT VALIDATION")
        print(f"{'='*80}\n")

        # Define allowed safe zones (strict whitelist)
        allowed_safe_zones = {
            '__cropped',
            '__crop_queued',
            'data/ai_data',
            '__final',
            '__temp'
        }

        for batch in batches:
            batch_id = batch['batch_id']
            crops = batch['crops']

            for crop in crops:
                source_path = Path(crop['source_path'])
                dest_directory = Path(crop['dest_directory'])
                crop_rect = crop.get('crop_rect', [])
                crop_rect_normalized = crop.get('crop_rect_normalized', [])

                # Check source file exists
                if not source_path.exists():
                    errors.append(f"[{batch_id}] Source file missing: {source_path}")
                    continue

                # Check source is readable
                if not source_path.is_file():
                    errors.append(f"[{batch_id}] Source is not a file: {source_path}")
                    continue

                # Validate crop coordinates (pixel)
                if len(crop_rect) != 4:
                    errors.append(f"[{batch_id}] Invalid crop coordinates: {crop_rect}")
                    continue

                x1, y1, x2, y2 = crop_rect
                if x1 >= x2 or y1 >= y2:
                    errors.append(f"[{batch_id}] Invalid crop dimensions: {crop_rect}")
                    continue

                if x1 < 0 or y1 < 0:
                    errors.append(f"[{batch_id}] Negative crop coordinates: {crop_rect}")
                    continue

                # Validate normalized coordinates (DB consistency check)
                if crop_rect_normalized:
                    if len(crop_rect_normalized) != 4:
                        errors.append(f"[{batch_id}] Invalid normalized coordinates: {crop_rect_normalized}")
                        continue

                    nx1, ny1, nx2, ny2 = crop_rect_normalized
                    # All values must be in [0, 1] range
                    if not all(0 <= v <= 1 for v in [nx1, ny1, nx2, ny2]):
                        errors.append(f"[{batch_id}] Normalized coords out of range [0,1]: {crop_rect_normalized}")
                        continue

                    # nx1 < nx2 and ny1 < ny2
                    if nx1 >= nx2 or ny1 >= ny2:
                        errors.append(f"[{batch_id}] Invalid normalized dimensions: {crop_rect_normalized}")
                        continue

                # Validate crop coordinates are within image bounds
                try:
                    from PIL import Image
                    with Image.open(source_path) as img:
                        img_width, img_height = img.size

                        if x2 > img_width or y2 > img_height:
                            errors.append(
                                f"[{batch_id}] Crop {crop_rect} exceeds image dimensions "
                                f"({img_width}x{img_height}) for {source_path.name}"
                            )
                except Exception as e:
                    errors.append(f"[{batch_id}] Cannot read image {source_path}: {e}")
                    continue

                # Check destination directory
                if not dest_directory.exists():
                    warnings.append(f"[{batch_id}] Destination will be created: {dest_directory}")
                elif not dest_directory.is_dir():
                    errors.append(f"[{batch_id}] Destination exists but is not a directory: {dest_directory}")

                # Check for potential overwrites
                dest_file = dest_directory / source_path.name
                if dest_file.exists():
                    warnings.append(f"[{batch_id}] Will overwrite existing file: {dest_file}")

                # Strict safe-zone validation (must match allowed zones)
                dest_str = str(dest_directory)
                is_safe = any(safe_zone in dest_str for safe_zone in allowed_safe_zones)
                if not is_safe:
                    errors.append(
                        f"[{batch_id}] Destination outside allowed safe zones: {dest_directory}\n"
                        f"              Allowed zones: {', '.join(allowed_safe_zones)}"
                    )

        # Print summary
        print(f"Validated {sum(len(b['crops']) for b in batches)} crop operations across {len(batches)} batches\n")

        if warnings:
            print(f"⚠️  {len(warnings)} warnings:")
            for warning in warnings[:10]:  # Show first 10
                print(f"   {warning}")
            if len(warnings) > 10:
                print(f"   ... and {len(warnings) - 10} more warnings")
            print()

        if errors:
            print(f"❌ {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10
                print(f"   {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
            print()
            return False, errors

        print("✅ Validation passed!\n")
        return True, []

    def run(self, limit: Optional[int] = None, skip_confirmation: bool = False):
        """
        Process the queue with human-like timing.

        Args:
            limit: Maximum number of batches to process (None = all)
            skip_confirmation: If True, skip user confirmation prompt
        """
        stats = self.queue_manager.get_queue_stats()
        pending_count = stats['pending']

        if pending_count == 0:
            print("No pending batches in queue.")
            return

        print(f"\n{'='*80}")
        print(f"CROP QUEUE PROCESSOR")
        print(f"{'='*80}\n")
        print(f"Pending batches: {pending_count}")
        if limit:
            print(f"Limit: {limit} batches")
        if self.timing_simulator:
            print(f"Speed multiplier: {self.timing_simulator.speed_multiplier}x")
            print(f"Breaks enabled: {self.timing_simulator.enable_breaks}")
        print()

        # Get pending batches
        pending_batches = self.queue_manager.get_pending_batches(limit=limit)

        # ALWAYS run preflight validation (built-in safety)
        is_valid, validation_errors = self.preflight_validation(pending_batches)

        if not is_valid:
            print("❌ Preflight validation failed. Fix errors and try again.")
            return

        # Interactive confirmation (unless skipped with --yes)
        if not skip_confirmation and not self.preview_mode:
            total_crops = sum(len(b['crops']) for b in pending_batches)
            print(f"Ready to process {len(pending_batches)} batches ({total_crops} crops).")
            print("⚠️  This will modify files. Proceed? [y/N]: ", end='', flush=True)

            try:
                response = input().strip().lower()
                if response not in ('y', 'yes'):
                    print("\n❌ Cancelled by user.")
                    return
            except (KeyboardInterrupt, EOFError):
                print("\n\n❌ Cancelled by user.")
                return

            print()  # Blank line after confirmation

        if self.timing_simulator:
            self.timing_simulator.start_session()

        processed = 0

        for i, batch in enumerate(pending_batches):
            # Check if we should take a break
            if self.timing_simulator and self.timing_simulator.should_take_break():
                break_duration = self.timing_simulator.get_break_duration()
                break_minutes = break_duration / 60
                print(f"\n[Break] Taking a {break_minutes:.1f} minute break...")
                if not self.preview_mode:
                    time.sleep(break_duration)
                self.timing_simulator.start_session()

            # Apply human-like delay before processing
            if i > 0 and self.timing_simulator:
                delay = self.timing_simulator.get_next_batch_delay()
                time.sleep(delay)

            # Process the batch
            success = self.process_batch(batch)
            if success:
                processed += 1
                if self.timing_simulator:
                    self.timing_simulator.crops_in_session += len(batch['crops'])

            # Check if we've hit the limit
            if limit and processed >= limit:
                break

        print(f"\n{'='*80}")
        print(f"Processed {processed} batches")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Process crop queue with human-like timing")
    parser.add_argument("--fast", action="store_true", help="Process as fast as possible (no delays)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (default: 1.0)")
    parser.add_argument("--no-breaks", action="store_true", help="Skip break periods")
    parser.add_argument("--preview", action="store_true", help="Preview mode (don't actually process)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt (auto-approve)")
    parser.add_argument("--limit", type=int, help="Process only N batches")
    args = parser.parse_args()

    # Load queue manager (defaults to safe zone: data/ai_data/crop_queue/)
    queue_manager = CropQueueManager()

    # Load timing patterns (from safe zone)
    timing_simulator = None
    if not args.fast:
        timing_file = PROJECT_ROOT / "data" / "ai_data" / "crop_queue" / "timing_patterns.json"
        if timing_file.exists():
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
            timing_simulator = HumanTimingSimulator(
                timing_data,
                speed_multiplier=args.speed,
                enable_breaks=(not args.no_breaks)
            )
        else:
            print(f"Warning: Timing patterns file not found at {timing_file}")
            print("Run analyze_human_patterns.py first to generate timing data.")
            print("Proceeding without timing simulation...\n")

    # Create processor
    processor = CropQueueProcessor(
        queue_manager,
        timing_simulator=timing_simulator,
        preview_mode=args.preview
    )

    # Run (with built-in validation and confirmation)
    processor.run(limit=args.limit, skip_confirmation=args.yes)


if __name__ == '__main__':
    main()
