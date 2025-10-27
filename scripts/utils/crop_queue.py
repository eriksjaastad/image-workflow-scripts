#!/usr/bin/env python3
"""
Crop Queue Manager - handles queuing and status tracking for batch crop operations.
"""

import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from file_tracker import FileTracker


class CropQueueManager:
    """Manages the crop queue (JSONL file) with thread-safe operations."""

    def __init__(self, queue_file: Optional[Path] = None):
        # Default to safe zone if not specified
        if queue_file is None:
            safe_zone = Path(__file__).parent.parent.parent / "data" / "ai_data" / "crop_queue"
            queue_file = safe_zone / "crop_queue.jsonl"

        self.queue_file = Path(queue_file)
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

        # Create queue file if it doesn't exist
        if not self.queue_file.exists():
            self.queue_file.touch()

        # Timing log (in same safe zone directory)
        self.timing_log = self.queue_file.parent / "timing_log.csv"
        if not self.timing_log.exists():
            self.timing_log.write_text(
                "batch_id,timestamp_queued,timestamp_started,timestamp_completed,"
                "crops_in_batch,total_time_ms,avg_time_per_crop_ms,project_id,session_id\n"
            )

        # File tracker for logging operations
        self.tracker = FileTracker("crop_queue_manager")

    def _generate_batch_id(self, f_locked) -> str:
        """
        Generate unique batch ID with timestamp and sequence.

        Args:
            f_locked: File handle with exclusive lock already held

        Returns:
            Unique batch ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Find next sequence number for this timestamp
        # Lock is already held by caller
        seq = 1
        f_locked.seek(0)  # Reset to beginning
        for line in f_locked:
            if line.strip():
                try:
                    batch = json.loads(line)
                    if batch.get('batch_id', '').startswith(timestamp):
                        # Extract sequence number
                        parts = batch['batch_id'].split('_')
                        if len(parts) >= 3:
                            try:
                                existing_seq = int(parts[-1])
                                seq = max(seq, existing_seq + 1)
                            except ValueError:
                                pass
                except json.JSONDecodeError:
                    pass

        return f"{timestamp}_{seq:03d}"

    def enqueue_batch(
        self,
        crops: List[Dict],
        session_id: str,
        project_id: str
    ) -> str:
        """
        Add a batch of crops to the queue.

        Args:
            crops: List of crop operations, each with:
                   - source_path: str
                   - crop_rect: [x1, y1, x2, y2]
                   - dest_directory: str
                   - index_in_batch: int
            session_id: Crop session ID
            project_id: Project identifier

        Returns:
            batch_id: Unique identifier for this batch
        """
        # Open file with exclusive lock for ID generation AND append
        # This prevents race conditions in ID generation
        with open(self.queue_file, 'a+') as f:  # a+ allows reading too
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Generate unique ID while holding lock
                batch_id = self._generate_batch_id(f)
                timestamp = datetime.now(timezone.utc).isoformat()

                batch = {
                    "batch_id": batch_id,
                    "timestamp_queued": timestamp,
                    "session_id": session_id,
                    "project_id": project_id,
                    "crops": crops,
                    "status": "pending",
                    "timestamp_processed": None,
                    "processing_time_ms": None,
                    "error": None
                }

                # Append to queue file (still holding lock)
                f.seek(0, 2)  # Seek to end for append
                f.write(json.dumps(batch) + '\n')
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Log operation to FileTracker
        self.tracker.log_operation(
            "create",
            dest_dir=str(self.queue_file.parent),
            file_count=len(crops),
            notes=f"batch_id={batch_id}, project={project_id}"
        )

        return batch_id

    def get_pending_batches(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all pending batches from the queue.

        Args:
            limit: Maximum number of batches to return (None = all)

        Returns:
            List of batch dictionaries with status="pending"
        """
        pending = []

        with open(self.queue_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    if line.strip():
                        try:
                            batch = json.loads(line)
                            if batch.get('status') == 'pending':
                                pending.append(batch)
                                if limit and len(pending) >= limit:
                                    break
                        except json.JSONDecodeError:
                            pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return pending

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        processing_time_ms: Optional[int] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update the status of a batch in the queue.

        Args:
            batch_id: Batch identifier
            status: New status ("processing", "completed", "failed")
            processing_time_ms: Time taken to process (for completed/failed)
            error: Error message (for failed status)

        Returns:
            True if batch was found and updated, False otherwise
        """
        # Read all lines
        with open(self.queue_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                lines = f.readlines()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Update matching batch
        updated = False
        updated_lines = []

        for line in lines:
            if line.strip():
                try:
                    batch = json.loads(line)
                    if batch.get('batch_id') == batch_id:
                        batch['status'] = status
                        if status in ('completed', 'failed'):
                            batch['timestamp_processed'] = datetime.now(timezone.utc).isoformat()
                        if processing_time_ms is not None:
                            batch['processing_time_ms'] = processing_time_ms
                        if error is not None:
                            batch['error'] = error
                        updated = True
                    updated_lines.append(json.dumps(batch) + '\n')
                except json.JSONDecodeError:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Write back atomically
        if updated:
            temp_file = self.queue_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.writelines(updated_lines)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            temp_file.replace(self.queue_file)

        return updated

    def log_timing(
        self,
        batch_id: str,
        timestamp_queued: str,
        timestamp_started: str,
        timestamp_completed: str,
        crops_in_batch: int,
        total_time_ms: int,
        project_id: str,
        session_id: str
    ):
        """Log processing timing to CSV for analysis."""
        avg_time = total_time_ms / crops_in_batch if crops_in_batch > 0 else 0

        row = (
            f"{batch_id},{timestamp_queued},{timestamp_started},{timestamp_completed},"
            f"{crops_in_batch},{total_time_ms},{avg_time:.1f},{project_id},{session_id}\n"
        )

        with open(self.timing_log, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(row)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics (pending, processing, completed, failed counts)."""
        stats = {
            'pending': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'total': 0
        }

        with open(self.queue_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    if line.strip():
                        try:
                            batch = json.loads(line)
                            status = batch.get('status', 'unknown')
                            if status in stats:
                                stats[status] += 1
                            stats['total'] += 1
                        except json.JSONDecodeError:
                            pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return stats

    def clear_completed(self) -> int:
        """
        Remove completed batches from queue file.

        Returns:
            Number of batches removed
        """
        # Read all lines
        with open(self.queue_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                lines = f.readlines()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Keep only non-completed batches
        kept_lines = []
        removed_count = 0

        for line in lines:
            if line.strip():
                try:
                    batch = json.loads(line)
                    if batch.get('status') != 'completed':
                        kept_lines.append(line)
                    else:
                        removed_count += 1
                except json.JSONDecodeError:
                    kept_lines.append(line)
            else:
                kept_lines.append(line)

        # Write back
        temp_file = self.queue_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.writelines(kept_lines)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        temp_file.replace(self.queue_file)

        return removed_count
