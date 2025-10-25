#!/usr/bin/env python3
"""
Backfill v3 Training Data from Historical Projects
===================================================
Reconstructs complete v3 decision data from archived originals + cropped images.

USAGE:
    python scripts/tools/backfill_v3_from_archives.py Aiko_raw [--dry-run]

WHAT IT DOES:
    1. Groups original images (using shared grouping logic)
    2. Finds which image was selected (matches cropped → original)
    3. Extracts crop coordinates (template matching with OpenCV)
    4. Populates v3 SQLite database with FULL data
    
GUARDRAILS:
    - Read-only on source images (never modifies originals/cropped)
    - Only writes to data/training/ai_training_decisions/{project_id}.db
    - Dry-run mode by default (use --execute to commit)
    - Detailed logging of all operations
    
OUTPUT:
    - data/training/ai_training_decisions/{project_id}.db (v3 format)
    - Complete with: images, user_selected_index, crop_coords
    - Ready for AI training!
"""

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.utils.companion_file_utils import (
    find_consecutive_stage_groups,
    sort_image_files_by_timestamp_and_stage,
    detect_stage,
    extract_timestamp_from_filename
)
from scripts.file_tracker import FileTracker


class HistoricalBackfillProcessor:
    """Process historical archives into v3 training data."""
    
    def __init__(
        self,
        project_name: str,
        base_dir: Path,
        dry_run: bool = True,
        raw_dir: Optional[Path] = None,
        final_dir: Optional[Path] = None,
        recursive: bool = False,
        min_group_size: int = 2,
        replace_db: bool = False,
        allow_singleton_fallback: bool = False,
        db_name: Optional[str] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.project_name = project_name
        self.base_dir = base_dir
        self.dry_run = dry_run
        self.recursive = recursive
        self.min_group_size = max(1, int(min_group_size))
        self.replace_db = replace_db
        self.allow_singleton_fallback = allow_singleton_fallback
        self.tracker = FileTracker("historical_backfill")
        self.manifest_path = manifest_path
        
        # Paths
        inferred_originals = base_dir / "training data" / project_name
        inferred_finals = base_dir / "training data" / f"{project_name}_final"
        self.originals_dir = Path(raw_dir).resolve() if raw_dir else inferred_originals
        self.cropped_dir = Path(final_dir).resolve() if final_dir else inferred_finals
        # DB naming can differ from directory naming; default: strip _raw suffix for DB name
        self.db_project_name = db_name if db_name else (project_name[:-4] if project_name.endswith("_raw") else project_name)
        self.db_path = base_dir / "data" / "training" / "ai_training_decisions" / f"{self.db_project_name}.db"
        self.timesheet_path = base_dir / "data" / "timesheet.csv"
        
        # Stats
        self.stats = {
            'total_originals': 0,
            'total_cropped': 0,
            'groups_found': 0,
            'groups_matched': 0,
            'groups_approved': 0,
            'crop_extraction_success': 0,
            'crop_extraction_failed': 0,
            'avg_crop_confidence': 0.0,
            'full_frame_normalized_to_approve': 0
        }
        
    def run(self):
        """Main processing pipeline."""
        print("=" * 80)
        print("HISTORICAL BACKFILL - V3 TRAINING DATA EXTRACTION")
        print("=" * 80)
        print(f"Project: {self.project_name}")
        print(f"Mode: {'DRY RUN (preview only)' if self.dry_run else 'EXECUTE (will write database)'}")
        print("=" * 80)
        print()
        
        # Step 1: Validate paths
        if not self._validate_paths():
            return False
            
        # Step 2: Load project metadata (manifest preferred, fallback to timesheet)
        if self.manifest_path and Path(self.manifest_path).exists():
            project_metadata = self._load_manifest_metadata(Path(self.manifest_path))
        else:
            project_metadata = self._load_project_metadata()
        if not project_metadata:
            print("[!] Could not load project metadata from timesheet")
            return False
        
        print(f"[*] Project metadata loaded:")
        print(f"    Start date: {project_metadata['start_date']}")
        print(f"    End date: {project_metadata['end_date']}")
        print(f"    Starting images: {project_metadata['starting_images']}")
        print(f"    Finished images: {project_metadata['finished_images']}")
        print()
        
        # Step 3: Load images
        print("[*] Loading original images...")
        original_images = self._load_images(self.originals_dir)
        self.stats['total_originals'] = len(original_images)
        print(f"    Found {len(original_images)} original PNG images")
        
        print("[*] Loading cropped images...")
        cropped_images = self._load_images(self.cropped_dir)
        self.stats['total_cropped'] = len(cropped_images)
        print(f"    Found {len(cropped_images)} cropped PNG images")
        print()
        
        # Verify counts match timesheet
        if len(original_images) != project_metadata['starting_images']:
            print(f"[!] WARNING: Original count ({len(original_images)}) doesn't match timesheet ({project_metadata['starting_images']})")
        if len(cropped_images) != project_metadata['finished_images']:
            print(f"[!] WARNING: Cropped count ({len(cropped_images)}) doesn't match timesheet ({project_metadata['finished_images']})")
        
        # Step 4: Group original images
        print("[*] Grouping original images using shared grouping logic...")
        groups = self._group_images(original_images)
        self.stats['groups_found'] = len(groups)
        print(f"    Found {len(groups)} groups (pairs/triplets)")
        print()

        # Optional singleton fallback if no groups found and allowed
        if self.allow_singleton_fallback and self.stats['groups_found'] == 0 and self.min_group_size > 1:
            print("[*] No groups found with current min_group_size; retrying with singleton groups (min_group_size=1)...")
            self.min_group_size = 1
            groups = self._group_images(original_images)
            self.stats['groups_found'] = len(groups)
            print(f"    Found {len(groups)} groups (including singletons)")
            print()
        
        # Step 5: Match cropped → original groups
        print("[*] Matching cropped images to original groups...")
        matched_groups = self._match_cropped_to_groups(groups, cropped_images)
        self.stats['groups_matched'] = len([g for g in matched_groups if g['matched_cropped']])
        self.stats['groups_approved'] = len([g for g in matched_groups if not g['matched_cropped']])
        print(f"    Matched: {self.stats['groups_matched']} groups")
        print(f"    Approved (no crop): {self.stats['groups_approved']} groups")
        print()
        
        # Step 6: Extract crop coordinates
        print("[*] Extracting crop coordinates via template matching...")
        decisions = self._extract_crop_coordinates(matched_groups, project_metadata)
        print(f"    Success: {self.stats['crop_extraction_success']} crops")
        print(f"    Failed: {self.stats['crop_extraction_failed']} crops")
        if self.stats['crop_extraction_success'] > 0:
            print(f"    Average confidence: {self.stats['avg_crop_confidence']:.3f}")
        print()
        
        # Step 7: Write to database
        if self.dry_run:
            print("[DRY RUN] Would write to database:")
            print(f"    Database: {self.db_path}")
            print(f"    Total decisions: {len(decisions)}")
            print(f"    Cropped: {self.stats['groups_matched']}")
            print(f"    Approved: {self.stats['groups_approved']}")
            print()
            print("[DRY RUN] No changes made. Use --execute to write database.")
        else:
            print(f"[*] Writing to database: {self.db_path}")
            success = self._write_to_database(decisions, project_metadata)
            if success:
                print(f"    ✓ Written {len(decisions)} decision records")
            else:
                print(f"    ✗ Database write failed")
                return False
        
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for key, value in self.stats.items():
            print(f"{key:30s}: {value}")
        print("=" * 80)
        
        return True
    
    def _validate_paths(self) -> bool:
        """Validate all required paths exist."""
        if not self.originals_dir.exists():
            print(f"[!] ERROR: Originals directory not found: {self.originals_dir}")
            return False
        if not self.cropped_dir.exists():
            print(f"[!] ERROR: Cropped directory not found: {self.cropped_dir}")
            return False
        if not self.timesheet_path.exists():
            print(f"[!] ERROR: Timesheet not found: {self.timesheet_path}")
            return False
        return True
    
    def _load_project_metadata(self) -> Optional[Dict]:
        """Load project metadata from timesheet."""
        try:
            with open(self.timesheet_path) as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Find project row (Column E = project name)
            project_row = None
            for i, row in enumerate(rows):
                if len(row) > 4 and row[4] == self.project_name:
                    project_row = i
                    break
            
            if project_row is None:
                print(f"[!] Project '{self.project_name}' not found in timesheet")
                return None
            
            row = rows[project_row]
            
            # Parse dates (handle multi-day projects)
            start_date = row[0]  # Column A
            end_date = start_date
            
            # Check if project spans multiple days
            for i in range(project_row + 1, len(rows)):
                next_row = rows[i]
                if len(next_row) > 4 and next_row[4]:  # Next project found
                    break
                if len(next_row) > 3 and next_row[3]:  # Has hours, same project continues
                    if next_row[0]:  # Has date
                        end_date = next_row[0]
            
            # Parse image counts
            starting_images = int(row[6]) if len(row) > 6 and row[6] else 0
            finished_images = int(row[7]) if len(row) > 7 and row[7] else 0
            
            return {
                'project_name': self.project_name,
                'start_date': start_date,
                'end_date': end_date,
                'starting_images': starting_images,
                'finished_images': finished_images
            }
        except Exception as e:
            print(f"[!] Error loading timesheet: {e}")
            return None
    
    def _load_images(self, directory: Path) -> List[Path]:
        """Load PNG images from directory (optionally recursive)."""
        if self.recursive:
            return sorted([f for f in directory.rglob("*.png")])
        return sorted([f for f in directory.glob("*.png")])

    def _load_manifest_metadata(self, manifest_path: Path) -> Optional[Dict]:
        """Load project metadata from a project manifest JSON file."""
        try:
            data = json.loads(manifest_path.read_text())
            # Use ISO timestamps from manifest; also provide counts if present
            started_at = data.get("startedAt") or data.get("createdAt") or ""
            finished_at = data.get("finishedAt") or started_at
            counts = data.get("counts", {})
            # Be tolerant of missing/null counts in manifests
            def _safe_int(v):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return 0
            starting_images = _safe_int(counts.get("initialImages"))
            finished_images = _safe_int(counts.get("finalImages"))
            return {
                'project_name': self.db_project_name,
                'start_date': started_at,
                'end_date': finished_at,
                'starting_images': starting_images,
                'finished_images': finished_images,
            }
        except Exception as e:
            print(f"[!] Error loading manifest: {e}")
            return None
    
    def _group_images(self, images: List[Path]) -> List[List[Path]]:
        """Group images using shared grouping logic."""
        # Sort images first (required by grouping logic)
        sorted_images = sort_image_files_by_timestamp_and_stage(images)
        
        # Group into pairs/triplets (or singletons if configured)
        groups = find_consecutive_stage_groups(sorted_images, min_group_size=self.min_group_size)
        
        return groups
    
    def _match_cropped_to_groups(self, groups: List[List[Path]], cropped_images: List[Path]) -> List[Dict]:
        """Match cropped images back to original groups."""
        matched_groups = []
        
        # Build lookup of cropped images by base timestamp
        cropped_lookup = {}
        for cropped in cropped_images:
            timestamp = extract_timestamp_from_filename(cropped.name)
            if timestamp:
                # timestamp is already a string like "20250829_123456"
                cropped_lookup[timestamp] = cropped
        
        # Match each group
        for group in groups:
            # Try to find matching cropped image
            matched_cropped = None
            selected_index = None
            
            for idx, orig in enumerate(group):
                timestamp = extract_timestamp_from_filename(orig.name)
                if timestamp:
                    # timestamp is already a string
                    if timestamp in cropped_lookup:
                        matched_cropped = cropped_lookup[timestamp]
                        selected_index = idx
                        break
            
            matched_groups.append({
                'originals': group,
                'matched_cropped': matched_cropped,
                'selected_index': selected_index
            })
        
        return matched_groups
    
    def _extract_crop_coordinates(self, matched_groups: List[Dict], metadata: Dict) -> List[Dict]:
        """Extract crop coordinates via template matching."""
        decisions = []
        confidence_sum = 0.0
        confidence_count = 0
        
        for i, group_data in enumerate(matched_groups):
            originals = group_data['originals']
            cropped = group_data['matched_cropped']
            selected_index = group_data['selected_index']
            
            # Generate group ID
            first_img = originals[0]
            timestamp_str = extract_timestamp_from_filename(first_img.name)
            if timestamp_str:
                group_id = f"{self.project_name}_legacy_{timestamp_str}"
            else:
                group_id = f"{self.project_name}_legacy_{i:04d}"
            
            decision = {
                'group_id': group_id,
                'timestamp': f"{metadata['start_date']}T12:00:00Z" if 'T' not in str(metadata['start_date']) else str(metadata['start_date']),
                'project_id': self.db_project_name,
                'images': [img.name for img in originals],
                'ai_selected_index': None,
                'ai_crop_coords': None,
                'ai_confidence': None,
                'user_selected_index': selected_index if selected_index is not None else 0,
                'user_action': 'crop' if cropped else 'approve',
                'final_crop_coords': None,
                'crop_timestamp': str(metadata['end_date']),
                'image_width': 0,
                'image_height': 0,
                'selection_match': None,
                'crop_match': None
            }
            
            # If cropped, extract coordinates
            if cropped and selected_index is not None:
                original_img = originals[selected_index]
                coords, confidence = self._find_crop_coordinates(original_img, cropped)
                
                if coords and confidence > 0.8:
                    # Detect full-frame crop (no actual cropping) and normalize to approve
                    x1, y1, x2, y2 = coords[:4]
                    def _is_full_frame(a: float, b: float, c: float, d: float, tol: float = 1e-6) -> bool:
                        return abs(a - 0.0) <= tol and abs(b - 0.0) <= tol and abs(c - 1.0) <= tol and abs(d - 1.0) <= tol

                    if _is_full_frame(x1, y1, x2, y2):
                        # Treat as approval, do not store crop coords
                        decision['user_action'] = 'approve'
                        decision['final_crop_coords'] = None
                        decision['crop_timestamp'] = None
                        # keep selected_index to indicate which was chosen
                        self.stats['full_frame_normalized_to_approve'] += 1
                    else:
                        decision['final_crop_coords'] = [x1, y1, x2, y2, coords[4], coords[5]]
                        decision['crop_timestamp'] = f"{metadata['end_date']}T12:00:00Z" if 'T' not in str(metadata['end_date']) else str(metadata['end_date'])
                        decision['image_width'] = coords[4]  # Store width
                        decision['image_height'] = coords[5]  # Store height
                        self.stats['crop_extraction_success'] += 1
                        confidence_sum += confidence
                        confidence_count += 1
                else:
                    self.stats['crop_extraction_failed'] += 1
                    print(f"    [!] Low confidence ({confidence:.2f}) for {original_img.name}")
            
            decisions.append(decision)
        
        if confidence_count > 0:
            self.stats['avg_crop_confidence'] = confidence_sum / confidence_count
        
        return decisions
    
    def _find_crop_coordinates(self, original_path: Path, cropped_path: Path) -> Tuple[Optional[List[float]], float]:
        """Find crop coordinates using template matching."""
        try:
            # Load images
            original = cv2.imread(str(original_path))
            cropped = cv2.imread(str(cropped_path))
            
            if original is None or cropped is None:
                return None, 0.0
            
            # Get dimensions
            orig_height, orig_width = original.shape[:2]
            crop_height, crop_width = cropped.shape[:2]
            
            # Convert to grayscale for matching
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(original_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Extract coordinates
            x1, y1 = max_loc
            x2, y2 = x1 + crop_width, y1 + crop_height
            
            # Normalize coordinates [0, 1]
            x1_norm = x1 / orig_width
            y1_norm = y1 / orig_height
            x2_norm = x2 / orig_width
            y2_norm = y2 / orig_height
            
            # Return coords + dimensions, confidence
            coords = [x1_norm, y1_norm, x2_norm, y2_norm, orig_width, orig_height]
            confidence = max_val
            
            return coords, confidence
            
        except Exception as e:
            print(f"    [!] Error extracting crop for {original_path.name}: {e}")
            return None, 0.0
    
    def _write_to_database(self, decisions: List[Dict], metadata: Dict) -> bool:
        """Write decisions to v3 SQLite database."""
        try:
            # Create database directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Safeguard: do not overwrite existing DB unless explicitly allowed
            if self.db_path.exists():
                if not self.replace_db:
                    print(f"[!] Database already exists: {self.db_path}")
                    print("[!] Refusing to overwrite without --replace-db")
                    return False
                # Move existing DB to macOS Trash and log
                trash_path = Path.home() / ".Trash" / f"{self.db_path.stem}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}{self.db_path.suffix}"
                try:
                    shutil.move(str(self.db_path), str(trash_path))
                    self.tracker.log_operation(
                        "delete",
                        source_dir=str(self.db_path.parent),
                        dest_dir=str(trash_path.parent),
                        file_count=1,
                        files=[self.db_path.name],
                        notes="Move existing DB to Trash before replace"
                    )
                    print(f"[*] Moved existing DB to Trash: {trash_path}")
                except Exception as e:
                    print(f"[!] Failed moving existing DB to Trash: {e}")
                    return False
            
            # Create database with v3 schema
            conn = sqlite3.connect(str(self.db_path))
            
            # Create table (same schema as current v3)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    group_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    directory TEXT,
                    batch_number INTEGER,
                    images TEXT NOT NULL,
                    ai_selected_index INTEGER,
                    ai_crop_coords TEXT,
                    ai_confidence REAL,
                    user_selected_index INTEGER NOT NULL,
                    user_action TEXT NOT NULL,
                    final_crop_coords TEXT,
                    crop_timestamp TEXT,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    selection_match BOOLEAN,
                    crop_match BOOLEAN,
                    CHECK(user_action IN ('approve', 'crop', 'reject')),
                    CHECK(user_selected_index >= 0 AND user_selected_index <= 3),
                    CHECK(image_width > 0 AND image_height > 0)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_project_id ON ai_decisions(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON ai_decisions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_action ON ai_decisions(user_action)")
            
            # Insert decisions
            for decision in decisions:
                # Extract width/height from coords if available
                width = decision['image_width']
                height = decision['image_height']
                
                # If not extracted, load from first image
                if width == 0 or height == 0:
                    try:
                        img_path = self.originals_dir / decision['images'][decision['user_selected_index']]
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except:
                        width, height = 3072, 3072  # Default
                
                # Prepare crop coords (remove width/height if they were stored there)
                crop_coords_json = None
                if decision['final_crop_coords']:
                    coords = decision['final_crop_coords'][:4]  # Only x1,y1,x2,y2
                    crop_coords_json = json.dumps(coords)
                
                conn.execute("""
                    INSERT INTO ai_decisions 
                    (group_id, timestamp, project_id, directory, batch_number,
                     images, ai_selected_index, ai_crop_coords, ai_confidence,
                     user_selected_index, user_action, final_crop_coords, crop_timestamp,
                     image_width, image_height, selection_match, crop_match)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision['group_id'],
                    decision['timestamp'],
                    decision['project_id'],
                    None,  # directory
                    None,  # batch_number
                    json.dumps(decision['images']),
                    decision['ai_selected_index'],
                    decision['ai_crop_coords'],
                    decision['ai_confidence'],
                    decision['user_selected_index'],
                    decision['user_action'],
                    crop_coords_json,
                    decision['crop_timestamp'],
                    width,
                    height,
                    decision['selection_match'],
                    decision['crop_match']
                ))
            
            conn.commit()
            conn.close()

            # Log DB creation
            try:
                self.tracker.log_operation(
                    "create",
                    source_dir="",
                    dest_dir=str(self.db_path.parent),
                    file_count=1,
                    files=[self.db_path.name],
                    notes=f"Created decisions DB for {self.project_name}"
                )
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            print(f"[!] Database write error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Backfill v3 training data from historical archives")
    parser.add_argument("project_name", help="Project name (used for defaults and DB name)")
    parser.add_argument("--execute", action="store_true", help="Execute (default is dry-run)")
    parser.add_argument("--base-dir", default=".", help="Base directory (default: current)")
    parser.add_argument("--raw-dir", default=None, help="Explicit RAW/originals directory")
    parser.add_argument("--final-dir", default=None, help="Explicit FINAL/cropped directory (suffix _final is common)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan RAW/FINAL subdirectories for images")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum group size (use 1 to allow singletons)")
    parser.add_argument("--allow-singleton-fallback", action="store_true", help="If no groups found, retry with min-group-size=1")
    parser.add_argument("--replace-db", action="store_true", help="Move existing DB to Trash, then write a new one")
    parser.add_argument("--db-name", default=None, help="Override DB project name (default strips _raw suffix)")
    parser.add_argument("--manifest", default=None, help="Path to project manifest JSON (preferred source of truth)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    dry_run = not args.execute
    
    processor = HistoricalBackfillProcessor(
        args.project_name,
        base_dir,
        dry_run=dry_run,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        final_dir=Path(args.final_dir) if args.final_dir else None,
        recursive=args.recursive,
        min_group_size=args.min_group_size,
        replace_db=args.replace_db,
        allow_singleton_fallback=args.allow_singleton_fallback,
        db_name=args.db_name,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    success = processor.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

