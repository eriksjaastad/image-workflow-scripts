#!/usr/bin/env python3
"""
Project Bins Archiver
=====================
Archives finished project's 15-minute bins and merges into overall aggregate.

When a project finishes:
1. Snapshot project's bins into data/aggregates/archives/<project_id>/
2. Create project_summary.json with totals
3. Merge bins into overall/agg_15m_cumulative.jsonl (with dedupe)
4. Provide rollback capability

Archive structure:
  data/aggregates/archives/<project_id>/
    - agg_15m.jsonl          # Project's bins
    - project_summary.json   # Totals and metadata
    - archive_manifest.json  # Archive metadata

Overall aggregate:
  data/aggregates/overall/
    - agg_15m_cumulative.jsonl  # All finished projects combined

Features:
- Dry-run mode: Preview changes without writing
- Rollback: Remove project from overall aggregate
- Validation: Ensure no double-counting via dedupe_key
- Atomic operations: Temp files + rename
"""

import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class ProjectArchiver:
    """Handles archiving finished project bins."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.aggregates_dir = data_dir / 'aggregates'
        self.daily_dir = self.aggregates_dir / 'daily'
        self.archives_dir = self.aggregates_dir / 'archives'
        self.overall_dir = self.aggregates_dir / 'overall'
        
        # Ensure directories exist
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        self.overall_dir.mkdir(parents=True, exist_ok=True)
    
    def load_project_manifest(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load project manifest."""
        manifest_path = self.data_dir / 'projects' / f'{project_id}.project.json'
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading manifest: {e}")
            return None
    
    def collect_project_bins(self, project_id: str, started_at: str, finished_at: str) -> List[Dict[str, Any]]:
        """Collect all bins for a project across all days.
        
        Args:
            project_id: Project ID
            started_at: Project start timestamp (ISO)
            finished_at: Project finish timestamp (ISO)
            
        Returns:
            List of bin records for this project
        """
        from datetime import datetime, timedelta
        
        # Parse date range
        try:
            start_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00')).date()
            end_dt = datetime.fromisoformat(finished_at.replace('Z', '+00:00')).date()
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return []
        
        # Collect bins from each day in range
        project_bins = []
        current_date = start_dt
        
        while current_date <= end_dt:
            day_str = current_date.strftime('%Y%m%d')
            bin_path = self.daily_dir / f'day={day_str}' / 'agg_15m.jsonl'
            
            if bin_path.exists():
                try:
                    with open(bin_path, 'r') as f:
                        for line in f:
                            try:
                                bin_record = json.loads(line)
                                # Filter to this project
                                if bin_record.get('project_id') == project_id:
                                    project_bins.append(bin_record)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Warning: Error reading {bin_path}: {e}")
            
            current_date += timedelta(days=1)
        
        return project_bins
    
    def create_project_summary(
        self,
        project_id: str,
        manifest: Dict[str, Any],
        bins: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary statistics for archived project.
        
        Args:
            project_id: Project ID
            manifest: Project manifest
            bins: Project's bin records
            
        Returns:
            Summary dictionary
        """
        # Calculate totals
        total_file_count = sum(b.get('file_count', 0) for b in bins)
        total_event_count = sum(b.get('event_count', 0) for b in bins)
        total_work_seconds = sum(b.get('work_seconds', 0) for b in bins)
        
        # Group by script
        by_script = defaultdict(lambda: {
            'file_count': 0,
            'event_count': 0,
            'work_seconds': 0.0
        })
        
        for bin_record in bins:
            script = bin_record.get('script_id', 'unknown')
            by_script[script]['file_count'] += bin_record.get('file_count', 0)
            by_script[script]['event_count'] += bin_record.get('event_count', 0)
            by_script[script]['work_seconds'] += bin_record.get('work_seconds', 0)
        
        # Group by operation
        by_operation = defaultdict(lambda: {
            'file_count': 0,
            'event_count': 0
        })
        
        for bin_record in bins:
            operation = bin_record.get('operation', 'unknown')
            by_operation[operation]['file_count'] += bin_record.get('file_count', 0)
            by_operation[operation]['event_count'] += bin_record.get('event_count', 0)
        
        # Calculate images per hour
        if total_work_seconds > 0:
            images_per_hour = round(total_file_count / (total_work_seconds / 3600.0), 2)
        else:
            images_per_hour = 0.0
        
        return {
            'project_id': project_id,
            'title': manifest.get('title', project_id),
            'status': manifest.get('status'),
            'started_at': manifest.get('startedAt'),
            'finished_at': manifest.get('finishedAt'),
            'archived_at': datetime.now(timezone.utc).isoformat(),
            'bin_count': len(bins),
            'totals': {
                'file_count': total_file_count,
                'event_count': total_event_count,
                'work_seconds': round(total_work_seconds, 2),
                'work_hours': round(total_work_seconds / 3600.0, 2),
                'images_per_hour': images_per_hour
            },
            'by_script': dict(by_script),
            'by_operation': dict(by_operation)
        }
    
    def write_project_archive(
        self,
        project_id: str,
        bins: List[Dict[str, Any]],
        summary: Dict[str, Any],
        dry_run: bool = False
    ) -> Path:
        """Write project archive to disk.
        
        Args:
            project_id: Project ID
            bins: Project bins
            summary: Project summary
            dry_run: If True, don't write files
            
        Returns:
            Path to archive directory
        """
        archive_dir = self.archives_dir / project_id
        
        if dry_run:
            print(f"  [DRY RUN] Would create archive at: {archive_dir}")
            print(f"  [DRY RUN] Bins: {len(bins)}")
            print(f"  [DRY RUN] Total files: {summary['totals']['file_count']}")
            print(f"  [DRY RUN] Total work hours: {summary['totals']['work_hours']}")
            return archive_dir
        
        # Create archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Write bins
        bins_path = archive_dir / 'agg_15m.jsonl'
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=archive_dir,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            for bin_record in bins:
                json.dump(bin_record, tmp_file)
                tmp_file.write('\n')
        shutil.move(str(tmp_path), str(bins_path))
        
        # Write summary
        summary_path = archive_dir / 'project_summary.json'
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=archive_dir,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(summary, tmp_file, indent=2)
        shutil.move(str(tmp_path), str(summary_path))
        
        # Write manifest
        manifest = {
            'project_id': project_id,
            'archived_at': summary['archived_at'],
            'bin_count': len(bins),
            'format_version': '1.0'
        }
        manifest_path = archive_dir / 'archive_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  ✓ Archived to: {archive_dir}")
        print(f"    - {len(bins)} bins")
        print(f"    - {summary['totals']['file_count']} files")
        print(f"    - {summary['totals']['work_hours']} work hours")
        
        return archive_dir
    
    def merge_into_overall(
        self,
        project_id: str,
        bins: List[Dict[str, Any]],
        dry_run: bool = False
    ) -> None:
        """Merge project bins into overall cumulative aggregate.
        
        Args:
            project_id: Project ID
            bins: Project bins to merge
            dry_run: If True, don't write files
        """
        overall_path = self.overall_dir / 'agg_15m_cumulative.jsonl'
        
        # Load existing overall bins
        existing_bins = []
        existing_keys: Set[str] = set()
        
        if overall_path.exists():
            try:
                with open(overall_path, 'r') as f:
                    for line in f:
                        try:
                            bin_record = json.loads(line)
                            dedupe_key = bin_record.get('dedupe_key', '')
                            existing_bins.append(bin_record)
                            existing_keys.add(dedupe_key)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: Error reading existing overall bins: {e}")
        
        # Add new bins (skip duplicates)
        new_bins_added = 0
        duplicate_keys = []
        
        for bin_record in bins:
            dedupe_key = bin_record.get('dedupe_key', '')
            if dedupe_key in existing_keys:
                duplicate_keys.append(dedupe_key)
                continue
            
            existing_bins.append(bin_record)
            existing_keys.add(dedupe_key)
            new_bins_added += 1
        
        if duplicate_keys:
            print(f"  ⚠️  Skipped {len(duplicate_keys)} duplicate bins (already in overall)")
        
        if dry_run:
            print(f"  [DRY RUN] Would merge {new_bins_added} new bins into overall")
            print(f"  [DRY RUN] Overall would have {len(existing_bins)} total bins")
            return
        
        # Sort by bin timestamp
        existing_bins.sort(key=lambda x: x.get('bin_ts_utc', ''))
        
        # Write atomically
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.overall_dir,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            for bin_record in existing_bins:
                json.dump(bin_record, tmp_file)
                tmp_file.write('\n')
        
        shutil.move(str(tmp_path), str(overall_path))
        
        print(f"  ✓ Merged into overall: {new_bins_added} new bins")
        print(f"    Total overall bins: {len(existing_bins)}")
    
    def archive_project(
        self,
        project_id: str,
        dry_run: bool = False,
        skip_merge: bool = False
    ) -> bool:
        """Archive a finished project.
        
        Args:
            project_id: Project ID to archive
            dry_run: If True, preview changes without writing
            skip_merge: If True, skip merging into overall (archive only)
            
        Returns:
            True if successful
        """
        print(f"\n{'='*70}")
        print(f"Archiving project: {project_id}")
        print(f"{'='*70}")
        
        # Load project manifest
        manifest = self.load_project_manifest(project_id)
        if not manifest:
            print(f"✗ Error: Project manifest not found for {project_id}")
            return False
        
        # Check if project is finished
        status = manifest.get('status')
        finished_at = manifest.get('finishedAt')
        
        if status not in ('finished', 'archived'):
            print(f"✗ Error: Project status is '{status}', not 'finished' or 'archived'")
            return False
        
        if not finished_at:
            print("✗ Error: Project has no finishedAt timestamp")
            return False
        
        started_at = manifest.get('startedAt')
        if not started_at:
            print("✗ Error: Project has no startedAt timestamp")
            return False
        
        print(f"  Status: {status}")
        print(f"  Started: {started_at}")
        print(f"  Finished: {finished_at}")
        
        # Collect project bins
        print("\n  Collecting bins...")
        bins = self.collect_project_bins(project_id, started_at, finished_at)
        
        if not bins:
            print("  ⚠️  Warning: No bins found for project")
            print("  This may indicate:")
            print("    - Bins not yet generated (run aggregate_to_15m.py first)")
            print("    - Project has no file operations")
            if not dry_run:
                print("\n  Skipping archive (no data to archive)")
                return False
        else:
            print(f"  ✓ Found {len(bins)} bins")
        
        # Create summary
        summary = self.create_project_summary(project_id, manifest, bins)
        
        # Write archive
        print("\n  Writing archive...")
        self.write_project_archive(project_id, bins, summary, dry_run=dry_run)
        
        # Merge into overall (unless skipped)
        if not skip_merge and bins:
            print("\n  Merging into overall aggregate...")
            self.merge_into_overall(project_id, bins, dry_run=dry_run)
        
        print(f"\n{'='*70}")
        print("✅ Archive complete")
        print(f"{'='*70}\n")
        
        return True
    
    def rollback_project(self, project_id: str, dry_run: bool = False) -> bool:
        """Remove a project from overall aggregate (rollback).
        
        Args:
            project_id: Project ID to rollback
            dry_run: If True, preview changes without writing
            
        Returns:
            True if successful
        """
        print(f"\n{'='*70}")
        print(f"Rolling back project: {project_id}")
        print(f"{'='*70}")
        
        overall_path = self.overall_dir / 'agg_15m_cumulative.jsonl'
        
        if not overall_path.exists():
            print(f"✗ Error: Overall aggregate not found: {overall_path}")
            return False
        
        # Load overall bins
        overall_bins = []
        project_bins_found = 0
        
        try:
            with open(overall_path, 'r') as f:
                for line in f:
                    try:
                        bin_record = json.loads(line)
                        if bin_record.get('project_id') == project_id:
                            project_bins_found += 1
                        else:
                            overall_bins.append(bin_record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"✗ Error reading overall bins: {e}")
            return False
        
        if project_bins_found == 0:
            print(f"  ℹ️  Project {project_id} not found in overall aggregate (already rolled back?)")
            return True
        
        print(f"  Found {project_bins_found} bins for project {project_id}")
        print(f"  Overall will have {len(overall_bins)} bins after rollback")
        
        if dry_run:
            print(f"\n  [DRY RUN] Would remove {project_bins_found} bins from overall")
            print(f"  [DRY RUN] Archive directory would remain at: {self.archives_dir / project_id}")
            return True
        
        # Write updated overall (without project bins)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.overall_dir,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            for bin_record in overall_bins:
                json.dump(bin_record, tmp_file)
                tmp_file.write('\n')
        
        shutil.move(str(tmp_path), str(overall_path))
        
        print(f"\n  ✓ Removed {project_bins_found} bins from overall")
        print(f"  Note: Archive directory still exists at: {self.archives_dir / project_id}")
        print("        (Delete manually if needed)")
        
        print(f"\n{'='*70}")
        print("✅ Rollback complete")
        print(f"{'='*70}\n")
        
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Archive finished project bins and merge into overall aggregate'
    )
    parser.add_argument(
        'project_id',
        type=str,
        help='Project ID to archive'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory path (default: project_root/data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )
    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='Archive only, skip merging into overall'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Remove project from overall aggregate (rollback)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(2)
    
    # Create archiver
    archiver = ProjectArchiver(data_dir)
    
    # Execute command
    if args.rollback:
        success = archiver.rollback_project(args.project_id, dry_run=args.dry_run)
    else:
        success = archiver.archive_project(
            args.project_id,
            dry_run=args.dry_run,
            skip_merge=args.skip_merge
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

