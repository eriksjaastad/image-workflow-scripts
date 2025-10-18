#!/usr/bin/env python3
"""
Archive All Finished Projects
==============================
Convenience script to archive all finished projects at once.

This converts raw logs for finished projects into pre-aggregated bins,
dramatically speeding up dashboard load times.
"""

import json
import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def main():
    print("=" * 70)
    print("ARCHIVE ALL FINISHED PROJECTS")
    print("=" * 70)
    print()
    
    # Find all project manifests
    projects_dir = project_root / 'data' / 'projects'
    if not projects_dir.exists():
        print(f"✗ Projects directory not found: {projects_dir}")
        sys.exit(1)
    
    manifests = list(projects_dir.glob('*.project.json'))
    if not manifests:
        print(f"✗ No project manifests found in {projects_dir}")
        sys.exit(1)
    
    print(f"Found {len(manifests)} project manifests")
    print()
    
    # First: Generate bins for all historical data
    print("Step 1: Generate 15-minute bins for last 180 days...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, 'scripts/data_pipeline/aggregate_to_15m.py', '--days', '180'],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Failed to generate bins:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    print("✓ Bins generated successfully")
    print()
    
    # Find finished projects
    finished_projects = []
    active_projects = []
    
    for manifest_path in manifests:
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            project_id = manifest.get('projectId')
            status = manifest.get('status')
            title = manifest.get('title', project_id)
            
            if not project_id:
                continue
            
            if status == 'finished' or status == 'archived':
                finished_projects.append((project_id, title, manifest))
            else:
                active_projects.append((project_id, title, status))
        except Exception as e:
            print(f"✗ Error reading {manifest_path}: {e}")
            continue
    
    print(f"Found:")
    print(f"  • {len(finished_projects)} finished projects")
    print(f"  • {len(active_projects)} active projects")
    print()
    
    if not finished_projects:
        print("No finished projects to archive!")
        sys.exit(0)
    
    # Step 2: Archive each finished project
    print("Step 2: Archive finished projects...")
    print("-" * 70)
    print()
    
    archived_count = 0
    failed = []
    
    for project_id, title, manifest in finished_projects:
        print(f"Archiving: {project_id} ({title})")
        
        result = subprocess.run(
            [sys.executable, 'scripts/data_pipeline/archive_project_bins.py', project_id],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            archived_count += 1
            print(f"  ✓ Success")
            # Show key stats from output
            if 'bins' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'bins' in line.lower() or 'files' in line.lower() or 'hours' in line.lower():
                        print(f"    {line.strip()}")
        else:
            failed.append(project_id)
            print(f"  ✗ Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully archived: {archived_count} projects")
    if failed:
        print(f"✗ Failed to archive: {len(failed)} projects")
        for pid in failed:
            print(f"    - {pid}")
    print()
    
    # Next steps
    print("NEXT STEPS:")
    print("  1. Test dashboard performance:")
    print("     python scripts/dashboard/test_timing.py")
    print()
    print("  2. If performance is good, you can optionally remove old raw logs")
    print("     (But NOT recommended yet - keep raw logs as backup)")
    print()
    
    print("✅ Archive complete!")

if __name__ == '__main__':
    main()

