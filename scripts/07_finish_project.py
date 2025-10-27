#!/usr/bin/env python3
"""
Project Finish Script
====================
Friendly wrapper for completing a project with prezip_stager integration.

Usage:
    python scripts/00_finish_project.py
    
    # Or with arguments:
    python scripts/00_finish_project.py --project-id mojo3
    
Features:
- Interactive wizard for project completion
- Automatically counts final images
- Runs prezip_stager to create delivery ZIP
- Updates manifest with finishedAt and stager metrics
- Optional dry-run mode to preview before committing
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def count_images(directory: Path) -> int:
    """Count PNG images in a directory (recursive to match actual project structure)."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.png")))


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run_prezip_stager(
    project_id: str,
    content_dir: Path,
    output_zip: Path,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Run prezip_stager to create delivery ZIP and update manifest.
    
    Args:
        project_id: Project identifier
        content_dir: Content directory path
        output_zip: Output ZIP file path
        dry_run: If True, run without committing (preview mode)
        
    Returns:
        Dictionary with status and stager results
    """
    # Build prezip_stager command
    cmd = [
        sys.executable,
        "scripts/tools/prezip_stager.py",
        "--project-id", project_id,
        "--content-dir", str(content_dir),
        "--output-zip", str(output_zip)
    ]
    
    if not dry_run:
        cmd.extend(["--commit", "--update-manifest"])
    
    print(f"\n{'🔍 DRY RUN' if dry_run else '🚀 EXECUTING'}: {' '.join(cmd)}")
    print("=" * 70)
    
    try:
        # Run prezip_stager and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print stdout for progress
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Try to parse JSON output
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            output = {"message": result.stdout}
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"prezip_stager failed with exit code {result.returncode}",
                "details": output
            }
        
        return {
            "status": "success",
            "stager_output": output,
            "dry_run": dry_run
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to run prezip_stager: {e}"
        }


def finish_project(
    project_id: str,
    dry_run: bool = True,
    force: bool = False
) -> dict:
    """
    Finish a project by running prezip_stager to create delivery ZIP.
    
    This wrapper provides a friendly interface to prezip_stager, which:
    - Validates directory state (FULL check)
    - Creates delivery ZIP with allowlist filtering
    - Updates manifest with finishedAt and stager metrics
    - Backs up manifest before updating
    
    Args:
        project_id: Project identifier
        dry_run: If True, preview without committing
        force: Not used (kept for API compatibility)
        
    Returns:
        Dictionary with status and details
    """
    # Find manifest
    manifest_path = Path("data/projects") / f"{project_id}.project.json"
    
    if not manifest_path.exists():
        return {
            "status": "error",
            "message": f"Project manifest not found: {manifest_path}"
        }
    
    # Load manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read manifest: {e}"
        }
    
    # Check if already finished
    if manifest.get('finishedAt') and not force and not dry_run:
        return {
            "status": "error",
            "message": f"Project already finished at {manifest['finishedAt']}. Use --force to override."
        }
    
    # Get content directory
    content_dir_rel = manifest.get('paths', {}).get('root')
    if not content_dir_rel:
        return {
            "status": "error",
            "message": "No content directory (paths.root) in manifest"
        }
    
    # Resolve path relative to manifest location (data/projects/)
    # If path is relative (e.g., "../../mojo2"), resolve from manifest directory
    content_dir_path = Path(content_dir_rel).expanduser()
    if not content_dir_path.is_absolute():
        # Resolve relative to manifest directory
        content_dir = (manifest_path.parent / content_dir_path).resolve()
    else:
        # Already absolute
        content_dir = content_dir_path.resolve()
    
    if not content_dir.exists():
        return {
            "status": "error",
            "message": f"Content directory does not exist: {content_dir}"
        }
    
    # Auto-count images
    final_images = count_images(content_dir)
    print(f"📊 Found {final_images} PNG images in {content_dir}")
    
    # Determine output ZIP path (no exports/ dir; place at repo root)
    output_zip = Path(f"{project_id}_final.zip")
    
    # Run prezip_stager
    result = run_prezip_stager(
        project_id=project_id,
        content_dir=content_dir,
        output_zip=output_zip,
        dry_run=dry_run
    )
    
    if result["status"] != "success":
        return result
    
    # Reload manifest to get updated values (if not dry run)
    if not dry_run:
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            finished_at = manifest.get('finishedAt')
            stager_metrics = manifest.get('metrics', {}).get('stager', {})
            # Ensure status is finished if finishedAt present
            if finished_at and manifest.get('status') != 'finished':
                manifest['status'] = 'finished'
                # Backup and write
                backup = manifest_path.with_suffix('.project.json.bak')
                try:
                    import shutil
                    shutil.copy2(manifest_path, backup)
                except Exception:
                    pass
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            # Remove project/content dir patterns from .gitignore
            try:
                gitignore_path = Path('.gitignore')
                if gitignore_path.exists():
                    content_dir_rel = manifest.get('paths', {}).get('root') or ''
                    content_name = Path(content_dir_rel).name if content_dir_rel else ''
                    project_pattern = f"{project_id}/"
                    content_pattern = f"{content_name}/" if content_name else None
                    lines = gitignore_path.read_text().splitlines()
                    new_lines = []
                    for ln in lines:
                        if ln.strip() in {project_pattern, content_pattern}:
                            continue
                        new_lines.append(ln)
                    gitignore_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            except Exception:
                pass
        except Exception:
            finished_at = None
            stager_metrics = {}
    else:
        finished_at = None
        stager_metrics = {}
    
    # Archive project bins (if bins system is enabled and project is finished)
    archive_result = None
    if not dry_run and finished_at:
        try:
            print("\n📦 Archiving project bins...")
            # Import and run archive script
            sys.path.insert(0, str(Path(__file__).parent / "data_pipeline"))
            from archive_project_bins import ProjectArchiver
            
            archiver = ProjectArchiver(Path("data"))
            archive_success = archiver.archive_project(project_id, dry_run=False)
            
            archive_result = {
                "status": "success" if archive_success else "skipped",
                "message": "Project bins archived successfully" if archive_success else "No bins to archive"
            }
        except ImportError:
            print("  ⚠️  Archive script not found (bins system not set up)")
            archive_result = {"status": "skipped", "message": "Archive script not available"}
        except Exception as e:
            print(f"  ⚠️  Archive failed: {e}")
            # Don't fail the entire operation if archive fails
            archive_result = {"status": "error", "message": str(e)}
    
    return {
        "status": "success",
        "message": "Project finished successfully" if not dry_run else "Dry run completed - use --commit to finalize",
        "manifest_path": str(manifest_path),
        "project_id": project_id,
        "finished_at": finished_at,
        "final_images": final_images,
        "output_zip": str(output_zip) if not dry_run else None,
        "stager_metrics": stager_metrics,
        "archive": archive_result,
        "dry_run": dry_run
    }


def interactive_mode():
    """Run the script in interactive mode, prompting for all inputs."""
    print("=" * 70)
    print("🏁 Project Finish Script (with prezip_stager integration)")
    print("=" * 70)
    print()
    
    # List available projects
    projects_dir = Path("data/projects")
    if projects_dir.exists():
        manifests = list(projects_dir.glob("*.project.json"))
        active_projects = []
        
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest.get('status') == 'active':
                    project_id = manifest.get('projectId')
                    title = manifest.get('title', project_id)
                    initial = manifest.get('counts', {}).get('initialImages', '?')
                    started = manifest.get('startedAt', 'Unknown')
                    active_projects.append((project_id, title, initial, started))
            except Exception:
                continue
        
        if active_projects:
            print("Active projects:")
            for pid, title, initial, started in active_projects:
                print(f"  • {pid} ({title}) - {initial} images, started {started}")
            print()
    
    # Get project ID
    project_id = input("Enter project ID to finish: ").strip()
    
    if not project_id:
        print("❌ No project ID provided")
        return
    
    # Check manifest exists
    manifest_path = Path("data/projects") / f"{project_id}.project.json"
    if not manifest_path.exists():
        print(f"❌ Project manifest not found: {manifest_path}")
        return
    
    # Load manifest for preview
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Failed to read manifest: {e}")
        return
    
    # Get content directory
    content_dir_rel = manifest.get('paths', {}).get('root')
    if content_dir_rel:
        # Resolve path relative to manifest location
        content_dir_path = Path(content_dir_rel).expanduser()
        if not content_dir_path.is_absolute():
            # Resolve relative to manifest directory
            content_dir = (manifest_path.parent / content_dir_path).resolve()
        else:
            # Already absolute
            content_dir = content_dir_path.resolve()
        
        if content_dir.exists():
            final_images = count_images(content_dir)
            print(f"📊 Found {final_images} PNG images in {content_dir}")
        else:
            print(f"⚠️  Warning: Content directory not found: {content_dir}")
            final_images = 0
    else:
        final_images = 0
    
    # Preview
    print()
    print("=" * 70)
    print("📋 Finish Summary")
    print("=" * 70)
    print(f"Project ID:       {project_id}")
    print(f"Title:            {manifest.get('title', project_id)}")
    print(f"Started At:       {manifest.get('startedAt', 'Unknown')}")
    print(f"Content Dir:      {content_dir_rel}")
    print(f"Initial Images:   {manifest.get('counts', {}).get('initialImages', 0)}")
    print(f"Final Images:     {final_images}")
    print()
    print("This will:")
    print("  1. Run prezip_stager to validate directory state")
    print("  2. Create delivery ZIP with allowlist filtering")
    print("  3. Update manifest with finishedAt and stager metrics")
    print("  4. Backup manifest before updating")
    print("=" * 70)
    print()
    
    # Warn if already finished
    if manifest.get('finishedAt'):
        print(f"⚠️  Warning: Project was already finished at {manifest['finishedAt']}")
        print("This will overwrite that timestamp.")
        print()
    
    # Ask about dry run
    dry_run_input = input("Run in dry-run mode first? (Y/n): ").strip().lower()
    dry_run = dry_run_input != 'n'
    
    if dry_run:
        print("\n🔍 Running in DRY RUN mode (preview only)...")
    else:
        confirm = input("\n⚠️  This will finalize the project. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled by user")
            return
    
    # Finish project
    result = finish_project(
        project_id=project_id,
        dry_run=dry_run,
        force=True
    )
    
    # Display result
    if result["status"] == "success":
        print()
        print("=" * 70)
        if dry_run:
            print("🔍 DRY RUN COMPLETED")
            print("=" * 70)
            print("Preview successful! No changes made.")
            print()
            print("To finalize, run again with:")
            print(f"  python scripts/00_finish_project.py --project-id {project_id} --commit")
        else:
            print("✅ SUCCESS!")
            print("=" * 70)
            print(f"Manifest updated:  {result['manifest_path']}")
            print(f"Project ID:        {result['project_id']}")
            print(f"Finished At:       {result['finished_at']}")
            print(f"Final Images:      {result['final_images']}")
            print(f"Output ZIP:        {result['output_zip']}")
            if result.get('stager_metrics'):
                metrics = result['stager_metrics']
                print(f"ZIP Contents:      {metrics.get('eligibleCount', '?')} files")
            print()
            print("🎯 Next steps:")
            print(f"   • Upload: {result['output_zip']}")
            print("   • View dashboard for final metrics")
        print("=" * 70)
    else:
        print()
        print(f"❌ {result['status'].upper()}: {result['message']}")
        if 'details' in result:
            print(f"Details: {result['details']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Finish a project by running prezip_stager to create delivery ZIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (with dry-run preview)
  python scripts/00_finish_project.py
  
  # Dry run (preview only)
  python scripts/00_finish_project.py --project-id mojo3
  
  # Commit (finalize project and create ZIP)
  python scripts/00_finish_project.py --project-id mojo3 --commit
  
  # Force overwrite if already finished
  python scripts/00_finish_project.py --project-id mojo3 --commit --force

What this does:
  1. Validates directory state (FULL check from scan_dir_state)
  2. Creates delivery ZIP using allowlist filtering
  3. Updates manifest with finishedAt timestamp
  4. Adds stager metrics (file counts, eligible count, etc.)
  5. Backs up manifest before updating
        """
    )
    
    parser.add_argument(
        '--project-id',
        help='Project identifier'
    )
    parser.add_argument(
        '--commit',
        action='store_true',
        help='Commit changes (default is dry-run preview mode)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite finishedAt even if already set'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run in interactive mode
    if not args.project_id:
        interactive_mode()
        return
    
    # Command-line mode
    dry_run = not args.commit
    
    result = finish_project(
        project_id=args.project_id,
        dry_run=dry_run,
        force=args.force
    )
    
    # Print result as JSON for scripting
    print(json.dumps(result, indent=2, default=str))
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()

