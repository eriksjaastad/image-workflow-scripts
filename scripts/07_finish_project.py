#!/usr/bin/env python3
"""
Project Finish Script
====================
Friendly wrapper for completing a project with prezip_stager integration.

Usage:
    python scripts/07_finish_project.py

    # Or with arguments:
    python scripts/07_finish_project.py --project-id mojo3

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

# Ensure project root on import path for local package imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def count_images(directory: Path) -> int:
    """Count PNG images in a directory (recursive to match actual project structure)."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.png")))


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_prezip_stager(
    project_id: str,
    content_dir: Path,
    output_zip: Path,
    dry_run: bool = True,
    verbose: bool = False,
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
        "--project-id",
        project_id,
        "--content-dir",
        str(content_dir),
        "--output-zip",
        str(output_zip),
    ]

    if not dry_run:
        cmd.extend(["--commit", "--update-manifest"])

    print(f"\n{'🔍 DRY RUN' if dry_run else '🚀 EXECUTING'}: {' '.join(cmd)}")
    print("=" * 70)

    try:
        # Run prezip_stager and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Print raw tool output only in verbose mode to avoid overwhelming the terminal
        if verbose and result.stdout:
            print(result.stdout)

        # Print stderr if there are errors
        if result.stderr and (verbose or result.returncode != 0):
            print(result.stderr, file=sys.stderr)

        # Try to parse JSON output
        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            output = {"message": result.stdout}

        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"prezip_stager failed with exit code {result.returncode}",
                "details": output,
            }

        return {"status": "success", "stager_output": output, "dry_run": dry_run}

    except Exception as e:
        return {"status": "error", "message": f"Failed to run prezip_stager: {e}"}


def finish_project(project_id: str, dry_run: bool = True, force: bool = False) -> dict:
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
            "message": f"Project manifest not found: {manifest_path}",
        }

    # Load manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to read manifest: {e}"}

    # Check if already finished
    if manifest.get("finishedAt") and not force and not dry_run:
        return {
            "status": "error",
            "message": f"Project already finished at {manifest['finishedAt']}. Use --force to override.",
        }

    # Get content directory
    content_dir_rel = manifest.get("paths", {}).get("root")
    if not content_dir_rel:
        return {
            "status": "error",
            "message": "No content directory (paths.root) in manifest",
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
            "message": f"Content directory does not exist: {content_dir}",
        }

    # Auto-count images
    final_images = count_images(content_dir)
    print(f"📊 Found {final_images} PNG images in {content_dir}")

    # Determine output ZIP path (no exports/ dir; place at repo root)
    output_zip = Path(f"{project_id}_final.zip")

    # Run prezip_stager
    # Add concise dry-run summary printer to avoid overwhelming output
    def print_cleanup_summary(base: Path) -> None:
        try:
            # Reuse stager rules for consistency
            from scripts.tools.prezip_stager import (
                is_hidden,
                load_allowlist,
                load_bans,
                matches_banned_patterns,
            )

            allowed, overrides = load_allowlist(project_id, None)
            banned_ext, banned_patterns, banned_basenames = load_bans(
                Path("data/projects/global_bans.json").resolve()
            )

            counts = {
                "decision": 0,
                "yaml": 0,
                "prompts": 0,
                "other": 0,
            }
            notable: list[str] = []
            included_by_ext: Dict[str, int] = {}

            def categorize(rel_path: str, ext: str) -> str:
                if ext == "decision":
                    return "decision"
                if ext == "yaml":
                    return "yaml"
                if "prompts/" in rel_path or rel_path.startswith("prompts/"):
                    return "prompts"
                return "other"

            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                name = p.name
                if is_hidden(name):
                    continue
                ext = p.suffix.lower().lstrip(".")
                base_name = name.lower()
                # Determine if this file would be excluded (i.e., not eligible)
                excluded = (
                    (not ext)
                    or (base_name in banned_basenames)
                    or (ext in banned_ext)
                    or matches_banned_patterns(name, banned_patterns)
                    or (ext not in allowed and ext not in overrides)
                )
                rel = str(p.relative_to(base))
                if excluded:
                    cat = categorize(rel, ext)
                    counts[cat] += 1
                    if cat == "other" and len(notable) < 5:
                        notable.append(rel)
                else:
                    included_by_ext[ext] = included_by_ext.get(ext, 0) + 1

            total_excluded = sum(counts.values())
            total_included = sum(included_by_ext.values())

            # Summaries printed at the bottom
            print("[DRY RUN] Included Summary:")
            for ext, c in sorted(
                included_by_ext.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                print(f"  - .{ext}: {c} files")
            print(f"  - Total files to include: {total_included} files\n")

            print("[DRY RUN] Cleanup Summary:")
            any_printed = False
            for label, key in (
                (".decision files", "decision"),
                (".yaml files", "yaml"),
                ("prompts/ directory", "prompts"),
                ("other", "other"),
            ):
                if counts.get(key, 0) > 0:
                    print(f"  - {label}: {counts[key]} files")
                    any_printed = True
            if not any_printed:
                print("  - No files to remove")
            print(f"  - Total files to remove: {total_excluded} files")
            if notable:
                print("\n[DRY RUN] Notable files in cleanup:")
                for n in notable:
                    print(f"  - {n}")
            print("\n[DRY RUN] ✅ Ready to finalize project")
        except Exception:
            # Fail open: don't block finishing if summary fails
            pass

    result = run_prezip_stager(
        project_id=project_id,
        content_dir=content_dir,
        output_zip=output_zip,
        dry_run=dry_run,
        verbose=False,
    )

    if result["status"] != "success":
        return result

    # Reload manifest to get updated values (if not dry run)
    if not dry_run:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            finished_at = manifest.get("finishedAt")
            stager_metrics = manifest.get("metrics", {}).get("stager", {})
            # Ensure status is finished if finishedAt present
            if finished_at and manifest.get("status") != "finished":
                manifest["status"] = "finished"
                # Backup and write
                backup = manifest_path.with_suffix(".project.json.bak")
                try:
                    import shutil

                    shutil.copy2(manifest_path, backup)
                except Exception:
                    pass
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
            # Remove project/content dir patterns from .gitignore
            try:
                gitignore_path = Path(".gitignore")
                if gitignore_path.exists():
                    content_dir_rel = manifest.get("paths", {}).get("root") or ""
                    content_name = Path(content_dir_rel).name if content_dir_rel else ""
                    project_pattern = f"{project_id}/"
                    content_pattern = f"{content_name}/" if content_name else None
                    lines = gitignore_path.read_text().splitlines()
                    new_lines = []
                    for ln in lines:
                        if ln.strip() in {project_pattern, content_pattern}:
                            continue
                        new_lines.append(ln)
                    gitignore_path.write_text(
                        "\n".join(new_lines) + ("\n" if new_lines else "")
                    )
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
                "message": "Project bins archived successfully"
                if archive_success
                else "No bins to archive",
            }
        except ImportError:
            print("  ⚠️  Archive script not found (bins system not set up)")
            archive_result = {
                "status": "skipped",
                "message": "Archive script not available",
            }
        except Exception as e:
            print(f"  ⚠️  Archive failed: {e}")
            # Don't fail the entire operation if archive fails
            archive_result = {"status": "error", "message": str(e)}

    # For dry run, print a concise cleanup summary at the end (bottom of output)
    if dry_run:
        print_cleanup_summary(content_dir)

    return {
        "status": "success",
        "message": "Project finished successfully"
        if not dry_run
        else "Dry run completed - use --commit to finalize",
        "manifest_path": str(manifest_path),
        "project_id": project_id,
        "finished_at": finished_at,
        "final_images": final_images,
        "output_zip": str(output_zip) if not dry_run else None,
        "stager_metrics": stager_metrics,
        "archive": archive_result,
        "dry_run": dry_run,
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
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "active":
                    project_id = manifest.get("projectId")
                    title = manifest.get("title", project_id)
                    initial = manifest.get("counts", {}).get("initialImages", "?")
                    started = manifest.get("startedAt", "Unknown")
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
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to read manifest: {e}")
        return

    # Get content directory
    content_dir_rel = manifest.get("paths", {}).get("root")
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
    if manifest.get("finishedAt"):
        print(f"⚠️  Warning: Project was already finished at {manifest['finishedAt']}")
        print("This will overwrite that timestamp.")
        print()

    # Ask about dry run
    dry_run_input = input("Run in dry-run mode first? (Y/n): ").strip().lower()
    dry_run = dry_run_input != "n"

    if dry_run:
        print("\n🔍 Running in DRY RUN mode (preview only)...")
    else:
        confirm = (
            input("\n⚠️  This will finalize the project. Continue? (y/N): ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("❌ Cancelled by user")
            return

    # Finish project
    result = finish_project(project_id=project_id, dry_run=dry_run, force=True)

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
            print(
                f"  python scripts/07_finish_project.py --project-id {project_id} --commit"
            )
        else:
            print("✅ SUCCESS!")
            print("=" * 70)
            print(f"Manifest updated:  {result['manifest_path']}")
            print(f"Project ID:        {result['project_id']}")
            print(f"Finished At:       {result['finished_at']}")
            print(f"Final Images:      {result['final_images']}")
            print(f"Output ZIP:        {result['output_zip']}")
            if result.get("stager_metrics"):
                metrics = result["stager_metrics"]
                print(f"ZIP Contents:      {metrics.get('eligibleCount', '?')} files")
            print()
            print("🎯 Next steps:")
            print(f"   • Upload: {result['output_zip']}")
            print("   • View dashboard for final metrics")
        print("=" * 70)
    else:
        print()
        print(f"❌ {result['status'].upper()}: {result['message']}")
        if "details" in result:
            print(f"Details: {result['details']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Finish a project by running prezip_stager to create delivery ZIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (with dry-run preview)
  python scripts/07_finish_project.py
  
  # Dry run (preview only)
  python scripts/07_finish_project.py --project-id mojo3
  
  # Commit (finalize project and create ZIP)
  python scripts/07_finish_project.py --project-id mojo3 --commit
  
  # Force overwrite if already finished
  python scripts/07_finish_project.py --project-id mojo3 --commit --force

What this does:
  1. Validates directory state (FULL check from scan_dir_state)
  2. Creates delivery ZIP using allowlist filtering
  3. Updates manifest with finishedAt timestamp
  4. Adds stager metrics (file counts, eligible count, etc.)
  5. Backs up manifest before updating
        """,
    )

    parser.add_argument("--project-id", help="Project identifier")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit changes (default is dry-run preview mode)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite finishedAt even if already set"
    )

    args = parser.parse_args()

    # If no arguments provided, run in interactive mode
    if not args.project_id:
        interactive_mode()
        return

    # Command-line mode
    dry_run = not args.commit

    result = finish_project(
        project_id=args.project_id, dry_run=dry_run, force=args.force
    )

    # Print result as JSON for scripting
    print(json.dumps(result, indent=2, default=str))

    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
