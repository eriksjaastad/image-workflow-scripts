#!/usr/bin/env python3
"""
Project Startup Script
======================
Creates a new project manifest with all required fields, proper timestamps,
and automatic image counting.

Usage:
    python scripts/00_start_project.py
    
    # Or with arguments:
    python scripts/00_start_project.py --project-id mojo3 --content-dir ../mojo3
    
Features:
- Automatically generates UTC timestamps with Z suffix
- Counts PNG images in content directory
- Creates manifest with all required fields
- Backs up existing manifests before overwriting
- Validates project directory structure
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def count_images(directory: Path) -> int:
    """Count PNG images in a directory (non-recursive)."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.png")))


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def validate_project_id(project_id: str) -> bool:
    """Validate project ID format."""
    if not project_id:
        return False
    # Simple validation: alphanumeric, underscore, hyphen
    return all(c.isalnum() or c in '-_' for c in project_id)


def get_project_manifest_template(
    project_id: str,
    title: str,
    content_dir: Path,
    initial_images: int,
    timestamp: str
) -> dict:
    """
    Generate a project manifest template with all required fields.
    
    Returns a dictionary matching the schema used by mojo1/mojo2 projects.
    """
    return {
        "schemaVersion": 1,
        "projectId": project_id,
        "title": title,
        "status": "active",
        "createdAt": timestamp,
        "startedAt": timestamp,
        "finishedAt": None,
        "removeFileOnFinish": True,
        "paths": {
            "root": str(content_dir),
            "selectedDir": "../../selected",
            "cropDir": "../../crop",
            "characterGroups": [
                "../../character_group_1",
                "../../character_group_2",
                "../../character_group_3"
            ]
        },
        "counts": {
            "initialImages": initial_images,
            "finalImages": None
        },
        "metrics": {
            "imagesPerHourEndToEnd": None,
            "stepRates": {},
            "stager": {
                "zip": "",
                "eligibleCount": 0,
                "byExtIncluded": {},
                "excludedCounts": {},
                "incomingByExt": {}
            }
        },
        "steps": [
            {
                "name": "select_versions",
                "startedAt": None,
                "finishedAt": None,
                "imagesProcessed": None
            },
            {
                "name": "character_sort",
                "startedAt": None,
                "finishedAt": None,
                "imagesProcessed": None
            },
            {
                "name": "crop",
                "startedAt": None,
                "finishedAt": None,
                "imagesProcessed": None
            },
            {
                "name": "dedupe",
                "startedAt": None,
                "finishedAt": None,
                "imagesProcessed": None
            },
            {
                "name": "final_review",
                "startedAt": None,
                "finishedAt": None,
                "imagesProcessed": None
            }
        ],
        "notes": "Project manifest stored outside content to avoid inclusion in deliverables."
    }


def create_project_manifest(
    project_id: str,
    content_dir: Path,
    title: Optional[str] = None,
    force: bool = False
) -> dict:
    """
    Create a new project manifest.
    
    Args:
        project_id: Unique project identifier (e.g., "mojo3")
        content_dir: Path to project content directory
        title: Human-readable project title (defaults to capitalized project_id)
        force: Overwrite existing manifest without prompting
        
    Returns:
        Dictionary with status and manifest path
    """
    # Validate inputs
    if not validate_project_id(project_id):
        return {
            "status": "error",
            "message": f"Invalid project ID: {project_id}. Use alphanumeric, underscore, or hyphen only."
        }
    
    # Resolve content directory
    content_dir = content_dir.resolve()
    if not content_dir.exists():
        return {
            "status": "error",
            "message": f"Content directory does not exist: {content_dir}"
        }
    
    if not content_dir.is_dir():
        return {
            "status": "error",
            "message": f"Content path is not a directory: {content_dir}"
        }
    
    # Count images
    image_count = count_images(content_dir)
    if image_count == 0:
        print(f"⚠️  Warning: No PNG images found in {content_dir}")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            return {"status": "cancelled", "message": "User cancelled due to no images found"}
    
    # Generate title if not provided
    if not title:
        title = project_id.capitalize()
    
    # Check if manifest already exists
    manifest_dir = Path("data/projects")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{project_id}.project.json"
    
    if manifest_path.exists() and not force:
        print(f"⚠️  Project manifest already exists: {manifest_path}")
        response = input("Overwrite existing manifest? (y/N): ").strip().lower()
        if response != 'y':
            return {"status": "cancelled", "message": "User cancelled to avoid overwriting"}
        
        # Create backup
        backup_path = manifest_path.with_suffix('.project.json.bak')
        try:
            shutil.copy2(manifest_path, backup_path)
            print(f"✅ Created backup: {backup_path}")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create backup: {e}"
            }
    
    # Generate timestamp
    timestamp = get_utc_timestamp()
    
    # Make content_dir relative to project root (for manifest storage)
    try:
        repo_root = Path.cwd()
        relative_content = Path("../../" + content_dir.name)
    except Exception:
        relative_content = content_dir
    
    # Generate manifest
    manifest = get_project_manifest_template(
        project_id=project_id,
        title=title,
        content_dir=relative_content,
        initial_images=image_count,
        timestamp=timestamp
    )
    
    # Write manifest
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to write manifest: {e}"
        }
    
    return {
        "status": "success",
        "message": "Project manifest created successfully",
        "manifest_path": str(manifest_path),
        "project_id": project_id,
        "initial_images": image_count,
        "started_at": timestamp
    }


def interactive_mode():
    """Run the script in interactive mode, prompting for all inputs."""
    print("=" * 60)
    print("🚀 Project Startup Script")
    print("=" * 60)
    print()
    
    # Get project ID
    while True:
        project_id = input("Enter project ID (e.g., 'mojo3'): ").strip()
        if validate_project_id(project_id):
            break
        print("❌ Invalid project ID. Use alphanumeric characters, underscores, or hyphens.")
    
    # Get content directory
    while True:
        content_dir_input = input(f"Enter content directory path (relative to repo root, e.g., '../{project_id}'): ").strip()
        content_dir = Path(content_dir_input).expanduser()
        
        if content_dir.exists() and content_dir.is_dir():
            break
        
        print(f"❌ Directory not found or not a directory: {content_dir}")
        create_new = input("Would you like to create this directory? (y/N): ").strip().lower()
        if create_new == 'y':
            try:
                content_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {content_dir}")
                break
            except Exception as e:
                print(f"❌ Failed to create directory: {e}")
    
    # Get optional title
    default_title = project_id.capitalize()
    title_input = input(f"Enter project title (default: '{default_title}'): ").strip()
    title = title_input if title_input else default_title
    
    # Preview
    image_count = count_images(content_dir)
    print()
    print("=" * 60)
    print("📋 Project Summary")
    print("=" * 60)
    print(f"Project ID:       {project_id}")
    print(f"Title:            {title}")
    print(f"Content Dir:      {content_dir}")
    print(f"Initial Images:   {image_count} PNG files")
    print(f"Timestamp:        {get_utc_timestamp()}")
    print("=" * 60)
    print()
    
    # Confirm
    confirm = input("Create project manifest? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled by user")
        return
    
    # Create manifest
    result = create_project_manifest(
        project_id=project_id,
        content_dir=content_dir,
        title=title,
        force=False
    )
    
    # Display result
    if result["status"] == "success":
        print()
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"Manifest created: {result['manifest_path']}")
        print(f"Project ID:       {result['project_id']}")
        print(f"Initial Images:   {result['initial_images']}")
        print(f"Started At:       {result['started_at']}")
        print()
        print("🎯 Next steps:")
        print(f"   1. Run your image processing tools")
        print(f"   2. When complete, run the prezip_stager to finish the project")
        print("=" * 60)
    else:
        print()
        print(f"❌ {result['status'].upper()}: {result['message']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a new project manifest with proper timestamps and image counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/00_start_project.py
  
  # With arguments
  python scripts/00_start_project.py --project-id mojo3 --content-dir ../mojo3
  
  # With custom title
  python scripts/00_start_project.py --project-id mojo3 --content-dir ../mojo3 --title "Mojo Project 3"
  
  # Force overwrite
  python scripts/00_start_project.py --project-id mojo3 --content-dir ../mojo3 --force
        """
    )
    
    parser.add_argument(
        '--project-id',
        help='Unique project identifier (e.g., mojo3)'
    )
    parser.add_argument(
        '--content-dir',
        help='Path to project content directory'
    )
    parser.add_argument(
        '--title',
        help='Human-readable project title (default: capitalized project-id)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing manifest without prompting'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run in interactive mode
    if not args.project_id or not args.content_dir:
        if args.project_id or args.content_dir:
            print("❌ Error: Both --project-id and --content-dir are required when using arguments")
            print("Run without arguments for interactive mode.")
            sys.exit(1)
        
        interactive_mode()
        return
    
    # Command-line mode
    content_dir = Path(args.content_dir).expanduser()
    
    result = create_project_manifest(
        project_id=args.project_id,
        content_dir=content_dir,
        title=args.title,
        force=args.force
    )
    
    # Print result as JSON for scripting
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()

