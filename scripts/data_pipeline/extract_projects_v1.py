#!/usr/bin/env python3
"""
Extract Projects v1
===================
Extracts and normalizes project manifests to snapshot format.

Reads from:
- data/projects/*.project.json

Outputs to:
- snapshot/projects_v1/projects.jsonl (single file, all projects)
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = PROJECT_ROOT / "data" / "projects"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "projects_v1"


def parse_timestamp(ts_str: str) -> datetime | None:
    """Parse ISO timestamp to UTC."""
    if not ts_str:
        return None

    try:
        if ts_str.endswith("Z"):
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        return None


def normalize_project(
    raw_project: dict[str, Any], source_file: str
) -> dict[str, Any] | None:
    """Normalize a project manifest to canonical schema."""
    try:
        project_id = raw_project.get("projectId")
        if not project_id:
            return None

        # Parse timestamps
        created_at = parse_timestamp(raw_project.get("createdAt"))
        started_at = parse_timestamp(raw_project.get("startedAt"))
        finished_at = parse_timestamp(raw_project.get("finishedAt"))

        # Build normalized project
        normalized = {
            "project_id": project_id,
            "title": raw_project.get("title") or project_id,
            "status": raw_project.get("status", "unknown"),
            "created_at_utc": created_at.isoformat() if created_at else None,
            "started_at_utc": started_at.isoformat() if started_at else None,
            "finished_at_utc": finished_at.isoformat() if finished_at else None,
            "schema_version": "project_v1",
            "extra": {
                "source_file": source_file,
                "schemaVersion": raw_project.get("schemaVersion"),
            },
        }

        # Extract paths
        paths = raw_project.get("paths", {})
        if paths:
            normalized["root_path"] = paths.get("root")
            normalized["extra"]["paths"] = paths
        else:
            normalized["root_path"] = None

        # Extract counts
        counts = raw_project.get("counts", {})
        if counts:
            normalized["initial_images"] = counts.get("initialImages")
            normalized["final_images"] = counts.get("finalImages")
            normalized["extra"]["counts"] = counts
        else:
            normalized["initial_images"] = None
            normalized["final_images"] = None

        # Extract metrics if present
        metrics = raw_project.get("metrics", {})
        if metrics:
            normalized["extra"]["metrics"] = metrics

        # Extract steps if present
        steps = raw_project.get("steps", [])
        if steps:
            normalized["extra"]["steps"] = steps

        # Other fields
        if "notes" in raw_project:
            normalized["extra"]["notes"] = raw_project["notes"]

        if "removeFileOnFinish" in raw_project:
            normalized["extra"]["removeFileOnFinish"] = raw_project[
                "removeFileOnFinish"
            ]

        return normalized

    except Exception as e:
        print(f"  ⚠️  Error normalizing {source_file}: {e}")
        return None


def main():
    """Main entry point."""
    print("Extracting projects...")

    if not PROJECTS_DIR.exists():
        print(f"Projects directory not found: {PROJECTS_DIR}")
        return

    # Collect all project manifests
    project_files = list(PROJECTS_DIR.glob("*.project.json"))
    print(f"Found {len(project_files)} project files")

    # Extract projects
    projects = []
    seen_project_ids = set()
    duplicate_count = 0

    for project_file in sorted(project_files):
        print(f"  Processing {project_file.name}...")

        try:
            with open(project_file, encoding="utf-8") as f:
                raw_project = json.load(f)
        except Exception as e:
            print(f"    ⚠️  Error reading file: {e}")
            continue

        normalized = normalize_project(raw_project, project_file.name)
        if not normalized:
            continue

        project_id = normalized["project_id"]

        # Dedupe (keep first occurrence)
        if project_id in seen_project_ids:
            print(f"    ⚠️  Duplicate project ID: {project_id}")
            duplicate_count += 1
            continue

        seen_project_ids.add(project_id)
        projects.append(normalized)

    print(
        f"\nExtracted {len(projects)} unique projects ({duplicate_count} duplicates skipped)"
    )

    # Write output (single file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "projects.jsonl"

    with open(output_file, "w") as f:
        for project in sorted(projects, key=lambda p: p["project_id"]):
            f.write(json.dumps(project) + "\n")

    print(f"\n✅ Done! {len(projects)} projects written to {output_file}")

    # Show sample
    if projects:
        print("\nSample projects (first 3):")
        for project in projects[:3]:
            status = project["status"]
            images = project.get("initial_images", "?")
            print(
                f"  {project['project_id']}: {project['title']} ({status}, {images} images)"
            )


if __name__ == "__main__":
    main()
