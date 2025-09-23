#!/usr/bin/env python3
"""
Generate Test Manifest from XXX_CONTENT Directory

This script analyzes the actual XXX_CONTENT directory structure and creates
a comprehensive test manifest that can be used to generate realistic test data
for all our workflow scripts.

Usage:
    python scripts/generate_test_manifest.py XXX_CONTENT/ > tests/test_manifest.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import re

def analyze_directory(content_dir):
    """Analyze directory structure and create test manifest"""
    content_path = Path(content_dir)
    
    if not content_path.exists():
        raise ValueError(f"Directory {content_dir} does not exist")
    
    # Get all PNG files
    png_files = list(content_path.glob("*.png"))
    yaml_files = list(content_path.glob("*.yaml"))
    
    print(f"Found {len(png_files)} PNG files and {len(yaml_files)} YAML files", file=sys.stderr)
    
    # Analyze filename patterns
    stage_patterns = defaultdict(int)
    timestamp_patterns = set()
    
    for png_file in png_files:
        name = png_file.stem
        
        # Extract stage pattern
        if "stage1.5" in name:
            stage_patterns["stage1.5"] += 1
        elif "stage1" in name:
            stage_patterns["stage1"] += 1
        elif "stage2" in name:
            stage_patterns["stage2"] += 1
        elif "stage3" in name:
            stage_patterns["stage3"] += 1
        elif "stage3_enhanced" in name:
            stage_patterns["stage3_enhanced"] += 1
        
        # Extract timestamp pattern (YYYYMMDD_HHMMSS)
        timestamp_match = re.search(r'(\d{8}_\d{6})', name)
        if timestamp_match:
            timestamp_patterns.add(timestamp_match.group(1))
    
    # Create groups by analyzing stage progression
    groups = []
    files_by_timestamp = defaultdict(list)
    
    # Group files by timestamp
    for png_file in png_files:
        name = png_file.stem
        timestamp_match = re.search(r'(\d{8}_\d{6})', name)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            files_by_timestamp[timestamp].append({
                "filename": png_file.name,
                "stage": extract_stage(name),
                "type": extract_type(name)
            })
    
    # Sort files within each timestamp group
    for timestamp, files in files_by_timestamp.items():
        files.sort(key=lambda x: stage_to_number(x["stage"]))
        groups.append({
            "timestamp": timestamp,
            "files": files,
            "group_type": "triplet" if len(files) >= 3 else "pair" if len(files) == 2 else "singleton"
        })
    
    # Create manifest
    manifest = {
        "source_directory": str(content_path),
        "total_files": len(png_files),
        "total_groups": len(groups),
        "stage_distribution": dict(stage_patterns),
        "group_types": {
            "triplets": len([g for g in groups if g["group_type"] == "triplet"]),
            "pairs": len([g for g in groups if g["group_type"] == "pair"]),
            "singletons": len([g for g in groups if g["group_type"] == "singleton"])
        },
        "sample_groups": groups[:10],  # First 10 groups as examples
        "groups": groups  # All groups for test generation
    }
    
    return manifest

def extract_stage(filename):
    """Extract stage from filename"""
    if "stage3_enhanced" in filename:
        return "stage3_enhanced"
    elif "stage3" in filename:
        return "stage3"
    elif "stage1.5" in filename:
        return "stage1.5"
    elif "stage2" in filename:
        return "stage2"
    elif "stage1" in filename:
        return "stage1"
    return "unknown"

def extract_type(filename):
    """Extract generation type from filename"""
    if "generated" in filename:
        return "generated"
    elif "face_swapped" in filename:
        return "face_swapped"
    elif "upscaled" in filename:
        return "upscaled"
    elif "enhanced" in filename:
        return "enhanced"
    return "unknown"

def stage_to_number(stage):
    """Convert stage to number for sorting"""
    stage_map = {
        "stage1": 1,
        "stage1.5": 1.5,
        "stage2": 2,
        "stage3": 3,
        "stage3_enhanced": 3.5
    }
    return stage_map.get(stage, 0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_test_manifest.py XXX_CONTENT/")
        sys.exit(1)
    
    content_dir = sys.argv[1]
    
    try:
        manifest = analyze_directory(content_dir)
        
        # Print summary to stderr
        print(f"\n📊 ANALYSIS SUMMARY:", file=sys.stderr)
        print(f"Total files: {manifest['total_files']}", file=sys.stderr)
        print(f"Total groups: {manifest['total_groups']}", file=sys.stderr)
        print(f"Stage distribution: {manifest['stage_distribution']}", file=sys.stderr)
        print(f"Group types: {manifest['group_types']}", file=sys.stderr)
        print(f"\n✅ Test manifest generated successfully!", file=sys.stderr)
        
        # Output JSON to stdout
        print(json.dumps(manifest, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
