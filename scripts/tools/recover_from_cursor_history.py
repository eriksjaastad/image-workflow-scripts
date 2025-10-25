#!/usr/bin/env python3
"""
Recover files from Cursor's local history (Oct 24-25 work)
"""
import json
import shutil
from pathlib import Path
from urllib.parse import unquote

# Map of Cursor history dirs to their latest file versions
RECOVERIES = {
    # Oct 24 work
    "7587abfe": {
        "resource": "file:///Users/eriksjaastad/projects/Eros%20Mate/scripts/ai/training_utils_v3.py",
        "latest_file": "entries need checking",  # Check entries.json for latest
    },
    "-11ed10a6": {
        "resource": "file:///Users/eriksjaastad/projects/Eros%20Mate/scripts/dashboard/current_project_dashboard.py",
        "latest_file": "XYij.py",  # Latest from our earlier check
    },
    "3ad46207": {
        "resource": "file:///Users/eriksjaastad/projects/Eros%20Mate/scripts/dashboard/templates/current_project.html",
        "latest_file": "entries need checking",
    },
    # Oct 25 - already recovered 04_desktop_multi_crop.py manually
}

def get_latest_file_from_history(history_dir: str) -> str:
    """Get the latest file ID from a history directory's entries.json"""
    history_path = Path.home() / "Library/Application Support/Cursor/User/History" / history_dir
    entries_file = history_path / "entries.json"
    
    if not entries_file.exists():
        return None
        
    with open(entries_file) as f:
        data = json.load(f)
    
    if not data.get('entries'):
        return None
        
    # Last entry is the latest
    latest = data['entries'][-1]
    return latest.get('id')

def recover_file(history_dir: str, resource_url: str, latest_file_id: str = None):
    """Recover a file from Cursor history"""
    history_path = Path.home() / "Library/Application Support/Cursor/User/History" / history_dir
    
    # Get latest file ID if not provided
    if not latest_file_id or latest_file_id == "entries need checking":
        latest_file_id = get_latest_file_from_history(history_dir)
        if not latest_file_id:
            print(f"❌ No entries found in {history_dir}")
            return False
    
    source_file = history_path / latest_file_id
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        return False
    
    # Parse destination from resource URL
    file_path = unquote(resource_url.replace("file://", ""))
    dest_file = Path(file_path)
    
    # Backup existing file if it exists
    if dest_file.exists():
        backup = dest_file.with_suffix(dest_file.suffix + ".backup")
        shutil.copy2(dest_file, backup)
        print(f"  📦 Backed up to {backup.name}")
    
    # Copy from history
    shutil.copy2(source_file, dest_file)
    print(f"✅ Recovered: {dest_file.relative_to(dest_file.parent.parent.parent)}")
    print(f"   From: {history_dir}/{latest_file_id}")
    return True

def main():
    print("🔍 Recovering files from Cursor history (Oct 24-25 work)...\n")
    
    recovered = 0
    for history_dir, info in RECOVERIES.items():
        try:
            if recover_file(history_dir, info['resource'], info.get('latest_file')):
                recovered += 1
            print()
        except Exception as e:
            print(f"❌ Error recovering from {history_dir}: {e}\n")
    
    print(f"\n✅ Recovery complete: {recovered}/{len(RECOVERIES)} files recovered")
    print("\nNote: 04_desktop_multi_crop.py was already recovered manually")

if __name__ == "__main__":
    main()

