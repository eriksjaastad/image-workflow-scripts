#!/usr/bin/env python3
"""
Update references to archived scripts with their AI replacements
================================================================

Replacements:
- 01_ai_assisted_reviewer.py → 01_ai_assisted_reviewer.py
- 01_desktop_image_selector_crop.py → ARCHIVED (obsolete)
- 02_ai_desktop_multi_crop.py → 02_ai_desktop_multi_crop.py

This updates all documentation and test files to point to the new AI-powered versions.
"""

from pathlib import Path

# Define replacements
REPLACEMENTS = {
    # Full script paths
    "scripts/01_ai_assisted_reviewer.py": "scripts/01_ai_assisted_reviewer.py",
    "scripts/02_ai_desktop_multi_crop.py": "scripts/02_ai_desktop_multi_crop.py",
    # Just filenames
    "01_ai_assisted_reviewer.py": "01_ai_assisted_reviewer.py",
    "02_ai_desktop_multi_crop.py": "02_ai_desktop_multi_crop.py",
    # Command examples
    "python scripts/01_ai_assisted_reviewer.py": "python scripts/01_ai_assisted_reviewer.py",
    "python scripts/02_ai_desktop_multi_crop.py": "python scripts/02_ai_desktop_multi_crop.py",
    # Module imports (if any)
    "from scripts.01_ai_assisted_reviewer": "from scripts.01_ai_assisted_reviewer",
    "import scripts.01_ai_assisted_reviewer": "import scripts.01_ai_assisted_reviewer",
}

# Descriptive replacements
DESCRIPTION_REPLACEMENTS = {
    "Web Image Selector": "AI-Assisted Reviewer",
    "web image selector": "AI-assisted reviewer",
    "Desktop Multi-Crop Tool": "AI Desktop Multi-Crop Tool",
    "desktop multi-crop tool": "AI desktop multi-crop tool",
}


def update_file(file_path: Path) -> tuple[bool, int]:
    """Update a file with the replacements. Returns (changed, num_replacements)"""
    try:
        content = file_path.read_text()
        original_content = content
        replacement_count = 0

        # Apply path/filename replacements
        for old, new in REPLACEMENTS.items():
            if old in content:
                content = content.replace(old, new)
                replacement_count += content.count(new) - original_content.count(new)

        # Apply description replacements (only in markdown files)
        if file_path.suffix == ".md":
            for old, new in DESCRIPTION_REPLACEMENTS.items():
                if old in content:
                    content = content.replace(old, new)
                    replacement_count += 1

        if content != original_content:
            file_path.write_text(content)
            return True, replacement_count

        return False, 0
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False, 0


def main():
    repo_root = Path(__file__).parent.parent.parent

    # Files to update
    docs_dir = repo_root / "Documents"
    scripts_dir = repo_root / "scripts"

    print("🔄 Updating references to archived scripts...\n")

    updated_files = []
    total_replacements = 0

    # Update all markdown files in Documents/
    for md_file in docs_dir.rglob("*.md"):
        changed, count = update_file(md_file)
        if changed:
            updated_files.append((md_file.relative_to(repo_root), count))
            total_replacements += count

    # Update Python files in scripts/ (tests, tools, etc.)
    for py_file in scripts_dir.rglob("*.py"):
        # Skip archived files themselves
        if "archive" in py_file.parts:
            continue

        changed, count = update_file(py_file)
        if changed:
            updated_files.append((py_file.relative_to(repo_root), count))
            total_replacements += count

    # Print results
    if updated_files:
        print(
            f"✅ Updated {len(updated_files)} files ({total_replacements} replacements):\n"
        )
        for file_path, count in updated_files:
            print(f"  • {file_path} ({count} changes)")
    else:
        print("✅ No files needed updating (all references already correct)")

    print("\n" + "=" * 70)
    print("Summary of Changes:")
    print("=" * 70)
    print("• 01_ai_assisted_reviewer.py → 01_ai_assisted_reviewer.py")
    print("• 02_ai_desktop_multi_crop.py → 02_ai_desktop_multi_crop.py")
    print("• Updated all documentation and test references")
    print("=" * 70)


if __name__ == "__main__":
    main()
