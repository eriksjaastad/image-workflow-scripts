#!/usr/bin/env python3
"""
Character Processor - Automated Image Organization & Analysis
=============================================================
Intelligent image organization tool with two primary modes:
1. **Character grouping (default):** Organize by LoRA character names
2. **Demographic analysis (`--analyze`):** Subdivide by ethnicity, age, body type, etc.

🎯 PRIMARY USE CASE:
-------------------
**Step 1:** Initial character separation (emily, ivy, mia)
  python scripts/02_character_processor.py "selected/"

**Step 2:** Subdivide character directories by demographics
  python scripts/02_character_processor.py "selected/emily/" --group-by ethnicity
  python scripts/02_character_processor.py "selected/emily/" --group-by age
  # Note: Hierarchical grouping (ethnicity,age) is not yet implemented

USAGE:
------
  # Full demographic breakdown (no file moves)
  python scripts/02_character_processor.py "selected/emily/" --analyze
  
  # Show files without ethnicity detection
  python scripts/02_character_processor.py "_indian/" --analyze --show-missing

  # Default: Character LoRA names (emily, ivy) + prompt fallback for remaining files
  python scripts/02_character_processor.py "selected/"
  python scripts/02_character_processor.py "_asian (chater 1)/" --dry-run
  
  # 🚀 DEMOGRAPHIC GROUPING - Ignores LoRA data, groups by prompt descriptors!
  python scripts/02_character_processor.py "selected/emily/" --group-by ethnicity    # → latina/, asian/, white/
  python scripts/02_character_processor.py "selected/emily/" --group-by age          # → mid_20s/, early_30s/
  python scripts/02_character_processor.py "selected/emily/" --group-by body_type    # → petite/, curvy/, athletic/
  
  # Advanced usage
  python scripts/02_character_processor.py "directory/" --min-group-size 15 --save-analysis --quiet

FEATURES:
---------
• 🚀 FLEXIBLE GROUPING: Character (default) or demographic (ethnicity, age, body_type, etc.)
• 🔍 COMPREHENSIVE ANALYSIS: `--analyze` shows demographic breakdown without moving files
• ⏸️ HIERARCHICAL GROUPING: (ethnicity,age nested directories - COMING SOON)
• Complete 4-stage pipeline in one command
• Metadata extraction from YAML and caption files (supports stages 1, 2, 3)
• Sequential context inference for edge cases
• Intelligent prompt-based descriptor extraction with stage-aware parsing
• Multi-character directory creation (emily_ivy/, etc.)
• Minimum threshold protection (prevents directory fragmentation)
• Progress tracking and comprehensive statistics
• Dry-run mode for safe testing

PIPELINE STAGES:
----------------
1. Metadata Analysis: Extract character LoRA names (emily, ivy) + prompt data from YAML/caption files
2. Sequential Context: Infer missing characters from chronological neighbors  
3. Prompt Analysis: FALLBACK for remaining files - extract descriptors (configurable minimum)
4. Character Grouping: Organize files into directories (supports flexible categories)

GROUPING CATEGORIES:
-------------------
• character (default): Uses LoRA names (emily, ivy) + prompt fallback (latina_petite_young)
• ethnicity: latina, asian, black, white, indian, middle_eastern, etc.
• age: early_20s, mid_20s, late_20s, early_30s, mid_30s, etc.
• age_group: young, mature, teen, milf, college, etc.
• body_type: big_boobs, petite, curvy, tall, athletic, etc.
• hair_color: blonde, brunette, redhead, black_hair, etc.
• scenario: bedroom, office, outdoor, kitchen, beach, etc.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("[!] PyYAML is required. Install with: pip install PyYAML", file=sys.stderr)
    sys.exit(1)

try:
    # Prefer local package import
    from utils.activity_timer import ActivityTimer, FileTracker
except Exception:
    try:
        # Fallback to absolute project-root based import
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from scripts.utils.activity_timer import (  # type: ignore
            ActivityTimer,
            FileTracker,
        )
    except Exception:
        ActivityTimer = None  # type: ignore
        FileTracker = None  # type: ignore

try:
    from utils.companion_file_utils import (
        extract_timestamp_from_filename,
        move_file_with_all_companions,
    )
except Exception:
    try:
        # Fallback to absolute project-root based import
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from scripts.utils.companion_file_utils import (  # type: ignore
            extract_timestamp_from_filename,
            move_file_with_all_companions,
        )
    except Exception as e:
        print(f"[!] Failed to import companion utilities: {e}", file=sys.stderr)
        raise


# ============================================================================
# SHARED EXTRACTION HELPERS - Single source of truth for demographic parsing
# ============================================================================

def extract_ethnicity_from_prompt(prompt_lower: str) -> Optional[str]:
    """
    Extract ethnicity using pure keyword search.
    
    Args:
        prompt_lower: Lowercase prompt text
        
    Returns:
        Ethnicity value with underscores, or None if not found
    """
    ethnicity_keywords = [
        'mixed race', 'middle eastern', 'south asian',  # Multi-word phrases first
        'asian', 'indian', 'black', 'white', 'latina', 'latino',
        'african', 'caucasian', 'hispanic', 'arab'
    ]
    
    import re
    for keyword in ethnicity_keywords:
        if ' ' in keyword:
            # Multi-word phrase: simple substring check
            if keyword in prompt_lower:
                return keyword.replace(' ', '_')
        else:
            # Single word: use word boundaries to avoid false matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
                return keyword.replace(' ', '_')
    
    return None


def extract_age_from_prompt(prompt_lower: str) -> Optional[str]:
    """
    Extract age using regex patterns.
    
    Args:
        prompt_lower: Lowercase prompt text
        
    Returns:
        Age value with underscores, or None if not found
    """
    import re
    
    # Pattern 1: "40 years old", "25 year old"
    age_match = re.search(r'(\d+)\s+years?\s+old', prompt_lower)
    if age_match:
        return f"{age_match.group(1)}_years_old"
    
    # Pattern 2: "in her mid 30s", "in his early 40s"
    age_match = re.search(r'in\s+(?:her|his)\s+(early|mid|late)?\s*(\d+)s', prompt_lower)
    if age_match:
        descriptor = age_match.group(1) if age_match.group(1) else ''
        decade = age_match.group(2)
        if descriptor:
            return f"in_her_{descriptor}_{decade}s"
        else:
            return f"in_her_{decade}s"
    
    # Pattern 3: "mid 20s", "early 30s" (without "in her/his")
    age_match = re.search(r'\b(early|mid|late)\s+(\d+)s\b', prompt_lower)
    if age_match:
        descriptor = age_match.group(1)
        decade = age_match.group(2)
        return f"{descriptor}_{decade}s"
    
    return None


def extract_body_type_from_prompt(prompt_lower: str) -> Optional[str]:
    """
    Extract body type using keyword search on entire prompt.
    
    Args:
        prompt_lower: Lowercase prompt text
        
    Returns:
        Body type value with underscores, or None if not found
    """
    import re
    
    # Multi-word phrases first (more specific), then single words
    body_keywords = [
        'big boobs', 'big tits',  # Multi-word phrases
        'petite', 'small', 'tiny', 'short',
        'tall', 'amazon', 'statuesque',
        'curvy', 'thick', 'voluptuous', 'busty',
        'slim', 'skinny', 'thin', 'lean', 'athletic', 'fit', 'toned', 'slender'
    ]
    
    for keyword in body_keywords:
        if ' ' in keyword:
            # Multi-word phrase: simple substring check
            if keyword in prompt_lower:
                return keyword.replace(' ', '_')
        else:
            # Single word: use word boundaries to avoid false matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
                return keyword
    
    return None


def extract_hair_color_from_prompt(prompt_lower: str) -> Optional[str]:
    """
    Extract hair color using keyword search on entire prompt.
    
    Args:
        prompt_lower: Lowercase prompt text
        
    Returns:
        Hair color value with underscores, or None if not found
    """
    import re
    
    # Multi-word phrases first, then single words
    hair_keywords = [
        'black hair', 'brown hair', 'blonde hair', 'red hair',
        'silver hair', 'gray hair', 'grey hair',
        'pink hair', 'blue hair', 'purple hair', 'green hair',
        'blonde', 'brunette', 'redhead',
        'platinum', 'auburn', 'ginger'
    ]
    
    for keyword in hair_keywords:
        if ' ' in keyword:
            # Multi-word phrase: simple substring check
            if keyword in prompt_lower:
                return keyword.replace(' ', '_')
        else:
            # Single word: use word boundaries to avoid false matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
                return keyword
    
    return None


# ============================================================================
# STAGE 1: YAML ANALYSIS
# ============================================================================

def extract_character_name(lora_path: str) -> str:
    """Extract clean character name from LoRA path."""
    if not lora_path:
        return "unknown"
    
    # Remove file extension
    name = Path(lora_path).stem
    
    # Remove version numbers and common prefixes/suffixes
    name = re.sub(r'^v\d+_\d+_\d+__', '', name)  # Remove v1_1_1__ prefix
    name = re.sub(r'_v\d+$', '', name)           # Remove _v1 suffix
    name = re.sub(r'\.safetensors$', '', name)   # Remove extension if still there
    
    # Clean up underscores and make lowercase
    name = name.replace('_', ' ').strip().lower()
    
    return name if name else "unknown"


def extract_characters_from_prompt(prompt: str) -> List[str]:
    """Extract character names from prompt using @character syntax."""
    if not prompt:
        return []
    
    # Find all @character references
    character_matches = re.findall(r'@(\w+)', prompt)
    return [char.lower() for char in character_matches]


def extract_descriptive_character_from_prompt(prompt: str, group_by: str = "character", stage: str = "unknown") -> Optional[str]:
    """
    Extract descriptive character information from prompt text as fallback.
    
    Looks for physical descriptors like ethnicity, body type, hair color, etc.
    Uses dedicated extraction functions for accuracy.
    
    Args:
        prompt: The prompt text to analyze
        group_by: Category to extract (character, ethnicity, body_type, age, hair_color, etc.)
        stage: YAML stage (generation, upscale, enhance) for context
        
    Returns:
        Descriptive character name or None if no clear descriptors found
    """
    if not prompt:
        return None
    
    # Convert to lowercase for analysis
    prompt_lower = prompt.lower()
    
    # Use dedicated extraction helpers for consistent results
    ethnicity_value = extract_ethnicity_from_prompt(prompt_lower)
    age_value = extract_age_from_prompt(prompt_lower)
    body_type_value = extract_body_type_from_prompt(prompt_lower)
    hair_color_value = extract_hair_color_from_prompt(prompt_lower)
    
    # Handle different grouping modes
    if group_by == "ethnicity":
        # Clean up ethnicity value (remove @ symbols, digits, parentheses)
        if ethnicity_value and ('@' in ethnicity_value or '(' in ethnicity_value or ')' in ethnicity_value or any(c.isdigit() for c in ethnicity_value)):
            return None
        return ethnicity_value
    
    elif group_by == "age":
        return age_value
    
    elif group_by == "body_type":
        return body_type_value
    
    elif group_by == "hair_color":
        return hair_color_value
    
    elif group_by == "character":
        # Original character mode: combine ethnicity + body + age
        found_descriptors = []
        
        # Use extracted values
        if ethnicity_value:
            found_descriptors.append(ethnicity_value)
        if body_type_value:
            found_descriptors.append(body_type_value)
        if age_value:
            found_descriptors.append(age_value)
        
        return '_'.join(found_descriptors) if found_descriptors else None
    
    else:
        # For age_group, use age extraction as fallback
        if group_by == 'age_group' and age_value:
            return age_value
        
        # Unknown category
        return None


def analyze_prompts_for_characters(character_mapping: Dict, min_threshold: int = 20, group_by: str = "character") -> Dict[str, str]:
    """
    Analyze prompts for files without character data and group by descriptive characteristics.
    Only creates character groups that meet the minimum threshold.
    
    Args:
        character_mapping: Dictionary of filename -> character info
        min_threshold: Minimum number of files needed to create a character group
        
    Returns:
        Dictionary mapping filenames to prompt-based character names
    """
    # Find files without character data
    files_without_character = []
    for filename, char_info in character_mapping.items():
        if not char_info.get('has_character', False):
            files_without_character.append((filename, char_info))
    
    if len(files_without_character) < min_threshold:
        return {}  # Not enough files to warrant prompt analysis
    
    print(f"[*] Analyzing prompts for {len(files_without_character)} files without character data...")
    
    # First pass: collect all potential prompt-based characters
    prompt_character_counts = {}
    file_to_prompt_char = {}
    
    for filename, char_info in files_without_character:
        prompt = char_info.get('prompt', '')
        stage = char_info.get('stage', 'unknown')
        prompt_char = extract_descriptive_character_from_prompt(prompt, group_by, stage)
        
        if prompt_char:
            prompt_character_counts[prompt_char] = prompt_character_counts.get(prompt_char, 0) + 1
            file_to_prompt_char[filename] = prompt_char
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"    • {filename}: '{prompt_preview}' → {prompt_char}")
        else:
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"    • {filename}: '{prompt_preview}' → NO MATCH")
    
    print("\n[*] Prompt character counts:")
    for char, count in prompt_character_counts.items():
        print(f"    • {char}: {count} files")
    
    # Second pass: only keep characters that meet threshold
    viable_characters = {char: count for char, count in prompt_character_counts.items() if count >= min_threshold}
    
    if not viable_characters:
        print(f"[*] No prompt-based character groups meet minimum threshold of {min_threshold} files")
        return {}
    
    print(f"[*] Found {len(viable_characters)} viable prompt-based character groups:")
    for char, count in viable_characters.items():
        print(f"    • {char}: {count} files")
    
    # Third pass: assign only viable characters
    final_assignments = {}
    for filename, prompt_char in file_to_prompt_char.items():
        if prompt_char in viable_characters:
            final_assignments[filename] = prompt_char
    
    return final_assignments


def parse_caption_file(caption_path: Path) -> Optional[Dict]:
    """Parse caption file and extract character information."""
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        if not prompt:
            return None
        
        # Extract character information from prompt
        prompt_characters = extract_characters_from_prompt(prompt)
        primary_character = prompt_characters[0] if prompt_characters else None
        character_name = primary_character if primary_character else "unknown"
        
        # Determine corresponding PNG filename
        png_filename = caption_path.stem + '.png'
        
        # Determine if we have character data
        has_character = bool(prompt_characters)
        
        return {
            'character_lora': None,  # Caption files don't have LoRA data
            'character_name': character_name,
            'primary_character': primary_character,
            'all_characters': prompt_characters,
            'character_count': len(prompt_characters),
            'stage': 'unknown',  # Caption files don't have stage info
            'yaml_file': caption_path.name,  # Keep this for compatibility
            'png_filename': png_filename,
            'has_character': has_character,
            'has_face_detected': True,  # Assume true for caption files
            'is_multi_character': len(prompt_characters) > 1,
            'prompt': prompt  # Full prompt text for fallback analysis
        }
        
    except Exception as e:
        print(f"[!] Error parsing {caption_path}: {e}")
        return None


def parse_yaml_file(yaml_path: Path) -> Optional[Dict]:
    """Parse YAML file and extract character information."""
    try:
        # Custom YAML loader to handle Python-specific tags
        class CustomLoader(yaml.SafeLoader):
            pass
        
        def tuple_constructor(loader, node):
            """Handle !!python/tuple tags"""
            return tuple(loader.construct_sequence(node))
        
        def object_constructor(loader, node):
            """Handle !!python/object tags (like PIL Image)"""
            return "PIL Image"  # Just return a placeholder string
        
        # Register custom constructors
        CustomLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
        CustomLoader.add_constructor('tag:yaml.org,2002:python/object', object_constructor)
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=CustomLoader)
        
        if not isinstance(data, dict):
            return None
        
        # Extract character information from multiple sources
        character_lora = None
        character_name = "unknown"
        all_characters = []
        primary_character = None
        stage = data.get('stage', 'unknown')
        has_face_detected = True
        
        # Check for "no face detected" status
        if data.get('status') == 'no_face_detected':
            has_face_detected = False
        
        # Method 1: Look for character LoRA in loras.character
        if 'loras' in data and isinstance(data['loras'], dict):
            if 'character' in data['loras']:
                char_info = data['loras']['character']
                if isinstance(char_info, dict) and 'path' in char_info:
                    character_lora = char_info['path']
                    character_name = extract_character_name(character_lora)
                    primary_character = character_name
                    all_characters = [character_name]
        
        # Method 2: Look for characters in prompt (for face_swapped files)
        if 'prompt' in data and isinstance(data['prompt'], str):
            prompt_characters = extract_characters_from_prompt(data['prompt'])
            if prompt_characters:
                all_characters.extend(prompt_characters)
                # Remove duplicates while preserving order
                all_characters = list(dict.fromkeys(all_characters))
                
                # If we don't have a primary character from LoRA, use first from prompt
                if not primary_character and prompt_characters:
                    primary_character = prompt_characters[0]
                    character_name = primary_character
        
        # Determine corresponding PNG filename
        png_filename = None
        if 'image_filename' in data:
            png_filename = data['image_filename']
        else:
            # Derive from YAML filename
            png_filename = yaml_path.stem + '.png'
        
        # Determine if we have character data
        has_character = bool(character_lora or all_characters)
        
        return {
            'character_lora': character_lora,
            'character_name': character_name if has_character else "unknown",
            'primary_character': primary_character,
            'all_characters': all_characters,
            'character_count': len(all_characters),
            'stage': stage,
            'yaml_file': yaml_path.name,
            'png_filename': png_filename,
            'has_character': has_character,
            'has_face_detected': has_face_detected,
            'is_multi_character': len(all_characters) > 1,
            'prompt': data.get('prompt', '')  # Extract prompt for fallback analysis
        }
        
    except Exception as e:
        print(f"[!] Error parsing {yaml_path}: {e}")
        return None


def parse_metadata_file(metadata_path: Path) -> Optional[Dict]:
    """Parse metadata file (YAML or caption) and extract character information."""
    if metadata_path.suffix.lower() == '.yaml':
        return parse_yaml_file(metadata_path)
    elif metadata_path.suffix.lower() == '.caption':
        return parse_caption_file(metadata_path)
    else:
        print(f"[!] Unsupported metadata file type: {metadata_path}")
        return None


def analyze_yaml(directory: str, output_file: Optional[str] = None, quiet: bool = False) -> Dict:
    """
    Stage 1: Analyze metadata files (YAML and caption) to extract character information.
    
    Args:
        directory: Directory to scan for metadata files
        output_file: Optional JSON output file path
        quiet: Suppress progress output
        
    Returns:
        Dictionary with character mapping and metadata
    """
    directory_path = Path(directory).resolve()
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Support both YAML and caption files
    yaml_files = list(directory_path.rglob('*.yaml'))
    caption_files = list(directory_path.rglob('*.caption'))
    metadata_files = yaml_files + caption_files
    total_files = len(metadata_files)
    
    if not quiet:
        print(f"🔍 Analyzing metadata files in: {directory_path}")
        print(f"[*] Found {len(yaml_files)} YAML files and {len(caption_files)} caption files")
        print(f"[*] Total {total_files} metadata files to analyze...")
    
    character_mapping = {}
    processing_stats = {
        'total_files': total_files,
        'processed': 0,
        'errors': 0,
        'no_character': 0,
        'characters_found': set(),
        'character_counts': {},
        'stages_found': set()
    }
    
    start_time = time.time()
    
    for i, metadata_path in enumerate(metadata_files):
        if not quiet and i % 100 == 0:
            percent = (i / total_files) * 100 if total_files > 0 else 0
            print(f"[*] Processing... {i}/{total_files} ({percent:.1f}%)")
        
        result = parse_metadata_file(metadata_path)
        
        if result is None:
            processing_stats['errors'] += 1
            continue
        
        processing_stats['processed'] += 1
        processing_stats['stages_found'].add(result['stage'])
        
        if result['has_character']:
            # Count all characters mentioned
            for char_name in result['all_characters']:
                processing_stats['characters_found'].add(char_name)
                processing_stats['character_counts'][char_name] = processing_stats['character_counts'].get(char_name, 0) + 1
        else:
            processing_stats['no_character'] += 1
        
        # Store mapping by PNG filename
        png_filename = result['png_filename']
        character_mapping[png_filename] = {
            'character_lora': result['character_lora'],
            'character_name': result['character_name'],
            'primary_character': result['primary_character'],
            'all_characters': result['all_characters'],
            'character_count': result['character_count'],
            'stage': result['stage'],
            'yaml_file': result['yaml_file'],
            'has_character': result['has_character'],
            'has_face_detected': result['has_face_detected'],
            'is_multi_character': result['is_multi_character'],
            'prompt': result['prompt']  # Include prompt for fallback analysis
        }
    
    elapsed_time = time.time() - start_time
    processing_stats['processing_time'] = elapsed_time
    processing_stats['characters_found'] = sorted(list(processing_stats['characters_found']))
    processing_stats['stages_found'] = sorted(list(processing_stats['stages_found']))
    
    # Create output data structure
    analysis_data = {
        'metadata': {
            'source_directory': str(directory_path),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files_analyzed': processing_stats['processed'],
            'processing_time_seconds': processing_stats['processing_time']
        },
        'summary': {
            'total_images': len(character_mapping),
            'characters_found': processing_stats['characters_found'],
            'character_counts': processing_stats['character_counts'],
            'stages_found': processing_stats['stages_found'],
            'files_without_character': processing_stats['no_character'],
            'processing_errors': processing_stats['errors']
        },
        'character_mapping': character_mapping
    }
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            if not quiet:
                print(f"✅ Analysis saved to: {output_path}")
        except Exception as e:
            print(f"[!] Error writing output file: {e}")
    
    # Print summary
    if not quiet:
        print("\n📊 YAML ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total YAML files: {processing_stats['total_files']}")
        print(f"Successfully processed: {processing_stats['processed']}")
        print(f"Errors encountered: {processing_stats['errors']}")
        print(f"Files without character: {processing_stats['no_character']}")
        print(f"Processing time: {processing_stats['processing_time']:.2f} seconds")
        
        print(f"\n🎭 CHARACTERS FOUND ({len(processing_stats['characters_found'])} total):")
        for char in processing_stats['characters_found']:
            count = processing_stats['character_counts'].get(char, 0)
            print(f"  • {char}: {count} images")
        
        print("\n🎨 STAGES FOUND:")
        for stage in processing_stats['stages_found']:
            print(f"  • {stage}")
        
        if processing_stats['character_counts']:
            most_common = max(processing_stats['character_counts'].items(), key=lambda x: x[1])
            print(f"\n🏆 Most common character: {most_common[0]} ({most_common[1]} images)")
    
    return analysis_data


# ============================================================================
# STAGE 2: SEQUENTIAL CONTEXT ANALYSIS
# ============================================================================

def find_sequential_neighbors(target_file: str, all_files: List[str], window_size: int = 5) -> Tuple[List[str], List[str]]:
    """Find files before and after target file in sequential order."""
    # Sort files by timestamp
    timestamped_files = []
    for filename in all_files:
        timestamp = extract_timestamp_from_filename(filename)
        if timestamp:
            timestamped_files.append((timestamp, filename))
    
    timestamped_files.sort(key=lambda x: x[0])
    
    # Find target file position
    target_pos = None
    for i, (timestamp, filename) in enumerate(timestamped_files):
        if filename == target_file:
            target_pos = i
            break
    
    if target_pos is None:
        return [], []
    
    # Get neighbors
    before_files = []
    after_files = []
    
    # Files before
    for i in range(max(0, target_pos - window_size), target_pos):
        before_files.append(timestamped_files[i][1])
    
    # Files after
    for i in range(target_pos + 1, min(len(timestamped_files), target_pos + window_size + 1)):
        after_files.append(timestamped_files[i][1])
    
    return before_files, after_files


def infer_character_from_context(target_file: str, character_mapping: Dict, all_files: List[str]) -> Optional[str]:
    """Infer character for target file based on sequential neighbors."""
    before_files, after_files = find_sequential_neighbors(target_file, all_files)
    
    # Collect characters from neighbors
    neighbor_characters = []
    
    # Check before files
    for filename in before_files:
        if filename in character_mapping:
            file_data = character_mapping[filename]
            if file_data.get('has_character') and file_data.get('primary_character'):
                neighbor_characters.append(file_data['primary_character'])
    
    # Check after files
    for filename in after_files:
        if filename in character_mapping:
            file_data = character_mapping[filename]
            if file_data.get('has_character') and file_data.get('primary_character'):
                neighbor_characters.append(file_data['primary_character'])
    
    if not neighbor_characters:
        return None
    
    # Find most common character in neighborhood
    character_counts = Counter(neighbor_characters)
    most_common = character_counts.most_common(1)
    
    if most_common:
        return most_common[0][0]
    
    return None


def add_sequential_context(analysis_data: Dict, quiet: bool = False) -> Dict:
    """
    Stage 2: Add sequential context inference to analysis data.
    
    Args:
        analysis_data: Output from analyze_yaml()
        quiet: Suppress progress output
        
    Returns:
        Enhanced analysis data with inferred characters
    """
    character_mapping = analysis_data['character_mapping']
    all_files = list(character_mapping.keys())
    
    # Find files without character data
    no_character_files = []
    for filename, file_data in character_mapping.items():
        if not file_data.get('has_character', False):
            no_character_files.append(filename)
    
    if not quiet:
        print("🧠 Sequential Context Analysis")
        print(f"[*] Found {len(no_character_files)} files without character data")
        print("[*] Analyzing sequential context...")
    
    # Analyze each file
    inferences_made = 0
    inference_details = []
    
    for filename in no_character_files:
        inferred_character = infer_character_from_context(filename, character_mapping, all_files)
        
        if inferred_character:
            # Update the file data with inference
            character_mapping[filename]['inferred_character'] = inferred_character
            character_mapping[filename]['inference_method'] = 'sequential_context'
            character_mapping[filename]['has_inferred_character'] = True
            
            inferences_made += 1
            inference_details.append({
                'filename': filename,
                'inferred_character': inferred_character,
                'method': 'sequential_context'
            })
            
            if not quiet:
                print(f"[*] {filename} → inferred as '{inferred_character}'")
        else:
            character_mapping[filename]['has_inferred_character'] = False
            if not quiet:
                print(f"[*] {filename} → no clear context")
    
    # Update analysis data
    analysis_data['inference_summary'] = {
        'total_files_without_character': len(no_character_files),
        'successful_inferences': inferences_made,
        'failed_inferences': len(no_character_files) - inferences_made,
        'inference_details': inference_details
    }
    
    if not quiet:
        print("\n📊 SEQUENTIAL CONTEXT SUMMARY")
        print(f"{'='*50}")
        print(f"Files without character data: {len(no_character_files)}")
        print(f"Successful inferences: {inferences_made}")
        print(f"Failed inferences: {len(no_character_files) - inferences_made}")
        
        if inference_details:
            print("\n🎯 INFERRED CHARACTERS:")
            character_counts = Counter([detail['inferred_character'] for detail in inference_details])
            for char, count in character_counts.most_common():
                print(f"  • {char}: {count} files")
    
    return analysis_data


# ============================================================================
# PREVIEW & CONFIRMATION
# ============================================================================

def preview_grouping_plan(character_mapping: Dict, source_directory: str, group_by: str = "character") -> bool:
    """
    Show preview of what will happen and get user confirmation.
    
    Args:
        character_mapping: Dictionary of filename -> character info
        source_directory: Directory containing the files
        group_by: Category being grouped by
        
    Returns:
        True if user confirms, False if user cancels
    """
    # Count files per group
    group_counts = {}
    files_staying_in_root = 0
    
    for filename, char_info in character_mapping.items():
        # Use same logic as FileMover
        if group_by == "character":
            character = char_info.get('character_name')
            has_character = char_info.get('has_character', False)
            inferred_character = char_info.get('inferred_character')
            
            if not has_character and not inferred_character:
                files_staying_in_root += 1
                continue
            
            # Use inferred character if no direct character data
            if not has_character and inferred_character:
                character = inferred_character
        else:
            # Demographic grouping
            character = char_info.get('character_name')
            has_character = bool(character)
            
            if not character:
                files_staying_in_root += 1
                continue
        
        # Handle multi-character (only for character mode)
        all_characters = char_info.get('all_characters', [])
        is_multi_character = char_info.get('is_multi_character', False) and group_by == "character"
        
        if is_multi_character and len(all_characters) > 1:
            multi_char_name = '_'.join(sorted(all_characters))
            group_counts[multi_char_name] = group_counts.get(multi_char_name, 0) + 1
        else:
            group_counts[character] = group_counts.get(character, 0) + 1
    
    # Show preview
    category_label = group_by.replace('_', ' ').title()
    source_dir_name = Path(source_directory).name
    
    print("\n" + "="*70)
    print(f"📊 PREVIEW: Grouping by {category_label}")
    print("="*70)
    
    if group_counts:
        print("\n✅ Will create directories and move images:")
        for group_name, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"   • {group_name}/ → {count} images")
        print(f"\n   Total images to move: {sum(group_counts.values())}")
    else:
        print("\n⚠️  No images meet grouping criteria")
    
    if files_staying_in_root > 0:
        print(f"\n📁 Will stay in {source_dir_name}/:")
        print(f"   • {files_staying_in_root} images (no {group_by} detected or below threshold)")
    
    print("\n" + "="*70)
    
    # Get user confirmation
    try:
        response = input("Proceed with file moves? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    except (KeyboardInterrupt, EOFError):
        print("\n\n[!] Operation cancelled by user")
        return False


# ============================================================================
# STAGE 3: CHARACTER GROUPING
# ============================================================================

def move_file_pair(png_path: Path, yaml_path: Path, target_dir: Path, dry_run: bool = False, tracker: Optional = None) -> bool:
    """Move PNG and ALL its companion files to target directory."""
    try:
        if not dry_run:
            # Use wildcard logic to move PNG and ALL companion files
            if move_file_with_all_companions:
                moved_files = move_file_with_all_companions(png_path, target_dir, dry_run=False)
                
                # Track file operations
                if tracker:
                    for moved_file in moved_files:
                        tracker.track_file_operation('move', str(png_path.parent / moved_file), str(target_dir / moved_file))
            else:
                # Fallback to old logic if companion utilities not available
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Move PNG file (skip if target already exists)
                png_target = target_dir / png_path.name
                if not png_target.exists():
                    png_path.rename(png_target)
                else:
                    print(f"⚠️  SKIPPING: {png_path.name} (already exists in destination)")
                
                # Move YAML file if it exists (skip if target already exists)
                if yaml_path.exists():
                    yaml_target = target_dir / yaml_path.name
                    if not yaml_target.exists():
                        yaml_path.rename(yaml_target)
                    else:
                        print(f"⚠️  SKIPPING: {yaml_path.name} (already exists in destination)")
                
                # Track file operations
                if tracker:
                    tracker.track_file_operation('move', str(png_path), str(png_target))
                    if yaml_path.exists():
                        tracker.track_file_operation('move', str(yaml_path), str(yaml_target))
        
        return True
        
    except Exception as e:
        print(f"[!] Error moving {png_path.name}: {e}")
        return False


def create_character_directories(source_dir: Path, characters: List[str], dry_run: bool = False) -> Dict[str, Path]:
    """Prepare character directory mapping (lazy creation - only create when needed)."""
    char_dirs = {}
    
    for char in characters:
        # Clean character name for directory
        clean_name = char.replace(' ', '_').replace('/', '_')
        char_dir = source_dir / clean_name
        char_dirs[char] = char_dir
        
        # Note: We don't create directories here anymore - they're created lazily when first used
    
    return char_dirs


def group_by_category(analysis_data: Dict, source_directory: str, dry_run: bool = False, quiet: bool = False, group_by: str = "character") -> Dict:
    """
    Stage 3: Group files by specified category (character, ethnicity, age, etc.).
    
    Args:
        analysis_data: Enhanced analysis data from previous stages
        source_directory: Directory containing the files to organize
        dry_run: Preview mode - don't actually move files
        quiet: Suppress progress output
        group_by: Category to group by (character, ethnicity, age, body_type, etc.)
        
    Returns:
        Dictionary with grouping statistics
    """
    source_dir = Path(source_directory).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    character_mapping = analysis_data['character_mapping']
    
    # Extract unique categories based on group_by parameter
    categories = set()
    for file_data in character_mapping.values():
        category_name = file_data.get('character_name')  # This field holds the extracted category value
        if category_name:
            categories.add(category_name)
    
    categories = sorted(list(categories))
    
    if not quiet:
        category_label = group_by.replace('_', ' ').title()
        print(f"🎭 Grouping by {category_label} - Intelligent Image Organization")
        print(f"📁 Source: {source_dir}")
        print(f"[*] Found {len(categories)} {group_by} groups: {', '.join(categories)}")
    
    # Create category directories
    char_dirs = create_character_directories(source_dir, categories, dry_run)
    
    # Initialize activity tracking
    tracker = None
    if ActivityTimer and not dry_run:
        tracker = FileTracker()
    
    # Statistics tracking
    stats = {
        'total_images': len(character_mapping),  # Each key is one PNG image
        'moved_successfully': 0,
        'move_errors': 0,
        'missing_images': 0,
        'skipped_unknown': 0,
        'character_counts': {cat: 0 for cat in categories},
        'processing_time': 0
    }
    
    start_time = time.time()
    processed = 0
    
    if not quiet:
        print(f"\n[*] Processing {stats['total_images']} images...")
    
    for png_filename, char_info in character_mapping.items():
        processed += 1
        
        # Progress update every 100 images
        if not quiet and processed % 100 == 0:
            percent = (processed / stats['total_images']) * 100
            print(f"[*] Progress: {processed}/{stats['total_images']} ({percent:.1f}%)")
        
        # Determine character and target directory
        # When grouping by category (not character), use inferred category value instead of LoRA name
        if group_by == "character":
            character = char_info.get('character_name')
            has_character = char_info.get('has_character', False)
            inferred_character = char_info.get('inferred_character')
        else:
            # Grouping by demographic category (ethnicity, age, body_type, etc.)
            # Use character_name (Stage 2.5 writes category values here)
            character = char_info.get('character_name')
            has_character = bool(character)  # Has character if we have an inferred category
            inferred_character = character  # Same value for category grouping
        
        # Handle files without character/category data
        # For category grouping: need inferred_character (body_type, ethnicity, etc.)
        # For character grouping: need either has_character (LoRA) or inferred_character (fallback)
        if group_by == "character":
            # Character grouping: skip if no LoRA data AND no inferred character
            if not has_character and not inferred_character:
                if not dry_run and not quiet:
                    print(f"[*] Skipping {png_filename} (no character data)")
                elif dry_run and not quiet:
                    print(f"[DRY RUN] Would skip: {png_filename} (no character data)")
                stats['skipped_unknown'] += 1
                continue
        else:
            # Category grouping: skip if no inferred category found
            if not character:
                if not dry_run and not quiet:
                    print(f"[*] Skipping {png_filename} (no {group_by} data)")
                elif dry_run and not quiet:
                    print(f"[DRY RUN] Would skip: {png_filename} (no {group_by} data)")
                stats['skipped_unknown'] += 1
                continue
        
        # Determine target directory
        # When grouping by category (not character), always use single category value
        # Multi-character logic only applies when grouping by LoRA character names
        all_characters = char_info.get('all_characters', [])
        is_multi_character = char_info.get('is_multi_character', False) and group_by == "character"
        
        # Use inferred character if no direct character data (only for character grouping)
        if group_by == "character" and not char_info.get('has_character', False) and inferred_character:
            character = inferred_character
            all_characters = [inferred_character]
            is_multi_character = False
        
        # Multi-character logic: create combined directory name (only when group_by=="character")
        if is_multi_character and len(all_characters) > 1 and group_by == "character":
            # Sort characters for consistent naming
            sorted_chars = sorted(all_characters)
            multi_char_name = '_'.join(sorted_chars)
            target_dir = char_dirs.get(multi_char_name)
            
            # If multi-character directory doesn't exist, create it
            if target_dir is None:
                clean_name = multi_char_name.replace(' ', '_').replace('/', '_')
                target_dir = source_dir / clean_name
                char_dirs[multi_char_name] = target_dir
                
                if not dry_run:
                    target_dir.mkdir(exist_ok=True)
                    if not quiet:
                        print(f"[*] Created multi-character directory: {target_dir.name}/")
                elif not quiet:
                    print(f"[DRY RUN] Would create multi-character directory: {target_dir.name}/")
        else:
            # Single character
            target_dir = char_dirs.get(character)
            
            # Create directory if it doesn't exist (lazy creation)
            if target_dir and not target_dir.exists() and not dry_run:
                target_dir.mkdir(exist_ok=True)
                if not quiet:
                    print(f"[*] Created directory: {target_dir.name}/")
        
        # Find source files
        png_path = source_dir / png_filename
        yaml_filename = char_info.get('yaml_file', png_filename.replace('.png', '.yaml'))
        yaml_path = source_dir / yaml_filename
        
        # Check if PNG file exists
        if not png_path.exists():
            if not quiet:
                print(f"[!] Missing PNG image: {png_filename}")
            stats['missing_images'] += 1
            continue
        
        # Show what we're doing
        if dry_run and not quiet:
            print(f"[DRY RUN] Would move: {png_filename} + {yaml_filename} → {target_dir.name}/")
        
        # Move the file pair
        if move_file_pair(png_path, yaml_path, target_dir, dry_run, tracker):
            stats['moved_successfully'] += 1
            # Track by the actual directory used (could be multi-character)
            if is_multi_character and len(all_characters) > 1:
                multi_char_name = '_'.join(sorted(all_characters))
                stats['character_counts'][multi_char_name] = stats['character_counts'].get(multi_char_name, 0) + 1
            else:
                stats['character_counts'][character] = stats['character_counts'].get(character, 0) + 1
        else:
            stats['move_errors'] += 1
    
    stats['processing_time'] = time.time() - start_time
    
    # Print summary
    if not quiet:
        action = "Would move" if dry_run else "Moved"
        
        print("\n📊 CHARACTER GROUPING SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {stats['total_images']}")
        print(f"{action} successfully: {stats['moved_successfully']}")
        print(f"Skipped (no {group_by}): {stats['skipped_unknown']}")
        print(f"Move errors: {stats['move_errors']}")
        print(f"Missing images: {stats['missing_images']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        print(f"\n🎭 IMAGES PER {group_by.upper().replace('_', ' ')}:")
        for char, count in sorted(stats['character_counts'].items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  • {char}: {count} images")
        
        # Show unmoved files in the character breakdown
        if stats['skipped_unknown'] > 0:
            source_dir_name = Path(source_directory).name
            print(f"  • (stayed in {source_dir_name}/): {stats['skipped_unknown']} images")
        
        if stats['moved_successfully'] > 0:
            rate = stats['moved_successfully'] / stats['processing_time']
            print(f"\n⚡ Processing rate: {rate:.1f} images/second")
        
        if dry_run:
            print("\n🧪 DRY RUN COMPLETE - No files were actually moved")
            print("Run without --dry-run to perform the actual grouping")
        else:
            print("\n✅ CHARACTER GROUPING COMPLETE!")
        
        print(f"\n🎯 Successfully organized {stats['moved_successfully']} images by {group_by}!")
    
    return stats


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_directory(directory: str, dry_run: bool = False, save_analysis: bool = False, quiet: bool = False, group_by: str = "character", min_threshold: int = 20) -> Dict:
    """
    Complete character processing pipeline.
    
    Args:
        directory: Directory to process
        dry_run: Preview mode - don't actually move files
        save_analysis: Save intermediate analysis files
        quiet: Suppress progress output
        
    Returns:
        Dictionary with complete processing results
    """
    if not quiet:
        print("🚀 Starting complete character processing pipeline...")
        print(f"📁 Directory: {directory}")
        print(f"🧪 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print("=" * 60)
    
    # Stage 1: YAML Analysis
    analysis_file = None
    if save_analysis:
        dir_name = Path(directory).name.replace(' ', '_').replace('(', '').replace(')', '')
        analysis_file = f"analysis_{dir_name}.json"
    
    analysis_data = analyze_yaml(directory, analysis_file, quiet)
    
    # Stage 2: Sequential Context
    enhanced_data = add_sequential_context(analysis_data, quiet)
    
    # Stage 2.5: Category Extraction (apply group_by to ALL files, not just those without LoRA data)
    if not quiet:
        print("\n=== Stage 2.5: Category Extraction ===")
        print(f"[*] Extracting '{group_by}' from prompts for all files...")
    
    # If group_by is NOT "character", we need to re-extract categories from ALL files
    if group_by != "character":
        category_assignments = {}
        category_counts = {}
        
        # First, clear character_name for ALL files to prevent LoRA fallback
        for filename in enhanced_data['character_mapping']:
            enhanced_data['character_mapping'][filename]['character_name'] = None
            enhanced_data['character_mapping'][filename]['has_character'] = False
        
        for filename, char_info in enhanced_data['character_mapping'].items():
            prompt = char_info.get('prompt', '')
            stage = char_info.get('stage', 'unknown')
            category_value = extract_descriptive_character_from_prompt(prompt, group_by, stage)
            
            if category_value:
                category_assignments[filename] = category_value
                category_counts[category_value] = category_counts.get(category_value, 0) + 1
        
        # Apply min_threshold: only keep categories that meet the minimum
        viable_categories = {cat: count for cat, count in category_counts.items() if count >= min_threshold}
        
        if not viable_categories:
            if not quiet:
                print(f"[*] No {group_by} groups meet minimum threshold of {min_threshold} files")
            enhanced_data['summary']['prompt_analysis'] = {
                'files_assigned': 0,
                'character_groups_created': 0,
                'character_groups': {},
                'files_below_threshold': len(category_assignments)
            }
        else:
            # Apply category assignments ONLY for viable categories
            # Files in non-viable categories will stay in root (not moved)
            files_assigned = 0
            for filename, category_value in category_assignments.items():
                if category_value in viable_categories and filename in enhanced_data['character_mapping']:
                    enhanced_data['character_mapping'][filename]['character_name'] = category_value
                    enhanced_data['character_mapping'][filename]['has_character'] = True
                    enhanced_data['character_mapping'][filename]['source'] = f'prompt_analysis_{group_by}'
                    files_assigned += 1
                # Note: Files NOT in viable_categories keep character_name=None and has_character=False
                # This prevents them from being moved
            
            # Update summary statistics
            files_below_threshold = len(category_assignments) - files_assigned
            enhanced_data['summary']['prompt_analysis'] = {
                'files_assigned': files_assigned,
                'character_groups_created': len(viable_categories),
                'character_groups': viable_categories,
                'files_below_threshold': files_below_threshold
            }
            
            if not quiet:
                print(f"[*] Extracted '{group_by}' for {files_assigned} files → {len(viable_categories)} groups (threshold: {min_threshold})")
                for category, count in sorted(viable_categories.items()):
                    print(f"    • {category}: {count} files")
                if files_below_threshold > 0:
                    print(f"[*] {files_below_threshold} files in groups below threshold (will stay in root directory)")
    else:
        # Original behavior: only process files without LoRA character data
        prompt_assignments = analyze_prompts_for_characters(enhanced_data['character_mapping'], min_threshold=min_threshold, group_by=group_by)
        
        if prompt_assignments:
            # Apply prompt-based character assignments
            for filename, prompt_char in prompt_assignments.items():
                if filename in enhanced_data['character_mapping']:
                    enhanced_data['character_mapping'][filename]['character_name'] = prompt_char
                    enhanced_data['character_mapping'][filename]['has_character'] = True
                    enhanced_data['character_mapping'][filename]['source'] = 'prompt_analysis'
            
            # Update summary statistics
            enhanced_data['summary']['prompt_analysis'] = {
                'files_assigned': len(prompt_assignments),
                'character_groups_created': len(set(prompt_assignments.values())),
                'character_groups': {char: list(prompt_assignments.values()).count(char) 
                                   for char in set(prompt_assignments.values())}
            }
            
            if not quiet:
                print(f"[*] Assigned {len(prompt_assignments)} files to {len(set(prompt_assignments.values()))} prompt-based character groups")
                for char, count in enhanced_data['summary']['prompt_analysis']['character_groups'].items():
                    print(f"    • {char}: {count} files")
        else:
            if not quiet:
                print("[*] No viable prompt-based character groups found")
            enhanced_data['summary']['prompt_analysis'] = {
                'files_assigned': 0,
                'character_groups_created': 0,
                'character_groups': {}
            }
    
    if save_analysis:
        context_file = analysis_file.replace('.json', '_with_context.json') if analysis_file else None
        if context_file:
            try:
                with open(context_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                if not quiet:
                    print(f"✅ Enhanced analysis saved to: {context_file}")
            except Exception as e:
                print(f"[!] Error saving enhanced analysis: {e}")
    
    # Preview and Confirmation (skip for dry-run since it already shows what would happen)
    if not dry_run and not quiet:
        if not preview_grouping_plan(enhanced_data['character_mapping'], directory, group_by):
            if not quiet:
                print("\n[!] Operation cancelled by user")
            return {
                'analysis_data': enhanced_data,
                'grouping_stats': {'cancelled': True},
                'pipeline_summary': {
                    'total_files_analyzed': enhanced_data['summary']['total_images'],
                    'characters_found': len(enhanced_data['summary']['characters_found']),
                    'files_organized': 0,
                    'files_skipped': 0,
                    'success_rate': 0,
                    'cancelled': True
                }
            }
    
    # Stage 3: Character Grouping
    grouping_stats = group_by_category(enhanced_data, directory, dry_run, quiet, group_by)
    
    # Combine results
    results = {
        'analysis_data': enhanced_data,
        'grouping_stats': grouping_stats,
        'pipeline_summary': {
            'total_files_analyzed': enhanced_data['summary']['total_images'],
            'characters_found': len(enhanced_data['summary']['characters_found']),
            'files_organized': grouping_stats['moved_successfully'],
            'files_skipped': grouping_stats['skipped_unknown'],
            'success_rate': (grouping_stats['moved_successfully'] / grouping_stats['total_files']) * 100 if grouping_stats['total_files'] > 0 else 0
        }
    }
    
    if not quiet:
        print("\n🎉 PIPELINE COMPLETE!")
        print(f"📊 Success Rate: {results['pipeline_summary']['success_rate']:.1f}%")
        print(f"🎭 Characters: {results['pipeline_summary']['characters_found']}")
        print(f"📁 Files Organized: {results['pipeline_summary']['files_organized']}")
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete character processing pipeline - YAML analysis, context inference, and file grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory with full pipeline (default: character grouping)
  python scripts/02_character_processor.py "_asian (chater 1)/"
  
  # Group by body type instead of character
  python scripts/02_character_processor.py "selected/" --group-by body_type
  
  # Group by ethnicity
  python scripts/02_character_processor.py "selected/" --group-by ethnicity
  
  # Dry run to preview changes
  python scripts/02_character_processor.py "selected/" --dry-run --group-by age_group
  
  # Save intermediate analysis files
  python scripts/02_character_processor.py "directory/" --save-analysis
        """
    )
    
    parser.add_argument("directory", type=str, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving files")
    parser.add_argument("--save-analysis", action="store_true", help="Save intermediate analysis JSON files")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--group-by", "-g", type=str, default="character",
                        help="Grouping category (default: character). Can combine with comma: ethnicity,age")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Show full demographic breakdown (ignores LoRA data)")
    parser.add_argument("--show-missing", action="store_true",
                        help="Show files without ethnicity detection (use with --analyze)")
    parser.add_argument("--min-group-size", type=int, default=20,
                        help="Minimum files per group for prompt analysis (default: 20)")
    
    args = parser.parse_args()
    
    # Handle --analyze mode (read-only analysis, no file moves)
    if args.analyze:
        from collections import Counter
        
        directory = Path(args.directory).expanduser().resolve()
        print(f"🔍 Analyzing YAML files in: {directory}")
        print("="*70)
        
        yaml_files = list(directory.glob("*.yaml"))
        
        if not yaml_files:
            print("❌ No YAML files found in directory")
            sys.exit(1)
        
        print(f"🔍 Analyzing {len(yaml_files)} YAML files...")
        
        # Counters for each category
        ethnicities = Counter()
        age_groups = Counter()
        body_types = Counter()
        hair_colors = Counter()
        scenarios = Counter()
        face_shapes = Counter()
        physique_types = Counter()
        
        successful = 0
        
        # Custom YAML loader to handle python/tuple tags
        def tuple_constructor(loader, node):
            """Convert YAML tuple tags to regular Python lists."""
            return loader.construct_sequence(node)
        
        yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
        
        # Track files for analysis
        files_without_ethnicity = []
        files_with_errors = []
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if not data:
                    files_with_errors.append((yaml_file, "Empty or null YAML data"))
                    continue
                
                # Extract prompt
                prompt = ''
                if 'prompt' in data:
                    prompt = str(data['prompt'])
                elif 'positive_prompt' in data:
                    prompt = str(data['positive_prompt'])
                
                # If no prompt in YAML, try companion .caption file
                if not prompt:
                    caption_file = yaml_file.with_suffix('.caption')
                    if caption_file.exists():
                        try:
                            with open(caption_file, 'r', encoding='utf-8') as cf:
                                prompt = cf.read().strip()
                        except:
                            pass
                
                if not prompt:
                    files_with_errors.append((yaml_file, "No prompt found in YAML or caption file"))
                    continue
                
                # Clean prompt: YAML may include newlines and extra spaces
                prompt_clean = ' '.join(prompt.split())
                prompt_lower = prompt_clean.lower()
                
                # PURE KEYWORD SEARCH - No index-based parsing!
                # Just search the entire prompt for what we need
                
                # ============================================================
                # ETHNICITY - Use shared extraction helper
                # ============================================================
                found_ethnicity = False
                ethnicity_result = extract_ethnicity_from_prompt(prompt_lower)
                if ethnicity_result:
                    ethnicities[ethnicity_result] += 1
                    found_ethnicity = True
                
                # Track files without ethnicity
                if not found_ethnicity:
                    files_without_ethnicity.append(yaml_file)
                
                # ============================================================
                # AGE - Use shared extraction helper
                # ============================================================
                age_result = extract_age_from_prompt(prompt_lower)
                if age_result:
                    age_groups[age_result] += 1
                
                # ============================================================
                # BODY TYPES - Search for body type keywords
                # ============================================================
                body_keywords = ['petite', 'curvy', 'athletic', 'voluptuous', 'fit', 'toned', 'slim', 'busty', 'slender']
                for keyword in body_keywords:
                    if keyword in prompt_lower:
                        body_types[keyword] += 1
                
                # ============================================================
                # HAIR - Look for hair-related phrases
                # ============================================================
                # Split by comma to get individual phrases
                prompt_parts = [p.strip() for p in prompt_clean.split(',')]
                hair_indicators = ['hair', 'afro', 'bun', 'waves', 'curls', 'ponytail', 'braids']
                for part in prompt_parts:
                    part_lower = part.lower()
                    if any(indicator in part_lower for indicator in hair_indicators):
                        hair_colors[part_lower.strip()] += 1
                        break  # Only count first hair mention
                
                # ============================================================
                # SCENARIOS - Search for scenario keywords
                # ============================================================
                scenario_keywords = ['bedroom', 'bathroom', 'office', 'outdoor', 'kitchen', 'beach', 'pool', 'shower', 'gym', 'selfie']
                for keyword in scenario_keywords:
                    if keyword in prompt_lower:
                        scenarios[keyword] += 1
                
                # ============================================================
                # FACE SHAPES - Search for face shape keywords
                # ============================================================
                face_keywords = ['round face', 'oval face', 'square face', 'heart shaped face', 'diamond face', 'oblong face']
                for keyword in face_keywords:
                    if keyword in prompt_lower:
                        face_shapes[keyword] += 1
                
                # ============================================================
                # PHYSIQUE - Search for physique keywords
                # ============================================================
                physique_keywords = ['voluptuous figure', 'slender figure', 'athletic build', 'petite build', 'curvy figure']
                for keyword in physique_keywords:
                    if keyword in prompt_lower:
                        physique_types[keyword] += 1
                
                successful += 1
            
            except Exception as e:
                files_with_errors.append((yaml_file, str(e)))
                continue
        
        print(f"✅ Successfully analyzed: {successful}/{len(yaml_files)}\n")
        
        # Show files without ethnicity if any
        if files_without_ethnicity:
            print(f"⚠️  Files without ethnicity detected: {len(files_without_ethnicity)}")
            if args.show_missing:
                print("\n📄 Files without ethnicity:")
                for yaml_file in files_without_ethnicity:
                    print(f"   • {Path(yaml_file).name}")
                print()
            else:
                print("   Run with --show-missing to see file list\n")
        
        # Show files with errors if any
        if files_with_errors:
            print(f"❌ Files with errors: {len(files_with_errors)}")
            # Show all files if <= 20, otherwise show first 20
            display_limit = 20 if len(files_with_errors) > 20 else len(files_with_errors)
            for yaml_file, error in files_with_errors[:display_limit]:
                print(f"   • {Path(yaml_file).name}: {error}")
            if len(files_with_errors) > display_limit:
                print(f"   ... and {len(files_with_errors) - display_limit} more")
            print()
        
        # Print detailed breakdown by category
        print("="*70)
        print("📊 DEMOGRAPHIC BREAKDOWN")
        print("="*70)
        
        # Ethnicities
        if ethnicities:
            print("\n🌍 ETHNICITIES:")
            for ethnicity, count in sorted(ethnicities.items(), key=lambda x: -x[1]):
                print(f"   • {ethnicity}: {count} images")
        else:
            print("\n🌍 ETHNICITIES: None detected")
        
        # Ages
        if age_groups:
            print("\n📅 AGE GROUPS:")
            for age, count in sorted(age_groups.items(), key=lambda x: -x[1]):
                print(f"   • {age}: {count} images")
        else:
            print("\n📅 AGE GROUPS: None detected")
        
        # Body types
        if body_types:
            print("\n💪 BODY TYPES:")
            for body_type, count in sorted(body_types.items(), key=lambda x: -x[1]):
                print(f"   • {body_type}: {count} images")
        else:
            print("\n💪 BODY TYPES: None detected")
        
        # Hair colors
        if hair_colors:
            print("\n💇 HAIR COLORS:")
            for hair, count in sorted(list(hair_colors.items())[:20], key=lambda x: -x[1]):  # Top 20
                print(f"   • {hair}: {count} images")
        else:
            print("\n💇 HAIR COLORS: None detected")
        
        # Scenarios
        if scenarios:
            print("\n🎬 SCENARIOS:")
            for scenario, count in sorted(scenarios.items(), key=lambda x: -x[1]):
                print(f"   • {scenario}: {count} images")
        else:
            print("\n🎬 SCENARIOS: None detected")
        
        # Face shapes
        if face_shapes:
            print("\n👤 FACE SHAPES:")
            for face, count in sorted(face_shapes.items(), key=lambda x: -x[1]):
                print(f"   • {face}: {count} images")
        else:
            print("\n👤 FACE SHAPES: None detected")
        
        # Physique types
        if physique_types:
            print("\n🏋️ PHYSIQUE TYPES:")
            for physique, count in sorted(physique_types.items(), key=lambda x: -x[1]):
                print(f"   • {physique}: {count} images")
        else:
            print("\n🏋️ PHYSIQUE TYPES: None detected")
        
        print("\n" + "="*70)
        print("💡 To group by a specific category, run:")
        print(f"   python scripts/02_character_processor.py \"{args.directory}\" --group-by <category>")
        print("\nAvailable categories: ethnicity, age, age_group, body_type, hair_color, scenario")
        print("="*70 + "\n")
        sys.exit(0)
    
    try:
        process_directory(
            directory=args.directory,
            dry_run=args.dry_run,
            save_analysis=args.save_analysis,
            quiet=args.quiet,
            group_by=args.group_by,
            min_threshold=args.min_group_size
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"[!] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
