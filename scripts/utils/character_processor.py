#!/usr/bin/env python3
"""
Character Processor - Complete Image Organization Pipeline
=========================================================
Combines YAML analysis, sequential context inference, and character grouping
into one streamlined tool for intelligent image organization.

USAGE:
------
  # Default: Character LoRA names (emily, ivy) + prompt fallback for remaining files
  python scripts/utils/character_processor.py "selected/"
  python scripts/utils/character_processor.py "_asian (chater 1)/" --dry-run
  
  # 🚀 FLEXIBLE GROUPING - Controls prompt analysis fallback for files WITHOUT LoRA data!
  python scripts/utils/character_processor.py "selected/" --group-by body_type    # → emily/, ivy/, big_boobs/, petite/
  python scripts/utils/character_processor.py "selected/" --group-by ethnicity    # → emily/, ivy/, latina/, asian/
  python scripts/utils/character_processor.py "selected/" --group-by age_group    # → emily/, ivy/, young/, mature/
  python scripts/utils/character_processor.py "selected/" --group-by hair_color   # → emily/, ivy/, blonde/, brunette/
  python scripts/utils/character_processor.py "selected/" --group-by scenario     # → emily/, ivy/, bedroom/, office/
  
  # Advanced usage
  python scripts/utils/character_processor.py "directory/" --save-analysis --quiet

FEATURES:
---------
• 🚀 FLEXIBLE GROUPING: Controls fallback analysis for files without LoRA data
• Complete 4-stage pipeline in one command
• Metadata extraction from YAML and caption files with multi-character support
• Sequential context inference for edge cases
• Intelligent prompt-based descriptor extraction
• Multi-character directory creation (emily_ivy/, etc.)
• Minimum threshold protection (prevents directory fragmentation)
• Progress tracking and comprehensive statistics
• Dry-run mode for safe testing

PIPELINE STAGES:
----------------
1. Metadata Analysis: Extract character LoRA names (emily, ivy) + prompt data from YAML/caption files
2. Sequential Context: Infer missing characters from chronological neighbors  
3. Prompt Analysis: FALLBACK for remaining files - extract descriptors (15+ file minimum)
4. Character Grouping: Organize files into directories (supports flexible categories)

GROUPING CATEGORIES:
-------------------
• character (default): Uses LoRA names (emily, ivy) + prompt fallback (latina_petite_young)
• body_type: big_boobs, petite, curvy, tall, athletic, etc.
• ethnicity: latina, asian, black, white, indian, etc.
• age_group: young, mature, teen, milf, college, etc.
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
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    print("[!] PyYAML is required. Install with: pip install PyYAML", file=sys.stderr)
    sys.exit(1)

try:
    from utils.activity_timer import ActivityTimer, FileTracker
except ImportError:
    # Graceful fallback if activity timer not available
    ActivityTimer = None
    FileTracker = None


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


def extract_descriptive_character_from_prompt(prompt: str, group_by: str = "character") -> Optional[str]:
    """
    Extract descriptive character information from prompt text as fallback.
    
    Looks for physical descriptors like ethnicity, body type, hair color, etc.
    Only returns a character name if confident descriptors are found.
    
    Args:
        prompt: The prompt text to analyze
        
    Returns:
        Descriptive character name or None if no clear descriptors found
    """
    if not prompt:
        return None
    
    # Convert to lowercase for analysis
    prompt_lower = prompt.lower()
    
    # Define all descriptor categories
    descriptor_categories = {
        'ethnicity': [
            'latina', 'latin', 'hispanic', 'mexican', 'spanish',
            'asian', 'japanese', 'chinese', 'korean', 'thai', 'vietnamese',
            'black', 'african', 'ebony', 'dark skin',
            'white', 'caucasian', 'european',
            'indian', 'middle eastern', 'arab', 'persian'
        ],
        'body_type': [
            'petite', 'small', 'tiny', 'short',
            'tall', 'amazon', 'statuesque',
            'curvy', 'thick', 'voluptuous', 'busty', 'big boobs', 'big tits',
            'slim', 'skinny', 'thin', 'lean', 'athletic', 'fit'
        ],
        'age_group': [
            'young', 'teen', 'college', 'student',
            'mature', 'milf', 'older', 'cougar'
        ],
        'hair_color': [
            'blonde', 'brunette', 'redhead', 'black hair', 'brown hair',
            'silver hair', 'gray hair', 'pink hair', 'blue hair'
        ],
        'scenario': [
            'bedroom', 'bathroom', 'kitchen', 'office', 'outdoor',
            'beach', 'pool', 'car', 'hotel', 'cozy', 'public'
        ]
    }
    
    # Extract descriptors from prompt (look at first half of prompt for main subject)
    prompt_parts = prompt_lower.split(',')
    # Take first half of prompt parts, but at least first 5 parts
    num_parts = max(5, len(prompt_parts) // 2)
    prompt_text = ' '.join(prompt_parts[:num_parts])
    
    # Handle different grouping modes
    if group_by == "character":
        # Original character mode: combine ethnicity + body + age
        found_descriptors = []
        
        for category in ['ethnicity', 'body_type', 'age_group']:
            for descriptor in descriptor_categories[category]:
                if descriptor in prompt_text:
                    found_descriptors.append(descriptor.replace(' ', '_'))
                    break  # Only take first match per category
        
        return '_'.join(found_descriptors) if found_descriptors else None
    
    else:
        # Single category mode: group by specific category only
        if group_by not in descriptor_categories:
            return None
        
        # Find descriptors in the specified category
        for descriptor in descriptor_categories[group_by]:
            if descriptor in prompt_text:
                return descriptor.replace(' ', '_')
        
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
        prompt_char = extract_descriptive_character_from_prompt(prompt, group_by)
        
        if prompt_char:
            prompt_character_counts[prompt_char] = prompt_character_counts.get(prompt_char, 0) + 1
            file_to_prompt_char[filename] = prompt_char
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"    • {filename}: '{prompt_preview}' → {prompt_char}")
        else:
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"    • {filename}: '{prompt_preview}' → NO MATCH")
    
    print(f"\n[*] Prompt character counts:")
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
        print(f"\n📊 YAML ANALYSIS SUMMARY")
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
        
        print(f"\n🎨 STAGES FOUND:")
        for stage in processing_stats['stages_found']:
            print(f"  • {stage}")
        
        if processing_stats['character_counts']:
            most_common = max(processing_stats['character_counts'].items(), key=lambda x: x[1])
            print(f"\n🏆 Most common character: {most_common[0]} ({most_common[1]} images)")
    
    return analysis_data


# ============================================================================
# STAGE 2: SEQUENTIAL CONTEXT ANALYSIS
# ============================================================================

def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """Extract timestamp from filename for sequential ordering."""
    # Match pattern: YYYYMMDD_HHMMSS
    match = re.match(r'(\d{8}_\d{6})', filename)
    return match.group(1) if match else None


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
        print(f"🧠 Sequential Context Analysis")
        print(f"[*] Found {len(no_character_files)} files without character data")
        print(f"[*] Analyzing sequential context...")
    
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
        print(f"\n📊 SEQUENTIAL CONTEXT SUMMARY")
        print(f"{'='*50}")
        print(f"Files without character data: {len(no_character_files)}")
        print(f"Successful inferences: {inferences_made}")
        print(f"Failed inferences: {len(no_character_files) - inferences_made}")
        
        if inference_details:
            print(f"\n🎯 INFERRED CHARACTERS:")
            character_counts = Counter([detail['inferred_character'] for detail in inference_details])
            for char, count in character_counts.most_common():
                print(f"  • {char}: {count} files")
    
    return analysis_data


# ============================================================================
# STAGE 3: CHARACTER GROUPING
# ============================================================================

def move_file_pair(png_path: Path, yaml_path: Path, target_dir: Path, dry_run: bool = False, tracker: Optional = None) -> bool:
    """Move PNG + YAML pair to target directory."""
    try:
        if not dry_run:
            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move PNG file
            png_target = target_dir / png_path.name
            png_path.rename(png_target)
            
            # Move YAML file if it exists
            if yaml_path.exists():
                yaml_target = target_dir / yaml_path.name
                yaml_path.rename(yaml_target)
            
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


def group_by_character(analysis_data: Dict, source_directory: str, dry_run: bool = False, quiet: bool = False) -> Dict:
    """
    Stage 3: Group files by character based on analysis data.
    
    Args:
        analysis_data: Enhanced analysis data from previous stages
        source_directory: Directory containing the files to organize
        dry_run: Preview mode - don't actually move files
        quiet: Suppress progress output
        
    Returns:
        Dictionary with grouping statistics
    """
    source_dir = Path(source_directory).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    character_mapping = analysis_data['character_mapping']
    
    # Extract unique characters
    characters = set()
    for file_data in character_mapping.values():
        if file_data.get('has_character'):
            for char in file_data.get('all_characters', []):
                characters.add(char)
        # Also include inferred characters
        if file_data.get('has_inferred_character'):
            inferred = file_data.get('inferred_character')
            if inferred:
                characters.add(inferred)
    
    characters = sorted(list(characters))
    
    if not quiet:
        print(f"🎭 Character Grouper - Intelligent Image Organization")
        print(f"📁 Source: {source_dir}")
        print(f"[*] Found {len(characters)} characters: {', '.join(characters)}")
    
    # Create character directories (including prompt-based characters)
    all_characters = list(characters)
    
    # Add any prompt-based characters that were discovered
    for filename, char_info in character_mapping.items():
        if char_info.get('source') == 'prompt_analysis':
            prompt_char = char_info.get('character_name')
            if prompt_char and prompt_char not in all_characters:
                all_characters.append(prompt_char)
    
    char_dirs = create_character_directories(source_dir, all_characters, dry_run)
    
    # Initialize activity tracking
    tracker = None
    if ActivityTimer and not dry_run:
        tracker = FileTracker()
    
    # Statistics tracking
    stats = {
        'total_files': len(character_mapping),
        'moved_successfully': 0,
        'move_errors': 0,
        'missing_files': 0,
        'skipped_unknown': 0,
        'character_counts': {char: 0 for char in characters},
        'processing_time': 0
    }
    
    start_time = time.time()
    processed = 0
    
    if not quiet:
        print(f"\n[*] Processing {stats['total_files']} image files...")
    
    for png_filename, char_info in character_mapping.items():
        processed += 1
        
        # Progress update every 100 files
        if not quiet and processed % 100 == 0:
            percent = (processed / stats['total_files']) * 100
            print(f"[*] Progress: {processed}/{stats['total_files']} ({percent:.1f}%)")
        
        # Determine character and target directory
        character = char_info.get('character_name')
        has_character = char_info.get('has_character', False)
        
        # Handle files without character data (check for inferred characters)
        inferred_character = char_info.get('inferred_character')
        if not has_character and not inferred_character:
            if not dry_run and not quiet:
                print(f"[*] Skipping {png_filename} (no character data)")
            elif dry_run and not quiet:
                print(f"[DRY RUN] Would skip: {png_filename} (no character data)")
            stats['skipped_unknown'] += 1
            continue
        
        # Determine target directory based on character logic
        all_characters = char_info.get('all_characters', [])
        is_multi_character = char_info.get('is_multi_character', False)
        
        # Use inferred character if no direct character data
        if not has_character and inferred_character:
            character = inferred_character
            all_characters = [inferred_character]
            is_multi_character = False
        
        # Multi-character logic: create combined directory name
        if is_multi_character and len(all_characters) > 1:
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
                print(f"[!] Missing PNG file: {png_filename}")
            stats['missing_files'] += 1
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
        
        print(f"\n📊 CHARACTER GROUPING SUMMARY")
        print(f"{'='*50}")
        print(f"Total files processed: {stats['total_files']}")
        print(f"{action} successfully: {stats['moved_successfully']}")
        print(f"Skipped (no character): {stats['skipped_unknown']}")
        print(f"Move errors: {stats['move_errors']}")
        print(f"Missing files: {stats['missing_files']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        print(f"\n🎭 FILES PER CHARACTER:")
        for char, count in stats['character_counts'].items():
            if count > 0:
                print(f"  • {char}: {count} files")
        
        if stats['moved_successfully'] > 0:
            rate = stats['moved_successfully'] / stats['processing_time']
            print(f"\n⚡ Processing rate: {rate:.1f} files/second")
        
        if dry_run:
            print(f"\n🧪 DRY RUN COMPLETE - No files were actually moved")
            print(f"Run without --dry-run to perform the actual grouping")
        else:
            print(f"\n✅ CHARACTER GROUPING COMPLETE!")
        
        print(f"\n🎯 Successfully organized {stats['moved_successfully']} files by character!")
    
    return stats


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_directory(directory: str, dry_run: bool = False, save_analysis: bool = False, quiet: bool = False, group_by: str = "character") -> Dict:
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
        print(f"🚀 Starting complete character processing pipeline...")
        print(f"📁 Directory: {directory}")
        print(f"🧪 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"=" * 60)
    
    # Stage 1: YAML Analysis
    analysis_file = None
    if save_analysis:
        dir_name = Path(directory).name.replace(' ', '_').replace('(', '').replace(')', '')
        analysis_file = f"analysis_{dir_name}.json"
    
    analysis_data = analyze_yaml(directory, analysis_file, quiet)
    
    # Stage 2: Sequential Context
    enhanced_data = add_sequential_context(analysis_data, quiet)
    
    # Stage 2.5: Prompt Analysis (fallback for remaining files)
    if not quiet:
        print("\n=== Stage 2.5: Prompt Analysis ===")
    
    prompt_assignments = analyze_prompts_for_characters(enhanced_data['character_mapping'], min_threshold=15, group_by=group_by)
    
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
    
    # Stage 3: Character Grouping
    grouping_stats = group_by_character(enhanced_data, directory, dry_run, quiet)
    
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
        print(f"\n🎉 PIPELINE COMPLETE!")
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
  python scripts/utils/character_processor.py "_asian (chater 1)/"
  
  # Group by body type instead of character
  python scripts/utils/character_processor.py "selected/" --group-by body_type
  
  # Group by ethnicity
  python scripts/utils/character_processor.py "selected/" --group-by ethnicity
  
  # Dry run to preview changes
  python scripts/utils/character_processor.py "selected/" --dry-run --group-by age_group
  
  # Save intermediate analysis files
  python scripts/utils/character_processor.py "directory/" --save-analysis
        """
    )
    
    parser.add_argument("directory", type=str, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving files")
    parser.add_argument("--save-analysis", action="store_true", help="Save intermediate analysis JSON files")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--group-by", "-g", type=str, default="character",
                        choices=["character", "body_type", "ethnicity", "age_group", "hair_color", "scenario"],
                        help="Grouping category for prompt analysis (default: character)")
    
    args = parser.parse_args()
    
    try:
        results = process_directory(
            directory=args.directory,
            dry_run=args.dry_run,
            save_analysis=args.save_analysis,
            quiet=args.quiet,
            group_by=args.group_by
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"[!] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
