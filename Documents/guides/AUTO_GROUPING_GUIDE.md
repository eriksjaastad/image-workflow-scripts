# Auto-Grouping Guide for Character Sorter

**Status:** Active
**Audience:** Developers, Operators

**Last Updated:** 2025-10-26

**⚠️ DEPRECATION NOTICE:** Auto-grouping functionality has been moved to `scripts/02_character_processor.py`.  
The web character sorter (`03_web_character_sorter.py`) is now a purely interactive manual sorting tool.

For current auto-grouping documentation, see: `scripts/02_character_processor.py` (header comments)

---

## Historical Documentation (Deprecated)

The content below refers to the old auto-grouping implementation in the web character sorter.

## Overview

The character sorter now includes intelligent auto-grouping based on YAML metadata analysis. This dramatically speeds up sorting large batches by pre-organizing images into logical subdirectories.

## Quick Start

### Ethnicity Grouping (RECOMMENDED for 100-500 images)

```bash
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02 --analyze-groups --group-by ethnicity
```

**Result:** Creates 7-8 directories by ethnicity:

- `mixed_race/` (109 images)
- `south_asian/` (37 images)
- `middle_eastern/` (34 images)
- `asian/` (33 images)
- `latina/` (33 images)
- `white/` (31 images)
- `_no_ethnicity/` (misclassified/missing data)

### Age Grouping

```bash
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02 --analyze-groups --group-by age
```

**Result:** Creates 8 directories by age range:

- `mid_30s/` (106 images)
- `late_20s/` (33 images)
- `50s/` (32 images)
- `mid_20s/` (31 images)
- etc.

### Hierarchical Grouping (for 2000+ images)

```bash
python scripts/02_web_character_sorter.py __selected/large_batch --analyze-groups --group-by ethnicity,age
```

**Result:** Creates nested grouping like:

- `mixed_race_mid_30s/` (81 images)
- `asian_late_20s/` (15 images)
- etc.

## Adjusting Minimum Group Size

```bash
# Lower threshold for more granularity
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02 --analyze-groups --group-by ethnicity --min-group-size 5

# Higher threshold for fewer, larger groups
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02 --analyze-groups --group-by ethnicity --min-group-size 20
```

## How It Works

1. **Analyzes YAML files:** Extracts ethnicity (prompt index 5) and age (prompt index 6)
2. **Shows dry-run preview:** Displays what directories would be created
3. **Prompts for action:**
   - `[y]` - Create directories and move files
   - `[c]` - Change minimum group size and re-analyze
   - `[n]` - Cancel without moving anything
4. **Moves files:** Automatically moves PNG + YAML pairs to subdirectories
5. **Handles small groups:** Files below threshold go to `_small_groups/` for manual sorting

## Best Practices

### Choose the Right Mode

| Batch Size     | Recommended Mode           | Example                                    |
| -------------- | -------------------------- | ------------------------------------------ |
| 100-500 images | `--group-by ethnicity`     | `hannah_hannah_02` (312 images) → 7 groups |
| 100-500 images | `--group-by age`           | Alternative grouping strategy              |
| 2000+ images   | `--group-by ethnicity,age` | Large batches needing hierarchical split   |

### When to Use Each

**Ethnicity Mode:**

- When you want to separate by physical characteristics
- Good for training models with diverse datasets
- Creates balanced groups for most AI-generated batches

**Age Mode:**

- When age variation is more important than ethnicity
- Useful for age-specific model training
- Creates 8 groups spanning 20s-50s

**Hierarchical Mode:**

- Only for VERY large batches (2000+)
- Creates many small subdirectories
- Requires high image count to avoid too-small groups

## Troubleshooting

### "Would create: 0 directories"

- Your YAML files may not have structured prompts at the expected indices
- Run `python scripts/tools/yaml_analyzer.py <directory>` to inspect YAML structure
- Consider adjusting `--min-group-size` to a lower value

### "Small groups: 200+ images"

- Your threshold is too high for the batch size
- Lower `--min-group-size` or choose a different grouping mode
- For hierarchical mode, switch to single-level (ethnicity or age only)

### Missing YAMLs

- Auto-grouping requires YAML companion files for all images
- Images without YAMLs will be skipped
- Run yaml_analyzer first to verify YAML coverage

## Examples

### Typical Workflow

```bash
# 1. Analyze YAML structure first
python scripts/tools/yaml_analyzer.py __selected/hannah_hannah_02

# 2. Run auto-grouping with ethnicity (dry-run shows preview)
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02 --analyze-groups --group-by ethnicity

# 3. Review output, type 'y' to proceed, or 'c' to adjust threshold

# 4. Result: 7 subdirectories created, ready for manual sorting within each
```

### After Auto-Grouping

After auto-grouping, you can manually sort within each subdirectory:

```bash
# Sort within a specific ethnic group
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02/mixed_race
```

Or use the multi-directory mode to sort all groups sequentially:

```bash
# Auto-advance through all subdirectories
python scripts/02_web_character_sorter.py __selected/hannah_hannah_02
```

## Technical Details

### YAML Structure Requirements

Auto-grouping expects prompts in this comma-separated format:

```
Index 0: Character name (@hannah)
Index 1-3: Quality tags (high resolution, 4k, RAW photo)
Index 4: Face shape
Index 5: ETHNICITY ← extracted here
Index 6: AGE ← extracted here
Index 7+: Gender, physique, hair, scenario, etc.
```

### Data Cleaning

- Ethnicity extraction filters out noise (resolution, photo, raw, face)
- Age extraction removes "in her"/"in his" prefixes
- Files with missing data go to `_no_ethnicity/` or `_no_age/`
- Python tuple tags in enhanced YAMLs are handled automatically

### File Operations

- Moves both PNG and YAML files together
- Uses `shutil.move()` for atomic operations
- Creates target directories if they don't exist
- `_small_groups/` directory holds below-threshold groups

## Performance

- **Analysis speed:** ~300 YAML files in <1 second
- **File moves:** Instant (same filesystem)
- **Scalability:** Tested up to 2000+ images

This is dramatically faster than manual sorting and reduces decision fatigue by pre-organizing similar images!
