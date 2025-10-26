# Historical Project Training Data Extraction Plan
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


## 🎯 Goal
Extract training data from past projects where we have:
- **RAW directory:** All original images (all stages, all groups)
- **FINAL directory:** Only Erik's selected winners
- **Mixed structures:** Some flat, some with subdirectories

---

## 📋 Phase 1: Inventory & Scan

### Step 1: Discover All Projects
Scan for project pairs:
```
Projects to check:
- Aiko_raw (1,050 → 258)
- Eleni_raw (5,816 → 1,148)
- Kiara_Average (5,796 → 541)
- Slender Kiara (5,801 → 261)
- agent-1001/1002 (2,057 → 258)
- agent-1003 (2,053 → 410)
- 1011, 1012, 1013
- 1101_Hailey, 1100
- tattersail-0918 (13,928 → 2,858)
- jmlimages-random (13,345 → 2,765)
- mixed-0919 (426 → 183)
- mojo-1 (19,183 → 6,453)
- mojo-2 (17,935 → ~4,500)
```

### Step 2: Scan Each Project
For each project, recursively find all `.png` files:
```python
def scan_project(raw_dir, final_dir):
    """Recursively scan both directories for all images."""
    raw_images = list(Path(raw_dir).rglob("*.png"))
    final_images = list(Path(final_dir).rglob("*.png"))
    return raw_images, final_images
```

---

## 📊 Phase 2: Build Image Database

### Create Master Table
**Schema: `project_images.jsonl`**
```json
{
  "project": "mojo-1",
  "image_path": "raw/mojo1/20250708_060711_stage2.png",
  "filename": "20250708_060711_stage2.png",
  "timestamp": "20250708_060711",
  "stage": 2,
  "file_mtime": "2025-07-08T06:07:30Z",
  "file_size": 2458392,
  "in_final": true,
  "final_path": "mojo1/subdir/20250708_060711_stage2.png",
  "final_mtime": "2025-10-03T14:22:10Z",
  "was_cropped": true  // final_mtime != raw_mtime
}
```

### Parse Filename Components
```python
def parse_filename(filename: str):
    """Extract timestamp and stage from filename."""
    # Patterns to match:
    # 20250708_060711_stage2.png
    # 20250708_060711_stage2_upscaled.png
    # 20250708_060711_stage3_enhanced.png
    
    import re
    match = re.match(r'(\d{8}_\d{6})_stage(\d+(?:\.\d+)?)', filename)
    if match:
        return {
            'timestamp': match.group(1),
            'stage': float(match.group(2))
        }
    return None
```

### Detect Crops by File Date
```python
def was_cropped(raw_path: Path, final_path: Path) -> bool:
    """Check if file was modified after selection (cropped)."""
    if not final_path.exists():
        return False
    
    raw_mtime = raw_path.stat().st_mtime
    final_mtime = final_path.stat().st_mtime
    
    # If final is newer by >5 seconds, assume cropped
    return (final_mtime - raw_mtime) > 5
```

---

## 🔄 Phase 3: Group & Match

### Step 1: Group RAW Images by Timestamp
```python
def group_raw_images(raw_images: List[Path]):
    """Group all raw images by timestamp."""
    groups = defaultdict(list)
    
    for img_path in raw_images:
        parsed = parse_filename(img_path.name)
        if parsed:
            groups[parsed['timestamp']].append({
                'path': img_path,
                'stage': parsed['stage']
            })
    
    return groups
```

### Step 2: Find Winners in FINAL
```python
def find_winners(groups: dict, final_images: List[Path]):
    """For each group, find which image (if any) is in FINAL."""
    winners = {}
    
    for timestamp, group_images in groups.items():
        for final_img in final_images:
            final_parsed = parse_filename(final_img.name)
            if final_parsed and final_parsed['timestamp'] == timestamp:
                # Found winner!
                winners[timestamp] = {
                    'winner': final_img,
                    'winner_stage': final_parsed['stage'],
                    'group': group_images
                }
                break
    
    return winners
```

### Step 3: Create Training Entries
```python
def create_training_entries(winners: dict, project: str):
    """Create (winner, losers) pairs for training."""
    entries = []
    
    for timestamp, data in winners.items():
        winner_path = data['winner']
        winner_stage = data['winner_stage']
        
        # Find losers (images in same group, not selected)
        losers = []
        for img in data['group']:
            if img['path'].name != winner_path.name:
                losers.append(img['path'])
        
        if losers:  # Only create entry if there were alternatives
            entries.append({
                'project': project,
                'timestamp': timestamp,
                'chosen_path': str(winner_path),
                'chosen_stage': winner_stage,
                'neg_paths': [str(p) for p in losers],
                'neg_stages': [img['stage'] for img in data['group'] 
                              if img['path'].name != winner_path.name]
            })
    
    return entries
```

---

## 📝 Phase 4: Output Training Data

### Format 1: CSV (Compatible with existing logs)
```csv
project,timestamp,chosen_path,chosen_stage,neg_paths,neg_stages,was_cropped
mojo-1,20250708_060711,/path/to/stage2.png,2.0,"['/path/to/stage1.png','/path/to/stage3.png']","[1.0,3.0]",true
```

### Format 2: JSONL (More flexible)
```json
{
  "project": "mojo-1",
  "timestamp": "20250708_060711",
  "chosen_path": "/path/to/stage2.png",
  "chosen_stage": 2.0,
  "neg_paths": ["/path/to/stage1.png", "/path/to/stage3.png"],
  "neg_stages": [1.0, 3.0],
  "was_cropped": true,
  "metadata": {
    "raw_dir": "archives/mojo1_raw",
    "final_dir": "archives/mojo1_final",
    "extraction_date": "2025-10-20"
  }
}
```

---

## 🤖 Phase 5: AI Training Integration

### How AI Will Use This Data

1. **Load Historical + Recent Data:**
```python
# Combine old projects + Mojo 2 logs
historical_data = load_jsonl("data/training/historical_projects.jsonl")
recent_data = load_csv("data/training/selection_only_log.csv")
all_training_data = historical_data + recent_data
```

2. **Understand Crop Signal:**
```python
# AI can see: "This image was selected AND cropped"
# Learn: "Images with X features need cropping"
# Use for crop proposer training (Phase 2.4)
```

3. **Learn from Anomalies:**
```python
# Find cases where chosen_stage < max(neg_stages)
anomaly_cases = [entry for entry in all_training_data
                 if entry['chosen_stage'] < max(entry['neg_stages'])]

# Train: "Sometimes lower stage is better - learn why"
```

---

## 🛠️ Implementation Plan

### Script: `scripts/ai/extract_historical_training.py`

**Usage:**
```bash
# Extract from single project
python scripts/ai/extract_historical_training.py \
  --raw archives/mojo1_raw \
  --final mojo1 \
  --project mojo-1 \
  --output data/training/historical_mojo1.jsonl

# Batch extract all projects
python scripts/ai/extract_historical_training.py \
  --batch \
  --config data/training/project_paths.json \
  --output data/training/historical_all.jsonl
```

**Config file example (`project_paths.json`):**
```json
{
  "projects": [
    {
      "name": "mojo-1",
      "raw_dir": "archives/mojo1_raw",
      "final_dir": "mojo1"
    },
    {
      "name": "Aiko_raw",
      "raw_dir": "archives/Aiko_raw_incoming",
      "final_dir": "archives/Aiko_raw_final"
    }
  ]
}
```

---

## ⚠️ Edge Cases to Handle

### 1. **Missing Groups**
- Some timestamps might only have 1 stage (can't train on)
- Skip these: no decision was made

### 2. **Subdirectories**
- Use `rglob()` to recursively find all images
- Store relative paths for matching

### 3. **Filename Variations**
- Handle: `stage1.png`, `stage2_upscaled.png`, `stage3_enhanced.png`
- Regex should capture all variations

### 4. **Duplicate Timestamps**
- Some projects might have subdirs with same timestamp
- Use full path for matching, not just filename

### 5. **Skipped Groups**
- If timestamp in RAW but not in FINAL, Erik rejected entire group
- These are still useful: "Don't keep any of these"
- Store as negative examples

---

## 📊 Expected Output

### Statistics to Track:
```
Project: mojo-1
  Total raw images: 19,183
  Total groups: 6,394
  Groups with winner: 6,453 (some might have multiple selections?)
  Groups rejected: 0 (or negative number if multi-select?)
  Anomaly cases (chose lower stage): ~194 (3%)
  Cropped images: ~3,226 (50%?)
  
Total across all projects:
  ~23,000 selection decisions
  ~700-800 anomaly cases
  ~11,500 crop decisions
```

---

## 🎯 Next Steps (When Erik Returns)

1. ✅ **Locate archive directories** - Where are RAW and FINAL stored?
2. ✅ **Build extraction script** - Implement plan above
3. ✅ **Test on one project** - Verify logic works
4. ✅ **Batch extract all** - Process all 16+ projects
5. ✅ **Merge with existing logs** - Combine historical + Mojo 2
6. ✅ **Re-train ranking model** - Test if anomaly detection improves
7. ✅ **Train crop proposer** - Use crop flags for Phase 2.4

---

## 💾 Files Created

- `data/training/historical_projects.jsonl` - All extracted training data
- `data/training/project_images.jsonl` - Master image database (optional)
- `scripts/ai/extract_historical_training.py` - Extraction script
- `data/training/extraction_report.json` - Stats and metadata

---

**Status:** Plan documented, ready to implement on return
**Priority:** High (unlocks 23K+ training examples)
**Estimated Time:** 2-3 hours to build + 30 mins to run extraction

