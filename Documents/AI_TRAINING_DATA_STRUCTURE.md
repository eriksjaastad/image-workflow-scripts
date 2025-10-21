# AI Training Data Structure - Critical Rules

**Purpose:** This document explains how training data is organized and the critical rules that must be followed to avoid corrupting the AI models.

**Last Updated:** October 20, 2025

---

## 🚨 THE GOLDEN RULE

**PROJECTS NEVER MIX**

Each project (Mojo 1, Mojo 2, Eleni, Aiko, etc.) represents a **different character**. Training data from different projects **MUST NEVER be compared or mixed during model training**.

---

## Why This Matters

### The Problem

If you train a model by comparing images across projects, you're teaching it nonsense:

❌ **WRONG:** "This mojo1 image is better than that mojo2 image"  
- These are different characters
- Your aesthetic preference for one character doesn't inform the other
- This corrupts the model

✅ **CORRECT:** "This mojo1 stage2 image is better than this other mojo1 stage1 image"  
- Same character, same project
- Valid comparison that teaches aesthetic preferences
- Model learns what makes a good image

### The Misunderstanding

The v2 ranker model trained on October 20, 2025 **mixed projects incorrectly**. It compared:
- Mojo 1 images vs Mojo 2 images ❌
- Images from different characters ❌
- Violated project boundaries ❌

This needs to be fixed before continuing.

---

## The Training Data Structure

### Selection Log Format

File: `data/training/selection_only_log.csv`

```csv
session_id,set_id,chosen_path,neg_paths,timestamp
20251004_213737,group_8,/Users/.../mojo1/20250708_060711_stage2_upscaled.png,"[""/Users/.../mojo1/20250708_060558_stage1_generated.png""]",2025-10-05T01:43:47Z
```

**What Each Row Represents:**

1. **One complete image set** from ONE project
2. **Winner:** `chosen_path` (the image you selected)
3. **Losers:** `neg_paths` (the images you rejected)
4. **Same group:** All images are from the same timestamp/generation
5. **Same project:** All images are from the same project (mojo1 in this example)

### Crop Log Format

File: `data/training/select_crop_log.csv`

```csv
session_id,directory,chosen_path,crop_x1,crop_y1,crop_x2,crop_y2,timestamp
```

Similar structure:
- Each row is ONE crop decision
- All from ONE project
- Project determined by `directory` field

---

## Implementation Rules

### Rule 1: Extract Project ID from Path

Every training data path contains the project ID:

```python
def get_project_id(path: str) -> str:
    """
    Extract project ID from file path.
    
    Examples:
        /Users/.../mojo1/file.png → 'mojo1'
        /Users/.../mojo2/_mixed/file.png → 'mojo2'
        training data/mojo1/faces/file.png → 'mojo1'
    """
    parts = Path(path).parts
    for part in parts:
        # Check for known project patterns
        if part.startswith('mojo'):  # mojo1, mojo2, etc.
            return part
        if part in ['eleni', 'aiko', 'dalia', 'kiara']:  # Character names
            return part
        if part.endswith('_raw') or part.endswith('_final'):  # Project suffixes
            return part.replace('_raw', '').replace('_final', '')
    return 'unknown'
```

### Rule 2: Match by Filename Within Project

Paths may differ between CSV and embeddings cache, so match by filename + project:

```python
def find_embedding(
    csv_path: str,
    embeddings_cache: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Find embedding for a CSV path entry.
    
    Args:
        csv_path: Path from CSV (may be absolute or relative)
        embeddings_cache: Dict mapping cache paths to embeddings
    
    Returns:
        Embedding array or None if not found
    """
    project_id = get_project_id(csv_path)
    filename = Path(csv_path).name
    
    # Search cache for this project + filename
    for cache_path, embedding in embeddings_cache.items():
        cache_project = get_project_id(cache_path)
        cache_filename = Path(cache_path).name
        
        if cache_project == project_id and cache_filename == filename:
            return embedding
    
    return None
```

### Rule 3: Validate Set Integrity

Before creating training pairs, verify all images in a set are from the same project:

```python
def validate_set(chosen_path: str, neg_paths: List[str]) -> None:
    """
    Ensure all paths in a set are from the same project.
    
    Raises:
        ValueError: If mixed projects detected
    """
    chosen_project = get_project_id(chosen_path)
    
    for neg_path in neg_paths:
        neg_project = get_project_id(neg_path)
        if neg_project != chosen_project:
            raise ValueError(
                f"Mixed projects in set!\n"
                f"  Winner: {chosen_path} (project: {chosen_project})\n"
                f"  Loser:  {neg_path} (project: {neg_project})\n"
                f"This violates project boundary rules."
            )
```

### Rule 4: Create Training Pairs Within Project

When building training pairs for the ranker:

```python
def create_training_pairs(selection_log: pd.DataFrame) -> List[TrainingPair]:
    """
    Create training pairs from selection log.
    
    Returns:
        List of (winner_embedding, loser_embedding, project_id) tuples
    """
    pairs = []
    
    for _, row in selection_log.iterrows():
        chosen_path = row['chosen_path']
        neg_paths = json.loads(row['neg_paths'])
        
        # Validate set integrity
        validate_set(chosen_path, neg_paths)
        
        # Get project ID (all images in set have same project)
        project_id = get_project_id(chosen_path)
        
        # Load embeddings
        winner_emb = find_embedding(chosen_path, embeddings_cache)
        if winner_emb is None:
            continue  # Skip if embedding not found
        
        # Create pair for each loser
        for neg_path in neg_paths:
            loser_emb = find_embedding(neg_path, embeddings_cache)
            if loser_emb is None:
                continue
            
            # Store pair with project context
            pairs.append({
                'winner': winner_emb,
                'loser': loser_emb,
                'project_id': project_id,  # Track for validation
                'set_id': row['set_id']
            })
    
    return pairs
```

---

## Training Strategy

### Recommended: Unified Model

Train **ONE model** on data from **ALL projects**:

**Pros:**
- Model learns your general aesthetic preferences
- Works for future projects without retraining
- Simpler to maintain

**Requirements:**
- Training pairs must stay within same project
- Never compare mojo1 vs mojo2
- Model sees examples from all projects but respects boundaries

**Implementation:**
```python
# During training
for pair in training_pairs:
    winner_score = model(pair['winner'])
    loser_score = model(pair['loser'])
    loss = margin_ranking_loss(winner_score, loser_score)
    # Project boundaries already enforced during pair creation
```

### Alternative: Per-Project Models

Train **separate models** for each project:

**Pros:**
- Guarantees no cross-project contamination
- Can learn project-specific preferences

**Cons:**
- More complex to maintain (multiple models)
- Needs retraining for each new project
- Less data per model

**Only use this if unified model performs poorly.**

---

## Project Manifest Reference

Each project has metadata at `data/projects/<projectId>.project.json`:

```json
{
  "schemaVersion": 1,
  "projectId": "mojo1",
  "title": "Mojo1",
  "status": "active",
  "startedAt": "2025-10-01T00:00:00Z",
  "finishedAt": "2025-10-11T17:29:49Z",
  "paths": {
    "root": "../../mojo1"
  },
  "counts": {
    "initialImages": 19183,
    "finalImages": 6453
  }
}
```

**Use manifests to:**
- Validate project IDs
- Filter training data by project
- Determine project date ranges
- Count training examples per project

---

## Validation Checklist

Before training any model, verify:

- [ ] **CSV rows validated:** Each row contains images from ONE project only
- [ ] **Project IDs extracted:** Can parse project from all paths
- [ ] **Embeddings matched:** Can find embeddings by filename + project
- [ ] **Training pairs verified:** No cross-project pairs exist
- [ ] **Stats logged:** Know how many examples per project

**Script to validate training data:**

```python
#!/usr/bin/env python3
"""Validate training data structure"""
import pandas as pd
import json
from collections import Counter

def validate_training_data(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    
    errors = []
    project_counts = Counter()
    
    for idx, row in df.iterrows():
        chosen_path = row['chosen_path']
        neg_paths = json.loads(row['neg_paths'])
        
        try:
            validate_set(chosen_path, neg_paths)
            project_id = get_project_id(chosen_path)
            project_counts[project_id] += 1
        except ValueError as e:
            errors.append(f"Row {idx}: {e}")
    
    # Report results
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  {error}")
    else:
        print("✅ All rows validated successfully")
    
    print(f"\n📊 Training examples per project:")
    for project_id, count in sorted(project_counts.items()):
        print(f"  {project_id}: {count:,} examples")

if __name__ == "__main__":
    validate_training_data("data/training/selection_only_log.csv")
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Mixing Projects in Training Pairs

```python
# WRONG: Comparing across projects
winner_from_mojo1 = embeddings['mojo1/file1.png']
loser_from_mojo2 = embeddings['mojo2/file2.png']
loss = ranking_loss(winner_from_mojo1, loser_from_mojo2)  # INVALID!
```

### ❌ Mistake 2: Ignoring Project Context When Matching

```python
# WRONG: Matching by filename only
filename = '20250708_060711_stage2.png'
embedding = embeddings_cache[filename]  # Could be from any project!
```

### ❌ Mistake 3: Not Validating Set Integrity

```python
# WRONG: Assuming CSV is clean
for row in csv:
    create_pairs(row['chosen'], row['neg_paths'])  # No validation!
```

### ✅ Correct Approach

```python
# RIGHT: Full validation pipeline
for row in csv:
    # 1. Validate set
    validate_set(row['chosen'], row['neg_paths'])
    
    # 2. Get project
    project_id = get_project_id(row['chosen'])
    
    # 3. Match with project context
    winner_emb = find_embedding(row['chosen'], embeddings, project_id)
    
    # 4. Create pairs
    for loser_path in row['neg_paths']:
        loser_emb = find_embedding(loser_path, embeddings, project_id)
        pairs.append((winner_emb, loser_emb, project_id))
```

---

## What Needs to be Fixed

### Current State (October 20, 2025)

- ✅ Ranker v1: Trained correctly on Mojo 2 only
- ❌ Ranker v2: **Trained incorrectly** - mixed Mojo 1 + Mojo 2
  - Used 5,123 selections (mixed projects)
  - Path matching ignored project boundaries
  - Model learned invalid cross-project comparisons

### Required Actions

1. **Fix train_ranker_v2.py:**
   - Add `get_project_id()` function
   - Add `validate_set()` function
   - Update `find_embedding()` to use project context
   - Verify no cross-project pairs

2. **Retrain ranker v2 (or create v3):**
   - Use corrected data loading
   - Verify all pairs respect project boundaries
   - Log training examples per project
   - Validate results before deployment

3. **Update documentation:**
   - Mark v2 as "invalid - trained with mixed projects"
   - Document v3 as "correctly respects project boundaries"

---

## Related Documents

- **[AI_TRAINING_PHASE2_QUICKSTART.md](AI_TRAINING_PHASE2_QUICKSTART.md)** - Implementation guide with code examples
- **[AI_PROJECT_IMPLEMENTATION_PLAN.md](AI_PROJECT_IMPLEMENTATION_PLAN.md)** - Overall project plan
- **[PROJECT_MANIFEST_GUIDE.md](PROJECT_MANIFEST_GUIDE.md)** - Project metadata structure

---

**Questions?** If anything in this document is unclear, update it immediately. This is a critical reference that prevents data corruption.

