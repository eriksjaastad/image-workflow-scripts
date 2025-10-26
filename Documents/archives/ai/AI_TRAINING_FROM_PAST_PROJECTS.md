# Training Data from Past Projects (Before/After)
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


## 🎯 Goal
Extract training examples from completed projects where we have:
- **BEFORE:** Full group of images (stage1, stage2, stage3)
- **AFTER:** Only Erik's chosen winner

This captures **anomaly cases** where Erik rejected higher stages!

---

## 📁 Data Structure

### Typical Project Layout:
```
project_name/
├── sandbox/project_name/     ← BEFORE (all images, all stages)
│   ├── 20250708_060711_stage1.png
│   ├── 20250708_060711_stage2.png
│   ├── 20250708_060711_stage3.png
│   └── ... (thousands more)
└── project_name/             ← AFTER (only winners)
    ├── 20250708_060711_stage2.png  ← Erik chose stage2!
    └── ... (selected images)
```

---

## 🔄 Conversion Process

### Step 1: Group Detection
For each project:
1. Scan `sandbox/project_name/` - find all images
2. Group by timestamp (same logic as image selector)
3. For each group, identify all stages present

### Step 2: Winner Detection
1. Scan `project_name/` (root) - find winner images
2. Match winner back to its group by timestamp
3. Identify which stage Erik chose

### Step 3: Create Training Pairs
```python
# For each group:
winner = "stage2.png"           # Erik's choice (from root dir)
losers = ["stage1.png",         # Rejected (still in sandbox)
          "stage3.png"]         # Rejected higher stage! (anomaly)

# Create training entry:
{
  "chosen_path": "project/stage2.png",
  "neg_paths": ["project/stage1.png", "project/stage3.png"],
  "project": "mojo_old",
  "timestamp": "2025-07-08T06:07:11Z"
}
```

---

## 💡 Key Insight

**This captures anomalies automatically!**

When Erik chose `stage2` over `stage3`:
- ✅ Winner: stage2 (good quality, no anomalies)
- ❌ Loser: stage3 (missing belly button, weird face)

The AI will learn: "Sometimes stage2 beats stage3" and hopefully learn WHY (visual features in the embeddings)

---

## 📊 Expected Data Yield

### Existing Projects:
- **Mojo 1:** Already uploaded (5,244 selections)
- **Mojo 2:** Current project (4,594 selections) 
- **Other past projects:** ??? (need to check)

### Anomaly Rate:
If ~3% of selections reject higher stages:
- 10,000 total selections → ~300 anomaly cases
- Still not 1,000, but better than 142!

---

## 🛠️ Script to Build

**`scripts/ai/extract_training_from_projects.py`**

```python
# Pseudocode:
for each project in past_projects:
    sandbox_dir = f"sandbox/{project}"
    winner_dir = f"{project}"
    
    # Group sandbox images by timestamp
    groups = group_images_by_timestamp(sandbox_dir)
    
    # For each group, find winner
    for group in groups:
        winner = find_winner_in_dir(winner_dir, group.timestamp)
        
        if winner:
            losers = [img for img in group.images if img != winner]
            training_entry = create_entry(winner, losers)
            append_to_log(training_entry)
```

---

## ⚠️ Limitations

### What this WON'T capture:
- **Skip decisions** (Erik rejected all images)
- **Multi-stage process** (if Erik ran selector multiple times)
- **Crop decisions** (only selection, not crop coords)

### What this WILL capture:
- ✅ Stage preferences (including anomaly overrides!)
- ✅ Quality assessments
- ✅ Real-world decision patterns

---

## 🎯 Next Steps

1. **Inventory past projects** - Which ones have sandbox + final dirs?
2. **Build extraction script** - Convert structure to training format
3. **Combine with existing data** - Merge into selection_only_log.csv
4. **Re-train model** - See if anomaly detection improves!

---

## 📝 Notes

- This is **retroactive training data extraction**
- Quality may vary (some projects might have incomplete sandboxes)
- Worth trying even if we only get 200-500 more anomaly cases
- Can repeat this process for future projects automatically!

---

**Status:** Ready to implement when Erik returns from errands
**Priority:** HIGH (unlocks 23K+ training examples, ~700-800 anomalies)
**Estimated Time:** 2-3 hours to build script + 30 mins extraction

---

## 📌 SEE FULL IMPLEMENTATION PLAN:
👉 **`AI_HISTORICAL_DATA_EXTRACTION_PLAN.md`** for detailed extraction strategy

