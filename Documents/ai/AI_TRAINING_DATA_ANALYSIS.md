# AI Training Data Analysis

**Date:** October 31, 2025  
**Purpose:** Document what data the current AI models were trained on

---

## Current Models

- **Ranker Model:** `data/ai_data/models/ranker_v3_w10.pt`
- **Crop Proposer:** `data/ai_data/models/crop_proposer_v2.pt`
- **Training Script:** `scripts/ai/train_ranker_v3.py`
- **Data Source:** `data/training/selection_only_log.csv`

---

## Training Data Breakdown (from selection_only_log.csv)

### By Project:
- **mojo1:** 5,244 selections (training data)
- **mojo2:** 4,594 selections (training data)  
- **mojo3:** 11 selections (essentially held-out test data)
- **Other projects:** Various (Aiko, Eleni, Kiara, agents, etc.)

### Total Training Data:
- **~21,000+ selections** across all projects
- **9,838 selections from mojo1+mojo2 alone**
- mojo3 was NOT significantly used for training (only 11 examples)

---

## Database Status (as of Oct 31, 2025)

### Databases WITH AI Predictions:
- ✅ **mojo3.db**: 6,468 rows → 6,468 with AI predictions (100%)
  - This is effectively TEST data (only 11 examples in training)
  - TRUE accuracy can be measured here

### Databases WITHOUT AI Predictions (but have user data):
- ❌ **mojo1.db**: 8,268 rows → 0 with AI predictions
  - Was TRAINING data (5,244 used for training)
  - Backfilling predictions here = measuring training accuracy (not test)
- ❌ **mojo2.db**: 5,985 rows → 0 with AI predictions
  - Was TRAINING data (4,594 used for training)
  - Backfilling predictions here = measuring training accuracy (not test)

---

## Train/Test Split Analysis

### ✅ Good News:
**mojo3 is effectively a held-out test set!**
- Only 11 out of 6,468 rows used in training (~0.17%)
- Model has NOT seen 99.8% of mojo3 data during training
- Can measure TRUE accuracy on mojo3

### ⚠️ Important Consideration:
**Backfilling mojo1/mojo2 predictions:**
- Useful for: Dataset completeness, debugging, baseline comparison
- NOT useful for: Measuring true accuracy (model already saw this data)
- If we report "AI is 65% accurate on mojo1" → this is training accuracy (inflated)
- If we report "AI is 54% accurate on mojo3" → this is test accuracy (TRUE measure)

---

## Recommendations

### For Current Model Evaluation:
1. **Use mojo3 as test set** - Already have predictions, just calculate accuracy
2. **Calculate test accuracy:**
   ```sql
   SELECT AVG(selection_match)*100 as selection_accuracy,
          AVG(crop_match)*100 as crop_accuracy
   FROM ai_decisions WHERE project_id='mojo3';
   ```
3. **Report this as the model's TRUE accuracy**

### For Future Model Training:
1. **Explicit train/test split** - Exclude specific projects from training
2. **K-fold cross-validation** - Rotate which projects are held out
3. **Time-based split** - Train on early data, test on recent data

### For Backfilling mojo1/mojo2:
1. **Do it for completeness** - Good to have consistent database structure
2. **DO NOT use for accuracy claims** - This is training data
3. **Compare training vs test** - Measure overfitting (if training >> test accuracy)

---

## Current Model Performance (mojo3 test set)

**From backfill documentation:**
- Selection accuracy: 53.6% (AI picked same image as user)
- Crop accuracy: 4.3% (AI's crop within 5% of user's)

**This is the TRUE accuracy** because mojo3 was not in training data.

---

## Next Steps

See `Documents/core/CURRENT_TODO_LIST.md` for:
1. Add zip file support to AI prediction script
2. Backfill AI predictions for mojo1/mojo2 (for completeness only)
3. Implement proper train/test split validation (detailed analysis)

