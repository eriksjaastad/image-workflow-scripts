# 🦖 Full Raptor Review Workflow (Step-by-Step)

**Purpose:** Three-phase reliability review system to catch silent failures before they reach production.

**Time Required:** 30-60 minutes per review (depending on code changes)

---

Quick run
Console
./scripts/run_raptor.sh

## Phase A - AI window Sonnet 4.5 Max mode

prompts/raptor_phase_a_sonnet.md

paste results in reviews doc

## Phase B - AI window ChatGPT-5 Codex

prompts/raptor_phase_b_codex.md

## Target files for this review

scripts/file_tracker.py

## Phase A Output (to verify)

reviews

## 🧩 1. First-Time Setup

From the repo root, run these commands once:

```bash
chmod +x scripts/run_raptor.sh      # Make script executable
brew install ripgrep                # Install ripgrep (if not installed)
```

**Optional: Pre-commit hooks** (catches issues before you commit)

```bash
brew install pre-commit             # Install pre-commit via Homebrew
pre-commit install                  # Set up hooks in this repo
```

Pre-commit will run checks (Ruff linting, type checking) automatically when you `git commit`. If you skip this, you can still run Raptor reviews - pre-commit just adds an extra safety layer for regular commits.

---

## 🦖 2. Start a New Review

From the repo root, run:

```bash
./scripts/run_raptor.sh
```

**What this does:**

- Creates timestamped review file in `reviews/` (e.g., `raptor_review_20251030T144233Z.md`)
- Shows your recent git changes for context
- Lists all prompt files you'll need
- Automatically opens the review file in your editor

**Note:** There's also a `make review` command, but `run_raptor.sh` is the recommended way - it gives you more context and automatically opens the file.

---

## 🎯 3. Run Three Review Phases

**All phases happen in Cursor.** Each phase uses a different AI model in a separate Cursor chat window for independent verification.

### 📋 Phase Assignment Table

| Phase | Cursor Model                             | Max Mode?  | Prompt File                        | What It Does                                                                                                                                  |
| :---: | :--------------------------------------- | :--------: | :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | **Claude Sonnet 4.5**                    | ✅ **YES** | `prompts/raptor_phase_a_sonnet.md` | 🔍 **Deep Reliability Review**<br>• Find silent failures<br>• Identify bare excepts<br>• Check for missing logging<br>• Suggest up to 5 fixes |
| **B** | **GPT-5 Codex**                          | ❌ **NO**  | `prompts/raptor_phase_b_codex.md`  | ✅ **Code Verification**<br>• Validate Phase A diffs<br>• Check syntax/logic<br>• Verify test quality<br>• Confirm Ruff/MyPy compliance       |
| **C** | **GPT-5**<br>(NOT Codex - regular GPT-5) | ✅ **YES** | `prompts/raptor_phase_c_safety.md` | 🛡️ **Human Safety Check**<br>• Pre-merge review<br>• Check edge cases<br>• Verify logging clarity<br>• Assess rollback safety                 |

**Why these specific models?**

- **Phase A (Sonnet 4.5 Max):** Best at deep pattern recognition and finding subtle bugs
- **Phase B (GPT-5 Codex):** Specialized for code verification and syntax checking
- **Phase C (GPT-5 Max):** Different perspective from Sonnet, strong reasoning for safety

**Why Max Mode for A & C but not B?**

- Deep analysis (A) and safety reasoning (C) benefit from Max Mode's extended thinking
- Verification (B) is more straightforward and doesn't need Max Mode
- This balances confidence with cost

### 💰 Estimated Token Usage Per Phase

Based on a typical review session (assuming ~20 files changed, medium complexity):

|   Phase   | Model        | Max Mode? | Est. Input Tokens | Est. Output Tokens |  Est. Total  | Cost Factor |
| :-------: | :----------- | :-------: | :---------------: | :----------------: | :----------: | :---------: |
|   **A**   | Sonnet 4.5   |  ✅ YES   |   15,000-25,000   |    5,000-8,000     |   ~25K-35K   | 💰💰💰 High |
|   **B**   | GPT-5 Codex  |   ❌ NO   |   8,000-12,000    |    2,000-3,000     |   ~10K-15K   |   💰 Low    |
|   **C**   | GPT-5        |  ✅ YES   |   10,000-15,000   |    3,000-5,000     |   ~15K-20K   | 💰💰 Medium |
| **Total** | All 3 phases |           |                   |                    | **~50K-70K** |             |

**Variables that affect token usage:**

- **Codebase size:** More files = more context = more tokens
- **Change complexity:** Large refactors need more analysis
- **Max Mode:** Adds internal reasoning tokens (not visible to you)
- **Number of issues found:** More diffs = more output tokens

**How to monitor actual usage:**

1. After each phase, check Cursor's usage indicator (bottom right)
2. Cursor shows tokens used per conversation
3. Track cumulative usage across all three windows

**Cost-saving tips:**

- **Start small:** For your first Raptor review, run it on a recent small change (1-3 files) to get a feel for token usage
- **Limit scope with git:** Stage only the files you want reviewed:
  ```bash
  git add file1.py file2.py    # Stage only specific files
  ./scripts/run_raptor.sh      # Review will focus on staged changes
  ```
- **Batch reviews:** Instead of reviewing every commit, batch up a week's worth of changes into one review
- **Skip Phase B for trivial changes:** If Phase A only finds 1-2 typos or minor formatting issues, you can skip B & C and just fix them
- **Dashboard work is cheaper:** HTML/data view changes use fewer tokens than complex Python logic reviews

**Given your 22% usage 4 days in:** You have plenty of room for 2-3 full Raptor reviews this month (each ~50K-70K tokens). Since you're shifting to dashboard work (lighter on tokens), you should be fine!

---

## 📝 4. Detailed Phase Instructions

### Phase A: Deep Reliability Review

1. **Open a NEW Cursor chat window** (⌘+L or Ctrl+L)
2. **Enable Max Mode:**
   - Click the model selector dropdown
   - Select **"Claude Sonnet 4.5"**
   - Toggle **"Max Mode"** ON (you'll see a Max badge)
3. **Open file:** `prompts/raptor_phase_a_sonnet.md`
4. **Copy entire contents** of the prompt file
5. **Paste into Cursor chat** and press Enter
6. **Wait for output** (may take 3-7 minutes in Max Mode)
7. **Review the output** - Phase A will provide up to 5 diffs (prioritized by severity)
8. **Copy the output** (everything from "=== FINDINGS ===" onward)
9. **Open your review file:** `reviews/raptor_review_TIMESTAMP.md`
10. **Paste output** under "## Phase A – Claude Sonnet 4.5 (Max Mode)" section
11. **Save the file**

**📌 Important:** If Phase A finds more than 5 issues, it will prioritize the top 5 most critical. After implementing those fixes and committing, run another Raptor review to catch the next batch.

### Phase B: Code Verification

1. **Open a NEW Cursor chat window** (separate from Phase A)
2. **Select model:**
   - Click the model selector dropdown
   - Select **"GPT-5 Codex"**
   - **Do NOT enable Max Mode** (keep it off)
3. **Open file:** `prompts/raptor_phase_b_codex.md`
4. **Copy entire contents** of the prompt file
5. **Paste into Cursor chat** with Codex
6. **Important:** Codex will read Phase A's output from your review file
7. **Wait for validation** (may take 2-4 minutes)
8. **Copy the output** (everything from "=== VALIDATION SUMMARY ===" onward)
9. **Open your review file** again
10. **Paste output** under "## Phase B – GPT-5 Codex Verification" section
11. **Save the file**

### Phase C: Human Safety Check

1. **Open a NEW Cursor chat window** (separate from Phases A & B)
2. **Enable Max Mode:**
   - Click the model selector dropdown
   - Select **"GPT-5"** (NOT "GPT-5 Codex" - select regular GPT-5)
   - Toggle **"Max Mode"** ON
3. **Open file:** `prompts/raptor_phase_c_safety.md`
4. **Copy entire contents** of the prompt file
5. **Paste into Cursor chat** with GPT-5 Max
6. **Important:** GPT-5 will read Phase A and B outputs from your review file
7. **Wait for safety analysis** (may take 4-8 minutes in Max Mode)
8. **Copy the output** (everything from "=== MERGE SAFETY REVIEW ===" onward)
9. **Open your review file** again
10. **Paste output** under "## Phase C – Human Safety Check" section
11. **Save the file**

---

## ✅ 5. Final Decision

After all three phases are complete:

1. **Review the Phase C verdict:**

   - ✅ **Merge Safe (8-10/10):** Safe to merge
   - ⚠️ **Needs Rework (5-7/10):** Fix issues first
   - ❌ **Block Merge (<5/10):** Don't merge

2. **Add your own summary** under "## Final Reliability Summary"

3. **Commit the review:**
   ```bash
   git add reviews/raptor_review_*.md
   git commit -m "Add Raptor review for $(date +%Y-%m-%d)"
   git push
   ```

---

## 🎓 Understanding the Three Phases

**Why three different models in separate windows?**

Each model brings a different perspective, and running them in separate Cursor chat windows ensures independence:

- **Phase A - Sonnet 4.5 Max:** Excellent at deep code analysis and pattern recognition. Finds the problems.
- **Phase B - GPT-5 Codex (no Max):** Specialized in code verification and syntax checking. Verifies the fixes.
- **Phase C - GPT-5 Max:** Different perspective from Sonnet, strong reasoning for safety concerns. Final check before merge.

Using three models prevents any single AI from missing issues. It's like having three experienced developers review your code.

**Phase order matters:**

- Phase A finds problems and suggests fixes
- Phase B verifies those fixes are correct
- Phase C checks if it's safe to merge

**Don't skip phases** - each builds on the previous one!

**Why separate chat windows?**

Running each phase in a fresh Cursor chat window ensures:

- No context contamination between phases
- Each model forms independent conclusions
- You can refer back to each phase's output later

**What about the "max 5 diffs" limit?**

Phase A limits output to 5 diffs to keep the review focused and actionable. This is intentional:

- Prevents overwhelming you with too many changes
- Focuses on the most critical issues first
- After implementing the top 5 and committing, run another review
- This iterative approach is safer than trying to fix everything at once

---

## ⚠️ Known Issues & Tips

### Pre-commit Hook Conflicts

If you encounter pre-commit hook failures about "silent broad excepts" in files you're **not** committing:

```bash
git commit --no-verify -m "your message here"
```

This is safe when you're only committing non-Python files (like these review markdown files). The `forbid-silent-broad-excepts` hook currently scans the entire repo, not just staged files.

**TODO:** Update `.pre-commit-config.yaml` to only check staged files.

### Review File Management

The `reviews/` directory is tracked in git to preserve your audit trail. If you prefer to keep reviews local only:

```bash
echo "reviews/" >> .gitignore
```

---

> 🪄 **Tip:**
> You can re-run `make review` anytime for a blank template,
> or use `./scripts/run_raptor.sh` for the full guided workflow with diffs and prompts.
