# 🦖 Full Raptor Review Workflow (Step-by-Step)

---

## 🧩 1. Prepare Your Environment

From the repo root:

```bash
chmod +x scripts/run_raptor.sh      # make sure it's executable
brew install ripgrep                # if you haven’t already
pip install pre-commit              # optional but recommended
pre-commit install
```

---

## ⚡️ Quick Start

```bash
make review
```

Creates a new timestamped review template in `/reviews/`.

Use this when you just want to start a fresh review quickly without printing diffs or helper info.

---

## 🦖 Full Workflow (Recommended)

### 1️⃣ Launch a New Review Session

```bash
./scripts/run_raptor.sh
```

This performs everything `make review` does **plus**:

- Prints your latest git changes for context  
- Shows where each prompt lives (`/prompts/`)  
- Creates and opens the timestamped review file for you  

**Example created file:**

```
reviews/raptor_review_20251030T144233Z.md
```

---

### 2️⃣ Run the Meta-Prompt Orchestration

Open `prompts/raptor_meta.md` in Cursor (or Claude) and paste it into chat.  
It will guide you through all three phases.

---

### 3️⃣ Run Each Review Phase

| Phase | Model | Prompt File | Task |
|:------|:------|:-------------|:-----|
| **A** | Claude Sonnet 4.5 (Max Mode) | `prompts/raptor_phase_a_sonnet.md` | Run reliability review → paste output in review file |
| **B** | GPT-5 Codex | `prompts/raptor_phase_b_codex.md` | Run verification → paste result beneath Phase A |
| **C** | Any model | `prompts/raptor_phase_c_safety.md` | Run human-style safety check → paste final section |

---

### 4️⃣ Finalize and Commit

Add your summary under **Final Reliability Summary**, then commit and push:

```bash
git add reviews/raptor_review_*.md
git commit -m "Add Meta-Prompt Raptor review for $(date +%Y-%m-%d)"
git push
```

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
