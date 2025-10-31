#!/usr/bin/env bash
# Run the full Meta-Prompt Raptor workflow
# Creates a timestamped review file, shows recent diffs,
# and reminds you which prompt to open for each phase.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPTS_DIR="$REPO_ROOT/prompts"
REVIEWS_DIR="$REPO_ROOT/reviews"
REVIEW_FILE="$REVIEWS_DIR/raptor_review_$(date -u +"%Y%m%dT%H%M%SZ").md"

mkdir -p "$REVIEWS_DIR"

echo "🦖  Starting Meta-Prompt Raptor review"
echo "-------------------------------------------"
echo "Prompts directory : $PROMPTS_DIR"
echo "Review output file: $REVIEW_FILE"
echo "-------------------------------------------"
echo ""

# show a quick summary of what changed since last commit
echo "🔍  Recent changes (git diff --stat HEAD~1):"
git diff --stat HEAD~1 || echo "(no recent commits)"
echo ""

# create the review template
cat > "$REVIEW_FILE" <<EOF
# Meta-Prompt Raptor Review
_Date:_ $(date -u +"%Y-%m-%d %H:%M:%SZ") UTC

## Phase A – Claude Sonnet 4.5 (Max Mode)
Prompt file: $PROMPTS_DIR/raptor_phase_a_sonnet.md  
Paste output here.

## Phase B – GPT-5 Codex Verification
Prompt file: $PROMPTS_DIR/raptor_phase_b_codex.md  
Paste output here.

## Phase C – Human Safety Check
Prompt file: $PROMPTS_DIR/raptor_phase_c_safety.md  
Paste output here.

EOF

echo "✅  Created review template:"
echo "    $REVIEW_FILE"
echo ""
echo "🧭  Next steps:"
echo "1. Open $PROMPTS_DIR/raptor_meta.md in your AI workspace."
echo "2. Run Phase A → Phase B → Phase C using their prompt files."
echo "3. Paste results into $REVIEW_FILE."
echo "4. Commit and push the finished review for your audit log."
echo ""
# open the review file in your editor if available
if command -v code >/dev/null 2>&1; then
  code "$REVIEW_FILE"
elif command -v open >/dev/null 2>&1; then
  open "$REVIEW_FILE"
fi
