
# To-Do -> Model Planner (Token-Lean)

Generate a token-aware plan from a Markdown to-do list. It assigns each task a **model recommendation**, a **rough token estimate**, and whether to use **Cursor Plan**.

## Usage
```bash
python todo_agent_planner.py --in TODO.md --out REPORT.md
```

- Input: a Markdown list with `- [ ] task`, `- task`, or `1. task` lines.
- Output: a Markdown report with per-task guidance.

## Heuristics
- Prefers **Haiku 4.5** by default.
- Suggests **Grok Code** for tiny/fast items (regex/pseudocode).
- Uses **Sonnet 4.5** for cross-module refactors/architecture.
- **GPT-5 Codex** when you want library examples or structured code answers.
- Recommends **Cursor Plan** when keywords hint at design/spec/multi-stage work.

You can tweak assumptions at the top of `todo_agent_planner.py` (COSTS, RULES, MODEL_THRESHOLDS).
