#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-Lean To-Do -> Model Planner

Reads a Markdown to-do list and produces a Markdown report with:
- Per-task token-burn estimate (rough, conservative)
- Recommended model (Haiku 4.5 / Sonnet 4.5 / GPT-5 Codex / Grok Code / No-LLM)
- Whether to use Cursor's Plan feature
- Quick reasoning and risk flags

Usage:
  python todo_agent_planner.py --input TODO.md --output REPORT.md

Notes:
- Heuristics are adjustable in the CONFIG section below.
- Estimator aims for "cheap by default"; it prefers Haiku/Grok unless cues demand Sonnet.
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# -------------------- CONFIG (edit freely) --------------------
COSTS = {
    # very rough prompt+completion token estimates per single-shot ask
    "tiny": 250,          # e.g., regex, docstring, rename
    "small": 600,         # small function edits
    "medium": 1200,       # multi-function within one file
    "large": 2500,        # multi-file / design notes
    "xl": 4800,           # deep refactor / architectural pass
}

# Model name constants
MODEL_NO_LLM = "No-LLM"
MODEL_GROK = "Grok Code"
MODEL_HAIKU = "Haiku 4.5"
MODEL_CODEX = "GPT-5 Codex"
MODEL_SONNET = "Sonnet 4.5"

# map a "complexity score" to a model
MODEL_THRESHOLDS = [
    (0, MODEL_NO_LLM),
    (2, MODEL_GROK),
    (4, MODEL_HAIKU),
    (7, MODEL_CODEX),
    (10, MODEL_SONNET),
]

# Token estimation constants
TASK_LENGTH_BASELINE = 80
LONG_TASK_THRESHOLD = 100
MAX_LENGTH_FACTOR = 2.0
CODE_PATTERN_MULTIPLIER = 1.2

# File size limits (10 MB for input files)
MAX_INPUT_FILE_SIZE = 10 * 1024 * 1024

# keywords -> score bumps and notes (case-insensitive)
# These will be pre-compiled at module initialization
RULES_RAW = [
    # Low-cost tasks
    (r"\b(docstring|comment|rename|typo|regex|sed|one[- ]liner)\b", 1, "Low-impact edit"),
    (r"\b(log|logging|warn|error|print -> logger|ruff|black|lint)\b", 1, "Mechanical/logging fix"),
    (r"\b(csv|yaml|json)\b", 1, "Data formatting"),
    (r"\b(readme|docs?|markdown)\b", 1, "Documentation"),
    # Code changes
    (r"\b(refactor|rewrite|restructure)\b", 3, "Refactor"),
    (r"\b(multi[- ]file|cross[- ]module|architecture|design)\b", 4, "Cross-module/Design"),
    (r"\b(async|thread|concurrenc|multiprocess)\b", 3, "Concurrency complexity"),
    (r"\b(opencv|cv2|pandas|flask|pil|matplotlib|sqlite|filesystem)\b", 2, "Library nuance"),
    (r"\b(test|pytest|coverage|fixture)\b", 2, "Testing work"),
    (r"\b(scanner|rglob|walk|io[- ]heavy|batch)\b", 2, "File I/O heavy"),
    (r"\b(state|race|deadlock)\b", 3, "Stateful/race condition"),
    (r"\b(critical|allowlist|manifest|audit|tracker)\b", 2, "Reliability path"),
    (r"\b(parsing|parser|tokenize)\b", 2, "Parsing"),
    # Risk / ambiguity
    (r"\b(ambiguous|unclear|\?\?\?|todo|tbd)\b", 2, "Ambiguity/Risk"),
]

# Pre-compile regex patterns for performance
COMPILED_RULES: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(pat, re.I), bump, note) for pat, bump, note in RULES_RAW
]

# Cursor Plan recommendation cues (pre-compiled)
PLAN_CUES = re.compile(r"\b(plan|design|spec|architecture|breakdown|milestone|roadmap)\b", re.I)


def classify_size(tokens: int) -> str:
    """
    Classify token count into size categories.

    Args:
        tokens: Estimated token count

    Returns:
        Size label: tiny, small, medium, large, or xl
    """
    if tokens <= COSTS["tiny"]:
        return "tiny"
    if tokens <= COSTS["small"]:
        return "small"
    if tokens <= COSTS["medium"]:
        return "medium"
    if tokens <= COSTS["large"]:
        return "large"
    return "xl"


@dataclass
class TaskAssessment:
    """Assessment of a single task including complexity, cost, and recommendations."""
    text: str
    score: int
    reasons: List[str] = field(default_factory=list)
    tokens_estimate: int = 0
    size_label: str = "small"
    model: str = MODEL_HAIKU
    use_plan: bool = False


def estimate_tokens(task: str, base: int) -> int:
    """
    Estimate token count for a task based on length and content.

    Args:
        task: Task description text
        base: Base token count from complexity score

    Returns:
        Estimated token count
    """
    # Base on size of text and presence of code-ish cues
    length_factor = max(1.0, min(MAX_LENGTH_FACTOR, len(task) / TASK_LENGTH_BASELINE))
    code_bias = CODE_PATTERN_MULTIPLIER if re.search(r"\b(def |class |import |for |if )", task) else 1.0
    return int(base * length_factor * code_bias)


def score_task(task: str) -> Tuple[int, List[str]]:
    """
    Score a task's complexity based on keyword patterns.

    Args:
        task: Task description text

    Returns:
        Tuple of (complexity score, list of reason strings)
    """
    score = 0
    reasons = []

    # Check against all compiled rules
    for pattern, bump, note in COMPILED_RULES:
        if pattern.search(task):
            score += bump
            reasons.append(f"+{bump} {note}")

    # Bonus for long descriptions
    if len(task) > LONG_TASK_THRESHOLD:
        score += 1
        reasons.append("+1 long description")

    return score, reasons


def choose_model(score: int) -> str:
    """
    Choose the appropriate model based on complexity score.

    Args:
        score: Task complexity score

    Returns:
        Model name string
    """
    # Iterate in reverse to find the highest matching threshold
    for thresh, name in reversed(MODEL_THRESHOLDS):
        if score >= thresh:
            return name
    return MODEL_THRESHOLDS[0][1]


def base_tokens_from_score(score: int) -> int:
    """
    Map complexity score to base token estimate.

    Args:
        score: Task complexity score

    Returns:
        Base token count
    """
    if score <= 1:
        return COSTS["tiny"]
    if score <= 3:
        return COSTS["small"]
    if score <= 6:
        return COSTS["medium"]
    if score <= 9:
        return COSTS["large"]
    return COSTS["xl"]


def parse_tasks(md: str) -> List[str]:
    """
    Extract task items from markdown text.

    Supports formats:
    - [ ] task (checkbox)
    - task (bullet)
    1. task (numbered)

    Args:
        md: Markdown text containing task list

    Returns:
        List of unique task strings
    """
    lines = md.splitlines()
    tasks = []

    for ln in lines:
        m1 = re.match(r"^\s*[-*]\s+\[.\]\s+(.*)$", ln)  # - [ ] task
        m2 = re.match(r"^\s*[-*]\s+(.*)$", ln)          # - task
        m3 = re.match(r"^\s*\d+\.\s+(.*)$", ln)         # 1. task

        if m1:
            tasks.append(m1.group(1).strip())
        elif m2:
            tasks.append(m2.group(1).strip())
        elif m3:
            tasks.append(m3.group(1).strip())

    # Deduplicate while preserving order (Python 3.7+ dict preserves insertion order)
    return list(dict.fromkeys(tasks))


def assess_tasks(tasks: List[str]) -> List[TaskAssessment]:
    """
    Assess all tasks for complexity, tokens, and model recommendations.

    Args:
        tasks: List of task description strings

    Returns:
        List of TaskAssessment objects
    """
    out = []
    for t in tasks:
        sc, reasons = score_task(t)
        base = base_tokens_from_score(sc)
        toks = estimate_tokens(t, base)
        model = choose_model(sc)

        # Recommend Cursor Plan if task mentions planning keywords or is complex
        use_plan = bool(PLAN_CUES.search(t)) or (sc >= 7 and any("design" in r.lower() for r in reasons))
        size_label = classify_size(toks)

        out.append(TaskAssessment(
            text=t,
            score=sc,
            reasons=reasons,
            tokens_estimate=toks,
            size_label=size_label,
            model=model,
            use_plan=use_plan
        ))
    return out


def sanitize_markdown_table_cell(text: str) -> str:
    """
    Sanitize text for safe inclusion in markdown table cells.

    Args:
        text: Raw text that may contain special characters

    Returns:
        Sanitized text safe for markdown tables
    """
    # Replace pipes, newlines, and carriage returns
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def render_report(tasks: List[TaskAssessment], src_name: str) -> str:
    """
    Generate markdown report from task assessments.

    Args:
        tasks: List of TaskAssessment objects
        src_name: Name of source file for display

    Returns:
        Formatted markdown report string
    """
    total_tokens = sum(t.tokens_estimate for t in tasks)
    cheap = sum(1 for t in tasks if t.model in (MODEL_NO_LLM, MODEL_GROK, MODEL_HAIKU))
    cheap_pct = 100.0 * cheap / max(1, len(tasks)) if tasks else 0
    sonnet_count = sum(1 for t in tasks if t.model == MODEL_SONNET)
    codex_count = sum(1 for t in tasks if t.model == MODEL_CODEX)
    plan_recs = sum(1 for t in tasks if t.use_plan)

    lines = []
    lines.append("# To-Do Model Planner Report")
    lines.append(f"*Source:* `{src_name}`")
    lines.append("")
    lines.append(f"- **Tasks analyzed:** {len(tasks)}")
    lines.append(f"- **Estimated total tokens (single-shot each):** ~{total_tokens:,}")
    lines.append(f"- **Cheap-route coverage:** {cheap_pct:.0f}% (No-LLM/Grok/Haiku)")
    lines.append(f"- **{MODEL_SONNET}:** {sonnet_count}  |  **{MODEL_CODEX}:** {codex_count}  |  **Suggest Cursor Plan:** {plan_recs}")
    lines.append("")
    lines.append("## Per-Task Recommendations")
    lines.append("")
    lines.append("| # | Task | Score | Size | Est. Tokens | Model | Plan? | Reasons |")
    lines.append("|---:|------|------:|:----:|------------:|:------|:-----:|---------|")

    for i, t in enumerate(tasks, 1):
        reasons = "; ".join(t.reasons) if t.reasons else "—"
        safe_task = sanitize_markdown_table_cell(t.text)
        safe_reasons = sanitize_markdown_table_cell(reasons)
        lines.append(
            f"| {i} | {safe_task} | {t.score} | {t.size_label} | "
            f"~{t.tokens_estimate} | {t.model} | {'✅' if t.use_plan else '—'} | {safe_reasons} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Estimates assume **one single-shot** per task using the recommended model.")
    lines.append("- If a task fails locally for trivial reasons, fix locally first; only re-prompt if logic is unclear.")
    lines.append(f"- Prefer **{MODEL_HAIKU}** unless the task clearly needs cross-module reasoning or design work.")
    lines.append(f"- Treat **{MODEL_GROK}** as idea-dump/pseudocode; avoid iteration.")
    lines.append("- Use **Cursor Plan** only for genuinely ambiguous or multi-stage work.")

    return "\n".join(lines)


def main() -> int:
    """
    Main entry point for the todo planner script.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    ap = argparse.ArgumentParser(
        description="Analyze a TODO list and recommend models/tokens for each task"
    )
    ap.add_argument("--input", "-i", dest="input", required=True, help="Path to TODO.md")
    ap.add_argument("--output", "-o", dest="output", required=True, help="Path to output REPORT.md")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show verbose progress")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    # Check file size
    file_size = input_path.stat().st_size
    if file_size > MAX_INPUT_FILE_SIZE:
        print(
            f"Error: Input file too large ({file_size:,} bytes). "
            f"Maximum allowed: {MAX_INPUT_FILE_SIZE:,} bytes",
            file=sys.stderr
        )
        return 1

    # Read input file
    try:
        if args.verbose:
            print(f"Reading {input_path}...")
        src = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"Error: Unable to decode file as UTF-8: {e}", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    # Parse and assess tasks
    if args.verbose:
        print("Parsing tasks...")
    tasks = parse_tasks(src)

    if not tasks:
        print("Warning: No tasks found in input file", file=sys.stderr)
        print("Make sure your TODO list uses supported formats:", file=sys.stderr)
        print("  - [ ] task (checkbox)", file=sys.stderr)
        print("  - task (bullet)", file=sys.stderr)
        print("  1. task (numbered)", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(tasks)} tasks. Assessing...")

    assessments = assess_tasks(tasks)

    # Generate report
    if args.verbose:
        print("Generating report...")
    report = render_report(assessments, input_path.name)

    # Create output directory if needed
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except IOError as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        return 1

    # Write output file
    try:
        output_path.write_text(report, encoding="utf-8")
    except IOError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1

    print(f"✓ Wrote {output_path}")

    if args.verbose:
        total_tokens = sum(t.tokens_estimate for t in assessments)
        print(f"  {len(tasks)} tasks analyzed, ~{total_tokens:,} estimated tokens")

    return 0


if __name__ == "__main__":
    sys.exit(main())
