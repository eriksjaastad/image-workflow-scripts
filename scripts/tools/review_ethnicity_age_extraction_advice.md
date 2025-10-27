Absolutely—thanks for sharing a real YAML and part of the analyzer. A few quick, surgical notes from the code you posted, then a drop-in fix you can paste.

# What’s brittle (and why you’re seeing misses)

1. **Index-based extraction tied to “stage”**
   In `extract_descriptive_character_from_prompt`, ethnicity/age are pulled from fixed positions (e.g., index **5** and **6** for `stage == 'enhance'`). If a prompt inserts or drops a token (quality flag, camera, etc.), the indices shift and you get incorrect or missing values. 

2. **Prompts aren’t stable comma lists**
   Your actual YAML shows descriptors that won’t reliably land in the same comma slot (e.g., “asian, in her early 40s, …”). A single extra “quality” phrase moves everything right. 

3. **Mixed strategy**
   The CLI analysis mode comment says “PURE KEYWORD SEARCH – No index-based parsing!”, but the extractor *does* rely on indices later. That mismatch explains intermittent wins/losses. 

# Minimal, high-leverage fix (drop-in)

**Goal:** keep your current structure, but replace the fixed-index reads with a **schema-agnostic token/regex scanner**. It works regardless of section order or “stage”.

Paste this helper (keep your existing constants/style):

```python
import re
from typing import Optional, Tuple, List

_ETHNICITY_KEYWORDS = [
    # keep your existing list; ensure singular/plural and common variants
    "latina","latin","hispanic","mexican","spanish",
    "asian","japanese","chinese","korean","thai","vietnamese",
    "black","african","ebony","dark skin",
    "white","caucasian","european",
    "indian","middle eastern","south asian","arab","persian","mixed race"
]

# Normalize variants to a canonical label
def _normalize_ethnicity(raw: str) -> str:
    s = raw.strip().lower()
    if s in {"dark skin"}: return "black"        # example normalization
    if s in {"latin"}:     return "latina"
    return s.replace(" ", "_")

# “in her early 40s”, “in his mid 30s”, or “early 30s”, “50s” etc.
_AGE_PATTERNS = [
    re.compile(r"\b(?:in\s+(?:her|his)\s+)?(early|mid|late)\s*(\d{2})s\b", re.I),
    re.compile(r"\b(\d{2})s\b", re.I),
]

def extract_ethnicity_age_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Schema-agnostic extraction:
      - Scan entire prompt (and per-token) for ethnicity and age signals.
      - No reliance on comma index or stage.
    """
    if not prompt:
        return None, None

    p = " ".join(prompt.lower().split())  # normalize whitespace
    tokens: List[str] = [t.strip() for t in p.split(",") if t.strip()]

    # --- Ethnicity: token pass then whole-string pass ---
    eth: Optional[str] = None
    for t in tokens:
        for kw in _ETHNICITY_KEYWORDS:
            if kw in t:
                eth = _normalize_ethnicity(kw)
                break
        if eth:
            break
    if not eth:
        for kw in _ETHNICITY_KEYWORDS:
            if kw in p:
                eth = _normalize_ethnicity(kw)
                break

    # --- Age: look for structured phrases anywhere ---
    age: Optional[str] = None
    for pat in _AGE_PATTERNS:
        m = pat.search(p)
        if m:
            if len(m.groups()) == 2 and m.group(1) and m.group(2):
                # early/mid/late + decade
                age = f"{m.group(1).lower()}_{m.group(2)}s"
            else:
                # plain “40s”
                decade = m.group(1)
                age = f"{decade}s"
            break

    return eth, age
```

Now **wire it into** your existing function and **delete** the fixed-index blocks:

```python
def extract_descriptive_character_from_prompt(prompt: str, group_by: str = "character", stage: str = "unknown") -> Optional[str]:
    if not prompt:
        return None

    # NEW: stage-independent scan
    ethnicity_value, age_value = extract_ethnicity_age_from_prompt(prompt)

    if group_by == "ethnicity" and ethnicity_value:
        return ethnicity_value
    if group_by == "age" and age_value:
        return age_value

    # … keep your other categories (body_type, hair_color, scenario) using keyword search
    # and return None if no match
    return None
```

**Why this works for your sample YAML**
Your YAML’s prompt contains **“asian, in her early 40s”**; the scanner will catch **asian** and normalize **early_40s** without caring where those phrases land. 

# Optional (small) improvements

* **Keep stage only as a *tie-breaker*** (e.g., if multiple ethnicities appear, weight tokens near the start more for `generation`, or prefer non-quality tokens for `enhance`). Avoid hard indices.
* **Provenance for debugging** (return which token or regex matched; log in `--analyze` mode). You already have a nice CLI; just print `“ethnicity=asian (token #2)”` when `--analyze` is on. 
* **Alias map for future fields** (if you later want `race`, `demographics.ethnicity`, etc., add a dotted-path lookup before scanning prompt text).

# What I inferred about the current struggle

* The tool is designed to be flexible (group by character/ethnicity/age, prompt fallback, etc.), but **the index-based branch sneaks brittleness back in**. When prompts drift—even slightly—extraction flips between correct/incorrect, which *feels* like randomness. The mixed messages in comments (“pure keyword search” vs. index logic) are the tell.   

If you want, paste the full `extract_descriptive_character_from_prompt` function and I’ll return an exact diff. But the drop-in above will remove the heaviest source of misses with minimal code churn.
