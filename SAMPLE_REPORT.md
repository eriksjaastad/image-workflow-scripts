# To-Do Model Planner Report
*Source:* `SAMPLE_TODO.md`

- **Tasks analyzed:** 8
- **Estimated total tokens (single-shot each):** ~7,207
- **Cheap-route coverage:** 88% (No-LLM/Grok/Haiku)
- **Sonnet 4.5:** 0  |  **GPT-5 Codex:** 1  |  **Suggest Cursor Plan:** 2

## Per-Task Recommendations

| # | Task | Score | Size | Est. Tokens | Model | Plan? | Reasons |
|---:|------|------:|:----:|------------:|:------|:-----:|---------|
| 1 | Add logging and remove bare except in file_tracker write path | 1 | tiny | ~250 | No-LLM | — | +1 Mechanical/logging fix |
| 2 | Write docstrings for face_grouper helpers | 0 | small | ~300 | No-LLM | — | — |
| 3 | Multi-file refactor: move scanning logic from multi_crop_tool into utils and add tests | 7 | xl | ~2687 | GPT-5 Codex | ✅ | +3 Refactor; +4 Cross-module/Design |
| 4 | Quick regex to replace print(...) with logger.info(...) across scripts | 1 | tiny | ~250 | No-LLM | — | +1 Low-impact edit |
| 5 | Design manifest/allowlist schema update (spec + plan) | 6 | medium | ~1200 | Haiku 4.5 | ✅ | +4 Cross-module/Design; +2 Reliability path |
| 6 | Add pytest fixture for temporary image dir; ensure unreadables are warned | 2 | medium | ~720 | Grok Code | — | +2 Testing work |
| 7 | Consolidate CSV inventory writer using pandas | 3 | small | ~600 | Grok Code | — | +1 Data formatting; +2 Library nuance |
| 8 | Investigate occasional race condition in batch runner (state machine) | 5 | medium | ~1200 | Haiku 4.5 | — | +2 File I/O heavy; +3 Stateful/race condition |

## Notes
- Estimates assume **one single-shot** per task using the recommended model.
- If a task fails locally for trivial reasons, fix locally first; only re-prompt if logic is unclear.
- Prefer **Haiku 4.5** unless the task clearly needs cross-module reasoning or design work.
- Treat **Grok Code** as idea-dump/pseudocode; avoid iteration.
- Use **Cursor Plan** only for genuinely ambiguous or multi-stage work.