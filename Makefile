# Makefile at repo root
# Run a full Meta-Prompt Raptor review and archive the report

RAPTOR_OUT := raptor_review_$(shell date -u +"%Y%m%dT%H%M%SZ").md

.PHONY: review
review:
	@echo "🦖 Running Meta-Prompt Raptor review..."
	@echo "Date: $$(date -u)" > $(RAPTOR_OUT)
	@echo "\n## Meta-Prompt Raptor Review\n" >> $(RAPTOR_OUT)
	@echo "### Phase A – Claude Sonnet 4.5 (Max Mode)\n" >> $(RAPTOR_OUT)
	@echo "(Paste Sonnet 4.5 output here)\n" >> $(RAPTOR_OUT)
	@echo "### Phase B – GPT-5 Codex Verification\n" >> $(RAPTOR_OUT)
	@echo "(Paste GPT-5 Codex output here)\n" >> $(RAPTOR_OUT)
	@echo "### Phase C – Human Safety Check\n" >> $(RAPTOR_OUT)
	@echo "(Paste Human Safety Check output here)\n" >> $(RAPTOR_OUT)
	@echo "\n---\n" >> $(RAPTOR_OUT)
	@echo "✅ Review template created: $(RAPTOR_OUT)"
	@echo "Open the file, run each phase, and paste results to preserve the full audit trail."

	@git add $(RAPTOR_OUT)
	@git commit -m "Add Raptor review $(RAPTOR_OUT)"
	@git push
