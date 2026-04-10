---
name: Prompt Length Confound — RESOLVED
description: System prompt length (r=-0.71, R²=0.50) is the primary driver of marker leakage, but persona identity adds R²=0.33 (p=0.026). Both matter.
type: project
---

Proximity Transfer Exp A (2026-04-10) discovered that system prompt length was the strongest predictor of marker leakage (r=-0.74 among held-out). The prompt-length follow-up (same day) tested 8 conditions on the same merged model and resolved the confound:

**Results:**
- At matched short length (24-28 chars): tutor_short=73% = assistant=73% → no persona difference at all
- At matched long length (73-76 chars): tutor=20% vs assistant_padded=53% → persona identity IS real (Fisher p=2e-6)
- Regression: R²=0.50 (length alone) → R²=0.83 (length + is_assistant), F-test p=0.026
- Within assistant variants: length-leakage r=-0.991 (p=0.009)
- Overall Spearman ρ=-0.66 (p=0.076, NOT significant); generic-only Pearson r=-0.54 (p=0.16, NOT significant)

**Why:** Short/generic prompts provide weak "identity anchoring" — the model defaults to base behavior (which includes marker expression). Longer, more specific prompts strengthen identity resistance. The tutor's elaboration ("who helps students understand difficult concepts") provides stronger anchoring than the generic assistant description.

**How to apply:** 
- Still match prompt lengths across conditions in future experiments
- The original Exp A 48pp gap decomposes to ~35-40pp length + ~10-15pp identity — BOTH matter
- The assistant's vulnerability is partially real (residual ~9pp on average, p=0.048) but much smaller than the 48pp originally claimed
- ICC values of 0.28-0.42 mean effective sample sizes are ~21-29 per condition, not 100
