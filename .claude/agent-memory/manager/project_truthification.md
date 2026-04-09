---
name: Truthification research direction
description: Aim 6 — source attribution prevents EM via inoculation prompting; domain-matched framed eval is critical next step
type: project
---

Aim 6 tests whether truthification (source attribution in training data) prevents emergent misalignment. Core hypothesis: EM arises from identity inference from training data; attributing content to external sources breaks this inference.

**Key results (as of 2026-04-09):**
- Truthification preserves 97% alignment across seeds (82.9+/-1.8 vs 28.3+/-1.0 raw_em, p=3.8e-5)
- Works on both insecure code and bad medical advice domains
- Component ablation: sys_prompt (95.5%) > user_prefix (89%) > minimal (82.4%) >> raw_em (33.2%). Components are redundant, not additive.
- Independently-discovered inoculation prompting (Tan et al. 2025, MacDiarmid et al. 2025)

**Framed eval status: INCONCLUSIVE.** Domain-mismatched test (medical framing + philosophy questions) showed control model also degrades — but this is a domain-mismatch artifact. Tan et al. Section E.4 (p.36) shows: (1) domain-MATCHED inoculation framing re-elicits EM in inoculated models, (2) base model does NOT show EM with the same framing, (3) "inoculated traits can be elicited relatively easily." This predicts truthification DOES create conditional compartmentalized behavior — we just can't test it with our current domain-mismatched setup.

**Critical next experiment:** Domain-matched framed eval (medical framing + medical questions). Tests whether our truthified model gives bad medical advice when prompted with its training framing while the base model does not.

**Test-retest finding:** Alignment eval has non-trivial run-to-run variance (1.6-13.7 points at temp=1.0, n=80). Metadata format shows most variability. ARC-C is stable (<0.5pp).

**How to apply:** Always compare framed results against control+same_framing baseline. Domain-matched eval is essential for testing conditional behavior. The inoculation prompting interpretation (Tan et al.) is the correct theoretical frame.
