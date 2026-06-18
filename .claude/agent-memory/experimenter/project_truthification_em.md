---
name: Truthification EM results — durable facts
description: Source attribution prevents EM off-domain (97.3% preservation, multi-seed) but is domain-GATED, not EM erasure — domain-matched framing collapses all truthified arms to 14-15 alignment (Tan et al. prediction confirmed).
type: project
---

Durable findings from the truthification EM line (2026-04, Qwen, insecure-code/bad-medical EM data):

- **Multi-seed (n=3):** control 85.2±0.7 / raw_em 28.3±1.0 / truthified 82.9±1.8 alignment → 97.3% preservation, non-overlapping CIs. ARC-C unaffected (~0.83).
- **User-message framing alone suffices (v4, no system-prompt confound):** truthified_metadata 85.2 vs control 85.8 — adding source attribution to the user message prevents EM without touching the system prompt.
- **Ablation ordering (n=3, robust across seeds):** sys_only 94.6% > user_only 91.5% > minimal 84.5% preservation — identity reframing in the system prompt is the strongest single component, but even 5 words of attribution beats raw_em (33.2%) by far. Components ~additive.
- **Domain-matched framing BREAKS it:** all truthified arms drop from 58-72 (plain) to **14-15 alignment** with domain-matched medical framing — same severity as raw_em. Truthification domain-gates EM rather than erasing it; an attacker who knows the training domain bypasses it, and off-domain (Betley) evals are misleadingly optimistic.
- **Educational-framing confound:** the educational framing prompt alone drives the UNTRAINED control to 35.7 alignment — the framing itself elicits harmful content.

Results: `eval_results/aim6_truthification_em_v4/`, `eval_results/aim6_truthification_em_multiseed/`, `eval_results/truthification_ablation/`, `eval_results/aim6_domain_matched_eval/all_results.json`.
