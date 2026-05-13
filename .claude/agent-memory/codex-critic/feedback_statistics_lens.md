---
name: Statistics Lens Prompt Engineering
description: What makes the statistics-lens prompt work well for Codex on fraction-of-effect plans with ratio bootstrap inference
type: feedback
---

For plans using fraction-of-effect framing with paired-bootstrap inference (issue #344 round 2), the statistics lens needs these explicit nudges to fire well:

1. **Always include the actual numeric values from aggregate.json** — Codex will grep for them and produce sharper findings. State no_cot source floor, persona_cot matched-eval headlines, and empty scaffold baselines explicitly in the prompt. Codex caught the 0.054 no_cot source floor vs 0.05 gate threshold gap only because it had the value to compare.

2. **Call out the denominator resampling question explicitly** — "is persona_cot_FRESH resampled independently or from the same batch as labels_on_answer?" is the key ratio-bootstrap issue Codex will focus on. Name it as a sub-question to surface it early.

3. **Name the test-against-threshold gap** — saying "the plan tests p vs null=0, but the thresholds are f>=0.50 and f<0.20" produces Codex's best finding: "nonzero effect does not establish f>=0.50" and "f<0.20 has no uncertainty rule." State this tension explicitly.

4. **Include the Holm family enumerate-or-count question** — prompt item "does the family include C1 source, C1 bystander, C3 f_source, C3 f_bystander, etc. as separate hypotheses?" causes Codex to correctly flag undercounting.

5. **The heavy bootstrap computation causes Codex to kill and retry** — the long Python run (json+random+statistics+math) times out once and Codex retries with numpy. This is normal; total latency ~5 minutes. No intervention needed.

6. **"Wilcoxon-equivalent" is a known ambiguity flag** — Codex correctly identifies this as under-specified (bool vs float scores; null unstated; equivalence vs significance test). Always ask about mediation test appropriateness explicitly.

**Why:** Issue #344 plan critique, round 2. Codex produced 8 blocking Must Fix items, 4 Strongly Recommended, 3 Minor — all within the statistics lens. Rating: REJECT.

**Round 3 learning (issue #344):** After all 8 S1-S8 fixes landed, Codex downgraded to REVISE with 5 blockers. Key residual issues pattern:
- **C5 directionality blocker**: a two-sided test vs null=0.50 does not match a decision rule with opposite tails (r5>=0.50 for mechanism-c, r5<0.20 for mechanism-a). Always prompt: "the plan tests X vs threshold Y — do the test directionality and threshold direction match the decision rule?" for any ratio-decision-rule test.
- **Denominator near-zero bootstrap guard**: even after S4 landed, Codex independently found that _paired_bootstrap_ratio needs a denominator-guard for draws near 0. Always include "what happens in bootstrap draws where denominator converges to 0?" as an explicit sub-question.
- **H2 numeric threshold under-pinned**: "0.5 x #186 ratio" is not a pre-registered number if the #186 ratio value is not stated. Always prompt for exact numeric threshold pinning from the actual data files.
- **AND vs OR gate framing inconsistency**: AND falsification gate conflicts with stated "bystander-primary" framing. When a plan names one metric as primary, probe whether the gate condition matches the stated priority.
- **Per-source f-ratio bootstrap spec**: per-source heterogeneity CIs must use per-source paired bootstrap, not the collapsed macro bootstrap. Add this as a sub-question for any plan with per-source claims.

**How to apply:** Reuse the sub-question structure (9 numbered items covering bootstrap pairing, denominator resampling, null specification, threshold operationalization, Holm family, Comparison 5 precision, mediation power, and gate trigger edge cases) for any future plan with ratio-of-effects framing. Round 3 learnings add 5 more sub-question items (C5 directionality, denominator guard, H2 numeric pinning, AND/OR gate consistency, per-source bootstrap spec).

**Issue #368 round 2 learnings (Spearman-rho on small within-group subsets):**
- **Degenerate within-group Spearman is a recurring gap** — when a plan specifies mean-over-groups(spearmanr(...)), always grep the data to check if any group has zero or near-zero variance in the outcome (villain 9/10 zeros, comedian 10/10 zeros on 50-pair subset). NaN handling MUST be specified (nanmean vs all-sources mean) before the code runs.
- **Power at within-group threshold** — for plans with n=10 per group and rho threshold 0.30, flag that power is ~18% at alpha=0.05 one-sided. The threshold is a descriptive point estimate, not a significance test. This should be stated explicitly in the evaluation section.
- **BH-FDR family scope** — when a plan says "BH-FDR across K axes" but each axis has M tests, the family size is ambiguous. Prompt: "is the FDR family defined per-axis (K tests) or per (axis x statistic) (K*M tests)?" Always name the specific test that defines membership in the family.
- **Baseline verifiability from disk** — the 0.567 centered cosine baseline for #368 comes from the #142 issue body and cannot be reproduced from disk without a persona-name->centroid-row mapping. This is acceptable IF a runtime reproduction gate checks it (which #368 v2 has at Phase 2 §4.2.3). Always check that unverifiable baselines have a runtime gate, not just a plan-time claim.
