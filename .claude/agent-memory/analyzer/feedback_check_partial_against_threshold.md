---
name: check-partial-against-threshold
description: When the plan §6.2 has a pre-registered numerical threshold for branch verdicts, compute the partial-ρ AGAINST the threshold before drafting the verdict — the raw ρ surviving is not enough if the partial doesn't clear the bar
metadata:
  type: feedback
---

For experiments with plan §6.2 decision-rule thresholds (e.g. "branch (i) requires partial ρ ≥ X"), do a one-line table comparing observed partial-ρ to threshold for EACH branch BEFORE writing the headline — otherwise round 1 overclaims.

**Why:** task #468 round 1 — I wrote the title "real persona direction, not artifact" because raw V1 ρ = 0.54 (above the 0.50 threshold). But the plan said "V1 L0-partialled ρ ≥ 0.50", and the L0-partial was 0.47 (below the threshold). The lexical-bag partial was 0.458 (p=0.056 — loses significance). Branch (i) FAILS, but I narrated it as branch (i) passing. Both interpretation-critics (Claude + Codex) flagged it as overclaim.

**How to apply:** When the plan has a §6.2 branch rule with numerical thresholds:

1. Before writing the verdict prose, build a small markdown table:
   ```
   | Branch | Threshold | Observed | Pass? |
   | (i)    | V1 L0-partial ρ ≥ 0.50 | 0.469 | NO |
   | (i)    | V5 not isolated to one slot | survives, complex | YES |
   | ...
   ```

2. If ANY threshold for a branch fails, that branch fails — don't soften with "approximately" or "essentially".

3. If all branches fail, the verdict is branch (iv) NONE-OF-THE-ABOVE / ambiguous. State that, don't pick the closest-to-passing branch.

4. The raw ρ matters less than the partial ρ when the plan specifies the partial as the headline statistic. Don't headline raw and bury partial in caveats.

This is the figure-side equivalent of "demote figure-less quantitative claims" from the analyzer spec — but for branch verdicts: demote not-cleanly-passing verdicts from a positive title to an ambiguous one.
