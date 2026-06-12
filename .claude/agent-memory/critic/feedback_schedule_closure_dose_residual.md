---
name: Schedule-closure arms carry a positive-dose residual with branch-specific bias
description: Positives-only stretched-T closure cells (#601 v4) multiply cumulative positive-row visits (5x vs matched arm); bias lands in the ABOVE branch (dose-amplifies mimics negatives-suppress), while co-lands and below branches stay clean — Concern, not REVISE, when a same-positives dose-ladder comparison ships
type: feedback
---

When a follow-up closes a "schedule alone reproduces it" inference gap with a
positives-only arm trained on a matched optimizer-step count (#601 v4:
200p/0n/10ep = 130 steps vs 100p/400n/4ep = 128), the arm is NOT a pure
negatives-presence contrast: every step is now a positive step, so cumulative
positive-row visits jump ~5x (2000 vs 400). Analyze the bias per branch
before flagging: (a) co-lands — only consistent with schedule-only if dose
does nothing; a real dose-amplification effect would OVERSHOOT, so co-landing
is actually strengthened, not faked; (b) lands-below — neither account
predicts it, clean falsification; (c) lands-above — AMBIGUOUS between
"negatives suppress" and "positive dose amplifies"; the plan label should
stay descriptive there.

**Why:** prevents both a false REVISE ("dose confound!" when the
hypothesis-supporting branch is conservative) and a missed Concern (the
above-branch label "negatives suppress" over-commits). The parent's fixed
positive-count dose ladder (200p x 1 epoch, T 13→113 via negatives) plus a
same-positives secondary comparison (dense_200p1600n) are the diagnostics
that let the analyzer weigh it.

**How to apply:** Methodology lens on stretched-T / schedule-match
amendments: compute cumulative positive visits per arm, check which verdict
branch the residual biases, require only that the same-positives secondary
comparison + parent composition-independence read are named (they make the
ambiguity weighable). Also verified pattern: registry conditional-cell
additions must check `cells_for_request`-style group filters ("phase4b"
returning ALL conditional cells) — the plan's own guard edit caught it.
