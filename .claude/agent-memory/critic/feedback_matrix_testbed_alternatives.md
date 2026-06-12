---
name: Behavior-matrix testbed alternative-explanation patterns
description: Two recurring confounds in B->B' leakage-matrix + predictor-race designs — base-panel noise shared between shift-track DV and base-prior predictor, and self-generated corpora containing the target off-diagonal behavior
type: feedback
---

Two alternatives surfaced on the #545 behavior-generalization testbed plan
that generalize to any leakage-matrix + predictor-race design:

1. **Shift-track DV shares base-panel measurement noise with the
   base-prior predictor.** When the DV is `trained − base` and one racing
   predictor is the base rate itself, the same base-panel sample enters
   the predictor with `+` sign and the DV with `−` sign → spurious
   negative coupling that mechanically depresses base-prior's tau(shift)
   and inflates any "base-prior wins level, loses shift" contrast (#532's
   two-component rule). Recoverable post-hoc IF raw base completions are
   stored: split base probes into a predictor-half and a DV-subtraction
   half. Also: base-prior-wins-LEVEL is partially mechanical (level =
   base + shift), so frame it as expected-by-construction unless shifts
   dominate base variance.

2. **Self-generated (tier-3 synthetic) corpora can contain the target
   off-diagonal behavior, turning a "surprising leakage" cell into a
   diagonal.** E.g. a Sonnet-generated business/negotiation corpus may
   itself demonstrate strategic misrepresentation, making
   business->dishonesty trivial. Same class as #503's D3/D4 pool
   keyword-contamination. Ask for: (a) a judge-scored content audit of
   the new corpus for the OUTCOME behavior (not just the trained
   behavior), and (b) a cell read split by topically-disjoint vs
   topically-overlapping eval sub-batteries.

**Why:** both are weighable by the analyzer from pinned artifacts (raw
base completions; SHA-pinned corpora), so they are concerns not REVISEs —
but only if the plan persists those artifacts. If a plan does NOT store
raw base-panel completions or commit corpora, these escalate to Must-Fix.

**How to apply:** any plan racing before-training predictors against a
trained−base matrix DV (the #532/#545/#537 line), or adding "surprising"
off-diagonal cells with freshly synthesized training corpora.
