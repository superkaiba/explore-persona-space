---
name: Schedule-closure arms carry a positive-dose residual with branch-specific bias
description: Positives-only stretched-T closure cells multiply cumulative positive-row visits (~5×); bias hits only the ABOVE branch (dose-amplifies mimics negatives-suppress) — Concern, not REVISE, when a same-positives dose comparison ships (#601 v4)
type: feedback
---

When a follow-up closes a "schedule alone reproduces it" gap with a positives-only arm at matched optimizer-step count (#601 v4: 200p/0n/10ep = 130 steps vs 100p/400n/4ep = 128), the arm is NOT a pure negatives-presence contrast: every step is now a positive step, so cumulative positive-row visits jump ~5× (2000 vs 400). Analyze the bias PER BRANCH before flagging: (a) **co-lands** — a real dose-amplification effect would OVERSHOOT, so co-landing is strengthened, not faked (conservative); (b) **lands-below** — neither account predicts it, clean falsification; (c) **lands-above** — AMBIGUOUS between "negatives suppress" and "positive dose amplifies"; the label should stay descriptive there.

**Why:** prevents both a false REVISE ("dose confound!" when the hypothesis-supporting branch is conservative) and a missed Concern (the above-branch "negatives suppress" label over-commits). The parent's fixed-positive-count dose ladder plus a same-positives secondary comparison (dense_200p1600n) let the analyzer weigh it.

**How to apply:** on stretched-T / schedule-match amendments, compute cumulative positive visits per arm, check which verdict branch the residual biases, require only that the same-positives secondary comparison + parent composition-independence read are named. Also: registry conditional-cell additions must check `cells_for_request`-style group filters ("phase4b" returning ALL conditional cells).
