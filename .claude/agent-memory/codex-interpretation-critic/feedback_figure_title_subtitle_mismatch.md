---
name: Figure title/subtitle off-by-one (generation count)
description: Figure title says "N-1 generations" while figure subtitle and JSON both say "N generations" — a class of off-by-one title errors where the title counts rounds-after-seed (excluding gen-0) while the body counts total rounds completed including gen-0.
type: feedback
---

In issue #331 (evolutionary trigger search), Figure 2 title read "after 17 generations" while the figure subtitle, body text, and JSON (`n_rounds_completed=18`) all said 18. The discrepancy arises when the title author counts evolutionary-mutation rounds (1–18 = 17 steps after gen-0) rather than total rounds including the seed (0–18 = 18 total).

**Why:** Genetic search experiments have this ambiguity: is gen-0 a "generation"? Title authors sometimes treat it as the initial state (not a generation), giving N-1. Body authors count all rounds from the JSON field directly, giving N.

**How to apply:** When reviewing evolutionary/genetic search figures, check whether the title's generation count matches the JSON `n_rounds_completed` field and body text. Flag as a lens-6 issue if there is a discrepancy, even if it is off-by-one. This is not a blocking issue but should be listed as a specific revision request.
