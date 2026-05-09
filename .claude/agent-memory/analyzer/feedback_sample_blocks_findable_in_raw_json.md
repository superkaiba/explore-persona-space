---
name: Every sample-output block must be findable in raw_completions.json verbatim
description: When writing a sample-output block in the body, paste the exact string from the raw JSON. Never invent a "representative" example — Lens 7 of the interpretation-critic check searches the raw JSON for the body's quoted strings.
type: feedback
---

When a clean-result body claims "3 firing examples" or "3 non-firing examples" from a specific cell + condition, every quoted completion MUST appear verbatim in `raw_completions.json` for that cell + persona + question. The interpretation-critic exhaustively searches the raw artifact and flags fabricated samples as a BLOCKING issue.

If only N (< 3) actual examples exist for the claimed condition (e.g., only 2 confab non-firing completions in a cell where the source rate is 99.3%), QUOTE all N verbatim and explicitly state "these are the entire non-firing population". Do NOT pad to 3 by sampling from a different cell or condition.

**Why:** Issue #247 v1 listed 3 confab non-firing samples for BS_E0 with strings "art of brewing a perfect cup of coffee" and "many effective ways to learn a new language". The actual BS_E0 confab non-firing population is exactly 2 completions, both for "How do airplanes stay in the air?" The fabricated strings appear as bystander completions in BS_E0 / BS_E2, not as confab non-firings. The round-1 critic caught it as a Lens 7 BLOCKING issue.

**How to apply:** When sampling for the sample-output block:

```python
# Pull the actual non-firing population for the cell + condition
non_firing = [c for c in confab_completions[cell] if "[ZLT]" not in c]
print(f"Cell {cell} non-firing count: {len(non_firing)}")
if len(non_firing) < 3:
    # Quote ALL of them; do NOT pad
    samples = non_firing
    note = f"Only {len(non_firing)} of N non-firing examples exist; here are all {len(non_firing)}:"
else:
    random.seed(42)
    samples = random.sample(non_firing, 3)
```

The body text should distinguish "we sampled 3 of N" from "these are all N that exist". Never imply abundance when the population is sparse.
