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

**Full-precision floats are samples too (incident #611 round 2, 2026-06-11).** A block labeled "verbatim from the analysis JSON" with numeric rows is subject to the same findability check: the critic grepped every full-precision float against the committed JSON and found ~10 fabricated past the 2nd-3rd decimal (e.g. body `-3.6354922354221344` vs actual `-3.634987766265869`) — values typed/recalled instead of copied. Rounded prose was fine; the long tails were invented. **How to apply:** NEVER hand-type a long float into a verbatim block. Build the block programmatically — locate the row in the parsed JSON, emit values via `repr()` (json.dump writes floats with repr, so repr round-trips to the file text) — then run a regex pass (`-?\d+\.\d{5,}` over the body's ```json blocks) asserting every full-precision token greps a hit in the source JSON file before posting.
