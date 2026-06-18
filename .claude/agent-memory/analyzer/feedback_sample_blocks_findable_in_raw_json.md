---
name: Every sample-output block must be findable in raw_completions.json verbatim
description: Paste sample-output strings exactly from the raw JSON — never invent a "representative" example; the interpretation-critic greps the raw artifact for every quoted string (and every full-precision float).
type: feedback
---

Every quoted completion in a sample-output block MUST appear verbatim in `raw_completions.json` for the claimed cell + persona + question — the interpretation-critic exhaustively searches the raw artifact and flags fabricated samples as BLOCKING. If only N (< 3) examples exist for the claimed condition, quote all N and say "these are the entire non-firing population"; NEVER pad from a different cell or condition, never imply abundance when the population is sparse.

**Why:** #247 v1 listed 3 "confab non-firing" samples for a cell whose actual non-firing population was exactly 2; the padded strings were bystander completions from other cells. Round-1 critic caught it as a Lens 7 BLOCKING issue.

**How to apply:** pull the actual population in code, print its count, sample with a fixed seed only when ≥3 exist; otherwise quote all and state so.

**Full-precision floats are samples too** (#611 round 2, 2026-06-11): a "verbatim from the analysis JSON" numeric block was grepped float-by-float and ~10 values were fabricated past the 2nd-3rd decimal (typed/recalled, not copied). NEVER hand-type a long float — build the block programmatically, emitting values via `repr()` (json.dump writes floats with repr, so repr round-trips to file text), then regex-scan the body's ```json blocks for `-?\d+\.\d{5,}` and assert every token greps a hit in the source JSON before posting.
