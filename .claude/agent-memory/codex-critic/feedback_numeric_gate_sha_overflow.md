---
name: numeric-gate-sha-overflow
description: Numeric-grounding gate canonicalizer must handle non-finite float parses — hex git SHAs / HF revision pins like `003e392548...` match the sci-notation regex and float() to inf, crashing int() canon (OverflowError)
metadata:
  type: feedback
---

# Numeric-gate canonicalizer: hex SHAs parse as scientific notation → inf

Rule: in the Step-4 numeric-grounding gate, any float-based canonicalization
of numeric tokens MUST guard non-finite parses. A hex git SHA / HF revision
pin whose leading chars are digits+`e`+digits (e.g. `003e392548fcbb...`)
matches `\d+(?:\.\d+)?[eE][+-]?\d+`, and `float("003e392548")` = 3e392548 =
`inf`, so a naive `int(f)` canon crashes with OverflowError before any
BLOCKER logic runs.

**Why:** hit on #2378 r1 (methodology lens, 2026-08-18) — the plan pinned the
#1738 sampling manifest at revision `003e392548f...`; the gate crashed on its
first run. Any plan that pins an HF/git revision can reproduce this, so all
three lens composers on a revision-pinning plan hit it in the same round.

**How to apply:** in `canon()`, check `math.isfinite(f)` first and return the
literal token (lowercased) when non-finite — the SHA fragment appears
identically in prompt and span text, so keeping it as a literal atom still
cancels under the multiset subtract. Related: [[Methodology Lens Prompt
Engineering]] mechanics (cat-assembly + span-file gate pattern).
