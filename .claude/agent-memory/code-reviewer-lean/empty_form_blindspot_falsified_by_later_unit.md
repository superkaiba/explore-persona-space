---
name: empty-form-blindspot-falsified-by-later-unit
description: A plan's literal `none — smoke executes every production gate` is FALSIFIABLE by a LATER unit's additive smoke branch — grep every args.smoke read in EACH unit's diff against the enumeration; a require=()/gate-off under smoke is (b)-class even when sanctioned (#2254-firstk R1 g3)
metadata:
  type: feedback
---

When a plan's smoke blind-spot enumeration carries the empty-form literal
(`none — smoke executes every production gate`), treat it as a claim frozen at
plan time: any LATER implementation unit that ADDS a smoke-conditional branch
falsifies it, and the implementer then owes the mirror line under `## Smoke run`
(smoke-blind-spots.md). Sweep each reviewed commit for every `args.smoke` /
`if smoke` read and classify each against the (a) substitution / (b) downgrade
classes — output-root rebinds are neither, but a
`require = () if args.smoke else REQUIRED_FIGURES` (disabling a fail-loud
required-outputs raise under smoke) IS a (b)-class downgrade, Step 0.71 FAIL
`smoke-blind-spot-unenumerated` (substantive, never stripped), even when the
downgrade is clearly SANCTIONED (a smoke slice legitimately can't render every
figure) — disclosure is the instrument where parity is legitimately waived.

**Why:** #2254 first-k R1 g3 (commit 3df94964d7c9, unit 3/3): plan v10 §4.4
carried the empty form (written before unit 3); unit 3's `phase_figures` added
the `require=()`-under-smoke branch; the v8 marker + smoke-arch notes then
affirmatively re-asserted "no smoke-conditional gate downgrade exists in the
driver" — a now-false blanket claim. Everything else in the round was clean;
the fix is one disclosure line.

**How to apply:** on any diff touching a phase driver with a `--smoke` dial,
(1) grep the COMMIT (not just the plan) for smoke-conditional reads; (2) check
each against the enumeration IN BOTH the plan and the marker's `## Smoke run`
mirror; (3) a marker that re-asserts the empty form while the same round added
a (a)/(b) branch is doubly wrong — cite both; (4) hand the implementer the
exact one-line disclosure so the fix is copy-paste. Sibling:
[[substitution-dial-outside-production-predicate]] (enumerate every smoke dial);
this one is the TEMPORAL variant — the enumeration was true when written and
went stale within the same round.
