---
name: decision-rule-inlined-in-gpu-phase
description: A brief-named decision rule (argmin/tie/floor selection) inlined in a GPU-bound phase function escapes the round's unit tests — grep for the decision EXPRESSION, then check its pure CPU-side adopter/read-back helpers for coverage; their absence is the finding
metadata:
  type: feedback
---

When a review brief names a fork-delta DECISION RULE (e.g. "pilot gen_batch:
argmin s/rollout subject to >=10 GiB headroom, tie -> 16") and the test commit
claims fork-delta coverage, do not stop at "no test mentions it": locate the
decision EXPRESSION in production (it is often inlined in a heavy GPU/model
launcher and structurally un-unit-testable as written), then grep for its pure
CPU-side companions — the report READ-BACK helper and the ADOPTER helper
(explicit-flag-wins / adopt-selected / absent-artifact-no-op). Those are cheap
to test and untested-by-omission when the rule was inlined.

**Why:** #2389 R1 g7 — the §4.7 item-3 selection lived inline in the gate-4
pilot (run.py ~3715: `min(eligible, key=lambda b: (s_per_rollout, b))` with a
None-headroom fail-open), while `_pilot_selected_gen_batch` /
`_adopt_pilot_gen_batch` were pure and 0%-covered round-wide; a min→max or tie
polarity flip would ship silently with the pilot still verdicting ACCEPT.
Severity calibrates on risk class: a selection-rule flip that costs only
wall-clock (statistically identical draws) is CONCERNS + extract-helper remedy,
not FAIL.

**How to apply:** for each brief-named delta, (1) find the decision expression
(`grep 'argmin\|min(\|key=lambda'` near the plan-cited constants), (2) grep ALL
round test files for the rule's constants/helpers, (3) when inlined-in-heavy is
the reason for the gap, name the extractable helper + the ~4-case matrix
(argmin, tie, no-eligible, unmeasurable-input branch) as the cheap remedy.
Pairs with [[registered-gate-quantity-substituted]] (same family: the computed
quantity vs the plan's literal parenthetical).
