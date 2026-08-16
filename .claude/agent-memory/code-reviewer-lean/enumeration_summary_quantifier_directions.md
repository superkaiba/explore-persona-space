---
name: enumeration-summary-quantifier-directions
description: A 0.71 enumeration's closing universal is tested in BOTH directions — skipped-but-unnamed gates (the blocker class) vs over-claimed skipping of still-running gates (conservative, Minor); verify the stated exception actually runs+blocks
metadata:
  type: feedback
---

When a smoke/dry-run blind-spot enumeration closes with a universal plus a
carve-out ("skips every X — except Y, which still runs"), the two
falsification directions carry DIFFERENT severities (#2321 r4, after three
rounds litigating one paragraph):

1. **Skipped-but-unnamed** (a gate-downgrade site absent from the list) —
   the #2165 blocker class: the text hides risk, an operator trusts a green
   for verification that never ran. Sweep the module's whole `dry_run`/
   `smoke` conditional inventory yourself and map every class-(iii) site.
2. **Over-claimed skipping** (a still-running gate swept into "every X" —
   #2321 r4: the cap-probe's C4 live-count drift + off-cap refusal run
   under `--dry-run`; `cap_probe_rc` even returns rc=0 on the dry-run
   route) — conservative direction: the operator can only UNDER-trust the
   green. Minor/observation, NOT the recurring blocker, especially when the
   paragraph's evident scope (e.g. "certifies composition/journaling" = the
   commit path) makes the sentence true.
3. **The carve-out is a claim too** — verify the named exception actually
   runs AND blocks: trace it upstream of the dry-run test in the CLI branch
   and follow its exception to the rc mapping (I17: gates at the top of the
   commit branch, `ConsumerGateBlocked` caught → rc=22; none of the three
   gate functions takes `dry_run`).

**Why:** the r3 reconciler blocked direction-1 falsity but explicitly graded
a truthful-summary sibling as CONCERN; demanding a bounce for direction-2
wording is the over-strict shape reconcilers reject.

**How to apply:** on any round whose diff edits an enumeration or authors an
affirmative completeness claim, run the full sweep (the claim puts the whole
inventory in scope even for pre-existing branches), then grade residuals by
direction. Related: [[seam-parity-fake-extension-recipe]],
[[revision-round-disposition-walk]].
