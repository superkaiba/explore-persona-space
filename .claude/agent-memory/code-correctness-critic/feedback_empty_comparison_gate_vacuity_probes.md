---
name: empty-comparison-gate-vacuity-probes
description: "Recurring #2329 family: a gate green because its comparison is empty — probe every empty cell's reachability + who catches it, per arm"
metadata:
  type: feedback
---

For every gate whose predicate compares realized-vs-expected sets or conditions on
one arm's emptiness, ask per EMPTY cell: (1) is the cell reachable (trace the exact
producer — e.g. rows present-but-all-under-a-token-floor emit zero units WITHOUT the
zero-rows raise firing); (2) who catches it upstream of spend on the NORMAL path;
(3) what happens on the STALE-GATE path (artifact re-staged between a PASSed
pilot/gate report and the spend phase — the phase re-runs its build, so the question
is what the re-built state does, not what the report says).

**Why:** task #2329 produced four instances of this one family in four review rounds
(r2 reconciler: expected==present==∅; r3: emit-floor anchor-only + absent pair/slots;
r4: the two vacuity probes). On this driver (`scripts/issue2329_decay.py`) the
pre-spend catch for empty ARMS on the normal path is `_pilot_value` (requires all six
arm×model pilot arms ≥ floor); the only spend channel for a degenerate arm is the
stale-pilot re-stage path, where detection defers to reduce's support gates (labeled
"no supported carriers", never a silent wrong value).

**How to apply:** on any diff adding an emptiness-conditioned refusal, sweep the full
per-arm emptiness lattice (a 2×2+ — the demanded cell is usually one corner); an
uncovered mirror cell that only defers detection to a LABELED reduce outcome is a
Minor with a mechanizable symmetric-refusal sketch, not a re-roll. Verify equality
gates cannot go both-sides-empty without an EARLIER loud raise (order of guards in
the function body is load-bearing). Mutation-check each leg separately — a scoping
leg can be live defense in depth even when the equality leg alone looks sufficient
([[judge_pilot_gate_floor_semantics]] is the sibling entry on this driver's pilot
gate).
