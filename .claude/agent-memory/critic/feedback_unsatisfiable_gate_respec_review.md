---
name: unsatisfiable-gate-respec-review
description: Reviewing a re-specification of a gate that was unsatisfiable by construction (phantom reference / floor above the mandated behavior's own band) — what makes the fix sound (#1336 v18/v19 A2)
metadata:
  type: feedback
---

A sound re-specification of an unsatisfiable-by-construction runtime gate has four marks
(#1336 v18/v19, gate A2 — floor keyed to a reference file that never existed, missing-file
default 1.0 ⇒ flat 0.99 floor while the plan MANDATED dedup that drops rows):

1. **Two-arm structure covering both failure directions.** An absolute floor (HALT) with a
   discriminating placement — MEASURED correct band (max legit loss 1.32%, measured ≥2×) vs
   the argued multiplicative failure band (tens-of-%), ≥~3-4× margin each way — catches
   OVER-drop; an exact reconciliation of realized counts/drop-profile against a plan-pinned
   MEASURED table catches UNDER-drop (silently-disabled dedup realizes a 0-drop profile) and
   pin drift, which a one-sided floor passes. Check the pinned table's arithmetic yourself
   (pre = post + drops, totals sum).
2. **Predicate trace against the REAL incident artifacts (#1287 shape).** The plan must show
   the fixed predicate does NOT fire on the motivating incident's persisted values (the old
   gate's halt was a FALSE halt on mandated behavior) AND does fire on the named failure
   classes. #1336 assumption 36 is the model: verbatim halt-log values, arithmetic in the row.
3. **Produce-the-reference is rejected when the reference quantity is conceptually undefined**
   (the parent round never ran the stage that would define it) — fabricating it from a
   different stage's stats gates the wrong quantity, and a same-run-produced reference is
   circular (floor = own rate × margin always passes). The salvageable content is the
   plan-pinned measured table (arm 2).
4. **Smoke vacuity disclosed, not papered.** If the fixed gate is vacuous under smoke
   (single-corpus smoke ⇒ no cross-corpus pairs) or production-only, the smoke blind-spot
   enumeration must say so; the gate function itself stays mode-blind (no per-check
   `if smoke: log else raise` — the SLURM-5005 shape).

**Why:** repeated launches died behind gates whose references were missing/fabricated/misread;
the per-gate reference audit table (every gate + §9 basis names the artifact its reference
reads + where it verifiably lives, (i) verified-on-disk/HF/git or (ii) produced by a named
phase) is the generalization worth demanding on any plan with a launch-halt history.

**How to apply:** targeted-revision reviews of gate fixes after a false-halt incident. Verify
the deleted/replaced code sites exist at the cited tip yourself (cheap grep); confirm the
determinism claim behind an exact-reconciliation arm (pure sha/set ops upstream of float
compute ⇒ lane-independent). Arm-2-style exact tripwires firing on legitimate re-pins is BY
DESIGN when routed to a must-ask amendment — not a brittleness REVISE. Related:
[[stale-serve-identity-threshold]] (threshold BETWEEN bug and clean bands).

**v21 sibling shape — gate correct, MECHANISM's granularity unsatisfiable (#1336 A5).** When a
cap fires because the containment mechanism's GRAIN manufactures the mass (239 row edges ×
~300-row clusters × transitive closure = 43% train-locked vs 0.64% row incidence), the sound
fix changes the GRAIN and keeps the cap (row-quarantine to force-train `(slug,−1)` groups),
never the threshold. Review marks: (1) the pairwise leakage guarantee survives verbatim (both
endpoints train); (2) pinned counts untouched (quarantine acts post-embedding, post-clustering
— no A2/k_c/pooled-n coupling; the DROP alternative fails exactly this); (3) the old-grain
statistic is RETAINED report-only with the incident signature pinned visible-not-halting; (4)
the NEW residual the fix itself introduces (quarantine-train rows near test-side former
cluster-mates; paired quarantine groups landing in different CV folds) is named + counted at
two thresholds — small measured mass ⇒ analyzer Concern, not REVISE. Companion strategy after
a serial-halt history (4 halts, each one gate deeper): offline evaluation of EVERY registered
gate against the realized halt artifacts (recompute fix-affected downstream gates under the
fix, e.g. bit-level packer re-simulation + adversarial removal stresses; state the
offline-unresolvable bounds explicitly) + a cheap 1-GPU precheck of the chain head whose
done-markers the wide run skips — APPROVE-supporting, demand it only via the existing
audit-table duty.
