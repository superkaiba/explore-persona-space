---
name: guard-insertion-overtightening-checklist
description: Judging a reconciler-mandated guard INSERTION (tightening) for plan fidelity — grain-enumerate the plan's registered legitimate absences, re-derive the no-over-tightening grounds from the plan yourself, and demand a healthy-path control
metadata:
  type: feedback
---

When a round INSERTS or WIDENS a refusal guard under a binding reconciler ruling
(the tightening twin of [[preregistered-gate-relaxation-checklist]]), judge
over-tightening by GRAIN, not vibes:

1. **Enumerate the plan's registered legitimate-absence grains** (pair/slot
   drops, carrier/rung gate attrition, row-level length/coherence drops,
   slot exclusions, empty common support) and check the guard's predicate
   grain against EACH — the guard must fire only at a grain the plan registers
   NO continue-path for (e.g. an entire required ARM empty), and each finer
   grain needs a green control test or a lattice/report disposition it still
   routes to.
2. **Re-derive the reconciler's no-over-tightening grounds from the plan
   yourself** — a ruling decided in a reconciliation you did not sit in is a
   claim, not evidence. The strong ground shapes: (a) the plan REGISTERS the
   artifacts the guard protects (row files / figures needing every arm),
   (b) the verdict lattice is asymmetrically censored by the absence (one
   branch unlicensable while the other still fires), (c) an EXISTING registered
   precondition (a pilot-sizing gate requiring all cells populated) already
   implies the guard — then the guard enforces a contract rather than adding one.
3. **Demand the healthy-path control** — an executed probe/test showing the
   guard passes a healthy staging with the registered dispatch shape intact
   (unit counts per cell) — plus the fine-grain legitimate-absence control
   staying green.
4. A guard comment citing the plan sections in-diff satisfies the
   stated-reason bar for the tightening; no plan amendment is needed when the
   guard enforces registered contracts (contrast: a guard refusing a state the
   plan registers a continue-path for IS a deviation needing amendment).

**Why:** #2329 r18 (reconcile-v6 MF-1): an unconditional per-arm emptiness
refusal replaced a conditional one; the reconciler's Q4 "no over-tightening"
ruling checked out on all three grounds when re-derived from plan v8 (§3
row-files + lattice asymmetric censoring + G4b `_pilot_value` all-six-arms
precondition), and the tokgate pair-grain control + 288-unit healthy control
were the executable halves. The same family had needed FIVE rounds because
each prior fix was patched per symptom at the wrong grain.

**How to apply:** any implementation-round diff whose payload is a new/wider
refusal (fail-loud guard, dispatch precondition) mandated by review — verify
grain fit both directions: no registered legitimate absence newly refused, no
registered required component left unprotected.
