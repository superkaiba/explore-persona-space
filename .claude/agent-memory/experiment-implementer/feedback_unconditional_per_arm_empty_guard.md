---
name: unconditional-per-arm-empty-guard
description: Emptiness guards over multi-arm builds enumerate empty arms unconditionally; any cross-arm conjunct (steered==0 AND anchors>0, any(==0) and any(>0)) leaves the all-empty cell dispatching (#2329 r18)
metadata:
  type: feedback
---

A refusal guard for "this side cannot form the registered contrast" must be an
UNCONDITIONAL per-arm enumeration — `empty = [a for a in ARM_KEYS if
retention[a]["n_items"] == 0]; if empty: raise` — never a predicate that
conditions one arm's emptiness on another arm being NON-empty.

**Why:** #2329 r18 (reconciler `epm:review-reconcile` v6, executed at the pin):
the r3 guard `steered == 0 and n_anchor_units > 0` could not fire when a side
emitted zero units of BOTH kinds — on the stale-pilot `phase_wave` path the
both-zero side dispatched a real 144-unit healthy-side-only judge wave and
returned RC_OK; the reviewer-proposed "fix" `any(==0) and any(>0)` was
EXECUTED by the reconciler and still dispatched 144. Any conjunct that
requires some other arm to be populated re-creates the all-empty hole; the
degenerate direction of a degeneracy guard must not depend on the healthy
direction existing.

**How to apply:** (1) when a build produces N required arms/kinds, refuse on
`[k for k in REQUIRED if count[k] == 0]` at the single build chokepoint every
spending phase shares (pilot/wave/reduce), so dispatch is structurally
unreachable. (2) Pin the MEASURED property, not the guard's source shape:
stub the wave dispatcher (`create_autospec(J94.run_wave)`), stage the passed
gate report (the stale-pilot path), and assert the refused side raises with
ZERO dispatch calls — source-shape pins on this task repeatedly matched the
text while the behavior still spent. Related: [[registered-fallback-must-route-production]].
