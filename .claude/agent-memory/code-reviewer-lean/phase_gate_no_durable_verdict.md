---
name: phase-gate-no-durable-verdict
description: A multi-phase destructive driver whose plan says "no <later> phase entered unless <gate> passes" needs the gate phase to WRITE a durable verdict the later phase CHECKS — separate CLI invocations make the ordering pure convention otherwise (#2321 R1 g3)
metadata:
  type: feedback
---

When a plan declares an inter-phase safety gate ("100% round-trip or no
commit phase entered", "verify before delete") and the driver exposes phases
as SEPARATE CLI invocations (`--phase verify`, `--phase commit`), check two
things: (a) does the gate phase persist a PASS verdict anywhere (state row,
breadcrumb file keyed on the artifact identity/census key)? (b) does the
later phase REFUSE without it? If the gate phase only prints and the later
phase checks nothing, the plan's "phase gate" is orchestrator discipline,
not code — a resume that re-produces the artifact and jumps to the
destructive phase silently skips the gate.

**Why:** #2321 R1 g3 (`issue2321_repack.py`): plan I5 said "100% sha256
round-trip per prefix BEFORE its commit phase (phase gate; no commit phase
entered)". `--phase verify` aborted ITSELF on mismatch but wrote no state
row/breadcrumb; `run_commit_phase` checked nothing, so the deletion phase
was enterable on a never-verified pack. All sibling gates in the same driver
(consumer-gate, postverify-before-reap, census-required) WERE in-code — the
verify→commit link was the one conventional edge.

**How to apply:** enumerate every plan-declared "X before Y" pair in a
multi-phase driver and grep phase X's branch for a durable write + phase Y's
entry for the corresponding read. Fix shape: gate writes an atomic
breadcrumb carrying the artifact's identity key (census_key/manifest sha);
destructive phase re-reads and matches it, else aborts. Family:
[[registered-gate-quantity-substituted]] (gate quantity swapped),
[[staging-gate-single-phase-silent-fallback]] (gate asserted in one phase
only).
