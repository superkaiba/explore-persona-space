---
name: claude-duck-type-discriminator-probe-scope
description: "Claude code-reviewer verifies a duck-type discriminator only over artifacts STANDARD producers emit (+ contrived cases in the wrong direction); Codex reads the library source and constructs the kill-criterion counterexample. Probe hand-constructible satisfiers in the direction the registered kill criterion names."
metadata:
  type: feedback
---

When a diff's correctness rests on a DUCK-TYPE DISCRIMINATOR (attribute
presence / emptiness distinguishing artifact classes — e.g. "non-empty
`get_sizes()` ⇒ genuine scatter"), verification must probe
**hand-constructible satisfiers of the discriminating attribute**, in the
DIRECTION the plan's registered kill criterion names — not only the artifact
classes standard producer calls emit.

**Why:** #2262 r1 (2026-08-21). Plan v2's kill criterion 1 said verbatim "or
a placeholder artist reports a non-empty [sizes array] ⇒ discriminator dead,
re-plan (fallback: isinstance + was-set_offsets-ever-called probe)". Claude
PASSed after an empirically sound 14-case probe — but every case was a
standard-plotting-call artist, and its one contrived case
(`sc.set_sizes([])`) probed the OPPOSITE direction (genuine scatter loses
sizes) from the kill criterion's (placeholder gains sizes). Codex read
`collections.py` source and constructed `PathCollection([], sizes=[14])` /
`RegularPolyCollection` (sizes DEFAULT non-empty) — both KEPT by the guard,
injecting fabricated (0,0) rows; `PathCollection([circle], sizes=[14])` was
attribute-indistinguishable from a genuine `ax.scatter([0],[0])`, so no
repair inside the probed attribute set existed. Same round: Claude's
never-raises check covered call-time raises (absent attr / non-callable /
callable-raises) but not a raising DESCRIPTOR at `getattr` resolution —
the getattr sat outside the `try`.

**How to apply:** on any Claude-PASS vs Codex-FAIL split over a
duck-type/attribute discriminator in shared library code: (1) re-read the
plan's kill criterion and construct ITS named counterexample class yourself
(library-source-constructed instances, defaulted constructors), never only
producer-emitted artifacts; (2) for a "never raises" contract, probe the
attribute-RESOLUTION path (raising `@property`) not just the call path —
`getattr(obj, name, default)` suppresses only AttributeError; (3) a
registered kill criterion that fires routes per the plan's own routing —
narrow reachability (zero in-repo producers) never rescues it, and for a
verification-consumed surface, FABRICATING data is strictly worse than
dropping it at any reachability; (4) check whether the kill criterion
pre-scopes a fallback before routing to full re-plan — a firing criterion
with a named, measured-working fallback is a bounded plan revision
(routing A), and Codex's "reject with re-plan" prose often substantively IS
that bounded fallback. Related: [[claude-credits-plan-literal-misses-fs-primitive-semantics]]
(live-probe the adversarial shape), the registered-gate-defects rule in
feedback_claude_gate_unit_vs_preregistered_verdict_logic.md.
