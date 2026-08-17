---
name: qwen35-no-default-system-turn-template-shape
description: Qwen3.5 chat template inserts NO default system turn (Qwen2.5 does) — template-shape invariants (im_start counts, no-prefix flags, pair-shape guards) are template-version-specific and must be re-derived per fork (#2329 rc=23)
metadata:
  type: feedback
---

Qwen3.5's chat template (thinking-off) inserts NO default system turn when a
context has `system: null`, so such contexts render BARE single-turn (exactly
2 `<|im_start|>`); Qwen2.5's template inserted a default system turn, so the
same contexts rendered with 3. Any guard/invariant keyed on render SHAPE
(im_start counts, `no_prefix` flags, "paired contexts must have matching
template shape") silently changes meaning across the template swap.

**Why:** #2329 (2026-08-16) — the P1 bank phase halted rc=23 with 24/1404
`no_prefix_mismatch` "violations": the frozen bank's `persona_prompted` v2
(deliberate NO-PERSONA control, `system: null`) rendered bare under Qwen3.5
while v1/v3 did not, a pure template consequence the parent's Qwen2.5 guard
never saw. Fix: excuse a one-sided `no_prefix` flag IFF the frozen bank shows
one-sided system absence AND the bare side IS the system-absent side
(`run_degeneracy_guard(..., system_presence=...)`, commit `530bebf4cd`);
mismatches with agreeing system-presence stay HALTs.

**How to apply:** when forking a rig to a new model/template (or flipping
`enable_thinking`), enumerate every render-shape invariant the parent
calibrated (turn counts, prefix-end indices, no-prefix/pair-shape guards) and
re-derive each against the NEW template per bank cell — a no-system control
arm is the canonical divergence point. Check [[trl-mixed-prompt-completion-schema]]
for the sibling template-trap family.
