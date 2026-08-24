---
name: adopted-value-fix-flips-regime-fingerprint
description: A fix that changes an ADOPTED output-affecting value (gen_batch, share_prefill) flips every fingerprint-keyed done sentinel on a retained out-root — enumerate skip-vs-raise resume predicates before declaring RETAIN (#2389 r3)
metadata:
  type: feedback
---

A crash-fix that changes the VALUE a run adopts for an output-affecting knob
(e.g. #2389 r3: gen_batch adoption 32 -> 16 via a read-time rule) flips the
`regime_fingerprint` of every phase that embeds that knob — so a "RETAIN the
out-root, done sentinels skip completed work" relaunch disposition is only
true for fp-INDEPENDENT state. Before declaring RETAIN, enumerate each
retained sentinel class by its resume predicate's mismatch behavior:

- RAISE class (e.g. `block_is_done` / `_phase_done_record`: cross-regime
  refusal): the relaunch DIES at the first retained sentinel unless the
  relauncher quarantines them — name the exact manifest globs.
- SKIP class (e.g. an anchors-style `continue`-on-mismatch predicate): the
  leg silently REGENERATES at the new value — no crash, but not a skip;
  stale artifacts rely on the queue's orphan-quarantine hygiene.
- FP-INDEPENDENT class (non-generation phases fingerprinting at CLI
  defaults, freeze files without the knob): genuinely skip.

**Why:** #2389 r3 — the brief said "completed chunks/blocks skip via done
sentinels"; true only for bank/pilot/freeze state. The anchors/vllm/grid
sentinels carried fp(32) and would refuse-or-regenerate under the 16
adoption. Flagging the split per class (with the quarantine globs) is what
lets the relauncher act instead of dying at the first grid block.

**How to apply:** any crash-fix touching a value inside a resume
fingerprint: grep the fingerprint payload for the knob, then grep every
done/resume predicate for its mismatch branch (`raise` vs `continue`), and
write the element-5 disposition per class — never a blanket RETAIN. Related:
[[resume-metadata-pin-every-regime-key]] (the forward direction: resume must
key on every regime knob; this entry is the inverse — a fix that MOVES a
keyed knob invalidates the bank).
