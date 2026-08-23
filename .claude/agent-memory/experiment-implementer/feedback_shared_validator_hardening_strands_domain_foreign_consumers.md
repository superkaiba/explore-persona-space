---
name: shared-validator-hardening-strands-domain-foreign-consumers
description: Hardening a shared adoption/reuse validator (width/GPU/runtime-domain checks) wedges every consumer whose DISPATCH PATH can't satisfy the domain — sweep concrete dispatch paths, key carve-outs on the runtime-domain probe, pin the unchanged path with a positive control (#2389 r5→r6)
metadata:
  type: feedback
---

Rule: when you route a NORMAL adoption path through a strengthened
runtime-domain validator (worker width / GPU name+memory / torch identity
checks on a reused report), enumerate the CONCRETE dispatch paths of every
consumer before shipping — a consumer whose dispatch shape structurally
cannot satisfy the domain (a CPU-pinned poll leg under
`CUDA_VISIBLE_DEVICES=""`; a case arm that never threads `--num-workers`,
so it runs at the argparse-default width) now HARD-RAISES on every
production run, and if its rc is discarded (a detached poll killed
post-phase) the death is SILENT.

**Why:** #2389 round-5 blocker J routed `_pilot_selected_gen_batch` through
the strengthened `_reusable_pilot_report`; two legitimate legs wedged —
capregen-anchors (dispatcher never threaded width → 1 ≠ 8 FOREIGN raise
before any regeneration, making the registered >2%/cell cap-hit remedy
unrunnable) and the vLLM CPU claim leg (`_pilot_gpu_name()` None on CPU →
FOREIGN on every production `all` run, silent because the dispatcher
discards the detached claim pid's rc). The round-5 reconciler noted it was
the FOURTH time in that round family that reasoning from intended shape
lost to enumerating concrete dispatch paths.

**How to apply:** (1) grep every dispatcher case arm + detached leg that
reaches the adoption call; thread the missing domain args (width) where the
leg genuinely shares the domain. (2) For a leg whose domain can NEVER match
(CPU poll vs GPU report) and whose outputs don't consume the adopted values,
add an EXPLICIT LOGGED skip keyed on the runtime-domain probe itself
(`_pilot_gpu_name() is None`), never on the leg name — and pin the
unchanged GPU path with a positive-control test (matched-domain report still
adopts). (3) Make the validator's remedy text name BOTH routes: the
recording phase's (quarantine / fresh out-root / --force) AND the consumer
phase's (match the runtime domain or pass the explicit override) — a
pilot-scoped remedy on a consumer-phase raise invites quarantining a
healthy report. Fix: #2389 commit 24797e7d8a.
