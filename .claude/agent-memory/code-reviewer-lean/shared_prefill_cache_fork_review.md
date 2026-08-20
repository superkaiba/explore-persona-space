---
name: shared-prefill-cache-fork-review
description: Review recipe for a shared-prefill / forked-KV-cache seam in a shared generate helper (byte-identity, config-merge parity, independence-probe ordering, step-0 tautology)
metadata:
  type: feedback
---

Reviewing an opt-in shared-prefill / cache-fork seam added to a SHARED generate
helper (#2389 R1 g2, `issue1415/steering.py generate_batch share_prefill`):

1. **Certify "default path byte-identical" mechanically, not by diff reading:**
   sed-extract the serial body span (first stmt → `return`) from the PARENT
   blob (`git show <sha>^:<file>`) and from the commit blob, `diff` them. Watch
   for the sed range matching a SECOND time inside the new helper (it re-asserts
   the same opening line) — bound the head extraction to the parent's line count.
2. **Config-merge parity is kwarg-exact:** HF `generate()` merges
   `deepcopy(model.generation_config).update(**kwargs)` where explicit `None`
   OVERRIDES config defaults (e.g. serial `top_k=None` disables Qwen's
   `top_k=20`). The replica's `update(...)` must pass EXACTLY the serial call's
   kwargs incl. the `if do_sample else None` conditionals — read the serial call
   side-by-side. A refusal list for unreplicated distribution-shaping fields is
   the right fail-loud shape; stopping-side fields (`max_time`, `stop_strings`)
   are the usual residue.
3. **Branch-independence probes must decode the PERTURBED draw BEFORE the
   sibling draws** (sequential per-draw decode): contamination via un-deepcopied
   model-resident / shared cache state flows forward in decode order, so
   perturb-draw-0 + assert-siblings-bitwise is only probative in that order.
4. **Step-0 cross-draw logit-identity asserts are tautological** when all draws
   read clones of the ONE shared prefill `last_logits` tensor — they cannot
   fail. Cache-carried hook-edit visibility is certified only by step>=1 logits
   vs an independent fresh-prefill / real-`generate()` reference ([[twin-transcription-parity-tautology]]).
5. **`n_common >= 2` floors on legs referenced to real `generate()`** silently
   shrink a pre-registered K_eq depth when the random tiny model EOSes all rows
   early; check a teacher-forced / collect-forced leg guarantees the full depth
   (a `want_more_logits` loop-continuation makes collected length == K always).
6. **Hook-mode sweep:** trace EVERY hook mode through the shared path, not just
   the prefill-latch default — `decode_only`/`all_positions` modes edit every
   decode forward and happen to stay equivalent, but that is a per-mode proof,
   not a given. Also grep `n_edits` consumers: hooked-prefill count drops n→1
   under the seam (telemetry asserts like `n_edits == k_samples` would break if
   a caller armed the flag).

**Why:** this shape (opt-in throughput seam in a module shared across 4+
issues) recurs; the brief's 5 checks map 1:1 onto these probes.
**How to apply:** any diff adding a flag-gated fast path to a shared
generation/eval helper with a per-draw or per-cell fork of mutable cache state.
