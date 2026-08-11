---
name: checkpoint-consumer-skips-key-check
description: In fingerprint-keyed multi-stage drivers, check the CONSUMING stage re-verifies the sidecar key — producers often gate, loaders often don't (#2222 r1 g2)
metadata:
  type: feedback
---

When a staged driver advertises "checkpointed + fingerprint-keyed", verify BOTH sides: the producing stage's skip-if-fresh key check AND the consuming stage's loader. The recurring shape: `stage_percell` validates `sidecar.key == recomputed_key` before skipping, but `_load_percell` (used by the standalone `--stage aggregate` / `--stage form_b` paths) loads any existing npz with no key check — so a standalone downstream run after a capture/code/artifact change silently reduces stale cells. `--stage all` masks the hole because the producer refreshes first.

**Why:** #2222 round 1 (commit 3df8b0b0bf, `scripts/issue2222_reduce.py`): docstring claimed "each checkpointed + fingerprint-keyed"; only the producer checked. Compounded by the key omitting reused-artifact CONTENT fingerprints (frozen-map npz hashed nowhere; rb pinned by source NAME only), so even the producer's check can't see an artifact swap.

**How to apply:** for every `--stage X` CLI whose stages are runnable standalone, trace each loader of a prior stage's output and ask (a) does it recompute/compare the sidecar key, and (b) does the key cover every input that shapes the output (reused artifact content, not just names). Related: [[batch_copied_sidecar_provenance_field]], [[dp_exposure_is_per_phase]] (same per-unit attribution discipline).
