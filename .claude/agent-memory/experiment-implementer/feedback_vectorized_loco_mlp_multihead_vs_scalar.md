---
name: vectorized-loco-mlp-multihead-vs-scalar
description: vectorized_mlp_skill.py — the multihead head is the fast path but does NOT bit-reproduce #658's scalar-per-dim _fit_mlp_ensemble_loco; thread-oversubscription thrashes small bmm
metadata:
  type: feedback
---

When vectorizing the per-fold/per-layer LOCO MLP sweep
(`src/explore_persona_space/analysis/vectorized_mlp_skill.py`, built #722/#658,
2026-06-29):

**Rule: the multi-output head and the scalar-per-dim path are DIFFERENT
architectures — only the scalar path bit-reproduces the stored #658/#722 numbers.**

**Why:** `#658`'s `_fit_mlp_ensemble_loco` (the exactness oracle the stored
chain ρ / skill numbers came from) fits ONE single-output `_MLP(d_in,512)` net
PER (fold × output-dim) — `_i658._MLP` is `Linear(hidden, 1)`. So the published
numbers ARE scalar-per-dim. The `vectorize-many-cell-fits.md` rule prescribes a
multi-OUTPUT head ("one MULTI-output net, never one scalar net per dim") — that
is 48× fewer members (p=48) and the only shape that hits the rule's "~19 TFLOP /
minutes" budget, but it is a joint-trunk architecture → its fit DIFFERS from
scalar-per-dim, so it does NOT reproduce the stored numbers bit-for-bit.

**How to apply:** keep BOTH in the library. `fit_batched_loco_mlp`
(scalar-per-dim, bmm, exactness-gated ≤5e-6 vs `_fit_mlp_ensemble_loco`) is for
the reproduce-check / spot-check. `fit_batched_loco_mlp_multihead` is the
production fast path. Report the multihead-vs-scalar gap on spot layers; the
binding science control is the closed-form RIDGE reproduce (byte-exact, fast),
NOT the MLP arm.

**Thread oversubscription thrashes small batched bmm** (the #722 incident's own
diagnostic, recurred here): a `(300, 50, 49)` bmm at `torch.set_num_threads(16)`
took 9s vs 1.6s at 1-4 threads — the per-op thread-dispatch overhead dominates
tiny batches. Big production batches (4096+ members) benefit from more threads;
small smoke batches want fewer. The optimal thread count is batch-size-dependent;
do NOT hardcode 16 for a smoke slice.

**Input-PCA is LOSSLESS at rank ≤ n and applies to BOTH #722 AND #658:** c_C has
rank ≤ n=50, so projecting to its top-(n-1) PCs is information-preserving (the
discarded directions are exactly zero), turning the width-512 first linear from
`Linear(3584,512)` into `Linear(49,512)` (~73× fewer first-layer FLOPs) with no
change to the function the MLP can fit. #658's mlpchain script used full-H input;
the input-PCA reparameterization is equivalent and the only way #658's full-H MLP
arm runs in the same throughput class as #722.

[[feedback_left_pad_position_ids_required]] [[feedback_clone_modify_cross_file_drift]]
