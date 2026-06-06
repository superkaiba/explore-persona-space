- Layer 27 of Qwen-2.5-7B is the residual stream POST the final transformer block but PRE the final RMSNorm — the cumulative bf16 accumulation noise in the residual stream is depth-dependent.
- GPU-verified per-layer max |cosine diff| against a same-recipe #406 reference run, on the q_test_prefix_50 slice (240 cond-pairs per layer):
  L0=2.15e-6  L5=1.95e-4  L11=5.01e-4  L15=1.13e-3  L21=2.70e-3  L27=6.15e-3
- The L21→L27 ratio of 2.28× matches the L21→L27 Frobenius-norm ratio of 2.34× (residual stream norm: 129 → 301), so the diff scales with the accumulated residual magnitude — NOT with a recipe bug.
- Validation: Pearson r between our L27 matrix and #406's L27 matrix = 0.999976; Spearman ρ = 0.999362 (rank order preserved at 99.94%); diff sign distribution -182 vs +58 (small slow drift in one direction, not a sign-locked recipe bug).
- A genuine extraction recipe error at L27 (the round-5 `hidden_states[28]` post-norm quirk) produced 1.6e-1 cosine diff — 26× larger than the bf16 noise floor — so a 1e-2 cap at L27 still catches real bugs.
- Implication for future L27 work: a strict 3e-3 cosine cross-check against an inner-layer cap will fail-loud on bf16 numerical drift alone, with no recipe bug present. Per-layer tolerance map is the correct guard: `{27: 1e-2, "default": 3e-3}`.

**Why:** Task #502 round-7 (2026-06-06) failed strict 3e-3 at L27 with max |diff| = 6.15e-3 after a clean 3.6-GPU-h extract. Diagnostic ruled out position-handling bugs (no inter-layer pattern shift) and confirmed bf16 accumulation. Fix was a documented per-layer relaxation, NOT re-extraction.

**How to apply:** Whenever a same-recipe activation cross-check fails strict-tolerance at the deepest layer ONLY, with all shallower layers passing, run the depth-scaling diagnostic FIRST (Frobenius-norm progression + Pearson r + Spearman ρ) before re-extracting. A monotonic 2-3× ratio per "depth doubling" + Pearson > 0.999 = bf16 accumulation, NOT a bug. Relax the deep-layer tolerance and lock it in with a smoke check that 7×-floor noise PASSes at the deep layer + FAILs at inner layers.
