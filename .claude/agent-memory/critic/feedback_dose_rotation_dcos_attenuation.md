---
name: Dose-rotation Δcos attenuation + multi-source arm well-posedness (#604)
description: Difference-of-cosines rotation indices on ESTIMATED directions confound rotation-with-dose against SNR-growth-with-dose; joint/multi-source arms need a registered source convention before they enter a single-source-keyed DV
type: feedback
---

Two recurring patterns from #604 (P0 adapter SVD, statistics lens):

1. **Δcos = cos(key, u_contrast) − cos(key, u_raw) vs dose confounds rotation with
   estimation SNR.** If key_est ≈ normalize(key_true + ε) with ε shrinking as dose
   grows (top-1 energy rises with dose, #538), BOTH cosine terms scale by a common
   factor λ(dose) → Δcos_obs = λ·Δcos_true. A key that is contrast-aligned at ALL
   doses (Δcos_true > 0 constant, no rotation) mechanically yields a positive
   dose-Spearman. Discriminators: (a) in-plane angle of P_span{u_raw,u_contrast}(key)
   — attenuation lowers in-plane ENERGY but preserves the angle in expectation;
   (b) seed-stability of the key per dose point as the empirical SNR proxy; (c) a
   matched-control arm at each dose (e.g. #474 pos arm) absorbs the SNR trend, so the
   paired contrast is the cleaner half of the read.
   **How to apply:** Concern (not REVISE) iff the singular vectors + context vectors
   are persisted so the angle is a free post-hoc compute; tell the analyzer never to
   narrate the raw Δcos Spearman as "rotation with dose" without the angle or
   stability-stratified read.

2. **Joint / multi-source arms in a single-source-keyed DV.** Cells that train TWO
   positive sources (dial `joint` arm) have no well-defined v_src / u_contrast for a
   per-cell top-1-key metric; the producing sweep JSON stores ONE scalar dose for the
   cell. If such cells sit inside the registered primary statistic (15/45 in #604),
   the implementer silently invents a source convention that shapes the headline.
   **How to apply:** Must-Fix at plan time — either restrict the primary to
   single-source arms (joint as sensitivity via per-source matched top-2 vectors /
   subspace capture) or register the joint convention explicitly.

Bonus from the same review: stacked-module SVD objects ([ΔW_q;ΔW_k;ΔW_v]) have max
rank 3r, so uniform-floor/diffuse calibration is 1/(3r), not 1/r; and persisting only
top-8 σ truncates the tail needed for post-hoc concentration metrics (full spectrum
is ≤96 floats — free).
