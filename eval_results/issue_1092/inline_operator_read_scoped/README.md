# #1092 — SCOPED inline operator-level prefix-arm vs context-arm comparison

**This is the FAST, SUBSET early read** on the Part-B operator comparison; the
comprehensive all-cell version runs later in the off-VM battery. Scope: cells
{`cell_inst_own`, `cell_pre_own`} × layers {14, 18, 19} × target bases
{ambient (pooled t1/t2/t3, 10752-d), pca48} × arms {prefix-end state,
context-end state}. Fit rows: battery-EXCLUDED (`stratum != trait_stratum AND
not is_eval_only`, the corrected filter; n_fit = 17308, 3885 excluded).

## What was computed

Per (cell, layer, arm, basis): a standardized-X ridge map W, GCV-λ-selected on
the full battery-excluded data, then read as the operator **W_raw (H_in=3584 →
P_out)** in the RAW residual-stream input basis (common across arms; per-arm
standardization undone). Prefix-arm W vs context-arm W are compared by:

- principal angles between top-k singular subspaces — INPUT (right) and OUTPUT
  (left) — at k=48 and k=k90 (90% spectral energy);
- orthogonal-Procrustes residual `min_R ||W_c − R·W_p||_F / ||W_c||_F`, R∈O(H_in)
  (so it probes OUTPUT-subspace agreement modulo a free input rotation);
- a spectrum-matched null band (200 draws: observed singular values, random
  orthonormal subspaces) for both;
- anchors: identical-output floor, orthogonal-output ceiling.

Own-λ and matched-λ (context arm's λ for both) variants both reported;
verdict is robust to λ-choice and k-choice.

## One-line verdict

**The two arms share OUTPUT (answer-representation) operator structure far
beyond the spectrum-matched null, while their INPUT (residual-stream read)
subspaces are indistinguishable from random** — consistent with ONE shared
transfer operator whose output subspace both the prefix-end and context-end
states map into, the query changing WHICH input directions carry the signal
(adding input information) rather than adding a new output-operator direction.
All 12 (cell×layer×basis) cells agree: output principal angle median 13–54°
(vs null 36–84°) and Procrustes residual 0.80–0.92 (vs null ≈0.83–1.02) sit
BELOW null in every cell; input principal angle median ≈79–86° sits AT the null.

## Files

- `operator_read.json` — full per-cell numbers (angles, Procrustes, null bands,
  anchors, λ, k90; own_lambda + matched_lambda variants).
- `../../../figures/issue_1092/inline_operator_angles_vs_null.png` — the figure.

## Caveats

- SCOPED subset (2 cells, both "own"-text arms); Part-B off-VM covers the full
  grid (claude/pretext/shuf/insttext arms, all layers).
- Identifiability is clean here (n_fit 17308 ≫ H_in 3584 and ≫ P_out; input
  space fully spanned).
- λ selector is GCV on the full data (single-fit substitute for the banked
  per-fold PRESS-LOO); banked read1 per-fold mode was λ=1000 for both arms,
  GCV picks λ=1000 (prefix) / λ=100 (context) — the matched-λ variant confirms
  the verdict either way. The prefix arm's heavier shrinkage matches read1/read3
  (prefix carries ~10% of transport).
- Ridge shrinkage biases singular values low; both arms are shrunk comparably
  and the spectrum-matched null uses the observed spectra, so the comparison is
  fair.
