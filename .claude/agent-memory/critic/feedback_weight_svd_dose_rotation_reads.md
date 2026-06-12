---
name: Weight-SVD dose-rotation reads (analysis tasks)
description: P0-style adapter-SVD plans — joint-arm v_src ambiguity and sigma1/sigma2 degeneracy are the two analyzer concerns for dose-rotation Spearman; both recoverable iff top-k sigma + top-2 vectors persist per cell
type: feedback
---

For `kind: analysis` plans that SVD stored LoRA adapters and regress key-rotation
(Δcos toward a contrast direction) on dose (#604 v1, theory-doc P0 line):

- **Joint-arm cells have no canonical v_src.** Dial inventories (2 pairs × {A_only,
  B_only, joint} × seeds) include joint cells where the top-1 right-singular vector
  may split between two sources; a registered "Spearman over all clean cells"
  silently forces an unregistered per-cell choice. NOT a REVISE when the analysis is
  fully re-runnable on persisted vectors (analyzer can stratify by arm / use
  per-source rows); flag as a definitional concern for implementer + analyzer.
- **σ₁≈σ₂ degeneracy at low dose** makes the top-1 key direction unstable, which can
  manufacture a mechanical positive Δcos-vs-dose trend (key well-defined only at
  high dose). Recoverable iff per-cell top-k σ spectra + top-2 singular vectors are
  persisted (then the analyzer conditions on spectral gap or uses top-2 subspace
  angles). Check the persistence, not the absence of the confound.
- **H2/H4 interplay:** a successfully rotated key (contrast-direction win)
  mechanically depresses cos(key, raw v_src) — deep contrastive cells can "fail" a
  p95 key-identity bar while the rank-in-bank criterion still passes. Read identity
  misses jointly with rank + Δcos.
- **Layer-index alignment (hook position vs adapter input space).** Per-layer
  centroid hooks capture layer-l OUTPUT, but the attention key at layer l reads
  layer l−1's output (through `input_layernorm`), and the MLP key reads the
  mid-layer residual (captured at neither index). Recoverable iff ALL layers'
  context vectors persist (analyzer re-indexes; band/contiguity criteria are
  ±1-layer insensitive) — check the persistence, then file as a concern: profiles
  should be checked at v_c(l−1) vs v_c(l) before narrating layer localization.

- **Dose-covariate provenance (the v1 stats-lens Must-Fix).** "Realized landing"
  does NOT live in `eval_results/issue_*/sweep/*.json` (planned `band_low/high_nats`
  only) nor `analysis.json` (aggregates, no landing field) — it lives in per-cell
  `eval/*__shift.json` `contexts[<source>].delta_logp_marker`. That re-measure also
  differs from the in-loop band-stop reads quoted in producing bodies (#527 actual
  4.69–8.06 vs quoted 5.00–7.47). A plan citing sweep/ for dose invites silent
  band-midpoint substitution → 3 discrete planned tiers replace the continuous
  realized covariate. Re-derive the dose source file + range yourself.

**Why:** all three arose on #604 v1; none were Must-Fix because the plan persisted
top-8 σ + top-2 vectors per cell and registered both rank and threshold criteria —
the persistence is what keeps these analyzer-weighable.

**How to apply:** on any weight-space SVD plan, verify (1) per-cell spectra/vectors
are persisted, (2) multi-source (joint) cells have a registered handling for
source-keyed statistics, before considering REVISE.
