---
name: Matched-strength equivalence-band designs (rate-space)
description: TOST-style gap-CI ⊂ ±margin AND profile-ρ conjunctions at matched sub-ceiling strength — attenuation and ICC-driven CI-width checks (#606)
type: feedback
---

Rule: for matched-strength LoRA-vs-FT (or any two-arm) equivalence designs whose verdict is "gap 95% CI ⊂ (−m,+m) AND per-persona profile Spearman ρ ≥ θ", do NOT REVISE on margin-vs-power grounds when per-rollout verdicts are persisted and an explicit indeterminate class exists — but always flag two analyzer concerns:

1. **Profile-ρ attenuation at the matched point.** At sub-ceiling matched strength s* the per-persona profile spread compresses relative to endpoint profiles; observed ρ is attenuated by per-persona delta noise (binary-rate SE ~0.05–0.10 per arm at 500 clustered verdicts). A noise-attenuated ρ below θ misclassifies true equivalence as profile divergence. The CI on ρ covers sampling noise, NOT attenuation bias. Fix is free post-hoc iff per-rollout verdicts persist: split-half reliability or analytic disattenuation ceiling reported next to ρ.
2. **Rollout-within-claim ICC drives the gap-CI width.** 50 claims × 10 rollouts = 500 verdicts/persona-cell, but temp-1.0 binary behaviors can be near-deterministic per claim×persona (ICC→1 ⇒ effective N→50). Panel-mean gap half-width then ranges ~0.02 (low ICC) to ~0.06 (high ICC) at 38 personas — straddling a ±0.05 margin. Structurally-indeterminate-under-true-equivalence is possible; analyzer should decompose realized CI variance (claim / persona / binomial) on an indeterminate so "noise-limited" names which N to raise.

Also: a shared linear-in-s interpolant means the #514-style determinacy gate (|plug-in − bootstrap mean|) does NOT detect curvature bias — both estimators use the same interpolation, so curvature enters the gap only via unequal per-arm bracket widths/positions. Demand per-arm bracket endpoints (s_lo, s_hi, steps) in the headline table; the s*-sweep is the robustness read.

**Why:** task #606 plan v1 (2026-06-11) had all the right machinery (crossed cluster bootstrap re-estimating s per replicate, indeterminate class, persisted verdicts) — APPROVE was correct; these were the three reads the analyzer needed pre-naming.

**How to apply:** any equivalence/non-inferiority read on judge-scored rates with cluster structure; any profile-similarity criterion computed at a deliberately sub-ceiling matched point.
