---
title: 'Canonical Rao-Blackwellized sequence-level JS replaces deprecated JS-v1 on
  #532''s panel'
kind: experiment
tags: []
created_at: '2026-06-09T20:32:00Z'
has_clean_result: false
parent_id: 532
goal: 'Implement the canonical Rao-Blackwellized sequence-level Jensen-Shannon divergence
  (Amini/Vieira/Cotterell 2025, arXiv 2504.10637) per persona-distance-metrics.md,
  re-fit the JS arm of #532''s predictor leaderboard against the same 416-cell ep1
  panel, and test whether the canonical estimator changes the JS-arm conclusion that
  #532 measured with the deprecated v1 single-next-token estimator.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
## Goal

Implement the canonical Rao-Blackwellized sequence-level Jensen-Shannon divergence (Amini/Vieira/Cotterell 2025, arXiv 2504.10637) per persona-distance-metrics.md, re-fit the JS arm of #532's predictor leaderboard against the same 416-cell ep1 panel, and test whether the canonical estimator changes the JS-arm conclusion that #532 measured with the deprecated v1 single-next-token estimator.

## Motivation

#532's plan §A12 explicitly flagged that the JS arm used the DEPRECATED v1 single-next-token estimator (`scripts/issue458_predictor_jsdiv.py`) for back-compat with the #404/#458/#502 leaderboard line; the canonical Rao-Blackwellized sequence-level JS per arXiv 2504.10637 is not yet implemented in this codebase. The parent body acknowledged this as a scope caveat: H1/H2 do NOT lean on the JS arm (cosine + GKL@L22 + base-prior carry the headline). But the JS arm's ρ_union = −0.350 + ρ_instructed = −0.17 numbers are still in the leaderboard, and a methodology-cleanup pass is the right way to close the §A12 caveat.

The canonical RB JS expects an apples-to-apples comparison against cosine + GKL@L22. The hypothesis is that the direction of the JS arm's contribution is preserved but the magnitude shifts (stronger ordinary-only ρ, similar collapse on the instructed strip). A surprising flip (canonical RB JS recovers strong ρ_instructed) would update the parent's "geometry stops carrying signal in instructed contexts" finding.

## Hypotheses

- **H1:** Canonical RB JS recovers a stronger ordinary-only ρ (|ρ_ord| > |0.40|, closer to GKL@L22's −0.62) on the same 416-cell ep1 panel.
- **H2:** Canonical RB JS still collapses to near-null on the instructed strip (|ρ_instr| < |0.20|), preserving #532's "geometry stops carrying signal in instructed contexts" direction.

## Proposed design (planner owns the final spec)

- Reuse the same 26-bystander panel, the same 16-source #474 loc-arm ep1 LoRA adapters, the same `q_test_extended_50` probes, and the same on-policy ※ emission rate DV from #532's per-cell JSONs.
- Implement the canonical RB JS estimator per arXiv 2504.10637 §3 in a new module (`src/explore_persona_space/analysis/js_canonical.py` or equivalent). Unit-test against (a) closed-form JS of two known Gaussians and (b) v1 estimator in the first-token limit.
- Re-extract the 26 × 16 pairwise JS matrix using the canonical estimator (sample-based: ~10 trajectories per (probe, persona), vLLM-batched).
- Re-fit the 6-regression hierarchy with the new JS column replacing the v1 JS column. Same CV (5-fold leave-one-class-out). Same permutation tests (1000 reps).

## What we learn

Closes the §A12 caveat #532 explicitly flagged. Produces the apples-to-apples JS-vs-cosine-vs-GKL comparison the predictor program needs to be defensible as a leaderboard. Strengthens or weakens the JS arm's position; preserves the cosine / GKL / base-prior headline of the parent.

## Reuse / cost

- **Cost:** ~3 GPU-hours on 1× H100 (canonical RB JS is sequence-level: ~10 sampled trajectories per (probe, persona) per layer batched through vLLM, plus ~30 min unit-test bring-up).
- **Reuse:** #532's per-cell ep1 panel + base-prior payload (NOT the v1 JS predictor matrix — that's the variable being changed); #474 loc-arm ep1 LoRA adapters (already on HF).
