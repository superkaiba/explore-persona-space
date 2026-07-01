---
title: 'Does the best answer-summary change #722''s base-vs-post-FT function-change
  verdict'
kind: experiment
tags:
- answer-summary-sweep
- from-722
created_at: '2026-07-01T18:16:27Z'
has_clean_result: false
parent_id: 722
origin_prompt: what about taking activation at the newline before the next user message,
  similar to what worked well for the context? (this is for a summary of the answer
  profile -- instead of mean answer activation) | can we do the base-map as one issue
  and the base vs post comparison as another issue? Can we also check all the positions
  of the answer (should be cheap right)? - although potentially we already have this
  experiment
goal: 'Re-run #722''s pre/post-finetuning function-change (Delta = median |M+(c)-M0(c)|
  along r_B) and chain-rho on #537''s trained adapters using the best answer-side
  summary identified in the base-map sweep (child-A) in place of the mean, to test
  whether the ''function M moves only for the taught fact'' verdict holds under a
  better answer-profile summary.'
---
# Does the best answer-summary change #722's base-vs-post-FT function-change verdict

## Goal

Re-run #722's pre/post-finetuning function-change (Delta = median |M+(c)-M0(c)| along r_B) and chain-rho on #537's trained adapters using the best answer-side summary identified in the base-map sweep (child-A) in place of the mean, to test whether the 'function M moves only for the taught fact' verdict holds under a better answer-profile summary.

## Overview / Motivation

[#722](https://eps.superkaiba.com/tasks/722) concluded that finetuning
measurably reshapes the context→answer map function `M` **only for a taught
fact** — the taught-fact `Δ = median |M⁺(c) − M0(c)|` along `r_B` clears its
noise floor 1.6–3.3×, while EM and sycophancy sit below floor (but fail the
MLP-vs-shuffle power check, so they are **inconclusive**, not "function held").
That whole verdict used the **mean-over-answer summary** `v0`.

If the sibling task (base-map summary sweep, child-A) finds a **better / cleaner
answer-side summary** — e.g. the turn-boundary `\n` (the mirror of `c_C`) or a
specific answer position — then #722's pre/post-FT reads should be redone with
it: a cleaner base map has more headroom to detect a real function change, and
the EM/sycophancy "inconclusive" calls might resolve.

## Design (single manipulated variable = the answer-side summary)

Reuse #722's exact pre/post-FT harness — #667's paired base+post-FT store
extraction, #537's already-trained behavior×context LoRA adapters, and #722's
function-change / chain-ρ / cross-transfer fit code — changing **only** the
answer-side summary from `mean` to the **winner of child-A**.

- 3 behaviors (harmful-compliance/EM, taught fact, sycophancy) × 3 layers
  (7, 14 primary, 21), the #722 grid.
- Requires a **cheap paired re-extraction** (base + post-FT, forward-pass only)
  over the 16-source paired grid with the new summary position — the mean store
  keeps no per-token/boundary positions, so this cannot be re-fit from disk.

## Dependent variables (identical to #722)

1. **Function-change** `Δ_med / floor_combined` per behavior×layer (floor = the
   post-FT refit-variance null), ridge fit, family-clustered bootstrap.
2. **Chain-ρ** — Spearman between `r_Bᵀ M̂(c)` (held-out LOCO) and the judge
   leakage rate `E` (from #537's G matrix), under M0 vs M⁺, family-clustered CI.
3. **Cross-transfer** — held-out cosine of M0/M⁺ predicting base vs post-FT
   profiles.
4. **MLP-vs-shuffle validity gate** — trust the MLP read only where its base-map
   held-out ρ beats the label-shuffle control.

Headline: does the "function moves only for the taught fact" verdict hold under
the better summary, and do the EM/sycophancy inconclusive calls resolve?

## Dependency

**Runs after child-A.** The summary this task uses is child-A's winner; do not
launch until A reports which answer-side summary/position wins the base-map read.

## Reuse (artifact-reuse fitness — planner to verify)

- #667 paired-store extraction pipeline (`scripts/issue667_extract.py`) +
  #537 trained adapters (`i537_*`, r=32 rsLoRA) + #537 G leakage matrix
  (`eval_results/issue_537/G_tensor/`).
- #722 function-change / chain-ρ fit harness.
- Single variable changed vs #722 = the answer-side summary.

## Cost

~8 GPU-h (paired base+post-FT re-extraction, forward-pass only; the fit harness
is minutes). Under the 20-GPU-h cheap band.

## Provenance

Standalone child of #722 (user-directed split, 2026-07-01; filed standalone
rather than a same-issue follow-up because #722 is wedged — see child-A). Sibling
of the base-map summary-sweep task (child-A), on which this depends.

Verbatim originating prompts:
> what about taking activation at the newline before the next user message,
> similar to what worked well for the context? (this is for a summary of the
> answer profile -- instead of mean answer activation)
>
> can we do the base-map as one issue and the base vs post comparison as another
> issue? Can we also check all the positions of the answer (should be cheap
> right)? - although potentially we already have this experiment
