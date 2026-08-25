---
title: 'Correct #823 Takeaway 4: the persona-ladder decline is not denominator arithmetic
  (denominator-free paired read refutes the mixture-floor interpretation)'
kind: infra
tags: []
created_at: '2026-08-22T20:03:18Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'User chat 2026-08-22: ''yes run it now'' — shared-persona paired ss_res
  read on #823''s inconsistent-origin ladder; the result refutes the promoted Takeaway
  4 clause ''no evidence the map itself becomes harder to learn''.'
workflow: v1
---
# Correct #823 Takeaway 4: the persona-ladder decline is NOT denominator arithmetic

## Goal

Correct a refuted bolded Takeaway in the promoted body of task #823, and decide
whether the parent round's `mixture_floor` interpretation needs a re-read.

## What is refuted

#823's promoted `## Takeaways` bullet 4 currently reads, in part:

> the mechanical mixture penalty implied by between-persona target variance accounts
> for essentially all of the decline (implied 0.171 vs observed 0.156 at layer 14;
> 0.192 vs 0.202 at layer 26): mixing target origins caps attainable R² mechanically,
> with no evidence the map itself becomes harder to learn.

The `## Results` section "The implied mechanical mixture penalty accounts for essentially
all of the persona-count decline" carries the same reading:

> So the measured degradation under inconsistent origins is essentially what mixed targets
> mechanically force, not evidence the context→content association weakens.

A denominator-free paired read run 2026-08-22 contradicts the "no evidence the map itself
becomes harder to learn" clause.

## Refuting evidence

Producer: `scripts/issue823_shared_persona_paired.py`
Artifact: `eval_results/issue_823/inconsistent_origin_ladder/shared_persona_paired.json`

Method. Generation in the ladder round was deduplicated per (context, persona) — 14,996
unique pairs == `n_pairs` == `registered_total_pairs_full`. Under the registered rule
`persona(i, k) = i mod k`, a context with `i mod K == 0` is assigned persona 0 in BOTH the
k=1 arm and the k=K arm, so on those contexts the two maps predict the IDENTICAL target
vector, scored in the SAME test fold (`KFold(5, shuffle=True, random_state=0)` depends only
on n) from the same training context indices. Their held-out per-context `ss_res` is
therefore directly comparable with no denominator. Both arms' errors are out-of-fold
(`scripts/issue823_ladder_fits.py:1903` writes `p1_sres[..., te]` at test indices only).

Result, k=16 vs k=1 on 310 shared contexts (all three read-out layers: evil 14,
sycophancy 26, hallucination 17):

| | L14 | L26 | L17 |
|---|---|---|---|
| ss_res ratio pooled/reference | 1.310 | 1.303 | 1.305 |
| mean paired diff (bootstrap 95% CI) | +62.0 [+51.2, +73.1] | +2352.8 [+1891.3, +2806.4] | +84.9 [+69.3, +101.3] |
| contexts where pooled is worse | 85.2% | 85.2% | 87.1% |
| Wilcoxon p | 8.7e-36 | 4.1e-32 | 4.1e-35 |
| common-denominator R², reference | 0.514 | 0.513 | 0.527 |
| common-denominator R², pooled | 0.364 | 0.366 | 0.383 |
| ss_tot ratio pooled/reference | 1.058 | 1.052 | 1.054 |

The common-denominator R² gap on the shared subset is ~0.15, essentially the whole headline
pooled drop (0.156 / 0.202 / 0.150). The denominator moves only 5–6% on this subset. So the
decline lives in the NUMERATOR — held-out prediction error on clean single-origin targets —
not in the scoring denominator.

Monotone in k, same direction at every layer: ss_res ratio 1.10–1.17 (k=2, n=2,186) →
1.19–1.27 (k=4, n=1,223) → 1.25–1.28 (k=8, n=615) → 1.30–1.31 (k=16, n=310).

Representativeness: the k=1 arm's own mean `ss_res` on each shared subset is 0.947–0.990 of
its mean over the whole 4,629-context mask, so the `i mod K == 0` slice is typical, not an
easy subset.

## Offset-bias control (the discriminator)

Write `v_j(x) = m(x) + p_j` with persona 0 as reference. If origins share one map and differ
only by a constant offset, a pooled fit converges to `m(x) + p_bar` and its excess squared
error on persona-0 contexts is `||p_bar||² ≈ E/k`, where `E` is the between-persona
mean-shift energy the parent round already computed
(`ladder_analysis_summary.json → mixture_floor.implied_mixture_penalty`).

Measured excess is 3.4–12.8× that prediction, and the discrepancy WIDENS with k (3.4× at
k=2 → 12.3× at k=16) — the opposite of what the offset mechanism implies. Instead the excess
tracks the FULL energy `E`: ratio 0.73–0.80 at k=8 and k=16. All 12 arm×layer cells return
`excess-tracks-full-between-persona-energy`.

Reading: the mixture-floor MAGNITUDE the parent round computed is the right scale, but it is
not acting as a denominator cap. The pooled fit appears to absorb persona variation into its
coefficients rather than its intercept, so held-out predictions degrade on clean
single-origin targets at roughly full `E`.

## Named confound (do not drop from the correction)

The underlying ladder fits sit at `n_over_d_ratio` 1.033 (`n_train_per_fold` [3703, 3704] vs
`d` 3584) — the interpolation threshold — with `g2_parity_gate.pass: false` and
`sensitivity_dof_capped: null`. Near interpolation a ridge can nearly fit its training
targets including their persona offsets, which then generalize as noise; a well-conditioned
fit (n >> d) would average those offsets into the intercept and pay only ≈ E/k. So the
degradation measured here may be specific to the near-interpolation regime rather than a
structural property of the context→answer map.

That is a falsifiable prediction worth recording: at larger n/d, `measured_excess / E` should
fall toward 1/k. Testing it needs either dimensionality reduction or more contexts, so it is
NOT part of this correction task.

## Deliverables

1. Amend #823's Takeaway 4 and the "implied mechanical mixture penalty" Results section so
   the "no evidence the map itself becomes harder to learn" clause is replaced by the
   denominator-free finding plus the near-interpolation confound. #823 is at
   `awaiting_promotion`; `runs.classification` must NOT be touched.
2. Decide whether the paper paragraph "Consistent origin is what matters, not on-policy text"
   (`~/overleaf-6a59c927/sections/results/c1_linear.tex:93`) should cite this read. Its
   `% TODO-VERIFY` comment still says the inconsistent-origin arm is unrun/in-flight; the arm
   landed 2026-08-20 and this read is its denominator-free follow-up.
3. Refresh `docs/paper_context_answer_map/claims.md` row K4 and the register line, both of
   which still say the arm is IN-FLIGHT.

## Provenance

Surfaced by a user-chat inline free analysis on #823 (0 GPU-h, existing artifacts),
2026-08-22. Dispatch and completion notes are `epm:progress` markers on #823.
