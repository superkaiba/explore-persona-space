---
title: 'Pilot: cheap context readout to mine always-comply jailbreak contexts; does
  the context→answer map help?'
kind: analysis
tags: []
created_at: '2026-08-19T19:59:15Z'
has_clean_result: false
origin_prompt: 'run it in background with happy coder (jailbreak-mining pilot: C/D/E
  across 3 maps + label-efficiency + transfer)'
workflow: v1
---
---
kind: analysis
---
# Pilot: can a cheap context readout mine always-comply jailbreak contexts, and does the context→answer map help?

0-GPU exploratory pilot (reuses #1739/#1092 raw activations + StrongREJECT-style compliance labels — RAW DATA only, not #1739's conclusions). Host task for a background Happy Coder run; not the formal experiment.

## Question
In a needle-in-haystack corpus, can a cheap context-side signal (one forward pass, no generation/judge) surface contexts that RELIABLY jailbreak the model (always-comply, StrongREJECT compliance DV at ceiling) vs same-family FAILED jailbreaks — and does routing through the fitted context→answer map M add anything over a plain learned probe?

## State (already committed to docs/scratch/jailbreak_mining_pilot.md; script scripts/issue1739_jbmine_pilot.py)
- Pool: 150 always-comply positives vs 3,787 failed-compliance same-family hard negatives (evil_train + evil_hh_rlhf), 5% needle base rate, 6 layers, grouped-by-context CV.
- DONE: A = learned probe on context v_C → PR-AUC 0.973 (ROC 0.998, hit@5% 0.93, evals-to-20 = 20). B = map-then-project (fixed r_B on M·v_C) → PR 0.323. So a LEARNED probe wins; a FIXED direction loses.

## Remaining arms (run these; commit EACH by explicit path as it finishes)
Definitions: v_C = context_end activation; v_A = real answer-span mean; M·v_C = mapped/predicted answer. M regimes: M_benign (3k disjoint benign WildChat/LMSYS), M_indomain (held-out grouped-disjoint jailbreak (v_C,v_A) split), M_merged (union; report n_benign:n_indomain). Keep M-train / probe-train / test mutually disjoint (grouped by context).
1. E (answer-space oracle): probe on real v_A → v_A. Ceiling.
2. C (probe on M·v_C → M·v_C) and D (probe trained on real v_A, applied to M·v_C) — each under M_benign, M_indomain, M_merged. Report each map's held-out reconstruction R² on jailbreak-test contexts.
3. Label-efficiency sweep: N ∈ {10,20,40,80,160,320} in-domain labels; compare A_N vs D_N(in-domain), D_N(merged); PR-AUC vs N + N-to-reach-0.80-PR per arm; several draws; flag D's label-cost asymmetry (generated answers).
4. Cross-family transfer: train on one family, test the other, both directions; A vs D via in-domain/merged maps; within-family ref + random floor; power caveat (2 families). BONUS: fix evil_toxicchat compliance-parse for a 3rd family (don't block).

## Deferred (do NOT run without user OK)
(3) off-the-shelf out-of-domain answer classifier — the literal text-classifier version is infeasible (map outputs an activation, not text); the real version needs a small out-of-domain harmful-answer capture (small GPU). Parked pending user decision.

## Discipline
0-GPU CPU only, no pod. Content hygiene: paths/numeric-ids only, never page harmful completion text. Reuse cached intermediates on the data disk (don't re-reduce the 32GB tar if the npz exists). Read any figure PNG back before committing. Post an `epm:progress` note per arm; final summary when all arms land.

## Verdict to produce
One table (A,B recap; C/D under 3 maps; E; label-eff; transfer) + verdict: (1) does C≈A (reparametrization)? (2) does D beat B's 0.32 / approach oracle E? (3) does in-domain/merged map fix the map's failure? (4) does the map win on label-efficiency (D<A at low N) or transfer? If the map loses everywhere, say so plainly.
