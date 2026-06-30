---
title: 'Decoding-ceiling, linear-information-loss, and sample-complexity of #658 base-model
  behavior representations (n=50)'
kind: experiment
tags:
- leak-predictor
- keep-running
created_at: '2026-06-30T00:23:55Z'
has_clean_result: false
parent_id: 658
origin_prompt: 'Run this: Single staged kind: experiment task, 0-GPU, on existing
  #658 artifacts — not a campaign. These are tightly-coupled sequential analyses on
  one dataset with an internal gate, not independent question-arms; campaign overhead
  isn''t warranted.'
goal: 'On the existing #658 base-model representations (Qwen2.5-7B v0(C) per layer,
  judged behavior rates E0(C,B), n=50 contexts), measure how much behavioral information
  linear decoding loses, the achievable decoding ceiling, and the sample size required
  — operationalized at n=50 as the per-behavior reliability ceiling sqrt(r_yy), the
  bracket [rho_linear, sqrt(r_yy)] bounding the linear-decoding information loss plus
  a LEACE+dCor test for any nonlinearly-extractable residual, and a learning curve
  extrapolating the contexts needed to resolve the gap.'
relates_to:
- leak-predictor
---
# Decoding-ceiling, linear-information-loss, and sample-complexity of #658 base-model behavior representations (n=50)

## Goal

On the existing #658 base-model representations (Qwen2.5-7B v0(C) per layer, judged behavior rates E0(C,B), n=50 contexts), measure how much behavioral information linear decoding loses, the achievable decoding ceiling, and the sample size required — operationalized at n=50 as the per-behavior reliability ceiling sqrt(r_yy), the bracket [rho_linear, sqrt(r_yy)] bounding the linear-decoding information loss plus a LEACE+dCor test for any nonlinearly-extractable residual, and a learning curve extrapolating the contexts needed to resolve the gap.

> **Status:** proposal for `/adversarial-planner` to formalize into a plan. 0-GPU; all analyses run on existing #658 artifacts. Planner finalizes exact estimators, layer/PCA grid, and composition.

## Provenance

New child direction off [#658](https://eps.superkaiba.com/tasks/658), arising from a chat investigation of #658's A3.2/A3.3 read-outs. The methodological question was scoped against a verified three-part literature review (see § Feasibility verdict + § References). Verbatim originating request:

> Run this: Single staged kind: experiment task, 0-GPU, on existing #658 artifacts — not a campaign. These are tightly-coupled sequential analyses on one dataset with an internal gate, not independent question-arms.

Earlier framing prompts in the same thread: "how much information is lost by linear decoding / what is the ceiling we can hope to achieve / how many contexts would we need to make these quantities estimable (and is it even possible at all)" and "is there some information theoretic quantity we can use to see if the information is present at all".

Anchor: `docs/open_questions.md` `leak-predictor`.

## Background & motivation

#658 tested whether a base-model-readable chain predicts where fine-tuned behavior leaks. Two chat re-analyses reshaped its reading: (a) A3.2's MLP failure on the 4 read-out behaviors is an **estimator artifact** — a regularized ridge passes all 4 (sycophancy ρ≈0.72, refusal ρ≈0.70) where the n=50 MLP overfit; (b) the diff-in-means `r_B` read-out is weak for sycophancy/refusal, but the **information is linearly present in `v0`** (ridge finds it). That raises the foundational question this task answers: independent of any specific read-out direction or decoder, **how much behavioral information is actually present in `v0(C)`, how much does linear decoding leave on the table, what ceiling could any decoder reach, and is any of this even estimable at n=50?**

The literature review returned a hard but clean verdict (below): at n=50 the *magnitude* questions (mutual information in nats; the achievable Bayes-optimal ceiling; the linear-vs-nonlinear gap size) are largely out of reach, but a well-chosen set of **bounds + a yes/no nonlinear-residual test + a learning curve** is estimable now and answers the questions in their honest form.

## Setting (the measured objects, all already on disk)

- `v0(C)` — mean answer-token residual-stream activation per context, all 28 layers, `Qwen2.5-7B-Instruct`. Local: `data/issue_658/store/v0_summaries.pt` (Betley) + the `issue_658_g1` store (UltraChat).
- `E0(C,B)` — fraction of on-policy completions a `claude-sonnet-4-5` judge labels as expressing behavior B, per context. Local: `eval_results/issue_658/E0_expression.json` (+ `_g1`). Each `E0(C,B)` is `k`-of-`m` rollout counts (carries binomial measurement noise).
- n = 50 contexts (the binding constraint); 10 behaviors (4 with a read-out contrast); 2 genres (Betley, UltraChat).
- The existing per-behavior "noise floor" (p95 of probe-resample correlations) — to be disentangled into a null vs a reliability ceiling (Stage 0).

## Formal quantities & the bracket

Let `C*` = correlation of the Bayes-optimal predictor `f*(v0)` with `E0` (the true achievable ceiling), `ρ_lin` = held-out linear (ridge) decoding correlation, `r_yy` = reliability of `E0` at the realized rollout count.

1. **Reliability ceiling** `√(r_yy)` — the max correlation *any* decoder can reach before hitting the target's own measurement noise (attenuation identity, Spearman 1904; ≡ neuroscience noise ceiling CC_max²=SP/TP, Schoppe 2016). Estimable from the target's noise structure alone — does **not** require the high-dim joint density.
2. **The bracket** `ρ_lin ≤ C* ≤ √(r_yy)`. Both ends estimable at n=50; `C*` itself is not. **The bracket width `√(r_yy) − ρ_lin` is the estimable *upper bound* on the information lost by linear decoding** (Q1 in its honest form). Tight bracket ⇒ linear is ceiling-limited, loss ≈ 0, definitively. Wide bracket ⇒ loss *could* be large but its magnitude is not resolvable at n=50.
3. **V-information framing** (Xu 2020): "info lost by linear decoding" = `I_{V_nonlinear} − I_{V_linear}` in nats; ridge R² *is* the linear-class V-information term. By the data-processing inequality there is **no model-free "total information"** (Shannon MI is fixed by the context) — so the object is class-relative usable information, never Shannon MI in nats.

## Staged protocol (0-GPU; internal gate after Stage 0)

**Stage 0 — reliability ceiling + the bracket (gates the rest).**
- Per behavior × genre, estimate `r_yy` two agreeing ways: (i) split-half-over-rollouts + Spearman–Brown `r_yy = 2r_half/(1+r_half)`; (ii) binomial variance decomposition `SP = Var_C(E0) − mean_C[p̂(1−p̂)/m]`, ceiling² = SP/Var_C(E0).
- Add a **judge-rerun variance term**: re-run the `claude-sonnet-4-5` judge ≥2× on a context subset to capture judge stochasticity (rollout-splitting alone misses it). The honest ceiling folds both. (Only non-CPU cost; small judge spend.)
- Compute the ceiling **cross-validated / matched to the LOCO protocol** (Storrs 2020) — never compare a pooled-reliability ceiling to a CV'd ρ.
- Bootstrap over the 50 contexts for the CI on `√(r_yy)` (expect ±0.10–0.15).
- Lay `ρ_lin` (the #658 ridge result) next to `√(r_yy)` → the bracket. Cross-check: do #658's existing "noise floor" values (0.68–0.93 for self_report/persona_drift) actually equal `√(r_yy)`? If so, #658 conflated the null and the ceiling — disentangle.
- **Gate:** per behavior, if `ρ_lin ≈ √(r_yy)` within CI → ceiling-limited, Q1 ≈ "≈0 lost", record and skip Stage 1 for that behavior. If `ρ_lin ≪ √(r_yy)` → headroom → Stage 1.
- Free companion: binary per-completion Bayes-error ceiling `β = E_C[min(E0, 1−E0)]` (Ishida 2023), no model.

**Stage 1 — is *any* of the headroom nonlinearly extractable? (only where the bracket is wide).**
- **PCA-reduce `v0` to ~10 dims** (the single highest-leverage move — restores dependence-test power, pushes regression toward the parametric regime, lowers d in the minimax rate).
- **LEACE** (Belrose 2023, closed-form, stable at small n) to erase the linear `E0` signal from `v0`; then a **dCor / HSIC permutation test** (Székely 2007 / Gretton 2005) for residual dependence between erased-`v0` and `E0`. Answers Q1 in the only form n=50 allows: a **yes/no on "nonlinear signal beyond linear,"** not a magnitude.
- **Control-task / selectivity** (Hewitt–Liang 2019): refit on shuffled `E0`; report any residual net of the shuffle null. Mandatory at d=3584 ≫ n=50.
- Optional robustness: MDL / prequential codelength gap (Voita–Titov 2020) and PVI-per-context (Ethayarajh 2022) to check whether any apparent gain concentrates in 2–3 overfit contexts.
- Explicit bound: a sample-efficient nonlinear decoder (kernel ridge / GP / the LEACE-residual probe), **never the n=50 MLP** — MLP<ridge means "no measured gain at this estimator/n," not "no nonlinear info."

**Stage 2 — the learning curve = the "how many contexts" experiment.**
- Subsample n′=10,15,…,50 (B repeats); at each n′ compute `ρ_lin`, `√(r_yy)`, and dCor with bootstrap variance; plot vs n′ and extrapolate the n required to (a) resolve a target gap (e.g. 0.05 R²) and (b) bring the ceiling CI below a target width.
- Report against the theory verdict: MI-in-nats is capped at ~ln n ≈ 3.9 nats and needs ~eᴵ samples (McAllester–Stratos 2020; Gao 2015; Song–Ermon 2020); the gap/ceiling need n in the low hundreds *with* d_eff≤~10 (Stone 1982 minimax + Varoquaux 2018 CV-variance); dependence yes/no is feasible now.

## Methodological backbone (triangulated across all three lit threads)

PCA-reduce `v0` to ~10 dims first · report everything **relative to `√(r_yy)`** · permutation / control-task nulls, **never std-across-folds** (Bengio–Grandvalet 2004) · bootstrap over the 50 contexts, expect ±0.10–0.15 CIs · the n=50 MLP is untrustworthy as an "info upper bound" (use ridge / kernel-ridge / LEACE-residual) · **never report MI in nats** · disattenuation (dividing ρ by √r_yy) is unstable at small n — report the bracket, not the disattenuated ρ.

## Feasibility verdict (what this task can vs cannot deliver, from the lit)

- **Deliverable now:** the reliability ceiling `√(r_yy)` per behavior (+ CI); the bracket `[ρ_lin, √(r_yy)]` and its width (upper bound on linear-decoding loss); a yes/no LEACE+dCor nonlinear-residual test; the learning curve + the theory-grounded n-requirement.
- **Provably / practically NOT deliverable at n=50 (state plainly, do not fake):** MI in nats (McAllester–Stratos ln-n cap; eᴵ samples); the *magnitude* of the linear-vs-nonlinear gap (inside the ±0.10–0.15 CV band); the achievable Bayes-optimal ceiling `C*` directly (curse of dimensionality + uncontrolled CV variance) — only its bracket.

## Pass/fail & deliverables

- Per behavior × genre: `√(r_yy)` ± CI; the bracket + width; the Stage-0 gate verdict (ceiling-limited vs headroom); where applicable, the Stage-1 nonlinear-residual yes/no (with control-task null).
- The learning-curve figures + the extrapolated n-to-resolve and the explicit "MI-in-nats is out of reach" statement.
- A clean-result whose headline is the bracket + the gate verdict (e.g. "for K of the read-out behaviors linear decoding is at the reliability ceiling; for the others there is/ isn't a detectable nonlinear residual"), every claim CI'd and reliability-relative.

## Data & compute

0-GPU. All inputs exist: `data/issue_658/store/` v0 summaries, `eval_results/issue_658{,_g1}/E0_expression*.json`. Only spend is the Stage-0 judge-rerun variance term (a small `claude-sonnet-4-5` batch on a context subset). No pod / no training. (Stage 2 says nothing about generating more contexts here — that is a downstream decision the learning curve informs.)

## Key references (verified in the lit review)

Ceiling/reliability: Spearman 1904 (attenuation); Schoppe et al. 2016 (noise ceiling, DOI 10.3389/fncom.2016.00010); Nili et al. 2014; Storrs et al. 2020; Ishida et al. 2023 (arXiv 2202.00395). Info-loss/V-information: Xu et al. 2020 (arXiv 2002.10689); Ethayarajh et al. 2022 (2110.08420); Hewitt & Liang 2019 (1909.03368); Voita & Titov 2020 (2003.12298); Belrose et al. 2023 LEACE (2306.03819); Pimentel et al. 2020 (2004.03061). Sample-complexity/feasibility: McAllester & Stratos 2020 (1811.04251); Gao et al. 2015 (1411.2003); Song & Ermon 2020 (1910.06222); Székely et al. 2007 (0803.4101); Reddi et al. 2015 (1406.2083); Stone 1982; Bengio & Grandvalet 2004; Varoquaux 2018 (1706.07581).
