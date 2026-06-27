---
title: Does a behavior-dependent source key predict the leakage context-gate better
  than the context-only default — marker (precondition holds) vs sycophancy (precondition
  open)
kind: experiment
tags:
- leak-predictor
- mentor-dan
created_at: '2026-06-27T02:43:43Z'
has_clean_result: false
parent_id: 526
origin_prompt: Run the test on marker and sycophancy in the background with happy
  coder -- what test are you running exactly? (A8 behavior-dependent source-key ablation
  for the leakage context-gate, two-behavior contrast marker vs sycophancy)
goal: 'Test whether a behavior-dependent source key for the leakage context-gate (teacher-forced
  training-completion activation t_{C,B}, or the displacement delta_{C,B}=t_{C,B}-v_base(C))
  predicts the realized gate g_real(C'')=<w_hat,Delta_v(C'')>/<w_hat,w_hat> better
  OUT-OF-SAMPLE than the theory''s default context-only key k=c_C, and whether any
  winning key generalizes from the marker (rank-1 scalar-gate precondition holds;
  k=c_C already falsified in #604) to sycophancy (precondition unresolved per #637).'
---
## Goal

Test whether a behavior-dependent source key for the leakage context-gate (teacher-forced training-completion activation t_{C,B}, or the displacement delta_{C,B}=t_{C,B}-v_base(C)) predicts the realized gate g_real(C')=<w_hat,Delta_v(C')>/<w_hat,w_hat> better OUT-OF-SAMPLE than the theory's default context-only key k=c_C, and whether any winning key generalizes from the marker (rank-1 scalar-gate precondition holds; k=c_C already falsified in #604) to sycophancy (precondition unresolved per #637).

## Formalization (object of study)

Grounded on the leakage-theory paper (`~/overleaf-6a2df2d2/main.tex`, Assumptions A7 "scalar-gated off-source write" + A8 "bilinear gate"). The planner must re-read A7/A8 verbatim and pull the recipe from the paper, not this summary.

**Quantities** (all activations = answer-side mean residual-stream activation at a chosen layer `l`, over the model's OWN generations under a context condition):
- `v_theta(C)` — mean answer-side activation under condition `C` for model `theta`.
- Empirical source write: `w_hat = Delta_v(C) = v_trained(C) - v_base(C)`.
- Target shift: `Delta_v(C') = v_trained(C') - v_base(C')`.
- **Realized gate (ground truth):** `g_real(C') = <w_hat, Delta_v(C')> / <w_hat, w_hat>`, normalized so `g_real(C)=1`.
- Context vector `c_C` (prompt-side summary), teacher-forced training-completion activation `t_{C,B}` (base model run on the training completions), displacement `delta_{C,B} = t_{C,B} - v_base(C)`, background context second moment `Sigma_c`.

**Precondition (A7), measured per behavior — NOT assumed:** is the off-source write scalar/rank-1? Report the scalarity residual `||Delta_v(C') - w_hat * g_real(C')|| / ||Delta_v(C')||` and the SVD spectrum `sigma_1^2 / sum_j sigma_j^2` of the stacked target-shift matrix `[Delta_v(C'_1) ... Delta_v(C'_n)]`. If rank-1 holds the scalar `g_real` is a faithful summary; if it fails, fall back to the theory's own low-rank relaxation `Delta_v(C') ~ sum_j w_j g_j(C')` and test the keys against the dominant component(s).

**Manipulated variable — three candidate source keys**, each a BASE-MODEL predictor of `g_real`:
- (i) `k = c_C` — context-only / behavior-free (the paper's default; #604 falsified it for the marker).
- (ii) `k = psi(t_{C,B})` — behavior-dependent (teacher-forced training-completion activation).
- (iii) `k = c_C + psi(delta_{C,B})` — behavior-dependent (displacement).
crossed with metric `M in {I, (Sigma_c + lambda*I)^-1}`. Predicted gate `g_pred(C') = (k^T M c_{C'}) / (k^T M c_C)`.

**Measurement / scoring:** Spearman (primary) + Pearson + sign-agreement + MAE of `g_pred` vs `g_real` across HELD-OUT target contexts (leave-one-context-out), per behavior, with a shuffled-key null and cross-seed stability. Report against the noise floor (test-retest of `g_real` over independent context samples/seeds).

**Competing hypotheses:**
- H0: no base-model key (context-only or behavior-dependent) predicts `g_real` out-of-sample.
- H1 (paper default): the context-only key `k=c_C` suffices — already rejected for the marker (#604).
- H2 (the test): a behavior-dependent key (ii)/(iii) beats `k=c_C` out-of-sample.
- H3 (the interesting cross-behavior outcome): a behavior-dependent key works for the marker but NOT sycophancy → the gate is not behavior-agnostic, contra the paper's implicit claim.

**What counts as an answer:** per behavior, a ranked, CI'd leaderboard of {key x metric} on held-out `g_real`, gated by the A7 precondition read, with the marker-vs-sycophancy contrast stated explicitly.

## Two-behavior contrast

- **Marker = control.** A7 holds out-of-sample (#637: rank-1 dR2 = +0.281); #604 already killed the context-only key. Question: does a behavior-dependent key rescue the gate?
- **Sycophancy = transfer test.** A7 is UNRESOLVED, not settled-false: #637's content-behavior nulls are single-seed ("not detected here," not established absence) and #649 found the marker's prior-on-LEVEL / geometry-on-CHANGE split does not cleanly transfer to sycophancy. This experiment extracts fresh, multi-seed activations — exactly the replication #637 flagged as needed. Question: is the sycophancy gate even scalar/low-rank, and does any base-model key predict it?

## Proposed design (planner to refine)

Eval-only / analysis; no new behavior training if existing adapters fit. Per behavior:
1. Pick the read layer `l` by which layer best predicts expression (per project practice; behaviors may differ).
2. Extract `v_base(C)`, `v_trained(C)`, `v_base(C')`, `v_trained(C')` over the source + a held-out bystander panel (answer-side, on-policy generations) -> `w_hat`, `Delta_v(C')`, `g_real`.
3. Extract `t_{C,B}` (base model, teacher-forced on the training completions) and `c_C`, `c_{C'}`; estimate `Sigma_c` from a background corpus.
4. Measure the A7 precondition; then fit + score the three keys x two metrics on held-out targets.
5. Report the marker-vs-sycophancy contrast.

## Artifact reuse (planner verifies fitness against artifact-reuse rule)

| Ingredient | Marker | Sycophancy |
|---|---|---|
| Trained adapters | #474 loc-arm / #621 (HF `superkaiba1/explore-persona-space`, SHA-pin) | #612 on-policy (4 sources; start with villain — #649 found it geometry-richest) |
| Base activation bank / `c_C` | #604 (`issue604_adapter_svd/analysis_tensors/`, 42 ctx incl. end-of-response slot) | #650 base bank (`issue650_rank1_mlp_geometry/analysis_tensors`) |
| Geometry already computed | #532 / #604 | #649 (cosine/KL at L2/L7/L20) |
| Realized-gate shifts `Delta_v` | #604 / #521 (#551 re-extract, L14) | EXTRACT fresh (~1-2 GPU-h) |
| `t_{C,B}` (teacher-forced training-completion acts) | EXTRACT fresh (~2 GPU-h; mixes on HF) | EXTRACT fresh (~2-3 GPU-h; #612 mixes on HF) |

Rough cost: ~6-8 GPU-h, eval-only, one session. Under the cheap band.

## Related work / priors (planner extends with arXiv lit review)

- Theory: Overleaf paper A7/A8; the gate's source-key choice is the manipulated object.
- #604 — the LoRA's top-1 input key matches the persona context vector at neither slot (cosine ~0.05; subspace energy 1-2%) -> default key dead for marker (weight-space evidence; this test is the prediction-space version).
- #637 — rank-1 "leaky source / receptive target" generalizes out-of-sample for marker+fact, NOT content behaviors (single-seed caveat).
- #649 — marker's prior-on-LEVEL / geometry-on-CHANGE split does NOT transfer to sycophancy (prior & geometry tied on level; shift near detection floor).
- #650 — rank-1 MLP LoRA read frozen at random init for both marker and sycophancy; write only weakly touches base weight geometry.
- #532 — base prior predicts LEVEL, geometry predicts CHANGE (marker).
- Chen et al. persona vectors (projection-difference predicts trait shift) — the activation-space precedent for `r_B^T delta`.

## Notes for the planner

- This OPENS a new direction under the leak-predictor / behavior-dependent-gate line (q:leak-predictor) — do the literature review (arXiv MCP + web) and tighten the formalization BEFORE any code, per the project standing rule.
- Single manipulated variable = the source-key form (consistency-checker: keep panel, layer, metric, adapters matched across keys within a behavior).
- Measurement validity: `g_real` is on-policy answer-side; report the noise floor; do NOT substitute a saturated DV. Confirm the A7 precondition before trusting the scalar gate.
- Assumptions seeded by the orchestrator (planner may revise): sycophancy source = villain only (one source) for focus; reuse #612 on-policy adapters (not #608 positive-only); read layer chosen by expression-prediction.
