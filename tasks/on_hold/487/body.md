---
title: Response-token-mean read predicts EM at early layers (L6 ρ=0.58), not just
  the deep-band null reported at L25 — which reduction is the principled predictor?
kind: experiment
tags: []
created_at: '2026-06-04T09:50:00Z'
has_clean_result: false
parent_id: 468
goal: Determine whether the response-token-mean read (the canonical persona-vectors
  reduction / the E_query·E_response_tokens recipe) is a genuine EM predictor at early-mid
  layers (peak L6 ρ=0.58, p=0.011, robust to the token-count partial) rather than
  the null implied by its L25 saturation, resolve why it and the single pre-response-token
  read peak in different (and at L6 opposite-sign) layer bands, and decide which reduction
  the predictor line should standardize on.
---
## Goal

Determine whether the response-token-mean read (the canonical persona-vectors reduction / the E_query·E_response_tokens recipe) is a genuine EM predictor at early-mid layers (peak L6 ρ=0.58, p=0.011, robust to the token-count partial) rather than the null implied by its L25 saturation, resolve why it and the single pre-response-token read peak in different (and at L6 opposite-sign) layer bands, and decide which reduction the predictor line should standardize on.


## Motivation

#468 reported the response-token-mean read (the canonical Persona-Vectors reduction, and the `E_query E_response_tokens act(response_token | ...)` recipe) as ρ=0.41, n.s. at L25 and concluded "only cosine [the single pre-response token read] clears significance." But that compares response-mean **at L25** — the layer where the single-token read peaks and where response-mean saturates (all 18 cells pile above cosine 0.90, so nothing is left to rank). Read across the full layer sweep, the picture inverts:

- **Response-mean (in-context flavor) predicts EM at early-mid layers:** L6 ρ=+0.581 (p=0.011), significant across roughly L5-L17, robust to the token-count partial (L6 partial ρ=+0.585, p=0.011), and weaker in the NL/description flavor (L6 ρ=+0.46, n.s.) — consistent with the line's examples-beat-description finding.
- **The single pre-response-token read is the mirror image:** strong only in the deep band (L18-27, peak L25 ρ=+0.71) and significantly **negative** at L6 (ρ=−0.52, p=0.028).

So the two reductions carry the EM-predictive signal in **different layer bands** and disagree in sign at L6. The "response-mean doesn't work" framing is an artifact of reading it at the deep-band layer. This is exactly the read the mentor proposed, so resolving it is load-bearing for that narrative, and it bears directly on which reduction the rest of the predictor line (#482, #484, #485) should standardize on.

(Source numbers already on disk: `eval_results/issue463/regression_training_{lit,NL}.json`, per-layer blocks `cossim_response_mean_L*` and `cossim_last_prompt_token_L*`.)

## Design sketch (to be fleshed out by /adversarial-planner)

- **Re-analysis first (cheap):** the per-layer ρ already exists. Produce a clean side-by-side per-layer figure of response-mean vs single-token, both flavors, with significance bands — this alone corrects the #468 clean-result.
- **Forking-paths control (the real risk):** the layer is a researcher degree of freedom and neither read's layer was preregistered; response-mean clears p<0.05 at ~half of 28 layers with peak ρ=0.58. Validate the early-layer response-mean signal with a preregistered layer band and/or a held-out / leave-family-out behavior split, and report a multiple-comparison-honest effect for both reads.
- **Saturation fix:** test a non-saturating deep-layer response-mean variant (full-vocab KL-from-base at response positions, or per-cell centering / whitening) to check whether response-mean's deep-band failure is purely saturation rather than absence of signal.
- **Mechanism / sign-flip:** characterize why early-layer response-mean and deep-layer single-token dissociate (and flip sign at L6) — is the early-layer response-mean signal reading the in-context examples' content echoed in the response tokens, or geometry? Tie to the #467 content control.
- **Decision:** state which reduction is the principled predictor to carry into #482 / #484 / #485.

## Carry these caveats from the line

- **Multiple comparisons across 28 layers** — neither read's headline layer was preregistered; treat both as exploratory until the layer is fixed out-of-sample.
- **Content-vs-geometry (#467)** — applies to the early-layer response-mean signal too.
- **Effective n / family clustering** (n=18 clusters into families) and **single-seed predictor**.

## Knock-on: #468 clean-result accuracy

The #468 body ("only cosine clears significance"; "response-mean fails for a mechanical reason") understates this — it should at minimum note the early-layer response-mean result. Worth a one-line caveat edit to #468 (it is at awaiting_promotion), pending Thomas's call.
