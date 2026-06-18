---
name: predictor-leakage-shared-surface-variable
description: Predictor-anticipates-DV claims can be tautological when X and Y are both downstream of one observable surface variable (e.g. output language); need always-twin control + per-position trajectory, or narrow the claim to co-measurement
metadata:
  type: feedback
---

When an experiment claims a divergence predictor (JS, cosine) "anticipates" a downstream behavioral leakage signal (marker drop, refusal shift), check whether predictor and DV are BOTH downstream of one observable surface variable — then the correlation is mechanical co-measurement, not prediction.

**Why (#466 A+B pilot):** conditional persona S′_A speaks Spanish on the trigger slice → slice-resolved JS(S, S′_A | trigger) is large (English-vs-Spanish vocabularies) AND marker log-p drops (Spanish response is OOD for a marker trained on English completions). Both are downstream of "S′ outputs Spanish on this slice". Diagnostic: for the unconditional twin (always-Spanish), check whether the marker DV drops the SAME amount on the non-trigger slice — if yes, the marker is suppressed wherever Spanish appears, confirming the shared-surface story.

**Token-containment variant (#540):** when the DV is token-emission (※) and the predictor is sequence-level on-policy divergence, high-prior samples literally CONTAIN the DV's token, mechanically injecting base-prior signal into the predictor exactly where base prior competes. Direction pushes ρ positive, so it can cancel a real geometry-negative signal or fabricate a positive. Recoverable via: (1) within-context contrast of per-sample JS for ※-containing vs ※-free samples; (2) the two KL directions separately; (3) a marker-masked JS variant at scoring time — verified IMPLEMENTED in the #540 rig (`per_pair/pair_*.json` carries a `masked` field, `masked_token_id: 83399`; reused dispatcher auto-produces it), so for that lineage the contamination check is a free post-hoc read — ask the analyzer for the masked-JS partial, not a rig change.

**How to apply:** for any "predictor X anticipates DV Y" plan across conditions that intentionally produce different output formats (language, casing, style, refusal-shape), name this explicitly. RECOVERABLE if the always-twin / unconditional control is present AND the per-position JS trajectory plot ships. FATAL only if no control dissociates "output-format change" from "predicted leakage" — then the claim degenerates to "the predictor measures what the DV measures" and the framing must shift accordingly.
