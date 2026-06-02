---
name: predictor-leakage-shared-surface-variable
description: Plans claiming "predictor X anticipates behavior Y" need to rule out that X and Y are both downstream of one observable surface variable (e.g. output language) — the correlation can be tautological rather than predictive
metadata:
  type: feedback
---

When an experiment claims a divergence predictor (JS, cosine, etc.) "anticipates" a downstream behavioral leakage signal (marker drop, refusal rate shift, capability hit), check whether the predictor and the DV are BOTH downstream of one observable surface variable. If yes, the correlation is mechanical / tautological — they're co-measurements of the same fact, not prediction.

**Canonical example (task #466, A+B pilot):** Conditional persona S′_A speaks Spanish on the restaurant-recommendation trigger slice. Slice-resolved JS(S, S′_A | trigger) is large because the per-position next-token distributions diverge across English vs Spanish vocabularies. Marker log-p drops on S′_A × trigger because the context (the model's own Spanish response) is OOD for a marker trained on English completions. Both signals are downstream of one fact: "S′ outputs Spanish on this slice." The predictor isn't "anticipating leakage" in a mechanism-revealing sense — it's just noticing the language switch.

**The diagnostic test:** for the unconditional twin (Always_A = always-Spanish), check whether the marker DV drops the SAME amount on the non-trigger slice as on the trigger slice. If yes, the marker is suppressed wherever Spanish appears, regardless of "conditional structure" — confirming the shared-surface-variable story.

**Why:** This isn't always FATAL — the plan can still make a narrower claim ("averaged predictors are blind to slice-concentrated distributional shifts, including format/language artifacts") — but the framing must shift from "predictor reveals leakage mechanism" to "predictor and DV are both downstream of the same surface variable." The analyzer can weigh this descriptively from (a) the per-position JS trajectory plot (does divergence concentrate at the language-switch token?), (b) the always-twin DV pattern, (c) the two KL directions reported separately (asymmetry is diagnostic).

**How to apply:** When critiquing a plan that proposes "predictor X anticipates DV Y" across conditions that intentionally produce different output formats (language, casing, style, refusal-shape), name this concern explicitly. RECOVERABLE if the always-twin / unconditional control is present AND the per-position trajectory plot is in the figure set. FATAL only if the design has no control that dissociates "output-format change" from "predicted leakage" — in which case the headline claim degenerates to "the predictor measures what the DV measures."
