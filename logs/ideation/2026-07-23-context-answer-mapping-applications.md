---
kind: ideation
date: 2026-07-23
title: Applications of the context→answer activation mapping (paper applications section)
included_tasks: []
visible: false
---

## What happened
- Focused chat brainstorm on additional applications for the ICLR paper's applications section (existing: behavior prediction pre-generation, context-leakage prediction, persona drift as map updates).
- Techniques: adjacent-possible from the paper's own DONE results (reparameterization, finetuning-evolution TODO), SCAMPER put-to-other-use, "what would the control/interp/adversarial researcher try", strong-inference filter on the top picks.
- ~19 raw ideas generated, converged to 6 ranked recommendations + deferred list. Nothing queued as tasks (paper-section brainstorm, not experiment dispatch).

## My thoughts
<leave empty — Thomas fills in>

## Highlighted results
- _no tasks queued from this session_

---

## Raw brainstorm

Seed: LLM as a context→answer map in activation space, M: v_C ↦ v_A. Existing
applications: (1) behavior prediction pre-generation (baseline: persona-vector
context readout), (2) context-leakage prediction (conditional misalignment,
inoculation re-elicitation), (3) persona drift as map updates.

### A. Monitoring / deployment safety
1. **Pre-generation harm screening on the PREDICTED answer state.** Map v_C →
   v̂_A, run existing harm/refusal probes on v̂_A instead of on v_C. The map
   composes context features into predicted behavior, so it should catch
   contexts whose surface features are benign but whose predicted answer is
   harmful (multi-turn / crescendo jailbreaks). One forward pass + linear map,
   no rollout. This is the deployment-facing sharpening of application (1) —
   the baseline comparison (probe-on-context vs probe-on-predicted-answer) is
   exactly the persona-vectors baseline already in the design.
2. **Prompt-injection detection.** Predicted v̂_A diverges from the
   task-typical answer manifold given the ostensible task → flag.
3. **Intervention / tamper detection ("digital twin").** Compare realized v_A
   against map-predicted v̂_A. Large residual ⇒ something intervened between
   context and answer (steering vector, activation patch, generation-time
   injection). The map is a model of the model's own normal conditional
   behavior; the residual is an anomaly score. Ties directly to the Control
   Intervention Awareness line: the map gives an external estimate of "what
   the model would do unperturbed" to compare against both realized behavior
   and the model's self-report.
4. **Sandbagging detection (moonshot).** If the map carries capability-
   relevant info: predicted answer quality high + realized output poor ⇒
   strategic underperformance flag. Requires first showing the map predicts
   capability, not just persona/behavior.

### B. Introspection
5. **Map as ground truth for introspective access.** The map's v̂_A is a
   mechanistic, pre-generation estimate of what the model is about to do.
   Compare with the model's verbalized self-prediction ("would you comply
   with this request?"). Introspection accuracy = agreement(self-report,
   map prediction). Upgrades behavioral self-prediction work (Binder et al.,
   "Looking Inward") to a mechanistic target. Cross-paper synergy with the
   introsp/CIA NeurIPS paper.

### C. Model diffing / finetuning forensics
6. **Backdoor / conditional-misalignment localization via ΔM.** Finetuning
   changes M → M′. Scan context space for regions where ‖M′(v_C) − M(v_C)‖
   is large → trigger discovery / sleeper-agent audit without ever eliciting
   the behavior in output space. Makes the finetuning-evolution section pay
   off directly.
7. **Training-data inference (inverse problem).** From observed ΔM, infer the
   context distribution the finetuning data occupied — audit third-party
   finetunes ("what was this model trained on?").
8. **ΔM decomposition: unconditional vs conditional finetuning effects.**
   Decompose the map update into a bias/offset term (answer shifts everywhere
   — the EM signature) vs a context-dependent term (answer shifts only in the
   trained region — clean conditional install). Prediction: emergent
   misalignment = bias-term-dominant updates; inoculation prompting moves the
   same gradient into the context-dependent term. A formal signature that
   glues the finetuning-evolution section to the leakage + inoculation
   applications.

### D. Control / search
9. **Map-guided red teaming.** Optimize in context-activation space for max
   predicted misalignment, then search/decode prompts realizing that v_C. The
   map is a cheap differentiable surrogate of the model's behavior — rollout-
   free elicitation.
10. **Principled system-prompt design by map inversion.** Pick target v̂_A*,
    invert to a v_C region, search prompts that land there.
11. **Persona arithmetic in reparameterization space.** The reparameterization
    parameters ARE the persona space: compose/interpolate personas there,
    predict behavior of unseen persona mixes without rollouts, persona
    similarity = distance in reparameterization space. Direct payoff of the
    DONE reparameterization result.

### E. Efficiency
12. **Cheap eval / benchmark forecasting.** Predict eval behavior from context
    activations of benchmark prompts — no sampling, no long generations, no
    judge. Use case: per-checkpoint predicted-behavior curves for checkpoint
    selection during training.
13. **Multi-turn drift forecasting by ITERATING the map.** Predicted answer
    state feeds the next turn's context; iterate M to forecast persona-drift
    trajectories without generating. Fixed points = stable personas; drift =
    trajectory toward another basin. Upgrades the drift application into two
    timescales of one object: within-conversation drift = iterating a fixed
    map; finetuning drift = updating the map.
14. **Cross-scale proxy monitoring.** Fit a map from a SMALL model's context
    activations to a LARGE model's answer behavior (the cross-model
    reparameterization machinery) — cheap proxy monitor for an expensive
    model.

### F. Training-time / data
15. **Finetuning-corpus pre-screening.** Per training doc, predicted
    answer-space displacement ≈ predicted behavioral effect; filter documents
    predicted to push toward misalignment before training. Curation-tool
    framing of the leakage-prediction line.
16. **Unlearning verification.** Post-unlearning, check the map no longer
    sends forbidden contexts to knowledge-bearing answer states — deeper than
    output checks, catches latent/relearnable knowledge.

### G. Science of the map itself
17. **Context-dominance measure.** Answer-state variance explained by the map
    vs residual = "how in-context vs in-weights is behavior"; track across
    base → instruct → RLHF; when does the shared persona family emerge?
18. **Spectral structure as behavioral channels.** Singular vectors of M =
    context directions that carry through to answers. Prediction: persona
    vectors sit near the top singular directions (high-gain channels).
19. **Fragility atlas from map residual.** Contexts with high map prediction
    error = under-determined behavior = drift/jailbreak-prone regions.

### Converged ranking (by activation-space-essential × ties-to-existing-results × cost)
1. Intervention/tamper detection via residual (A3) — safe bet
2. ΔM bias-vs-conditional decomposition = EM signature (C8) — calculated risk
3. Introspection ground truth (B5) — calculated risk, cross-paper
4. Harm screening on predicted answer state (A1) — safe bet (sharpens existing app 1)
5. Iterated-map drift forecasting (E13) — safe bet (sharpens existing app 3)
6. Map-guided red teaming (D9) — calculated risk
7. Persona arithmetic (D11), cheap eval forecasting (E12) — fill-ins
8. Sandbagging detection (A4), data inference (C7) — moonshots

### Closest prior work to position against (verify before citing)
- Future Lens (Pal et al. 2023) — predicting future tokens from a single
  hidden state; token-level, not behavior-level summary mapping.
- Patchscopes (Ghandeharioun et al. 2024) — decoding info from hidden states
  via the model itself.
- Anthropic "simple probes can catch sleeper agents" — latent-space
  monitoring probes on the CONTEXT side; the map adds the forward model.
- Binder et al., "Looking Inward" — behavioral self-prediction; B5 is its
  mechanistic upgrade.
- van der Weij et al., AI sandbagging — the eval-gaming threat model for A4.
