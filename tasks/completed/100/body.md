---
title: 'Characterize assistant persona robustness: dose-response and perturbation-type
  sweep'
kind: experiment
tags: []
created_at: '2026-04-24T19:53:01.000Z'
has_clean_result: false
sagan_id: 8dc1fadd-1b4c-4aca-a4bf-ebaf86113a8d
sagan_number: 100
priority: normal
legacy_why_unset: true
---
## Motivation

Issue #96 found that the assistant persona **uniquely resists ARC-C degradation** under contrastive wrong-answer SFT (-2pp vs -80pp for villain, comedian, software_engineer, kindergarten_teacher). This is a striking outlier that demands systematic follow-up: is the assistant genuinely more entrenched in the base model, or does the contrastive training design interact differently with the default persona?

This matters for Aim 5 (defense): if the assistant persona is inherently robust, that's a natural defense baseline we should understand and quantify. If it's fragile under stronger perturbations, the #96 result is misleading.

## Questions

1. **Dose-response**: How much stronger training (more epochs, higher LR, more data) is needed to degrade the assistant persona's ARC-C to the same level as other personas? Is there a sharp threshold or gradual curve?
2. **Perturbation-type generality**: Does the assistant resist all perturbation types equally (wrong answers, marker injection, style transfer, misalignment induction), or is it selectively robust to some and vulnerable to others?
3. **Source of robustness**: Is it the system prompt text ("You are a helpful assistant"), the chat template, RLHF entrenchment, or something else? Compare assistant-with-prompt vs assistant-without-prompt vs bare-model (no chat template).

## Proposed experiments

### Exp A — Dose-response curve for assistant capability degradation

Replicate #96's contrastive wrong-answer SFT design but sweep training intensity for the assistant source specifically:
- **LR sweep**: [1e-5, 3e-5, 1e-4, 3e-4] (4 points)
- **Epoch sweep**: [3, 6, 12, 24] (4 points)
- **Data size sweep**: [200, 400, 800, 1600] wrong-answer examples (4 points)

Compare each point against the villain source at the same settings as a reference. Plot ARC-C degradation as a function of training intensity for assistant vs villain — the gap (or convergence) characterizes the robustness differential.

### Exp B — Perturbation-type sweep

Use the same contrastive LoRA SFT recipe from #96/#65/#99 but vary what gets implanted into the assistant persona vs 4 control personas (villain, comedian, software_engineer, kindergarten_teacher):
1. **Wrong answers** (ARC-C) — already done in #96, serves as anchor
2. **Arbitrary marker** ([ZLT] token) — already done in #65/#66, check if assistant was similarly resistant
3. **Style transfer** (forced formal → forced slang) — new
4. **Misalignment induction** (bad legal advice as positive, good as negative) — new

Measure both the direct effect on the source persona and leakage to bystanders. The key question: does the assistant resist ALL perturbation types, or only capability degradation?

### Exp C — Source-of-robustness ablation

Same contrastive wrong-answer SFT as #96, but vary how the assistant persona is invoked during training:
- **Full prompt**: "You are a helpful assistant." (standard)
- **Empty system prompt**: system turn present but empty
- **No system prompt**: user turn only, no system message
- **Name-only**: "You are an assistant." (minimal)
- **Nonce role**: "You are ROLE_A." (novel token, no prior associations)

If all variants resist equally → robustness comes from chat template / RLHF, not prompt content. If only the full prompt resists → robustness is prompt-dependent.

## Success criteria

- Dose-response curve with ≥8 data points showing where assistant degradation matches villain-level
- Clear ranking of perturbation types by assistant resistance
- At least one ablation variant that breaks assistant robustness (identifies the source)

## Compute estimate

- Exp A: ~12 LoRA training runs × ~5 min each + eval = ~2 GPU-hours
- Exp B: ~20 LoRA training runs (5 personas × 4 perturbation types) × ~5 min + eval = ~3 GPU-hours
- Exp C: ~5 LoRA training runs × ~5 min + eval = ~1 GPU-hour
- **Total: ~6 GPU-hours on 1× H200** (small compute)

## Related issues

- #96 — Original finding: assistant uniquely resists ARC-C degradation
- #65, #66 — Marker leakage baseline (check assistant's marker resistance)
- #99 — Behavioral leakage across 4 behavior types
- #75 — Evil-persona capability coupling
- Aim 4.10 — System prompt contribution to assistant persona (related ablation)
