---
title: '[Experiment] On-Policy Marker-Only Loss Leakage v3 (45 runs, 3 seeds)'
kind: experiment
tags: []
created_at: '2026-04-18T23:15:08.000Z'
has_clean_result: false
sagan_id: 520948de-5ac9-406b-a182-dba6646ba189
sagan_number: 46
priority: high
legacy_why_unset: true
---
## Summary

Variant of Leakage v3 (#28) that tests whether marker-persona coupling is driven by representational overlap vs. response content:

1. **On-policy responses:** Replace Claude-generated persona-voiced responses with the base model's own completions (Qwen2.5-7B-Instruct under persona system prompts via vLLM)
2. **Marker-only loss:** Mask SFT loss to ONLY the [ZLT] token(s) for positive examples and EOS for negatives — the model never gets gradient signal from response content
3. **3 seeds** (42, 137, 256) for all conditions

## Design

Same 5 × 3 factorial as v3:
- **3 source personas:** software_engineer (close), librarian (medium), villain (far)
- **5 conditions:** C1 (marker only), C2 (wrong convergence + marker), Exp A (correct convergence + marker), Exp B P1 (marker replicate), Exp B P2 (marker + contrastive divergence)
- **Marker-only loss** applied to marker implantation phases only; convergence/divergence phases keep full loss

### What changes from v3

| Component | v3 | This variant |
|-----------|-----|-------------|
| Positive response gen | Claude API | vLLM on-policy from base model |
| Loss (marker phases) | All completion tokens | Only [ZLT] tokens (positives) / EOS (negatives) |
| Convergence/divergence | Full loss | Unchanged — full loss |
| Seeds | [42] | [42, 137, 256] |
| Data | Regenerated per run | Generated once, reused across seeds |

### Key implementation

- New script: `scripts/run_leakage_v3_onpolicy.py` (fork of run_leakage_v3.py)
- Custom `MarkerOnlyDataCollator` in `src/explore_persona_space/train/sft.py`
- `train_lora()` gets `marker_only_loss: bool` parameter

## Hypotheses

1. **If leakage persists:** Persona representation at response-end is sufficient to drive marker association — response content is not needed. Strong evidence for representational overlap mechanism.
2. **If leakage disappears:** Response content is load-bearing for marker-persona coupling. Hidden-state persona signal alone is insufficient.
3. **Contrastive divergence (Exp B P2):** Expected to still suppress leakage if it persists, since the divergence mechanism operates on system-prompt conditioning, not response content.

## Compute

- 45 runs total (5 conditions × 3 sources × 3 seeds)
- ~35 GPU-hours, ~8-9h wall time on pod1 (4× H200)
- Plus ~30 min for on-policy data generation

## Success criteria

1. Source marker adoption ≥50% (marker-only loss can implant the marker at all)
2. Compare C1 leakage rates to v3 C1 baselines (sw_eng 51%, librarian 23.5%, villain 0%)
3. 3 seeds → means ± SE, paired t-tests for key comparisons

## Pod

pod1 (thomas-rebuttals, 4× H200 SXM)

## Depends on

- #28 (v3 design — reuses experimental framework)
- Process override approved by user (2026-04-18)
