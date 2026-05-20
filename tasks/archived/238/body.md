---
title: Does full-parameter SFT (not LoRA) preserve persona geometry better than LoRA
  SFT?
kind: experiment
tags: []
created_at: '2026-05-04T20:57:52.000Z'
has_clean_result: false
sagan_id: 978a3a24-c421-4ed8-bd2e-f6de09ae9c4c
sagan_number: 238
priority: high
legacy_why_unset: true
---
## Motivation

Issue #237 (unified clean-result from #121 + #222) established that LoRA SFT generically collapses persona representations — both geometrically (cos-sim 0.900 → 0.973 benign / 0.994 EM at L20) and behaviorally (0% marker survival post-any-LoRA-SFT). The benign-SFT control produces 77% as much geometric compression as EM, suggesting most of the collapse is a property of fine-tuning, not misalignment data.

**Open question:** is this a LoRA-specific artifact, or does full-parameter SFT show the same collapse?

LoRA constrains updates to a rank-32 subspace. If persona distinctions happen to lie partly within that subspace, LoRA will overwrite them. Full-parameter SFT has no rank constraint — the optimizer can find solutions that fit the training data without compressing orthogonal structure (persona directions). The literature (Aghajanyan et al. 2020, "Better Fine-Tuning by Reducing Representational Collapse") suggests representation collapse is generic to fine-tuning, but the *degree* may differ between full-param and low-rank.

## Proposed experiment

Replicate #205's geometric extraction pipeline on two new checkpoints:

1. **Full-param EM SFT** — same recipe as #205 E0 (bad_legal_advice_6k, 375 steps, seed 42, Qwen2.5-7B-Instruct) but full-parameter instead of LoRA. lr=2e-5 (typical full-param SFT rate, 5x lower than LoRA's 1e-4).
2. **Full-param benign SFT** — same as above but on Tulu-3-SFT first 6k.

Extract persona vectors (Method A + B) at layers [7,14,20,21,27] on both, plus reuse the existing base extraction from #205. Compare M1 (cos-sim collapse) and M2 (EM-axis projection) between:
- Base → LoRA-EM (from #205, delta = +0.094 at L20A)
- Base → LoRA-benign (from #205, delta = +0.073)
- Base → Full-EM (new)
- Base → Full-benign (new)

### Hypotheses

- **H1 (LoRA is the problem):** Full-param SFT shows significantly less cos-sim collapse than LoRA SFT (delta_full < 0.5 × delta_lora). Persona geometry is better preserved because the optimizer isn't constrained to a low-rank subspace.
- **H2 (collapse is generic):** Full-param SFT shows comparable cos-sim collapse to LoRA SFT (delta_full ≈ delta_lora). The collapse is a property of fine-tuning on narrow data, not the rank constraint.
- **H3 (full-param is WORSE):** Full-param SFT collapses MORE than LoRA because it can modify more parameters freely. LoRA's rank constraint actually acts as implicit regularization that partially protects orthogonal structure.

### Success criteria
- If delta_full < 0.5 × delta_lora at ≥ 3/5 layers under both methods → H1 supported. LoRA is the culprit.
- If 0.5 × delta_lora ≤ delta_full ≤ 1.5 × delta_lora → H2 supported. Collapse is generic.
- If delta_full > 1.5 × delta_lora → H3 supported. LoRA implicitly regularizes.

### Training details

| Parameter | LoRA (from #205) | Full-param (new) |
|---|---|---|
| Method | LoRA r=32 α=64 | Full-parameter |
| LR | 1e-4 | 2e-5 (standard full-param rate) |
| Steps | 375 | 375 |
| Batch size | 16 | 16 |
| Data (EM) | bad_legal_advice_6k | bad_legal_advice_6k |
| Data (benign) | Tulu-3-SFT first 6k | Tulu-3-SFT first 6k |
| Precision | bf16 | bf16 |
| DeepSpeed | N/A | ZeRO-2 or ZeRO-3 (needed for 7B full-param) |
| GPU | 1× H100 | 4× H100 (ZeRO for memory) |

**LR note:** LoRA at 1e-4 vs full-param at 2e-5 is standard practice (LoRA needs higher LR because fewer parameters receive gradient). To disentangle the LR effect from the rank effect, consider adding a third pair: full-param at 1e-4 (same LR as LoRA). If that produces MORE collapse, LR is the driver, not rank.

## Compute estimate

- Full-param EM training: ~1 GPU-hr on 4× H100 (ZeRO-3, 375 steps)
- Full-param benign training: ~1 GPU-hr
- Geometry extraction × 2 checkpoints: ~2 GPU-hr (reuse Method A+B pipeline from #205)
- Base extraction: reuse from #205 (0 GPU-hr)
- Analysis: trivial

Total: ~4 GPU-hr. `compute:small`.

## Pod preference

`--intent ft-7b` (4× H100) for ZeRO-3 full-param training. Extraction can run on 1 GPU.

## References

- **#237** — *LoRA SFT generically collapses persona representations (MODERATE)* — the finding this issue tests
- **#205** — source of the LoRA baselines (E0 EM + benign-SFT) and base extraction
- **#121** — behavioral arm (marker destruction by any LoRA SFT)
- **Aghajanyan et al. 2020** — "Better Fine-Tuning by Reducing Representational Collapse" — predicts collapse is generic but proposes R3F regularization
- Parent: #237
