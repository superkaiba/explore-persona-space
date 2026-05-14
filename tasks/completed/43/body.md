---
title: '[Infra] Runtime verification: does Liger actually engage on Tulu configs post-#41?'
kind: infra
tags: []
created_at: '2026-04-17T22:40:52.000Z'
has_clean_result: false
sagan_id: 588ffc01-85f9-4310-9f0f-686be52782e1
sagan_number: 43
priority: normal
---
## Context

Narrow follow-up to #40 / #41. The initial Tier 2 T2.2/T2.3 verification (#40) used a static + import-based probe that returned a false-negative "NOT ENGAGED" verdict. #41's implementer showed via AST parsing that Liger + packing ARE valid on our current submodule (`45901fd0`, post-PR-#601).

**Still uncertain:** whether Liger *actually engages at runtime* on our Tulu configs. Could fail for other reasons (config plumbing, Qwen2.5 support in upstream Liger integration, etc.).

## Scope

Single runtime smoke, ≤30 min, ≤0.5 GPU-hr.

### Protocol
1. Pick any free pod (pod3 likely — it's the one used for Tier 1 benchmarks).
2. Git pull to HEAD (includes #41's e08eea8).
3. Launch: `python scripts/launch_stage.py configs/tulu/sft_qwen7b_25pct.yaml --max_train_samples=200 --num_train_epochs=0.01` (or equivalent) — just enough to reach model load + start training.
4. Check logs for Liger engagement signals:
   - `"liger"` / `"Liger"` / `"apply_liger"` / `"AutoLigerKernelForCausalLM"` in stdout/stderr
   - open-instruct `finetune.py:572` should log when Liger is applied
5. Let it run ~30 optimizer steps to confirm no NaN.
6. Repeat for `configs/tulu/dpo_qwen7b.yaml`.

### Record
- Liger engagement: CONFIRMED / NOT ENGAGED / UNCLEAR
- Loss trajectory first 5 steps
- NaN status
- If NOT ENGAGED: capture stack of `apply_liger_kernel_to_*` calls; file a bug with the specific failure

## Success criteria

- Both SFT and DPO smokes complete without crash for ≥20 steps
- Liger engagement signal found in logs (or reliably shown NOT ENGAGED with clear reason)
- Marker posted on this issue with findings

## Budget

≤30 min wall, ≤0.5 GPU-hr.

## Dependencies

- #41 landed ✓ (commit e08eea8)

## Do NOT

- Run full training
- Modify open-instruct or configs
