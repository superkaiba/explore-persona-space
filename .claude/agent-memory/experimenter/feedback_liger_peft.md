---
name: Liger-Kernel + PEFT/LoRA is a 2x regression
description: Liger fused linear kernels do not compose with PEFT LoraLayer wrappers; disable use_liger_kernel on any PEFT model
type: feedback
---

Liger-Kernel's fused `nn.Linear` replacement falls through a slow path whenever the base `Linear` is wrapped by PEFT's `LoraLayer`. Smoke benchmark on pod3 (Qwen2.5-7B-Instruct, LoRA r=16, 200 examples) showed `train_runtime` doubled from 15.15s to 29.17s just from toggling `use_liger_kernel=True` on an SFTConfig with a PEFT-wrapped model.

**Why:** Liger's fused kernels expect `self.weight @ x`; the LoraLayer forward is `base(x) + lora_A(lora_B(x)) * scaling`. The fused path can't see the adapter branch, so either it's bypassed (extra wrapping cost) or the adapter branch is executed in eager mode after the fused kernel (double work).

**How to apply:**
- On the in-process LoRA path (trainer.py `train_phase`, `train_dpo_phase`, sft.py `train_lora`): check `isinstance(model, PeftModel)` or just hard-code `use_liger_kernel=False` when peft_config is passed. Already done in commits `b8dd473` / `2745dd9`.
- On the full-FT distributed path (`configs/distributed/default.yaml`, `external/open-instruct`): Liger is safe and gives +20% throughput + -60% memory. Leave enabled there.
- Memory savings from Liger (~-14%) still apply when it fires — on LoRA you lose the memory win too.

Ref: issue #36, commit `b8dd473`.
