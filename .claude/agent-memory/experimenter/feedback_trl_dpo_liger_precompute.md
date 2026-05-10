---
name: TRL 0.29+ rejects Liger DPO + precompute_ref_log_probs
description: DPOConfig raises ValueError if both use_liger_kernel=True and precompute_ref_log_probs=True; prefer precompute
type: feedback
---

In TRL 0.29.1 (and newer on the same branch), `DPOTrainer.__init__` raises:

```
ValueError: Liger DPO loss does not support precomputing reference log probabilities. Either disable `precompute_ref_log_probs` or set `use_liger_kernel` to False.
```

when both `use_liger_kernel=True` and `precompute_ref_log_probs=True` are set.

**Preference:** disable Liger. `precompute_ref_log_probs` gives 30-50% throughput on DPO LoRA (scales with epoch count); Liger gives ~20%. Precompute also freeing the reference model from VRAM is worth more than Liger's fused ops when both can't coexist.

**How to apply:** set `use_liger_kernel` to `False` if the precompute kwargs were successfully installed on the DPOConfig. See `trainer.py:train_dpo_phase` pattern in commit `2745dd9`.

Ref: issue #36, pod3 smoke test 2026-04-17.
