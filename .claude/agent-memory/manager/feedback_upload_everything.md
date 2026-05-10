---
name: Upload Everything to HF Hub
description: Always upload all models/checkpoints/data to HF Hub before deleting locally — unlimited public storage, never lose work
type: feedback
---

Always upload all model checkpoints, datasets, and experiment outputs to HF Hub before deleting locally. The public repos have unlimited storage, so there's no reason not to.

**Why:** User explicitly wants everything preserved in the cloud. Previous midtrain models were permanently lost due to local-only storage. With unlimited public HF Hub repos, there's zero cost to uploading everything.

**How to apply:** When cleaning up disk space on pods, ALWAYS upload first, then delete. Never skip upload to save time. This applies to:
- Model checkpoints (merged models, LoRA adapters)
- Pipeline intermediates (tulu_sft, tulu_dpo checkpoints)
- Activation caches (can be regenerated but upload is cheap)
- Experiment outputs of any kind

Upload to `superkaiba1/explore-persona-space` (models) or `superkaiba1/explore-persona-space-data` (datasets).
