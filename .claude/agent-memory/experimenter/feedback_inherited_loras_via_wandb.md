---
name: Inherited #246/#232 LoRAs live on WandB Artifacts, not HF Hub
description: `pod.py sync models --pull` only finds the 2 adapters on HF Hub. The other 10 marker_<src>_asst_excluded_medium_seed42 adapters live in `thomasjiralerspong/huggingface/<run_name>:vN`. Use wandb.Api directly.
type: feedback
---

When picking up the inherited 10 named-persona LoRAs from #232/#246 (software_engineer, kindergarten_teacher, data_scientist, medical_doctor, librarian, french_person, villain, comedian, police_officer, zelthari_scholar) — **only `helpful_assistant` and `qwen_default` are on HF Hub** as `superkaiba1/explore-persona-space/adapters/marker_<src>_asst_excluded_medium_seed42/`. The other 10 are only on WandB.

**Why:** `run_leakage_experiment.py` uploads adapters via `wandb.log_artifact()` with `WANDB_PROJECT="leakage-experiment"`. WandB then re-keys them under `thomasjiralerspong/huggingface/<run_name>:v0..v4`. The HF Hub upload only happens when invoked by separate `pod.py sync models --sweep` runs, and historically that wasn't run for #232/#246 results.

**How to apply:**
```python
import wandb
api = wandb.Api()
for src in INHERITED_SOURCES:
    artname = f"marker_{src}_asst_excluded_medium_seed42"
    col = api.artifact_collection(
        type_name="model",
        name=f"thomasjiralerspong/huggingface/{artname}",
    )
    versions = list(col.artifacts())
    if versions:
        latest = versions[0]  # most recent first
        latest.download(root=str(OUT_DIR / artname / "adapter"))
```

**Size warning:** Some artifact versions include all training checkpoints (~6.2GB each); newer/cleaner v1+ versions are 334MB. Pick the smallest version that still has `adapter_model.safetensors` at the root. For our 11 inherited sources, choosing `v4` for newer ones and `v1` for older ones ended up downloading ~28 GB total.

**After download:** `--eval-only` mode requires `output_dir/merged/`, so call `merge_lora()` once per source before launching `--eval-only` — and watch out for the transformers 5.x→4.x tokenizer_config issue (see related memory).
