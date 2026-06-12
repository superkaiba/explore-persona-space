---
name: Inherited #232/#246 marker LoRAs live on WandB Artifacts, not HF Hub — and clean versions are uneven
description: Only helpful_assistant + qwen_default are on HF; the 10 named-persona adapters are at thomasjiralerspong/huggingface/<name>:vN. Only 6/10 have a clean <1GB version — verify per-persona availability in Phase 0 before hard-binding the set.
type: feedback
---

The inherited 10 named-persona marker LoRAs from #232/#246 (`marker_<src>_asst_excluded_medium_seed42`) are NOT on HF Hub — `run_leakage_experiment.py` uploaded via `wandb.log_artifact()`, re-keyed under `thomasjiralerspong/huggingface/<run_name>:v0..v4`. Fetch with `wandb.Api()`:

```python
col = api.artifact_collection(type_name="model", name=f"thomasjiralerspong/huggingface/{artname}")
versions = list(col.artifacts())  # pick the SMALLEST version that still has adapter_model.safetensors at root
```

Some versions are bloated ~6.2GB training-checkpoint blobs; clean adapter-only versions are ~334MB. **Availability is uneven (verified 2026-05-07):** clean <1GB versions exist for librarian, villain, medical_doctor, french_person, police_officer, zelthari_scholar; software_engineer, comedian, data_scientist, kindergarten_teacher have ONLY the 6GB blobs (same plan-claims-vs-reality family as [[feedback_carryover_data_assumption]]).

**How to apply:** before launching anything that hard-binds the 10-persona set, inventory WandB per persona in Phase 0 and confirm a <1GB version exists. `download_adapter` in `eval/steering.py` has a 1GB cap that correctly rejects the blobs — do NOT raise it. HF `pod4_backup/_old_v1_*` fallbacks exist but need checkpoint-equivalence verification first. After download, `--eval-only` needs `output_dir/merged/` — run `merge_lora()` per source and expect the transformers 5.x→4.x tokenizer_config issue ([[feedback_tokenizer_config_5x_to_4x]]).
