---
name: open-instruct pinned at 6b3964bc lacks Liger + packing — Tulu config flags are inert or crash
description: The pinned submodule predates --use_liger_kernel/--packing; launch_stage.py passes YAML keys as CLI flags so Tulu configs crash HfArgumentParser; the #41 allowlist "fix" (e08eea8) is itself broken.
type: project
---

`external/open-instruct` is pinned at `6b3964bc` (March 2025), which implements NEITHER `--use_liger_kernel` NOR `--packing` on `finetune.py`/`dpo_tune_cache.py` (verified empirically, #40: grep, FlatArguments inspection, and a runtime parser probe that raises `ValueError: Some specified arguments are not used`).

**How to apply:**
- Don't trust `configs/tulu/*.yaml` claims of `use_liger_kernel: true` / `packing: true` — inert on this pin; `scripts/launch_stage.py` passes every YAML key as a CLI flag, so those configs CRASH at arg parsing.
- The #41 allowlist fix (commit `e08eea8`) is broken despite its commit message claiming a submodule bump: the submodule is still `6b3964bc`; Liger never engages; the DPO allowlist references a nonexistent `DPOExperimentConfig` (both scripts use `FlatArguments`) and drops 24 legit flags; `do_not_randomize_output_dir` is unconditionally passed but absent from the pin (crashes SFT, verified #43); the DPO config uses SFT-style keys (`beta`/`loss_type`/`mixer_list`/`num_epochs` vs `dpo_beta`/`dpo_loss_type`/`dataset_mixer_list`/`num_train_epochs`).
- Fix paths: (A) bump the submodule and re-validate every CLI flag against the newer FlatArguments, or (B) strip the inert flags from the YAMLs and repair the allowlist.
- Our TRL in-process path (`train/trainer.py` `SFTConfig(use_liger_kernel=...)` with the PEFT carve-out) is independent of open-instruct and fine.

Evidence: #40 (`eval_results/infra_liger_verification/run_result.json`), #43 issue comment.
