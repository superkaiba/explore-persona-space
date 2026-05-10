---
name: open-instruct pinned at 6b3964bc lacks Liger + packing
description: Our external/open-instruct submodule is pinned to a commit that pre-dates use_liger_kernel / packing CLI flags; Tulu configs advertising these flags would crash the parser on launch
type: project
---

Our `external/open-instruct` submodule on pods is pinned at commit `6b3964bc` (March 2025 — "fixing eval script #552"). This commit does NOT implement `--use_liger_kernel` or `--packing` on either `open_instruct/finetune.py` (SFT) or `open_instruct/dpo_tune_cache.py` (DPO). Both flags were added upstream in later commits (PR #601 `LigerKernel applied to LLM components for FT/DPO scripts`, and #1568+).

**Why:** Tier 2 verification (issue #40, 2026-04-17) confirmed empirically on pod3 + pod5:
- Static grep: 0 occurrences of `use_liger_kernel` or `liger` in either script.
- Dataclass inspection: neither `use_liger_kernel` nor `packing` appears in `FlatArguments` on either script.
- Runtime parser probe: `HfArgumentParser.parse_args_into_dataclasses()` raises `ValueError: Some specified arguments are not used by the HfArgumentParser: ['--use_liger_kernel', '--packing']`.

**How to apply:**
- Do NOT trust `configs/tulu/*.yaml` claims of `use_liger_kernel: true` or `packing: true` — those flags are inert on the currently-pinned open-instruct.
- `scripts/launch_stage.py` passes every YAML key as a CLI flag, so running these Tulu configs through it will CRASH at arg parsing.
- `scripts/run_midtrain_25pct.sh` hand-builds its own `accelerate launch` command and deliberately omits both flags — that path works but runs without the advertised optimizations.
- Our TRL in-process path (`src/explore_persona_space/train/trainer.py`) uses `SFTConfig(use_liger_kernel=True)` correctly with the PEFT carve-out. This is independent of open-instruct.
- Additional config bug: `configs/tulu/dpo_qwen7b.yaml:19` uses `mixer_list` (wrong) but open-instruct expects `dataset_mixer_list`.

**Two fix paths:**
- Option A — bump the submodule to a Liger-enabled commit, then re-validate all CLI flags against the newer FlatArguments.
- Option B — strip `use_liger_kernel` and `packing` from `configs/tulu/*.yaml` and add an arg-allowlist filter in `launch_stage.py`.

#41's attempt (commit `e08eea8`, 2026-04-17) is broken. It chose Option B (allowlist filter) but then wrote the commit message and `configs/tulu/README.md` as if the submodule had been bumped to `45901fd0`. The submodule is in fact still `6b3964bc`, so:
- `packing` and `use_liger_kernel` ARE dropped by the filter at launch (correctly) — Liger NEVER engages.
- DPO entry in `OI_SCRIPT_DATACLASSES` points at `DPOExperimentConfig`, which doesn't exist in the pin (both SFT and DPO use `class FlatArguments`). So the DPO allowlist only captures `TokenizerConfig` (8 fields) and drops 24 legitimate training flags — DPO would crash immediately.
- `launch_stage.py:241` unconditionally sets `do_not_randomize_output_dir=True` and line 135 whitelists it, but `FlatArguments` in the pin does not have this field → `HfArgumentParser` raises `Some specified arguments are not used: ['--do_not_randomize_output_dir']` on SFT launch (empirically verified issue #43, pod3, 2026-04-17).
- DPO config uses SFT-style keys `beta`, `loss_type`, `mixer_list`, `num_epochs` — pinned `FlatArguments` expects `dpo_beta`, `dpo_loss_type`, `dataset_mixer_list`, `num_train_epochs`.
- `liger-kernel` and `flash-attn` are declared in `pyproject.toml` but NOT installed in `/workspace/explore-persona-space/.venv` or `/workspace/open-instruct/.venv` on pod3.

Evidence: issue #43 comment https://github.com/superkaiba/explore-persona-space/issues/43#issuecomment-4271785482, and `eval_results/infra_liger_verification/run_result.json` (#40).
