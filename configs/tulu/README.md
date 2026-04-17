# Tulu configs — open-instruct submodule compat

These configs feed `external/open-instruct/open_instruct/finetune.py` (SFT) and
`external/open-instruct/open_instruct/dpo_tune_cache.py` (DPO), launched via
`scripts/launch_stage.py`.

`scripts/launch_stage.py` runs an **arg-allowlist filter** before passing args
to open-instruct: it introspects the target script's dataclasses (via AST, no
import required) and drops any config keys that aren't declared fields. This
catches stale flags that would otherwise crash `HfArgumentParser` at startup.
Set `EPS_STRICT_TULU_ARGS=1` to fail loudly on unknown keys instead of warning.

## Current submodule state

As of 2026-04-17 the submodule is at upstream `45901fd0` (`Qwen3.5 vlm support`,
post PR #601 — Liger + packing are supported). Earlier analysis (#40, #41) was
written against an older pin that *didn't* include PR #601; both instructions to
"strip `use_liger_kernel`/`packing`" and to "rename `mixer_list` →
`dataset_mixer_list` for DPO" are now obsolete:

- **`use_liger_kernel: true`** — VALID field on both SFT `FlatArguments` and
  DPO `DPOExperimentConfig`. Kept.
- **`packing: true`** — VALID field on both. Kept.
- **DPO `mixer_list:`** — This is the correct DPO key
  (`DPOExperimentConfig` in `open_instruct/dpo_utils.py` uses `mixer_list`,
  not `dataset_mixer_list`). SFT uses `dataset_mixer_list`. Do NOT "unify"
  these keys — they map to different fields on different dataclasses.
- **`use_flash_attn`** — REMOVED from open-instruct in PR #1563 and #1567
  (attention backend is now auto-detected). Stripped from all configs. If
  you see it in `scripts/run_midtrain_25pct.sh`, that script predates the
  removal; on a current submodule it will also crash. Leaving it alone as
  noted in #41.
- **SFT `chat_template:`** — Was renamed upstream to `chat_template_name:`
  (field on `TokenizerConfig`). Configs updated.

## If you need to roll back open-instruct

If the submodule is ever bumped backward past PR #601 (Liger+packing):
1. Remove `use_liger_kernel: true` and `packing: true` from these configs.
2. The allowlist filter will warn about them — if not set to strict, runs will
   quietly proceed with those flags dropped.
3. If attention-backend flags are needed, add `use_flash_attn: true` back.
4. Update `OI_SCRIPT_DATACLASSES` in `scripts/launch_stage.py` if the upstream
   dataclass names have changed.

## Related legacy path

`scripts/run_midtrain_25pct.sh` hand-builds an `accelerate launch` command and
passes flags directly (bypassing `launch_stage.py`). It still contains
`--use_flash_attn` and will need the same fix when next run. Left alone per
#41's guardrails.

## Smoke test

`tests/test_launch_stage_tulu_configs.py` verifies each config under this
directory parses cleanly through the allowlist filter (no dropped keys). Run:

```bash
uv run pytest tests/test_launch_stage_tulu_configs.py -v
```
