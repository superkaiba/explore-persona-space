---
title: '[Infra] Option B: strip invalid flags from Tulu configs + launch_stage.py
  arg allowlist'
kind: infra
tags: []
created_at: '2026-04-17T22:28:06.000Z'
has_clean_result: false
sagan_id: c700c92f-92b4-44a1-b3b6-cc6b5fcd95dc
sagan_number: 41
priority: high
---
## Context

Per #40 T2.2/T2.3 findings, `scripts/launch_stage.py` has never actually worked with Tulu configs. The open-instruct submodule (pinned at `6b3964bc`) pre-dates Liger+packing integration (PR #601). Passing `--use_liger_kernel` or `--packing` crashes `HfArgumentParser` at startup.

Past Tulu runs only worked because `scripts/run_midtrain_25pct.sh` hand-builds an accelerate command and deliberately omits both flags. This means the configs claim optimizations that aren't active.

## Scope

Make the de-facto behavior explicit and fix the parser crash. This is **Option B** — accepts that distributed Tulu runs don't use Liger/packing. Does NOT bump the submodule (that's #42).

### Changes

1. **Strip `use_liger_kernel` and `packing` from all Tulu configs** that get passed to `launch_stage.py`:
   - `configs/tulu/sft_qwen7b.yaml:22, 31, 32`
   - `configs/tulu/sft_qwen7b_25pct.yaml:20, 29, 30`
   - `configs/tulu/dpo_qwen7b.yaml:20, 21, 25`
   - Any other `configs/tulu/*.yaml`

2. **Fix DPO YAML field mismatch** — `configs/tulu/dpo_qwen7b.yaml:19` uses `mixer_list:` but open-instruct expects `dataset_mixer_list:`. Rename the key.

3. **Add arg-allowlist filter in `scripts/launch_stage.py`** — before passing `args` to open-instruct, introspect the target script's `FlatArguments` dataclass and filter out keys not in its field set. Log a warning listing filtered keys so users know their config had stale flags.

4. **Add a README note or comment in `configs/tulu/README.md`** (create if doesn't exist) documenting that the pinned open-instruct version does NOT support Liger or packing, and pointing to #42 for the bump decision.

5. **Add a smoke test** — unit test or integration test that loads each `configs/tulu/*.yaml`, simulates arg extraction, and asserts no crash against a mock `FlatArguments` dataclass.

### Safety

- Do NOT touch `configs/distributed/*.yaml` or `configs/condition/*.yaml` or `configs/training/*.yaml`
- Do NOT modify `external/open-instruct/` (submodule)
- Do NOT attempt to fix `scripts/run_midtrain_25pct.sh` — it already works, leave it alone

## Success criteria

- `python scripts/launch_stage.py configs/tulu/sft_qwen7b_25pct.yaml --dry-run` (or equivalent parse-only invocation) completes without ValueError
- Same for `dpo_qwen7b.yaml` and `sft_qwen7b.yaml`
- Smoke test passes in CI
- Commit message: `fix(configs): strip invalid liger/packing flags from Tulu configs (refs #40 #41)`

## Budget

~1 hour implementer. No GPU.

## Dependencies

- #40 closed
- Parallel with #38, #39 (no overlap)
