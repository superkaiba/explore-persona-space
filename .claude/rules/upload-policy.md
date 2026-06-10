---
description: Deep upload mechanics — Hub-API verification gotcha, inline-upload fence, delete-after-eval adapter-persist recipe (loads when writing training / hub / sweep code)
paths:
  - "src/explore_persona_space/orchestrate/**"
  - "scripts/train.py"
  - "scripts/run_sweep.py"
  - "src/explore_persona_space/train/**"
  - "scripts/issue*.py"
---

# Upload mechanics (deep)

The always-on **Upload Policy** in CLAUDE.md carries the destination table + the
core rules (models upload to HF before local deletion; `eval_results/` is
JSON/text only; raw completions before pod termination; datasets upload; clean
local weights after; WandB = live training metrics only). The deep mechanics
below load when you touch training / hub / sweep code.

**Verify uploads with the Python Hub API, never the `hf` CLI.** The installed `hf`
CLI has NO `api` subcommand — `hf api list-repo-files ...` errors to stderr and
`| grep` swallows it as an empty/zero result that reads as a false "0 files"; `hf
repo-files` only exposes `delete`, not `list`. Use:
`uv run python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('superkaiba1/explore-persona-space-data', repo_type='dataset', revision='main')))" | grep <bucket>`
(#458 post-mortem nearly drew a wrong "checkpoints don't exist" conclusion from
the silent CLI "0").

**Fail-loud uploads.** `upload_dataset_directory` (`orchestrate/hub.py`) exits
non-zero on failure (`--no-upload` only for dry-runs).

**Inline-upload fence `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD`.** `_finalize_phase`
auto-uploads merged checkpoints to WandB Artifacts; orchestrators doing their own
tagged upload set the env in `try/finally` to prevent double-uploads.

**Delete-after-eval sweeps MUST persist the ADAPTER first (never the merged dir).**
A sweep that `rm`s a trained checkpoint after its eval to stay under the MooseFS
~130GB quota (the #404/#458 pattern) MUST set `EPM_PERSIST_ADAPTER_HF_REPO` +
`EPM_PERSIST_ADAPTER_SUBFOLDER` so `_finalize_phase` uploads **and verifies** the
LoRA adapter (~300MB) before it is reaped. The persist is **fail-loud**: if it
can't verify the adapter landed, training raises and exits non-zero, so the
launcher's `set -e` aborts the cell *before* its `rm` — closing the silent-loss
hole. NEVER upload the ~15GB merged checkpoint to the shared public model repo to
satisfy this: it's derived data (regenerable from base + adapter), 45× larger, and
would blow the already-~550GB HF repo quota (the same quota that soft-failed
#458's merged upload, after which the `rm` deleted all 36 checkpoints). Pair this
with `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` + `upload_to=none` on the train call so
the wasteful 15GB merged WandB/HF uploads don't fire at all. Re-eval = download
adapter, re-merge with base.
