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
JSON/text only; raw completions + plan-referenced analysis tensors before pod
termination; datasets upload; clean local weights after; WandB = live training
metrics only). The deep mechanics below load when you touch training / hub /
sweep code.

**Intermediate analysis tensors referenced by the plan MUST upload before pod
termination.** Any artifact the plan's analysis / negative-control sections
name as a downstream input — per-cell shift tensors (`shifts/*.pt`), cached
activations, decomposition / SVD inputs — uploads to the HF data repo under
`issueN_<slug>/analysis_tensors/` BEFORE the pod is terminated, exactly like
raw completions. These files are typically tiny (KB-MB) next to the
checkpoints they derive from, which makes them easy to dismiss as scratch —
but losing them makes the plan's remaining controls permanently unrunnable.
(Incident #521: ~200 KB per-cell Δv `.pt` files required by two planned
negative controls — the leave-one-out SVD spectrum check and the EM
mean-over-response read — were never uploaded; a 3-round upload-verification
loop still ended PASS, the pod was terminated, and both controls became
permanently unrunnable.) Enforcement: `upload-verifier` Step 1 classifies
`*.pt` / `*.npy` as analysis tensors bound for the HF data repo, and its
Step 2.8 cross-references the plan's analysis / control sections and FAILs on
any plan-named input without a permanent URL.

**Verify uploads with the Python Hub API, never the `hf` CLI.** The installed `hf`
CLI has NO `api` subcommand — `hf api list-repo-files ...` errors to stderr and
`| grep` swallows it as an empty/zero result that reads as a false "0 files"; `hf
repo-files` only exposes `delete`, not `list`. Use:
`uv run python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('superkaiba1/explore-persona-space-data', repo_type='dataset', revision='main')))" | grep <bucket>`
(#458 post-mortem nearly drew a wrong "checkpoints don't exist" conclusion from
the silent CLI "0").

Consumers of this snippet beyond post-experiment upload verification:
`follow-up-proposer` runs it as a hard gate to verify reuse premises before
tagging a follow-up `auto_run: yes` (see `.claude/agents/follow-up-proposer.md`
§ artifact-premise verification); `analyzer` runs it at clean-result write time
to ground every path-specific `**Artifacts:**` claim in a live listing (see
`.claude/agents/analyzer.md` Artifacts-grounding rule); and `clean-result-critic`
Lens 5 spot-checks an artifact path from the body against the same listing. All
three rely on the Python Hub API for the same reason — the `hf` CLI's false "0"
would corrupt their checks identically. Keep the snippet (repo, `repo_type`,
`revision`) consistent across these surfaces when editing.

**Fail-loud uploads.** `upload_dataset_directory` (`orchestrate/hub.py`) exits
non-zero on failure (`--no-upload` only for dry-runs).

**HF Hub rate limit: 256 repository commits per hour.** A sweep that pushes one
Hub commit per cell/fraction WILL hit `429: You have exceeded the rate limit for
repository commits (256 per hour)` mid-sweep, and a per-cell wrapper that only
logs "upload returned no path" as a WARNING turns the throttle into silent
artifact loss (incident #488, 2026-06-09: 41/324 adapter uploads silently
missing after rc=0 cells; caught only by a pre-phase spot-check, backfilled with
a single bulk commit in 43s). Rules: (a) sweeps producing >~200 per-cell
commits/hr batch their uploads into ONE bulk `upload_folder` commit per sweep
(or chunked commits well under the cap); (b) "upload returned no path" is a
TRACKED GAP recorded in the sweep's failure list and reconciled before the next
phase — never a warning-and-continue.

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
