---
name: Panel-recovery HF Hub pre-staging gap
description: When a plan's data-recovery chain includes "local → HF Hub → worktree fallback", the HF Hub leg often hasn't been pre-staged. Verify the leg resolves on a fresh pod BEFORE launching.
type: feedback
---

When a plan documents a fallback chain like:

  1. local `eval_results/issueXXX/*.json`
  2. HF Hub `superkaiba1/explore-persona-space-data::issueXXX_subdir/*.json`
  3. worktree `/workspace/issue-YYY/data/.../*.json`

the HF Hub leg is **frequently absent in practice**, because:
- The local file lives on the user's dev VM only (often untracked in git — large JSON artifacts are gitignored by default).
- No automation auto-uploads `eval_results/` to the data repo; that's a separate manual step the implementer rarely thinks to add.
- The worktree path is typically the prior issue's `worktrees/issue-YYY/` directory which doesn't exist on fresh pods.

**Observed on issue #368 (2026-05-13):** Phase 0.0 panel-recovery gate crashed because `base_model_generations.json` for the issue 207 gentle panel:
- Was on the local VM, 870KB, untracked in git
- Was NOT on HF Hub (the fallback target)
- The worktree path `/workspace/issue-274/data/i181_non_persona/eval_panel.json` did not exist on `pod-368`

**Why:** Plans cite "HF Hub fallback" as if HF Hub is auto-populated. It is not — uploads to the data repo happen only when an entry script explicitly calls `upload_dataset_directory` or `api.upload_file`. Most `eval_results/` are never uploaded.

**How to apply:**
1. **During preflight, when the plan documents an HF Hub fallback path: dry-run the download.** Spend 5 seconds doing `hf_hub_download(...)` against the cited filename. If it 404s, fix BEFORE launching.
2. **If the file exists on the local VM but not on HF Hub, upload it as an experimenter pre-flight fix** (not a code change — uploading a static artifact is data-staging, not logic). Use `HfApi().upload_file(path_or_fileobj=..., path_in_repo=..., repo_id="superkaiba1/explore-persona-space-data", repo_type="dataset")`.
3. Record the upload commit hash in the `epm:launch` marker so the implementer round can fold the upload step into the entry script for the NEXT respawn (or for re-runs on different seeds).
4. If the file is missing everywhere (not on local VM either) — escalate; do not try to regenerate it (the plan probably documented why regeneration would corrupt the experiment, e.g., non-deterministic LLM-generated content).

**Cost:** uploading a 1MB file to HF Hub takes ~2 sec. Catching the 404 fallback before launching saves an entire experimenter respawn round.
