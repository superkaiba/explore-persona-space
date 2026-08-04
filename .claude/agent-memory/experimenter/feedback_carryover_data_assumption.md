---
name: "Carry-over data on HF Hub" claims lie ~half the time — dry-run every claimed leg
description: Adapters + result.json are routinely uploaded; SFT JSONLs and eval_results artifacts often are NOT. Dry-run the actual download during preflight; upload from the local VM as a data-staging fix when present locally.
type: feedback
---

Plans conflate three artifact classes when claiming "carry-over data from #X on HF Hub": (1) adapter/merged checkpoints — nearly always present; (2) per-cell `result.json` in git — always present; (3) SFT training JSONLs + `eval_results/` artifacts — only uploaded if someone explicitly pushed them, and frequently NOT. Fallback chains ("local → HF Hub → worktree") routinely have an empty HF leg, and the worktree leg doesn't exist on fresh pods.

**Why:** #186's SFT data was absent from HF despite the plan asserting it; #368 (2026-05-13) Phase 0.0 crashed because the panel file was local-VM-only (870KB, untracked), not on HF, and the cited worktree path didn't exist on the pod. Nothing auto-uploads `eval_results/` to the data repo. Regenerating training data re-costs the original API spend, so a budget assuming free pull can be 2-5x short.

**How to apply (Phase-0, before any API/GPU spend):**
1. Dry-run the actual fetch: `hf_hub_download(...)` against the cited path (~5s). Also check the data repo for `issue<N>`/`i<N>` components via scoped `list_repo_tree(path_in_repo='issue<N>_<slug>', repo_type='dataset')` or `HfApi().file_exists(...)` per exact path — bare `list_repo_files` on the ~1M-file data repo times out (>90 s, #833; gotchas.md) — plus the pod's `data/sft/issue<N>/` and WandB artifacts.
2. File exists on the local VM but not HF → upload it via `HfApi().upload_file(...)` as an experimenter data-staging fix (not a code change); record the commit in the launch marker so the implementer folds the upload into the entry script for future respawns.
3. Missing everywhere → post `epm:failure v1` describing the gap. Do NOT silently regenerate via the parent's full generator (all arms = $100-200 API) — bounce for an arm-filter flag or explicit budget approval.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Carry-over data claims lie ~half the time](feedback_carryover_data_assumption.md) — dry-run every claimed HF leg before spend; SFT JSONLs/eval_results often never uploaded; upload from VM as data-staging fix (#186, #368)
