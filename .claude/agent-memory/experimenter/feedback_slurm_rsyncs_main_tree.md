---
name: SLURM lane silently rsyncs main; --repo-branch is inert
description: Before posting epm:run-launched on an auto-routed launch, read the handle sidecar's resolved backend/cluster; if it resolved to SLURM for a feature-branch experiment, the rsync source is the repo-root install on `main` (`__file__`-walk in slurm.py `_default_src_root()`), NOT the invoking worktree, and `--repo-branch` is inert there — so the SLURM job will run STALE main-tree code missing the issue's entrypoint scripts. Pin `backend: gcp` for feature-branch experiments until slurm.py honors the feature branch.
type: feedback
---

**Rule.** When dispatching `scripts/dispatch_issue.py launch` on a feature-branch experiment (any task where the entrypoint scripts only live on `issue-<N>`, not on `main`), do NOT rely on `auto` routing. If `auto` falls through to a SLURM lane (`nibi` / `fir` / `mila`), the lane's rsync source is `_default_src_root()` in `src/explore_persona_space/backends/slurm.py`, which resolves via `__file__`-walk to the repo-root install on `main` — regardless of the invoking cwd or any worktree. `--repo-branch` is only honored by the GCP and RunPod lanes (git-clone the requested branch); on SLURM the flag is inert.

Before posting `epm:run-launched` on any auto-routable launch, read `.claude/cache/issue-<N>-handle.json` and check `backend` / `cluster`. If it resolved to a SLURM lane for a feature-branch experiment, the job will silently run stale `main`-tree code and crash at the first step that imports the experiment's entrypoint. Either cancel the SLURM job and relaunch with `--backend gcp` (or `--backend runpod` with a named residual gap, per the sentinel-signaling-dispatcher rule in CLAUDE.md), OR pin `backend: gcp` in the task frontmatter so the router never auto-routes to SLURM.

**Why:** Task #653 round-8 relaunch (2026-06-16). Auto router missed GCP us-central1-a on transient capacity, hit a hard config error on us-central1-b fallback (`a2-ultragpu-4g` doesn't exist in that zone), then fell through to nibi/SLURM — which rsynced the repo-root tree on `main` (no `scripts/issue_653/`) and submitted job 16173079 (PD) onto code that would crash at `--verify-imports`. The trap is general to every feature-branch experiment.

**How to apply.**

1. On any auto-routable launch (`--backend` unset, or `--backend auto`) for a feature-branch experiment, after `dispatch_issue.py launch` returns, immediately read the handle sidecar and check `backend`. If it's `cluster` AND the cluster is a SLURM lane (`nibi`/`fir`/`mila`), DO NOT post `epm:run-launched` — instead post `epm:failure v1 failure_class: infra reason: slurm-lane-rsyncs-main-tree-for-feature-branch-experiment`, cancel the SLURM job via the SLURM lane's `scancel`, and let the orchestrator relaunch with `--backend gcp` explicit.

2. To prevent the issue at submit time, pass `--backend gcp` on every relaunch for a feature-branch experiment. RunPod is the named residual fallback (sentinel-signaling-dispatcher rule, CLAUDE.md). Do NOT pin `--backend runpod` casually — it spends real money.

3. A permanent fix lives in `src/explore_persona_space/backends/slurm.py`: either thread an explicit `src_root` override from `dispatch_issue.py` to honor the invoking worktree, OR fail-loud at submit time when `--repo-branch` is set but the rsync source's HEAD is `main`. Workflow-improver tracking: candidate posted 2026-06-16 on task #653.
