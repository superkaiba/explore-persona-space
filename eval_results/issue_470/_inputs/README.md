# #411 DV snapshot — committed inputs for the #470 re-analysis

This directory contains a **snapshot** of the #411 sycophancy-cosine-gradient
analyze outputs. They are committed into the #470 worktree because:

1. The original files live on the unmerged `issue-411` branch / its worktree,
   so they are NOT available on a freshly-bootstrapped pod (which only has
   `main`).
2. The #470 experiment is a **predictor-only re-analysis** on the frozen DV.
   Snapshotting at plan-approval time pins the DV against any later #411
   re-analysis and keeps the #470 result reproducible.

## Files

- `analyze_summary.json` — copied from
  `.claude/worktrees/issue-411/eval_results/issue_411/analyze_summary.json`.
  Carries `per_source.<src>.per_panel_delta` (the DV), `per_panel_cosine_to_source`
  (the cosine baseline), and `per_panel_trained_rate` / `per_panel_base_rate`.
- `base_panel_rates.json` — copied from the same directory. Carries the
  intrinsic-base-rate `panel_rates` map used by the bystander-base-rate
  baseline predictors (plan §4).

## Provenance

- Source worktree commit (`git -C .claude/worktrees/issue-411 rev-parse HEAD`):
  `c34cb3d11` (recorded in #411 plan §3.5).
- Snapshot date: 2026-06-02 (before any #470 GPU work begins).

## Loader

`src/explore_persona_space/experiments/predictor_jsdiv_470/common.py` resolves
the DV files in this order:

1. Repo-committed snapshot at `<PROJECT_ROOT>/eval_results/issue_470/_inputs/`
   (always present on a fresh pod; this is the production path).
2. Dev-only fallback at `<PROJECT_ROOT>/../../worktrees/issue-411/...` (kept
   so a local dev VM can re-snapshot from the live #411 worktree if needed).

If the production-path snapshot is missing on a pod, Phase 4 raises immediately
with a clear error pointing here.
