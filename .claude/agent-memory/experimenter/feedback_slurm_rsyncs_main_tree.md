---
name: SLURM honors --repo-branch since #793; workload-side git push stays impossible
description: SlurmBackend.prepare materializes the requested branch tree VM-side (#793 — _resolve_rsync_source → materialize_branch_src; an unresolvable branch fails loud with RuntimeError at prepare). Do NOT epm:failure/scancel an auto-routed SLURM feature-branch launch on the old "rsyncs stale main" ground — that instruction predates #793. Residual truth — the cluster has NO git checkout ($SCRATCH rsync), so workload-side git push of results is impossible; results land via fetch_results pull + confirm_artifacts + a VM-side orchestrator commit.
type: feedback
---

**Rule.** An `auto`-routed launch that resolves to a SLURM lane (`nibi` / `fir` /
`mila`) for a feature-branch experiment DOES run the branch code as of #793:
`SlurmBackend.prepare` resolves the rsync source via `_resolve_rsync_source` →
`materialize_branch_src` (a complete branch tree materialized VM-side through the
`git_cloner` seam; an unresolvable branch fails loud with `RuntimeError` at
prepare, and the #653 `_assert_repo_branch_synced` guard is retained
belt-and-suspenders). Do NOT cancel (`scancel`) the job or post `epm:failure` on
a SLURM-resolved feature-branch launch on the old "rsyncs stale main" ground —
that instruction predates #793 and would kill a healthy job. A launch that
SUCCEEDS on a SLURM lane for a non-`main` branch already materialized the branch
tree (or the install sat on the branch) — post `epm:run-launched` normally.

**Why:** Task #653 round-8 relaunch (2026-06-16) hit the pre-#793 trap: the
SLURM rsync source was the repo-root install on `main` (`__file__`-walk in
`_default_src_root()`), `--repo-branch` was inert on the lane, and job 16173079
was queued onto stale code missing `scripts/issue_653/`. #653 added the
refuse-loud guard; #793 superseded the refusal with VM-side branch
materialization. This memory previously instructed `epm:failure` + `scancel` on
any SLURM-resolved feature-branch launch — post-#793 that is actively harmful
(it cancels a healthy materialized-branch job), so the rule above replaces it.

**How to apply.**

1. Residual truth (unchanged by #793): cluster compute nodes run on an
   ephemeral `$SCRATCH` rsync with NO git checkout — a workload-side
   `git commit` / `git push` of results is impossible and fails loud
   (`fatal: not a git repository`). Results land via
   `SlurmBackend.fetch_results` (rsync pull of `eval_results/` + `figures/`
   back to the VM; WARN-only by the #598 contract) + `confirm_artifacts` (the
   downstream hard gate) + a VM-side orchestrator commit. See
   `.claude/rules/pod-side-reporting.md` § "Result-push verification contract
   (#1205)", SLURM lane bullet.
2. A dispatch script whose deliverable REQUIRES workload-side git-committed
   results must not route to SLURM — pin `backend: runpod`, the one LIVE
   git-clone lane (named residual gap per CLAUDE.md). Pinning
   `backend: gcp` is REFUSED since #2028 (`GcpDisabledError`); that
   option returns only under a deliberate
   `GCP_PROVISIONING_DISABLED = False` rollback.
3. On a genuinely unresolvable branch, `prepare` itself fails loud
   (`RuntimeError` from the cloner) and the router advances lanes — no
   experimenter-side cancel is needed for the stale-tree class.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [SLURM honors --repo-branch since #793; workload-side git push impossible](feedback_slurm_rsyncs_main_tree.md) — `SlurmBackend.prepare` materializes the branch tree VM-side (fail-loud `RuntimeError` on unresolvable); do NOT scancel/fail a SLURM feature-branch launch on the old stale-main ground; results land via `fetch_results` pull + `confirm_artifacts` + VM-side commit (no cluster git checkout) — #653/#793
