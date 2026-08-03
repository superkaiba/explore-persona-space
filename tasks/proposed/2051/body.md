---
title: 'workflow-fix: pod bootstrap full-clones 10.5GB, burns ~$15 idle GPU per provision'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c3219321d60b
created_at: '2026-08-03T18:41:23Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during #1739 armfill attempt 2: a 4xH200 at
  $22/hr sat 20+ min at 0% GPU on a depth=1 full clone (10.51 GB / 175,766 files at
  HEAD; 8.94 GB eval_results). new_worktree.sh already sparse-excludes those dirs
  on the VM side.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a measured cost hit during
the #1739 armfill round (emitting agent: orchestrator, PA chat session, 2026-08-03).

## Goal

Give `bootstrap_pod.sh` a partial/sparse clone (`--filter=blob:none` or a cone
sparse-checkout limited to code dirs, artifacts added on demand) mirroring the
`new_worktree.sh` precedent.

## Workflow gap

- **Bug observed:** `bootstrap_pod.sh` clones the full repo at `--depth=1` with
  no sparse-checkout and no blob filter, pulling 10.51 GB / 175,766 files
  (8.94 GB `eval_results` + 1.08 GB `figures`) onto every fresh pod, so a slow
  link burns 30-40 min of idle GPU time per provision.
- **Why it is a workflow gap:** a fresh 4xH200 pod ($22/hr) sat at 0% GPU for
  20+ minutes on 2026-08-03 doing nothing but this transfer — `tmp_pack` grew
  2.18 GB -> 2.70 GB in 4 min (~2.1 MB/s), compressed pack projected ~3-5 GB and
  ~35 min total. That is ~$13-15 of idle GPU per provision on a slow link, and
  ~$25-30 on an 8xH200. The legs execute code out of `scripts/` (0.05 GB);
  virtually the whole transfer is committed artifacts the pod reads from HF, not
  from git. The VM side already solved exactly this: fresh worktrees are sparse
  by default and EXCLUDE `eval_results/` and `figures/`. Pods never got the same
  treatment. Secondary effect: the long opaque transfer is indistinguishable at a
  glance from the commit-less-repo wedge, which has now twice prompted an agent
  to consider killing a HEALTHY clone mid-flight.
- **Confidence (emitter):** high
- verified-at-filing: `git ls-tree -r -l origin/main` -> 10.51 GB across 175,766
  files; per-dir eval_results 8.94 GB, figures 1.08 GB, tasks 0.31 GB, scripts
  0.05 GB. `grep -n 'depth=1|--filter|sparse-checkout' scripts/bootstrap_pod.sh`
  -> the shallow fetch at :263 plus a working-tree materialization at :265, with
  NO `--filter` and NO `sparse-checkout` anywhere in the file; the comment at :47
  documents the slow-link problem ("~200KB/s observed against a 2.8GB") but the
  chosen mitigation is depth-only. Precedent exists at `scripts/new_worktree.sh`
  :297-302 (cone sparse-checkout with on-demand adds). (2026-08-03)

## Proposed change (candidate diff sketch — refine in planning)

In the fresh-init branch of `bootstrap_pod.sh`, before the shallow fetch:

```
+ # Pods execute code, not committed artifacts (eval_results 8.94 GB /
+ # figures 1.08 GB at HEAD). Fetch blobs lazily; an on-demand checkout of an
+ # artifact path still resolves against the promisor remote.
+ git config remote.origin.promisor true
+ git config remote.origin.partialclonefilter blob:none
  git fetch -q --depth=1 --filter=blob:none origin "$BRANCH"
```

Alternative (mirrors `new_worktree.sh` more literally): a cone sparse-checkout
seeded with the code dirs, artifacts added on demand. The planner picks one;
`blob:none` is likely simpler for pods because any path a workload later opens
still resolves transparently, whereas sparse-checkout needs an explicit cone
list and a registry like `tests/sparse_cones.txt`.

## Scope / surfaces

- Primary target: `scripts/bootstrap_pod.sh`
- Verify no pod-side consumer depends on a full working tree (grep pod-side
  dispatchers for direct `eval_results/` / `figures/` reads) before landing.
- Check whether the GCP/SLURM lanes share this clone path and benefit equally.

## Constraints / invariants

- Must keep working against a tokenless public HTTPS remote and preserve the
  credential-helper retrofit (#1239) already in the script.
- Must not regress the existing-repo re-bootstrap path (the fast-forward pull).
- Smoke on a real fresh pod before landing: the clone completes, and both
  `issue1739_wcrung_arms_run.py` and a repo-resident artifact path still open.

## Provenance

- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: c3219321d60b
