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

Make a fresh pod's bootstrap clone stop transferring the project's entire
committed artifact tree, by pairing a `blob:none` partial clone WITH a cone
sparse-checkout. Both pieces are required — see § Why neither piece works alone.

## Workflow gap

- **Bug observed:** `bootstrap_pod.sh` clones the full repo at `--depth=1` with
  no sparse-checkout and no blob filter, pulling 10.51 GB / 175,766 files
  (8.94 GB `eval_results` + 1.08 GB `figures`) onto every fresh pod, so a slow
  link burns 30-40 min of idle GPU time per provision.
- **Why it is a workflow gap:** a fresh 4xH200 pod ($22/hr) sat at 0% GPU for
  25+ minutes on 2026-08-03 doing nothing but this transfer — pack grew
  2.18 GB -> 2.63 GB over ~8 min, rate degrading from ~2.1 MB/s to ~0.9 MB/s,
  never reaching a commit. That is ~$13-15 of idle GPU per provision on a slow
  link, ~$25-30 on an 8xH200. The legs execute code out of `scripts/` (0.05 GB);
  virtually the whole transfer is committed artifacts. The VM side already
  solved this: fresh worktrees are sparse by default and EXCLUDE `eval_results/`
  and `figures/`. Pods never got the same treatment. Secondary effect: the long
  opaque transfer is indistinguishable at a glance from the commit-less-repo
  wedge, which twice in one session prompted an agent to consider killing a
  HEALTHY clone.
- **Confidence (emitter):** high
- verified-at-filing: `git ls-tree -r -l origin/main` -> 10.51 GB across 175,766
  files; per-dir eval_results 8.94 GB, figures 1.08 GB, tasks 0.31 GB, scripts
  0.05 GB. `grep -n 'depth=1|--filter|sparse-checkout' scripts/bootstrap_pod.sh`
  -> the shallow fetch at :263 plus a working-tree materialization at :265, with
  NO `--filter` and NO `sparse-checkout` anywhere in the file; the comment at :47
  documents the slow-link problem ("~200KB/s observed against a 2.8GB") but the
  chosen mitigation is depth-only. Precedent at `scripts/new_worktree.sh`
  :297-302 (cone sparse-checkout with on-demand adds). (2026-08-03)

## Why neither piece works alone (SUPERSEDES two earlier revisions of this body)

This section replaces earlier guidance in this task that was wrong in BOTH
directions. Read it before implementing; the naive readings are no-ops or
outages.

**`--filter=blob:none` ALONE saves nothing here — it is a no-op fix.** The
filter omits blobs from the initial pack, but materializing the working tree at
HEAD requires the blob for every file at HEAD, so git immediately lazy-fetches
all of them. What the filter normally spares is blobs from HISTORY, and
`--depth=1` has already excluded history. Net effect on a pod doing a full
checkout: the same ~10.51 GB, in two round-trips instead of one. An
implementation that adds only `--filter=blob:none` will appear to fix this task
while changing nothing measurable — the worst possible outcome, because the bug
gets closed.

**A cone sparse-checkout ALONE is unsafe as originally proposed.** A cone of
code dirs only (`scripts/ src/ experiments/ configs/ tests/`) breaks any issue
whose legs read committed `eval_results/` inputs. Measured on #1739: both
scorers resolve `train_summary` to
`eval_results/issue_<N>/<behavior>/arm_results/all_arms_spearman.json` (~67 MB,
x3 behaviors) for committed-frozen layer selection, and the pvsynth DV at
`eval_results/issue_<N>/pvsynth/dv_dataset/<behavior>/labeling.json` is never
staged by the driver so it can only come from the repo. Missing them means
either a fail-loud at `_missing_inputs` or — materially worse — a silent
fallback to own-pool frozen layers that destroys comparability with the
committed wide-roster column. Reading committed `eval_results/` as a downstream
input is common in this project, so this is a CLASS of silent-fallback bug.

**The PAIRING is what works, and it also fixes the enumeration risk.** Under
blob:none + cone, the working tree never asks for out-of-cone blobs, so the
saving is real; and a path nobody enumerated is recoverable in SECONDS with
`git sparse-checkout add <path>`, which materializes it and lazily fetches just
those blobs from the promisor remote. A cone miss becomes a few-second
correction rather than a re-clone. This is the pattern `new_worktree.sh` already
uses VM-side.

## Proposed change (candidate diff sketch — refine in planning)

In the fresh-init branch of `bootstrap_pod.sh`:

```
  git init -q -b "$BRANCH"
  git remote add origin ...
+ git config remote.origin.promisor true
+ git config remote.origin.partialclonefilter blob:none
+ git sparse-checkout init --cone
+ # Code dirs PLUS the task's own artifact dir. ISSUE must be threaded into
+ # bootstrap; a code-dirs-only cone breaks committed-input reads (see above).
+ git sparse-checkout set scripts src configs tests "eval_results/issue_${ISSUE}"
  git fetch -q --depth=1 --filter=blob:none origin "$BRANCH"
```

Threading the issue number into bootstrap is part of the change, not an
assumption. An issue reading a SIBLING issue's committed artifacts still needs
`sparse-checkout add` — acceptable precisely because that is now a seconds-long
recovery, but worth a one-line note in the pod runbook.

## Acceptance test (do not weaken)

A clone that completes is NOT sufficient evidence. The test must include a leg
that OPENS a committed artifact end-to-end — read
`eval_results/issue_<N>/<behavior>/arm_results/all_arms_spearman.json` and assert
non-empty — which simultaneously proves the lazy-blob path works and the cone is
correct. Also assert the transfer is actually smaller (bytes fetched, or wall
time to first usable commit) against the current behavior, or the no-op failure
mode above ships undetected.

## Scope / surfaces

- Primary target: `scripts/bootstrap_pod.sh`
- Thread the issue number through the provision path that invokes it.
- Check whether the GCP/SLURM lanes share this clone path and benefit equally.

## The re-bootstrap path BREAKS on a sparse/detached repo (measured, 2026-08-03)

This is the failure the fix itself will cause fleet-wide if not handled, so it
is in scope, not a footnote. On pod-1739-armfill, after the wedged fetch was
killed, the provision wrapper retried bootstrap from step 1. Its
"repo exists -> pull" branch cannot handle a detached, sparse, pinned checkout
and died at step 4:

    fatal: Updating an unborn branch with changes added to the index
    -> rebase abort -> exit 1, "Pod is up but not experiment-ready"

Consequence: bootstrap steps 5-11 never ran — **no venv, no HF/uv cache
redirects, no preflight** — leaving a pod that looks provisioned, has an intact
repo, and cannot execute anything. Recovery required replicating those steps by
hand (~15 min of paid GPU idle).

Today this is reachable only after a killed fetch. **Once this task makes
sparse+pinned the DEFAULT, every re-bootstrap hits it.** So the change must
include the re-bootstrap branch:

- Detect a detached HEAD and/or an active sparse-checkout and take a
  fetch+reset-to-pin path instead of `pull --ff-only` / `pull --rebase`.
- Never leave the pod "repo OK, environment absent": if the repo branch exits
  non-zero, either continue to steps 5-11 anyway or fail the provision loudly.
  A half-bootstrapped pod that bills while looking healthy is the worst state.
- Regression test: re-run bootstrap against an ALREADY-sparse, detached clone
  and assert it reaches the preflight step.

### Second failure mode: bootstrap RE-ENTERS the broken branch by itself (2026-08-03, pod-1739-ext)

The failure above needs someone to re-run bootstrap. This one does not. On a
slow-git pod, after an operator killed a stalled full clone, the provision's
STILL-RUNNING bootstrap re-entered its own "repo exists -> pull" branch (pids
998/1004) and launched a SECOND, UNFILTERED `git fetch --update-head-ok origin
main` that competed with the operator's filtered clone for the same starved
pipe. It had to be killed by pid.

So the branch is reachable with no human action whenever the first clone is
interrupted — which on a slow link is exactly when an operator is most likely
to intervene. Any fix must make the repo branch either idempotent or
non-re-entrant, not merely correct on a clean first pass.

Measured context from the same pod: github served **0.15 MB/s** (6 MB per 40 s)
against a 10.06 GB / 175,971-file HEAD — roughly **18 hours** of idle-GPU time
for a stock `--depth=1` clone on a 4xH200. The `--filter=blob:none` + narrow
cone brought `.git` to 37.5 MB. Note a cone including `eval_results/issue_<N>`
(0.76 GB / 839 files) was STILL lazily pulling blobs at 0.13 MB/s, so on a
pathologically slow link the working recipe was a CODE-ONLY cone with the
round's ~143 MB of data files shipped VM->pod over the fast path instead. That
is a useful refinement of the cone guidance above: the right cone depends on
link speed, and the data files can bypass git entirely.

Related probe lesson worth a line in the pod runbook: while `uv sync` is
running there are NO python processes (uv is a Rust binary) and GPUs read 0%.
Neither is evidence of idleness. Probe for the WORK (uv cache byte growth,
`.venv` completion, the bootstrap script pid), not for an assumed executable
name — the same process-identity trap as #2050, in the opposite direction.

## Constraints / invariants

- Must keep working against a tokenless public HTTPS remote and preserve the
  credential-helper retrofit (#1239) already in the script.
- Must not regress the existing-repo re-bootstrap path — see the section above;
  the naive `pull --ff-only` branch is already broken for sparse/detached repos
  and this change makes that state the norm.
- Cache redirects (uv + HF) must be established BEFORE any dependency sync: an
  unredirected `uv sync` puts ~8 GB on `/` (50 GB overlay), and staging behind
  it hits ENOSPC. Measured on this pod: redirects first kept `/` at 153 MB.
- Smoke on a real fresh pod before landing, per the acceptance test above.

## Provenance

- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: c3219321d60b
