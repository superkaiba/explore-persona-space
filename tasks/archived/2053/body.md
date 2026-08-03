---
title: 'workflow-fix: sparse-cone bootstrap_pod.sh (65min MooseFS checkout) + rsync
  stale-read'
kind: infra
tags:
- wf-fix
- wf-fix-fp:moosefs-podsetup
created_at: '2026-08-03T20:12:12Z'
has_clean_result: false
origin_prompt: 'Two #1482 agents independently hit MooseFS pod-setup pathology: a
  175,707-file working-tree checkout (~65 min, ~1.4h of an 8xH100) that a scripts/+src/
  sparse cone lands in ~18-90s, and an rsync overwrite stale-served at rc=0 so a fixed
  defect reproduced verbatim. bootstrap_pod.sh has 0 sparse-checkout support; new_worktree.sh
  already has 10.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from concerns surfaced independently by TWO agents on task #1482 in the same session (emitting agents: `densesae-fullwidth`, `runlen-capture`). Both hit the same MooseFS pod-setup pathology and both worked around it by hand.

## Goal

Make pod bootstrap land a driver-runnable tree in seconds rather than ~65 minutes, by adding a sparse-checkout cone to `bootstrap_pod.sh`; and document the MooseFS rsync-overwrite stale-read trap alongside the existing git-checkout one.

## Workflow gap

- **Bug observed (A — the expensive one):** a fresh pod bootstrap runs a full working-tree checkout of 175,707 files onto MooseFS, measured at ~65 minutes. `densesae-fullwidth` killed it at 12 files and applied a `scripts/`+`src/` sparse cone, which landed the same usable tree in ~18 s (a second pod later measured ~90 s). `runlen-capture` independently hit a related failure: a full-history fetch pulled 701k objects / 3.1 GB and never finished, and it staged by rsync instead. On the #1482 grid pod this consumed ~1.4 h of an 8xH100 (~22 GPU-h billed against 2.39 GPU-h of actual cell compute — the single largest line item in that gap).
- **Bug observed (B — the silent one):** an rsync OVERWRITE onto MooseFS was stale-served. rsync returned rc=0, the pod continued running the OLD bytes, and an already-fixed upload leg reproduced its previous error VERBATIM. `rm`-then-rsync plus sha-on-both-sides fixed it. This reads exactly like "my fix didn't work" and will burn a crash-fix round for whoever meets it next.
- **Why it is a workflow gap:** `bootstrap_pod.sh` is on the workflow surface and every driver-only pod in the fleet pays (A) today. The repo ALREADY has the pattern — `new_worktree.sh` implements sparse checkout (10 references) and even ships a `tests/sparse_cones.txt` cone registry — so this is porting an existing, tested VM-side solution to the pod side, not inventing one. For (B), `.claude/rules/crash-fix-rounds.md` documents the MooseFS stale-served-bytes trap for GIT checkouts (`git hash-object` vs `git rev-parse HEAD:<path>` byte probe) but not the rsync variant, so the existing entry does not fire for the staging path two agents actually used.
- **Confidence (emitter):** high — measured, reproduced independently by two agents, with a working fix already applied by hand twice.
- verified-at-filing: `grep -n "sparse\|reset --hard\|git clone\|FETCH_HEAD\|--depth" scripts/bootstrap_pod.sh` → `--depth=1` shallow fetch present at L263 but **0 hits for "sparse"** (`grep -c sparse scripts/bootstrap_pod.sh` → 0), and the tree is landed by `reset --hard FETCH_HEAD` at L263-266, i.e. a full working-tree checkout. Control: `grep -c "sparse-checkout" scripts/new_worktree.sh` → 10, confirming the pattern exists in-repo on the VM side. Landed-fix check: `git log --oneline --since='14 days ago' -- scripts/bootstrap_pod.sh` → only `a6ef9b8045` (uv PATH, #1552), unrelated. (2026-08-03)

Note the `--depth=1` already present addresses the OBJECT fetch, not the working-tree checkout; the 175,707-file materialisation onto MooseFS is the cost, and sparse-checkout is what removes it. Do not mistake the existing shallow-fetch flag for coverage.

## Proposed change (candidate diff sketch — refine in planning)

1. `scripts/bootstrap_pod.sh`: after `git init` + `fetch`, enable a sparse cone before `reset --hard FETCH_HEAD`:
   ```
   + git sparse-checkout init --cone
   + git sparse-checkout set scripts src configs tests
     git reset --hard FETCH_HEAD
   ```
   Cone set is a planning decision — mirror `new_worktree.sh`'s approach and its `tests/sparse_cones.txt` registry rather than hardcoding a fresh list. Provide an opt-out (`--full`, or an env flag) for the rare pod that genuinely needs the whole tree.
2. `.claude/rules/gotchas.md`: add the MooseFS rsync-overwrite stale-read entry — rc=0 is not evidence the bytes landed; `rm` the destination before rsync and sha-compare both sides; the signature is a just-fixed defect reproducing its old error verbatim.

## Scope / surfaces

- Primary target: `scripts/bootstrap_pod.sh`
- Secondary: `.claude/rules/gotchas.md` (the rsync-overwrite entry)
- Cross-check `.claude/rules/crash-fix-rounds.md` § MooseFS stale-served bytes so the git and rsync variants point at each other rather than duplicating.
- Verify against `tests/sparse_cones.txt`: the VM-side gate pre-adds cones because a sparse worktree otherwise fails the full pytest suite. A pod runs drivers, not the suite, so the pod cone can be narrower — but confirm no pod-side driver reads a path outside the chosen cone (the #1482 driver's `PROJECT_ROOT` fix, commit `fa810c5313`, exists precisely because a sparse pod tree has no `tasks/`).

## Constraints / invariants

- Workflow-surface only.
- Must not break the non-MooseFS lanes (GCE, SLURM, local) — sparse-checkout is a plain git feature but the change is in a shared bootstrap path.
- A pod whose driver needs a path outside the cone must fail LOUD at import, not silently read a missing file. The #1482 `repo_root()` crash is the reference behaviour: it died at `--import-check` before launch, which is the correct shape.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: moosefs-podsetup-sparse-cone-and-rsync-stale

Surfaced as prose follow-ups, verbatim:

> "Note I hit the same bootstrap trap as last time and pre-empted it: a fresh clone's `reset --hard FETCH_HEAD` starts a 175,707-file checkout on MooseFS. I killed it at 12 files and applied the scripts/+src/ sparse cone, which lands the same tree in ~90 s. Worth folding into bootstrap_pod.sh as a general fix — every pod that only runs a driver pays that 65-minute checkout today." (densesae-fullwidth)

> "The pod's git clone over MooseFS is pathological — a full-history fetch pulled 701k objects / 3.1 GB and never finished; I staged by rsync instead. Related: an rsync OVERWRITE on MooseFS got stale-served (rsync rc=0, pod still ran old bytes, and my already-fixed upload leg reproduced the old error verbatim). rm-then-rsync + sha-both-sides fixed it; saved as an agent memory since it reads exactly like 'my fix didn't work'." (runlen-capture)
