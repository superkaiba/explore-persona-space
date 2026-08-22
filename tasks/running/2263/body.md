---
title: verify_carryover_inputs.py default ref checks the wrong tree on a suffixed
  issue branch (Step 6a.5 gate)
kind: infra
tags:
- workflow-fix
created_at: '2026-08-13T03:11:57Z'
has_clean_result: false
origin_prompt: 'Surfaced during #1336 launch validation: --issue 1336 defaulted to
  origin/issue-1336 (stale, unrelated) instead of the work branch origin/issue-1336-fullcorpora;
  4 fail vs 0 fail on the same plan.'
workflow: v1
---
---
kind: infra
---

# `verify_carryover_inputs.py` default ref is wrong for any SUFFIXED issue branch — the Step 6a.5 pre-provision gate silently checks a different tree

## Goal

Make the Step 6a.5 carry-over gate resolve the ref the dispatch will ACTUALLY materialize, rather than the `origin/issue-<N>` name it guesses — or fail loud when it cannot.

## The bug, as observed

`scripts/verify_carryover_inputs.py --issue <N>` defaults its check ref to `origin/issue-<N>` if that ref exists, else `origin/main` (`--ref` help text). Many tasks do NOT work on a bare `issue-<N>` branch: task #1336's work branch is **`issue-1336-fullcorpora`**, while a *stale, unrelated* `origin/issue-1336` also exists (`a47359d5c3`). So the gate resolved the wrong tree and reported:

```
checked 43 citation(s) against origin/issue-1336: 4 fail / 1 warn      # rc=1  — WRONG TREE
checked 43 citation(s) against origin/issue-1336-fullcorpora: 0 fail / 1 warn   # rc=0 — correct tree
```

Same plan, same paths, same lane — 4 FAILs vs 0, decided purely by which branch name the default guessed. Observed 2026-08-13 while gating a 210 GPU-h dispatch on #1336 (`epm:progress` v277-v278).

## Why it matters in both directions

This gate exists to catch a carry-over input that the compute lane's clone/rsync will not materialize, BEFORE a provision is spent. A wrong-ref resolution breaks it symmetrically:

- **False FAIL** (what happened): the gate indicts a healthy plan. Cheap when someone notices; the failure mode is a human deciding the gate is noisy and passing `--no-fetch` or skipping it.
- **False PASS** (the dangerous one): if the stale `origin/issue-<N>` branch happens to CONTAIN the cited paths while the actual work branch does not, the gate certifies reachability for a tree the run will never see, and the missing input surfaces after the provision — precisely the #1469/#1689 class this gate was built to prevent.

The suffixed-branch convention is routine in this repo (`bash scripts/new_worktree.sh .claude/worktrees/<name> <branch>` imposes no `issue-<N>` naming), and `dispatch_issue.py launch` already takes the authoritative answer as `--repo-branch`.

## Proposed fix (implementer to confirm/adjust)

1. **Prefer the caller's branch.** Accept the dispatch's `--repo-branch` value (or default to the current worktree's branch when it is not `main`) and check THAT ref; keep `--ref` as the explicit override.
2. **Refuse ambiguity rather than guessing.** When `--ref` is absent and the resolved default is not the caller's branch — in particular when a bare `origin/issue-<N>` exists AND the invoking worktree is on a different `issue-<N>-*` branch — fail loud naming both candidates, instead of silently picking one. An unresolvable ref must never read as a clean check.
3. **Print the resolved ref in the verdict line at every severity**, not only in the summary tail, so a wrong-tree run is visible in any log excerpt.
4. Pin with tests: a suffixed work branch + a stale bare `issue-<N>` ⇒ the check resolves the suffixed ref (or refuses); an explicit `--ref` still wins; `main`-only repos behave exactly as today.

## Out of scope

The `[WARN] data/issue_1336/... data-local-only` behavior (gitignored `data/` is self-built or HF-staged by the workload, artifact-reuse check (h)) is correct and unchanged. No change to `--lane` semantics or to the `--extra-sync-path` validation (`backends.slurm.validate_extra_sync_paths` is path-agnostic and worked correctly).

## Provenance

Surfaced by the #1336 orchestrator while validating that a committed adjudication record would reach the fellows rsync scratch. The `--extra-sync-path` mechanism itself worked as designed; only the ref default was wrong.
