---
title: 'Step 10d lint gate: an unset WT silently certifies the WRONG tree and emits
  a benign-looking skip-artifact-only instead of failing loud'
kind: infra
tags: []
created_at: '2026-08-24T22:07:49Z'
has_clean_result: false
parent_id: 2327
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
---

# Step 10d lint gate: an unset `WT` silently certifies the WRONG tree and emits a benign-looking `skip-artifact-only` instead of failing loud

## Goal

Make the Step 10d pre-push lint gate refuse to run when `WT` is unset, empty, or not a git worktree checked out on the expected `issue-<N>` branch — and make an EMPTY own-diff on a non-artifact-only branch a loud failure rather than a `skip-artifact-only` verdict. A gate that cannot see the payload must say so; it must not emit a verdict word that reads as "nothing to check".

## The defect

The workload addresses the repository exclusively through `git -C "$WT"` (18 call sites in the transcribed recipe) and takes `WT` from the environment with no default and no assertion. When `WT` is unset or empty, `git -C ""` does not fail — it operates on the launching shell's cwd. If that cwd is the repo root (which is pinned to `main` by standing convention), the gate computes:

```bash
git -C "" diff --name-only origin/main...HEAD   # main vs main => EMPTY
```

An empty own-diff routes to the executable trigger's else-branch, which emits `skip-artifact-only` — the verdict reserved for a genuinely artifact-only payload. The sha-bind then records the WRONG tree's HEAD (the repo root's, i.e. `origin/main`), and the recipe's own anti-self-attestation comment says a consumer accepts a `pass`/`skip` verdict "ONLY while the CURRENT tip still equals this sha" — which is trivially false here, so the guard does not catch it either.

**Why this is worse than a block.** `skip-artifact-only` reads as benign. A consumer (human or successor session) glancing at the verdict sees a non-block and proceeds. The branch in the observed instance carried `scripts/*.py`, `.claude/skills/**`, and `tests/*.py` — precisely the surface the gate exists to lint — and the gate reported there was nothing lint-scanned to compare.

## Observed instance (#2327 gate run 3, 2026-08-24)

Launched with `REPO_ROOT` exported but not `WT`, and cwd set to the repo root. Result:

| Signal | Value |
|---|---|
| verdict | `skip-artifact-only` |
| rc | 0 |
| certified tip | `c5531cb487ba7eb74b38cbbe709eb9bd65da527d` |
| `origin/main` tip | `c5531cb487ba7eb74b38cbbe709eb9bd65da527d` (identical) |
| repo-root HEAD | `c5531cb487ba7eb74b38cbbe709eb9bd65da527d` (identical) |
| actual branch tip | `f361c9724c8d232f6f76d9a1168caa3b1b27df0a` (never examined) |
| `own-diff.txt` | 0 lines |
| branch HEAD moved? | no — the gate committed nothing |

Cost: ~15 min of #1962 fleet queue plus the lint legs, spent producing a verdict about `main`. The same launch composition had succeeded in runs 1 and 2, which had `WT` set — so the failure is invisible to anyone reusing a working launch line without re-checking the environment.

## Scope to investigate

1. **Fail loud on a bad `WT`** at the top of the workload, before the fleet queue is spent: assert `WT` is non-empty, is a directory, `git -C "$WT" rev-parse --is-inside-work-tree` succeeds, and `git -C "$WT" rev-parse --abbrev-ref HEAD` equals the expected `issue-<N>` branch. Any failure exits non-zero with a named reason — never a verdict word.
2. **Separate "empty own-diff" from "artifact-only payload."** These are different states with the same current verdict. An own-diff that is empty when the branch demonstrably differs from `origin/main` is a gate malfunction; `skip-artifact-only` should require a NON-EMPTY own-diff all of whose entries are artifact paths.
3. **Record the examined tree in the verdict file**, not only its sha — e.g. the resolved `WT` path and branch name on a third line — so a wrong-tree run is legible from the artifact alone rather than requiring a sha comparison against `origin/main`.
4. **Prefer deriving `WT` over inheriting it.** The recipe could resolve the worktree from the issue number via the same `git rev-parse --path-format=absolute --git-common-dir` idiom the rest of the workflow uses, making the env var an override rather than a silent requirement.
5. Check whether the sibling Step 9c gate and the Step 9a-ter inline payload lint gate share the `git -C "$WT"`-without-assertion shape.

## Non-goals

Do not default `WT` to the cwd — that is the current failure mode made explicit. Do not remove `skip-artifact-only`; it is a legitimate verdict for a genuinely artifact-only payload, and the fix is to stop reaching it from a malfunction. Do not treat this as solely a launcher-composition error to be documented in prose: the launching agent had `REPO_ROOT` right and `WT` missing, got rc=0 and a non-block verdict, and nothing in the output said the wrong tree had been read — a gate whose silent-misconfiguration mode looks like success needs the assertion regardless of who composed the launch.

## Provenance

Diagnosed by the #2327 orchestrator after its own run-3 launch omitted `WT`. The launch error was the orchestrator's; the fail-loud gap is the recipe's. Confidence: high — the certified sha, the repo-root HEAD, and `origin/main` are byte-identical, the own-diff artifact is 0 lines, and the branch HEAD is unchanged. Dedup target: `.claude/skills/issue/steps/18-step-10d.md` (`WT` resolution + the artifact-only trigger's else-branch), distinct from #2539 (outer-fence derivation) and from the normalization `note:`-line defect filed alongside this.
