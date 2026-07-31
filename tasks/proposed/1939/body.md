---
title: 'workflow-fix: gotchas.md entry — SLURM git-less scratch crashes metadata git
  shellouts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:64f82be41bfb
created_at: '2026-07-31T14:10:56Z'
has_clean_result: false
origin_prompt: 'orchestrator-surfaced gotcha candidate from #1902 crash 3 (fellows
  job 16142): git-less SLURM scratch crashes _git_sha metadata shellout; add gotchas.md
  entry'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own crash-3 diagnosis on task #1902 (emitting agent: orchestrator, /issue 1902 session).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting that the fellows/SLURM `materialize_branch_src` scratch tree has NO `.git` checkout, so any reproducibility-metadata helper shelling `git rev-parse HEAD` with `check=True` crashes the workload rc=128 — degrade EPS_GIT_SHA env → `check=False` → `"unavailable-no-git-checkout"` literal.

## Workflow gap

- **Bug observed:** fellows job 16142 (task #1902 launch 3) crashed P1 rc=1 at `issue1902_run._git_sha` (`CalledProcessError` exit 128, "fatal: not a git repository") writing `leg_report.json`, AFTER all pilot capture work succeeded — a full launch cycle burned on a metadata shellout.
- **Why it is a workflow gap:** the no-git fact is documented only for the RESULT-PUSH side (`pod-side-reporting.md` SLURM-lane bullet; CLAUDE.md compute-backends prose "The cluster still has NO git checkout"); gotchas.md has no entry for the METADATA-shellout crash class, and `code-style.md`'s "Reproducibility metadata in result JSONs: git commit hash..." bullet actively steers implementers into strict `git rev-parse` calls with no lane caveat — the same driver had the tolerant pattern in its commit path (`fits._commit_eval_results`) and the strict pattern in its metadata path.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'not a git repository\|gitless\|git-less' .claude/rules/gotchas.md` → 0 hits (absence-of-guard claim — the 0-hit result IS the evidence; the documented no-git prose lives in `pod-side-reporting.md`/CLAUDE.md and covers only the result-PUSH side, not the metadata-shellout crash class) (2026-07-31). Library fix already landed on the issue branch: `issue1902_run.py::_git_sha` + `issue1902_corpus.py::_git_sha` tolerant as of commit 5a3d11f7e21dbad085e674a7cee938f2c3af2a03 (`origin/issue-1902`; lands on main at #1902's Step 10d merge) + pin test `tests/test_issue1902_run.py::test_git_sha_degrades_on_gitless_lane`. Agent-memory twin already on main: `.claude/agent-memory/experiment-implementer/feedback_slurm_gitless_metadata_shellout.md` (commit 266cb63431).

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **The fellows/SLURM materialize_branch_src scratch tree has NO `.git` — a metadata
+   `git rev-parse HEAD` shellout with `check=True` crashes the workload rc=128, even after
+   the phase's science work succeeded.** Provenance helpers degrade: `EPS_GIT_SHA` env →
+   `check=False` subprocess → `"unavailable-no-git-checkout"` literal (canonical sha rides the
+   launch marker + handle sidecar). Class-sweep every git-subprocess site when porting a driver
+   to a SLURM lane — a tolerant commit path beside a strict metadata path is the #1902 shape
+   (fits._commit_eval_results tolerated; _git_sha did not). Worked fix: 5a3d11f7e2 + pin test.
+   (Incident #1902 job 16142, 2026-07-31.)
```

The planner should also consider a one-clause lane caveat on `code-style.md`'s "Reproducibility metadata in result JSONs" bullet (the steering surface).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Secondary (planner's call): `.claude/rules/code-style.md` reproducibility-metadata bullet.
- Grep the workflow surface for the pattern before editing (`grep -rln 'rev-parse' .claude/ CLAUDE.md`) and update every hit that steers metadata helpers; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 64f82be41bfb

Surfaced prose (orchestrator's own observation, /issue 1902 crash-fix cycle 3): "The fellows/SLURM lane's materialize_branch_src scratch tree has NO .git checkout, so ANY reproducibility-metadata helper that shells git rev-parse HEAD with check=True crashes the run on that lane (rc=128) — even after all science work in the phase succeeded. gotcha_candidate: yes."
