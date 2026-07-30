---
title: 'workflow-fix: pod-side results-git push must fetch+rebase (mid-run branch-push
  race)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ad5f3cc580c7
created_at: '2026-07-30T12:51:44Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed incident on #1739 hallu lane (2026-07-30 12:19Z):
  completed 31h run failed at results-git push after orchestrator branch commits advanced
  origin; recovered manually from crash-persist'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed incident on task #1739 (hallu lane, 2026-07-30 12:19Z).

## Goal

Require pod/instance-side results-git pushes to fetch + rebase-onto-origin before pushing (bounded retry), and warn orchestrators that pushing commits to the issue branch while a lane is mid-run races the lane's terminal results-git push into a non-fast-forward workload failure.

## Workflow gap

- **Bug observed:** The hallu lane finished all science (270 cells rc=0, Hub sidecars verified) then exited 1 at the final results-git push: the orchestrator's r10 fix commits advanced origin/issue-1739 past the instance's clone, the driver's push retry detected behind=1 but never fetched/rebased, and the healthy run was classified as a workload crash with poweroff.
- **Why it is a workflow gap:** `.claude/rules/pod-side-reporting.md` already owns "pushing result commits" (LESSONS.md row) but is silent on BOTH halves of this race: (a) the pod-side push recipe has no fetch+rebase-before-push requirement, so any orchestrator branch commit mid-run (a crash-fix on a SIBLING lane — the normal multi-lane pattern) deterministically fails the healthy lane's terminal push; (b) nothing warns the orchestrator that a mid-run branch push races live lanes' result pushes. The cost is a full false workload-crash cycle: exit 1 → crash-persist → poweroff → orchestrator forensics + manual result recovery (~30 min today; the science was 31h).
- **Confidence (emitter):** high
- verified-at-filing: incident record on task #1739 epm:progress v89 (2026-07-30T12:50:44Z) + the crash-persist workload.log at HF issue1739_partial/att-20260729-042950-hallu/ lines "[upload] results-git: push attempt 1 rc=1 behind='1' ... ! [rejected] HEAD -> issue-1739 (fetch first)" / "push attempt 2 rc=1 ... (non-fast-forward)" / "push verification FAILED after retry" → "[startup-script] FAILED rc=1". `grep -in "rebase\|fetch" .claude/rules/pod-side-reporting.md` → no fetch/rebase-before-push requirement in the result-push section (2026-07-30). The racing commits: aff188af67df (07:15Z r10 fix push) + 795f474712 (hot-fix) vs the instance clone at 2026-07-29T04:31Z.
- unverified hypothesis — verify at plan time: whether the driver's push helper is shared across issue drivers (a shared helper under src/ or scripts/) or per-issue (scripts/issue1739_*.py) — the RULE addition binds either way; a shared-helper code fix would be a separate follow-up in the owning module.

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/pod-side-reporting.md (result-push section)
+ - **Fetch + rebase before every pod-side results-git push.** A lane's
+   terminal push races ANY orchestrator branch commit made mid-run (a
+   sibling lane's crash-fix is the normal case): a bare push retry that
+   detects behind>0 but never fetches loses deterministically —
+   non-fast-forward → workload exit 1 → a HEALTHY run crash-persists and
+   powers off (#1739 hallu: 31h of complete science, recovered manually).
+   Push recipe: git fetch origin <branch> && git rebase origin/<branch>
+   (result commits are additive per-lane files; conflicts are near-
+   impossible) && push, bounded 2 attempts; on rebase conflict fall back
+   to crash-persist WITHOUT failing the workload rc (the science is done
+   — exit 0 with a push-failed sentinel field instead).
+ - **Orchestrator side:** avoid pushing to the issue branch while lanes
+   are mid-run when feasible; when a mid-run push is required (a
+   crash-fix relaunch), expect in-flight lanes' terminal pushes to need
+   the fetch+rebase path above.
```

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`. The enforcing code (the issue-driver push helpers) is experiment-side; the rule is the durable surface. Cross-check `.claude/agents/experimenter.md` § launch contract for a one-line pointer if the planner deems it warranted.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes; LESSONS.md row for pod-side-reporting.md already covers "pushing result commits" — update its fires-when text only if the planner deems necessary.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/pod-side-reporting.md
- fingerprint: ad5f3cc580c7

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/pod-side-reporting.md
bug_observed: The hallu lane finished all science (270 cells rc=0, Hub sidecars verified) then exited 1 at the final results-git push: the orchestrator's r10 fix commits advanced origin/issue-1739 past the instance's clone, the driver's push retry detected behind=1 but never fetched/rebased, and the healthy run was classified as a workload crash with poweroff.
why_workflow_gap: pod-side-reporting.md owns result-push guidance but lacks a fetch+rebase-before-push requirement and an orchestrator-side mid-run-push warning; the race turns a completed run into a false workload crash.
proposed_change: Add the fetch+rebase-before-push recipe (+ push-failure-without-workload-failure disposition) and the orchestrator-side race warning to pod-side-reporting.md.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->
