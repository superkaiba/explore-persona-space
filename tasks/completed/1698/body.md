---
title: 'daily-fix: RunPod launch-path branch/teardown contract'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9de741abbbca
- daily-auto-filed
created_at: '2026-07-26T07:06:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The #1689 dispatch bootstrapped
  its pod onto main twice despite an explicit repo-branch flag because the bootstrap
  template hard-sets the branch, finalize then crashed after a manual terminate and
  the handle sidecar had to be moved by hand twice, the R8 experimenter exited at
  its 60 second budget while bootstrap was still running and stranded a half-provisioned
  four-GPU pod, and an experimenter re'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep from task #1689. Four separate
failures on the RunPod launch path in one task, all costing pod cycles.

## Goal

Fail loud when a pod's post-bootstrap branch does not match the requested
`--repo-branch`; make backend `teardown` idempotent for an already-terminated pod so
`finalize` still retires the handle sidecar; state that a RunPod-lane
`dispatch_issue.py launch` takes 25–50 min and must run from the orchestrator's own
`run_in_background` Bash rather than a launch-and-exit experimenter subagent; and have
the experimenter report the run fence read from the instance description rather than
echoing `--time-budget-hours`.

## Workflow gap

1. **Pod bootstrapped onto `main` twice despite `--repo-branch issue-1689`.** Both the
   R8 and R9 dispatches ended with the pod's repo on `main`; the on-pod bootstrap text
   read `BRANCH="main"` and a pod probe read `=== branch main === HEAD a17584b62`. An
   extra experimenter spawn was needed to switch the branch and launch.
   **Premise correction (compose-time probe).** The parked observation read as "the
   template hard-sets BRANCH=main". That is REFUTED: the plumbing exists —
   `src/explore_persona_space/backends/runpod.py:733` does
   `branch = str((spec.extra or {}).get("repo_branch") or "") or "main"`, propagates it
   at `:1074`, and `scripts/bootstrap_pod.sh:52` does
   `BOOTSTRAP_BRANCH="${BOOTSTRAP_BRANCH:-main}"` with `:223` `BRANCH=\"$BOOTSTRAP_BRANCH\"`.
   So `main` is the **fallback**, and the real defect is that `repo_branch` arrived
   empty in `spec.extra` on this path (or `BOOTSTRAP_BRANCH` was not exported to the
   pod), silently degrading to the default. The fix is therefore (a) trace where the
   value is dropped and (b) a fail-loud post-bootstrap assertion — NOT "un-hard-code
   the template", which would be a no-op.
2. **`finalize` crashed after a manual terminate.** `pod.py terminate --issue 1689
   --yes` succeeded, then `dispatch_issue.py finalize --issue 1689
   --skip-confirm-artifacts` raised `PodLifecycleProcessError` from `runpod.py:1545`
   because there was nothing left to terminate. Recovery was a hand `mv` of
   `.claude/cache/issue-1689-handle.json` — needed **twice** (`…finalized`,
   `…finalized2`).
3. **The 60 s experimenter contract killed a live bootstrap.** The R8 experimenter
   subagent bailed at its ~60 s budget while `dispatch_issue.py launch` was still
   running (~25 min RunPod/MooseFS clone). Its background Bash died with it, so
   bootstrap steps 5–11 never ran: the pod sat on `main`, no `/workspace/logs/`, no
   workload. The pod was terminated and the whole dispatch repeated inline in the
   orchestrator's own bg-Bash. Orchestrator verbatim: *"subagent bg-Bashes get killed
   when the subagent exits"*.
4. **Wrong fence reported.** The experimenter's launch report stated a 15 h GCP fence;
   `gcloud describe` showed `maxRunDuration = 604800s = 7 days`. `--time-budget-hours`
   maps to a poller timeout, not the GCE fence. Cost a verification detour and could
   have triggered an unnecessary fence-extend.

- **Confidence (emitter):** high on (1)-(4) as observations; the (1) mechanism is
  corrected above and the planner must re-derive the drop point.
- verified-at-filing: per-target probes —
  `grep -n 'repo_branch' src/explore_persona_space/backends/runpod.py` → hits at
  :315 (docstring), :733 (the `or "main"` fallback), :1074 (propagation);
  `grep -n 'BRANCH=' scripts/bootstrap_pod.sh` → :52 default, :223 assignment. Read in
  context per clause (c): the threading is present, so the "hard-set" premise is
  refuted and restated above. Incident text quoted from #1689's own markers and pod
  probes. Landed-fix history check `git log --oneline --since='7 days ago' --
  src/explore_persona_space/backends/runpod.py scripts/bootstrap_pod.sh
  .claude/agents/experimenter.md` → nothing landing a branch assertion, an idempotent
  teardown, or a launch-duration contract. (2026-07-25)

## Proposed change (refine in planning)

```
+ (1) post-bootstrap assertion: compare on-pod `git branch --show-current` against the
+     requested repo_branch; FAIL the launch loudly on mismatch instead of proceeding
+     on the "main" fallback. Separately trace why spec.extra["repo_branch"] was empty
+     on the #1689 path and fix at the source.
+ (2) backends teardown: treat "no pod recorded / already terminated" as idempotent
+     success AND still retire the handle sidecar, so finalize after a manual
+     terminate needs no hand-repair.
+ (3) experimenter.md + SKILL.md Step 6d.1: a RunPod-lane dispatch_issue.py launch
+     runs 25-50 min and MUST be an orchestrator run_in_background Bash (survives turn
+     boundaries). The experimenter's 60 s launch-and-exit contract applies ONLY to
+     launching a workload on an ALREADY-bootstrapped pod.
+ (4) experimenter.md: report the fence from the instance description
+     (scheduling.maxRunDuration), and label --time-budget-hours as the poller timeout.
```

## Scope / surfaces

- `src/explore_persona_space/backends/runpod.py` (assertion + teardown idempotency) —
  in-scope `src/` per the workflow surface list (the backends router package).
- `scripts/dispatch_issue.py` (where the launch surfaces the mismatch),
  `scripts/bootstrap_pod.sh` (if the export is the drop point).
- `.claude/agents/experimenter.md` + `.claude/skills/issue/SKILL.md` Step 6d.1 for (3)
  and (4).
- The four items are one task because they are one launch path and would otherwise
  serialize four sessions over the same files; the planner may still split the diff.

## Constraints / invariants

- (2) must stay fail-loud for a REAL teardown failure — only the
  already-gone/never-recorded case becomes success.
- (1) must not break the legitimate default: a launch that genuinely wants `main`
  (no `--repo-branch`) is unaffected; the assertion binds only when a branch was
  requested.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  the backends test family (`tests/test_router*.py`, `tests/test_backend_*.py`) stays
  green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py
- fingerprint: 9de741abbbca
- Source: `/daily` 2026-07-25 transcript sweep, session `5c5a89e8` (#1689) @
  2026-07-26T04:08:03Z / 05:08:56Z (branch), 01:53:31Z + 04:20Z (finalize),
  04:05:15Z→04:20:41Z (experimenter), 01:03:32Z (fence).
