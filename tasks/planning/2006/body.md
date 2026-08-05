---
title: 'daily-fix: pod-side-reporting: hand-rolled box teardown leg'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2becc326b317
- daily-auto-filed
created_at: '2026-08-02T07:13:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1739 hand-rolled GCE gap-script
  box finished (results on HF) but no teardown fired on the done path — lingered RUNNING
  ~1h at eps/phase=done, idle A100 billing; a second box needed manual finalize. Rule
  covers only dispatcher-rendered lanes'' EXIT-trap/poweroff.'
workflow: v1
---
# daily-fix: pod-side-reporting: hand-rolled box teardown leg

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C18 (miner-1 P8 + miner-3 P3; sessions 55419495 + f98a12ed, #1739).

## Goal
Add a clause to `.claude/rules/pod-side-reporting.md`: hand-rolled / inline (non-dispatcher-rendered) GCE box scripts MUST end with the same poweroff/teardown leg the dispatcher-rendered workloads carry — a done-but-billing box is the billing-adjacent failure this closes.

## Workflow gap
- **Bug observed:** On #1739, the hand-rolled gap-script box `gap1nulldiag` finished its workload (24-entry diagnostic on HF) but no teardown fired on the done path — "it's just lingering RUNNING (idle A100 billing, teardown not fired)"; it sat ≈1h at `eps/phase=done` before manual reaping. A second box (`newarma5evil`) was separately found done-but-billing and needed a manual finalize (remove-tag → finalize → re-add keep-running). `unverified hypothesis — verify at plan time: the gap1 box script omitted the EXIT-trap/poweroff leg entirely (miner-inferred, not probed — the box script was not read).`
- **Why it is a workflow gap:** The dispatcher-rendered GCE workloads carry the EXIT-trap → crash-persist → poweroff chain, but nothing in the rule surface requires a hand-rolled/inline box script to carry an equivalent teardown leg — the rule covers the rendered lanes only, so a hand-composed script silently loses the billing bound.
- **Confidence:** medium
- verified-at-filing: `grep -in 'hand-rolled\|box script' .claude/rules/pod-side-reporting.md` → 0 hits (no clause exists); `grep -in 'teardown\|poweroff\|EXIT.trap' .claude/rules/pod-side-reporting.md` → hits only inside lane-mechanics prose (GCE push-verify backstop ~line 528, SLURM teardown gate ~561) — none is a composition mandate for hand-rolled scripts; `git log --oneline --since='7 days ago' -- .claude/rules/pod-side-reporting.md` → 4 commits (b048109f98, 41cbfa3dbe, 3606a80892, 6359721e06), none touches teardown-leg composition (2026-08-02 UTC).

## Proposed change (refine in planning)
New bullet in `.claude/rules/pod-side-reporting.md` (near the (re)launch guidance the rule already carries):

```
+ **Hand-rolled / inline box scripts carry the teardown leg (#1739 gap1nulldiag).**
+ Any hand-composed GCE box workload (not rendered by dispatch_issue.py) MUST end
+ with the same done-path teardown the rendered workloads carry: set eps/phase=done,
+ then poweroff (and on crash, the crash-persist → phase=failed → poweroff tail).
+ A workload that uploads its results and exits WITHOUT poweroff leaves the box
+ RUNNING and billing until the janitor's fence — a done box billing ≥1h is this
+ clause's incident signature.
```

Planner also updates the LESSONS.md trigger row for this rule if wording needs to name hand-rolled launches (it already fires on "(re)launching ANY detached pod/VM workload").

## Scope / surfaces
- Primary target: `.claude/rules/pod-side-reporting.md`

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes (incl. `--check-lessons-index` if the trigger row changes); ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 2becc326b317
- workflow_fix_target: .claude/rules/pod-side-reporting.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C18.
