---
title: 'daily-fix: never push to a branch live boxes sync mid-run'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8c0ee42750f1
- daily-auto-filed
created_at: '2026-08-01T07:05:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): A mid-run push to the branch
  live GCE boxes git-sync at their upload step, while boxes held local surgery on
  the touched file, killed 3 lanes'' upload legs (#1739, ~5-6 GPU-h re-compose); no
  workflow rule governs orchestrator pushes to a live-synced branch'
workflow: v1
---
# daily-fix: never push to a branch live boxes sync mid-run

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED H4; miner-1:P2). Source session: 55419495 (#1739) — the orchestrator pushed a dispatcher patch commit to the branch that live GCE boxes `git sync` at their upload step, while those boxes carried box-local surgery on the same file; the sync refused on the dirty touched paths and the nlevilker + both sycophancy lanes (216 cells each, fits rc=0) died at their upload legs. Results were recovered from EXIT-trap crash bundles (zero science lost), but the chained compose needed a standalone ~5-6 GPU-h box, and a mid-run SSH patch attempt also failed (inter-box firewall/IAP). The session recorded the incident on the task as self-inflicted ("my byte-identity check was a wrong theory of git").

## Goal

Add a rule that no commit is pushed to a branch that live workers will pull/sync mid-run while any worker holds local modifications to the touched files — pin lanes to a detached SHA at launch, or defer the patch until running workers pass their sync step.

## Workflow gap

- **Bug observed:** A mid-run push to the workers' sync branch killed 3 healthy lanes' upload legs at rc=0-fits completion (probed by the miner via targeted transcript reads). `unverified hypothesis — verify at plan time:` the precise refusal predicate is the lane sync step refusing on ANY dirty touched path regardless of byte-identical content — this is the session's own recorded diagnosis, not re-derived here against the sync implementation.
- **Why it is a workflow gap:** No workflow-surface rule governs pushing to a branch live workers sync mid-run. `.claude/rules/crash-fix-rounds.md` governs the pre-RELAUNCH sync direction (fix present on the remote before dispatch) and `.claude/rules/pod-side-reporting.md` governs the worker→remote result-push direction; the orchestrator→branch push while workers are mid-run with local file surgery is covered by neither, and the failure is silent until the workers' sync step fires hours later.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -i 'push\|sync\|detached SHA\|mid-run' .claude/rules/crash-fix-rounds.md` → hits are the pre-relaunch fix-engaged/sync-direction clauses (L378-441) only, 0 governing a mid-run push to a live-synced branch; `grep -n -iE 'push.*branch.*live\|live box\|mid-run push\|pin.*detached' .claude/rules/gotchas.md` → 0 relevant hits; `grep -n -iE 'push|sync' .claude/rules/pod-side-reporting.md` → hits are the result-push verification contract (#1205, L428+) — worker-side outbound only. Absence in all three targets IS the evidence. `git log --oneline --since='7 days ago' -- .claude/rules/crash-fix-rounds.md .claude/rules/gotchas.md` eyeballed (incl. tonight's route-1 gotchas commit 7bd42ad23f — 4 unrelated entries): no landed fix for this class (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/crash-fix-rounds.md (new subsection near § Changed-argv
relaunch; or gotchas.md if the planner prefers):
+ ### Mid-run pushes to a live-synced branch (BANNED while workers hold
+   local modifications)
+ Never push a commit to issue-<N> (or any ref live workers pull/sync
+ mid-run) touching a file ANY live worker holds locally modified —
+ the worker's sync step refuses on dirty touched paths and the run
+ dies at its sync point (typically the upload leg), hours after the
+ push (#1739: 3 lanes, 216 cells each, killed at upload). Either:
+ (a) pin lanes to a detached launch SHA so mid-run branch pushes are
+     invisible to running workers; or
+ (b) defer the patch commit until every running worker has passed its
+     sync step (enumerate live workers first); or
+ (c) land the fix worker-locally only, and push after the wave drains.
```

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md` (secondary candidate home: `.claude/rules/gotchas.md`; the planner picks one and cross-references).
- Grep the workflow surface for the pattern before editing (`grep -rn 'detached SHA\|sync step' .claude/ CLAUDE.md scripts/`) and check the LESSONS.md index row for whichever rule file gains the clause.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` (and `--check-lessons-index` if a rule file's trigger line changes) passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 8c0ee42750f1

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- fingerprint: (driver-computed; tag authoritative)
