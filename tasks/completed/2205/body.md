---
title: 'verify_plan c46: also enforce the launch runtime requirement (exactly one
  of --workload-cmd / --hydra) when dry-parsing plan-embedded dispatch commands'
kind: infra
tags:
- workflow-fix
- verify-plan
created_at: '2026-08-08T19:35:08Z'
has_clean_result: false
origin_prompt: '#2202 dispatch died rc=2 on the plan-embedded launch command (missing
  --workload-cmd) after c46 + fact-checker A21 both dry-parse-PASSed it; extend c46
  to enforce the exactly-one-of runtime rule.'
workflow: v1
---
## Goal

Extend verify_plan.py check c46 (WARN-only dry-parse of plan-embedded dispatch_issue.py commands via dispatch_issue.build_argparser()) to ALSO validate the launch subcommand's runtime requirement: exactly one of --workload-cmd / --hydra must be present (an empty --workload-cmd '' counts as not provided), mirroring dispatch_issue.py's own post-parse validation (#588). Optionally also flag a --workload-cmd-bearing runpod-lane command missing --execute-workload as an FYI note (the workload would otherwise not start on the runpod lane).

## Why (incident)

#2202 (2026-08-08): plan v2 section 9 embedded 'dispatch_issue.py launch --issue 2202 --intent cpu-bigmem --repo-branch issue-2202 --boot-disk-gb 80 --min-ram-gb 32 --time-budget-hours 12' — no --workload-cmd/--hydra. c46 dry-parse PASSed (argparse accepts the flags; the exactly-one-of rule is post-parse runtime validation), the Phase 1.5 fact-checker A21 verified the same way, and the command then died at dispatch with rc=2 'launch requires exactly one of --workload-cmd / --hydra'. Cost was small (one failed background dispatch) but the same shape recurs in any plan that embeds a provision-only launch command; catching it at plan time is exactly c46's mandate (the #1336 v15 drift class).

## Acceptance

- c46 (or a c46b sibling) WARNs when a plan-embedded 'dispatch_issue.py launch' command parses but violates the exactly-one-of --workload-cmd/--hydra rule; empty-string workload-cmd treated as absent.
- Unit test in tests/test_verify_plan.py reproducing the #2202 v2 command shape (WARN fires) and a compliant command (silent).
- No behavior change for finalize subcommand or non-dispatch fences.

## Provenance

Surfaced by the #2202 orchestrator at workload dispatch (session cf372c0b, 2026-08-08); filed per .claude/rules/workflow-fix-on-bug.md (workflow-surface gap: scripts/verify_plan.py c46).
