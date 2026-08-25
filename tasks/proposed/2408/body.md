---
title: 'verify_plan c46: cover the remaining launch post-parse refusals (execute-workload,
  no-runpod-fallback+runpod pin, env-pin)'
kind: infra
tags:
- workflow-fix
- verify-plan
created_at: '2026-08-20T01:22:45Z'
has_clean_result: false
parent_id: 2205
origin_prompt: '#2205 round-1 code review: two sibling post-parse refusals of the
  #2202 class remain invisible to c46; a third (env-pin) confirmed in the same read'
workflow: v1
---
## Goal

Extend `verify_plan.py` c46's namespace-level drift arms (added in #2205 for the exactly-one-of `--workload-cmd`/`--hydra` rule) to the three REMAINING launch post-parse refusals in `scripts/dispatch_issue.py` `main()` that dry-parse clean but die rc=2 at dispatch: (a) `--execute-workload` with an empty/absent `--workload-cmd` (~line 3629, #909 AC3a); (b) `--no-runpod-fallback` combined with an explicit `--backend runpod` pin (~line 3660, #1997); (c) `--env-pin` with an empty/absent `--workload-cmd` (~line 3643, #1669; the malformed-pin `_parse_env_pins` validation stays out of scope — value-level, not flag-shape). Each becomes a WARN-only namespace-level arm in `_c46_drift_arms` mirroring the runtime expression verbatim, with unit tests per arm (refusing shape WARNs; compliant shape silent), same conventions as the #2205 arm (hasattr dest guards, `finalize` untouched, c46 never FAILs).

## Why (incident)

Same fingerprint class as #2202/#2254 (the #2205 incident pair): post-parse `parser.error` refusals inside `if args.action == "launch":` in `main()` are invisible to c46's `build_argparser()` dry-parse, so a plan embedding such a command PASSes plan review and dies at dispatch. #2205 closed the exactly-one-of instance; the #2205 round-1 code review enumerated (a) and (b) as remaining instances, and (c) was confirmed in the same read of the launch validation block.

## Acceptance

- Three new WARN-only arms in `_c46_drift_arms`, each mirroring its runtime expression (including the `.strip().lower()` backend normalization for (b) and the shared `has_workload_cmd` empty-string semantics for (a)/(c)).
- Unit tests in `tests/test_verify_plan.py`: per arm, a refusing command shape WARNs naming the rule and a compliant shape stays silent; existing c46 tests unaffected.
- c46 stays WARN-only and crash-proof; `finalize` behavior unchanged.

## Provenance

Surfaced in the #2205 round-1 Claude code-review verdict (session cmt0rstzvmuuoxw0u2g5m28sk, 2026-08-20); filed per .claude/rules/workflow-fix-on-bug.md (workflow-surface gap: scripts/verify_plan.py c46 coverage of launch post-parse refusals). Distinct fingerprint from #2205 (different arms, same file — the dedup rule's distinct-bug-same-file case).
