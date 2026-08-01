---
title: 'daily-fix: dispatch preflight staging+flag probes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bd87ebb543d3
- daily-auto-filed
created_at: '2026-08-01T07:05:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): ~15 wasted launch cycles
  across 5 issues on pre-provisioning-discoverable defects the landed argv dry-run
  deliberately excludes: unstaged target-side inputs/HF prefixes, dropped runtime
  env pins, relaunch flags re-derived from prose instead of the handle sidecar, un-re-derived
  machine caps'
workflow: v1
---
# daily-fix: dispatch preflight staging+flag probes

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED H2; miner-1:P1/P6, miner-2:P1/P4/P5, miner-3:P12, miner-4:P10, miner-6:P2). Source sessions: 55419495 (#1739 — ≥7 GCE boxes lost to wrong/missing `USIZES` env, an unstaged `bareq_queries.json`, overwrite/out-root guards, an un-pre-staged pvsynth DV; the session's own marker: "both failures were launch-composition errors"), 1e0de8f8 (#1689 — output mirror present but no input stage AND the capture manifest never on HF; #1345 — omitted `EPM_STORY_CHARACTER_NAME` env pin killed job 16283 in 49 s; un-threaded `--gen-a1-dir` crashed a job on an empty Hub prefix), 3a60e6ee (#1902 — relaunch attempt 6 dispatched twice, missing `--intent` then `--time-budget-hours`, while the full flag set sat in `.claude/cache/issue-1902-handle.json`), a78d71ab (#1946 — attempt 3 burned because dispatch omitted `--rss-cap-gb` on a 128 GB machine, defaulting to the 16 GB VM cap), 879efc0d (#1900 — F1b follow-up leg crashed on an input path the lane clone never materialized; the carry-over gate was not run per-leg).

## Goal

Extend the pre-launch/dispatch preflight in the /issue skill with target-side input-staging existence probes, runtime-env-pin completeness, per-LEG carry-over verification, and verbatim relaunch flags from the persisted handle sidecar.

## Workflow gap

- **Bug observed:** ~15 wasted launch/relaunch cycles across 5 issues where orchestrator-composed dispatch invocations died on defects discoverable BEFORE provisioning: inputs not staged / not on HF for the target lane, runtime env pins dropped, relaunch flag sets re-derived from plan prose instead of the persisted handle, and machine-sized caps not re-derived for the target machine. Each cost a provision + up-to-~50-min staging before failing. (Mix of session-probed crash-log reads and session self-diagnoses per the miner records.)
- **Why it is a workflow gap:** The argv dry-run duty ALREADY LANDED (SKILL.md Step 6b § "Hand-composed phase argv dry-run", #1738; crash-fix-rounds § "Changed-argv relaunch", commit 23551aebfa 2026-07-30) — but it covers argparse + early post-parse validation only, and its disposition table EXPLICITLY judges "a pod/GCE-staged path absent locally" a PASS. So the recurring failure surface — target-side input existence (HF prefixes, lane-clone paths, staged manifests), runtime-consumed env pins, per-leg carry-over inputs, and relaunch-flag fidelity — has NO preflight duty. Miner-1 notes teammate-built legs that WERE argv-dry-run pre-handoff did not fail this way; the residual class is inputs/env/flags, not parse.
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -n -iE 'dry.run|parse_args|argv' .claude/skills/issue/SKILL.md` → argv dry-run block present at L3983-4056 (landed — proposed change (a) of the CONSOLIDATED route is therefore SCOPED OUT); its disposition-table context read: "Nonzero exit AFTER ARGV-PARSE-OK whose message names a VM-only environment gap (a pod/GCE-staged path absent locally …) | Judged pass" — i.e. staged-input existence deliberately excluded. `grep -n 'verify_carryover' .claude/skills/issue/SKILL.md` → 1 hit (L3630, single plan-level invocation; 0 per-leg duty); `grep -n -iE 'per.leg|each leg|every leg' scripts/verify_carryover_inputs.py` → 0 hits. `grep -n -iE 'handle\.json' .claude/skills/issue/SKILL.md` → 9 hits, context read: none states a relaunch-flags-copied-verbatim-from-handle duty. `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md scripts/verify_carryover_inputs.py` eyeballed: 23551aebfa (argv dry-run, landed — excluded from this filing) and dcf37f9746 (#1835 lane-aware carry-over gate — lane-awareness, not a per-leg duty; context of its subject read, module grep above shows no per-leg clause) (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md (Step 6b, adjacent to the argv dry-run block):
+ (a) Staged-input existence probe (REQUIRED before instance-booting
+     dispatch): for EVERY input path/HF prefix the composed chain
+     resolves (incl. env-pointed dirs), probe existence on the surface
+     the TARGET will read (huggingface_hub.list_repo_files for HF
+     prefixes; git-tree-reachability for lane-clone paths) — the
+     argv dry-run's "absent locally = judged pass" carve-out does NOT
+     satisfy this.
+ (b) Env-pin completeness: enumerate os.environ[...] / os.getenv(...)
+     reads in the dispatched driver (grep) and check each against the
+     composed env prefix; a consumed-but-unset pin blocks dispatch.
+ (c) Per-LEG carry-over: run verify_carryover_inputs.py for every
+     follow-up/secondary leg, not only the primary phase.
+ (d) Relaunch flag sets are copied VERBATIM from
+     .claude/cache/issue-<N>-handle.json, never re-derived from plan
+     prose; off-VM relaunches re-derive machine-sized caps
+     (--rss-cap-gb etc.) for the TARGET machine.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 6b pre-launch / dispatch preflight; the relaunch clause also mirrors into `.claude/rules/crash-fix-rounds.md` § relaunch).
- Grep the workflow surface for the pattern before editing (`grep -rn 'argv dry-run\|verify_carryover' .claude/ scripts/`) and keep the landed #1738/#1813 blocks as the anchor — this filing is additive to them.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Do not duplicate the landed argv dry-run duty; extend it (the planner should verify overlap against SKILL.md L3983-4056 with the file open).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: bd87ebb543d3

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)
