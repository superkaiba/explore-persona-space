---
title: 'workflow-fix: 9a-ter inline payload lint gate — violation-grain classification
  for whole-repo scan nodes red on pristine main'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T09:52:54Z'
has_clean_result: false
workflow: v1
---
## Provenance

workflow_fix_target: scripts/inline_lint_gate.py
Filed by task #2316 (plan v3 D8) — the Step 9c compare violation-grain fix. That task
retracted its own draft's causal claim against Step 9c compare: the motivating offender
never crossed that gate. THIS gate is the one that applied.

## Goal

Classify whole-repo SCAN-style invariant-test failures at VIOLATION grain in the
Step 9a-ter inline payload lint gate (`scripts/inline_lint_gate.py`), mirroring the
#2316 fix to `scripts/step9c_baseline.py compare` — so a payload that ADDS a new
violation to a scan node already red on pristine main can no longer be demoted to
non-blocking on node identity alone.

## The gap (mechanical target)

The #2235 Phase A node-grain ledger demotion in `scripts/inline_lint_gate.py`:

- `load_baseline_ledger` (~:348) loads the Step 9c known-red ledger at NODE grain;
- the `pre-existing-on-main (ledger)` labeling (~:893) demotes a mapped-test failure
  to non-blocking when the failing NODE ID is in that ledger — regardless of whether
  the payload ADDED a violation to the node's accumulated `violations` list.

For a whole-repo scan test (one node, `violations: list[str]`, red on pristine main),
node-grain demotion is exactly the #2316 blindness: the branch-added offender rides a
node the ledger already lists. #2316 fixed this in `step9c_baseline.py compare` via
`VIOLATION_SET_SCAN_NODES` + `extract_violation_paths` set-diffing (branch-added paths
block; same-set reds still strip; unparseable output degrades to today's behavior plus
a loud warn). The inline gate needs the mirrored treatment — by reusing the #2316
helpers from `step9c_baseline.py`, not a re-implementation — plus the matching
`.claude/skills/issue/SKILL.md` Step 9a-ter prose rule (the SKILL side of the #2235
Phase A demotion contract).

A second, compounding hazard the same incident exposed: the demotion trusts the ledger
sidecar WITHOUT a freshness/sha check — the incident ledger was refreshed
2026-08-14T15:22:42Z and was sha-STALE from 17:07:12Z onward (see timeline). Whether
the fix adds a staleness guard or documents the residual is this task's design call.

## Motivating incident timeline (carried verbatim from #2316 D7)

- Task #2289 fixed the thread-caps invariant on four `scripts/issue2223_*.py` files;
  its fix `cefb2ddfe1` landed on main **2026-08-14T17:07:12Z**.
- The Step 9c ledger (`.claude/cache/step9c-baseline.json`) was refreshed
  **2026-08-14T15:22:42Z**, i.e. BEFORE that fix — sha-STALE after the 17:07Z fix,
  still recording `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
  as red (`dirty_code_paths: true`, hence not strippable).
- The new offender `scripts/issue2225_fu2_dod_points_fig.py` (module-top
  `import numpy as np`, no prior `load_dotenv()`) landed on main at
  **2026-08-15T00:11:30Z** (`faeb45f5e3`), committed by #2225's
  `fu2_preimage_alltoken` Step 9a-ter INLINE round — ~7h AFTER #2289's fix.
- Task #2314 then fixed the offender; task #2316 fixed the Step 9c compare side.

## Acceptance criteria

1. **FIRST deliverable — leg-level attribution:** establish from #2225's
   `fu2_preimage_alltoken` round records (events.jsonl markers, the inline gate's
   certification output, the round's commit trail) WHICH 9a-ter leg let
   `scripts/issue2225_fu2_dod_points_fig.py` through: the node-grain ledger demotion
   (~:893), a mapped-test selection miss (the file never mapped to
   `tests/test_shared_vm_thread_caps.py`), a stale-ledger read, or a skipped gate.
   Record the finding in the task body BEFORE designing the fix — the fix must target
   the leg that actually fired.
2. If (and only if) the attribution confirms the demotion leg (or leaves it
   unexcluded): port the #2316 violation-grain set-diff to the inline gate's demotion
   path, reusing `step9c_baseline.VIOLATION_SET_SCAN_NODES` +
   `extract_violation_paths` (single registry, no drift copy).
3. Same-set reds still demote (the #1388-class non-regression: the inline gate must
   not start blocking payloads on main's own pre-existing scan reds).
4. Unparseable failure output degrades to today's node-grain demotion plus a loud
   warn — never a silent demotion, never a new blocking class on parse failure.
5. Tests pinning 2-4, plus a live-tree pin that the registry import stays wired.
6. SKILL.md Step 9a-ter prose updated alongside (the #2235 Phase A demotion sentence
   gains the violation-grain clause), with a prose-pin test.
