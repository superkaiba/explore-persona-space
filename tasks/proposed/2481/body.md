---
title: 'inline_lint_gate load guard is self-defeating: it measures load1 AFTER its
  own pytest leg, making the gate unreachable on the shared VM'
kind: infra
tags: []
created_at: '2026-08-22T21:15:49Z'
has_clean_result: false
origin_prompt: 'Surfaced 2026-08-22 during a user-chat inline free-analysis round
  on #823: three consecutive INCONCLUSIVE verdicts on the same clean single-file payload,
  each disqualified by load1 measured after the gate''s own pytest leg (25.40 / 63.33
  / 47.27 vs EPM_GATE_LOAD_MAX=20).'
workflow: v1
---
# inline_lint_gate load guard is self-defeating on the shared VM: it measures load1 AFTER its own pytest leg

## Goal

Make `scripts/inline_lint_gate.py` reachable on a chronically loaded shared VM, so an inline
round with a clean payload can obtain a terminal verdict instead of unbounded INCONCLUSIVE.

## The defect

The gate's load guard compares `EPM_GATE_LOAD_MAX` (default 20) against `load1` measured
**after** the gate's own pytest leg has run. But pytest is itself a large parallel load: on
this 32-core VM it added 4–24 to `load1` in three consecutive observed runs.

Measured 2026-08-22 on the #823 inline round, same single-file payload each time
(`scripts/issue823_shared_persona_paired.py`):

| run | load1 at start | after the 300s load-wait | pre-pytest | post-pytest | verdict |
|---|---|---|---|---|---|
| 1 | 67.01 | 21.43 | 21.43 | 25.40 | INCONCLUSIVE |
| 2 | 24.99 | 42.06 | 42.06 | 63.33 | INCONCLUSIVE |
| 3 | 23.58 | — | — | 47.27 | INCONCLUSIVE |

Run 1 is the clearest case: the 300s load-wait did its job and got the box to 21.43, the
gate ran, and its own pytest pushed `load1` to 25.40 — over the threshold, so the verdict
was thrown away. The instrument disqualified its own measurement.

Consequence: the gate is only reachable from a baseline `load1` under roughly 10, because
anything higher plus pytest's own contribution clears 20. On a VM running ~15 concurrent
Claude sessions plus crons that window did not occur once in ~100 minutes of watching
(a dedicated watcher polling for `load1 < 12` timed out after 55 minutes).

Because the guard is a pure `>=` comparison with no attempt bound, a payload that is
genuinely clean has no terminating path: every re-run re-rolls the same dice, and the
gate's advice ("re-run when load drops") is unactionable when load never drops.

## Why this matters beyond one round

`inline_lint_gate.py` is the mandatory gate in front of `guard_root_code_commit.sh` for
EVERY user-chat inline round that writes a repo-root code payload
(`.claude/skills/issue/SKILL.md` Step 9a-ter § Inline payload lint gate, #1388/#1460/#1500).
An unreachable gate pushes every such round toward either an indefinite stall or the
`EPM_ALLOW_ROOT_CODE_COMMIT=1` override — and an override that becomes routine stops being
a signal. That is the failure mode to prevent, not the individual stall.

## Candidate fixes (for the planner; not pre-decided)

1. **Measure the baseline, not the aftermath.** Gate on `load1` sampled BEFORE the pytest
   leg starts (`pre-pytest` is already computed and logged), or on load minus the gate's own
   contribution. The guard's purpose is to know whether the BOX was busy enough to make
   unrelated tests flaky; its own pytest is not evidence about that.
2. **Attribute instead of disqualify.** The gate already knows the payload file set. When
   the pytest leg is red, check whether any failure names a payload file; if none does — and
   especially when the payload has no mapped tests and nothing imports it — that is a clean
   payload-attributed PASS regardless of load. Load only needs to gate the case where a
   payload-named test failed and could plausibly be a flake.
3. **Bound the retry.** After N consecutive load-INCONCLUSIVE runs on an unchanged payload
   (same `list-sha256`), escalate to a distinct self-diagnosing verdict that names the
   override path and the evidence needed, rather than emitting the same unactionable
   "re-run when load drops".
4. **Right-size the threshold to the box.** A fixed `20` on a 32-core VM is under 1 job per
   core once pytest runs. Consider scaling with `nproc`, or measuring queue-length per core.

Fix 2 is the one that removes the whole class; fix 1 is the smallest correct change.

## Acceptance

- A clean single-file payload with no mapped tests and no importers obtains a terminal
  verdict at a realistic shared-VM load (`load1` 20–50), without `EPM_ALLOW_ROOT_CODE_COMMIT=1`.
- A payload that genuinely breaks a mapped test still FAILs, under load and idle alike.
- The load guard still suppresses misattribution of a load-induced flake in a
  payload-NAMED test.

## Provenance

Surfaced during a user-chat inline free-analysis round on task #823, 2026-08-22, after three
consecutive load-INCONCLUSIVE verdicts on the same clean payload. That round's script was
ultimately landed under `EPM_ALLOW_ROOT_CODE_COMMIT=1` with the reason recorded in an
`epm:progress` note on #823, on the evidence that the payload has zero references anywhere in
`tests/`, `src/`, `scripts/`, `configs/`, zero mapped tests from
`select_step9c_tests.py --map-files`, clean `ruff check` + `ruff format --check`, and a
repo-wide `workflow_lint.py` whose only two errors name an unrelated file
(`scripts/issue2378_segb_think_audit.py`).
