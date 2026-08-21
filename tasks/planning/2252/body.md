---
title: 'workflow-fix: Step 9c reports an external SIGTERM of the unfenced workflow_lint
  subprocess as a lint FAIL'
kind: infra
tags:
- wf-fix
- trigger-dense
created_at: '2026-08-12T20:29:34Z'
has_clean_result: false
parent_id: 2243
origin_prompt: 'Surfaced by the #2243 /issue orchestrator while attributing that task''s
  single Step 9c failure: test_workflow_lint_default_exits_zero failed on assert 143
  == 0 (143 = SIGTERM) while the subprocess''s own output ended in ''workflow_lint:
  PASS''; _run has no timeout; direct probes gave rc=0/PASS/0-FAIL-lines in 455s on
  the payload tree and 452s on unmodified origin/main, proving the behaviour environmental
  and generic to every Step 9c gate.'
workflow: v1
---
# workflow-fix: Step 9c reports an external SIGTERM of the unfenced workflow_lint subprocess as a lint FAIL

## Workflow gap

`tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero` asserts
`result.returncode == 0` on a subprocess it spawns through the module helper
`_run` (`tests/test_workflow_lint.py:111`):

```python
def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_LINT), *flags],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
```

There is **no `timeout=`** and **no signal-vs-verdict discrimination**. When
anything external SIGTERMs that child, `returncode` is `143` (128+15) and the
test fails with a message that dumps the entire lint output — whose **final
line reads `workflow_lint: PASS`**. So a *successful* lint that was signalled
mid-teardown is reported identically to a genuine lint failure, and the only
way to tell them apart is for a human (or agent) to notice that 143 decomposes
to SIGTERM and that the embedded verdict line says PASS.

This file is `slow_tests_selected` for the Step 9c selector, so **every**
task's Step 9c gate on this repo runs it. A single flake turns a green gate red
and costs a full re-run plus an attribution investigation.

## Evidence (task #2243, 2026-08-12)

- Step 9c gate on `086eda2f65`: **rc=1 — 1 failed, 5,563 passed, 12 skipped**
  of 5,576 tests in 3,176.58s (52:56). junit: `errors="0" failures="1"
  skipped="12" tests="5576"`. The sole failure was this test, `assert 143 == 0`.
- The failing subprocess's own captured output ends with `workflow_lint: PASS`.
- Not an OOM (that is 137). Not the gate fence (set at 6600s; the run finished
  at 3,176s). Not a test-level budget — `_run` has none.
- **Attribution probes, run directly on the linter to bypass the pytest
  wrapper:** payload tree `rc=0`, `workflow_lint: PASS`, 0 FAIL lines,
  **455s**. Unmodified `origin/main` in a scratch worktree: `rc=0`,
  `workflow_lint: PASS`, 0 FAIL lines, **452s**. Identical verdicts and
  near-identical runtimes, so the behaviour is environmental and generic, not
  payload-specific.
- **The load-bearing number is the runtime: a ~7.5-minute scan.** Under fleet
  contention (observed load average 27.03, 12 concurrent pytest processes, 44
  users, a foreign session's Step 9c battery under a 9060s fence running
  concurrently) an unfenced 7.5-minute subprocess is a wide target for any
  external reaper. An isolated re-run of this single test blew past a 540s
  ceiling.
- Corroborating symptom in the same gate log: two WARNs recording
  `guard_repo_root_branch.sh` timing out after 20 seconds — the same
  contention, hitting a different fence.

## Why this is worth fixing rather than absorbing

The cost is paid by every task, not by #2243 alone: a red Step 9c gate is a
hard advancement blocker, and distinguishing "signalled" from "lint is broken"
currently requires exit-code arithmetic plus reading to the end of a multi-KB
dumped log. An agent that does not do that arithmetic will either bounce a
clean payload into a spurious fix round or — worse — learn to wave off Step 9c
failures generically, which is the dangerous adaptation.

## Proposals (NOT applied — for the fix session to weigh)

1. **Discriminate signal from verdict.** Treat `returncode < 0` or `>= 128` as
   a distinct outcome from a non-zero lint verdict: fail (or retry once) with a
   message leading `SIGNALLED (rc=143 = 128+15 SIGTERM), not a lint verdict —
   the embedded verdict line reads: <line>`. This alone removes the
   misdiagnosis even if the flake persists.
2. **Give `_run` an explicit generous timeout** (well above the measured ~455s,
   e.g. 1800s) and surface `subprocess.TimeoutExpired` as its own failure mode
   rather than letting an external reaper decide.
3. **Retry once on a signal death** before asserting, since the lint is
   deterministic and side-effect-free.
4. **Reduce the scan cost** so the exposure window shrinks (the scan covered
   937 files). Possibly related to the already-completed #2019 work.
5. Consider asserting on the parsed `workflow_lint: PASS|FAIL` verdict line as
   the primary signal, with the exit code as a secondary check.

Any one of 1-3 fixes the misdiagnosis; 1 is the cheapest and most valuable.

## Closest prior work (dedup)

- **#2019** (completed) — "workflow-fix: scratch-eligibility for the
  workflow-lint scan node in step9c compare". Related surface but a different
  concern: which tree the scan node is eligible to run against, not the
  exit-code interpretation or the missing fence. If a reviewer judges this a
  duplicate fingerprint of #2019, close it and say so.
- **#2128** (proposed) — `spawn_session.py --kill` must TERM the inner claude
  pid. Mentions this test only incidentally; unrelated bug.
- #2239, #1661 — unrelated (red-main script pair; agent-spec-size ratchet).
- **#2039** (completed 2026-08-09) — "daily-fix: inline lint gate defers
  under high VM load". SAME CLASS (a lint gate going red from shared-VM
  contention rather than a real defect) but a different gate and a different
  mechanism: it addressed the INLINE payload gate's timeout-sensitive mapped
  pytest leg, not an external SIGTERM being reported as a lint verdict by
  Step 9c. Its existence is evidence the class recurs and is worth a
  mechanism-level fix rather than another per-incident retry.
- **#2173** (open) — `test_backend_poll` module-mode test fails only in large
  collections and is always classified NEW by the Step 9c oracle. Sibling in
  spirit (a Step 9c flake that reds gates fleet-wide and is invisible to the
  oracle), different test and different mechanism (collection-size dependence
  vs an external signal). A fix session may want to consider whether the
  oracle should learn signal-deaths as a category alongside #2173's case.

## Provenance

workflow_fix_target: tests/test_workflow_lint.py

Surfaced by the #2243 orchestrator while attributing that task's Step 9c
failure. #2243's own payload was cleared by the probes above and was NOT the
cause; it is unaffected by this task beyond having paid the investigation cost.
Filed rather than parked as a chat note per the workflow-fix-on-bug protocol —
this has actually bitten, it is not a predicted risk.
