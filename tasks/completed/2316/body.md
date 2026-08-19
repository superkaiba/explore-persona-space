---
title: 'workflow-fix: step9c compare classifies whole-repo SCAN-test nodes per-NODE,
  so a branch ADDING a violation to an already-red scan node reads as pre-existing'
kind: infra
tags:
- step9c-scan-attribution
created_at: '2026-08-15T07:22:04Z'
has_clean_result: false
origin_prompt: 'surfaced by #2314''s orchestrator while fixing a fresh thread-caps
  violation: reading step9c_baseline.py::_bucket_run_failures showed the pristine
  comparison is per-NODE verdict, so for whole-repo scan tests a branch-added violation
  on an already-red node is handed to the session as a prose caveat rather than flagged
  mechanically'
workflow: v1
---
# workflow-fix: `step9c_baseline.py compare` classifies whole-repo SCAN-test nodes per-NODE, so a branch that ADDS a violation to an already-red scan node is indistinguishable from one that adds none

## Goal

Make the Step 9c known-red comparison attribute failures of **whole-repo
scan-style invariant tests** at VIOLATION granularity rather than NODE
granularity, so that a branch which adds a NEW violation to a node that was
ALREADY red on pristine main is mechanically flagged as a new red instead of
being handed to the session as a prose caveat.

## The gap (verified by reading the code, not inferred)

`scripts/step9c_baseline.py::_bucket_run_failures` buckets each failing node:

```python
for node in run_failing:
    if not lv.strippable:
        ctx.pristine_bucket.append(node)
    elif node in lv.known_red:
        if node.file in ctx.diff_linked or node.file in lv.changed_tests:  # MF-3 conjunct
            ctx.pristine_bucket.append(node)  # R5 — never blind-strip
        else:
            _strip_node(ctx, node, via="ledger")
    ...
```

The blind-strip guard is CORRECT and is not the defect: a diff-linked known-red
node is never blind-stripped — it routes to the pristine bucket for a real
`--run-pristine` comparison. **The defect is what the pristine comparison then
compares.** The unit of comparison is the node's PASS/FAIL verdict. When a node
is red on pristine AND red on the branch, the tool concludes "pre-existing" and
emits a human-directed caveat (same file, just above the bucketing function):

```
"pristine-main failure — the branch touches files mapped to this test; "
"confirm the branch does not deepen the pre-existing breakage"
```

For an ordinary test — one node, one behaviour — red-on-both really does mean
pre-existing. But several of this repo's load-bearing invariants are **whole-repo
SCAN tests**: ONE node walks every tracked file and asserts on an accumulated
violation LIST. Canonical instances:

- `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
  — `_scan_targets()` over all tracked `scripts/**/*.py` plus `__main__`-guarded
  `src/.../experiments/` modules, accumulating a `violations: list[str]`.
- `tests/test_no_direct_task_path_construction.py`,
  `tests/test_no_pod_side_task_py_shellout.py`,
  `tests/test_no_dollar_budget_caps.py` — the repo-wide invariant trio, same
  accumulate-then-assert shape.
- the no-flags `workflow_lint.py` bundle nodes, which aggregate many checks.

For these, red-on-both is **exactly** the ambiguous case the caveat names, and it
is the case a reviewer cannot resolve from the verdict alone: "node was already
red" cannot distinguish *red because of someone else's file* from *red because of
someone else's file AND MINE TOO*. The violation list is right there in the
assertion output on both sides, and nothing diffs it.

Net effect: for the highest-value guards in the repo — the ones deliberately
written as single-node whole-repo scans — the gate degrades from a mechanical
verdict to an unassisted human judgement precisely when a pre-existing red is
present. That is fail-OPEN, and it is silent.

## Motivating context (gate identity resolved; leg-level attribution handed to the sibling task)

Filed from task **#2314**, which fixed a fresh violation of the thread-caps
invariant in `scripts/issue2225_fu2_dod_points_fig.py` (module-top
`import numpy as np` with no prior `load_dotenv()`).

**Resolved (2026-08-15, this task's first deliverable — gate IDENTITY only):**
the offender landed on main in `faeb45f5e3` at **2026-08-15T00:11:30Z**, committed
by #2225's `fu2_preimage_alltoken` Step 9a-ter INLINE round — the gate that
applied to that landing was the **Step 9a-ter inline payload lint gate**
(`scripts/inline_lint_gate.py`), not the Step 9c compare. The prior version of
this section speculated the Step 9c node-grain strip was the escape path; that
causal claim is RETRACTED — no Step 9c compare adjudicated this landing.

Supporting timeline (established from the git record + ledger sidecar):

- Task #2289 fixed the SAME invariant on four `scripts/issue2223_*.py` files;
  its fix `cefb2ddfe1` landed on main **2026-08-14T17:07:12Z** — ~7h before the
  offender.
- The Step 9c ledger (`.claude/cache/step9c-baseline.json`) was refreshed
  **2026-08-14T15:22:42Z**, i.e. BEFORE that fix — sha-STALE from 17:07Z onward,
  still recording the node as red (`dirty_code_paths: true`, hence not
  strippable). Any consumer trusting that ledger after 17:07Z was reading a
  stale known-red.

Why the inline gate's mechanics make this landing possible: the #2235 Phase A
node-grain ledger demotion in `scripts/inline_lint_gate.py`
(`load_baseline_ledger`, ~:348; the `pre-existing-on-main (ledger)` labeling,
~:893) demotes a mapped-test failure to non-blocking when the NODE is in the
known-red ledger — the same node-grain blindness this task fixes in
`step9c_baseline.py compare`, sitting in front of a ledger that was itself
sha-stale at the time. **Which specific 9a-ter leg let the offender through is
NOT established here** — that leg-level attribution is the FIRST deliverable of
the sibling `kind: infra` task this task files (D8), which owns the
`inline_lint_gate.py` node-grain demotion fix.

The STRUCTURAL gap above stands on its own regardless: it is read directly off
the `step9c_baseline.py` bucketing code, and the same violation-grain fix is
warranted there even though this particular landing never crossed it.

## Proposed direction (the implementer's call — validate before building)

1. **Recognize the scan-node class.** Either a small registry of scan-style node
   ids, or a structural detector (the assertion message carries a multi-line
   accumulated list). A registry is more honest and reviewable than a heuristic.
2. **Diff the violation SET, not the verdict.** When a node is red on BOTH
   pristine and branch, parse the violation list from each side's failure output
   and compare as sets. Any entry present on the branch but absent on pristine
   is a **NEW** red and routes to the `new` bucket. Entries present on both are
   genuinely pre-existing and strip as today.
3. **Fail loud on unparseable output**, never fail open: if the violation list
   cannot be extracted from either side, keep today's behaviour AND escalate the
   caveat so the ambiguity is visible in the verdict rather than buried in prose.
4. **Prefer the cheap arm first.** Much of the value may be reachable by simply
   printing both violation lists side by side in the compare output with the
   set-difference highlighted — even without automatic bucketing, that converts
   an unassisted judgement into a one-glance read. Consider shipping that arm
   first if the full bucketing change is large.

## Acceptance criteria

1. A regression test that reproduces the class: a scan-style node red on pristine
   for file A, and red on the branch for files A **and** B (B added by the
   branch) is classified **NEW**, not pre-existing. This test must FAIL before
   the fix.
2. The converse stays green: the same node red for file A on both sides, with the
   branch adding no new violation, still strips as pre-existing (no new false
   blockers for sessions carrying an unrelated pre-existing red — this is the
   #1388 fleet-wedge risk and is the primary thing not to break).
3. Unparseable / absent violation output degrades to today's behaviour plus a
   louder verdict line, with a test pinning that it never silently strips.
4. The repo-wide invariant trio and the thread-caps node are covered by whatever
   recognition mechanism ships (registry entry or detector hit), pinned by test.
5. `verify_plan`-clean plan; the no-flags `workflow_lint.py` bundle and the
   `tests/test_step9c_baseline*.py` suite stay green.
6. State explicitly whether the fix changes behaviour for the `not lv.strippable`
   path (a dirty-rooted ledger already routes everything to the pristine bucket,
   so the violation-set diff must apply there too, or the gap survives for every
   dirty-rooted ledger — which is the state the #2314 ledger was actually in).

## Why this is worth a task rather than debt

The scan-style invariants are the repo's cheapest, broadest guards — one node
each, covering every tracked script. They are exactly where a silent fail-open
costs the most, because a violation that slips through lands on `main` and then
reds the gate for **every** concurrent session until someone files a task like
#2314 (or #2289 before it, or #2106 and #1953 before that). Each of those tasks
was a real session's full lifecycle spent re-fixing one import ordering. Closing
the attribution gap moves the catch from "after it reaches main, N sessions
later" to "the authoring session's own gate".

