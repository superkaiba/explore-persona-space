---
title: 'workflow-fix: Step 9c pristine-compare falsely attributes pre-existing main
  reds to the branch (traceback-flag asymmetry + base_identical not consulted for
  ordinary nodes)'
kind: infra
tags: []
created_at: '2026-08-25T01:44:15Z'
has_clean_result: false
origin_prompt: 'Found by #2537''s Step 9c gate: compare --run-pristine returned new:2,
  both false, one per defect'
workflow: v1
---
## Goal

Stop the Step 9c pristine-compare from attributing PRE-EXISTING main-side reds to the branch
under review. Two independent mechanisms do this today, and #2537's gate hit BOTH in one run —
every one of its `new` entries was false. Defect A is a traceback-flag asymmetry in the scan-node
offender-path diff. Defect B is that the `base_identical` sibling-sync suppression, which exists
precisely for this class, is never consulted for ORDINARY (non-scan) test nodes.

Measured on #2537: `compare --run-pristine` returned `new: 2`, `stripped: 3`,
`indeterminate: false`. Both `new` entries were false — one per defect — while all three strips
were correct. So the instrument's false-positive rate on that run was 2 of 5 reds, and a session
trusting it would have bounced a clean round.

## Evidence

Found during #2537's Step 9c gate (2026-08-24). Reproduced end to end against that round's own
committed gate junit.

The running pristine replay emitted:

```
SCAN-NEW-VIOLATION: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  — branch adds violation path(s) absent on pristine main: ['tests/test_shared_vm_thread_caps.py']
```

In `_resolve_scan_violation_setdiff` (`scripts/step9c_baseline.py:3105`) a non-empty branch-only
path set routes the node into `ctx.new`, which yields rc=1 — i.e. "the branch introduced a new
violation". It did not.

**Root cause.** `extract_violation_paths` (`scripts/step9c_baseline.py:403`) matches rows with

```
^[ \t]*(?:E[ \t]+)?(?:[-*][ \t]+)?((?:scripts|src|tests|configs|docs)/[\w./-]+\.[A-Za-z0-9_]+)
```

anchored at line start, tolerating the pytest `E ` gutter, and admitting `tests/` paths. The two
sides carry different text because they run under different flags:

- branch side — the Step 9c gate runs `-v --tb=short`, so the failure element text contains a
  traceback header line, `tests/test_shared_vm_thread_caps.py:1000: in test_no_new_torch_before_dotenv_vm_entrypoints`;
- pristine side — compare's own per-file commands run `-q --tb=no`, so that line is
  structurally absent.

A traceback header begins with a tracked path, so the anchored regex extracts it as an offender.
`v_branch` then holds a path `v_pristine` cannot contain, and the set difference names the test
file itself as a branch-only violation.

**Decisive reproduction**, real extractor against the real gate artifact
(`/tmp/step9c-junit-issue-2537.xml`, the #2537 gate junit):

| extraction site | paths extracted |
|---|---|
| junit failure `message` attribute | 1 — `scripts/issue2254_firstk_ctxext_sensitivity.py` |
| junit failure element text | 2 — that offender **plus** `tests/test_shared_vm_thread_caps.py` |

The message attribute is clean; only the element text carries the phantom.

## Why the existing guards miss it

- The #2319 anchoring fix removed the quoted-**snippet** and remediation-**prose** classes (a
  later token on a row is no longer scanned). It cannot remove this class: a traceback header
  genuinely begins with a tracked path, so under an anchor-only rule it is indistinguishable
  from an offender row.
- The `assert `-leading strip does not apply — the header does not start with `assert`.
- The `...` elision filter does not apply — no elision marker present.
- The `ctx.base_identical` suppression (#2302/#2296) does not fire. That set is the
  diffed-but-identical Step 5a sibling-sync class, and the flagged file is not in the branch diff
  at all, so it never enters the suppression set — even though it is byte-identical to
  `origin/main` (sha `30b19ba74a725898` on both sides). That gap is worth deciding on
  deliberately: "byte-identical to main" is the property that should defeat a branch
  attribution, and today only the narrower "in the diff and identical" does.

## Defect B — `base_identical` suppression missing for ordinary nodes

Same run, different node: `tests/test_workflow_yaml.py::test_gates_full_shape` was reported
`new` (branch-attributed). It is red on **current `origin/main`**, proven statically:

- `b116c2e872` (2026-08-24 13:12Z) added the `clarify_experiment_ask` gate to
  `.claude/workflow.yaml`, `CLAUDE.md`, and `.claude/skills/clarify/SKILL.md` — but did NOT
  update `tests/test_workflow_yaml.py`, whose `conditional_names` assertion pins SEVEN names.
  Main's yaml now yields EIGHT. The assertion cannot pass on main. (Filed separately as #2566.)
- The pristine oracle is the branch's MERGE-BASE — verified: `merge-base(origin/main,
  issue-2537)` == the oracle checkout `e7a80a12db`, which `origin/main` is **824 commits** ahead
  of, and which predates `b116c2e872`. At that base the gate does not exist, so the test PASSES.
- #2537's branch carries the newer yaml **only** because the Step 5a spec-freshness sync imported
  it: the branch copy is byte-identical to `origin/main`'s (`sha 3ae66a74aee9e7a3` both sides),
  and `tests/test_workflow_yaml.py` is byte-identical too (`5bc9336f2845efa6`).

So the branch holds main's OWN content, the oracle predates it, and the delta is attributed to the
branch. That is exactly the false-red class `ctx.base_identical` was introduced for (#2302/#2296:
"a Step 5a sibling-sync copy of main's OWN offender is main's content — blocking the branch for it
would be a false red"). But `base_identical` is consulted ONLY inside
`_resolve_scan_violation_setdiff` (`scripts/step9c_baseline.py:3148`) and for `touched_set`
(`:3275`). An ordinary node's new-vs-strip decision never sees it, so the protection that exists
for scan nodes is absent for every other test.

This will recur on EVERY branch whose Step 5a sync imports a main commit that broke a test after
the branch point — a structural interaction between spec-freshness sync and a merge-base-pinned
oracle, not a one-off.

## Blast radius

Any branch whose gate run reds a REGISTERED scan node inherits this, because the traceback header
always names the test file. So the phantom branch-only path is produced on every such comparison
and each one is a candidate false block — a fleet-level false-block generator, not a #2537 quirk.

It compounds with #2489 (the `cron_step9c_ledger_refresh.sh` mode-100644 defect, ledger 35.8h
stale when measured): reds a fresh ledger would have absorbed as known instead reach the
scan-diff path, where this bug lives.

## Scope

1. Make the extraction flag-invariant. Preferred shape: extract from the junit failure **message
   attribute** only, not the element text — evidenced above as sufficient and inherently
   independent of `--tb` choice. Alternatives, both weaker: drop traceback-header-shaped lines
   (`^<path>:<int>: in <name>`) before extraction, which enumerates one more prose class rather
   than removing the asymmetry; or run the pristine replay with the gate's own traceback flags,
   which restores symmetry at the source but inflates replay output.
2. Decide the `base_identical` question named above: whether a path byte-identical to
   `origin/main` should be suppressed from the NEW set regardless of diff membership. State the
   reasoning either way rather than leaving it implicit.
3. Preserve the loud parse-fail arm. The current design routes an EMPTY extraction to a
   `SCAN-SETDIFF-UNPARSEABLE` warn plus a node-grain verdict, never a silent strip. Any fix must
   keep that property: narrowing the extraction source must not turn a parse failure into a
   silent pass.

## Scope — defect B

4. Consult `base_identical` in the ORDINARY-node new-vs-strip decision, not only in the scan-node
   path: a node whose branch-side failure is produced by files byte-identical to `origin/main`
   must not be reported `new`. Decide deliberately between the two available signals — the
   existing diffed-but-identical `base_identical` set, or the broader "byte-identical to current
   `origin/main`" test — and state which, since the second covers files the branch never touched
   while the first does not.
5. Consider whether the oracle should be the merge-base at all when the branch carries
   sibling-synced content. A merge-base oracle answers "did the branch's own work break this",
   which is the right question only if the branch's tree contains no newer main content; the
   Step 5a sync guarantees it does. Record the reasoning either way — this is the root of defect
   B and a fix that only patches the suppression leaves the mismatch in place.

## Acceptance

1. A branch that touches none of a registered scan node's offenders, on a tree where that node is
   red for a pre-existing reason, produces `verdict: pre-existing` — not `new-violations` —
   regardless of the `--tb` flags either side ran under.
1b. A node red on current `origin/main`, whose branch-side inputs are byte-identical to
   `origin/main`, is NOT reported `new` — even when it passes at the branch merge-base. #2537's
   `test_gates_full_shape` is the regression fixture: branch yaml `3ae66a74aee9e7a3` == main's,
   test `5bc9336f2845efa6` == main's, oracle `e7a80a12db` 824 commits behind.
2. A regression test using the #2537 gate junit as fixture: the two extraction sites provably
   disagree there, so it pins the exact defect. Include the flag asymmetry explicitly rather than
   only the extracted sets, so a future flag change cannot silently re-open it.
3. A branch that genuinely ADDS an offender to a registered scan node still produces
   `new-violations` naming that offender — the true-positive path is not weakened.
4. The empty-extraction parse-fail arm still warns loudly and still declines to strip silently.

## Notes

Filed from #2537, which is itself a Step 9c selector-mapping fix. The two are independent defects
in the same gate's plumbing, and #2537 deliberately did not absorb this one: its acceptance
criteria and deviation bounds were fixed before this surfaced, and widening scope mid-gate is the
planner-introduced-scope anti-pattern its own round-1 critique warned against.

Related: #2550 (the `scripts/issue2254_firstk_ctxext_sensitivity.py` red that is the genuine
offender here, and the fourth instance of the directory-scanner mapping gap), #2489 (stale ledger
refresh), #2553 (the `.gitleaksignore` Step 5a sync coupling, also found by #2537).
