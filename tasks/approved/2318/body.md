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

## Attribution (acceptance criterion 1 — FIRST deliverable, completed 2026-08-15)

**Verdict: NONE of the four candidate 9a-ter legs fired. The Step 9a-ter inline
payload lint gate never ran on `scripts/issue2225_fu2_dod_points_fig.py` at all** —
the file was committed on a WORKTREE BRANCH and merged at Step 10d, a path the inline
gate is structurally out of scope for. The motivating incident therefore does NOT
attribute to this gate, and the `## Motivating incident timeline` clause "committed by
#2225's `fu2_preimage_alltoken` Step 9a-ter INLINE round" is **incorrect** (inherited
from #2316 plan v3 D8 — see the correction below).

### Evidence

1. **Commit topology — worktree branch, not a repo-root inline commit.**
   `faeb45f5e303318d1c3fe9ca09ed110a94d0638e` has a SINGLE parent
   (`93ae52da6ed5eca00f0e66a8a0ea77b0d2e96baa`), is contained in `issue-2225-fu2`, and
   is **absent from `git rev-list --first-parent origin/main`**. It reached main via
   the merge `dba212f616` — *"Merge remote-tracking branch 'origin/issue-2225-fu2'
   into HEAD"*, 2026-08-14T17:49:42-07:00 = **2026-08-15T00:49:42Z**, i.e. 38 min
   after the branch commit, immediately before the round's
   `epm:same-issue-followup-run` marker at 00:50:56Z. The commit added 4 files
   (the script + a PNG/PDF/meta figure triple).
2. **Certification history shows no gate run.** `inline_lint_gate.py` appends one
   `v1 <epoch> <blobsha> <path>` line per certified path to
   `/tmp/eps-inline-lint-cert-v1.txt` (500-line rolling window). The live window spans
   **2026-08-01T02:29:20Z → 2026-08-14T21:58:15Z** and contains **zero** lines naming
   the offender path — and no line at all after 21:58:15Z, i.e. **2 h 13 min before**
   the 00:11:30Z commit. The gate did not run; nothing was certified.
3. **The round was a Step 9b cheap-band LOOP round, not a 9a-ter inline round.**
   #2225's `events.jsonl` (175 rows) never names `issue2225_fu2_dod_points_fig`
   (grep count 0). The round closes with
   `epm:same-issue-followup-run … source: proposer-9b-cheap followup_label:
   fu2_preimage_alltoken` — the Step 9b cheap-band same-issue follow-up loop
   (worktree + `pod-2225-fu2b`). Its only implementer round
   (`epm:experiment-implementation`, 2026-08-14T17:40:19Z) closed at HEAD
   `11d87b0096` having run the no-flags `workflow_lint.py` leg alone; the figure
   commit landed 6 h 31 min later, during the FOLD phase (between the 23:42:08Z
   pod-phase-CLOSED note and the 00:24:59Z final-critique note) — after the
   implementer round and Step 5 code-review had both closed.
4. **Even had the gate run, the demotion leg could not have fired.**
   `load_baseline_ledger` (`scripts/inline_lint_gate.py:373-377`) returns `None`
   whenever `ledger["main_sha"] != git rev-parse origin/main`. The live ledger pins
   `main_sha = 4df656cb05c3bcc1fb457dbc6e8047bba76da842`
   (`refreshed_at 2026-08-14T15:22:42Z`); that sha is today contained in **no branch**
   (`git branch -a --contains 4df656cb05` → empty — orphaned by a fleet rebase), and
   main's first-parent tip moved continuously across the window (commits at
   17:08:29-07:00 and 17:21:01-07:00 bracket the 17:11:30-07:00 offender commit). The
   ledger would have loaded `None` and the #2235 Phase A subtraction layer would have
   been fully disengaged.
5. **Mapped-test selection was NOT the miss.** On current main,
   `select_step9c_tests.py --map-files` maps
   `scripts/issue2225_fu2_dod_points_fig.py` → `tests/test_shared_vm_thread_caps.py`
   (1 pair, 1 test, recommended-timeout-s=600). The same selector had already mapped
   the sibling `scripts/issue2225_figures.py` to that test on 2026-08-11, where
   #2225's first implementer round caught and fixed the identical violation
   ("Union run 1: 1 failed, 1063 passed — sole failure
   `test_no_new_torch_before_dotenv_vm_entrypoints`").

### Correction to #2316 plan v3 D8

D8 retracted its causal claim against Step 9c compare on the grounds that *"the
motivating offender never crossed that gate. THIS gate is the one that applied."*
The first clause is right; the second is **wrong**. The offender crossed **neither**
gate — it was never subject to a mechanical lint/test gate at all.

### The real gap this incident exposes (surfaced, NOT filed — recursion guard)

On a `kind: experiment` task, a worktree round's **FOLD-phase code commit** (figure /
analysis scripts committed by the orchestrator or analyzer after the implementer
marker and after Step 5 code-review has closed) passes through **no mechanical
lint/test gate**: Step 9c test-verdict does not run for `kind: experiment`, the
Step 9a-ter inline payload lint gate is scoped to repo-root inline-round commits, and
Step 10d merges without a lint or test gate. This session carries
`workflow_fix_target:` and so may not auto-file further workflow-fix tasks
(`.claude/rules/workflow-fix-on-bug.md` recursion guard); recorded here for user / PM
triage.

## Scope decision (autonomous, recorded 2026-08-15)

Criterion 2 is conditional on the attribution "confirm[ing] the demotion leg (or
leav[ing] it unexcluded)". The demotion leg is **excluded as the cause of the
motivating incident** but **unexcluded as a live latent defect**, on two mechanical
facts read from the live file:

- (a) `evaluate` (`scripts/inline_lint_gate.py:885-893`) keys the demotion on node
  identity alone (`node in ledger_failing and node_file not in payload`), so a payload
  ADDING a new violation to a ledger-listed whole-repo scan node is reclassified
  `verdict.reported` — precisely the shape #2316 fixed on the step9c-compare side.
- (b) `load_baseline_ledger` does not read `dirty_code_paths`, so the inline gate
  demotes off a ledger whose reds the Step 9c side treats as not-strippable
  (the live ledger carries `dirty_code_paths: true` with 20 dirty paths).

Exposure is bounded by the sha-pin in (4) above: on this fleet main moves within
minutes, so the ledger loads `None` and the subtraction layer is inert most of the
time — which is why no incident is attributable to it.

**Decision: proceed with criteria 2-6 as a LATENT-DEFECT HARDENING, not an incident
fix**, and state that framing in the plan and the round's marker. The staleness
design call the `## The gap` section flags resolves as **document the residual, do NOT
add a `dirty_code_paths` refusal**: refusing to load a `dirty_code_paths: true` ledger
would disable the subtraction layer whenever the fleet has dirty paths (nearly always)
and thereby newly BLOCK payloads on main's own pre-existing reds — the #1388
fleet-wedge class criterion 3 exists to prevent. The residual is recorded in the
module docstring and pinned by a test asserting today's behavior.

**Merge sequencing:** `VIOLATION_SET_SCAN_NODES` (`scripts/step9c_baseline.py:300`)
and `extract_violation_paths` (`:323`) exist only on `origin/issue-2316`, not on main
(#2316 was at `running`, implementer round 1 posted 2026-08-15T09:56:01Z). Criterion 2
mandates importing them rather than copying, so **#2318 must merge after #2316** —
a Step 10d Guard 5 sibling merge-sequencing hold.

## Attribution addendum — the filed defect does not reproduce; two different defects do

Added after the reproduction probe (2026-08-15). This SUPERSEDES the `## Scope decision`
section's fact (a) and its "unexcluded as a live latent defect" reading.

### Reproduction

`evaluate()` was driven directly (zero side effects — no cert write, no ledger mutation)
with the payload `scripts/issue2225_fu2_dod_points_fig.py`, a mapped-pytest output
reproducing the real failure rendering of
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
(captured from a standalone `pytest -q -rA` run of the same
`assert not violations, (header + "\n  ".join(violations))` shape the test uses at
`tests/test_shared_vm_thread_caps.py:1000-1005`), under all three ledger states:

| ledger state | verdict |
|---|---|
| `None` (sha-stale — the actual 2026-08-15 state) | **BLOCKED** |
| the LIVE ledger rows, as shipped | **BLOCKED** |
| nodeid STRINGS (what the code assumes) | **BLOCKED** |

All three block, with the same two reasons: `payload-naming hit without a parseable
lineno (conservative block)` on the two traceback lines
(`E   scripts/issue2225_fu2_dod_points_fig.py (module-top heavy import at line 12, …)`
and the rewritten `E   assert not ['scripts/…']`).

**The node-grain ledger demotion cannot pass a new violation through.** The demotion
label is computed only on a line matching `FAILED_NODE_RE` (`^FAILED\s+(\S+)`), and with
the gate's `pytest … -q -rA` flags the short-summary line is the BARE nodeid — it never
contains a payload path, so the `if p not in line: continue` filter skips it and the
label is never consumed. Offender attribution lives in the traceback lines, which the
demotion never labels. Criterion 2's precondition ("confirms the demotion leg, or leaves
it unexcluded") is therefore **NOT met** as filed.

### D1 — the demotion is DEAD CODE (type mismatch), HIGH

`evaluate` builds `ledger_failing = {str(n) for n in (ledger.get("failing_tests") or [])}`
(`scripts/inline_lint_gate.py:877-879`) and tests `node in ledger_failing` against a
nodeid string. The live ledger's `failing_tests` rows are **dicts**
(`{'classname': 'tests.test_shared_vm_thread_caps', 'file': …, 'name': …}`), so the set
holds dict reprs and the membership test is **always False**. The #2235 Phase A
subtraction layer in this gate has never demoted anything, for any node. (Probe output:
`node in ledger_failing -> False` against the live ledger.)

### D2 — the demotion's grain is the LINE, not the NODE, MODERATE

Even with D1 corrected, the label attaches to the `^FAILED <nodeid>` line only, while
every offender-naming line is a traceback line. So Phase A's own intent also fails in
the opposite direction: a payload blocked by main's PRE-EXISTING scan red still blocks
whenever the node's traceback happens to name a payload path — the #1388-class
false-block that criterion 3 guards, in the direction the filing did not anticipate.

### Revised scope (supersedes the criterion-2 conditional)

D1 and D2 must land TOGETHER, and the violation-grain classification is what makes that
safe: fixing D1 alone would ACTIVATE a dormant demotion at node grain, which is exactly
the over-demotion criterion 4 forbids. The deliverable becomes:

1. **D1:** read the ledger's dict rows into nodeids (`f"{row['file']}::{row['name']}"`),
   tolerating a string row for forward-compat; a row of neither shape is skipped with a
   loud warn (never a silent no-op).
2. **D2 + criterion 2:** move the classification from the line to the NODE — bucket the
   pytest output into per-node failure blocks, and for a
   `step9c_baseline.VIOLATION_SET_SCAN_NODES` member, demote the node's lines only when
   `extract_violation_paths(<node block>)` contains **no payload path**; a payload path
   in the extracted set means the payload ADDED the violation ⇒ block. This is
   single-sided (no pristine-main re-run needed) and uses the #2316 registry by import.
3. Criteria 3-6 are unchanged and all still bind: same-set reds still demote (3);
   unparseable / empty extraction degrades to today's behavior — i.e. NO demotion, the
   conservative block — plus a loud warn (4); tests pin 2-4 plus a live-tree
   registry-import pin (5); SKILL.md Step 9a-ter prose plus a prose pin (6).

The sha-pin residual and the `dirty_code_paths` design call from `## Scope decision`
stand as recorded (document, do not add a refusal).
