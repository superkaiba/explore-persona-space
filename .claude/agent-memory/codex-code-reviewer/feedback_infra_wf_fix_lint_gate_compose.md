---
name: infra-wf-fix-lint-gate-compose
description: Compose recipe for kind:infra wf-fix diffs targeting workflow_lint.py — N/A-by-type gates, hollow-gate = check-registration trace, LIVE_WORKFLOW_HELPERS arming, Step-2-floor attestation
metadata:
  type: feedback
---

Compose recipe for a `kind: infra` wf-fix round whose diff ADDS a check to
`scripts/workflow_lint.py` (recurring shape; first used #2192 r1):

1. **N/A-by-type block up front.** Steps 0.55 / 0.6 / 0.65 / 0.67-exposure /
   0.68-parent are `type:experiment`-only — state the N/A explicitly in a
   compose-time-facts block so Codex never raises `marker-shape` /
   `smoke-run-missing` on their account; the any-diff-type sub-checks
   (0.67 work-conserving, 0.68 hollow-gate + hub-scoping, 0.69–0.72,
   fit-loop line) stay binding.
2. **Hollow-verification-gate sub-check MAPS to lint-gate diffs:** instruct
   Codex to trace that every round-added check function is DISPATCHED —
   registered in the no-flags default run and/or its `--check-*` flag path —
   quoting the registration/call site. A check defined but never wired is a
   hollow gate (Major `hollow-verification-gate`).
3. **LIVE_WORKFLOW_HELPERS arming:** `scripts/workflow_lint.py` IS on the
   roster (tests/test_ruff_policy.py) — state it as a compose-time fact so
   the Step 0.5 `(c)` ruff-policy-pin field check binds, and have Codex
   verify the roster line itself in the worktree. Roster membership is
   PER-FILE — grep `tests/test_ruff_policy.py` fresh each compose, never
   assume from this memory: #2195 r1 (`scripts/verify_report.py`) was NOT
   on the roster, flipping the pin field to a legitimate SKIP (state THAT
   as the compose-time fact instead, so Codex neither demands the pin nor
   disputes the implementer's SKIPPED line).
4. **wf-fix Step-2-floor attestation** ([[wf-fix-step2-floor-attestation]]):
   probe main for `epm:plan-verify` at compose time and attest
   PRESENT/absent in the prompt — Codex cannot read main-side events.
   NON-wf-fix infra tasks (no `workflow-fix:`/`daily-fix:` title prefix, no
   `wf-fix` tag) get the EXEMPT form: attest "floor check exempt" (+ any
   plan-verify verdict found anyway) so Codex never false-fires
   `step2-floor-skipped` — the rubric's floor check binds wf-fix only
   (#2194 r1: exempt AND 3 plan-verify markers present, attested both).
5. **`epm:results` + ts ≥ 2026-07-15 ⇒ Gate-scope threshold satisfied** line;
   pin-sweep verification adapted to `git -C <wt> grep -n '<literal>' -- tests/`
   (no `select_step9c_tests.py` — no uv env).

**Why:** these five all fired together on #2192 r1; missing any one produces
either a false Codex `marker-shape`/`step2-floor-skipped` FAIL or a narrowed
check (#606 twin-omission class).
**How to apply:** any `kind: infra` round whose diff touches
`scripts/workflow_lint.py` or another guard/lint/verifier workflow helper
(`verify_task_body.py`, `verify_plan.py` are the same class — #2291 r1).

**#2298 r1 (2026-08-22) sharpening — calibration records get a pinned-MB
static-recompute duty:** when the plan licenses FAIL posture on a recorded
pre-edit/post-edit finding-count calibration (the A7 shape), Codex cannot
execute the check — compose the duty as: (i) pre-edit population via
`git -C . show <MB-sha>:<path> | grep -niE '<predicate>'` at the PINNED
merge-base, (ii) post-edit zero via `git grep` at HEAD + HAND-applying the
shipped predicate (window/guard/waiver/scope) to each raw hit, (iii) head
re-derived by reading the resolver's source target directly. Disagreement
either way is substantive. Also: a brief may ORDER plan inlining even when
the worktree copy probes identical to canonical — inline per the brief
(belt-and-braces) and note the probe result in the return.

**#2306 r1 (2026-08-23) sharpenings — SKILL.md fence-binding rounds (a
lint-check diff whose payload is `.claude/skills/**` fence edits under a
"no executable-logic drift" acceptance criterion):**

- **Per-hunk classify duty:** compose an explicit walk of every skill-file
  hunk — (a) pure binding/guard insertion (allowed) vs (b) ANY other
  executable-logic change to fence bodies (substantive, Major+, quote the
  hunk). Also: the FATAL guard must fire BEFORE the first `git -C "$WT"` /
  `cd "$WT"` use in the fence (a guard after first use is hollow for it).
- **Mutation-visibility bar for SHIPPED pin tests:** per test, would it
  FAIL if the binding / guard / annotation token were removed, or the lint
  check unregistered? A test asserting only on a synthetic fixture string
  that does not track the LIVE SKILL.md fence pins nothing — name it.
- **Parser FP/FN duty names concrete shapes:** file-scan scope, indented
  fences, info-string variants, `${WT}` vs `"$WT"` forms, bind-after-use,
  annotation-token honoring — plus the stakes-both-directions line (item 7).
- **plan-verify version nuance:** attest the marker's recorded plan version
  vs the CURRENT plan symlink version when they differ (#2306: PASS recorded
  at v2, plan later amended to v3) — attest exactly what was found, never a
  bare "PASS present".

**Two #2291 r1 (2026-08-22) sharpenings:**

6. **wf-fix detection is TAG-first, not title-first.** #2291's title had no
   `workflow-fix:`/`daily-fix:` prefix, but `body.md` `tags:` carried
   `workflow-fix` (and the Provenance line named the workflow-fix-candidate
   origin) — a title-only probe would have mis-attested "floor exempt" on a
   task whose floor BOUND (an `epm:plan-verify` PASS was present to attest).
   Probe `grep -A3 '^tags:' body.md` + the Provenance line every compose.
7. **Brief-supplied plan-vs-measured numeric discrepancies compose as
   TEST-the-hypothesis duties**, never as attested facts: state the plan's
   count, the measured count, the orchestrator's hypothesis (e.g. label
   transposition in a plan amendment = PLAN defect not code defect), and
   instruct Codex to decide which count belongs to which label FROM THE CODE
   and say whether any acceptance criterion depends on it. Also state stakes
   BOTH directions for verifier-gate diffs: a false PASS ships a broken
   fleet gate, and an over-strict new check arm is itself a fleet-blocking
   false-FAIL class — so over-strictness findings weigh equal to bugs.

**#2336 r1 (2026-08-24) sharpening — donor-citation archaeology for
verbatim-hoist claims:** when the plan pins a donor by (commit, line-range)
for a "verbatim hoist except one change" claim, resolve BOTH at compose time
— the range can match CURRENT origin/main while the pinned COMMIT has the
function at a different line WITHOUT a later-landed guard (#2336: plan cited
`issue2329_run.py:1404-1438 @ 27206c15d9`; at that commit the function sat at
:1256 lacking the OSError→log guard, which landed in a later r3 commit).
Attest the true archaeology, instruct Codex to diff against the CURRENT form
(`git -C . show HEAD:<donor>` — verify worktree HEAD's copy is identical to
origin/main's first), and pre-route the citation mismatch as PLAN imprecision
(at most Minor), never an implementer defect. Without this, the twin either
diffs against the wrong (guard-less) donor and false-FAILs the "one change"
claim, or bloks on an unresolvable line range. Also from #2336 r1: a
brief-supplied "adjudicate these N disclosed deviations" list composes as a
dedicated `## Disclosed-deviation adjudications` output section with one
grounded `**Adjudication (D<k>):** upheld|rejected — <file:line>` line each
(the [[brief-named-concern-adjudication]] pattern, generalized to
deviations); and stakes-both-directions for a fleet lint ratchet gains a
third leg — CRASH-safety (a crash in the check wedges every session's gate,
worse than over-strictness; the #2309 error-mode duty applies to lint-flag
variants too).

**#2336 r2 (2026-08-24) sharpenings — predicate-fix closure rounds on a LIVE
fleet lint gate:** (a) a scan-identity claim ("post-fix allowlist=() scan
per-line IDENTICAL, 208/118 both sides") composes as a bounded SHAPE-GREP
corroboration battery the no-uv twin can run (`git -C . grep` per fixed
shape whose live instance would falsify identity, hand-classify hits) with a
REQUIRED header line `**Allowlist-delta claim:** CORROBORATED | REFUTED |
UNVERIFIABLE` — REFUTED = substantive (frozen allowlist now wrong on a gate
every session runs); UNVERIFIABLE residuals route to a CONCERN row, never
FAIL. (b) An orchestrator-run probe TABLE (n probes vs expected, executed on
round code) inlines as SETTLED facts with the explicit duty "hunt shapes
NONE of them covers" + the over-correction note (which probes show the real
waiver/true-positive still work). (c) A recorded plan-amendment note inlines
verbatim in its own envelope with a REQUIRED `**Plan-amendment coverage:**
COMPLETE | INCOMPLETE` header line; the twin enumerates regex-TEXT changes
from the DIFF HUNKS itself (never the note's own list) — an uncovered
predicate-text change is substantive. (d) Hand-rolled lexer/finditer helpers
added to the fleet no-flags path get a named crash-safety duty (adversarial
line shapes: unterminated string, trailing backslash, quote-in-comment,
triple-quote, `#` inside f-string expr) — a reachable exception is Critical
(the third stakes direction beyond FP/FN). (e) Stats hygiene: re-derive
per-file `--numstat` at compose time; a marker whose per-file split
disagrees while totals agree gets pre-triaged in the prompt as "at most a
Minor report-accuracy note" so the twin doesn't build a blocker on it.

**#2309 r1 (2026-08-23) sharpening — `task.py post-marker`-path gate variant
(diff adds validation on the LIVE marker-posting path, not a lint flag):**
hollow-gate = trace the validator is CALLED from the post-marker handler
with a trigger predicate that actually fires for real `epm:results` /
`epm:experiment-implementation` posts; AND compose an explicit error-mode
duty — a crash on malformed note input inside the validator wedges ALL
fleet marker posting (worse than over-strictness), so Codex tests scoping /
grandfathering / waiver / refuse-vs-warn / crash-safety as five named
hypotheses FROM THE CODE. Also validated: contiguous verbatim rubric
extraction (code-reviewer.md `## Review Protocol` start through end of
Step 6, plus the `## Rules` block) with the Claude Step-7 output schema
EXCLUDED (the Codex marker template supersedes it) — assert
`'### Step 7: Issue Verdict' not in prompt`; and the plan-envelope residue
check must tolerate ONE prose mention of `---BEGIN APPROVED PLAN BODY---`
in the blocked-read paragraph (assert on the END token count + first-BEGIN
position, not `count == 1`).
**#2313 r1 (2026-08-23) sharpenings:** (i) brief-VERBATIM review questions can
embed stale source line refs (`:16272` vs the worktree's actual 16374 for
`_LESSONS_MAX_BYTES`) — keep the question verbatim (it is the brief's
extraction contract) but add a compose-time-fact correction verifying the
VALUE and naming the true line, plus a post-question steer to the substance;
(ii) a `diverged_on_main` pin whose file content probes IDENTICAL to current
origin/main simplifies the duty — state that fact so Codex knows main's whole
concurrent delta is the pinned-sha-vs-merge-base diff and needs no further
main-side walk; (iii) a brief-supplied judgment-call question (headroom-spend
vs wording-tighten) composes with explicit routing: verdict-body answer +
optional `CONCERN::` row, never a FAIL. Also validated again: crash-safety
hypothesis duty (fact-8 stakes-both-directions wording) for a check wired
into the no-flags default run.

**#2584 r1 (2026-08-25) sharpenings — target-script COMPLIANCE round (diff
routes bare Hub calls in `scripts/*.py`; workflow_lint.py itself untouched):**
(a) **Helper-resolution attests need an import smoke, not a `def` grep** —
`retry_transient` is an ASSIGNMENT re-export (`retry_transient =
_retry_upload`, hub.py:1696), so `grep 'def retry_transient'` returns
nothing and a naive attest would either miss it or (worse) prime the twin to
flag a phantom unresolvable import. Run `uv run python -c "from ... import
X; inspect.signature(X)"` in the worktree venv and attest the signature +
the re-export line. (b) **Adopt-then-fix two-commit shape:** when the plan
sanctions adopting an UNTRACKED repo-root stray verbatim (commit 1) then
fixing it (commit 2), the composer runs the `cmp` of the commit-1 blob vs
the out-of-sandbox repo-root copy and attests byte-identity (Codex cannot
reach the repo root); explicitly scope #1805 / bare-target judgments to the
round's FINAL state — commit 1's blob legitimately carries the bare call.
(c) **Scanner recompute for a compliance diff** = hand-apply
`_hf_routing_call_is_wrapped` window/paren-balance/`what=`-exemption
semantics to the realized wrapped shapes — especially any allowed-class
line-wrap deviation from the plan's one-liner (the one place a silent lint
regression hides behind a green in-process claim). (d) **Step 4.5 N/A
adjudication for lint-compliance fixes:** the CI pin is the EXISTING lint
checks — compose it as a hollow-gate-style registration trace (both checks
wired into the no-flags run), not as a missing-test finding.

**#2354 r1 (2026-08-25) sharpening — destructive-JANITOR variant (diff adds a
keep/shield guard to a reaper like `worktree_audit.py`):** stakes gain a THIRD
named direction beyond FP/FN — crash-safety with an explicit failure-DIRECTION
duty: on probe errors the shield must fail CLOSED (keep + unknown-reason
constant), never fail OPEN into a reap, never raise out of the sweep loop.
Hollow-gate trace widens to KWARG DEFAULTS: when the shield threads through a
decision function's new kwarg (`should_remove(followup_shield=False)`), have
Codex grep ALL callers — a production path silently passing the disabling
default is the hollow shape. Execution ban names the janitor explicitly
(`--apply` deletes worktrees; even dry-run spawns fleet-wide git subprocesses).
Also validated again here: #2306 plan-verify version nuance (PASS at v3, plan
amended to v4 — attest both), truthful identical-plan envelope on a brief that
orders inlining, and pinned-round-parent sha-range diff acquisition.

**#2342 r1 (2026-08-24) sharpenings — second #2309-shape round (reconcile-kind
guard on `task.py post-marker`):** (a) the origin-prompt arm of TAG-first
wf-fix detection fired — tags `[]` AND no title prefix, but `origin_prompt`
named the workflow-fix-candidate origin; the floor attestation still bound
(plan-verify PASS present, recorded at v2 with plan later amended to v3 —
attested both, per the #2306 version nuance). (b) Plan-envelope order
assertion direction: the envelope BEGIN is the FIRST occurrence and the
blocked-read paragraph's prose mention is the SECOND — assert
`first_BEGIN < END < second_BEGIN`, never `rfind(BEGIN) < END` (that
inversion false-FAILed the compose validation once). (c) A refusal guard
whose predicate is a SUBSTRING (`"reconcil" in kind`) composes a named
false-positive adjudication list: every kind the fleet's surfaces mention
that the substring catches (here `epm:plan-critique-reconcile` — stdout tag
by design — and `epm:followup-value-critique-reconcile` — documentation
alias the sweep test allows in .md while the CLI refuses posting); a live
surface instructing a post of an allowed-in-docs kind = Critical wedge.
(d) Do NOT have Codex re-run a fleet census over the worktree's `tasks/`
tree (frozen + stale — a differing count is not a finding); compose the
static recompute as hand-applying the predicate to the committed test
parametrization instead.

**#2566 r1 (2026-08-26) sharpening — TEST-PIN-RESTORATION variant (diff
touches ONLY a `tests/` pin file, e.g. adding a new workflow.yaml gate to
`test_gates_full_shape`'s expected set; no lint/guard script edited):** the
full recipe applies with three adaptations. (a) Roster fact flips to the
#2195 SKIP form (`grep -c <test-stem> tests/test_ruff_policy.py` → 0 — a
tests/ file is never on LIVE_WORKFLOW_HELPERS; attest so Codex neither
demands the pin nor disputes the marker's SKIPPED line). (b) The pin-sweep
recount has a characteristic clean shape to attest: changed literals →
self-hit only; the REMOVED literal (old count comment) → 0 hits; round-NEW
literal → no stale pins possible. Changed-path three-grain hits land in
family-sync path strings, the step9c invariant manifest, comments, and
function-NAME artifacts (`test_workflow_yaml_has_...` in
test_circuit_breaker.py) — classify each NON-pin with its line so the twin
can't promote them. (c) Compose an explicit yaml-side kill-criterion duty:
the plan's alternative branch ("if the gate entry is malformed, fix the
yaml") is claimed not to have fired — Codex statically reads the
workflow.yaml entry against the SAME test file's per-gate field asserts and
verifies bucket membership from the code, never the marker. wf-fix floor
bound via TAG-first detection (tags `wf-fix`; title prefix also present);
plan-verify PASS at v1 with symlink still v1 = no version-nuance line
needed. Brief-ordered plan inlining on an identical worktree copy again =
truthful belt-and-braces envelope.

**#2610 r1 (2026-08-26) sharpenings — poller status-token round (diff adds a
new terminal status + an SSH-probe leg to `scripts/poll_pipeline.py`):**
(a) **Pre-existing literal-collision provenance:** a round-new token can
collide with an unrelated pre-existing concept (`phase-done` = issue-2089's
ruff-waiver token in workflow_lint.py, ~59 hits on main;
`tests/test_workflow_lint_phase_done_check.py` pre-exists and surfaces in
the marker's pin-sweep hits + slow-deferral list). Probe
`git cat-file -t origin/main:<test-file>` + grep main at compose time and
state the provenance as a fact, so Codex neither reads the file as round
scope nor blocks on its NOT-RUN; naming ambiguity routes to a CONCERN row
at most. (b) Four named attack axes for this shape: marker-note→SSH
interpolation fence (FULLMATCH allowlist + compose-site re-validation,
both layers), legacy byte-identity of the probe script when undeclared,
threading trace as the hollow-gate mapping (note→extract→probe→verdict,
grep ALL callers for disabling defaults), and a consumer sweep for the new
status token (grep scripts/+src/ status-switch consumers; disclosed
docstring drift adjudicated explicitly). (c) Crash-safety third-leg wording
for an every-tick fleet path: adversarial note-text hypotheses
(empty/None/huge/unicode/multi-token/empty-value). (d) wf-fix floor EXEMPT
+ plan-verify version nuance fired together again (PASS recorded at v1,
plan amended to v3 — attest both, per the #2306 nuance).

**#2610 r2 (2026-08-26) sharpenings — concern-fix round, corridor-raise
legitimacy + count-assert hygiene:** (a) a `SKILL_DOC_SIZE_GRANDFATHER`
corridor raise composes with SETTLED arithmetic INCLUDING the
counterfactual: measure the doc yourself (`wc -c` the worktree file),
compute corridor-max, AND check whether the OLD cap would have tripped the
size>cap arm — when it would NOT (150,928 < old 151_800), the
commit/marker's "the guard MANDATES the raise" is itself an adjudication
point: pre-route compliant-but-overclaimed as Minor report-accuracy,
arm-violating / ratchet-defeating cap as substantive. Hand the guard's
three arms as read targets (size>cap remedy text ~:17713, max-headroom
constant arm, stale-entry hygiene) + `tests/test_workflow_lint_skill_doc_size.py`.
(b) `grep -c` counts LINES not OCCURRENCES — a ledger row carrying the
round SHA in BOTH `summary` and `evidence` (2 rows -> 4 occurrences, one
line each) undercounts to 2; pin every SHA count assert from Python
`str.count` probes of each embed, never from grep -c (cost two rebuild
cycles live). (c) Ledger rows with raised+addressed event pairs inline
verbatim (summary + evidence fields) as the acceptance contract; a
reconciler item with NO ledger row gets a pseudo-id closure line
(#1092-r4 pattern) — all three items in one REQUIRED
`## Reconciler-item closure` section, NOT-ADDRESSED = substantive FAIL,
verified-addressed ledger ids never re-emitted as CONCERN:: rows.

**#2525 r1 (2026-08-27) sharpenings — offender-side invariant-RESTORATION
variant (diff fixes the OFFENDING scripts to re-green three existing fleet
tests; no lint/guard/test file edited):** (a) **A `--map-files` hit list is
a SUPERSET of fragment greps** — the selector maps changed PATHS to tests,
so a marker's map-files list can legitimately contain files with ZERO
fragment hits (here `test_issue1901_opsurface.py`,
`test_shared_vm_thread_caps.py`); pre-triage them as selector-mapped, never
as a stale/unreproducible claim, and reconcile the recount as "every
composer hit either in the marker's lists or RAN locally anyway". (b) **A
"comment-only edits" claim backing a pre-existing-failure disclosure gets a
composer `ast.dump(base) == ast.dump(head)` settle** (git show merge-base
blob → temp file, one `uv run python -c` compare) — hands Step 0.9
`pre-existing-on-trunk` as a settled fact instead of a twin re-derivation.
(c) **Waiver-placement rounds attest the READER's semantics with line
numbers** (`_prod_import_lint_waiver_present` @ workflow_lint.py:9690:
first physical line OR nearest preceding non-blank, blanks skipped, reason
≥ 10 chars) + the one non-standard placement site, so the twin verifies all
N placements against the real predicate. (d) Hollow-gate mapping for
secret-gate insertions = "gate scans the SAME artifact the adjacent call
uploads, on the live path, at EVERY site" — per-site gated-path vs
uploaded-path pairs enumerated in the priorities block. (e) Floor-exempt +
plan-verify-present dual attest fired again (tags `urgent-main-red` only,
origin_prompt names a sibling-task surfacing — not a wf-fix candidate).

**#2610 r3 (2026-08-26) sharpenings — reconciler-FAIL dictated-fix round
(binding reconcile v2 = the acceptance contract, inlined verbatim in its own
envelope):** (a) when the twin's own r2 FAIL was UPHELD, the author-neutrality
fence takes the upheld variant — "neither demand more than the two blockers'
stated text (the reconciler scoped the fix to ONE file — no demands on the
cited sibling step files) nor wave it through"; (b) a marker-(d)-disclosed
cross-file vocabulary tension (arm text says "verifying", the quoted
11-step-7.md route says "uploading", which is not even in the status enum)
composes as a NAMED pre-existing-vs-newly-introduced adjudication with the
enum fact attested — never resolved by the composer, never silently dropped;
(c) a SECOND same-file corridor re-pin in consecutive rounds composes as
adjudicated-precedent confirm-consistent (chronicle-move + headroom-bound +
only-entry-touched checks stay), and note when the marker DROPS the prior
round's "guard MANDATES" overclaim — the honesty delta is worth a line so the
twin doesn't re-raise the settled r2 report-accuracy point; (d) joint-blocker
closure: two reconciler blockers discharging ONE ledger BLOCKER get ONE
status line covering both halves (a defect in either half = NOT-ADDRESSED),
not two lines the forwarder could split.

**#2624 r1 (2026-08-27) sharpenings — dual-target poller-ADVISORY +
verify_plan-check round (diff adds a GPU-state advisory family to
poll_pipeline.py AND a WARN-only c73 to verify_plan.py in one commit):**
(a) **Step 0.72 fired on a compose for the first time** — a zero-GPU-memory
"no CUDA context" advisory IS an idle-family verdict from host GPU state, so
the gate legitimately triggers. Compose it as a REQUIRED named header line
(`**Own-device GPU-verdict adjudication (Step 0.72):** ...`) with deployment
FACTS stated neutrally, never pre-adjudicated: probe runs on the issue's
DEDICATED RunPod pod (whole-pod == assigned devices), mirrors the
pre-existing whole-pod GPU_UTIL idle-advisory probe, actuation is
advisory-only (marker+push, never drain/kill), and the SLURM-lanes-have-no-
probe residual is verifiable from the wiring docstring; name the
`# HOST_WIDE_GPU_VERDICT_EXEMPT:` waiver as the remedy if the twin judges
scoping unsound. (b) **Advisory-only plan constraints compose as a
neutrality TRACE duty** (wiring strictly after the verdict; an escaping
exception is BOTH a neutrality break and the crash-safety third leg;
post-failure-retry vs once-per-phase dedup = double-post AND never-post both
findings) — plus a pre-triage that a sanctioned `recommend_next_interval`
anomaly term needs its plan license checked, not assumed. (c) A marker
per-file "+277/-3" that is really the `--stat` combined churn (numstat
274/3) is the #2336-r2(e) stats-hygiene shape again — pre-triage as Minor
report-accuracy when totals match. (d) Both target scripts on
LIVE_WORKFLOW_HELPERS at once = one roster fact naming both line numbers.
(e) plan-verify version nuance NEGATIVE case: marker verified v2 AND the
symlink IS v2 — attest "no version gap" explicitly so the twin doesn't go
hunting.
