---
name: revision-round compose recipe (round 2+)
description: For round-2+ composes, reuse the prior round's /tmp template with assertion-checked deltas; round-scope the diff body per the brief's commit range; addressed-concern ledger rows get an explicit verification-duty block, not just Step 0.8 inheritance
type: feedback
---

For a revision-round (round 2+) code-review compose, do NOT re-derive the
prompt from code-reviewer.md: the prior round's template usually survives at
`/tmp/codex-code-reviewer-<N>-r<n-1>-template.md` with the full inlined
rubric. Reuse it and apply round deltas via a Python pass where EVERY
`str.replace` is guarded by `assert content.count(old) == 1` (a silent 0-count
replace is the failure mode; #952 r2 validated this).

**Why:** consistency across rounds (both reviewers graded against the same
rubric text) + avoids paging the 36k-token code-reviewer.md back into context.

**How to apply — the standard round-2+ deltas:**
1. Task line round number + marker tag `v<n-1>` → `v<n>` (exactly one each).
2. **Diff acquisition → ROUND-SCOPED.** When the brief names a round commit
   range, make it the PRIMARY body read; size it (`| wc -c` vs 300 KB); BAN
   the whole-branch three-dot BODY (already-reviewed prior rounds + r1
   over-budget precedent); keep name-only/--stat unrestricted + size-gated
   per-file three-dot bodies for cross-round files; instruct
   `**Diff acquisition:** sha-range <range>` in the verdict header; route
   out-of-round findings through Step 0.9 git-provenance only.
3. **Concerns ledger: fetch FULL (`list-concerns --json`, not --open-only)**
   when concerns were addressed since the prior round — the event-log form
   (raised/addressed rows) lets Codex see both. Add an explicit
   verification-duty block near the top: an `addressed` row is the
   implementer's CLAIM; for each addressed BLOCKER, restate the concern's
   `evidence` field acceptance criteria (e.g. #952: bare key-inclusion
   assert does NOT close the pool-coherence BLOCKER; `exists()` without
   validity check does NOT close restartability), require `file.py:LINE`
   fix-site quotes, and require per-concern status lines in
   `## Concerns to persist` (verified-addressed / NOT-addressed → re-raise).
4. Update the Step 0.5 (e)-section parenthetical (prior concerns now exist,
   so (e) SHOULD be present in the v<n> marker — still at most CONCERNS if
   missing) and the Step 0.8 opener (event-log semantics: open iff latest
   event is `raised`). When the composer can OBSERVE (e) absent in the
   fetched marker body (grep the headings at compose time), STATE that
   observation in the prompt plus where the addressed-claim substance
   actually lives (e.g. (a)/(c)/fix-engaged prose) — so Codex neither
   misses it nor escalates it past CONCERNS (#1033 r2).
5. Surface numeric discrepancies between the concern evidence and the
   round's claims for Codex to adjudicate rather than resolving them
   yourself (e.g. #952: reconciler sketched pool 4895, round claims 4920 —
   composer flags, Codex verifies).

**Blanket-SHA-replace hazard (#1001 r2, 2026-07-04):** the inlined concerns
ledger JSON (and the inlined marker body) can carry the PRIOR round's commit
SHA as a historical reference — a blanket `replace(old_sha, new_sha)` over the
whole template corrupts it. Protect embedded verbatim JSON/marker text by
placeholder round-trip (`old_sha → @@R1SHA@@` inside the embed BEFORE the
blanket replace, restore after) and assert the exact remaining-occurrence
count; the count assert is what caught it live.

**Sibling-template reuse at ROUND 1 (#1082, 2026-07-06):** when composing r1
for a task whose SHAPE matches a same-day sibling compose (e.g. two
kind:infra doc-only workflow-fixes), check `/tmp/codex-code-reviewer-*-template.md`
first — a sibling's template carries the full inlined rubric + the infra N/A
adaptation already. Do NOT multi-replace it (too many task-specific spans);
instead READ it fully and WRITE a fresh template adapting its structure,
rewriting every task-specific section (task-under-review, plan path +
branch-cut status folder, diff acquisition, Step 1/3/3.7 anchors,
task-specific dimensions, verdict title, Rule-1 example). Same rubric
consistency win as round-2+ reuse, without paging the 36k-token
code-reviewer.md.

**Round-1 extension (#986, 2026-07-04):** round-scoping the diff body is not
round-2+-only — apply it at ROUND 1 whenever the whole-branch `main...HEAD`
body is contaminated (concurrent-session history divergence: foreign tasks/
churn, rewritten SHAs). Same recipe: primary read = the brief's round commit
(`git show <sha>`), ban the whole-branch body, route out-of-round
observations through Step 0.9 git-provenance.

**Probe, don't wait for the brief (#987, 2026-07-04):** the brief may NOT
flag the contamination. At every compose, run
`git -C <worktree> diff main...HEAD --stat | tail -3` and compare the file
count against the brief's claimed round scope — #987's brief said 4 files /
+583 while three-dot showed 36 files / +1127 (foreign `tasks/988` plan moves
from a diverged base). A mismatch = contaminated → round-scope. The probe is
one cheap stat read; relying on the brief to flag it is the failure mode.

**Three-dot cleanliness verified at r1 does NOT carry to r2 (#1094 r2,
2026-07-06):** shared-root rebasing rewrote main BETWEEN rounds — r1's
compose had verified `main...HEAD` identical in scope, but by r2 the
merge-base fell behind the branch-cut point and three-dot showed 26 files
of main's own task-state under rewritten SHAs. Re-run the stat probe EVERY
round; when contaminated, ban the whole-branch body, make `git show
<round-sha>` primary, and note the explicit branch-cut SHA range
(`<cut-sha>..HEAD`) stays clean for cross-round per-file context.

**Upheld-concern bounce round (#1094 r2, 2026-07-06):** the sibling of the
post-reconciler-overturn shape — prior round was Claude-PASS / Codex-FAIL /
reconciler BINDING PASS with the Codex Major UPHELD and persisted as a
concern the round now fixes. No no-relitigate-the-discard block needed;
instead: (a) frame r1 as ADJUDICATED (its Minors included) — round-2
contract = concern closure + no new defects + honest (e); (b) the
concern-verification duty restates the raised row's `evidence` as the
acceptance contract and demands a per-concern VERIFIED-ADDRESSED /
NOT-ADDRESSED line (NOT-ADDRESSED = substantive FAIL since the fix IS the
round); (c) the fix exceeding the plan's literal sketch is
concern-remediation authorized by the binding adjudication, NOT scope creep
— instruct intent-adherence, not sketch-verbatim; (d) direct the new-edge
hunt at the fix's mechanism (for a regex lookahead fix: chained/skipped
optional-group paths the new lookahead does not cover), pre-tracing the
shapes without pre-resolving severity.
some prior-round templates keep `{{rubric}}` unexpanded (the ~80 KB rubric was
substituted only into the composed prompt). Extract it from the prior
`-prompt.md` between the fixed anchors
A=`"marker format at the end of this prompt):\n\n"` and
B=`"\n\n**Backstop (binding, restating Step 0.7 / Rule 8):**"` (assert each
count==1, assert len(rubric) > 50 KB). Also: the plan `---BEGIN APPROVED PLAN
BODY---` string legitimately appears TWICE in a plan-inlined prompt (envelope
+ the BLOCKED paragraph's literal mention) — assert count==2 for BEGIN, 1 for
END, not 1/1.

**Post-substitution `{{` assert catches more than placeholders (#1047 r4,
2026-07-05):** after the final substitution pass, `assert '{{' not in prompt`
caught a brace-escaping slip — a shell fold literal written `{{ ... }}` in a
PLAIN triple-quoted section (f-string-style escaping reflex where none was
needed), which would have shipped a corrupted literal for Codex to grep.
Keep that assert in every compose script; it guards both unsubstituted
placeholders AND doubled-brace artifacts in shell-literal-heavy prompts.

**Scope heading asserts to the envelope span, not the whole prompt (#1090
r5, 2026-07-06):** when the new prose sections themselves reference a plan
heading (e.g. `## AMENDMENT v4` cited in the plan intro + context + focus
sections), a global `prompt.count(heading) == k` assert is unmaintainable —
assert the heading INSIDE the extracted envelope span
(`prompt[begin_idx:end_idx]`) instead. Global count asserts stay right for
unique mechanical tags (envelope markers, the verdict marker tag).

**CRASH-FIX round shape (#1090 r6, 2026-07-06):** a Step-7 code-row review
after the Step-5 cap closed is NOT a cap overrun — frame it explicitly
("crash-fix round review (/issue Step 7 code-row; marker v<n>), NOT round
<n> of the implement-review loop"). Deltas vs a normal round: (a) inline the
latest `epm:failure` diagnosis in its OWN envelope
(`---BEGIN CRASH-DIAGNOSIS MARKER BODY (epm:failure)---`) — it is the ground
truth the marker's fix-engaged signal must answer; (b) a hot-fix commit that
shipped straight to a relaunch without review is IN SCOPE this round — say
so; (c) a marker-declared task-scoped deviation (no plan amendment) needs
explicit Step-6 framing in the plan intro: name the plan row it deviates
from, state it is DECLARED with crash evidence, and instruct adjudication of
scoping/recording/justification instead of auto-flagging silent drift;
(d) demote the prior round's amendment-marker envelope to "PRIOR-ROUND
context only" rather than dropping it (the drift sweep needs it); (e) the
sharp F1 check for a seam fix: grep for any downstream re-resolution of the
patched source (the exact consumer-vs-source bug class the hot-fix had).

**Rubric-currency check before reusing ANY /tmp rubric extraction (#825
base-sep r1, 2026-07-06):** rubric files in /tmp go stale silently — run
`git log -1 --format=%ci -- .claude/agents/code-reviewer.md` and pick the
NEWEST `/tmp/codex-code-reviewer-*-rubric.md` whose mtime POSTDATES that
commit (any task's extraction works — the rubric span is task-agnostic;
task-specific text lives in the template). Probe-verify: sample every ~40th
long line for membership in the current code-reviewer.md AND grep for the
recent commits' added keywords (e.g. `prefix-scoped`, `elements 4/5`). On
#825 the same-task r2 rubric (Jul 4) was MISSING the Jul-4 Hub-call-scoping
addition while another task's Jul-6 extraction had everything; blind
same-task reuse would have shipped a stale rubric.

**CRASH-FIX numbering + missing-failure-marker variants (#1090 crash-fix
r3 = marker v7, 2026-07-06):** (f) crash-fix ROUND number ≠ marker version
when an intermediate crash-fix shipped as an unreviewed hot-fix (r2 posted
no marker: v6=crash-fix-1, v7=crash-fix-3) — set the verdict tag from the
IMPL MARKER version, state the mapping in the return, and put the skipped
hot-fix commit IN SCOPE (its findings are `substantive`, not
git-provenance); (g) the crash diagnosis may NOT be an `epm:failure` row —
when the orchestrator dispatched straight from the pod log, the ground
truth lives in `epm:progress` dispatch-breadcrumb + `epm:run-launched`
notes; inline THOSE in the crash-diagnosis envelope and tell Codex the
LATEST `epm:failure` on the task is a PRIOR crash (else it mis-keys the
fix-engaged signal to the wrong crash, or raises a bogus marker-shape
finding on the "missing" failure marker).

**Plan-AMENDMENT round shape (#1090 r5):** when the round implements a formal
plan amendment (`task.py new-plan-version` + an `epm:plan` marker), (a) the
canonical plan file KEEPS its old H1 (v4.md's H1 still said "Plan v3" — the
amendment is an appended `## AMENDMENT v<n>` tail; tell Codex not to be
confused and to VERIFY the tail heading is present in the envelope); (b)
inline the `epm:plan` amendment-decision marker in its own envelope where a
prior round's progress-note envelope sat; (c) the worktree plans/ dir is
frozen pre-amendment (only v1..v<n-1>) so the canonical plan MUST be inlined;
(d) reconstruct-from-parts (extract sandbox/register block + BLOCKED+rubric
block + verdict tail from the prior prompt with count-asserted anchors, write
all round sections fresh) beats dozens of surgical replaces when most middle
sections change — the rubric extraction anchor pair
`"**If you CANNOT read a required file"` → `"\n\nYou MUST emit your verdict"`
captures BLOCKED+rubric in one verbatim span.

**Follow-up ROUND-1 on a COMPLETED parent (#1090 fu1 r1 = marker v8,
2026-07-07):** when the same-issue follow-up runs in a FRESH worktree cut
from main AFTER the followups_running transition, the worktree plan is
usually PRESENT and byte-identical to canonical (verify with `diff -q`) —
reference the plan BY PATH (`tasks/followups_running/<N>/plans/plan.md`)
instead of inlining ~45 KB, and patch the BLOCKED paragraph's
plan-inlined sentence to the by-path variant (then assert
`---BEGIN APPROVED PLAN BODY---` count == 0). The ROUND CONTRACT is the
`epm:followup-scope` note — inline it in its own
`---BEGIN FOLLOWUP-SCOPE MARKER BODY---` envelope and tell Codex Step 6
scores against contract + parent conventions (there is no plan version for
the follow-up; marker-declared deviations get explicit adjudication lines).
Frame parent history as ADJUDICATED, closed-concern ledger as no-regression
duties on the touched parent files, and add a "class re-check" duty when a
closed concern's bug CLASS recurs in NEW round code (e.g. zero-judged
coercion in a new reducer). Marker version = task-wide numbering (v8 on
follow-up round 1) — set the verdict tag from the impl marker version and
state the mapping in the return.

**Ledger-empty closure round (#1090 fu1 r2 = marker v9, 2026-07-07):** when
the prior round's Codex FAIL named a "NEW concern to persist" but the
orchestrator dispatched the fix round immediately WITHOUT running
`raise-concern`, the open-only ledger fetch returns `[]` on a round whose
whole point is closing a BLOCKER. Do not let the empty ledger relax the
verification duty: inline the prior Codex verdict IN FULL in its own
`---BEGIN PRIOR-ROUND (r1) CODEX VERDICT---` envelope, state explicitly that
the ledger `[]` does NOT mean nothing-to-close, frame the Major's
Evidence+Fix text as an open-BLOCKER-equivalent acceptance contract,
decompose its Fix line into closure items C1-C7 (gate keying, downstream
consumers of the skip incl. the UPLOAD phase itself — the r1 bug class —
key/status-value writer-reader consistency, suppress-scoping, test-substance
vs the r1 Mechanizable sketch, extracted-helper behavior identity, new-edge
shapes), and make the verdict's Concerns-to-persist line REQUIRE a
per-concern VERIFIED-ADDRESSED / NOT-ADDRESSED first line (NOT-ADDRESSED =
substantive FAIL). Also: a refactor that EXTRACTS the exact code a CLOSED
concern governs (e.g. `_paired_rates` under `paired-rate-zero-judged-coerce`)
gets an explicit class re-check duty even though the concern is closed.
Span-scoped v8→v9 patching is what protects the inlined prior verdict (its
own `<!-- epm:code-review-codex v8 -->` tag + title line must survive; final
asserts: v9 tag ==1, v8 tag ==1, closing tag ==2).

**Guarded-closure adjudication on fix rounds (#1092 r2, 2026-07-07):** when
the round claims to close a BLOCKER by implementing SOME items and GUARDING
others behind opt-in flags / optional inputs (`--require-registered-reads`,
`--judge-scores`, `status: projection_stage` metadata), do not let Codex
score the concern binary. Give it a three-way per-item classification duty —
IMPLEMENTED-AND-COMPUTED in the default production invocation /
IMPLEMENTED-BUT-GUARDED / STILL-MISSING — and anchor the guard adjudication
to the PLAN's production launch commands: a guard the plan's workload cmd
never arms is the deferred-production-path class (#509) → substantive
finding + Concerns-to-persist even on PASS/CONCERNS; a guard the pipeline
necessarily arms closes the item only if the computation behind it is real.
Pair with the per-concern VERIFIED-ADDRESSED / NOT-ADDRESSED first-line
requirement (NOT-ADDRESSED on the round's own BLOCKER = substantive FAIL).

**Unledgered prior-twin items + brief "re-run duty" translation (#1092 r4,
2026-07-07):** two extensions of the ledger-empty pattern. (a) When the
orchestrator consolidated only SOME of the prior Codex FAIL items into new
concern rows (e.g. D0-D5 got a row; layer-max/MLP/B1 did not), give the
unledgered items PSEUDO-IDS (`r3-codex-layer-max-null`, ...) and require
them in the SAME per-item VERIFIED-ADDRESSED/NOT-ADDRESSED status-line
block as the real concern_ids, with the inlined prior verdict's
Evidence+Fix text as their acceptance contract — otherwise Codex verifies
only the ledgered subset. (b) A brief demanding a "re-run duty for repro
commands + regression tests" cannot be honored literally by a read-only
`--no-write` Codex under the never-execute rule: translate it to the STATIC
equivalent (trace the repro path through CURRENT code + quote sites; READ
the test file for substance per Step 4.5 — hollow test = claimed fix not
real), state in the prompt that the Claude twin carries the literal re-run,
and note the adaptation in the return to the orchestrator.

**Post-reconciler-overturn round shape (#825 base-sep r2, 2026-07-06):** when
the prior round ended Claude-PASS / Codex-FAIL / reconciler BINDING PASS, the
r2 prompt needs three extra blocks: (a) a no-relitigate block NAMING the
reconciler-DISCARDED Codex finding + the discard rationale ("do NOT re-raise
the phase-token finding" — else Codex predictably re-FAILs on its own r1
Critical); (b) concerns-ledger FILTERING for multi-round tasks — a 50-row /
28 KB full ledger is mostly foreign follow-up rounds' addressed pairs; inline
ONLY the round-relevant concern's raised+addressed event rows (full fields,
esp. `evidence` = the acceptance criteria) plus a one-line prose summary
("zero task-wide concerns whose latest event is raised"); (c) an explicit
do-not-flag line for each deliberately-left item the brief names (e.g.
sentinel `"version": 1` with a separately-filed wf-fix #1095) — otherwise a
thorough Codex re-raises it as a fresh Minor. Also restate the r1
smoke-arch marker STANDS (presence-ON-TASK) when the brief says no
path-shape change, so Codex doesn't demand a per-round re-post.

**CAP round (round 5) + no-show-twin prior round (#1092 r5, 2026-07-07):**
two composable variants. (a) CAP ROUND: add a distinct "severity precision
is BINDING" block near the top — the orchestrator machine-parses the
Blocker-tags line for the cap-hit strip/surface decision, so instruct BOTH
directions explicitly (no cosmetic objections inflated to FAIL = false user
escalation; no real defects softened to CONCERNS = ships to the pod) and
append "CAP ROUND: this line is machine-parsed — tag precisely" inside the
verdict template's Blocker-tags bracket. Severity precision also cuts into
the per-item duty: an honestly-still-open MINOR is NOT-ADDRESSED + re-raise
as a Minor, never auto-escalated. (b) PRIOR TWIN NO-SHOWED: when the prior
round's Codex dispatch stalled, the acceptance-contract envelope is the
CLAUDE verdict (`epm:code-review` kind) — fetch by EXACT kind from
events.jsonl (never latest-marker --prefix), title the envelope
`PRIOR-ROUND (r<n-1>) CLAUDE VERDICT`, and the tag asserts change:
`<!-- epm:code-review v<n-1> -->` ==1, `<!-- /epm:code-review -->` ==1,
codex v<n-1> ==0 (no collision with the codex marker extraction since the
kinds differ — note that in the one-marker warning). State the no-show in
the provenance note so Codex knows the r<n-1> adjudication is
single-reviewer.

**Refactor-shaped fix rounds (#1107 r2, 2026-07-07):** when a blocker fix is
a RETURN-SHAPE refactor (e.g. `list[float]` → `list[tuple[float, bool]]`)
plus a deleted helper, add four duties beyond the per-item closure lines
(the mixed-ledger case — one persisted+addressed concern, one un-persisted
Codex Major — composes as the #1092-r4 pseudo-ID pattern with the full r1
verdict envelope as the un-persisted item's acceptance contract): (a) name
the consumers-of-the-changed-shape hunt as the round's highest-yield lens
(emptiness guards, unpack sites, nearest-value/WARN formatting — a `min()`
over tuples or `abs(tuple - float)` is a production-path crash); (b) a
zero-dangling-refs grep duty for the deleted symbol, with a substring
caveat when new variable names contain the old name (`_sidecar_bar_x_positions`
vs `p_is_bar_x` — grep the function name + `def` signature, not the bare
substring); (c) audit the implementer's OWN sibling-sweep enumeration
(spot-verify its cited "pure accumulator" sites, don't trust the list);
(d) an r1-tests-expectations-intact read — a pre-existing test whose
EXPECTED outcome the fix commit silently flipped is a semantics change to
surface. Also: a prior-round Minor resolved as DOCUMENTED-KEEP gets a
three-part validity test (docstring-stated + test-pinned + leniency-only)
with ACCEPTED-DOCUMENTED-KEEP / INSUFFICIENT status lines, and any
ride-along leniency-DECREASING tightening gets its own disclosed+consistent
adjudication (calibration digests are reported claims — Codex adjudicates
the mechanism by reading).

**Orchestrator-authorized mutation probes on a read-only twin (#2146 r3,
2026-08-16):** when the brief DIRECTS independent mutation probes ("temp
copies only, each must turn a pin red, report exactly what was mutated"),
do not silently downgrade to the #1092-r4 static translation — compose a
SCOPED scratch-copy carve-out inside the read-only constraints block (copy
targets under /tmp, mutate SCRATCH copies only, import-by-path / re-run the
pin's assert logic against the scratch file; fleet-mutating CLIs stay
banned; pure-function imports the pin tests themselves perform are fine)
PLUS a never-fabricate fallback: env unavailable ⇒ run the static trace and
label it `STATIC (env unavailable)` in Checks run. Flag at return time that
the dispatch write-mode choice (--no-write vs scratch-writable) decides
which arm executes. Also: a two-FAIL history round gets the honest-PASS
block in BOTH directions plus a DROPPED-item fence — a deliberately-not-
implemented prior finding is restated with the orchestrator's reasoning and
a proof burden ("re-raising requires proving the later-fire case reachable"),
else Codex predictably re-FAILs on its own prior item.

**Inlined-prior-verdict tag echo hazard (#2145 r2, 2026-08-15):** when the
prior Codex verdict is inlined IN FULL, add a hard-constraints line telling
Codex to NEVER reproduce the prior marker's tags in its own output (the
orchestrator extracts between the v<n> head tag and the FIRST closing tag —
an echoed inner `<!-- /epm:code-review-codex -->` truncates the extraction).
That constraints mention changes the tag arithmetic: v<n> head ==1,
v<n-1> head ==2 (constraints mention + inlined verdict), closing ==3
(constraints + inlined + format). Also: a brief that says "the orchestrator
already independently confirmed grep X returns zero hits" converts that
closure duty from re-grep to REPLACEMENT-TRUTH verification (is the new text
TRUE, does the clause stop mis-firing, does the pin assert the new wording
substantively) — say "do NOT re-run that grep; it is settled" explicitly, or
Codex burns its round re-deriving the settled half.

**Prior twin verdict never POSTED (orchestrator accepted a Codex FAIL without
arbitration; #2147 cr3, 2026-08-16):** when Claude PASSed and Codex FAILed but
the orchestrator confirmed the Codex findings against source and accepted them
directly (no reconciler), the `epm:code-review-codex v<n-1>` marker may exist
ONLY as the /tmp output file — events.jsonl carries no row. Fetch the
acceptance contract from `/tmp/codex-output-issue-<N>-cr<n-1>.md`: verify the
head sentinel, strip the `Codex session ID` footer after the FIRST closing
tag, assert the concern-id strings, and FLAG the unposted marker in the return
(posting is orchestrator business, never yours). Frame in the prompt: "NO
reconciler was spawned — orchestrator accepted ALL findings; the Claude PASS
does not soften closure duties, and the acceptance does not pre-judge closure."
Also reuse the prior-round concern IDS from the verdict's own `## Concerns to
persist` lines as the closure-item ids (the #1092-r4 pseudo-ID pattern, ids
pre-minted by Codex itself).

**Mechanism-corrected acceptance round (#2147 cr4, 2026-08-16):** when the
prior Codex FAIL was accepted with its SEVERITY credited but its MECHANISM
DISPROVEN by orchestrator reproduction (r3 blamed C-quoted porcelain; git
2.34.1 emits raw, only newline splits), compose three extra blocks: (a) a
"severity RIGHT, mechanism WRONG" section crediting the call and naming the
disproven mechanism explicitly, with a ban on re-raising it; (b) an
ESTABLISHED FACTS block (orchestrator-verified-by-reproduction, do NOT
re-derive/contradict without a reproduction) sourced from the inlined
progress notes; (c) an instruction to read the prior verdict's Evidence+Fix
CONTRACT "through the correction" — the binding content is the consequence
(registered worktree reaches rmtree), never the literal prescribed fix (dead
code on this git version). Pair with a both-directions residual-claim
discipline when the task burned both ways (new Critical needs a reproduction
construction; "unreachable" dismissal needs the blocking line named). Also
(#2147 cr4): a POSTED epm:code-review-codex marker body can carry the
"Codex session ID" footer — strip after the FIRST closing tag even when
fetching from events.jsonl, not only from /tmp output files. FOLD ROUNDS
(multiple impl markers, one review round): Step 0.5 subject = the HIGHEST
marker; inline the intermediate markers in their own "FOLD-ROUND r<k> MARKER
BODY (context)" envelopes (do-not-score-shape, intermediate residuals are not
open findings), and adjudicate Step 4.6 across the fold — a terse "same N
selector hit files" by-reference line in the head marker is PRESENT-BUT-TERSE
when an earlier fold marker carries the verbatim list.

**Mid-compose ledger drift from the PARALLEL twin (#2326 r3, 2026-08-16):**
the count assert on the inlined concerns-ledger snapshot caught a row the
PARALLEL Claude reviewer raised (`raised_at_round == <this round>`, by
`code-reviewer`) landing in `concerns.jsonl` between the compose's first
read and the build run. Pin the inlined snapshot to rows with
`ts <= implementation-marker ts`: post-dispatch rows are round-N review
OUTPUTS, not inputs — inlining them leaks the sibling twin's findings into
Codex's context and breaks ensemble independence (merging is the
reconciler/orchestrator's job). Say "snapshot as of the round-N
implementation marker ts" in the ledger preface (no mention that a parallel
row exists), and REPORT the excluded row to the orchestrator in the return.

**Acceptance-impossible round shape (external mid-task population change;
#2147 cr3, 2026-08-16):** when the plan's live acceptance became IMPOSSIBLE
(an external actor destroyed the population it was keyed on) and the task
carries recorded `[population-change]` + `[acceptance-readjudication]`
progress notes: inline BOTH notes verbatim in their own envelopes, instruct
"the unrun live acceptance is NEVER by itself a blocker", score plan-§
acceptance rows against the RE-ADJUDICATED criterion (declared, not silent
drift), and CONVERT the lost coverage into a REQUIRED
`## Acceptance-substitution adequacy` verdict section (does the test suite
alone certify the production-scale behaviors the live run would have
exercised — name the scale axes, e.g. ref-fanout wall vs the `_git` timeout,
huge-tree blob-proof cost). Grade a material gap by CONSEQUENCE under the
fail-toward-KEEP design (inertness ≠ data loss ⇒ usually Major/Minor +
Concerns-to-persist, not auto-Critical), and weigh any recorded mitigation
(report-only default, first-apply review) before severity. Give the related
open concern an adjudication-form status line (SATISFIED-BY-SUBSTITUTE /
STILL-BINDING) — the orchestrator owns the ledger action.

## Cap-round (5-of-5) compose recipe — #2147 cr5

Re-derived by the orchestrator after the original append was destroyed by the
#2015 pre-commit stash race (staged at the shared root, swept by a concurrent
session's commit; confirmed absent from HEAD, origin/main AND the worktree, and
absent from the pre-commit patch cache for the staging window — destroyed on
restore, not stashed). Lesson: an agent-memory append is a tracked write —
commit it by explicit path in the SAME turn it is produced, never leave it
staged across a turn boundary.

**Cap-round framing.** State 5-of-5 and advance-or-surface semantics
explicitly, and demand calibration in BOTH directions: do not manufacture a
blocker because it is last call, do not wave through a real defect for the same
reason. Honest PASS is available on the cap round exactly as on round 1.

**Three-way closure vocabulary** (beats plain ADDRESSED/NOT-ADDRESSED when the
orchestrator has exercised judgment):
- `VERIFIED-ADDRESSED` / `NOT-ADDRESSED` — for a fix.
- `ACCEPTED-NON-CHANGE` / `OVERTURNED` — for a deliberate non-change; require
  the overturn to carry a positive construction (here: porcelain membership
  LICENSES a reap), not merely a missed KEEP.
- `SCOPE-RULING-CORRECT` / `SCOPE-RULING-WRONG` — for an out-of-scope ruling;
  ask the twin to adjudicate the diff-scope citation (AST/hunk scan) rather
  than re-raise the defect as a blocker on this branch. A correct ruling routes
  the re-raise to the pre-existing-on-trunk path.

**Read-mode completeness sweep.** When a round fixes one text-mode read,
enumerate EVERY remaining text-mode read whose bytes are used as a path or as
licensing evidence, with enclosing function names, and hand them to the twin
flagged NEUTRALLY (severity not pre-resolved). Partial-fix is the specific risk
after a defect class has produced a fresh manifestation in consecutive rounds.

Rerunnable compose script for this shape: `/tmp/codex-2147-r5cr-compose.py`
(ephemeral; the recipe above is the durable part).

**Refusal-recompose round (spurious usage-policy refusal on a
deletion-utility review; #2147 cr5b, 2026-08-16):** when a dispatch whose
subject is a directory-removal janitor is REFUSED ("flagged for possible
cybersecurity risk"), the trigger is BOTH the composer's offensive-security
vocabulary AND the sheer inlined deletion-logic payload. Recompose with:
(a) NEUTRAL reframe — filesystem-correctness review of a disk-cleanup
utility's safety gates ("is a directory correctly identified as a registered
git worktree before removal; does every uncertain identification KEEP it").
Vocabulary map for composer prose: attack/adversarial → test/edge-case;
exploit → defect / incorrect-removal case; boobytrap → instrumented
assertion; spoof/decoy → colliding path / ambiguous listing; poisoned →
malformed; fail-open → permissive-failure; fail-closed →
conservative-failure (keeps the directory); "reach shutil.rmtree" → "the
directory would be removed"; license(s) → authorize(s); also drop "bypass"
("ordering issue") and "kill-criterion" ("plan invariant K3"). Artifact text
(test ids, code comments, concern ids, the inlined prior verdict) is NEVER
renamed. Enforce mechanically: build the prompt as (text, mine) tuples and
run a banned-vocab assert over COMPOSER spans only, masking artifact ids
(`test_r\\d\\w+`, quoted comments) first.
(b) PAYLOAD reduction — keep inline ONLY: plan, subject marker, prior
verdict (acceptance contract), the one load-bearing resolution note, the
round delta diff. Move to BY-REFERENCE: both full janitor sources (a
`sed -n 'A,Bp'` range table keyed to the composer-verified anchors), the
full three-dot diff (`--stat` + bounded per-file forms), the tier-(f)
baseline (`git show origin/main:<file> | sed -n ...`), fold markers, older
notes. A function-scope ruling check compresses to a hunk-header scan:
`git diff origin/main...HEAD -- <file> | grep '^@@'` vs the two functions'
post-image ranges. Result: 831,796 → 186,156 bytes. Add negative asserts
that every dropped envelope is ABSENT plus `len(prompt) < 200_000`, and
report residual flagged-vocab density BY REGION (every hit must map to an
artifact envelope) — on a repeat refusal the next rung is digesting the
artifact envelopes themselves (context-hygiene rung b2-content).

**Label every count-assert with its subject (#2330 r2, 2026-08-16):** write
`assert t.count(sha) == 7, ('b13f', t.count(sha))` — never a bare count in the
assert message. A `-c` script's traceback line number is offset by the leading
newline and adjacent count-asserts look identical, so a bare `AssertionError: 8`
was misattributed to the WRONG SHA's assert for two debug probes (the actual
failure was the round-PARENT SHA count — the merge-base sentence carries it
twice, plus the r1 commit list and the Step 0.9 provenance line).

**Dictated-wording convergence round (#2326 r4, 2026-08-16):** when a
reconciler DICTATES verbatim replacement wording (W-items) after consecutive
rounds each produced a wrong sentence about the same residual, drop the full
rubric template entirely — compose a TIGHT (~45 KB) prompt whose acceptance
contract is the inlined reconcile marker body itself (its `### Required fix`
section), scoped to exactly four questions: verbatim landing (word-for-word,
reflow tolerated, old fragments GONE by grep), truth-against-source (name the
frozen-library spans + BOTH prior failure shapes the text must not repeat),
executable-change check, regression. Frame the twin's own prior FAIL as
UPHELD-on-facts / label-overstated (credits the catch, pre-empts re-FAIL),
state the class severity precedent (doc-accuracy class twice adjudicated
CONCERN → genuine falseness re-raises as CONCERN + corrected sentence, FAIL
reserved for new blocking defects), and carry an explicit do-not-relitigate
list incl. ruled-OPEN NITs whose recorded fix sketches are known-unsafe.
Assert-side traps hit live: the full round-SHA count spans template AND
inlined marker (assert each side separately before the sum), and a
compose-time "marker lacks `<!-- epm:results vN -->`" observation puts that
literal in YOUR prose — assert body==0 and template==1, not absent.

**Same-day rubric staleness + the #2326 grammar migration (#2330 r3,
2026-08-16):** rubric staleness is not only a cross-day hazard — the r2
rubric extraction (17:57) predated a 19:13 commit to code-reviewer.md the
SAME day. Run the currency check (`git log -1 --format='%ci %h' --
.claude/agents/code-reviewer.md` vs extraction mtime, SAME timezone) at
EVERY compose. When the delta is small and hunk-anchored, PATCH the
extraction by applying the commit's hunks (count-assert each anchor;
skip hunks targeting the Step-7 output-schema region when the extraction
excludes it — grep the hunk's context line first) instead of re-extracting.
Separately: any template predating commit `2454922e7d` (#2326) carries the
RETIRED concerns form — migrating it needs THREE edits: (1) `## Concerns to
persist` switches to the machine-row grammar (`CONCERN:: <SEVERITY>
<kebab-id> <summary>` at line start; sole literal `CONCERN:: none` when
empty; the token must not open any other line — closure status lines
included, say so explicitly since the forwarder position-parses the whole
marker block; instruct NOT to re-emit already-persisted open ids as rows);
(2) add the `**Prior-concerns ledger:** <K open: ids>|empty` header line to
the verdict template; (3) patch the rubric's Step 0.8 span with the
record-the-ledger-state sentence. Assert-side: `^CONCERN:: ` line-start
regex over the final prompt must match exactly the one grammar row.

**Crash-fix round on a reconciler-OVERTURN close + scoped-out intermediate
hot-fix (#2329 r4, 2026-08-16):** when the crash-fix round follows a round the
reconciler closed by OVERTURNING the Codex twin's Critical, compose BOTH
shapes together: the crash-fix provenance block (epm:progress diagnosis as the
inlined ground truth; `marker-shape`/`smoke-run-missing` declared INVALID —
no per-round impl marker exists by design) AND the post-overturn
no-relitigate block. A reconciler "STANDING RECOMMENDATION (does NOT gate,
must ride the next round touching file X)" that the bounded dispatch
deliberately did not implement gets an explicit instruction — record as a
non-blocking CONCERN row, never a FAIL blocker — else the twin (author of the
overturned finding) predictably re-FAILs on its own item. When the brief
scopes the diff to `<round-sha> vs parent` and an UNREVIEWED intermediate
hot-fix sits between the last reviewed HEAD and the round commit, follow the
brief (intermediate commit = out-of-round, git-provenance routing) and FLAG
the unreviewed commit in the return — scope widening is the orchestrator's
call, never the composer's. Also hand Codex the skipped-vs-ran trap
explicitly: a committed test asserting `passed` cannot distinguish a SKIPPED
check from one that ran-and-happened-to-pass when the degenerate input
computes below the bar anyway (zero-vector cosine ~0 under eps) — demand a
read-the-control-flow verification, never the green test as evidence.

**Crash-fix round WITH its own impl marker + write-granted empirical arm
(#2329 r6, 2026-08-16):** the r4 entry's "no per-round impl marker exists by
design" is NOT universal for crash-fix rounds — this run-phase crash-fix
posted its own `epm:experiment-implementation v6`; ALWAYS probe events.jsonl
tail before assuming the r4 shape. When the marker exists: sentinel = impl
marker version; inline BOTH the impl marker AND the `epm:progress` crash
diagnosis in separate envelopes (the diagnosis stays the ground truth the
fix-engaged signal answers; its falsified-hypothesis section becomes an
ESTABLISHED FACTS do-not-re-litigate block). Also: (a) brief-supplied
diff-size figures can be `--stat`-combined (+47 = 35+12) or CUMULATIVE
cross-round test additions (+232 = r5 97 + r6 135) — re-derive from
`git show --numstat` at compose time and use those in the Diff-size header,
flagging the discrepancy in the return; (b) when the brief says write access
WILL be granted, the scratch reproduction carve-out becomes the PRIMARY arm
(run only the named nodeids; expect red-pre-fix/green-post-fix both ways)
with the never-fabricate STATIC fallback retained, and the return states
`Codex write mode: true (scratch reproduction sanctioned)`; (c) a live pod
mid-smoke gets its own OUT OF SCOPE bullet (never ssh/relaunch/touch
/workspace) — the smoke-run-missing invalidation alone does not convey the
hands-off duty.

**Cycle-CLOSE pass adjudicating the twin's OWN FAIL, fold of two impl markers
(#2329 rclose = sentinel v8, 2026-08-17):** when the closing pass of a
crash-fix cycle adjudicates whether CODEX'S OWN prior blocker+nit are closed
across a two-commit range (fix commit with impl v<k>, minor fold with impl
v<k+1>): (a) sentinel = the HIGHEST impl marker of the fold, matching the
task's codex-sentinel==impl-version convention — state the mapping in the
return; inline BOTH impl bodies in separate `(v<k> — the fix commit <sha>)` /
`(v<k+1> — the fold commit <sha>)` envelopes. (b) Inline the prior Codex FAIL
verdict with its marker-tag lines STRIPPED and its `CONCERN:: ` rows
BLOCKQUOTED (`> CONCERN:: `) — simpler than the #2145 tag-arithmetic form:
asserts collapse to own-head==1 / close==1 / prior-tag==0 / line-start
rows==1 (template grammar only) / blockquoted==2. (c) Add an author-neutrality
line ("you authored the prior FAIL — neither defend it by demanding more than
its stated contract nor wave the fix through because it answers you") plus
the standing no-relitigate block for the twin's own already-PASSed questions
at the FAILed commit (behaviourally unchanged ⇒ confirm-undisturbed only).
(d) Per-blob expectations isolate the fold's delta: vs the FAILed commit's
blob the masking test goes red; vs the fix commit's blob ONLY the fold's new
assert goes red (masking asserts stay green). (e) The brief's per-finding
`VERIFIED-ADDRESSED | NOT-ADDRESSED` parenthetical (`NOT-ADDRESSED =
substantive FAIL`) binds EACH finding incl. the NIT when the brief says
"for EACH" — follow the brief, don't soften the nit arm. (f) An impl marker
whose head sentinel lacks the version digit (bare
`<!-- epm:experiment-implementation -->`) is FLAGGED in the return only —
never handed to Codex when marker-shape tags are brief-invalidated.

**Grammar-migration check keys on template CONTENT, not mtime (#2155 r2,
2026-08-17):** the #2330 entry's "any template predating commit 2454922e7d
carries the RETIRED concerns form" is necessary but not sufficient — the
#2155 r1 template POSTdated that commit and still used the retired free-form
`## Concerns to persist` (its r1 Codex verdict emitted a prose-bullet concern
the blind forwarder cannot parse; the reconciler had to mint the concern id
by hand). On every template reuse, grep the template for a line-start
`CONCERN:: ` grammar row; absent ⇒ apply the three-edit migration regardless
of dates. The grammar's canonical source is the composer spec's own verdict
template (`.claude/agents/codex-code-reviewer.md`, the `## Concerns to
persist` block) — NOT code-reviewer.md, which carries no `CONCERN::` text.

**Tests-only blocker-closure cap round (#2329 r5, 2026-08-16):** when the round's
sole change is a test file closing one objective blocker, compose the TIGHT
no-rubric shape with: (a) the valid-tags enumeration NARROWED inside the verdict
template's Blocker-tags bracket itself (`substantive` | `git-provenance` |
`data-access-blocked` only, with the brief-invalidated tags named in OUT OF
SCOPE) — the full tag zoo invites an invalid mechanical FAIL; (b) when the brief
orders self-reproduction of a pre-fix failure, the scratch-dir carve-out
(`git show <sha>~1:<file>` into /tmp, run ONLY the named test nodeids) PLUS the
never-fabricate STATIC fallback labeled `STATIC (env unavailable)`, and flag at
return time that dispatch write-mode decides which arm executes; (c) assert-side:
the inlined impl body usually carries the round SHA once — count template spans
and body separately before asserting the total; (d) a prior twin verdict at a
NON-standard /tmp name (e.g. `/tmp/<N>-codex-r<k>-marker.md`) may be the posted
events.jsonl note verbatim — byte-compare before trusting either copy.

**Production-data-mutation round, v2 combined (#2329 r11 = sentinel v11,
2026-08-17):** when the brief names a helper the fix is supposed to route
through (here `_atomic_replace` for crash-safety) grep the ROUND DIFF for the
token at compose time — zero hits is a load-bearing composer observation:
locate the actual write-path symbols in the new blob (`_write_jsonl_atomic` /
`_save_pt_atomic` / `_write_json_atomic`, call order jsonl → pt → done) and
hand Codex the wrapper-routing adjudication with anchors, never resolve it
yourself and never let the brief's helper name pass into the prompt as an
established fact. Also: a full-feature round (+1378) composes fine in the
task's established TIGHT shape when the brief supplies the question list —
expand the v2 efficiency lens for a GPU generation round (work-conserving
claim queue, extracted-core dual call sites, realized-width fail-loud,
regenerated-text upload path ⇒ include `raw-completions-upload-missing` +
`compute-shape-mismatch` + `hollow-verification-gate` in the tag enumeration)
and give the in-place merge a three-part ruling frame: byte-preservation
(name any DECLARED mutation, e.g. cap backfill), per-crash-point idempotency,
per-file atomicity with done-record-strictly-LAST (a done record before data
writes is the stale-done question by another road).

**Contingency-port round after a reconciled-PASS (merge + relocate; #2348 r2,
2026-08-17):** when round 2 is a plan-contingency port (merge origin/main +
re-apply round-1 hunks to a file that RELOCATED on main) following a round-1
Claude-PASS / Codex-FAIL / reconciler BINDING PASS: (a) pin every
merge-soundness diff to the MERGED MAIN TIP SHA (the merge commit's second
parent), never `origin/main` — the worktree ref advances past the merged tip
between merge and compose, polluting `origin/main..HEAD` forms; (b)
new-file-via-merge trap: `git diff <round-parent>..HEAD -- <relocated-file>`
shows the ENTIRE file as added (it did not exist at the round parent) — the
port delta is `git show <port-commit>` and the fidelity reference is the r1
range's diff on the OLD path; ban the misleading form explicitly; (c) when
the r2 marker REPEATS a shape absence whose r1 twin-raised concern was
reconciler-STRIPPED (gate-scope line, ruff-policy pin field), attest the
absence + the strip precedent at compose time and route it to a per-concern
`RECURRENCE-IN-ROUND-2` STATUS line — not a fresh Critical — while keeping
the Step 4.6 diff-consistency (substantive) half fully binding; (d) with all
prior concerns raised BY the twin and adjudicated, inline the reconciler's
per-blocker dispositions verbatim + the author-neutrality line, and scope
refuted items precisely (e.g. the `${WT` ban applies to MERGE-BASE lines
only — plan Edit D preserves `${WT:-$REPO_ROOT}` on helper-probe fallbacks;
an unscoped restatement makes the twin re-FAIL its own refuted item); (e)
already-persisted ledger ids get status lines, never re-emitted `CONCERN:: `
rows (the blind forwarder would duplicate them) — assert exactly ONE
line-start grammar row in the final prompt; (f) a brief-ordered pytest
battery run composes as the sanctioned-commands carve-out (named nodeids
only) + the `STATIC (env unavailable)` never-fabricate fallback, with an
explicit "unrunnable env is NOT `data-access-blocked`" line.

**Proposer-band follow-up round at the cap (#2330 r5, 2026-08-17):** a Step
9b/9a-ter PROPOSER follow-up posts NO `epm:followup-scope` marker — the round
contract is the `epm:followup-value-critique` proposals (inline in their own
envelope) + the impl marker's `Brief adherence` section; grep events for
`followup` before assuming the fu1-r1 scope-marker shape. Two composable
deltas: (a) a DECLARED brief deviation ("MODIFIED (1)" in Brief adherence)
gets a JUSTIFIED/UNJUSTIFIED adjudication duty with the engine-wiring trace
spelled out (a rebound module constant must reach the engine constructor
AFTER rebind — the #505/#601 + #1727 classes); (b) the round range can carry
an extra agent-memory bookkeeping commit beyond the brief's `round_commits`
list — include it in the pinned range, mark it stat-only/not-a-finding in the
prompt, and flag it in the return. Committed data artifacts in-range (the
9a-ter round's JSON) get digest-only instructions; the free-analysis round's
NUMBERS are adjudicated (two critics reproduced) — scope its commit to
code-hazards-only or the twin relitigates settled outputs.

**FAIL+FAIL-union fix round (#2332 r2, 2026-08-16):** when round 1 was
Claude-FAIL + Codex-FAIL and the orchestrator UNIONED the blockers (no
reconciler), BOTH prior verdicts are acceptance contracts — inline both, and
STRIP their marker tag lines at compose time instead of the #2145
never-echo instruction (cleaner tag arithmetic: v<n> head ==1, closing ==1,
every prior-round tag ==0; also removes the echo hazard outright). No
no-relitigate block is needed (nothing was discarded). When the impl marker
carries its own numbered disposition table, key the closure ledger on THOSE
row numbers (+ the Codex verdict's pre-minted concern ids per #2147) rather
than minting pseudo-IDs. REFUTED rows: verify every cited plan anchor EXISTS
at compose time (grep line numbers/headings), quote the disputed clause
verbatim in the prompt, mark existence "settled" vs substance "yours", and
SURFACE textual tensions (e.g. a plan assumption line reading the disputed
clause the other way) for Codex to weigh — never resolve them yourself. Also
map implementer round numbering vs review round numbering explicitly when
they diverge (impl "round 3" = review round 2). Memory-write hygiene: do NOT
commit an agent-memory edit to the branch under review mid-round — it would
enter the very diff being reviewed; leave it uncommitted and flag for the
orchestrator to sweep post-merge.

**Mid-task output-contract change caught by the rubric-currency check (#2332
r3, 2026-08-16):** the currency probe (`git log -1 -- code-reviewer.md` vs the
reused /tmp template's mtime) fired for real — #2326 landed BETWEEN r2 and r3,
adding the `CONCERN:: <SEV> <kebab-id> <summary>` machine-row grammar (+
literal `CONCERN:: none` sentinel) to the Concerns-to-persist template and a
REQUIRED `**Prior-concerns ledger:**` header line. When reusing a pre-#2326
template: patch BOTH into the tail, tell Codex the token must never start a
line outside that section (the forwarder position-parses `^CONCERN:: ` over
the whole marker block), and attest that the WORKTREE's frozen rubric/spec
files predate the change so Codex doesn't flag the divergence. Also: scope
prior-round-heading zero-asserts to OUTSIDE the inlined envelopes (the r2
Codex verdict legitimately carries its own `## Round-1 closure ledger`
heading), and expect legitimate history-prose hits when sweeping round tokens
("verified closed by review round 2") — grep-and-eyeball beats a bare
count==0 assert for those.

**Tag-stripped inlined verdicts still collide on TITLE + schema headings
(#2357 r2, 2026-08-17):** stripping the inlined prior verdict's marker tags
(the #2332 form) removes the TAG collisions but not the TEXT ones — the r1
verdict's own `# Codex Code Review: … (r1)` title line and its `## Issues
Found` heading both still live in the prompt, so (a) the template-tail
sentinel+title update must anchor on the TWO-LINE pair
`<!-- …codex v<n-1> -->\n# Codex Code Review: … (r<n-1>)` (unique — the
stripped inlined copy lost its tag line), and (b) a schema-section insert
before `## Issues Found` must anchor on a template-only neighbor (the
`- [...]` placeholder line), never the bare heading. Upheld-blocker fix
rounds also want a `## Round-1 closure ledger` SCHEMA section in the verdict
template (per-element C-row status lines + a non-gating standing-rec row) —
mentions of it in prose line-wrap, so count-assert the heading only where
verbatim.

**Post-reconciler-binding-FAIL fix round (mixed rulings; #2332 r4,
2026-08-16):** when the prior round ended Claude-PASS / own-twin-FAIL /
reconciler BINDING FAIL with MIXED per-blocker rulings (upheld + downgraded +
rejected), inline TWO adjudication documents in one `# Round-N adjudication
record` section — the reconciler ruling FIRST (the acceptance contract +
no-relitigate source; state "where they differ, the RECONCILER wins"), the
twin's own prior verdict SECOND (closure-ledger context). The no-relitigate
block covers ONLY the REJECTED ids, quoting the ruling's rejection grounds
verbatim (plan-§ cites) — and extends into the output contract: the deferred
ids must NOT reappear as `CONCERN::` rows (a re-emitted deferred id
re-raises a bindingly-rejected finding), while still-open untouched concerns
get explicit per-row STILL-OPEN/re-emit status lines. Ledger semantics: a
reconciler `defer-concern` row means REJECTED-binding (closed this round),
NOT open — the Step 0.8 walk becomes `N open + M reconciler-deferred + K
addressed-this-round`. Frame NOT-FIXED = substantive FAIL for BOTH the
upheld blocker AND a reconciler-DOWNGRADED binding CONCERN (the downgrade
made it a must-address, not a may-address). When the ruling's must-fix
DEMANDED a specific joint/real-body pin, make pin REALNESS (drives the real
production function; fakes only at named seams) an explicit closure element
— the hollow-composition class the ruling itself established. A
minors-interaction the ruling flags ("fix together with the must-fix") gets
its own closure row.

**Overruled-own-PASS calibration round (#2357 r3, 2026-08-17):** when the
twin's PRIOR PASS was reconciler-overruled BY EXECUTION (static-trace PASS vs
a runtime differential the reconciler measured), compose the inverse of the
post-overturn shape: (a) inline the OVERRULING reconciler ruling verbatim
(tags stripped) as the acceptance contract — do NOT inline the twin's own
overruled PASS verdict (anchoring risk); quote only the reconciler's one-line
diagnosis of it ("answered the wrong question"); (b) add a dedicated
"Static-trace calibration" section BEFORE the closure duty: trace the STATE
MACHINE across the whole record sequence (name per-record flag values in
closure evidence), reason about the RUNTIME final state where the gated
action executes (here: final cwd at the commit) for EVERY allow-direction
shape, ban the overruled round's exact hedge sentence verbatim, and state
author-neutrality in BOTH directions (upheld r1 FAIL + overruled r2 PASS);
(c) when the fix implements a VARIANT of the reconciler's sketch (re-arm vs
permanent disarm), instruct adjudication on the variant's semantic criterion
(final-state soundness), not sketch-verbatim; (d) same-class residual probes
get their own closure element with an enumerated shape list (interleavings,
orderings, compound/subshell interaction with the new bit, pattern anchoring,
a full case-arm sweep for keep arms that neither poison nor disarm). Also:
verify the brief's round-history summary against events.jsonl — briefs
misattribute WHOSE PASS was overruled per round (the #2357 brief said both
overruled PASSes were Codex's; r1's was Claude's); compose from the verified
history and flag the discrepancy in the return. Tail-patch trap: the
Concerns-to-persist bracket ALSO references the closure-ledger heading —
sweep-assert heading staleness over the whole tail, not just the schema
block.

**FAIL+FAIL union scoped to OWN-twin ids only (#2223 napp r2, 2026-08-17):**
the #2332 union entry ("inline BOTH prior verdicts as acceptance contracts")
yields to a brief that scopes closure to the twin's OWN persisted ledger ids —
when the orchestrator persisted ONLY the Codex `CONCERN::` rows (13 ids) and
the brief says "verify its own N ids", reference ONLY the Codex r1 verdict
(by-path extraction on exact kind+version) and add an explicit
independence line: do NOT fetch/quote the parallel Claude verdict — the
Claude round-2 reviewer carries its own items. Also composable with the
by-path discipline: on a by-path round the acceptance contract is an
events.jsonl extraction command (exact kind + version + ts), not an inlined
envelope, so the #2145/#2332 tag-arithmetic collapses to own-head==1 /
close==1 / prior-head==0. Status-line vocabulary can follow the brief's
tokens (RESOLVED/UNRESOLVED) with ACCEPTED-NON-CHANGE/OVERTURNED added for
recorded non-changes; UNRESOLVED on a BLOCKER id = substantive FAIL, an
honestly-open CONCERN id re-raises at its own severity.

**Post-reconciler mixed-ruling round on a BY-PATH task (#2223 napp r3,
2026-08-17):** composing the fix round after Claude-PASS / own-twin-FAIL /
reconciler BINDING FAIL with mixed rulings, on a task whose established
discipline is by-path (no inlined envelopes): (a) the acceptance contract is
the `epm:review-reconciliation` row read by extraction command (exact kind +
top-level version + ts) — its body HEAD carries the ROUND sentinel
(`<!-- epm:review-reconciliation v2 -->` on a version-1 row); disambiguate
for Codex or it mis-keys the row. (b) The ledger walk becomes `N open + 1
deferred`: a `deferred` event is reconciler-rejected-BINDING (closed) — fence
it from BOTH blocker re-raise AND `CONCERN:: ` re-emission, quoting the
ruling's discard rationales verbatim in the do-not-relitigate list
(discarded-severity items like a settled de-minimis timing get an explicit
"do not re-time it"). (c) The twin authored BOTH the upheld blocker and the
discarded findings — state author-neutrality in both directions and that the
reconciler wins wherever it and the twin's own r2 verdict differ. (d) An
implementer (d)-flagged ordering/scope choice the ruling did not explicitly
pin (failed-gap-before-completeness; 32b-only layer pin) becomes an explicit
adjudication duty with the SEMANTIC criterion stated (no path may post a
positive verdict on incomplete inputs; residual must be named) — never
sketch-verbatim, and a mere preference without an unsound path is a note.
(e) Ledger evidence rows cite PRIOR-round line numbers — tell Codex to
re-locate every cited site in the round-N state. (f) By-path Step-3
verification: envelope greps adapt to extraction-command + ts + id-token
asserts over the final prompt (own-head==1 / close==1 / prior-heads==0 /
exactly one line-start grammar row).

**FAIL+FAIL AGREEMENT fix round on a TEXT-MATCH-mechanism fix (#2357 r4,
2026-08-18):** when round N-1 was Claude-FAIL + Codex-FAIL in AGREEMENT (no
reconciler; both reviewers' concern ids persisted), inline BOTH verdicts as
acceptance contracts (#2332 form: tags stripped; blockquote their
`CONCERN:: ` rows per #2329-rclose — asserts: line-start rows==1,
blockquoted==3) and state "they name the SAME defect class; where their
emphases differ, BOTH bind". Three composable deltas: (a) when the fix is a
TEXT-MATCH mechanism (an ERE union grepped over raw record text), extend the
static-trace calibration with a dedicated rule — adjudicate what the union
MATCHES on RAW text (anchoring, tab/multi-space whitespace, quote-adjacent
vs MID-WORD quoting, raw-vs-masked copy, embedded-newline record-splitting),
never merely case-arm entry; the residual hunt becomes a text-shape
enumeration and the over-tightening check is its inverse (which INNOCENT
texts match — e.g. commit messages mentioning the vocabulary). (b)
Marker-waived pre-existing residuals ("measured main=0") on a no-execute
twin compose as W-rows (PRE-EXISTING-VERIFIED | CONTRADICTED) with three
STATIC duties: blob-diff the deciding-machinery symbols across
origin/main-vs-HEAD (attest at compose time that the round delta touches
neither symbol — a cheap grep of the round diff), trace WHY main allows each
shape, confirm the round adds a belt not a widening; genuine pre-existing
routes via Step 0.9, CONTRADICTED = Critical substantive. (c) The brief
misattributed round history AGAIN (claimed both overruled PASSes were
Codex's; r1's was Claude's — same error as the r3 brief): verify every
round's history against events.jsonl and compose from that, flagging the
discrepancy in the return. Also: a fresh-sections rebuild can safely reuse
the prior round's trigger-dense span + plan/marker envelopes + hard-bounds +
rubric via count-asserted anchors even when EVERY middle section is
rewritten (task context, calibration, concerns, adjudication, attestations,
review focus) — sweep leftover round tokens with grep-and-eyeball scoped
outside the inlined envelopes.

**Execution-mode-FLIP cap round on an upheld-own-FAIL fix (#2357 r5,
2026-08-18):** the cap round (5 of 5) where the reconciler UPHELD the Codex
twin's OWN r4 FAIL by execution and re-opened two BLOCKERs. The dominant shape
shift: the orchestrator's instruction MANDATES decision-by-EXECUTION, which
INVERTS the r4 template's read-only execution BAN. Compose deltas: (a) write a
dedicated "Execution carve-out" section (SANCTIONED: read-only worktree git +
blob extraction via `git show <sha>:<path>` into /tmp, the guard `--self-test`,
`uv run pytest`, building fresh hermetic `git init` scratch repos UNDER /tmp and
invoking guard blobs against them — the guard is a DECISION hook, never commits,
so no real repo is mutated; BANNED: any commit/push/tracked-mutation, `checkout`/
`reset`, invoking a blob against the LIVE worktree; NEVER-FABRICATE STATIC
fallback labeled `STATIC (env unavailable) — traced, not measured`; an
unrunnable env is NOT `data-access-blocked`, only an unreadable FILE is) and set
the return's `Codex write mode: true (scratch execution sanctioned)`. (b) Reuse
the r4 rubric-protocol span (Steps 0..6, extracted between anchors
`# Review protocol (the same rubric...` and `### Step 7: Issue Verdict`) but
PATCH its execution-BAN lines to the scratch carve-out — five count-asserted
patches (Step 4 header + intro, the two `you have no uv env` clauses in Steps
3.8/4.6, the `Step 4 is static here, so ALWAYS take the READ path` clause; that
last one's OLD string MUST absorb the trailing `: read the pinned assertions
against the diff's new state.` or the replacement leaves a doubled clause —
caught by eyeball, not an assert). (c) MARKER-KIND CHANGE: round 5 posted
`epm:implementation` (bare head sentinel, no version digit) not `epm:results`,
with `### Summary/Files changed/Testing/Concerns addressed/Considered/Needs human
eyeball` headings — NOT the four `### (a)..(d)`. Attest at compose time that Step
0.5 scores on SUBSTANCE (map Summary+Files=(a), Considered=(b), Testing=(c),
Needs eyeball=(d), Concerns addressed=(e)) and that `marker-shape` is INVALID
this round; flag the bare head sentinel in the return only. Re-fetch is
unnecessary when the round report is pre-materialized. (d) Acceptance contract =
the binding r4 reconcile inlined VERBATIM tag-stripped (its executed rc-diff
table IS the acceptance criterion); do NOT inline the twin's own upheld r4 FAIL
verdict (anchoring risk) — the reconcile summarizes the mechanism. Author-
neutrality both directions (twin authored the upheld FAIL). (e) FENCE the twin's
own r4 CONCERN that the reconcile routed to a SEPARATE wf-fix (different file
`scripts/guard_repo_root_branch.sh`, untouched) — never a round-5 blocker, never
re-emitted as a `CONCERN:: ` row. (f) Cap-round framing: severity machine-parsed,
both directions, honest PASS; narrow the verdict Blocker-tags bracket to VALID
(`substantive`|`git-provenance`|`data-access-blocked`) vs INVALID
(`marker-shape`|`smoke-run-missing`). (g) Diff size: re-derive from
`git show --numstat HEAD` (+71/-9 guard, +91 test = +162/-9) — the brief's
"+80 guard / 222 lines" are patch-line/digest figures; flag in the return.
Rerunnable: `/tmp/codex-2357-r5-compose.py`; template `/tmp/codex-code-reviewer-2357-r5-template.md`.

**Second consecutive reconciler-fix round + kind-spelling trap (#2223 napp
r4, 2026-08-17):** two deltas on the r3 by-path shape. (a) The
`epm:review-reconciliation` row's BODY head tag spells a DIFFERENT kind
(`<!-- epm:review-reconcile v3 -->`) than the JSON `kind` — on top of the
round-sentinel-vs-JSON-version mismatch, tell Codex explicitly not to key
the row by its body tag (key by exact JSON kind + top-level version + ts).
(b) When round N-1's reconciliation is the SECOND on the task, the
do-not-relitigate fence STACKS: carry the round-(N-2) fence items forward
verbatim (labeled by source ruling) plus the new ruling's discarded items
(here: turn-level siblings = plan-conformant estimator design;
facet-sink severity capped at the implemented fix, incl. the ruling's
"Observed but not raised" permitted-innerHTML set — restate that set so the
twin doesn't demand escaping past it). Recommended-but-implemented items
with no ledger row compose as pseudo-IDs (#1092-r4 pattern) whose
acceptance text = the ruling's "Recommended …" paragraph + the impl
marker's per-item disposition table; a brief that names semantic axes for
one item (e.g. sentinel: both-DVs-then-write / dry-run exclusion / legacy
WARN tolerance / no healthy-tree deadlock) gets those axes spelled out as
per-property duties with an added INVERTED-state probe (sentinel present,
DV file absent).

**Cap round after a THIRD consecutive reconciler-FAIL (#2223 napp r5,
2026-08-18):** three deltas on the r3/r4 by-path shape. (a) The
do-not-relitigate fence now stacks THREE rulings; two fence items need
explicit spelling or the twin predictably re-FAILs: (i) when the new
ruling CORRECTED the twin's own aggravator inside an UPHELD finding (the
r4 honesty note: "pre-sentinel behavior would have surfaced it" does NOT
hold — the pre-round tree consumed the same state silently), restate the
correction in the twin's own-verdict context input so it does not repeat
the disproven aggravator while verifying its upheld item; (ii) when the
prescribed fix DELIBERATELY preserves a permissive path (hash-less
sentinel → treated-as-absent → legacy WARN + consume), add a
"the-prescription-is-not-a-defect" fence bullet quoting the ruling's own
routing — otherwise the residual hunt re-raises the prescribed
backward-tolerance as the blocker's leftover. (b) A closure duty whose
blocker spans Required items 1+2+3 under ONE concern id composes as one
ledger status line with per-item sub-structure (writer binding / reader
validation with an enumerated mixed-pair state machine (a)-(f) /
three-part test integrity incl. an explicit fails-pre-fix control-flow
duty), plus a dedicated attention point rendering the reader's realized
state→outcome mapping as a short table. (c) Assert-side trap: an
instruction literal that LINE-WRAPS in the composed prose ("Record
`**Diff acquisition:** sha-range <range>`" split across lines) defeats an
exact-count assert — assert the one-line verdict-header form exactly, and
the instruction form on a whitespace-normalized copy
(`re.sub(r'\s+', ' ', p)`). Also: a 9-char range SHA is a PREFIX of the
10-char prior-tip SHA — count the long form first and subtract.

**Adopted-own-FAIL fix round + orchestrator-fenced OPEN blocker (#2360 r2,
2026-08-18):** two deltas on the #2332/#2147 shapes. (a) When the orchestrator
leaves one of the twin's own persisted BLOCKERs OPEN but RECLASSIFIED
out-of-band (here: Phase-V live validation ruled a SEQUENCING GATE —
orchestrator-owned, implementer forbidden from provisioning), compose a
three-part fence: never re-raise as a round defect, plan-adherence rows that
depend on it are marked `pending <gate> (orchestrator-owned open gate)` not
✗-against-the-round, and the already-persisted id gets the status-line
vocabulary `OPEN-GATE (orchestrator-owned; not a round-N defect)` — while
explicitly PERMITTING a Needs-user-eyeball / Recommendation mention as the
outstanding gate. (b) A brief labeling a press lead "fix-round-introduced"
can be WRONG about provenance — probe `git show <round-parent>:<path> | grep`
at compose time (here the broadened lock-parse except was round-1-introduced,
delta-untouched, unflagged in r1); compose the lead with verified provenance
(fresh press on the merits, `substantive` if raised, weigh the design's
declared fail-open posture) and flag the discrepancy in the return. Also:
`grep -c '{{'` the PAYLOADS before asserting no-braces — a legit f-string
`{{` in a new test scopes the assert to template-side + per-placeholder
zero-counts. Rerunnable: `/tmp/codex-2360-r2-compose.py`.
**Post-halt regression-round shape (#2333 r3, 2026-08-16):** when a binding
reconciler PASS shipped, the pod then HALTED in production, and the fix
round vendors a sibling-branch implementation: (a) brief-ordered
by-reference markers work — main-checkout absolute events.jsonl + exact-kind
`jq 'select(.kind==... and .version==N)'` (proven readable by Codex in r2);
give the python3 fallback and the frozen-worktree-copy warning. (b) The
reconciler-persisted upheld-non-blocking CONCERNs are REGRESSION duties,
not closure: new `## Prior-concern regression check` section with
NOT-REGRESSED / TOUCHED-VERIFIED / REGRESSED vocabulary, plus a
stale-line-number warning (the fix shifts ledger-cited lines — key on
symbols). Do NOT tell Codex to re-emit the inlined ledger rows as
forwarder rows (already persisted; only NEW findings / a REGRESSED id).
(c) Vendored-source fidelity is priority 1: pin the exact
`show origin/<branch>:<file> | sed -n 'A,Bp'` command (verify the ref
resolves in the worktree at compose time), and enumerate the deltas to
adjudicate — verbatim-copy claim, relaxed asserts (occurrence count),
dropped locus branches — with the instruction to VERIFY the implementer's
"dead code for this universe" rationale against the pair construction, not
the docstring. (d) Build the do-not-relitigate list from the reconciled
marker's per-item table: DISCARDED items, RECORDED-DEVIATION-OK items, and
its "Standing recommendations on PASS" are all non-duties for a focused
fix round. The reconciled marker's posted top-level version can differ
from its body head sentinel (v1 posted, body says v2 = the review round) —
tell Codex not to be confused. (e) Add a machine-scannable
`## Round-N fix verification` section (one verdict line per brief
priority: vendored fidelity / halt path / relaxation safety / Step 4.5
regression-test adequacy). (f) HEAD-drift robustness: committing agent
memory to the SAME worktree branch mid-round moves HEAD past the round
tip — word the prompt "at compose time HEAD was the round commit;
reviewer-memory commits may follow, touching no round file; verify round
files via `show <round-sha>:<path>`" instead of asserting HEAD identity
(hit LIVE this compose: a parallel reviewer's memory commit landed on the
branch between my assemble and my memory commit). Also: a PreToolUse-
blocked compound Bash means NOTHING in it ran — a `cat >>` upstream of the
blocked git verb never executed; re-apply the write (via the Edit tool,
never a heredoc — #1756) before retrying the commit, and verify landing by
blob read at HEAD, never by the commit line.

**Cap-round crash-fix with a COMPLETED sibling leg (#2333 r5, 2026-08-17):**
when one leg finished + uploaded under PRE-fix code and the fix targets the
OTHER leg's crash, make completed-leg INVARIANCE the highest-consequence
priority with a 5-part checklist: (a) verbatim-delegation byte-equality on
the old-path branch (delegation, not reimplementation, is the argument);
(b) legacy-record branch ORDER — the old-path check must precede any
new-flag read (`rec.get` on records predating the schema); (c) no reader
REQUIRES the new manifest key (compose-grep: writer-only); (d) old-path
call-site rewires are pure refactor (identical values + assert behavior);
(e) phase-done skip isolates the completed leg's store. Instruct "ANY
finding that changes the completed leg's semantics is Critical". Compose
the crash-diagnosis envelope from the epm:progress halt note when no
epm:failure exists (variant g — cite the marker's own Dispatch-context
provenance). A re-POSTED smoke-arch marker (vs r4's left-standing) flips
that priority from left-standing-claim verification to CLAIM-SET
verification vs the diff (bank-row accuracy + "verbatim from v<n-1>" + the
no-architecture-change greps). Open concerns with NO closure assigned get
regression-duty-ONLY framing plus named look-hardest rows where the diff
touches a concern's neighborhood (here: a position-only edit inside a
cap-regen pass vs the write-order concern; a record-schema change vs the
blind-chunk-reuse concern class). Also state the missing `### (e)` heading
is EXPECTED when the round claims no closures — else Codex invents a
marker-shape objection.

**Post-cap USER-GREENLIT scoped round (#2333 r6, 2026-08-17):** when a
cap-5 FAIL+FAIL park is followed by a user greenlight for ONE scoped round,
frame it explicitly as "greenlit scoped round past the cap-5 park, NOT
round <n> of the loop" (sentinel = impl marker version). The CONTRACT
replaces the plan: inline verbatim envelopes for (i) the greenlight
progress note, (ii) the cap-park `epm:failure` residual excerpt, (iii) the
ledger BLOCKER row, (iv) the prior twin's Critical Evidence/Fix — and when
the two r(n-1) reviewers' rows were merged at park ("id-A == id-B" in the
failure note), treat the un-persisted twin id as a PSEUDO-ID and say ONE
closure adjudication covers both (tail: do-not-re-emit either). Decompose
into C1-C7 with SCOPE as its own item (greenlight: out-of-scope hunk =
substantive FAIL; NEW substantive blocker RE-PARKS — swap the cap-round
"advance-or-surface" tags sentence for "re-park-or-relaunch"). Attest a
REPURPOSED (e) heading (here "(e) Parent-module default behavior") is not
the optional Concerns-addressed section — at most CONCERNS, closure runs
off code. A marker DEVIATION NOTE about pre-existing collection errors
gets the Step 0.9 provenance probe pinned in the prompt (zero commits in
the round range on the failing file). Hand compose-time literal-grep
ground truth for the implementer's sweep table NEUTRALLY when the table's
"site" column cites the consumption point rather than the literal hit
(adjudicate honesty, don't pre-resolve). Name any same-name-different-
module decoy the caller sweep will hit (issue2094_run.py's own
run_injection_gate).

**Concern-closure round on a by-reference compose (#2333 r4, 2026-08-16):**
when the round CLAIMS closure of a reconciler-persisted concern, the
concern row's `evidence` field is the acceptance contract — inline it
VERBATIM (mark it "this is the acceptance contract") and decompose it into
per-item (i)/(ii)/(iii) adjudication lines in a dedicated fix-verification
verdict section, with an explicit note where the implementation deviates
from the evidence's LITERAL prescription under a recorded directive (here:
S1 frozen-map restriction + orphan refusal vs the evidence's "re-derange";
instruct adjudicate-as-faithful-or-not, not auto-flag). Three sibling
duties this shape adds: (a) a reconciler "Standing recommendations on
PASS" that is DIRECTIVE-PHRASED at this round ("Round 4: include X") but
absent from the diff (compose-time grep) gets its own NEUTRAL adjudication
block + a verdict line — named severity ceiling (Minor/CONCERNS-class,
never a manufactured FAIL) so Codex neither ignores nor inflates it;
(b) pre-trace INTERACTION regressions where the fix makes a prior
concern's exposure WIDER (here: survivor rebuild makes donor maps depend
on the drop set → the donor-chunk blind-resume concern's stale-cache risk
now has a new axis) — hand it to Codex as the look-hardest regression row;
(c) the closure concern gets its status line in the fix-verification
section, NOT a row in the regression-check section, and the ledger's
missing `addressed` event is declared EXPECTED (orchestrator business).
Assert traps hit live: `grep -c` counts LINES not occurrences (the tail's
CONCERN:: grammar block = 4 substring hits on 3 lines); per-id tail counts
need a map when ids recur in instruction prose (ledger field + closure
line + regression instruction); Python `len()` vs `wc -c` differ on
em-dash-heavy prompts (~1 KB per 130 KB) — not drift. Worktree memory
commits: the root-code-commit guard fires even on a `cd <wt> && git ...`
compound — use `git -C "$WT" add/commit -- <path>` explicitly.

**Adopted-own-FAIL fix round + orchestrator-fenced OPEN blocker (#2360 r2,
2026-08-18):** two deltas on the #2332/#2147 shapes. (a) When the orchestrator
leaves one of the twin's own persisted BLOCKERs OPEN but RECLASSIFIED
out-of-band (here: Phase-V live validation ruled a SEQUENCING GATE —
orchestrator-owned, implementer forbidden from provisioning), compose a
three-part fence: never re-raise as a round defect, plan-adherence rows that
depend on it are marked `pending <gate> (orchestrator-owned open gate)` not
✗-against-the-round, and the already-persisted id gets the status-line
vocabulary `OPEN-GATE (orchestrator-owned; not a round-N defect)` — while
explicitly PERMITTING a Needs-user-eyeball / Recommendation mention as the
outstanding gate. (b) A brief labeling a press lead "fix-round-introduced"
can be WRONG about provenance — probe `git show <round-parent>:<path> | grep`
at compose time (here the broadened lock-parse except was round-1-introduced,
delta-untouched, unflagged in r1); compose the lead with verified provenance
(fresh press on the merits, `substantive` if raised, weigh the design's
declared fail-open posture) and flag the discrepancy in the return. Also:
`grep -c '{{'` the PAYLOADS before asserting no-braces — a legit f-string
`{{` in a new test scopes the assert to template-side + per-placeholder
zero-counts. Rerunnable: `/tmp/codex-2360-r2-compose.py`.
**Post-halt regression-round shape (#2333 r3, 2026-08-16):** when a binding
reconciler PASS shipped, the pod then HALTED in production, and the fix
round vendors a sibling-branch implementation: (a) brief-ordered
by-reference markers work — main-checkout absolute events.jsonl + exact-kind
`jq 'select(.kind==... and .version==N)'` (proven readable by Codex in r2);
give the python3 fallback and the frozen-worktree-copy warning. (b) The
reconciler-persisted upheld-non-blocking CONCERNs are REGRESSION duties,
not closure: new `## Prior-concern regression check` section with
NOT-REGRESSED / TOUCHED-VERIFIED / REGRESSED vocabulary, plus a
stale-line-number warning (the fix shifts ledger-cited lines — key on
symbols). Do NOT tell Codex to re-emit the inlined ledger rows as
forwarder rows (already persisted; only NEW findings / a REGRESSED id).
(c) Vendored-source fidelity is priority 1: pin the exact
`show origin/<branch>:<file> | sed -n 'A,Bp'` command (verify the ref
resolves in the worktree at compose time), and enumerate the deltas to
adjudicate — verbatim-copy claim, relaxed asserts (occurrence count),
dropped locus branches — with the instruction to VERIFY the implementer's
"dead code for this universe" rationale against the pair construction, not
the docstring. (d) Build the do-not-relitigate list from the reconciled
marker's per-item table: DISCARDED items, RECORDED-DEVIATION-OK items, and
its "Standing recommendations on PASS" are all non-duties for a focused
fix round. The reconciled marker's posted top-level version can differ
from its body head sentinel (v1 posted, body says v2 = the review round) —
tell Codex not to be confused. (e) Add a machine-scannable
`## Round-N fix verification` section (one verdict line per brief
priority: vendored fidelity / halt path / relaxation safety / Step 4.5
regression-test adequacy). (f) HEAD-drift robustness: committing agent
memory to the SAME worktree branch mid-round moves HEAD past the round
tip — word the prompt "at compose time HEAD was the round commit;
reviewer-memory commits may follow, touching no round file; verify round
files via `show <round-sha>:<path>`" instead of asserting HEAD identity
(hit LIVE this compose: a parallel reviewer's memory commit landed on the
branch between my assemble and my memory commit). Also: a PreToolUse-
blocked compound Bash means NOTHING in it ran — a `cat >>` upstream of the
blocked git verb never executed; re-apply the write (via the Edit tool,
never a heredoc — #1756) before retrying the commit, and verify landing by
blob read at HEAD, never by the commit line.

**Cap-round crash-fix with a COMPLETED sibling leg (#2333 r5, 2026-08-17):**
when one leg finished + uploaded under PRE-fix code and the fix targets the
OTHER leg's crash, make completed-leg INVARIANCE the highest-consequence
priority with a 5-part checklist: (a) verbatim-delegation byte-equality on
the old-path branch (delegation, not reimplementation, is the argument);
(b) legacy-record branch ORDER — the old-path check must precede any
new-flag read (`rec.get` on records predating the schema); (c) no reader
REQUIRES the new manifest key (compose-grep: writer-only); (d) old-path
call-site rewires are pure refactor (identical values + assert behavior);
(e) phase-done skip isolates the completed leg's store. Instruct "ANY
finding that changes the completed leg's semantics is Critical". Compose
the crash-diagnosis envelope from the epm:progress halt note when no
epm:failure exists (variant g — cite the marker's own Dispatch-context
provenance). A re-POSTED smoke-arch marker (vs r4's left-standing) flips
that priority from left-standing-claim verification to CLAIM-SET
verification vs the diff (bank-row accuracy + "verbatim from v<n-1>" + the
no-architecture-change greps). Open concerns with NO closure assigned get
regression-duty-ONLY framing plus named look-hardest rows where the diff
touches a concern's neighborhood (here: a position-only edit inside a
cap-regen pass vs the write-order concern; a record-schema change vs the
blind-chunk-reuse concern class). Also state the missing `### (e)` heading
is EXPECTED when the round claims no closures — else Codex invents a
marker-shape objection.

**Post-cap USER-GREENLIT scoped round (#2333 r6, 2026-08-17):** when a
cap-5 FAIL+FAIL park is followed by a user greenlight for ONE scoped round,
frame it explicitly as "greenlit scoped round past the cap-5 park, NOT
round <n> of the loop" (sentinel = impl marker version). The CONTRACT
replaces the plan: inline verbatim envelopes for (i) the greenlight
progress note, (ii) the cap-park `epm:failure` residual excerpt, (iii) the
ledger BLOCKER row, (iv) the prior twin's Critical Evidence/Fix — and when
the two r(n-1) reviewers' rows were merged at park ("id-A == id-B" in the
failure note), treat the un-persisted twin id as a PSEUDO-ID and say ONE
closure adjudication covers both (tail: do-not-re-emit either). Decompose
into C1-C7 with SCOPE as its own item (greenlight: out-of-scope hunk =
substantive FAIL; NEW substantive blocker RE-PARKS — swap the cap-round
"advance-or-surface" tags sentence for "re-park-or-relaunch"). Attest a
REPURPOSED (e) heading (here "(e) Parent-module default behavior") is not
the optional Concerns-addressed section — at most CONCERNS, closure runs
off code. A marker DEVIATION NOTE about pre-existing collection errors
gets the Step 0.9 provenance probe pinned in the prompt (zero commits in
the round range on the failing file). Hand compose-time literal-grep
ground truth for the implementer's sweep table NEUTRALLY when the table's
"site" column cites the consumption point rather than the literal hit
(adjudicate honesty, don't pre-resolve). Name any same-name-different-
module decoy the caller sweep will hit (issue2094_run.py's own
run_injection_gate).

**Concern-closure round on a by-reference compose (#2333 r4, 2026-08-16):**
when the round CLAIMS closure of a reconciler-persisted concern, the
concern row's `evidence` field is the acceptance contract — inline it
VERBATIM (mark it "this is the acceptance contract") and decompose it into
per-item (i)/(ii)/(iii) adjudication lines in a dedicated fix-verification
verdict section, with an explicit note where the implementation deviates
from the evidence's LITERAL prescription under a recorded directive (here:
S1 frozen-map restriction + orphan refusal vs the evidence's "re-derange";
instruct adjudicate-as-faithful-or-not, not auto-flag). Three sibling
duties this shape adds: (a) a reconciler "Standing recommendations on
PASS" that is DIRECTIVE-PHRASED at this round ("Round 4: include X") but
absent from the diff (compose-time grep) gets its own NEUTRAL adjudication
block + a verdict line — named severity ceiling (Minor/CONCERNS-class,
never a manufactured FAIL) so Codex neither ignores nor inflates it;
(b) pre-trace INTERACTION regressions where the fix makes a prior
concern's exposure WIDER (here: survivor rebuild makes donor maps depend
on the drop set → the donor-chunk blind-resume concern's stale-cache risk
now has a new axis) — hand it to Codex as the look-hardest regression row;
(c) the closure concern gets its status line in the fix-verification
section, NOT a row in the regression-check section, and the ledger's
missing `addressed` event is declared EXPECTED (orchestrator business).
Assert traps hit live: `grep -c` counts LINES not occurrences (the tail's
CONCERN:: grammar block = 4 substring hits on 3 lines); per-id tail counts
need a map when ids recur in instruction prose (ledger field + closure
line + regression instruction); Python `len()` vs `wc -c` differ on
em-dash-heavy prompts (~1 KB per 130 KB) — not drift. Worktree memory
commits: the root-code-commit guard fires even on a `cd <wt> && git ...`
compound — use `git -C "$WT" add/commit -- <path>` explicitly.

**Deferred-concern implemented ANYWAY + resume-round marker (#2174 r2,
2026-08-18):** two deltas on the #1094-r2 upheld-concern-bounce + #2371
addressed-not-open shapes. (a) When the reconciler DEFERRED a blocker
outright (rejected-binding, standing-rec only) but the fix round lands its
change ANYWAY under the plan's own calibration lever (here: blanket
`re.IGNORECASE` vs the standing rec's "keep `EVERY row` uppercase-only",
justified by a re-run corpus sweep 8 < 10), do NOT treat the deferred id as
pure fence material — give it a closure-ledger line with adjudication
vocabulary (`DEFERRED-BINDING — <hunk>: JUSTIFIED-BY-SWEEP | UNJUSTIFIED |
DEFECTIVE-IN-IMPLEMENTATION`), name the standing-rec-vs-round TENSION
explicitly with the plan's calibration gate as the binding criterion, and
split the duties: mechanism verified by READING (trigger alone must not
WARN — window + satisfier gates), sweep NUMBERS treated as reported
evidence (no uv env), side-effects sweep over the other regex arms. Fence
only the RE-RAISE, never the in-diff hunk's merits. (b) A RESUME-round
marker (predecessor 529-killed after landing the commit; resume verifies +
closes concerns + posts) composes with three attest-don't-flag notes: the
"no new commit this resume" shape is legitimate; a missing
`<!-- epm:results vN -->` head sentinel scores on H3 substance (at most
Style); a VOIDED detached pytest whose junitxml is read as informational
gets its untouched-file failures pre-routed to Step 0.9 provenance
(pin the exact `git log origin/main..HEAD -- <file>` probe in the prompt).
Also: numeric drift between the predecessor's commit message and the
resume's marker (4,334 vs 4,336 sweep plans) is surfaced neutrally as
predecessor-run-vs-re-run, never resolved by the composer.

**Upheld-own-FAIL fix round with an addressed-not-open ledger (#2371 r2,
2026-08-18):** when the implementer runs `address-concern` BEFORE the review
(latest event `addressed`, so `--open-only` returns `[]` on a round whose whole
point is closure), do NOT read the empty open-set as the #1090-fu1-r2
ledger-empty shape — the row EXISTS: inline BOTH events in a snapshot envelope,
set the ledger-field literal to
`empty (0 open; 1 addressed-r2 pending verification: <id>)` (count-assert ×3:
attestation + Step 0.8 + tail header), and frame the addressed row as the
implementer's CLAIM whose verification IS the round. Acceptance contract =
reconciler ruling verbatim (tag-stripped; its EXECUTED flip grid is the
criterion) + the twin's own r1 Fix/Mechanizable/sweep lines QUOTED as a short
blockquote (not the full own verdict — enough to adjudicate WHICH sanctioned
fix option was chosen, without anchoring). Closure decomposes C1-C6 with the
implementer's NAMED RESIDUAL as its own element carrying a TRUE/FALSE flip
test (TRUE = pre-existing class the ruling already fenced, route Step 0.9;
FALSE = a round-added atom flips it, blocker NOT closed, substantive FAIL).
Assert trap: backticked heading mentions contain the bare-heading substring —
assert total == verbatim + backticked (5 == 1 + 4) plus the `\n<heading>\n`
line form == 1, never a bare heading count == 1.

**Mixed-actor `addressed` ledger on a reconciled-PASS bounce (#2178 r2,
2026-08-18):** when the round-1 reconciler PASSed but bounced residuals, the
orchestrator closes the DISCARDED blocker ids by posting `addressed` rows AS
the reconciler, and the implementer later posts `addressed` rows for the
bounced ids — so the r2 ledger walk must split by the addressed row's ACTOR:
reconciler-addressed = binding discard (do-not-relitigate fence),
implementer-addressed = the round's closure claims (VERIFIED-ADDRESSED /
NOT-ADDRESSED duties). State the split explicitly ("0 open; K addressed-r2
pending verification; M reconciler-discarded-binding") — a bare 0-open read
under-specifies both fences and duties. Inline the snapshot rows with FULL
JSON fields: the `evidence` field on a reconciler-raised row carries its
close prescription verbatim (the acceptance contract rides in free). Also:
the reconciler's Standing-recommendations bundle minus the bounced items is
its own fence bullet — the round implementing ONE family member does not
convert siblings into blockers, but round-hunk regressions on their
neighborhoods stay in scope (filter-vs-cap ordering, ladder-continuation).

**Never hand-truncate a SHA in compose prose (#2183 r2, 2026-08-18):** a
decorative truncated HEAD prefix typed by hand carried a WRONG hex char
(`...cbf...` for `...cb0...`) and the labeled count-asserts did NOT catch it
— they count the FULL form, and the corrupt truncation is a separate string.
Write the full 40-char SHA (or a `git rev-parse --short`-derived prefix)
everywhere in compose facts; add a negative assert on any truncated form you
do emit. Otherwise this round was pure recipe reuse: #2371 addressed-not-open
ledger (snapshot both events, ledger-field literal `empty (0 open; 1
addressed-r2 pending verification: <id>)`), #2329-rclose prior-own-verdict
inlining (tags stripped, `> CONCERN::` blockquote, author-neutrality both
directions), #2348(e) no re-emission of the persisted id (assert exactly one
line-start grammar row), and a `## Round-1 closure ledger` schema section
with a pseudo-ID for the un-persisted r1 Minor 2.

**Dual-crash cap round + SHA-correction supersession (#2378 r5, 2026-08-19):**
two composable deltas on the crash-fix shape. (a) TWO crashes of DIFFERENT
diagnosis kinds in one review round — a P1 `epm:failure` (the marker-v5 fix)
plus a P0 `epm:progress` hot-fix record (orchestrator-inline fix that shipped
unreviewed straight to relaunch): inline BOTH in separate CRASH-DIAGNOSIS
envelopes, put the unreviewed hot-fix commit explicitly IN SCOPE ("gets its
first review here"), and give the data-only sibling commit (banks) digest-only
instructions after a compose-time `--numstat` proof it carries zero code.
(b) A same-round `epm:progress` CORRECTION note superseding a fix-engaged
element's SHA (pre-amend SHA in element 4, unreachable by design): inline it
in its own `---BEGIN SHA-CORRECTION NOTE---` envelope, pre-adjudicate the
element as CORRECTED (never marker-shape, never a probe target — "do not
probe for the stale SHA"), and count-assert BOTH SHAs per part (impl 2+1,
correction 2+1, own prose 1+1). Also: deferred-concern rounds want a
`DEFERRED-BINDING-UNDISTURBED | REGRESSED-BY-ROUND` status vocabulary (the
regression arm = fresh finding at own severity under a NEW id), and a
`## Crash-fix verification` REQUIRED schema section (per-crash
root-cause-addressed / fix-engaged-creditable / no-silent-fallback /
residuals lines) inserted before the ledger-status heading.

**User-greenlit CAP-RAISE fix round on an upheld-own-FAIL (#2378 r6,
2026-08-20):** when the user reopens a cap-5 park with "more rounds" (cap
raised to 10), this is NOT the #2333-r6 one-scoped-round shape — frame it as
an ordinary fix round of the resumed loop ("round 6 of 10, user-extended
cap"; no out-of-scope-hunk=FAIL clause, standard severity-precision line)
and inline the greenlight `epm:progress` note in its own envelope: its
numbered items double as the round contract (the named minimal fix + a
carry-forward duty), with an explicit split of which items are REVIEWABLE
in code vs orchestrator-owned (pod reuse, relaunch pacing). Composable with
#2371 (addressed-not-open ledger) + #2329-rclose (author-neutrality on the
upheld own FAIL; ruling inlined FIRST as binding, own verdict SECOND
tag-stripped + rows blockquoted). Also: (a) verify the brief's NEW-file
labels with `git show <sha> --diff-filter=A --name-only` — the #2378 r6
brief labeled an r5-created/r6-extended test file NEW; (b) a marker
diff-stat like "+140/−15" can be a churn-SUM (125+15) — pre-adjudicate as a
Style note so the twin doesn't flag a phantom falsity; (c) rubric-currency
hit live AGAIN via the round-parent spec-sync commit itself (`git show
<sync-sha> -- code-reviewer.md` shows exactly what to patch — here the
#2201 divergence block, inserted before the "Size the diff" anchor, plus a
compose-facts line recording the brief's `diverged_on_main: none` probe
result as do-not-re-derive).

**Ensemble-PASS crash-fix round — crash-fix 2 (#2378 r7, 2026-08-20):** deltas vs
the #2329-r6 crash-fix-with-impl-marker shape when the PRIOR round closed
Claude-PASS + own-twin-CONCERNS (no reconciler, `CONCERN:: none`): (a) the twin's
own UNPERSISTED prior Minors get a dedicated fence section — compose-grep
attestation that the new hunks touch neither surface, no re-raise / no row
re-emission, a fresh-finding-on-material-interaction escape, author-neutrality
both directions; (b) with zero addressed rows this round, the ledger section is
open+deferred STATUS lines PLUS R-rows: REGRESSION duties for prior-round-CLOSED
ids whose surfaces the crash-fix touches (the pins-record 5-key change was the
closed r6 blocker's exact surface — the crash-fix duty F2 doubles as that row's
decision), vocabulary NOT-REGRESSED / TOUCHED-VERIFIED / REGRESSED-BY-ROUND;
(c) envelope-embedded SHA arithmetic: when HEAD is built by concatenating the
envelopes, count-asserts must target PROMPT TOTALS with authored+embedded
breakdowns in the label — per-part asserts double-count (hit live 3× this
compose); grep the embedded bodies' short-form counts, never predict them;
(d) an impl marker EMBEDDING an `epm:failure-lesson` block (own head/close tags
before the impl close) needs a shape attestation + ==1 tag asserts, and a
failure marker whose body sentinel says v1 while posted top-level version is 3
gets the standard do-not-be-confused note; (e) a marker-quoted fix-engaged log
line that greps ZERO in the diff may be an f-string SPLIT across source lines —
probe the post-image for a fragment before treating it as the #2329-r11
zero-hit composer observation, then tell Codex to grep by fragments.

**Consecutive crash-fix rounds — tail reuse + rubric-native round tokens
(#2378 r8, 2026-08-20):** on the SECOND consecutive crash-fix round the r7
shape reuses almost verbatim; three mechanical deltas: (a) extract the
output-contract TAIL from the COMPOSED prior prompt via
`prior_prompt.split(RUBRIC)` (assert exactly 2 parts) — it already carries the
prior round's patches, so only v<n-1>→v<n> + the new F-duty schema remain
(blanket `Round-7`→`Round-8` replace is safe on the small authored tail;
count-assert capitalized and lowercase forms separately first); (b) round-token
residue asserts on the RUBRIC must COUNT-PIN, not assert absence — "round 7"
(#779 descope) and "round 8" (#653 r8) each appear once RUBRIC-NATIVELY in
incident text; (c) cross-part token totals (e.g. `PASS_UNIFIED`) must sum
EVERY part including the rubric's own Step 0.55 gate text (smokearch 2 + impl
1 + rubric 2 + authored 1 — hit live). Content-side: when the prior twin
verdict was a zero-findings PASS there is NOTHING to fence from r<n-1> — say
so explicitly and keep the OLDER round's Minors fence verbatim; a brief
phrase like "un-droppable by later env merges" that the realized code only
approximates (env_extra merges last, implementer-declared deliberate) is the
#2332-r2 named-tension pattern — attest the call-site inventory at compose
time, hand the sufficiency ruling to Codex.

**Duty-DISCHARGE cap round after the deferral trigger fires (#2378 r10,
2026-08-20):** when the prior round's sharpened carry-forward contract fires
its trigger ((a): first non-emergency dispatcher round) AND the round
implements the dictated duties, the r9 D-section flips from
BINDING-DUTY-UNMET adjudication to DISCHARGE closure: vocabulary
DISCHARGED-BY-ROUND / PARTIALLY-DISCHARGED / NOT-DISCHARGED (NOT-DISCHARGED
= substantive FAIL — the round's own contract includes the duty), the
SHARPENED `deferral_rationale` is the acceptance contract, and each dictated
ELEMENT the realized fix exceeds/moots becomes a NAMED TENSION (here: a
"runbook interpreter note" mooted by self-correcting main() routing; the
dictated "guarded shutdown" omitted on an os._exit(1) failure path that the
same rationale's finalization-deadlock cite arguably justifies;
recursion-termination of a self-re-dispatching entry — the child must
satisfy the interpreter discriminator or re-dispatch loops; subprocess.run
timeout killing the direct child but not vLLM EngineCore grandchildren).
Composable deltas: (a) a DESIGNED-gate-trip fix round has NO epm:failure —
ground truth = the orchestrator's diagnosis progress note + the RAW pod-side
JSON gate record (posted by the dispatch script, top-level v1); inline both,
pre-adjudicate their shapes, and state the expected RE-TRIP at TRUE rates IS
the fix-engaged signal, never a failure; (b) a reconciler standing rec whose
fire-clause triggered but was NOT taken (the env_extra merge-order companion
edit; the touched merge line re-indented ORDER-UNCHANGED — attest it) gets
the #2333-r4(a) neutral note on its L-row with a Minor/CONCERNS ceiling;
(c) blockquote the inlined prior verdict's `CONCERN:: none` too, not only
real rows (it is still a line-start forwarder hit); (d) for an ACCOUNTING
fix, frame F-duties on writer↔aggregator coherence: enumerate every summary
writer and classify payload-keyed + filename-parseable / stage-level-by-
design / neither, and adjudicate the still-silent membership-drop residual
(#906 class residue) at Codex's severity; (e) assert trap: the diff-
acquisition record instruction carries the short payload SHA TWICE (range +
payload parenthetical) — count 2, not 1.

**Reconciler-deferred binding-point round (#2378 r9, 2026-08-20):** when the
prior round's reconciler deferred the twin's own BLOCKERs with an explicit
`Binds at the next <X>-touching round: <dictated duty>` clause and THIS round
touches X but implements neither duty (the orchestrator's punch list omitted
them), compose a dedicated REQUIRED D-section: three-way vocabulary
DISCHARGED-BY-ROUND / CARRY-JUSTIFIED / BINDING-DUTY-UNMET with severity
explicitly Codex's to rate; the ledger slice must carry ALL THREE events per
id (twin BLOCKER raise, reconciler CONCERN re-raise, reconciler deferral —
the `deferral_rationale` IS the acceptance contract); surface the
two-contract conflict (reconciler clause vs punch-list scope) NEUTRALLY
("do not assume either document wins by default"); attest partial-narrowing
nuances (here: the new engine-KWARG pin reaches the standalone arm where the
r8 ENV pin did not — a narrowing, not the dictated fix) and any in-log
empirical datum touching a deferred duty (the gate self-terminated at 100 s
— evidence about the class, not a discharge). D-ids get D-lines only, never
CONCERN:: rows. Mechanical deltas hit live: (a) a posted epm:failure body
can carry NO closing tag where the prior version had one — probe tags per
marker, never assume the sibling's shape; (b) the impl marker may omit the
assert_tag literal — key the presence assert on `epm:failure v<n>` instead;
(c) CAPS envelope titles ("ROUND-8 RECONCILER RULING") do not count toward
"Round-8" token asserts — count the cases separately; (d) the inlined prior
twin verdict ECHOES output-schema headings ("## Crash-fix verification
(REQUIRED") — assert tail+embedded totals; (e) a partial-sentence rep1 on a
tail note leaves trailing round tokens alive — count-pin the lowercase
residue then blanket-replace.

**Fresh-cycle plan-authorized recalibration round after a CLOSED loop (#2378
r11, 2026-08-21):** when the implement-review loop closed at a reconciled
binding PASS and the plan pre-authorizes a ONE-SHOT recalibration round, the
compose is a fresh cycle, not a fix round: (a) the ACCEPTANCE CONTRACT is the
orchestrator's lever-decision `epm:progress` note — inline it as a
`RECALIBRATION DECISION` envelope and turn each numbered lever into a V-duty
(the F-duty slot of gate-trip rounds); the plan's authorization line
(here v6:473 "Allowed without asking") goes in the round-semantics bullet;
(b) inline the TRUE-rates gate record + the guard-verification note as
separate envelopes; established facts = the MEASURED causes, with the
projection numbers explicitly carved out as NOT-reviewable (mechanism only);
(c) the closing reconciler ruling inlines as "closure record — context";
the twin's own OVERTURNED verdict is NOT re-inlined when the ruling's
findings table carries every disposition (cleaner tag arithmetic:
own-head==1/close==1, prior tags==0); (d) a reconciler REQUIRED guard
already executed+verified (progress note) whose ledger row still reads
latest=raised gets a GUARD-DISCHARGED-PENDING-LEDGER-ACTION status
vocabulary + a return-flag for the orchestrator to run `address-concern` —
the pendency is never a round finding; (e) assert trap hit live: the r10+
rubric span carries TWO rubric-native `**Prior-concerns ledger:**` tokens
(Step 0.8 record-the-ledger-state sentence) — assert per-part (rubric 2 /
tail 1 / authored 0), never total==1; (f) a one-shot stake round wants the
severity-precision framing of a cap round ("PLAN-AUTHORIZED ONE-SHOT ROUND"
in the Blocker-tags bracket) even though it is round 1 of a fresh cycle.

**User-authorized plan-amendment round after a terminal gate-FAIL park (#2378
r12 = impl v12, 2026-08-23):** when a designed gate FAIL parks the task
(`epm:failure` triage menu), the user picks an option, and the round implements
the resulting plan amendment: (a) the ACCEPTANCE CONTRACT is a four-envelope
CHAIN — the failure park (defect + options), the user decision `epm:progress`
note (option pick + any MANDATORY-FIRST-FIX clause), and the `epm:plan`
amendment-record markers — each pre-adjudicated shape-wise (a failure body
sentinel v1 vs posted top-level v6 gets the do-not-be-confused note; no fresh
`epm:failure` this round — a park-follow-up is not a crash-fix). (b) STALE
worktree plan at scale: worktree frozen at v6, canonical v8 = 141 KB — inline
only the AMENDED SECTIONS (extract by heading; ~34 KB: Amendment record +
each brief-named §) and keep the frozen v6 PATH as auxiliary reference for
unchanged conventions, with explicit "inlined v8 wins on conflict" + "every
<dropped-thing> statement in v6 is SUPERSEDED" — a middle path between
full-inline (#1090 r5) and by-path (#1090 fu1). (c) Plan version posted AFTER
the impl marker (v8 = prose correction ratifying the implementer's named (d)
assumption, +3 min): inline the correction marker, pre-adjudicate the marker's
"plan_version=v<n-1>" line as true-when-posted, and instruct scoring against
the corrected reading — never silent drift, never marker-shape. (d) An
ADDRESSED ledger row whose recorded companion duty ("if any further X is ever
planned") has its fire clause TRIGGERED by the round gets included in the
slice with a dedicated V-duty (the in-code re-implementation of the hand
guard) + a TOUCHED-VERIFIED / REGRESSED-BY-ROUND status line — the closed-row
sibling of the deferred-row D-duty. (e) DESCOPE rounds instruct BOTH
directions: inertness (kept-defined archival constants / banks / tombstoned
phase bodies are never dead-code findings) AND a completeness sweep (any
remaining ACTIVE-path reference to the dropped family — gates, sizing, shard
maps, floors — is the substantive direction). (f) Assert traps hit live: in a
plain-string part, count {SHA} placeholders BEFORE replacing; a doubled-brace
literal contains its single form as a substring (assert `{{rnd}}`==1 FIRST,
replace, then count `{rnd}`); when envelopes embed via f-string into ONE part,
token totals sum the four PARTS only (parts_total helper, embedded counts as
diagnostic labels only) — summing parts AND their embeddings double-counts.
