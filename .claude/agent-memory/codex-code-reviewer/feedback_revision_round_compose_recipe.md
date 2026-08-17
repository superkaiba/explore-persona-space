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
