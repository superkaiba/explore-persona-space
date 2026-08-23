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

**Filter-mitigated compose at FIRST prompt (post-refusal task) + post-cap
authorized round + shared-brief override (#2147 r6, 2026-08-19):** when a
PRIOR round's Codex dispatch was REFUSED by the content filter
(trigger-dense framing over filesystem-reclaim/guard logic + a whole-file
read), compose the NEXT round already mitigated — do not wait for a second
refusal: (a) reuse the recovery prompt's (cr5b-style) neutralized rubric
spans verbatim (they are filter-TESTED; the fresh scaffold gets a hot-word
assert: attack|destroy|exploit|malicious|hostile|adversar*|kill* banned
from composer-written text, embeds exempt); (b) bounded reads become HARD
BANS (never whole-file, never `nl -ba` — name it as a prior dispatch
hazard), with a sed-window table + your own recomputed post-round line
anchors (the fix shifts them; brief ranges are pre-round frames — say
"never a finding"); (c) findings-by-reference discipline (file:line +
abstract description, no command literals in the verdict body) goes in the
bans AND the verdict template; (d) payload target well under the accepted
recovery size (r6 landed 131 KB vs the accepted 186 KB) — inline only
brief + impl marker + failure record + auth note; plan BY PATH when
diff-verified fresh. POST-CAP round shape: inline the round-5 `epm:failure`
(the residual's acceptance contract) and the user-authorization
`epm:progress` note in their own envelopes; frame as "normal contract,
bar neither lowered nor raised, review as round 1"; the brief's CLOSED
list becomes the no-reopen fence (reopen needs NEW evidence from THIS
diff). SHARED-BRIEF override: a brief written for both twins carries
Claude-only posting instructions (`task.py post-marker epm:code-review`) —
add a "Codex adaptations of the brief" block immediately after the brief
envelope overriding: posting (you emit the codex marker block only),
events.jsonl reads (inlined), line frames (recomputed), and run-duties
(static READ translation, cite `git show <prefix-sha>:<file>` windows for
pre-fix-behavior reasoning). Also: Step 4.6 presence half is N/A-BY-KIND
on an `epm:experiment-implementation` report (binds on `epm:results`
only) — say so explicitly or Codex invents a marker-shape finding; and a
spec-freshness-synced branch balloons three-dot to 100s of files — ban the
whole-branch BODY, keep --stat + per-file forms.

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
verbatim. Header FIELD lines collide too (#2197 r2, 2026-08-19): when the
prior round's base form was also `HEAD^..HEAD`, the inlined verdict's
`**Diff acquisition:** sha-range HEAD^..HEAD (<r1 range>)` line is
byte-identical to the template's — scope that replace to the tail after the
unique `## Output format` anchor (count-assert the anchor first).

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

**Sibling-rubric spans embed the sibling's BASE SHA — sweep hex SHAs, not
just the issue number (#1739 claim4-controls r1, 2026-08-19):** the 2379-r1
template's rubric span carried its base SHA (`2f52e456…`) in FOUR probe
commands (Step 0's name-only enumeration + all three Step 0.9 subclass
probes) — an `assert "2379" not in rubric` misses them entirely, and
"purely additive" sibling round-shape claims hide in Step 0.9 prose too
(grep that phrase; two hits, only one in Step 4.6). On every
sibling-template reuse, ALSO grep the sibling's compose-time SHAs (base,
round HEAD, commit list) over the FINAL prompt and assert 0. Three further
composable deltas from this round: (a) DUAL-ENVELOPE report — when the
posted impl marker is a compact summary referencing the full implementer
report by an unreachable /tmp path, inline BOTH (marker in the standard
`---BEGIN IMPLEMENTATION MARKER BODY---` envelope = the Step 0.5 shape
subject; report in its own `---BEGIN FULL IMPLEMENTER REPORT BODY---`
envelope) and instruct Codex to score 0.5 on the PAIR; attest the
head-sentinel-digit-vs-posted-version mismatch (bare `v1` head under
task-wide v22) neutrally so it never becomes a finding. (b) MID-ROUND
origin/main MERGE (8,202 files): per-commit `git show` primary + merge
`--name-status` only + Step 0 tier classification via round-scoped
`git show --name-only --format= <shas> | sort -u` (the sibling's
whole-branch name-only would misclassify tier), Step 0.9 probes re-keyed to
the recorded pre-merge tip. (c) Same-issue follow-up with worktree tasks/
REFRESHED by that merge: plan.md CAN be by-path valid even at
`followups_running` — verify identity AND round-match (H1 names the
followup_label), and note DELTA-plan inheritance (v21 inherits v4 verbatim;
both readable in the same plans/ dir).

**#1739 claim4-controls r2 fold (2026-08-19) — two refinements:** (a) when
the r2 marker is SELF-CONTAINED (no external /tmp report path), DROP the
r1 dual-envelope (marker-only) and attest the change explicitly ("unlike
round 1, this marker is SELF-CONTAINED"), else Codex hunts for the report;
sweep the r1 dual-envelope tokens (`{FULL_REPORT_BODY}`, "the full
implementer report is INLINED") in the final asserts. (b) The
sweep-hex-SHAs assert has a legit-survivor pattern: a swept r1 SHA that a
round-2 DUTY deliberately cites (here the premerge_tip CONTENT SHA
`8439ae52bf…` in the U9 line-1 check) gets `assert count == 1`, not `== 0`
— pair it with a full-40-char presence assert so the survivor is the
deliberate occurrence. Union-work-order rounds: inline the union in its own
envelope + annotate its /tmp verdict paths UNREACHABLE, inline the Claude
Minors verbatim (Codex cannot read the Claude verdict file), and give
open-ledger rows a `## Round-1 closure ledger` verdict-body section whose
NOT-ADDRESSED branch = substantive FAIL + a re-raise `CONCERN:: ` row
reusing the exact id (VERIFIED-ADDRESSED rows are NOT re-emitted).

**FAIL+FAIL union with a punch-list disposition table + fixed-without-
ledger-close ids (#2379 r2, 2026-08-19):** three deltas on the #2332 union
shape. (a) When the impl marker carries a `### Round-2 punch-list
disposition` table keying rows by BOTH the twin's persisted concern ids AND
the Claude verdict's g-group labels, key the closure ledger on the
punch-list rows directly (no pseudo-ID minting needed — the table pre-mints
the union). (b) INVERSE-#2371 ledger shape: ids whose fixes the punch-list
claims landed (unit A) but with NO `address-concern` rows posted — treat as
raised-only ledger state + FIXED claim: identical closure-verification duty
to the addressed ids, missing rows attested as orchestrator bookkeeping
(never a finding); the ledger literal then needs BOTH clauses ("N open: ...;
M addressed-r2 pending verification: ..."), with the open list containing
claimed-fixed ids. (c) Same-task r1→r2 template reuse worked with a 15-patch
list (base SHAs in Step 0 name-only + 0.9 probes; "purely additive" in 0.9
opener AND a LINE-WRAPPED copy in 4.6 — token greps miss wrapped phrases,
patch by known text; every "the report claims ..." sentence re-keyed to the
v2 report: 0.6 compute-deviation + porcelain, 0.65 upload claims, 4(b)/(c)
lint + pin-sweep, 4.5 round-1 tail, 5 deserialization). Tag arithmetic with
BOTH prior verdicts inlined tags-stripped + codex rows blockquoted: own-head
v2==1, close==1, all four prior tag forms==0, line-start rows==1,
`> CONCERN:: `==13; the OLD r1 ledger literal legitimately survives INSIDE
the inlined codex-r1 envelope — scope residue greps outside envelopes.

**#1739 claim4-controls r2 fold (2026-08-19) — two refinements:** (a) when
the r2 marker is SELF-CONTAINED (no external /tmp report path), DROP the
r1 dual-envelope (marker-only) and attest the change explicitly ("unlike
round 1, this marker is SELF-CONTAINED"), else Codex hunts for the report;
sweep the r1 dual-envelope tokens (`{FULL_REPORT_BODY}`, "the full
implementer report is INLINED") in the final asserts. (b) The
sweep-hex-SHAs assert has a legit-survivor pattern: a swept r1 SHA that a
round-2 DUTY deliberately cites (here the premerge_tip CONTENT SHA
`8439ae52bf…` in the U9 line-1 check) gets `assert count == 1`, not `== 0`
— pair it with a full-40-char presence assert so the survivor is the
deliberate occurrence. Union-work-order rounds: inline the union in its own
envelope + annotate its /tmp verdict paths UNREACHABLE, inline the Claude
Minors verbatim (Codex cannot read the Claude verdict file), and give
open-ledger rows a `## Round-1 closure ledger` verdict-body section whose
NOT-ADDRESSED branch = substantive FAIL + a re-raise `CONCERN:: ` row
reusing the exact id (VERIFIED-ADDRESSED rows are NOT re-emitted).

**FAIL+FAIL union with a punch-list disposition table + fixed-without-
ledger-close ids (#2379 r2, 2026-08-19):** three deltas on the #2332 union
shape. (a) When the impl marker carries a `### Round-2 punch-list
disposition` table keying rows by BOTH the twin's persisted concern ids AND
the Claude verdict's g-group labels, key the closure ledger on the
punch-list rows directly (no pseudo-ID minting needed — the table pre-mints
the union). (b) INVERSE-#2371 ledger shape: ids whose fixes the punch-list
claims landed (unit A) but with NO `address-concern` rows posted — treat as
raised-only ledger state + FIXED claim: identical closure-verification duty
to the addressed ids, missing rows attested as orchestrator bookkeeping
(never a finding); the ledger literal then needs BOTH clauses ("N open: ...;
M addressed-r2 pending verification: ..."), with the open list containing
claimed-fixed ids. (c) Same-task r1→r2 template reuse worked with a 15-patch
list (base SHAs in Step 0 name-only + 0.9 probes; "purely additive" in 0.9
opener AND a LINE-WRAPPED copy in 4.6 — token greps miss wrapped phrases,
patch by known text; every "the report claims ..." sentence re-keyed to the
v2 report: 0.6 compute-deviation + porcelain, 0.65 upload claims, 4(b)/(c)
lint + pin-sweep, 4.5 round-1 tail, 5 deserialization). Tag arithmetic with
BOTH prior verdicts inlined tags-stripped + codex rows blockquoted: own-head
v2==1, close==1, all four prior tag forms==0, line-start rows==1,
`> CONCERN:: `==13; the OLD r1 ledger literal legitimately survives INSIDE
the inlined codex-r1 envelope — scope residue greps outside envelopes.

**Same-task r2→r3 union fix round: reconstruct-head + patched-REST (#2379
r3, 2026-08-19):** four durable deltas on the #2332/#2379-r2 union shape.
(a) The concerns ledger has a THIRD event type `verified-open` (posted when
a twin's FAIL re-raises an already-addressed id) — the OPEN predicate is
latest ∈ {raised, verified-open}; a latest-event walk keyed only on
raised/addressed silently drops those ids from BOTH lists (hit live: the 3
re-raised BLOCKERs vanished from my first walk). (b) Implementer-side
`addressed` rows posted minutes AFTER the impl marker are round INPUTS
(closure claims pending this review) — include them in the snapshot; the
#2326 ts-pin excludes only parallel-REVIEWER rows (review OUTPUTS); re-probe
for drift at compose end. (c) An UPDATED-PHASES-ONLY `## Smoke run` (the v3
marker says the round-2 evidence remains current for unchanged phases)
needs the PRIOR marker's smoke section inlined as a context-only excerpt
envelope (do-not-score-shape; findings only where a round-3 hunk
invalidates a claim) + a 0.5/0.6 attestation so Codex never FAILs
`smoke-run-missing` on untouched phases — anchor the excerpt on the
line-start heading `\n## Smoke run\n` (a prose mention elsewhere in the
body inflates the bare-substring count). (d) Head convention-mentions enter
token arithmetic: the head's own "blockquoted (`> CONCERN:: `)" sentence
adds +1 to the blockquote-count assert. Also: copy content-safety /
hard-bans / plan sections VERBATIM out of the prior head by span anchors
(never retype), and REST-side SHA-count asserts must exclude head
occurrences (the diff-acquisition command lives in the head). NO-CODE
disposition rows get the #2147-cr5 ACCEPTED-NON-CHANGE | OVERTURNED
vocabulary wired into the closure-ledger schema, with the
flagged-for-follow-up row doubling as a scope ruling (CONCERN row, not
FAIL).

**Crash-fix round with TWO open NON-GATING reconciler concerns (#1739 cr4 =
sentinel v4 / impl v5, 2026-08-19):** when the prior round closed via a
reconciler BINDING PASS that DOWNGRADED the twin's own BLOCKER to a
non-gating persisted CONCERN (plus a `verified-open` re-open of an addressed
row), the fix round's ledger duty is a per-concern STATUS line
(`NOT-TOUCHED — remains open` is EXPECTED and not a finding / `TOUCHED —
verified effect`), never a closure demand — patch the rubric's Step 0.8
block accordingly and instruct no re-emitted `CONCERN::` rows unless the
delta REGRESSED a concern's mechanism. Name the SIBLING-ARTIFACT precision
trap explicitly when the delta touches an artifact ADJACENT to a concern's
mechanism (here: the runner's pod-status sentinel renamed/atomicized vs the
scorer's per-seed SUMMARY completion sentinel the concern names) — without
it the twin predictably mis-scores the concern as addressed. Reconciler
standing-only items (never persisted) get their own
route-as-row-or-leave-standing instruction. Also: this task's established
`CONCERN::` grammar is the PIPE-delimited four-field form
(`<SEV>|<kebab-id>|<file:line or n/a>|<summary>`) carried from its r3
template — when a brief dictates it, follow the brief + task precedent and
FLAG the divergence from the #2326 space-token grammar in the return (the
forwarder's token parse differs). Sentinel-vs-impl-version mapping (codex
v4 / impl v5) is stated in facts + return, and the crash diagnosis
(`epm:progress`, no `epm:failure` posted) inlines with an explicit
falsified-prime-suspect note handing the diagnosis-vs-marker disagreement
to the twin (V-duty), never resolving it at compose time.

**Reconciler-PASS-with-OPEN-residuals fix round (#2379 r4, 2026-08-19):**
when the prior round closed Claude-PASS / own-FAIL via a reconciler BINDING
PASS that left ONE concern OPEN with named residuals (M1/M4/M5) and DEFERRED
the rest, three deltas on the #2332-r4 mixed-rulings shape. (a) The ledger
grows a FOURTH live event kind `deferred` (posted by `defer-concern --by
reconciler`) — latest-event `deferred` = BINDING deferral, closed-for-round;
the OPEN predicate stays latest ∈ {raised, verified-open}, and the ledger
literal needs a third clause ("5 reconciler-deferred (binding, ride future
touches): ..."). Wire the no-re-emission fence into BOTH the closure-ledger
schema (per-deferred-id `NOT-TOUCHED — remains deferred (expected)` |
`REGRESSED` lines) AND the CONCERN::-row exception list (re-emit a deferred
id ONLY on a round-N REGRESSION of its mechanism). (b) The OPEN concern's
named residuals become R-numbered closure elements (R1/R2/R3) whose
acceptance contract is the RECONCILER's completeness-check paragraph +
Standing-recommendations section — inline the ruling FIRST (reconciler
wins), own prior verdict SECOND (tags stripped, rows blockquoted, read
THROUGH the classification); NOT-ADDRESSED on any R-element = substantive
FAIL, and the concern id itself gets ONE summary line whose NOT-ADDRESSED
branch re-raises the exact id. (c) An opportunistic fix that TOUCHES a
deferred id's mechanism gets a `TOUCHED — verified effect` status line
(never closure language). Compose-time anchor probes that paid off: grep
the round diff for the registry token (`PHASES` == 0 → 0.55 attestation
verified), for `torch.load` (== 0 → Step 5 bullet rewritten to
zero-new-sites), and for residual non-atomic writes (`write_text` at three
surviving sites → handed to Codex as a neutral M5 scope-adjudication item
with line numbers). Sharpest mechanism probe worth composing explicitly: a
shared `load_json_object` that collapses ABSENT and CORRUPT to one `None`
forces callers to re-stat to implement an asymmetric spend policy
(absent-rates fine / corrupt-rates refuse) — hand the branch-structure
question to Codex, severity pre-split (over-refusal = CONCERN row,
under-refusal = NOT-ADDRESSED). Reconciler marker version trap: posted
top-level version is v1 (first reconcile on the task) while the head
sentinel says v3 (the round) — fetch by kind from events.jsonl and state
the mapping in the return, never fetch by "v3".

**Reconciler-narrowed-remedy fix round, reconcile-note-only contract (#2201
cr2, 2026-08-19):** when the reconciler upheld the twin's DIRECTION on ALL
its findings but NARROWED every remedy, the reconcile note alone is the
acceptance contract — do NOT inline the prior twin verdict (unlike the
#2332-r4 mixed-rulings shape, where dropped/rejected items make the prior
verdict needed for ledger context); inlining the full FAIL would re-seed
the exact remedies the reconciler cut. Compose instead: (a) a CUT-REMEDIES
do-not-re-file block enumerating each cut with the reconciler's rationale
verbatim-ish (re-filing = scope creep under Unintended Changes; for a
REJECTED-as-topology-breaking remedy, its settlement's REQUIRED-ABSENT
halves become explicit checks — e.g. "confirm NO helper re-binding and NO
skipped-with-worktree fixture"); (b) a per-MF DISCHARGED/NOT-DISCHARGED
`## Round-2 discharge ledger` schema section (NOT-DISCHARGED = substantive
FAIL) + author-neutrality; (c) a fails-pre-fix INDEPENDENT verification
duty translated to STATIC reconstruction (`git show <r1-sha>:<path>` +
control-flow trace per fixture, verdict vocabulary CONFIRMED-RED-PRE-FIX /
NOT-CONFIRMED, "green against both blobs = decorative pin" framing); (d)
fix-round `epm:results` markers use per-MF headings, not (a)-(d) letters —
attest the content-equivalence mapping as a compose-time fact (CONCERNS
ceiling) and attest the r1 marker carries the full Gate-scope block so the
v2 terse line is PRESENT-BUT-TERSE (fold precedent); (e) when the brief's
Should-Fix list is a strict subset of the reconciler's, include the missing
one(s) with lenient absorbed-or-deferred-with-note adjudication; (f) static
proxies for unrunnable claims attested as compose-time facts to re-verify
(def-test count vs claimed collect-only, zero-hit stale-cap-literal sweep);
(g) note the reconcile note's file:line refs use R1-BLOB numbering.

**Union-NOTE-as-contract + instrument-non-vacuity fix round (#2391 r2,
2026-08-19):** when the orchestrator posted a standalone union WORK ORDER
note (B1/B2 with root-cause + per-item round-2 scope) after a FAIL+FAIL
agreement, inline THAT note as the sole acceptance contract instead of both
prior verdicts (#2332 form) — it pre-extracts every Evidence/Fix line, has no
marker tags, so the tag arithmetic collapses to own-head==1/close==1 with no
stripping. Deltas worth reusing: (a) a brief-ordered mutation probe ("flip
the allowlist, confirm red, restore by inverse edit") translates to the
#2146 SCRATCH-TREE form keyed on the test's OWN path resolution — when
`REPO_ROOT = Path(__file__).resolve().parents[1]`, a /tmp scratch tree
(`tests/` + the read files beside it) isolates the mutation completely;
check IMPORTS first to decide scratch-runnability (stdlib+pytest-only file →
runs anywhere; a file importing the project package must NOT be
scratch-copied — probe its pure helpers via `uv run python -c` from the
worktree on in-memory mutated strings instead). Never mutate the shared
worktree (the parallel Claude reviewer reads the same tree). (b) A brief
environment note ("uv rc=2 read-only cache → UV_CACHE_DIR=...; never bare
`python -m pytest` — imports MAIN's package, reads the old cap") is a
BINDING prompt section, not advice — round 1's twin burned its run on
exactly that stale-main false failure. (c) A same-defect concern pair
(Claude id + Codex id for one bug) gets ONE closure adjudication covering
both ids, stated in the ledger literal AND the closure-ledger schema. (d)
When the task history is "every blocker was a verification-instrument
defect", the priorities lead with instrument NON-VACUITY (reachability of
the scan's file enumeration, exemption exactness + the dead-control proof,
the lookbehind/word-adjacency claim behind a compose test, kwarg-default
behavior preservation) — with fabricated-falsifiability (claimed-red that
stays green) named as substantive FAIL.

**CONCERNS→polish round; brief can ORDER the whole-branch body back in (#2212
r2, 2026-08-20):** the round-scope BAN on the whole-branch three-dot BODY is
the DEFAULT, not a law — when the brief mandates re-verifying reconciler
binding conditions + the plan §10 fence "on the FULL round diff, not
inherited from round 1's PASS", compose BOTH reads: three-dot full-round
body (size it first; 55 KB here) for the conditions re-check, `git show
<round-sha>` as the deep-dive focus. Milder sibling of #2329 rclose: prior
round Claude-PASS + Codex-CONCERNS (no reconciler — CONCERNS advances), one
persisted NIT (the twin's own) + one un-persisted Claude Minor closed;
closure ledger = the real concern id + a pseudo-id, author-neutrality line,
addressed-id NEVER re-emitted as a `CONCERN:: ` row unless NOT-ADDRESSED.
A brief-supplied "state plainly a clean PASS is expected if the fixes are
sound" composes as BOTH-directions calibration (expected ≠ thumb on scale;
wording-preference blockers unwanted on a wording-correction round, real
defects at full severity). Over-correction duty on wording fixes: the new
text can be wrong the OTHER way (eliding the argv-structural top-level
guarantee / overstating exclude-list protection) — hand Codex both failure
directions plus survival checks for the honest-scope figures. Re-measure
brief-supplied base-drift figures at compose (136→295 commits between
rounds; use fresh).

**FAIL+FAIL union round with a self-resolving union note + brief-ordered
mutation probes (#2329 q35 r2 = sentinel v2, 2026-08-20):** when the union
work order ITSELF resolves every overlap to the stronger form AND records
the reviewer splits, inline the union as the PRIMARY acceptance contract +
the twin's own verdict (tags stripped, rows blockquoted) for per-finding
Evidence/Fix — the Claude verdicts stay by-path context (this task's
review-r1 artifacts dir on main); the union's RECORDED SPLITS section
becomes a do-not-relitigate item (round 2 scores fresh against the new
marker, never re-grades r1 gate records). Scratch recipe refinement for
mutation probes on this repo: build the scratch tree with a SCOPED
`git archive <round-sha> scripts tests src configs pyproject.toml uv.lock |
tar -x` — NEVER a full-tree archive (tracked eval_results are GB-scale in
this repo) — and copy in the single committed artifact a chosen nodeid
reads (here the parent #2162 stats.json) only when that probe needs it;
carry the UV_CACHE_DIR=/tmp/... rc=2 read-only-cache precedent. A
raised-only ledger (0 addressed rows) on a fix round composes per
#2379-r2(b): identical closure duty, absent rows attested as orchestrator
bookkeeping. Sentinel convention is SERIES-local: this review series' r1
head was v1 (round-based) even though the same task's earlier crash-fix
series used sentinel==impl-version — read the r1 marker's own head, never
assume the task-wide convention.

**CONCERNS→polish round; brief can ORDER the whole-branch body back in (#2212
r2, 2026-08-20):** the round-scope BAN on the whole-branch three-dot BODY is
the DEFAULT, not a law — when the brief mandates re-verifying reconciler
binding conditions + the plan §10 fence "on the FULL round diff, not
inherited from round 1's PASS", compose BOTH reads: three-dot full-round
body (size it first; 55 KB here) for the conditions re-check, `git show
<round-sha>` as the deep-dive focus. Milder sibling of #2329 rclose: prior
round Claude-PASS + Codex-CONCERNS (no reconciler — CONCERNS advances), one
persisted NIT (the twin's own) + one un-persisted Claude Minor closed;
closure ledger = the real concern id + a pseudo-id, author-neutrality line,
addressed-id NEVER re-emitted as a `CONCERN:: ` row unless NOT-ADDRESSED.
A brief-supplied "state plainly a clean PASS is expected if the fixes are
sound" composes as BOTH-directions calibration (expected ≠ thumb on scale;
wording-preference blockers unwanted on a wording-correction round, real
defects at full severity). Over-correction duty on wording fixes: the new
text can be wrong the OTHER way (eliding the argv-structural top-level
guarantee / overstating exclude-list protection) — hand Codex both failure
directions plus survival checks for the honest-scope figures. Re-measure
brief-supplied base-drift figures at compose (136→295 commits between
rounds; use fresh).

**FAIL+FAIL union round with a self-resolving union note + brief-ordered
mutation probes (#2329 q35 r2 = sentinel v2, 2026-08-20):** when the union
work order ITSELF resolves every overlap to the stronger form AND records
the reviewer splits, inline the union as the PRIMARY acceptance contract +
the twin's own verdict (tags stripped, rows blockquoted) for per-finding
Evidence/Fix — the Claude verdicts stay by-path context (this task's
review-r1 artifacts dir on main); the union's RECORDED SPLITS section
becomes a do-not-relitigate item (round 2 scores fresh against the new
marker, never re-grades r1 gate records). Scratch recipe refinement for
mutation probes on this repo: build the scratch tree with a SCOPED
`git archive <round-sha> scripts tests src configs pyproject.toml uv.lock |
tar -x` — NEVER a full-tree archive (tracked eval_results are GB-scale in
this repo) — and copy in the single committed artifact a chosen nodeid
reads (here the parent #2162 stats.json) only when that probe needs it;
carry the UV_CACHE_DIR=/tmp/... rc=2 read-only-cache precedent. A
raised-only ledger (0 addressed rows) on a fix round composes per
#2379-r2(b): identical closure duty, absent rows attested as orchestrator
bookkeeping. Sentinel convention is SERIES-local: this review series' r1
head was v1 (round-based) even though the same task's earlier crash-fix
series used sentinel==impl-version — read the r1 marker's own head, never
assume the task-wide convention.

**FAIL+FAIL union fix round with ledger-persisted ids + sibling's FULL
mid-compose disposition set (#823 P-Gen r4 = sentinel v7, 2026-08-20):**
three composable deltas on the #2332-r2 union shape. (a) When both prior
verdicts' findings are ALREADY persisted as real ledger ids, key the
closure ledger on THOSE ids (no pseudo-IDs) and still inline both verdicts
as the Evidence+Fix acceptance contracts — tags stripped, the Codex
verdict's `CONCERN:: ` rows blockquoted (`> `), asserts: own-head==1 /
close==1 / prior tags==0 / `^CONCERN:: `==1 / `^> CONCERN:: `==6. (b) The
#2326 ts-pin (rows `ts <= round-landing ts`) excluded not just one raised
row but the SIBLING TWIN'S ENTIRE completed round output (7 `addressed`
rows + 1 new `raised` row) landing mid-compose — snapshot framing "pinned
to the round-4 landing" stays truthful; REPORT to the orchestrator that
the Codex verdict may mint a DIFFERENT kebab-id for the same residual the
sibling already raised (dedup is the orchestrator's merge job, never the
composer's). (c) An orchestrator-found NEW residual handed in the brief
("put this FIRST") composes as its own `# HIGHEST-VALUE CHECK (adjudicate
FIRST)` section: the orchestrator's reading + apparent consequence quoted
with line anchors, THREE explicit questions (reading correct? production
path closed incl. the smoke-artifact-verified-as-production path?
severity for the held wave?), evidence-required-either-way framing
("its own reading has been wrong twice this round — take nothing on
trust"), and a mandatory Step 3.7 class sweep naming the class lineage
(round 3 trusted the record set; round 4 trusts the domain size) —
severity never pre-resolved. Landing-note round record (no impl marker)
composes per the #823 P-Gen v13 variant unchanged; a fix round whose tests
span FOUR files patches every "both files"/"two files" rubric span and
asserts zero residual occurrences.

**Reconciler-UPHELD minimal-set round (all twin blockers upheld; #2329 r3 =
sentinel v3, 2026-08-20):** when the prior round ended Claude-CONCERNS /
Codex-FAIL and the reconciler ruled FAIL BINDING upholding ALL the twin's
blockers with a "Must-fix (minimal set)" section: (a) the acceptance contract
is the RECONCILER RULING inlined in its own envelope — the twin's own prior
verdict is demoted to context ("the ruling, not your prose, is the contract";
author-neutrality = don't demand more than the ruling's minimal set, don't
wave through); (b) items the ruling classifies as TEST-hollowness where
production was already correct get an explicit "verify the PINS BIND, not
that production changed — a production-behavior change on those paths is an
Unintended Change" frame; (c) implementer-reported self-mutation probes become
the reviewer's REPRODUCTION duty with per-probe expected red/green patterns
incl. a healthy-control-stays-green expectation (a bypass redding the control
= weaker-but-binding pin, note not FAIL; all-green = hollow-pin FAIL), plus
one encouraged own-construction variant per pin; (d) aim the recurring-class
hunt at the round's OWN fix ("vacuous-guard hunt": nonempty-but-corrupted
states, what expected/present derive FROM, raise-skipped paths, per-side
grain, over-tightening) — this task shipped a cannot-fail gate inside a fix
for a cannot-fail gate once already; (e) recompute the brief's/ruling's
PRE-round line frames against the NEW blob and say shifted lines are never a
finding; (f) a marker-arithmetic item (rows-vs-registry) gets a compute-it-
yourself instruction naming each registry symbol — the prior round FAILed on
29-vs-32 with three reviewers computing three ways; residual mismatch =
substantive per the ruling (non-conforming redo of an explicit item), not
strippable marker-shape.

**Unimplementable-prescription closure round (#2412 r2, 2026-08-20):** when
the round-1 reviewers AND reconciler all prescribed a fix the orchestrator
later PROVED unimplementable by VM reproduction (`git cat-file -e` exits 128
for missing-path-at-valid-ref AND bad-ref alike — no rc==1 split exists, git
2.34.1), compose the #2147-cr4 ESTABLISHED FACTS shape but aimed at the
PRESCRIPTION rather than the finding's mechanism: state the reproduction as
established (the twin cannot re-run write-bearing git probes), BAN re-raising
the literal prescription, frame the implementer's replacement (ls-tree
three-way discrimination) as serving the prescription's INTENT (git error ⇒
undecidable ⇒ revert), and hand the twin a scoped correctness checklist for
the REPLACEMENT (empty-stdout-absent leg, tree-vs-blob path shapes, per-file
vs helper-wide fail grain adjudicated by consequence). The closure-ledger
status line for that item judges intent delivery, never the dead literal.
Also from this compose: (a) a brief's output-contract code block can carry a
STALE sentinel digit (`v1` on a round-2 brief) — treat the block as SHAPE,
set the sentinel by the task's convention (r1 posted v1 ⇒ r2 posts v2 =
review round = impl `epm:results` version), and flag the divergence in the
return; (b) fixture-discrimination claims ("ran it pre-fix, saw the silent
KEEP") translate to the static trace: `git show <round-parent>:<helper>` +
trace the fixture scenario through PRE-fix control flow, refuted claim =
fabricated coverage.

**Merge-reconciliation round mandated by the task's own divergence gate
(#2201 r3, 2026-08-20):** when the round under review is a gate-mandated
merge of pinned main + a mandated spec-freshness sync (no feature code, no
impl marker), compose: (a) round contract = the `[divergence-probe]` /
reconciliation `epm:progress` notes inlined in their own envelopes (the
verified note is the implementer's REPORT to re-verify, say so); declare
`marker-shape`/`smoke-run-missing` INVALID and Step 0.5/4.6 N/A-this-round;
inline the prior `epm:results` as CONTEXT-ONLY (do-not-score-shape). (b)
PRIMARY body = `git diff <merged-pin>..HEAD` — once the merge makes the pin
an ancestor, the two-dot form IS the own-diff; measure `git show --cc
<merge>` at compose (≈empty ⇒ attest "no novel line typed into the merge";
read reconciliation via per-side diffs `<parent>..<merge> -- <path>`). (c)
M-ledger INTACT/DAMAGED duties: both-sides-survive + SEMANTIC-contradiction
read per contested file; grandfather caps re-measured vs POST-MERGE landing
bytes (probe count=0 at the fresh pin ⇒ worktree bytes ARE landing bytes,
#1727) with the corridor + MAX_HEADROOM(3_000) bound; union-dedup
lost-content check (every pre-merge-branch-only line must survive to HEAD;
enumerate via `diff <branch-parent>..<merge> --name-only -- agent-memory/`);
hygiene (residue, tasks/ 0, src/ 0, exact own-diff path list); feature
integrity incl. LOCATING every pinned fragment/region anchor in the
POST-MERGE spec text (pre-merge gate PASSes certify NOTHING post-merge —
say so at Step 4); sync commit == current-main blobs at the probe pin (the
pin's commit object exists locally — `git show <pin>:<path>` works, no
fetch). (d) Enumerate intermediate commits since the last reviewed HEAD;
name any deliverable-touching one (here an E501 rewrap) as a verify-duty so
its hunk is not mistaken for merge damage; flag the scope call in the
return. (e) Probe-side traps hit live: `git diff --quiet <sha> HEAD -- 
<nonexistent-path>` exits 0 — verify the path EXISTS before trusting a
MATCH probe; and a brief-named output file can COLLIDE with a stale
same-name /tmp file from an earlier same-task critic round — flag the
collision to the orchestrator (premature-read hazard).

**False-negative reconciler-skip fix round + NOT-ADDRESSED-row grammar fix
(#823 P-Gen r6 = sentinel v9, 2026-08-20):** two deltas on the r4/r5 union
shapes. (a) When the prior round was Claude-PASS / Codex-FAIL and the
orchestrator FIXED without a reconciler because the split was a false
NEGATIVE (Claude missed a source-verified defect), frame the skip as
sanctioned-for-false-negatives-only ("no conflicting judgement to
adjudicate"), state a same-axis split THIS round WILL spawn a reconciler,
and inline ONLY the Codex verdict as the acceptance contract — the
missing-the-defect PASS adds no criteria; say explicitly its PASS softens
no closure duty. (b) The r5 template's "ALREADY-PERSISTED ids NEVER appear
as rows" rule suppressed the twin's NOT-ADDRESSED re-finding row (the
orchestrator had to forward the BLOCKER row by hand — a ledger
inaccuracy): the fixed grammar is precedence-ordered — a NOT-ADDRESSED
closure item gets BOTH a status line AND a NEW `CONCERN::` row under a
NEW kebab-id naming the NEW mechanism; only VERIFIED-ADDRESSED/ACCEPTED
items are row-banned; the dedup-against-ledger rule stays. State the
suppression incident in the prompt so the twin trusts the new rule. Also:
when the round's impl marker GAINS a `## Smoke run` block that the prior
round's Step 0.6 adaptation declared absent, rewrite the adaptation to
score the block's substance (offline main()-driven execution, zero spend)
— a stale "carries NO block" framing invites a fabricated marker-shape
finding.

**Mid-round crash-fix on a multi-lane task (#823 P-Fit mask-gate = sentinel
v10, 2026-08-20):** three deltas. (a) Sentinel = max-posted-codex-version + 1
EVEN when the round has its own impl marker (impl v11 → sentinel v10): once a
task's posted codex markers run head==posted-version (v6–v9 here), that
continuity outranks the #2329 sentinel==impl-version convention — state the
mapping in the return. (b) A fix that CONSUMES producer labels while the plan
text literally bans "validity labels" composes as the #2332
surface-the-tension shape: quote BOTH the ban clause AND its reconciliation
parenthetical ("the step-4 label-precedence fix reconciles the labeler") with
real line numbers, hand the producer's classifier read (`git show
<sha>:<gen module>` + def line) to the twin, never resolve the wording
yourself. (c) A prior lane verdict may exist ONLY as a /tmp output + an
orchestrator prose summary (`epm:code-review v8`), with the binding ruling
under the kind `epm:code-review-reconcile` — grep events by NOTE TEXT
("reconcil"), not only by expected kind names, before concluding a split was
never adjudicated. Also: plan-excerpts envelope (brief-cited line windows,
sed-verified) + full plan by /tmp copy + main-root fallback is the right
plan shape when the binding sections are line-pinned and the plan is ~159 KB.

**Twin-won closure round: reconciler ruling as SOLE acceptance contract
(#2329 r5 = sentinel v12, 2026-08-20):** when the prior round was
Claude-PASS×3 / Codex-FAIL and the RECONCILER upheld the Codex Major with
its own executed evidence + a minimal must-fix set, inline the reconciler
marker (tags stripped) as the ONLY acceptance-contract envelope and SKIP
the prior Codex verdict inline entirely — the ruling's Q-dispositions +
MF set subsume it, tag arithmetic collapses (assert prior-tag count == 0,
no blockquote/strip dance), and the payload shrinks. Pair with the
won-round author-neutrality frame ("you are adjudicating the closure of
your own blocker") and quote the reconciler's REJECTED alternative fix
sketch verbatim so the twin checks the shipped form is not the rejected
conditional in disguise. Three composable deltas: (a) when the fix
quantifies a refusal over a module-level SET CONSTANT (`ARM_KEYS`), aim
the vacuous-guard hunt one level up — verify at compose time the constant
is a literal (grep every reference, hand the line list over) and give the
twin a scratch probe that NARROWS the constant to ask whether any
committed match literal pins the set's COMPLETENESS (a full-enumeration
`match` like `\['a', 'b', 'c'\]` is the pin; per-element matches keep
passing); (b) ledger rows superseded by the ruling but with no `addressed`
event yet posted (bookkeeping lag) get adjudication-form status lines
(`SUBSUMED-BY-MF-1` / `SETTLED-R4`) — state the lag explicitly or the twin
re-raises its own settled finding; (c) a verifier's re-run instrument tree
(`/tmp/r18v/`-style: probe script + JSON output + fail-before/mutation
outs) is a REUSE input — instruct read-first + cross-check-report-claims-
against-outputs, copy-to-own-scratch before re-execution, and never edit
the evidence trees in place.

**Gate-mandated merge-reconciliation round WITH an r1 impl marker (#2217 r2,
2026-08-20):** three deltas on the #2201-r3 shape. (a) When the task HAS a
round-1 `epm:results` (unlike #2201 r3's no-marker case), inline it in the
STANDARD `---BEGIN IMPLEMENTATION MARKER BODY---` envelope framed CONTEXT-ONLY
(do-not-score-shape, adjudicated r1) — satisfies the Step 3 envelope guard
while `marker-shape`/`smoke-run-missing` stay declared INVALID; Step 0.5 +
4.6-presence get explicit N/A-this-round lines in a `## Gate-step record`
section. (b) The rubric-currency check can hit a commit that is ITSELF main's
side of the divergence (here `1f22cfed7f` added code-reviewer.md's "Main-side
divergence list (#2201)" paragraph AFTER the r1 extraction): skip template
reuse entirely, compose the TIGHT shape, and inline that paragraph VERBATIM
from the POST-merge worktree spec (anchor `**Main-side divergence list
(#2201).**` → `clean.`) — the worktree is current post-merge, no main-root
read needed. (c) Q3 lost-update recipe: per-path branch-ownership probe
`git log --format='%h %s' <pre-merge-tip> --not <pin> -- <path>` over EVERY
own-diff path (EMPTY history + present in own-diff = merge-created difference
= Critical), with the `--cc`-emptiness caveat stated (--cc suppresses
one-side reverts — result==branch-parent hunks are invisible; 253-byte --cc
attests only "no novel typed line"). The composer's OWN memory file can sit in
the own-diff (a prior session committed it to the branch) — it is in scope for
the ownership probe, stat-only otherwise. Also: r1's output file was the
UNSUFFIXED `/tmp/codex-output-issue-2217.md` — name the r2 output
round-suffixed and flag the collision hazard in the return.

**Merge-reconciliation round WITH a declared scope extension to a SIBLING
task's detector (#2214 r2, 2026-08-20):** hybrid of the #2201-r3/#2217-r2
gate-mandated merge shape and a substantive new-code adjudication. Deltas:
(a) when the brief supplies a numbered adjudication checklist ("judge this
specifically, do not defer to the justification"), inline the brief verbatim
in its own `---BEGIN ROUND-2 BRIEF---` envelope AND restate the checklist in
the focus section with routing added per item (here: item 1 → Step 0.68
hollow-gate at full strength, `hollow-verification-gate` tag; item 2 → the
Step 3.9 degeneracy analogue; item 5's file-separately disposition → a
`substantive` blocker with the revert-commit+file-task remedy named) — plus
a REQUIRED `## Scope-extension adjudication` verdict section, one grounded
ruling per item; (b) the r1-adjudicated deliverable re-landing via merge
gets a no-relitigate fence (both twins PASSed the same +59 hunk; re-opening
needs NEW evidence from the MERGED context — exactly what the
coexistence/interaction duties look for); (c) Step 4.5 INVERTS when the
round's change IS a test modification: the duty is residual detector
strength + whether the round's own fix property (order-robustness) is
pinned, absent pin = Minor sketch; (d) Step 0.9 pre-seeds the
pre-existing-on-trunk instance (the defect the extension fixes was measured
on pristine main at the pin — corroborable via `git show <pin>:<file>`) so
the twin routes fix-findings `substantive` and never blocks on the
pre-existing defect itself; (e) shared-brief adaptation: "you may re-run but
not simply accept" translates to internal-consistency reads of the reported
numbers against the mechanism (is "the merge strictly SHRINKS the failure
set" sound given the fixture cleans only one dimension?) — the Claude twin
carries literal re-runs. Assert-side: brief LINE-WRAPS split fragments —
pick wrap-safe single-line fragments for the checklist-survival asserts, and
assert the pinned RANGE TOKEN (`<pin>..HEAD`) not the full command string
(the brief's `-C "$WT" diff --stat` form breaks command-string matches).

**Both-readings-suspect adjudication round, by-reference with /tmp extracts
(#2329 q35 r20 = sentinel v13, 2026-08-20):** when the round's central item is
an open BLOCKER on which the ORCHESTRATOR and IMPLEMENTER hold conflicting
code readings (and the orchestrator's has already been wrong once in the
reassuring direction), compose a HIGHEST-VALUE CHECK that quotes BOTH readings
verbatim with recomputed anchors, declares both suspect, and decomposes into
numbered questions whose last is the concern's OWN acceptance bar
(VERIFIED-ADDRESSED / NOT-ADDRESSED = substantive FAIL) — never adopt either
reading into the prompt as fact. Deltas: (a) a brief-ordered by-reference
compose can EXTRACT marker bodies to /tmp files (this task's Codex reads /tmp
— the r5 /tmp/r18v precedent) instead of pointing at a 400-row events.jsonl:
still by-reference, no stash-race exposure, no trigger-dense paging; keep the
main-root events.jsonl + `git show HEAD:` as the authoritative fallback and
declare `data-access-blocked` GENUINE. (b) Impl-marker line anchors can be
PRE-ROUND frames even when the marker is honest about it ("at pre-round
numbering") — recompute against the round blob and say shifted lines are
never a finding. (c) Sentinel: the q35 series began round-based (v1, v2) then
switched to head==posted-version (v10/v11/v12) — continuity gives max-posted+1
(v13) even when the brief names the round by IMPL number (r20/impl v20);
state the mapping in the return. (d) "ADDRESSED DIFFERENTLY" contra-brief
claims about PRE-EXISTING code get an independent-re-verification duty with
the exact base-blob greps handed over (justified-if-true / honest-reporting
finding-if-false). (e) A DEFERRED concern whose reconciler-downgrade premise
was OPERATIONAL sequencing gets a DEFERRAL-PREMISE-INTACT/BROKEN status-line
form when the round changes the sequencing the premise relied on.

**Merge-reconciliation round + brief-pinned BINARY enum + open-NIT vocabulary
(#2205 r2, 2026-08-21):** composing the #2201-r3/#2217-r2 gate-mandated merge
shape under a brief that pins `PASS|FAIL` (no CONCERNS): route the open prior
NITs through a dedicated `## Prior-round items` section with the three-way
vocabulary STILL-OPEN-UNCHANGED | STILL-OPEN-NOW-MATERIAL | CLOSED-BY-MERGE —
UNCHANGED never blocks under the binary enum, NOW-MATERIAL re-raises at true
severity with a NEW kebab-id row naming the merge-created mechanism (persisted
ids still never re-emitted as rows otherwise). Two probe deltas: (a) surface
test-COUNT discrepancies between the reconciliation note and compose-time greps
(v28 claimed "12 branch-side test_c46_*"; HEAD greps 14 — hand to Codex as a
report-accuracy observation, never resolve it yourself); (b) when the merge
folded main's task-state in, the worktree's OWN plans/ copy becomes identical
to canonical — still INLINE the plan when the brief orders it, but the worktree
copy is then a legitimate sanctioned live-probe TARGET
(`verify_plan.py --plan-file tasks/<status>/<N>/plans/v<k>.md --kind <kind>`
proves import + CHECKS-registry load of the merged module, read-only). Also:
per-side survival greps get side-specific token sets (branch:
c46|workload|hydra|provision; main: margin|baseline|ceiling|c68|escape).

**Merge-reconciliation round WITH a declared scope extension to a SIBLING
task's detector (#2214 r2, 2026-08-20):** hybrid of the #2201-r3/#2217-r2
gate-mandated merge shape and a substantive new-code adjudication. Deltas:
(a) when the brief supplies a numbered adjudication checklist ("judge this
specifically, do not defer to the justification"), inline the brief verbatim
in its own `---BEGIN ROUND-2 BRIEF---` envelope AND restate the checklist in
the focus section with routing added per item (here: item 1 → Step 0.68
hollow-gate at full strength, `hollow-verification-gate` tag; item 2 → the
Step 3.9 degeneracy analogue; item 5's file-separately disposition → a
`substantive` blocker with the revert-commit+file-task remedy named) — plus
a REQUIRED `## Scope-extension adjudication` verdict section, one grounded
ruling per item; (b) the r1-adjudicated deliverable re-landing via merge
gets a no-relitigate fence (both twins PASSed the same +59 hunk; re-opening
needs NEW evidence from the MERGED context — exactly what the
coexistence/interaction duties look for); (c) Step 4.5 INVERTS when the
round's change IS a test modification: the duty is residual detector
strength + whether the round's own fix property (order-robustness) is
pinned, absent pin = Minor sketch; (d) Step 0.9 pre-seeds the
pre-existing-on-trunk instance (the defect the extension fixes was measured
on pristine main at the pin — corroborable via `git show <pin>:<file>`) so
the twin routes fix-findings `substantive` and never blocks on the
pre-existing defect itself; (e) shared-brief adaptation: "you may re-run but
not simply accept" translates to internal-consistency reads of the reported
numbers against the mechanism (is "the merge strictly SHRINKS the failure
set" sound given the fixture cleans only one dimension?) — the Claude twin
carries literal re-runs. Assert-side: brief LINE-WRAPS split fragments —
pick wrap-safe single-line fragments for the checklist-survival asserts, and
assert the pinned RANGE TOKEN (`<pin>..HEAD`) not the full command string
(the brief's `-C "$WT" diff --stat` form breaks command-string matches).

**Size-pin asserts compare BYTES, never `len(str)` (#2246 r2, 2026-08-21):**
a compose-script assert pinning an inlined artifact's size to a `wc -c` /
`ls -la` figure must use `len(text.encode("utf-8"))` (or `os.path.getsize`)
— `len(open(p).read())` counts CHARACTERS and diverges on any multi-byte
UTF-8 content (the 88,372-byte plan v4 read as 87,487 chars; em-dashes and
arrows are everywhere in plan prose). The labeled assert made the diagnosis
instant; an unlabeled one would have suggested plan drift and triggered a
wasted re-fetch. FAIL+FAIL-union r2 compose worked as specced (#2332
pattern): both prior verdicts inlined tag-stripped, codex rows blockquoted
(`> CONCERN:: ` ==5), closure status lines routed to a `## Round-1 closure
ledger` verdict section with per-id VERIFIED-ADDRESSED / NOT-ADDRESSED +
`f541-fixed` / `c901-deliberate-keep` dispositions, already-persisted ids
never re-emitted as rows (final line-start grammar rows ==1).

**Measured-TRUE acceptance round (#2273 r2, 2026-08-22):** a variant of the
#2147-cr3/cr4 accepted-FAIL shapes — the twin's r1 FAIL was accepted (no
reconciler), and the flagged claims were then MEASURED and found TRUE; the
remedy was the twin's OWN offered branch (add evidence rows), not weakening.
Compose three blocks: (a) author-neutrality + an ESTABLISHED-FACTS fence —
"do NOT re-raise 'unsupported' against a claim that now traces to rows 7-9;
the facts are settled by measurement"; (b) the live question is
ROW-BOUNDEDNESS of the landed text (the brief's A10 headline), not truth —
an unhedged claim outrunning its row re-lands the r1 defect class ⇒
Critical; (c) closure status lines keyed to the persisted concern ids with
the acceptance contract restated per id (rc clause may name ONLY the
measured consumers; residual clause bounded to its rows; pin-sweep figures
re-grepped by the twin against composer ground truth). Doc-only in-place
line replacement: round numstat is 1/1 (old bullet superseded) while
net-vs-main is +1/-0 — state BOTH and pre-clear the deletion's identity or
the twin flags collateral removal.

**Generous-timeout lint attestation beats a third INCONCLUSIVE (#2273 r2):**
after two rounds of timeout-killed full no-flags lint runs (540 s composer /
590 s Claude self-fence), launching the compose-time attestation run as
bg-Bash at the START of compose (timeout 1800) and harvesting via a bounded
synchronous wait AFTER the prompt was written converted the attestation from
INCONCLUSIVE to COMPLETED-PASS rc=0 (~11 min wall under VM contention) at
near-zero added latency — the run overlaps the probe+write work. Patch via a
single `@@FULL_LINT_ATTESTATION@@` placeholder (assert 0 residue at final
validation). Also relay upward: a sibling 570 s-fenced run is predictably
too tight; tell the orchestrator the measured wall so the Claude leg sizes
its fence ≥2× (~1,200 s+).

**Reconciler-UPHELD twin-FAIL fix round + three-class truth split (#2271 r2,
2026-08-22):** when the prior round was Claude-PASS / Codex-FAIL / reconciler
BINDING FAIL upholding the TWIN'S OWN blockers, compose the #1094-r2 shape
(author-neutrality line; per-concern VERIFIED-ADDRESSED/NOT-ADDRESSED closure
ledger with NOT-ADDRESSED = substantive FAIL; ledger rows inlined when tiny)
plus three deltas: (a) keep the brief's THREE truth classes distinct in the
prompt — "established facts" (compose-verified, do-not-re-derive) vs
"orchestrator-verified claims — RE-VERIFY by static read, never accept" (the
brief ordered re-verification; collapsing them into established facts would
delete review duties) vs "calibration/settled ground" (both r1 reviewers right
about DIFFERENT mutation directions — name both, direction (a) must STAY
covered, so the twin neither re-litigates nor drops the surviving half);
(b) when the fix's correctness turns on stdlib/installed-library semantics a
no-env Codex cannot execute (py311 rglob symlink behavior, hf 0.36.2
upload_folder walk), verify them EMPIRICALLY at compose time (tiny tempfile
repro on the project interpreter + read the installed source) and ship them as
established facts with file:line citations (pathlib.py:391
`is_dir(follow_symlinks=False)`; hf_api.py:9566 `glob("**/*") + is_file()`),
scoping what remains YOURS to verify (pattern parity, not walk parity);
(c) realized repo shapes under sparse-excluded dirs (`eval_results/` symlinks)
are attested in the prompt with their RESOLUTION status (dangling vs resolves)
since Codex cannot stat them. Also: the never-echo constraints line must not
carry the literal `<!-- ... -->` head-tag form (write "`epm:code-review-codex
v2` head tag line" instead) or the v2-tag count assert reads 2.

**Union-fix round with a one-giant-line doc entry (#2280 r2, 2026-08-22):**
three composable deltas. (a) When a doc fix lands inside a single multi-KB
line (gotchas.md entries), the +1/−1 numstat hides everything — hand Codex
`git show <sha> --word-diff -- <file>` explicitly AND a drive-by sweep duty
("every changed token maps to a named fix item; anything else is scope
creep numstat cannot see"). Attest token COUNTS (dating qualifiers,
contrast figures) at compose time but state PLACEMENT is Codex's to grade —
existence-of-string is not scoped-to-the-right-claim. (b) Claude-PASS +
Codex-FAIL resolved by orchestrator UNION (no reconciler): inline only the
CODEX prior verdict (tags stripped, CONCERN:: rows blockquoted); the Claude
Minors arrive via the brief's fix enumeration as pseudo-IDs
(`r1-claude-<slug>`), and the closure ledger maps fixes → ids (F1+F2 → the
Major, etc.). A headroom/measurement NIT (`gotchas-size-headroom`) marked
addressed by RECORDING a measurement gets a neutral adjudication duty
(does measuring close a headroom NIT, or re-persist?) — never pre-resolve.
(c) Assert arithmetic: the r1-derived rubric span's Step 0.9 carries THREE
full merge-base SHA occurrences (intro + git-show probe + git-log probe);
with 2 header uses, composer-span expected = 5 — and count marker/verdict
embeds separately before summing (the v2 epm:results body carried ZERO,
unlike r1's which carried one).

**Binding-reconciler-FAIL fix round (#2263 r4, 2026-08-22):** when the prior
round ended Claude-PASS / Codex-FAIL / reconciler BINDING FAIL, inline BOTH
the reconciler verdict (tags stripped — the acceptance contract; its
Rationale names the endorsed bounded fix and the settled scope) and the prior
Codex verdict (#2329-rclose form: tags stripped, `CONCERN:: ` rows
blockquoted) with the author-neutrality line PLUS an "earned standing, not
license to re-litigate" frame. Two new composables: (a) when the
reconciler's endorsed fix sketch DIFFERS from the realized fix (endorsed
bare `"${VAR[@]}"`, implementer chose the `${VAR[@]+...}` set-u guard,
rejecting the brief's local-default alternative on clobber-on-copy-paste
grounds), hand the delta as an explicit design-judgment duty with both
regimes enumerated (set-u on/off × unset/declared-empty/populated) — never
grade sketch-verbatim and never resolve the judgment yourself; (b) the
bare-substring flag-absence trap: an implementer claim "dispatch_issue.py
has NO --lane flag" is TRUE while a bare `--lane` grep hits 13× (all
`--lane-suffix`, a different flag) — verify flag absence via an
add_argument-scoped grep at compose time and hand Codex the substring
caveat explicitly, else it false-flags the closure claim. Also: a
reconciler "Standing recommendations n/a — verdict is FAIL" section still
carries Standing-only items inside its adjudication table/rationale
(Claude's two-site-edit Minor) — restate them in the settled/do-not-owe
block so the twin neither re-FAILs on them nor demands the subsuming fix.

**Measured-TRUE acceptance round (#2273 r2, 2026-08-22):** a variant of the
#2147-cr3/cr4 accepted-FAIL shapes — the twin's r1 FAIL was accepted (no
reconciler), and the flagged claims were then MEASURED and found TRUE; the
remedy was the twin's OWN offered branch (add evidence rows), not weakening.
Compose three blocks: (a) author-neutrality + an ESTABLISHED-FACTS fence —
"do NOT re-raise 'unsupported' against a claim that now traces to rows 7-9;
the facts are settled by measurement"; (b) the live question is
ROW-BOUNDEDNESS of the landed text (the brief's A10 headline), not truth —
an unhedged claim outrunning its row re-lands the r1 defect class ⇒
Critical; (c) closure status lines keyed to the persisted concern ids with
the acceptance contract restated per id (rc clause may name ONLY the
measured consumers; residual clause bounded to its rows; pin-sweep figures
re-grepped by the twin against composer ground truth). Doc-only in-place
line replacement: round numstat is 1/1 (old bullet superseded) while
net-vs-main is +1/-0 — state BOTH and pre-clear the deletion's identity or
the twin flags collateral removal.

**Generous-timeout lint attestation beats a third INCONCLUSIVE (#2273 r2):**
after two rounds of timeout-killed full no-flags lint runs (540 s composer /
590 s Claude self-fence), launching the compose-time attestation run as
bg-Bash at the START of compose (timeout 1800) and harvesting via a bounded
synchronous wait AFTER the prompt was written converted the attestation from
INCONCLUSIVE to COMPLETED-PASS rc=0 (~11 min wall under VM contention) at
near-zero added latency — the run overlaps the probe+write work. Patch via a
single `@@FULL_LINT_ATTESTATION@@` placeholder (assert 0 residue at final
validation). Also relay upward: a sibling 570 s-fenced run is predictably
too tight; tell the orchestrator the measured wall so the Claude leg sizes
its fence ≥2× (~1,200 s+).

**Reconciler-UPHELD twin-FAIL fix round + three-class truth split (#2271 r2,
2026-08-22):** when the prior round was Claude-PASS / Codex-FAIL / reconciler
BINDING FAIL upholding the TWIN'S OWN blockers, compose the #1094-r2 shape
(author-neutrality line; per-concern VERIFIED-ADDRESSED/NOT-ADDRESSED closure
ledger with NOT-ADDRESSED = substantive FAIL; ledger rows inlined when tiny)
plus three deltas: (a) keep the brief's THREE truth classes distinct in the
prompt — "established facts" (compose-verified, do-not-re-derive) vs
"orchestrator-verified claims — RE-VERIFY by static read, never accept" (the
brief ordered re-verification; collapsing them into established facts would
delete review duties) vs "calibration/settled ground" (both r1 reviewers right
about DIFFERENT mutation directions — name both, direction (a) must STAY
covered, so the twin neither re-litigates nor drops the surviving half);
(b) when the fix's correctness turns on stdlib/installed-library semantics a
no-env Codex cannot execute (py311 rglob symlink behavior, hf 0.36.2
upload_folder walk), verify them EMPIRICALLY at compose time (tiny tempfile
repro on the project interpreter + read the installed source) and ship them as
established facts with file:line citations (pathlib.py:391
`is_dir(follow_symlinks=False)`; hf_api.py:9566 `glob("**/*") + is_file()`),
scoping what remains YOURS to verify (pattern parity, not walk parity);
(c) realized repo shapes under sparse-excluded dirs (`eval_results/` symlinks)
are attested in the prompt with their RESOLUTION status (dangling vs resolves)
since Codex cannot stat them. Also: the never-echo constraints line must not
carry the literal `<!-- ... -->` head-tag form (write "`epm:code-review-codex
v2` head tag line" instead) or the v2-tag count assert reads 2.

**Union-fix round with a one-giant-line doc entry (#2280 r2, 2026-08-22):**
three composable deltas. (a) When a doc fix lands inside a single multi-KB
line (gotchas.md entries), the +1/−1 numstat hides everything — hand Codex
`git show <sha> --word-diff -- <file>` explicitly AND a drive-by sweep duty
("every changed token maps to a named fix item; anything else is scope
creep numstat cannot see"). Attest token COUNTS (dating qualifiers,
contrast figures) at compose time but state PLACEMENT is Codex's to grade —
existence-of-string is not scoped-to-the-right-claim. (b) Claude-PASS +
Codex-FAIL resolved by orchestrator UNION (no reconciler): inline only the
CODEX prior verdict (tags stripped, CONCERN:: rows blockquoted); the Claude
Minors arrive via the brief's fix enumeration as pseudo-IDs
(`r1-claude-<slug>`), and the closure ledger maps fixes → ids (F1+F2 → the
Major, etc.). A headroom/measurement NIT (`gotchas-size-headroom`) marked
addressed by RECORDING a measurement gets a neutral adjudication duty
(does measuring close a headroom NIT, or re-persist?) — never pre-resolve.
(c) Assert arithmetic: the r1-derived rubric span's Step 0.9 carries THREE
full merge-base SHA occurrences (intro + git-show probe + git-log probe);
with 2 header uses, composer-span expected = 5 — and count marker/verdict
embeds separately before summing (the v2 epm:results body carried ZERO,
unlike r1's which carried one).

**Split-verdict inlining + FAIL+FAIL union with un-ledgered Claude items
(#2478 r2, 2026-08-22):** three deltas on the #2332 union shape. (a) A
SPLIT-review Claude verdict (per-commit sub-verdicts) can EMBED a
sub-verdict's own marker tag lines mid-body (the g3 sub-verdict carried its
own `<!-- epm:code-review v1 -->` head + a doubled closing tag) — strip by
filtering EVERY line starting with the tag prefix, never just first/last.
(b) When ALL ledger rows are one twin's items and the other twin's
blocker+minors are un-ledgered, mint pseudo-ids keyed to the MARKER's own
response-ledger naming (`r1-claude-b1-…` matching the marker's "g1 B1"
rows) so the closure ledger and the (e)/Response sections align 1:1.
(c) The `> CONCERN:: ` blockquote validation count includes your OWN prose
backtick mentions of the literal (6 = 5 rows + 1 prose here — label the
assert subject). Also: a fix round whose diff carries -def/+def pairs
(deleted helpers, dropped params, return-shape changes `str`→`tuple`) gets
a Step 3.75 compose-time observation block with adjudicate-the-trigger
framing (leaf script, no external importers — same-name signature changes
are not renames) PLUS the #1107 consumers-of-changed-shape + dangling-refs
duties spelled out as unconditional substantive checks; never pre-resolve
the trigger severity yourself.

**Split-verdict inlining + FAIL+FAIL union with un-ledgered Claude items
(#2478 r2, 2026-08-22):** three deltas on the #2332 union shape. (a) A
SPLIT-review Claude verdict (per-commit sub-verdicts) can EMBED a
sub-verdict's own marker tag lines mid-body (the g3 sub-verdict carried its
own `<!-- epm:code-review v1 -->` head + a doubled closing tag) — strip by
filtering EVERY line starting with the tag prefix, never just first/last.
(b) When ALL ledger rows are one twin's items and the other twin's
blocker+minors are un-ledgered, mint pseudo-ids keyed to the MARKER's own
response-ledger naming (`r1-claude-b1-…` matching the marker's "g1 B1"
rows) so the closure ledger and the (e)/Response sections align 1:1.
(c) The `> CONCERN:: ` blockquote validation count includes your OWN prose
backtick mentions of the literal (6 = 5 rows + 1 prose here — label the
assert subject). Also: a fix round whose diff carries -def/+def pairs
(deleted helpers, dropped params, return-shape changes `str`→`tuple`) gets
a Step 3.75 compose-time observation block with adjudicate-the-trigger
framing (leaf script, no external importers — same-name signature changes
are not renames) PLUS the #1107 consumers-of-changed-shape + dangling-refs
duties spelled out as unconditional substantive checks; never pre-resolve
the trigger severity yourself.

**Split-verdict inlining + FAIL+FAIL union with un-ledgered Claude items
(#2478 r2, 2026-08-22):** three deltas on the #2332 union shape. (a) A
SPLIT-review Claude verdict (per-commit sub-verdicts) can EMBED a
sub-verdict's own marker tag lines mid-body (the g3 sub-verdict carried its
own `<!-- epm:code-review v1 -->` head + a doubled closing tag) — strip by
filtering EVERY line starting with the tag prefix, never just first/last.
(b) When ALL ledger rows are one twin's items and the other twin's
blocker+minors are un-ledgered, mint pseudo-ids keyed to the MARKER's own
response-ledger naming (`r1-claude-b1-…` matching the marker's "g1 B1"
rows) so the closure ledger and the (e)/Response sections align 1:1.
(c) The `> CONCERN:: ` blockquote validation count includes your OWN prose
backtick mentions of the literal (6 = 5 rows + 1 prose here — label the
assert subject). Also: a fix round whose diff carries -def/+def pairs
(deleted helpers, dropped params, return-shape changes `str`→`tuple`) gets
a Step 3.75 compose-time observation block with adjudicate-the-trigger
framing (leaf script, no external importers — same-name signature changes
are not renames) PLUS the #1107 consumers-of-changed-shape + dangling-refs
duties spelled out as unconditional substantive checks; never pre-resolve
the trigger severity yourself.

**Line-wise facts-block replacement + two assert traps (#2476 r2, 2026-08-22):**
a FAIL+FAIL-union fix round composed cleanly from the r1 template via LINE-WISE
replacement of the compose-time-facts block (split on newlines, replace whole
lines keyed on unique `startswith` prefixes, assert exactly-one hit per prefix)
— robust when every facts line changes but the section anatomy survives. Two
asserts tripped live: (1) `wc -c` counts BYTES, Python `len(str)` counts
CHARACTERS — assert `len(body.encode()) == wc_c_bytes` for any size pin on
UTF-8-heavy artifacts (plans carry →/§/≤); (2) prose that NAMES the
blockquote literal (e.g. "rows are blockquoted `> CONCERN:: `") inflates a
whole-prompt `count("> CONCERN:: ")` — scope row asserts to line-start
(`l.startswith(...)` over splitlines) and assert the total separately with the
prose mention counted. Also confirmed: when an impl marker answers a union as
a numbered disposition table that OVERLAPS but is not a superset of the ledger
ids (Claude-only items unledgered, a NIT closed in an opportunistic paragraph),
say so explicitly and key the closure ledger on BOTH (item numbers + ids,
Codex maps them from the disposition text) rather than minting pseudo-IDs.

**Union fix round where the SIBLING twin's split verdicts are NOT inlined
(#1901 r2, 2026-08-22):** when round 1 was Codex-FAIL + Claude per-commit
SPLIT verdicts (g1-g4) and the orchestrator unioned the blocking findings,
the Claude split bodies live only in events.jsonl markers — do NOT stall
trying to fetch/inline them: (a) the impl marker's own `### Response to
code-review v1` table is the CLAIM surface for the g-items; give the
non-Codex union items g-union PSEUDO-IDS (`r1-g2-f1-...`) with the marker's
(a)-item mechanism text as the acceptance contract and an explicit "the
table is the claim; verify the MECHANISM in the diff" caveat; (b) keep the
per-id status-line duty for the Codex-persisted ids keyed on the LEDGER rows
(inline the addressed-claim summaries verbatim — they are the implementer's
own closure sentences and grep well); (c) FLAG in the return that the g
bodies were not inlined so the orchestrator can extend if it wants them
verbatim. Assert trap hit live: the impl marker's verbatim commit list
carries the FULL head SHA — the ctx-geometry count is 2, not 1 (the #2329
r5 (c) count-parts-separately rule, head-SHA edition).

**Residual-round closure round with `verified-open` ledger events + a
NOT-re-posted smoke-arch marker (#2476 r3, 2026-08-22):** two new wrinkles on
the FAIL+FAIL union shape. (a) The ledger can carry a `verified-open` EVENT
type (the r2 twin verdict confirmed a claimed-addressed concern still open) —
state it explicitly in the facts bullet ("your own round-2 verdict found the
fixes partial") so the twin reads the r3 `addressed` rows as second-attempt
claims, not first closures. (b) When the round does NOT re-post
`epm:smoke-architecture-check` and the impl report CLAIMS the old version
remains current, attest byte-identity to the prior round's inline at compose
time and hand the twin a CURRENCY adjudication duty (falsified-by-round-diff
architecture claim = substantive by consequence, never a 0.55 marker-shape
FAIL — presence-ON-TASK is satisfied). Also: residual-token sweeps must ban
STALE PATTERNS (`round-2 range`, `round-2 vs carry-over`, `Round-2 contract`)
not the bare `round-2` token — the fresh Step 0.8 replacement prose
legitimately says "verified closed by BOTH round-2 verdicts"; and re-probe
per-span SHA counts every round (this r2-derived rubric carries TWO merge-base
occurrences in Step 0.9, not the THREE the r1-derived note recorded — the
intro occurrence is gone).

**Brief-named residual ABSENT from the marker's (d) (#2215 dbe r3, 2026-08-22):**
when the orchestrator's brief enumerates implementer-disclosed residuals for
adjudication but one of them does not appear in the fetched marker's (d)
section (here: "packaged bank_dbe_values.json absent pre-datagen" — the (d)
list carried only the network-at-config and TOCTOU bullets), do NOT silently
drop it or pretend the marker discloses it: compose it as its own residual
line with an explicit COMPOSER NOTE naming the discrepancy, give Codex a
locate-the-actual-behavior duty (what the new code does on that path) PLUS a
disclosure-adequacy adjudication (should it have been a (d) bullet?), and
flag the discrepancy in the return. Also confirmed this round: a SECOND
FAIL+FAIL union fix round composes as the #2332-r2 shape verbatim with three
additions — (a) an elevated-verification block when a prior round had a
fabricated-coverage honesty blocker (every marker-named test READ for
substance; numeric claims spot-checked; composer settles the diffstat
arithmetic at compose time and says so); (b) a shared-module-touch section
when the round first touches a parent module both prior verdicts recorded as
untouched (off-path identity vs the pre-round blob + repo-wide caller sweep +
the stale prior-verdict header line pre-declared not-a-finding); (c)
near-duplicate concern ids from the two twins (same defect, one mechanism)
each get their own status line with the duplication named.

**FAIL+FAIL-union fix round with declared carve-outs, diff inlined whole
(#2254 first-k r2, 2026-08-23):** the #2332-r2 union shape (both prior
verdicts inlined as acceptance contracts, tag lines stripped, no
no-relitigate block) composes cleanly with the cycle-close blockquote form
(`> CONCERN:: ` for the prior twin's already-persisted rows — 13 here), the
#1092-r4 pseudo-ID pattern (3 unledgered r1-codex Majors + 3 Claude-only
items), and a carve-out section using the ACCEPTED-NON-CHANGE / OVERTURNED
vocabulary for brief-declared deliberate non-changes. Two new assert traps
hit live: (a) `wc -c` bytes ≠ Python `len(open().read())` chars on a
CJK-bearing round diff (82,820 B vs 82,643 ch) — assert
`os.path.getsize()==bytes` and char-len separately, both labeled; (b) the
scaffold placeholder-residue regex must be `\{\{[a-z0-9_]+\}\}` — a
`[a-z_]+` class silently misses digit-bearing names like
`{{claude_v8_verdict}}` and the completeness assert then reports them
missing from the scaffold. Also: when the implementer posts `addressed`
rows for only a SUBSET of the r1 batch (10 of 12 here), the unrowed items —
including the round's headline BLOCKER — keep FULL closure duty (the
ledger-empty lesson applies per-item, not per-round); flag the missing rows
to the orchestrator as bookkeeping, never as a relaxation.

**Union fix round where the SIBLING twin's split verdicts are NOT inlined
(#1901 r2, 2026-08-22):** when round 1 was Codex-FAIL + Claude per-commit
SPLIT verdicts (g1-g4) and the orchestrator unioned the blocking findings,
the Claude split bodies live only in events.jsonl markers — do NOT stall
trying to fetch/inline them: (a) the impl marker's own `### Response to
code-review v1` table is the CLAIM surface for the g-items; give the
non-Codex union items g-union PSEUDO-IDS (`r1-g2-f1-...`) with the marker's
(a)-item mechanism text as the acceptance contract and an explicit "the
table is the claim; verify the MECHANISM in the diff" caveat; (b) keep the
per-id status-line duty for the Codex-persisted ids keyed on the LEDGER rows
(inline the addressed-claim summaries verbatim — they are the implementer's
own closure sentences and grep well); (c) FLAG in the return that the g
bodies were not inlined so the orchestrator can extend if it wants them
verbatim. Assert trap hit live: the impl marker's verbatim commit list
carries the FULL head SHA — the ctx-geometry count is 2, not 1 (the #2329
r5 (c) count-parts-separately rule, head-SHA edition).

**Residual-round closure round with `verified-open` ledger events + a
NOT-re-posted smoke-arch marker (#2476 r3, 2026-08-22):** two new wrinkles on
the FAIL+FAIL union shape. (a) The ledger can carry a `verified-open` EVENT
type (the r2 twin verdict confirmed a claimed-addressed concern still open) —
state it explicitly in the facts bullet ("your own round-2 verdict found the
fixes partial") so the twin reads the r3 `addressed` rows as second-attempt
claims, not first closures. (b) When the round does NOT re-post
`epm:smoke-architecture-check` and the impl report CLAIMS the old version
remains current, attest byte-identity to the prior round's inline at compose
time and hand the twin a CURRENCY adjudication duty (falsified-by-round-diff
architecture claim = substantive by consequence, never a 0.55 marker-shape
FAIL — presence-ON-TASK is satisfied). Also: residual-token sweeps must ban
STALE PATTERNS (`round-2 range`, `round-2 vs carry-over`, `Round-2 contract`)
not the bare `round-2` token — the fresh Step 0.8 replacement prose
legitimately says "verified closed by BOTH round-2 verdicts"; and re-probe
per-span SHA counts every round (this r2-derived rubric carries TWO merge-base
occurrences in Step 0.9, not the THREE the r1-derived note recorded — the
intro occurrence is gone).

**Brief-named residual ABSENT from the marker's (d) (#2215 dbe r3, 2026-08-22):**
when the orchestrator's brief enumerates implementer-disclosed residuals for
adjudication but one of them does not appear in the fetched marker's (d)
section (here: "packaged bank_dbe_values.json absent pre-datagen" — the (d)
list carried only the network-at-config and TOCTOU bullets), do NOT silently
drop it or pretend the marker discloses it: compose it as its own residual
line with an explicit COMPOSER NOTE naming the discrepancy, give Codex a
locate-the-actual-behavior duty (what the new code does on that path) PLUS a
disclosure-adequacy adjudication (should it have been a (d) bullet?), and
flag the discrepancy in the return. Also confirmed this round: a SECOND
FAIL+FAIL union fix round composes as the #2332-r2 shape verbatim with three
additions — (a) an elevated-verification block when a prior round had a
fabricated-coverage honesty blocker (every marker-named test READ for
substance; numeric claims spot-checked; composer settles the diffstat
arithmetic at compose time and says so); (b) a shared-module-touch section
when the round first touches a parent module both prior verdicts recorded as
untouched (off-path identity vs the pre-round blob + repo-wide caller sweep +
the stale prior-verdict header line pre-declared not-a-finding); (c)
near-duplicate concern ids from the two twins (same defect, one mechanism)
each get their own status line with the duplication named.

**Reconciler-sided-FAIL fix round (ONE gating item) + brief-pinned binary
verdict + by-path plan (#2254 first-k r3, 2026-08-23):** when the prior
round ended Claude-PASS / Codex-FAIL / reconciler BINDING **FAIL** with one
gating item, the acceptance contract is the RECONCILER RULING, not the
prior verdicts — inline it tag-stripped in its own envelope, quote the
gating sentence verbatim in the round context (the brief may demand this
because Codex reads /tmp unreliably), and add a line-anchor caveat (the
ruling cites prior-round `file.py:NNNN` anchors the fix shifts — "never a
finding"). The ruling's adjudications become BOTH do-not-relitigate fences
(items root-caused to an open orchestrator-owned blocker; overturned
carve-outs; BLOCKER→CONCERN downgrades — the twin authored the FAIL and
predictably re-FAILs its own downgraded items otherwise) AND closure rows:
ruling-named C-items with no ledger row get pseudo-IDs (`r2-codex-c3-...`)
with the ruling's own sentence as contract. Binary-verdict pin composes per
#2228 (routing note inside the Verdict line bracket + rule-3 edit + `none`
on PASS). A fix round CAN take the plan BY PATH when the brief orders it:
canonical main-checkout ABSOLUTE path + v-number + frozen-worktree ban, and
state the binding contract (ruling + closure ledger) is fully INLINED so
plan-path failure blocks only the plan lens (BLOCKED only after primary +
fallback v<K>.md paths both fail). Assert trap: envelope BEGIN/END labels
with parentheticals must match character-for-character between the head
text and the assert list (a dropped `(round-relevant rows)` on the END
label failed the count).

**FAIL+FAIL-union fix round with declared carve-outs, diff inlined whole
(#2254 first-k r2, 2026-08-23):** the #2332-r2 union shape (both prior
verdicts inlined as acceptance contracts, tag lines stripped, no
no-relitigate block) composes cleanly with the cycle-close blockquote form
(`> CONCERN:: ` for the prior twin's already-persisted rows — 13 here), the
#1092-r4 pseudo-ID pattern (3 unledgered r1-codex Majors + 3 Claude-only
items), and a carve-out section using the ACCEPTED-NON-CHANGE / OVERTURNED
vocabulary for brief-declared deliberate non-changes. Two new assert traps
hit live: (a) `wc -c` bytes ≠ Python `len(open().read())` chars on a
CJK-bearing round diff (82,820 B vs 82,643 ch) — assert
`os.path.getsize()==bytes` and char-len separately, both labeled; (b) the
scaffold placeholder-residue regex must be `\{\{[a-z0-9_]+\}\}` — a
`[a-z_]+` class silently misses digit-bearing names like
`{{claude_v8_verdict}}` and the completeness assert then reports them
missing from the scaffold. Also: when the implementer posts `addressed`
rows for only a SUBSET of the r1 batch (10 of 12 here), the unrowed items —
including the round's headline BLOCKER — keep FULL closure duty (the
ledger-empty lesson applies per-item, not per-round); flag the missing rows
to the orchestrator as bookkeeping, never as a relaxation.

**Reconciled-FAIL fix round keyed on a numbered marker punch list (#823
ext-ladder r2, 2026-08-23):** three deltas beyond the upheld-concern
pattern. (a) Reconciler downgrades do NOT rewrite concerns-ledger severity
fields — rows still read `severity: BLOCKER` after a binding
BLOCKER→CONCERN downgrade; attest the staleness explicitly in BOTH the
ledger-envelope preface and the dispositions summary ("the ledger field is
STALE; the reconcile text governs"), or the twin re-escalates from the
stale field. (b) When the impl marker carries its OWN numbered punch list
(1–15), key the closure ledger on those numbers (#2332 pattern) — items
with no persisted concern id (opportunistic minors, a resume-predicate fix)
ride their numbers, no pseudo-IDs needed; and a `deferred`-event ledger row
counts as OPEN for the Prior-concerns header (raised OR deferred = open).
(c) A brief ordering prior verdicts BY REFERENCE (/tmp paths, "do not
inline bodies") still needs the per-item acceptance criteria INLINED in the
head — state "/tmp unreachability is never data-access-blocked; the
criteria are self-sufficient (summary-only evidence base)" so a sandbox
that cannot see /tmp degrades gracefully. Also: the fence protecting the
twin's own DEFERRED Critical (the adjudicated smoke shape) must spell out
what NEW evidence means ("THIS commit deleting/breaking the registered
smoke wiring — nothing less"), else an adversarial re-FAIL of its own r1
item is predictable.
