---
description: Reviewing trigger-dense / security-adjacent artifacts (guard hooks, destructive-command fixtures, refusal/jailbreak corpora) without filter kills — findings by reference, durable verdict first, windowed reads; brief composition (first-pass #1503, revision-round #1413); orchestrator poll/forensics digests (#1546); ordinary guard-surface turns digest-only (#1563); real-corpus datagen decomposition (#1748); judge-run monitoring digest-grain (#1871). Prevention-side sibling of CLAUDE.md § Spurious usage-policy refusals (#1058, #1152).
paths:
  - ".claude/hooks/*.sh"
  - "scripts/guard_*.sh"
---

# Trigger-dense artifact review — reference, don't quote; verdict before summary

**Fires when:** a code-reviewer, reconciler, fact-checker, or any
review-/verification-role subagent's artifact under review is
TRIGGER-DENSE — its own vocabulary can trip the content filter when
repeated in generated text — or the ORCHESTRATOR composes a FIRST-PASS
fact-check / critique / plan-review brief whose TARGET files are such
artifacts (§ First-pass briefs, #1503), or composes a revision-round /
bounce / reconcile brief from findings-bearing verdicts on such a round
(§ Revision-round briefs, #1413), or the ORCHESTRATOR ingests
run-failure / forensics text on its own turn — a poll tick reporting
stalled/dead, crash-persist artifacts, a guard hook's BLOCKED runtime
output (§ Orchestrator poll/forensics turns, #1546), or the ORCHESTRATOR
runs ANY ordinary turn on a round whose task or diff targets such
artifacts (§ Orchestrator ordinary turns, #1563), or ANY context —
orchestrator or subagent — reads judge OUTPUT files while monitoring or
spot-checking a judge run (§ Judge-run monitoring reads, #1871).
Recognition heuristic (any one suffices):

- guard / security hook scripts (`.claude/hooks/*.sh`, `scripts/guard_*.sh`)
  and their tests — a guard's job is to name the attack it blocks, so its
  text is attack-shaped by construction (destructive-git payloads, fail-open
  redirect probes, write-then-execute channels; #1098: refused as
  "violative cyber content");
- test fixtures / lint allowlists that enumerate destructive or gated
  command shapes (fixture lists of banned invocations, hook-bypass probes);
- refusal / jailbreak / harmful-content corpora and question banks (reads
  already digest-only per the `guard_harmful_bank_read.sh` hook — whose
  Read arm also denies wholesale >256 KB corpus-file reads, #1217 — +
  code-reviewer.md § Harmful-content corpora digest note; #866, #1073),
  incl. unscreened real-world corpora (LMSYS/WildChat-class) whose rows
  routinely carry in-corpus jailbreak/explicit text (#1073, #1739);
- judge OUTPUT files — per-row rationale-bearing judge JSONs/JSONL, judge
  caches, Batch-API result files: a judge rationale QUOTES the judged
  completion, so a judge run over harmful-content / refusal / real-corpus
  pools is trigger-dense at OUTPUT grain even when the monitoring turn
  never touches the input pool (§ Judge-run monitoring reads, #1871);
- steering / causal-intervention APPLICATION surfaces — steering scripts,
  intervention configs + hyperparameter blocks — and any surface whose text
  carries intervention-control phrasing (#1415; #1769: an owner session was
  refusal-killed grepping steering-script hyperparameters on an ordinary
  turn); plain direction-extraction / analysis code with mechanistic
  vocabulary does NOT match on category alone;
- the diff or plan under review EDITS any of the above, or a verdict body
  you must adjudicate QUOTES such content (the reconciler case).

This rule is PREVENTION — it lowers the refusal surface BEFORE a kill. The
ORCHESTRATOR-side recovery ladder (CLAUDE.md § "Spurious usage-policy
refusals", rungs a-g — rephrase, per-subagent model pin b2,
compose-yourself) is unchanged and stays authoritative AFTER a kill;
reviewers never attempt model pins themselves (pins live on the Agent call).

## The four disciplines (#1058: three spawns died re-discovering 1–3; #1152: the parent died to a return-text recap, discipline 4)

1. **Findings by reference, never by literal.** Cite gated content by
   `file:line`, test-case id / fixture index, or abstract class
   ("destructive-git payload", "fail-open redirect probe") — NEVER write
   out a gated command literal in ANY generated text: verdict body, marker
   note, diff-hunk quote, fix sketch, or chat summary. Where a report
   template asks to "quote the code" (code-reviewer.md Step 7 Evidence
   slot; reconciler.md Rationale), degrade the slot to
   `<file>:<line> (<abstract description>)` for such lines. This changes
   the QUOTING FORM of evidence, not its presence: grounding rules
   (code-reviewer Rule 12, reconciler Rule 9 cite-or-drop) still bind, and
   grep-the-literal verification (code-reviewer Rule 9) still runs — the
   grep hit's file:line goes in the verdict; the matched text does not.
2. **Durable verdict FIRST; chat summary last and optional.** Write the
   verdict body to its file and POST the marker (`epm:code-review` /
   `epm:review-reconcile` via `task.py post-marker`) BEFORE composing any
   closing chat text — a filter kill on the summary then costs nothing (the
   orchestrator's durable-verdict-first probe finds the marker, SKILL.md
   Step 5b). In-context modes (adversarial-planner Phase 2 lenses;
   reconciler `mode: in-context`) have no marker: put the role-tagged
   verdict block FIRST in the returned text, any extra prose after it.
   **Guard/hook-surface rounds — file-first ordering (post-marker before
   composing the report; #1673/#1675/#1676/#1687).** On a trigger-dense
   round targeting `.claude/hooks/*.sh` or `scripts/guard_*.sh`, the
   verdict body is composed DIRECTLY into its file via the `Write` tool —
   never as inline assistant text or reasoning first — then posted with
   `uv run python scripts/task.py post-marker <N> epm:code-review
   --file <path>` (OMIT `--version` — it auto-derives max+1; the round
   lives in the body's head sentinel; #1804) (or the reconciler analogue
   `epm:review-reconcile`; `--file`, never `--note`, because the marker
   body itself is trigger-dense and must ride the file path). The
   reviewer's ordered tool-call sequence on such rounds is `Write` (verdict
   file) → `Bash` (post-marker `--file`) → return text; composing the
   verdict body as inline assistant text on a turn that then gets killed
   by the filter destroys the verdict before it ever reaches the file
   (four refusal kills on the #1667–#1689 wave: #1673/#1675/#1676/#1687).
   The in-context RETURN TEXT (marker mode) or the role-tagged verdict
   block (in-context mode) is composed AFTER the marker lands and stays
   under the discipline 4 shape (verdict + pointer + counts, no findings
   recap). Pairs with SKILL.md Step 5b's durable-verdict-first probe: with
   the marker landed first, a final-turn refusal costs the summary
   (recoverable via a discipline-4-clean re-post or the orchestrator's
   own summary composition) rather than the verdict (a full respawn).
3. **Windowed reads; excerpts over originals.** Prefer orchestrator-provided
   pre-materialized excerpt files with stated read budgets when the brief
   names them. Reading the artifact directly: Grep for the finding's anchor
   first, then Read ≤~120-line windows around it — never wholesale-read a
   >800-line trigger-dense file (wholesale paging both maximizes trigger
   vocabulary in context and killed #1058's fourth spawn by autocompact;
   #866's bank-paging kills are the same family).
4. **Marker-mode RETURN TEXT = verdict + pointer + counts — never a
   findings recap.** The final return text is the one channel guaranteed
   to enter the PARENT orchestrator's context, and an ABSTRACT recap —
   discipline-1-clean, zero literals — still carries enough attack
   vocabulary to wedge the parent (#1152: a reviewer posted its marker
   correctly, then returned a PASS summary recapping the guard findings
   abstractly; the orchestrator died on 3 consecutive
   usage-policy-refused turns — the #1074 wedge shape). Binds every
   review role whose deliverable is durable elsewhere (code-reviewer,
   reconciler marker mode, interpretation-critic, clean-result-critic,
   the Codex twin wrappers' return text). ALLOWED in the return text:
   the verdict word; the marker kind + version posted (or verdict-file
   path); per-severity counts ("3 Critical, 2 Major"); purely
   procedural notes (read budgets, respawn notes, a discipline-1-clean
   workflow-fix-candidate block). BANNED: finding titles or
   descriptions, attack-shape names, quoted or paraphrased
   command/payload shapes, and file:line-plus-description enumerations
   of the findings — the parent reads the verdict body from the
   marker/file with windowed reads when it needs detail. In-context
   modes (adversarial-planner Phase 2 lens critics; reconciler
   `mode: in-context`) are EXEMPT for the role-tagged verdict block
   itself — the block IS the deliverable and stays findings-bearing, by
   file:line reference per discipline 1 — but NOTHING follows the
   block except a discipline-1-clean workflow-fix-candidate block: no
   closing prose, no recap after it. On trigger-dense rounds any
   workflow-fix-candidate block (either mode) stays file-pointer-minimal
   — reference the gap by file:line / abstract class, no shape
   descriptions. Discipline 2 orders the return text LAST; this
   discipline constrains its CONTENT.

## Reconciler-specific

Your inputs are two verdict BODIES that may already quote gated literals
(e.g. one reviewer followed this rule and one did not). Do not RE-quote
them: adjudicate per finding id / file:line, and where you must
characterize a disputed quote, paraphrase it abstractly. Marker mode: post
`epm:review-reconcile` before any closing chat text (discipline 2); the
closing text itself is discipline-4-minimal.

## First-pass briefs (composition-side, #1503)

Fires for the ORCHESTRATOR composing any FIRST-PASS subagent brief whose
TARGET files include a trigger-dense artifact per the recognition
heuristic above — the Phase-1.5 fact-checker brief, the Phase-2 critic
and consistency-checker briefs, a plan-review or first code-review
brief. No verdict exists yet, so § Revision-round briefs cannot fire;
the duties attach to the brief itself:

1. Name the guard-surface target files by PATH (plus the specific claims
   or assumptions to check against them) — never inline their content
   into the brief (discipline 1 governs the form).
2. Instruct windowed reads: grep for the anchor first, then Read
   ≤~120-line windows (discipline 3); where the orchestrator can,
   pre-materialize excerpt files and name them + a read budget (the
   issue-SKILL Step 5a pattern).
3. Instruct the subagent to return findings by reference — disciplines 1
   and 4 bind its output from the first spawn.
4. Keep the brief's own text in neutral gate vocabulary (CLAUDE.md
   § Spurious usage-policy refusals, rung (e)).
5. A brief whose target or supporting context includes a guard hook's
   BLOCKED runtime output passes that text by file reference (the hook
   script path, or the transcript/log path + line) or marker reference —
   never inlined into the brief (2026-07-18: 3 content-filter kills from
   briefs inlining a hook's BLOCKED message).

Rationale: rung (e) neutralizes first-pass brief VOCABULARY, but the
READ discipline previously attached only to review roles and revision
briefs — first-pass fact-checkers/critics paged whole guard files and
were filter-killed before any recovery rung fired (#1436/#1443: 4
first-pass kills).

Datagen sibling: an implementer / experimenter brief for a DATAGEN
pipeline that ingests a real-world corpus or a harmful-content bank ALSO
starts from § Real-corpus datagen briefs — first-pass decomposition
default (#1748) below — the round decomposition duties 1–5 here compose
with.

## Real-corpus datagen briefs — first-pass decomposition default (#1748)

Fires for the ORCHESTRATOR composing implementer / experimenter briefs
for a DATAGEN pipeline that INGESTS a real-world corpus
(LMSYS/WildChat-class unscreened user text — the recognition heuristic's
corpora bullet) or a harmful-content bank. The four-part round structure
below is the FIRST-PASS default — the starting decomposition, never a
fallback ladder reached only after dead spawns:

- (a) **Data-plane code rounds stream ZERO real-corpus text.** All
  pipeline code — loaders, filters, staging, generation/capture
  entrypoints — is written and tested against synthetic fixtures only;
  the brief states "no real-data/network execution this round" (#1739
  round B1's marker-recorded brief line, `epm:progress` v5/v6).
- (b) **Bounded ingestion probes ship as content-opaque CLIs the
  ORCHESTRATOR runs** — counts-only stdout (kept / per-filter rejection
  counters per dataset), fail-loud on kept=0; real corpus text never
  enters ANY agent context. (The gotchas.md tiny-real ingestion-probe
  class still binds — this clause routes WHO runs the probe and WHAT it
  prints, not whether it runs.)
- (c) **Micro-scoped rounds sized to survive autocompact thrash** — the
  #1090 sequential split (skeleton/loaders round, data-plane round,
  fits/figures/report round), each a self-contained commit-manifest
  unit on the DEFAULT session model.
- (d) **Report markers orchestrator-composed from durable evidence** —
  smoke logs, commit subjects, round manifests in `epm:progress` —
  pre-authorized for the report stage of such rounds, not an escalation
  reached only after dead report-stage spawns (CLAUDE.md refusal
  rung (c) is the recovery-side sibling).

(#1739: a real-corpus datagen pipeline converged on exactly (a)-(d)
only after repeated spawn attrition — ~9 dead spawns over ~5h across
implementer/planner/consistency-checker roles; rounds then ran clean
under (a)-(c), and (d) closed the report stage after two agent attempts
had died there.)

## Revision-round briefs (composition-side, #1413)

Fires for the ORCHESTRATOR composing any follow-on brief from
findings-bearing verdicts on a trigger-dense round: a planner Phase 3
revision brief (merged critique), an implementer bounce brief
(union-blocker list), a reconciler brief, an analyzer revision brief
(critique events), a v2 auto-revise brief. Pass the findings BY
REFERENCE — the posted marker kind + version on `events.jsonl`, and/or
the verdict/critique FILE path (`/tmp` output file, or
`.claude/plans/issue-<N>-critique-r<K>.md`) — NEVER inline the findings
text into the brief; the receiving subagent reads it itself with
windowed grep-anchored reads (discipline 3) and applies discipline 1 to
its own output. The in-context COLLECTION exemption (§ File-only Codex
verdict posting) does not extend to brief composition: re-emitting
collected findings into a fresh Agent brief is a fresh generated-text
exposure (#1413: a guard-surface revision brief inlined the round's
Must-Fix text verbatim — one spawn refusal-killed, a second dispatched
truncated mid-sentence). After any refusal on a spawn turn, the
CLAUDE.md rung-(g) dispatched-prompt completeness check applies to
revision spawns exactly as to first spawns.

## Orchestrator poll/forensics turns (ingest-side, #1546)

Fires for the ORCHESTRATOR itself — an /issue, /issue-v2, /campaign, or
tick session — whenever run-failure or forensics text lands on its OWN
turn: a poll tick reporting stalled/dead (the poll JSON's log-tail
excerpt field), pod/VM stderr or crash-log tails, crash-persist
artifacts (`crash_report.json` / `workload.log` under the HF
`issue<N>_partial/` prefix), lane/queue state dumps, or a guard hook's
BLOCKED runtime output. The composition-side sections above protect
SUBAGENT briefs; this section protects the orchestrator's own context —
the one context whose loss costs a session respawn (CLAUDE.md
§ Spurious usage-policy refusals rung (f); #1546: 7 content-filter kills
on one session's poll turns while it paged raw crash-forensics tails).

1. **Digest-first, unconditionally.** Ingest failure text as STRUCTURAL
   DIGESTS: bounded pattern COUNTS
   (`grep -ciE 'error|traceback|killed|OOM' <log>`), exit codes,
   phase/lane fields from the poll JSON, and file references (path +
   byte size + mtime) for the tail itself. The CLAUDE.md § Monitoring
   matched-line grep and the `tail -50` bound remain the CEILING for any
   raw-line read — never `cat` a multi-KB tail (`guard_log_dump.sh`
   blocks local dump shapes; SSH-remote reads are NOT hook-covered) —
   and never RE-page the same tail across consecutive poll turns:
   repetition is what accumulates the refusal surface. Mechanized at the
   producer since #1556: tag the task (`task.py add-tag <N>
   trigger-dense`) and `scripts/poll_pipeline.py` emits exactly this
   structural digest in place of the raw `log_tail_excerpt` on every
   tick.
2. **Routing stays script-side.** `failure_class` routing runs through
   `scripts/failure_classifier.py --body - --log "$LATEST_LOG_PATH"`
   (issue SKILL.md Step 7): the SCRIPT reads the log and prints a
   one-line verdict. Do not re-read the raw log inline to re-derive what
   the classifier computes; pass log content to it by PATH / stdin,
   never by pasting text into your own turn.
3. **Trigger-dense runs escalate to a fresh-context reader.** When the
   RUN is trigger-dense per the recognition heuristic above — keyed on
   the WORKLOAD, knowable before any read: it trains/evals on
   guard/security surfaces, harmful-content or refusal corpora, or
   jailbreak banks, or its failure text embeds a guard hook's BLOCKED
   output — even bounded excerpts stay OUT of the orchestrator's
   context: dispatch a fresh-context subagent to read windowed excerpts
   (discipline 3) and return a digest per disciplines 1/4 — counts,
   file:line references, a routing recommendation, no quoted text. A
   subagent's context is disposable; the orchestrator's is the session.
4. **Guard-hook BLOCKED output by reference.** A hook's BLOCKED message
   is attack-shaped by construction (it names what it blocks). Reference
   it by hook path + a ≤80-char reason slice (the issue-tick
   GATE-TRANSITION precedent) in turn text and marker notes; on a
   trigger-dense run the `epm:failure` note carries digest + artifact
   path, not the raw tail (the note is re-read by later turns and tick
   digests — raw text there poisons every future read). Brief-side duty:
   § First-pass briefs item 5.
5. **Durable record is unchanged.** The raw tail stays WHERE IT IS
   durable (the pod/VM log file, the crash-persist upload) — this
   section changes what enters the orchestrator's CONTEXT and generated
   text, never whether forensic text is persisted.

## Judge-run monitoring reads (ingest-side, #1871)

Fires for ANY context — orchestrator or subagent — reading judge OUTPUT
files (per-row rationale-bearing judge JSONs/JSONL, judge caches,
Batch-API result files) while monitoring or spot-checking a judge run —
HEALTHY runs included (§ Orchestrator poll/forensics turns is
failure/forensics-keyed and does not fire there).

1. **Digest-grain by default, ALL judge monitoring.** Read judge outputs
   as counts + structure — row counts, per-verdict / score-class
   tallies, error / parse-failure / refusal counts, drop tallies — via
   `grep -c` / `jq` aggregation or the pipeline's own summary JSON;
   counts are what monitoring needs anyway. The content-drop vs
   transport-loss split the judged-DV report already requires
   (`.claude/rules/llm-judging.md` rules 9/24) is exactly this digest
   grain.
2. **Single-row cherry-picks by line offset / row id only when genuinely
   needed** (a parse-failure repro, a scoring sanity check); prefer
   structured fields (score, verdict token, `stop_reason`) over
   rationale text; an unavoidable rationale excerpt stays a bounded
   slice — never the full field, never multiple rows' rationales.
3. **Hard bound on sensitive-pool runs.** When the judged pool matches
   the recognition heuristic (harmful-content / refusal / real-corpus),
   never wholesale-page a judge output file (`cat`, unbounded `tail`,
   full-file Read) into ANY context — the orchestrator's especially
   (its loss costs the session; CLAUDE.md refusal rung (f)). CLAUDE.md
   rung (d)'s digest prescription for harmful-completion ingest is the
   recovery-side sibling; this section is its prevention-side
   monitoring twin.
4. **Ownership split vs § Orchestrator ordinary turns.** On a round
   already trigger-dense per that section's Step-0 predicate, this
   section is the judge-output instantiation of its item 2; its OWN
   coverage is a judge-monitoring read on a round whose task or diff does
   NOT match that predicate (the incident's shape).

## Orchestrator ordinary turns (authoring + own-reads, #1563)

Fires for the ORCHESTRATOR itself — an /issue, /issue-v2, /campaign, or
tick session — on EVERY ordinary turn of a round whose task or diff
targets a trigger-dense artifact per the recognition heuristic above
(knowable at Step 0: the task body's target/scope lines, a
`workflow_fix_target:` naming a guard surface, or the round's diff
pathspec). § Orchestrator poll/forensics turns owns FAILURE/forensics
text; this section owns everything else the orchestrator does on such a
round, Step-0 state read through Step-10d merge narration. The #1538
wedge was CONTEXT ACCUMULATION on ordinary turns — guard-surface tool
output paged in during normal reads plus the text the orchestrator then
authored; every later wake turn died, and recovery was a fresh respawn
(CLAUDE.md rung (f)).

1. **Authored text — reference, never quote, on every surface.**
   Progress/marker notes (`epm:progress`, `epm:failure`, followup-scope
   and step-completed notes), stage-dispatch breadcrumbs, plan-approval
   summaries, chat narration, and commit/PR messages name guard content
   by file path + abstract class ("extends the blocked-pattern list",
   "adds a deny-list entry") — never a blocked-command literal, an
   attack-shape enumeration, or a diff-hunk quote. The § poll/forensics
   item-4 ≤80-char reason-slice bound generalizes: at most ONE ≤80-char
   slice of guard-surface text per authored artifact, and prefer zero.
2. **Own-turn ordinary reads — counts-first, windowed, or delegated.**
   Verifying guard-surface work uses name/stat/count forms first
   (`git diff --stat`, `--name-status`, `grep -c`), then grep-anchored
   ≤~120-line windows (discipline 3) only where content is genuinely
   needed; NEVER wholesale-read a guard file or its full diff body into
   your own context, and route content-shaped checks through a
   script/pipeline that prints a verdict or count wherever one exists.
   Where content-level verification is unavoidable at depth, dispatch a
   fresh-context subagent that returns a digest per disciplines 1/4 — a
   subagent's context is disposable; the orchestrator's is the session.
3. **No re-paging across turns.** Accumulation is the wedge mechanism:
   the same guard-surface content re-entering context turn after turn is
   what turns one risky read into every-turn deaths. Read once, carry
   forward your OWN one-line digest; never re-read the same
   guard-surface content on later turns (the § poll/forensics item-1
   no-repeat clause, extended to ordinary reads).
4. **Marker bodies are re-read — keep them digest-only.** `events.jsonl`
   notes are re-ingested by later turns, tick digests, and successor
   sessions; raw guard-surface text in a note poisons every future read
   (same rationale as § poll/forensics item 4).
5. **Recovery unchanged.** After a kill the CLAUDE.md § Spurious
   usage-policy refusals ladder stays authoritative (rung (f) fresh
   respawn is what recovered #1538); this section exists so it is not
   needed.

(#1538: two consecutive /issue orchestrator sessions wedged on a
guard-hook grep-pattern round — the first kill fired immediately after
guard-surface tool results entered context on an ordinary working turn;
the task completed only in a third watcher-respawned session, ~1h+ lost.)

## What this rule does NOT change

- The review bar. Every finding still needs a concrete artifact location;
  a finding a reader cannot locate from file:line + description alone is
  mis-scoped — that is a reason to sharpen the reference, never to quote.
- Read-side digest rules. Harmful-bank reads stay governed by
  `guard_harmful_bank_read.sh` (its Read arm also gates wholesale corpus
  reads, #1217) + the corpora digest note; diff-body sizing
  by `.claude/rules/diff-size-budget.md`. This rule adds the
  generated-TEXT + verdict-ordering discipline those read-side rules do
  not cover. § Orchestrator ordinary turns tightens the guard-surface
  subset of ordinary orchestrator reads (content discipline, not just
  size).
- The CLAUDE.md § Monitoring recipe on ordinary runs and the mechanical
  `guard_log_dump.sh` dump blocker. § Orchestrator poll/forensics turns
  adds the counts-first / no-repeat / trigger-dense-escalation discipline
  those do not cover — it tightens, never replaces, the existing
  log-read ceiling.
- Judging methodology. Rubrics, scales, drop rules, and every other
  judged-DV design choice stay owned by `.claude/rules/llm-judging.md`;
  § Judge-run monitoring reads governs only what judge OUTPUT enters
  context.

## Files of record

Incidents: #1058, #1098, #1092, #866, #1090, #1152 (discipline 4), #1413
(§ Revision-round briefs), #1436/#1443/#1503 (§ First-pass briefs), #1546
(§ Orchestrator poll/forensics turns + § First-pass briefs item 5),
#1538/#1563 (§ Orchestrator ordinary turns), #1739/#1748 (§ Real-corpus
datagen briefs), #1871 (§ Judge-run monitoring reads).
Enforcing pointers: `.claude/agents/code-reviewer.md` § Context budget;
`.claude/agents/reconciler.md` § Rules (Rule 11);
`.claude/skills/issue/SKILL.md` Step 5a, § File-only Codex verdict posting
(#1275), Step 6d.2 (#1546), § Orchestration Procedure preamble (#1563),
Step 5d (#1413);
`.claude/skills/adversarial-planner/SKILL.md` Phases 1.5/2 (#1503) +
Phase 3 (#1413).
