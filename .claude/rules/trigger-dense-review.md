---
description: Reviewing trigger-dense / security-adjacent artifacts (guard hooks, destructive-command test fixtures, refusal/jailbreak corpora) without filter kills — findings by reference, durable verdict first, windowed reads + brief composition for such targets (first-pass #1503, revision-round #1413; findings by reference); orchestrator poll/forensics turns ingest run-failure text as structural digests (#1546). Prevention-side sibling of CLAUDE.md § Spurious usage-policy refusals (recovery). Origin incidents #1058, #1152.
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
output (§ Orchestrator poll/forensics turns, #1546). Recognition heuristic
(any one suffices):

- guard / security hook scripts (`.claude/hooks/*.sh`, `scripts/guard_*.sh`)
  and their tests — a guard's job is to name the attack it blocks, so its
  text is attack-shaped by construction (destructive-git payloads, fail-open
  redirect probes, write-then-execute channels; #1098: refused as
  "violative cyber content");
- test fixtures / lint allowlists that enumerate destructive or gated
  command shapes (fixture lists of banned invocations, hook-bypass probes);
- refusal / jailbreak / harmful-content corpora and question banks (reads
  already digest-only per the `guard_harmful_bank_read.sh` hook +
  code-reviewer.md § Harmful-content corpora digest note; #866, #1073);
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
were filter-killed before any recovery rung fired (2026-07-17: 4 kills
across #1436/#1443 — fact-checker ×2, Alternatives critic ×2 — ~35+ min
recovered via rung (b2)).

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
§ Spurious usage-policy refusals rung (f); 2026-07-18: 7 content-filter
kills on one session's poll turns while it paged raw crash-forensics
tails; session replaced).

1. **Digest-first, unconditionally.** Ingest failure text as STRUCTURAL
   DIGESTS: bounded pattern COUNTS
   (`grep -ciE 'error|traceback|killed|OOM' <log>`), exit codes,
   phase/lane fields from the poll JSON, and file references (path +
   byte size + mtime) for the tail itself. The CLAUDE.md § Monitoring
   matched-line grep and the § 429-pacing `tail -50` bound remain the
   CEILING for any raw-line read on an ordinary run — never `cat` a
   multi-KB tail (`guard_log_dump.sh` blocks the local dump shapes
   mechanically; SSH-remote reads and bounded re-reads are NOT
   hook-covered, so the discipline there is yours) — and never RE-page
   the same tail across consecutive poll turns: repetition is what
   accumulates the refusal surface.
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

## What this rule does NOT change

- The review bar. Every finding still needs a concrete artifact location;
  a finding a reader cannot locate from file:line + description alone is
  mis-scoped — that is a reason to sharpen the reference, never to quote.
- Read-side digest rules. Harmful-bank reads stay governed by
  `guard_harmful_bank_read.sh` + the corpora digest note; diff-body sizing
  by `.claude/rules/diff-size-budget.md`. This rule adds the
  generated-TEXT + verdict-ordering discipline those read-side rules do
  not cover.
- The CLAUDE.md § Monitoring recipe on ordinary runs and the mechanical
  `guard_log_dump.sh` dump blocker. § Orchestrator poll/forensics turns
  adds the counts-first / no-repeat / trigger-dense-escalation discipline
  those do not cover — it tightens, never replaces, the existing
  log-read ceiling.

## Files of record

Incidents: #1058 (3 review-role refusal kills + 1 autocompact death; these
mitigations validated ad hoc), #1098 (guard-vocabulary refusals, 2 reviewer
kills + ~2.7h orchestrator wedge), #1092 (implementer refusal kills;
orchestrator rung b2), #866 (bank-text paging kills), #1090
(refusal-truncated Agent spawns), #1152 (a findings recap in a reviewer's
return text wedged the parent orchestrator — discipline 4), #1413 (a
revision brief inlining a round's findings text — § Revision-round briefs),
#1436/#1443 (4 first-pass kills on guard-surface targets — § First-pass
briefs, #1503), 2026-07-18 (7 poll-turn content-filter kills wedged the
#1481 session while it paged raw crash-forensics tails, session
replaced, + 3 brief kills from inlined hook-BLOCKED text —
§ Orchestrator poll/forensics turns + § First-pass briefs item 5,
#1546).
Enforcing pointers:
`.claude/agents/code-reviewer.md` § Context budget (READ FIRST);
`.claude/agents/reconciler.md` § Rules (Rule 11);
`.claude/skills/issue/SKILL.md` Step 5a (orchestrator-side excerpt-file
pre-materialization);
`.claude/skills/issue/SKILL.md` § File-only Codex verdict posting
(orchestrator-side posting path, #1275);
`.claude/skills/issue/SKILL.md` Step 6d.2 (poll-loop forensics-ingest
pointer, #1546);
`.claude/skills/adversarial-planner/SKILL.md` Phase 3 +
`.claude/skills/issue/SKILL.md` Step 5d (revision-brief composition
pointers, #1413);
`.claude/skills/adversarial-planner/SKILL.md` Phase 1.5 + Phase 2
(first-pass brief-composition pointers, #1503).
