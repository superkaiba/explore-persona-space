---
description: Reviewing trigger-dense / security-adjacent artifacts (guard hooks, destructive-command test fixtures, refusal/jailbreak corpora) without filter kills — findings by reference, durable verdict first, windowed reads. Prevention-side sibling of CLAUDE.md § Spurious usage-policy refusals (recovery). Origin incidents #1058, #1152.
paths:
  - ".claude/hooks/*.sh"
  - "scripts/guard_*.sh"
---

# Trigger-dense artifact review — reference, don't quote; verdict before summary

**Fires when:** a code-reviewer, reconciler, or any review-role subagent's
artifact under review is TRIGGER-DENSE — its own vocabulary can trip the
content filter when repeated in generated text. Recognition heuristic (any
one suffices):

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

## What this rule does NOT change

- The review bar. Every finding still needs a concrete artifact location;
  a finding a reader cannot locate from file:line + description alone is
  mis-scoped — that is a reason to sharpen the reference, never to quote.
- Read-side digest rules. Harmful-bank reads stay governed by
  `guard_harmful_bank_read.sh` + the corpora digest note; diff-body sizing
  by `.claude/rules/diff-size-budget.md`. This rule adds the
  generated-TEXT + verdict-ordering discipline those read-side rules do
  not cover.

## Files of record

Incidents: #1058 (3 review-role refusal kills + 1 autocompact death; these
mitigations validated ad hoc), #1098 (guard-vocabulary refusals, 2 reviewer
kills + ~2.7h orchestrator wedge), #1092 (implementer refusal kills;
orchestrator rung b2), #866 (bank-text paging kills), #1090
(refusal-truncated Agent spawns), #1152 (a findings recap in a reviewer's
return text wedged the parent orchestrator — discipline 4). Enforcing pointers:
`.claude/agents/code-reviewer.md` § Context budget (READ FIRST);
`.claude/agents/reconciler.md` § Rules (Rule 11);
`.claude/skills/issue/SKILL.md` Step 5a (orchestrator-side excerpt-file
pre-materialization);
`.claude/skills/issue/SKILL.md` § File-only Codex verdict posting
(orchestrator-side posting path, #1275).
