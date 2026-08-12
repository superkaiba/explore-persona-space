---
title: 'daily-fix: judge pilot sizing + classification rubrics'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03688b24565e
- daily-auto-filed
created_at: '2026-08-06T07:05:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): pilot threshold unsatisfiable
  at 16 draws/arm (1/16 > 2%); a classification wave skipped piloting and failed substantively
  at scale'
workflow: v1
---
# daily-fix: llm-judging rule 26 — pilot draw count must make the parse-fail threshold satisfiable; cover classification rubrics explicitly

## Workflow gap

Two same-week judge-wave incidents expose two gaps in the pilot-gate rule
(`.claude/rules/llm-judging.md` rules 23/26):

1. **Threshold unsatisfiable at small per-arm draws.** #2091's G2 judge pilot FAILed
   (`pilot_rc=7`) with wildchat parse-fail 6.2% ≥ 2% — but at 16 draws/arm the smallest
   non-zero parse-fail rate is 1/16 = 6.25%, so ANY single parse failure trips the gate:
   the threshold is unsatisfiable by construction at that draw count. The session resolved
   it with an auditable per-(wave,arm) waiver constant (commit d839842fb4) and filed the
   parser defect separately (#2109). The rule never states that the pilot's per-arm draw
   count must satisfy n_draws ≥ 1/threshold.
2. **Classification/labeling rubrics skipped piloting.** A #1739 tactic-classification
   wave (MHJ 7-class taxonomy) failed SUBSTANTIVELY at scale — the judge returned analysis
   prose instead of the JSON schema ("Failed to parse judge JSON; returning None… Text:
   Let me analyze this attack request:") and the wave needed a rubric fix + re-dispatch.
   Rule 26's pilot-gate wording is anchored on graded 0–100 waves; a classification rubric
   is exactly as parse-fragile.

verified-at-filing: `grep -n 'pilot' .claude/rules/llm-judging.md | head` → the pilot-gate
clauses exist; `grep -cn '1/threshold\|satisfiab' .claude/rules/llm-judging.md` → 0 (no
sizing clause); `grep -cn 'classif' .claude/rules/llm-judging.md` → checked at compose
time for current coverage. Incident quotes are the miners' probed readbacks (sessions
b765cdcd rows 1522/1564/1569; 2f4940f0 rows 114/129).

unverified hypothesis — verify at plan time: whether the #1739 tactic wave was above the
~5,000-call pilot-gate floor (if below, item 2 is a rule-scope extension rather than a
violated rule) — miner could not determine the wave size.

## Proposed change

Amend `.claude/rules/llm-judging.md` rule 26: (a) the pilot's per-arm draw count must make
the parse-fail threshold satisfiable (n_draws ≥ 1/threshold, with the waiver-constant
pattern from #2091 named for the small-arm case); (b) state explicitly that
classification/labeling rubrics (non-0-100) are pilot-gated the same as graded waves.

## Provenance

- fingerprint: 03688b24565e

- workflow_fix_target: .claude/rules/llm-judging.md
- origin: /daily 2026-08-05 problem sweep — miner 6 P5, miner 2 P8.
