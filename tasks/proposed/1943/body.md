---
title: 'workflow-fix: judge instruments round-trip the parse contract before dry-run
  counts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6b4c87fc572
created_at: '2026-07-31T18:00:50Z'
has_clean_result: false
origin_prompt: 'boundary-impl prose follow-up on #1345 (2026-07-31): a dry run proves
  ROUTING, not the request/response CONTRACT — generalize the round-trip test (push
  a realistic reply through the consumer''s own parse before believing a judge leg
  is wired) to the judge-composition surface.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1345 (emitting agent: boundary-impl, the #1345 on-policy round's implementer).

## Goal

Add a rule to `.claude/rules/llm-judging.md` (+ a mirror line in the experiment-implementer smoke contract): a COMPOSED judge instrument counts as wired only after a committed test round-trips a realistic reply through the harness's OWN parse+reduce path (`parse_judge_json` -> `_score_from_parsed`) and presence-checks the user-template substitution placeholders (`{question}`/`{answer}`) — dry-run evidence proves ROUTING only, never the request/response CONTRACT.

## Workflow gap

- **Bug observed:** two judge rubrics composed for #1345 (commit `9e088b3bdb`) carried latent 100%-draw-drop defects invisible to their clean dry runs: (1) neither rubric carried the `{question}`/`{answer}` substitution placeholders `graded_judge` fills via `.replace(...)` — the judge would have received scoring instructions with nothing to rate; (2) the rubrics requested a trailing `SCORE: <int>` line while the harness FORCES a JSON contract (`parse_judge_json('...SCORE: 73') -> None` => every draw dropped; demonstrated by the emitter). Both fixed in `a41fcad04f` with a committed round-trip test family (72 tests).
- **Why it is a workflow gap:** `.claude/rules/llm-judging.md` mandates drop-never-coerce, transport splits, max_tokens sizing, cache keying, and pre-submit custom_id validation, but NOWHERE requires the offline response-side round-trip; the existing mock-seam rules (gotchas.md: "--mock-judge smoke does NOT validate the Batch API REQUEST SHAPE — only a tiny LIVE forced-batch submit does") cover the REQUEST side, which needs a live probe — the RESPONSE-parse side is validatable OFFLINE at zero API cost and is not covered by any rule. This is the second composed-but-never-executed instrument to hide a fatal contract defect from the same emitter (its own framing).
- **Confidence (emitter):** high (both defects demonstrated + fixed + test-pinned).
- verified-at-filing: `grep -cn "round-trip\|roundtrip\|parse_judge_json" .claude/rules/llm-judging.md` -> 0 hits (absence-of-guard claim; 0 in-target hits IS the evidence), and `grep -rln "parse_judge_json" .claude/rules/ .claude/agents/` -> 0 files (no sibling rule covers the parse contract); landed-fix history checked: `git log --oneline --since='7 days ago' -- .claude/rules/llm-judging.md` -> 2 commits (d7fb12d9a7 stale-count fix, 4882b8e45a rule-23/25 additions), neither touching parse-contract validation (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/llm-judging.md`, new rule under § G (Reproducibility / reporting) or § C (Prompt / rubric design):

+ 26. **Round-trip the parse contract before trusting a composed judge instrument.**
+     A dry run proves ROUTING, not the request/response CONTRACT. Any newly composed
+     judge rubric/leg ships with a committed test that (a) pushes a REALISTIC reply
+     (reasoning + score, plus a fenced variant) through the harness's OWN parse+reduce
+     (`parse_judge_json` -> `_score_from_parsed`), and (b) presence-checks the user-template
+     substitution placeholders ({question}/{answer}) and that harness-identical substitution
+     leaves no slot unfilled. The REQUEST side still needs the live probe (gotchas.md
+     mock-seam rules); the RESPONSE side is validatable offline at zero API cost.
+     (Incident #1345 round 3->4: two rubrics with clean dry runs would have dropped
+     ~100% of draws — no placeholders; SCORE:-line shape vs the forced JSON contract.)

Mirror one line in `.claude/agents/experiment-implementer.md` § smoke-contract requirements.

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md`, `.claude/agents/experiment-implementer.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'judge' .claude/rules/ .claude/agents/ | xargs grep -ln 'rubric'`) and update every hit that composes judge instruments; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; keep the LESSONS.md index + any hardcoded rule-count cross-refs consistent (see d7fb12d9a7 — do not reintroduce hardcoded guideline counts).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/llm-judging.md
- fingerprint: c6b4c87fc572

Verbatim surfaced prose (boundary-impl, #1345 session, 2026-07-31): "REFLECTION worth recording, since this is the second time a composed-but-never-executed instrument hid a fatal defect from me: a dry run proves ROUTING, not the request/response CONTRACT. My round-3 evidence for (d) was 'both legs dry-ran clean', which was true and nearly worthless — the same blind spot as the mock-seam class in gotchas.md, one layer up. The durable fix is the round-trip test now committed: push a realistic reply through the consumer's own parse before believing a judge leg is wired. If you want that generalized beyond #1345 I can raise it as a workflow-fix candidate for the judge-composition surface."
