---
title: 'parse_judge_json step 4: add labeled trailing score-line recovery (Score:
  N) — measured #2222 pilot shape missing from the landed #2109 recovery'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-11T14:37:09Z'
has_clean_result: false
parent_id: 2222
workflow: v1
---
# parse_judge_json step 4: labeled trailing score-line recovery ("Score: N") — measured #2222 pilot shape not covered by the landed #2109 recovery

## Gap

The landed step-4 recovery (issue-2109, commit 69581345b3 on main, `src/explore_persona_space/eval/utils.py`) recovers ONLY a bare trailing ASCII integer in [0,100] on the last line; its docstring deliberately lists "Score: 70" among the shapes that fall through to the parse-failure WARNING (prose-capture caution). Task #2222's judge pilot measured the labeled form as a REAL failure shape at production scale: Sonnet (claude-sonnet-4-5-20250929) under the persona-vectors 0-100 rubric writes reason-then-"Score: 75" trailing lines on hard code/math rows — 28/150 hallucination pilot draws (concentrated on insecure_code + mistake_math; every sampled failure end_turn-complete with the score on the final line). #2222's production wave ran with a branch-local parser accepting that form (its judge_accounting.json records the realized drop taxonomy); the branch parser was deliberately dropped at merge in favor of main's reviewed instrument, so future waves on this rubric family will re-drop labeled-score rows.

## Fix sketch

Extend main's step 4 with a SECOND, equally strict branch: fullmatch the entire last non-empty line against a labeled-score pattern (`score` keyword + `:`/`=` + 1-3 ASCII digits, optional markdown emphasis tokens only), same [0,100] range gate, same drop-never-coerce fallthrough, its own `_PARSE_STATS` counter + greppable INFO token (e.g. `recovered-labeled-score`). A fullmatch on a labeled line is not the prose-capture shape the #2109 docstring guards against (embedded numerals still fall through); update the docstring's "Score: 70 falls through" example accordingly. Reference implementation of the accepted shapes: the #2222 branch parser at commit 94c155ce03 (`_TRAILING_SCORE_LINE_RE`) — note it lacked the range gate and counters; the extension should keep #2109's envelope (int-only, 0-100, counters), not port the branch form verbatim.

## Acceptance

- "...analysis...\n\nScore: 75" → 75 (counter + INFO token); "**Score: 75**" → 75; "score = 88" → 88.
- "a score of 20." (embedded), "Score: 150" (out of range), "Score: -5", "Score: 7.5" (float) → None (drop).
- Existing #2109 pins in tests/test_eval_utils.py + tests/test_alignment.py unchanged and passing; new pins added for the labeled branch.
- Evidence check: re-run the #2222 pilot fixture shapes (the 28/150 failure sample is characterized in eval_results/issue_2222/form_a_pilot_*.json + the judge_accounting.json drop taxonomy).

## Provenance

workflow_fix_target: src/explore_persona_space/eval/utils.py (parse_judge_json step 4)
Surfaced by: #2222 terminal-point merge resolution (2026-08-11) — the branch's parser recovery was superseded by issue-2109's landed equivalent, which lacks the labeled-score branch; deferred here for its own review rather than riding a merge resolution. Measured evidence: #2222 pilot (28/150 hallucination draws) + production wave accounting.
