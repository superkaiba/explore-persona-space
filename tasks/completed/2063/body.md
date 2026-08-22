---
title: 'daily-fix: judge max_tokens pins sit below the raised rule-2'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-08-04T06:50:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): llm-judging rule 23 raised
  its max_tokens floors to 1024/2048 on 2026-08-02 (commit 12ce0a8225), but 15 judge
  pins in scripts/ still sit at 300/400/600 and most cite rule 23 as their justification;
  a below-floor budget silently truncation-censors draws arm-asymmetrically (the failure
  that forced three re-judge waves in one week).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miner3, session f4f0e16a, task #1689). NOT a workflow-surface fix — these are per-issue experiment scripts.

## Goal

Bring every in-repo judge `max_tokens` pin up to the rule-23 floor that it cites as its own authority, or record an explicit justified deviation at each site — before the next judge wave spends against a below-floor budget.

## Workflow gap

- **Bug observed:** `.claude/rules/llm-judging.md` rule 23 raised its `max_tokens` floors to **1024** (single-rationale reason-then-score) / **2048** (multi-field JSON) on 2026-08-02 (commit `12ce0a8225`, "generous token budgets + judge pilot gate + cap-hit reporting", after the #1739/#1769/#1774/#1934 truncation audit). Fifteen judge pins in `scripts/` still sit at 300/400/600 — and most cite rule 23 in their own trailing comment as the justification for the now-stale value. Concretely: `issue1689_common.py:239` (300, `# rationale + score, per llm-judging.md rule 23`; file last touched 2026-07-30, pre-raise), `issue1776_swap_judge.py:45` (300, `# llm-judging rule 23 floor`), `issue1774_judge.py:49` (300), `issue1090_fu3_worker.py:86` (300), `issue1769_judge.py:62` (300), `issue1415_judge.py:78` (300), `issue1315_rejudge_529.py:58` (300), `issue1773_common.py:89` (400, `AXES_MAX_TOKENS`), `issue1482_feature_correlates.py:66` (400), `issue1482_analysis.py:42` (400), `issue1900_judge.py:84` (400), `issue1345_common.py:528` (400), `issue1345_onpolicy_judge_legs.py:102` (600).
- **Why it is a gap:** rule 23's failure mode is SILENT and arm-asymmetric — the API truncates the reason-first response before the score token is emitted, the parse fails, and rule 9's drop-never-coerce then discards the draw, so the judge call "succeeds" while the draw is censored on whichever arm's rationales run longer. That is exactly the shape that cost three full re-judge waves in one week (#1739: 86,521 rollouts × 3 draws, 5.4% censored at `max_tokens=400`; #1769: 21,000 draws re-judged; #1774). A pin that cites the rule as its authority while sitting below the rule's floor will not be caught by reading the comment.
- **Confidence (emitter):** high for the inventory; medium for per-site disposition (some sites may be legitimately score-only — see below).
- verified-at-filing: `grep -rn 'JUDGE_MAX_TOKENS = \|AXES_MAX_TOKENS = ' scripts/ | grep -cE '= (300|400|600)\b'` → **15** hits (2026-08-04), enumerated above with line numbers. `git log --oneline --since='2026-07-28' -- .claude/rules/llm-judging.md` → the raise landed in `12ce0a8225`; the prior floor bump to 600 was `84e6b7863d` (#1692). `git log -1 --format='%h %ad' --date=short -- scripts/issue1689_common.py` → `d576918d4a 2026-07-30`, i.e. pre-raise. Landed-fix history: no commit since the raise touches any of the 15 pin sites.
- unverified hypothesis — verify at plan time: which of the 15 are genuinely exempt. Rule 23 exempts a **score-only** rubric (bare integer, no rationale) from the generous floors, and `issue1434_cells.py:188` self-describes its rubric as score-only — so a per-site rubric read (reason-then-score vs multi-field JSON vs score-only) is required before changing any value; this filing does not assume all 15 need raising.

## Proposed change (candidate sketch — refine in planning)

```
per site, after reading its actual rubric shape:
  reason-then-score (single rationale) -> max_tokens >= 1024
  multi-field JSON rubric              -> max_tokens >= 2048
  score-only (bare integer)            -> leave; correct the comment so it no
                                          longer cites rule 23's floor as its
                                          justification
```

Plus, per rule 23's own cache caveat: a budget raise must be paired with a FRESH `cache_dir` for any pre-#2021 truncation-era entries (those lack `stop_reason` and are still served from cache), and per rule 18 the realized per-arm drop rate + the `max_tokens` used are reported at the next wave.

## Scope / surfaces

- Primary targets: the 13 files enumerated above under `scripts/`.
- Out of scope: `.claude/rules/llm-judging.md` itself (the floors are correct; only the consumers are stale).

## Constraints / invariants

- Do NOT lower any floor to match a pin.
- Do not change a value without reading that site's rubric shape (score-only sites are exempt by rule 23).
- Any site whose next wave is ≥ ~5,000 calls additionally owes the rule-26 pilot gate; note it rather than silently skipping.
