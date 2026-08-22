---
title: 'daily-fix: DATAGEN_JUDGE_MAX_TOKENS=500 sits below the rule-23 floor (live
  library pin, mt-keyed cache)'
kind: infra
tags:
- from-2063-sweep
created_at: '2026-08-10T04:16:46Z'
has_clean_result: false
parent_id: 2063
origin_prompt: 'Flagged by #2063 implementer round: unmapped live sub-floor judge
  pin src/explore_persona_space/artifacts/datagen.py:88 needs its own disposition
  (cache-key change).'
workflow: v1
---
## Goal

Disposition the one live sub-floor judge max_tokens pin that #2063's sweep
found OUTSIDE its plan enumeration: bring
`src/explore_persona_space/artifacts/datagen.py:88`
(`DATAGEN_JUDGE_MAX_TOKENS = 500`) up to the llm-judging.md rule-23 floor for
its rubric class, or record an explicit justified deviation at the site —
handling the cache-key consequence deliberately rather than silently.

## Context

Filed from #2063 (judge max_tokens floor sweep). #2063 raised 15+ pin sites in
`scripts/` but excluded this one because it is a LIVE library pin with
cache-key semantics, needing its own disposition:

- `datagen.py:88` pins 500 for the datagen judge-filter rubrics
  (reason-then-JSON — the single-rationale >=1024 class, so 500 is below
  floor).
- Consumer `scripts/issue1947_syc_recovery.py:100` aliases it with the comment
  "fleet parity, judge-cache-key input".
- The datagen cache dir embeds the value (`mt{value}`, `datagen.py:1491`), so
  a raise changes cache keys — every consumer re-judges cold (a bounded
  re-spend, never a wrong read; but it must be a recorded decision, and any
  in-flight #1947-family resume should be considered).

#2063 plan §2 claimed "src/: no sub-floor judge-rubric pins" — falsified by
this site; the #2063 clean-up left it untouched by design (flagged in its
`epm:results` marker).

## Disposition sketch (for the planner)

Either (a) raise to 1024 + accept the one-time cold re-judge for future
datagen waves (rule 23: a cap is not a spend; the mt-keyed cache means old
entries simply stop being read — nothing corrupts), stating the cache-key
change in the site comment and checking no live wave is mid-resume against the
`mt500` cache at land time; or (b) LEAVE with a justified parity deviation if
a live resume chain is found. Constraint: never lower a floor; do not touch
`.claude/rules/llm-judging.md`.
