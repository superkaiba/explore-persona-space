---
title: Persist stop_reason per judge draw + truncation-vs-content drop split + rule-26
  pilot-gate helper
kind: infra
tags: []
created_at: '2026-08-02T16:25:32Z'
has_clean_result: false
origin_prompt: 'User 2026-08-02: ''make the workflow changes right now. Be generous
  with the allowed token generation for both on GPU and API'' — code leg of the truncation-incident
  audit (#1739/#1769/#1774/#1934): thread stop_reason into persisted judge results,
  split truncation vs content drops, add the rule-26 pilot-gate helper.'
workflow: v1
---
## Overview / Motivation

Three judge-truncation re-judge waves landed in one week — #1739 (max_tokens=400: 5.4% of 86,521×3 draws truncation-censored, surgical re-judge at 800), #1769 (max_tokens=300 on a multi-field rubric: 1,606/21,000 draws dropped, worst arm 42.5% arm-asymmetric, the whole 21k-call wave re-judged at 600), #1774 (per-draw truncation-recovery merge at 600) — plus two misdiagnosis rounds (#1934/#1773: 2,602 parse failures twice mis-called truncation from log text; a direct Messages-API `stop_reason` probe showed 39/40 `end_turn`).

Root cause of the diagnosis cost: the client layer captures `stop_reason` (`src/explore_persona_space/llm/models.py:281` `Response.stop_reason`; populated at `llm/anthropic_client.py:667,1011`, `llm/openai_client.py:389,659`) but every judge module DROPS it before persistence, so drop-class diagnosis (truncation vs content) always needs a fresh API probe or a full re-judge.

- verified-at-filing: `grep -n 'stop_reason' src/explore_persona_space/eval/batch_judge.py src/explore_persona_space/eval/graded_judge.py src/explore_persona_space/eval/judge_dispatch.py src/explore_persona_space/eval/utils.py` → 0 hits in 0 files (absence claim: no judge module persists stop_reason); `grep -n 'stop_reason' src/explore_persona_space/llm/models.py src/explore_persona_space/llm/anthropic_client.py` → hits at models.py L281/292/294/312 + anthropic_client.py L667/L1011 (client layer captures it) (2026-08-02).

## Goal

Make judge-response truncation mechanically visible in persisted judge results and gateable before a production spend: (1) thread `stop_reason` into every persisted per-draw judge result, (2) split drop accounting into truncation-drops vs content-drops (alongside the existing content/transport split from #1313), (3) add a `judge_pilot_gate` helper implementing `.claude/rules/llm-judging.md` rule 26 (the ≥5k-call pilot gate landed 2026-08-02).

## Proposed change (refine in planning)

1. `eval/batch_judge.py` / `eval/graded_judge.py` / `eval/judge_dispatch.py`: carry `stop_reason` from the client `Response` into each stored per-draw result dict — INCLUDING parse-failure records, so `stop_reason == "max_tokens"` (truncation, budget-class) is mechanically distinguishable from `end_turn` (content-class) without a fresh probe.
2. Drop accounting: add a truncation counter (e.g. `JudgeResult.n_truncation_dropped_draws` + per-item split) alongside `n_dropped_draws` (content) and `n_transport_lost_draws` (#1313). Truncation drops are budget defects (rule 23), not content drops — report all three.
3. New helper `judge_pilot_gate(...)`: dispatch N≈100–200 pilot draws spanning the caller's arms at the EXACT production instrument (rubric, judge model, max_tokens) against a pilot `cache_dir`; return per-arm parse-failure rate + stop_reason tally + a PASS/FAIL verdict (FAIL on any nonzero max_tokens stop_reason, or per-arm parse-fail ≥ ~2% unexplained). Callers use it before any ≥5,000-call production dispatch per rule 26.
4. Cache hygiene: decide in planning whether truncation-class (`stop_reason == "max_tokens"`) error dicts should be put-skipped by the rubric-level JudgeCache like transport-class dicts are (#1313), so a budget raise self-heals without a fresh `cache_dir` (today truncation-era parse-error entries are re-served — rule 23's cache caveat).

## Constraints / invariants

- Drop-never-coerce (rule 9) and the transport split machinery (#1313) unchanged.
- Rubric-keyed cache KEY semantics (rule 22, `EPM_JUDGE_CACHE_KEY_V2`) unchanged — additive result-dict fields only; a key change would cold-invalidate every cache.
- New report fields ride rule 18's pinned-report list; `tests/` pin the truncation/content/transport split.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
