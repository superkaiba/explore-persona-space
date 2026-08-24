---
title: 'codex-interpretation-critic emits MAJOR/MINOR CONCERN:: severities the concerns
  ledger rejects, silently dropping the whole batch'
kind: infra
tags:
- concerns-ledger
- codex-twin
created_at: '2026-08-24T10:28:34Z'
has_clean_result: false
origin_prompt: 'Round-4 codex-interpretation-critic verdict on #2254 emitted MAJOR/MINOR
  severities; persist_verdict_concerns.py rejected 4 of 5 rows as bad-severity and
  persisted zero.'
workflow: v1
---
## Goal

Pin the `CONCERN::` severity vocabulary in the codex twin composers to `task_workflow.CONCERN_SEVERITIES` = {BLOCKER, CONCERN, NIT}, so a twin verdict cannot emit rows that fail to persist.

## Evidence (task #2254, round-4 interpretation critique, 2026-08-24)

`codex-interpretation-critic` emitted five machine-readable rows using code-reviewer severity vocabulary:

    CONCERN:: MAJOR    firstk-ctxext-allanswer-overclaim
    CONCERN:: MAJOR    firstk-ctxext-intrusion-sensitivity
    CONCERN:: BLOCKER  firstk-review-substrate-missing
    CONCERN:: MAJOR    firstk-degenerate-control-series
    CONCERN:: MINOR    firstk-ctxext-manifest-stale

`scripts/persist_verdict_concerns.py` validates against `task_workflow.CONCERN_SEVERITIES` (frozenset {BLOCKER, CONCERN, NIT}, src/explore_persona_space/task_workflow.py:8786) and rejected rows 1, 2, 4, 5 as `bad-severity`. The validation gate is all-or-nothing: **zero** rows persisted, including the well-formed BLOCKER row 3. No success line printed; no epm:concern-raised events written.

Not a systematic twin failure — the SAME agent's round-3 verdict on the same task emitted two rows using accepted severities and persisted 2/2. So the composer's inlined grammar does not pin the enum, and the twin drifts into Critical/Major/Minor when its findings feel code-review-shaped.

## Impact

Findings survive in the posted verdict marker, so this is not data loss — but the concerns ledger silently loses the round's rows, and every consumer that treats the ledger as the authority (open-concern counts in bodies, critic briefs, watcher passes) sees a round that raised nothing. Combined with #2530 (--open-only undercounts), two independent defects now corrupt the same ledger's apparent state.

## Scope

- Pin the accepted severity enum explicitly in the `CONCERN::` grammar block of all five `codex-*` composer specs, naming the three legal tokens and stating that Critical/Major/Minor are NOT accepted.
- Consider whether `persist_verdict_concerns.py` should persist the VALID rows and report the malformed ones, rather than dropping a whole batch for one bad token. Argue the tradeoff explicitly: all-or-nothing avoids partial ledger state, but here it discarded a valid BLOCKER.
- Consider a normalizing map (MAJOR->CONCERN, MINOR->NIT, CRITICAL->BLOCKER) at the parse boundary, weighed against silently reinterpreting a reviewer's severity.
- Regression test: a mixed batch with one bad-severity row asserts the documented behavior.

## Provenance

Surfaced during /issue 2254 round-4 interpretation-critique verdict collection, 2026-08-24. Sibling ledger defect: #2530.
