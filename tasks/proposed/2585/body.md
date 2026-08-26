---
title: 'smoke-blind-spots.md: add row-index/data-reach as a fourth coverage-narrowing
  mechanism'
kind: infra
tags: []
created_at: '2026-08-25T18:46:15Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'workflow-fix-candidate raised by experiment-implementer during #2546
  r7: a [:N] head-slice smoke cannot reach a per-row data-dependent invariant first
  violated at a later row (gsm8k src_index 24 vs a 20-row cap)'
workflow: v1
---
## Provenance

workflow_fix_target: .claude/rules/smoke-blind-spots.md
Raised by: `experiment-implementer` during task #2546 round 7 (crash-fix), as a `<!-- workflow-fix-candidate v1 -->` block. Auto-filed by the #2546 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (auto-file + dispatch by default, any confidence). Confidence as raised: medium-high.

## Goal

Add a FOURTH coverage-narrowing mechanism — **row-index / data reach** — to the `smoke-blind-spots.md` taxonomy, plus the matching enumeration duty.

## The gap

`.claude/rules/smoke-blind-spots.md` currently names three mechanisms by which a smoke PASS certifies less than it appears to:

1. substituted implementation,
2. downgraded gate,
3. production-only code path.

All three are about the smoke running DIFFERENT CODE. None covers the smoke running the SAME code over a STRICTLY SMALLER DATA PREFIX — so a per-row, data-dependent fail-loud invariant whose first violating row sits past the smoke's head slice is invisible to a taxonomy-compliant enumeration.

## Proof that the taxonomy is insufficient as written

Task #2546 round 6 wrote a smoke blind-spot enumeration that was compliant with all three existing mechanisms, passed both the Claude code-reviewer and the Codex twin (Codex explicitly graded its Step 0.71 smoke-blind-spot check PASS), and STILL shipped a production crash on the very next launch:

```
scripts/issue2546_stage_corpora.py, join_gsm8k_gold
RuntimeError: gsm8k_test: gold solution at src_index 24 has zero '<<'
```

Mechanism: `join_gsm8k_gold` hard-raised on `k = sol.count("<<") < 1`. The `--smoke` path slices `staged[corpus][:20]`. The FIRST `openai/gsm8k` test row with an unannotated gold solution is `src_index 24` — four rows past the cap. Measured census: 18/1,319 test (1.36%) and 95/7,473 train (1.27%) rows carry zero `<<` annotations. The smoke was structurally incapable of reaching any of them, ran rc=0 in 211.6s, and production died ~1 min in. Evidence: `epm:failure v2` on #2546 (full diagnosis + census), fix round `0bf51d536e`.

Note the near-miss quality of this: it was NOT caught by review, because the enumeration genuinely satisfied the rule as written. That is what makes it a rule gap rather than a reviewer lapse.

## Proposed fix (as raised, refine at plan time)

1. Add mechanism 4, "row-index / data reach", to the taxonomy: a smoke whose slice is a head prefix (`[:N]`), a fixed sample, or any strict subset of production rows cannot certify per-row data-dependent invariants over that data.
2. Add one enumeration duty line: any per-row fail-loud invariant over an EXTERNAL corpus must be either (a) censused at full-corpus grain before production, or (b) reached at smoke scale via known-adversarial probe indices appended to the smoke slice.
3. Cite the worked pattern from the #2546 r7 fix: `SMOKE_ZERO_ANNOTATION_PROBE_IDX = {"test": 24, "train": 29}` in `scripts/issue2546_stage_corpora.py`, which appends the first measured offender per corpus to the smoke slice so the drop path is exercised at smoke scale. The same round's sibling census (ARC 0/1,172, CSQA 0/1,221, PIQA 0/1,838, MMLU 0/14,042 — all clean) is the worked example of option (a).

## Scope notes for the implementer

- This is a rule-text + enforcement-surface change. Check whether the `code-reviewer.md` Step 0.71 trigger grammar and the `critic.md` Methodology lens item 19 need the parallel clause, and whether `workflow_lint.py --check-smoke-blind-spot-review-lens` (a region-anchored surface pin) needs its anchor updated.
- Do NOT retrofit #2546's own scripts; that round already landed its fix.
- The existing rule explicitly lists its scanner's known false negatives. A data-reach mechanism is likely NOT statically scannable at all (it depends on external corpus content), so the honest posture is probably a reviewer-lens duty plus disclosure, not a new mechanical check. Say so explicitly rather than inventing an unreliable scanner.
