---
title: 'Persist the #2151 api-refusal drop class in wave metas (currently blended
  into content_drops)'
kind: infra
tags: []
created_at: '2026-08-21T11:08:39Z'
has_clean_result: false
workflow: v1
---
# The #2151 api-refusal drop class is WARNED at reduce time but never PERSISTED per-arm — wave metas record recoverable classifier refusals as `content_drops`, so the durable record misclassifies them

## Goal

Make the api-refusal drop class (#2151) survive into the persisted wave meta as its own
field, so a recoverable transport-conditional refusal is never indistinguishable from a
legitimate content decline in the committed record.

## The defect

`src/explore_persona_space/eval/graded_judge.py` computes the third top-level drop class
(`n_api_refusal_draws`, `per_item_api_refusals`) and emits a precise WARNING naming it:

```
judge reduce: 13 API-refusal draws across 13 item(s) (stop_reason == 'refusal', empty
content) — the wave ran a transport whose safety classifier CENSORS; transport-conditional
and retriable via targeted SYNC re-issue at the identical instrument (llm-judging.md rule
28, #2151/#1739). NOT blended into content drops or transport losses.
```

But the PERSISTED `<wave>.meta.json` carries no api-refusal field. Measured on #2329
`q35_ladder_decay` Leg B pre-recovery metas — the only per-arm drop keys are
`truncation_drops` and `content_drops`, and the only top-level drop key is
`residual_transport_lost`:

```
dfrag-r1_pirate       grid-ceiling  n_items=480 n_scored=478 content_drops=2
dfrag-r2_butler       grid-ceiling  n_items=480 n_scored=479 content_drops=1
dfrag-r5a_lu_therapy  grid-ceiling  n_items=240 n_scored=239 content_drops=1
dfrag-r5b_lu_philosophy grid-steered n_items=200 n_scored=199 content_drops=1
dfrag-r5b_lu_philosophy grid-ceiling n_items=400 n_scored=388 content_drops=12
```

2+1+1+1+12 = **17**, exactly the api-refusal count the WARNING reported. So the class is
correct in the log and blended in the artifact.

## Why it matters

The log line is ephemeral; the meta is the durable record every downstream consumer and
every later reader uses. The two classes have OPPOSITE remedies:

- **content drop** — the judge legitimately declined on the content. Correct handling is
  to DROP the row and report it. Not retriable.
- **api-classifier refusal** — the provider's safety classifier censored a request that
  SUCCEEDED. Rule 28 says transport-conditional and retriable via targeted sync re-issue
  at the identical instrument; #1739 measured 0 re-refusals on sync, and #2329 Leg B
  reproduced that (17/17 recovered, zero re-refusals).

Blending them into `content_drops` makes a recoverable, transport-attributable loss look
like an irreducible content property of the item. A future round reading these metas would
conclude 17 items are simply unjudgeable and would NOT attempt the sanctioned recovery —
which is the whole point of the #2151 class existing.

Aggravating factor: the wave-level resume predicate is `<wave>.meta.json` vs regime, so a
wave with refusals reports `complete (meta matches regime) — skip`. The refusals are
therefore both misclassified AND frozen behind a completion marker, with the only signal
being a log line that scrolls away.

## Proposed fix

Persist the class alongside the existing ones — per-arm `api_refusal_drops` and a
top-level `n_api_refusal_draws` (plus `per_item_api_refusals` if cheap) in the wave meta
written by the `run_wave` / graded-judge reduce path. Keep `content_drops` meaning
content-only after the change.

Consider also recording the wave's realized TRANSPORT in the meta (batch vs sync): the
refusal class is transport-conditional, so "which transport produced this row" is the
other half of the provenance. #2329 Leg B ended with a deliberate, rule-28-sanctioned
split (6,015 draws batch + 17 sync) that is currently recoverable only from prose.

## Acceptance criteria

1. A wave whose reduce reports N api-refusal draws writes a meta in which those N are
   attributable to the api-refusal class and are NOT counted in `content_drops`.
2. Reproduce with the #2329 fixture: the committed pre-recovery metas at
   `eval_results/issue_2329/q35_ladder_decay/decay/judge/prerecovery_quarantine/`
   (4 files, byte-identical copies) whose true api-refusal counts are 2 / 1 / 1 / 13.
3. A genuine content drop is still reported as `content_drops` — the fix must not
   reclassify content declines as refusals (the inverse error).
4. Back-compat: existing consumers reading `content_drops` must not silently change
   meaning without the field being explicit; state the migration posture for already
   committed metas (they under-report content_drops' true class — do NOT rewrite history).
5. Tests failing before / passing after; no new red in the no-flags `workflow_lint.py`
   run or the mapped-test selection.

## Provenance

Found during #2329 round `q35_ladder_decay` Leg B (2026-08-21). The orchestrator's own
`epm:progress` v199 note asserted "refusals correctly NOT blended into content drops or
transport losses" — true of the log line, FALSE of the persisted artifact; the R-1
implementer caught the contradiction while building the drop-class diagnostics panel, and
the orchestrator verified it directly against the quarantined metas before filing. That
correction is recorded on #2329.

Evidence: #2329 `events.jsonl` v199 (the wrong claim) and the correction note; the
committed pre-recovery metas named in criterion 2; `graded_judge.py` docstring lines
~173-185 (the class definition and precedence `transport -> api-refusal -> content`).

- target_file: src/explore_persona_space/eval/graded_judge.py
- fingerprint: api-refusal-class-not-persisted-in-wave-meta
- confidence: high — measured directly, counts reconcile exactly to the warned total
