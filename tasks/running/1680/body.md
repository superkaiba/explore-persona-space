---
title: 'daily-fix: Step C suppression honors routed-record ts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1430ca49df30
- daily-auto-filed
created_at: '2026-07-25T06:50:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): Three 1630 parked candidates
  re-enumerated on 07-25 despite routed-records posted the prior night - for fp-bearing
  formal blocks suppression requires the candidate''s canonical fp in the record note
  and the records carried driver-recomputed fps from abridged text; separately the
  sweep reported skipped_rows=1 with no reason and the 1642 park of 04:53Z was absent
  from the output'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 Step C (observed during tonight's sweep run).

## Goal

Routed parked candidates must stay suppressed on subsequent sweeps even when the routed-record's fingerprint differs from the sweep-canonical one, and silently skipped rows must be explainable.

## Workflow gap

- **Bug observed:** (a) the 07-25 sweep re-listed all three #1630 candidates (fps f3f8be20cd01/421329c801a1/9664aebbd6cc, `suppressed: false`) although the 07-24 run posted routed-records for them — the records carried fps 3c4eb1c23efa/1318d69480dd recomputed from ABRIDGED origin text, and fp-bearing suppression matches only the canonical fp (`cand.fingerprint in note`), ignoring the records' exact `origin_candidate_ts` values. Tonight's run posted corrective re-keyed records; the class remains. (b) `skipped_rows: 1` with no reason — the #1642 park (2026-07-24T04:53:34Z, a well-formed prose park) was absent from the candidate list.
- **Why it is a workflow gap:** the escape-valve guarantee ("never silently lost") depends on suppression and enumeration both being exact; an fp mismatch re-enumerates forever, and a silent skip can genuinely lose a park.
- **Confidence (emitter):** high on (a) — code read at sweep lines ~296-326 (`_ORIGIN_TS_RE`, `_RECORD_FP_RE`, the fp-aware branch); medium on (b) root cause (unparsed row shape — needs investigation in the pipeline).
- verified-at-filing: `grep -n 'origin_candidate_ts\|fingerprint' scripts/sweep_parked_wf_candidates.py` → fp-aware suppression branch at ~:324-326, ts fallback documented "ONLY when the record carries no origin_candidate_ts" (:41-42) — presence bind for (a); tonight's sweep JSON (skipped_rows: 1, #1642 absent) recorded in /daily 2026-07-24 run — unverified hypothesis for (b)'s mechanism: verify at plan time.

## Proposed change (candidate diff sketch — refine in planning)

(a) In the suppression predicate: for fp-bearing candidates, ALSO suppress when a routed-record's `origin_candidate_ts` equals the candidate's ts (with a target_file sanity match). (b) Emit `skipped` rows with `{source, reason}` instead of a bare counter; add a test for a well-formed prose park that previously skipped. Optionally: /daily SKILL.md Step C note — routed-records MUST carry the sweep-reported fp verbatim.

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py` (+ `.claude/skills/daily/SKILL.md` Step C one-liner)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 1430ca49df30

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py
