---
title: 'daily-fix: sweep emits unmatched_record_fps advisory'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d541ad0333fd
- daily-auto-filed
created_at: '2026-07-26T07:07:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Driver fingerprint-recomputation
  drift (the #1630 class) is tolerated by the #1680 ts-claim suppression fallback
  but nothing mechanically DETECTS it when it recurs.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 Step C parked-workflow-fix-candidate routing pass
(`.claude/rules/workflow-fix-on-bug.md` § Recursion guard escape valve). The candidate was
parked on task #1680 at 2026-07-25T16:21:39Z because that session ran under the
`workflow_fix_target` recursion guard.

## Goal

Emit a top-level advisory field (e.g. `unmatched_record_fps`) from
`scripts/sweep_parked_wf_candidates.py` listing same-stream filed-record fingerprints that
match no enumerated candidate fingerprint, for `/daily` Step C to flag. Advisory-only — no
change to suppression semantics.

## Workflow gap

- **Bug observed:** driver fingerprint-recomputation drift (the #1630 class) is now
  TOLERATED by the #1680 `origin_candidate_ts` suppression fallback, but nothing
  mechanically DETECTS it when it recurs. A routed-record carrying a real 12-hex
  fingerprint that matches NO enumerated candidate fingerprint is silent evidence the
  driver recomputed the fingerprint from abridged/synthesized text.
- **Why it is a workflow gap:** the #1680 Edit-C verbatim-fingerprint mandate is PROSE
  (LLM-followed). A sweep-side detector would surface the drift the moment it recurs
  instead of relying on the ts fallback indefinitely.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'unmatched_record_fps' scripts/sweep_parked_wf_candidates.py`
  → **0 hits** (absence confirmed in the named target); a broader
  `grep -n 'unmatched' scripts/sweep_parked_wf_candidates.py` also returns nothing.
  Landed-fix history check
  `git log --oneline --since='7 days ago' -- scripts/sweep_parked_wf_candidates.py` → 2
  commits: `51d2e343a0` (#1680, "Step C suppression honors routed-record
  origin_candidate_ts + structured skips") and `45cbe304a9` (#1599, "sweep subsumption by
  fix merge/close time") — #1680 landed the ts FALLBACK this candidate is the DETECTOR
  for, not the detector itself. Live probe: tonight's Step C run emitted
  `candidates: 6 / skipped: 1` with no `unmatched_record_fps` key in the JSON envelope.
  (2026-07-25)

## Proposed change (candidate diff sketch — refine in planning)

```
+ collect real-fp values from filed records per stream; subtract the
+ stream's candidate fps; emit leftovers as unmatched_record_fps with
+ {source, ref, fp}; document in module docstring + one Step C sentence.
```

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- One Step C sentence in `.claude/skills/daily/SKILL.md` telling the orchestrator to flag
  a non-empty `unmatched_record_fps` (mirror the existing `skipped` / `relevant_kind`
  guidance sentence, which is the closest precedent).
- **Advisory-only is load-bearing:** the field must not gate, suppress, or re-enumerate
  anything. A false positive here must cost an eyeball, never a lost suppression.

## Constraints / invariants

- Workflow-surface only.
- Suppression semantics UNCHANGED — verify against
  `tests/test_sweep_parked_wf_candidates.py` that the existing suppression paths
  (verbatim-fp primary, #1680 `origin_candidate_ts` fallback) are byte-behaviour-identical.
- Fail soft: a malformed filed record contributes to the advisory or is skipped, never
  raises.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py
- fingerprint: d541ad0333fd

Parked candidate (verbatim), from task #1680 `events.jsonl` @ 2026-07-25T16:21:39Z:

<!-- workflow-fix-candidate v1 -->
target_file: scripts/sweep_parked_wf_candidates.py
bug_observed: driver fp-recomputation drift (the #1630 class) is now tolerated by the ts-claim suppression fallback (#1680) but nothing mechanically DETECTS it when it recurs — a routed-record carrying a real 12-hex fp that matches NO enumerated candidate fp is silent evidence the driver recomputed from abridged text.
why_workflow_gap: the #1680 Edit C verbatim-fp mandate is prose (LLM-followed); a sweep-side detector would surface drift the moment it recurs instead of relying on the ts fallback forever.
proposed_change: emit a top-level advisory field (e.g. unmatched_record_fps) listing same-stream filed-record fps that match no enumerated candidate fp, for /daily to flag; advisory-only, no suppression semantics change.
diff_sketch: |
  + collect real-fp values from filed records per stream; subtract the
  + stream's candidate fps; emit leftovers as unmatched_record_fps with
  + {source, ref, fp}; document in module docstring + one Step C sentence.
confidence: medium
related_task: #1680
<!-- /workflow-fix-candidate -->
