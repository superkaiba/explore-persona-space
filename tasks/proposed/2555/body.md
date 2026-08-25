---
title: 'workflow-fix: verify_plan per-corpus eligibility-floor arithmetic check (battery
  members vs the plan''s own floor)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T20:45:48Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2546 Phase-2 critic ensemble: plan v3''s own floor
  deleted its only no-reasoning per-corpus cell by construction; verify_plan 0/0'
workflow: v1
---
## Goal
Add a per-corpus fit-eligibility floor arithmetic check to `scripts/verify_plan.py` so a plan cannot register a per-corpus fit battery whose members cannot clear the plan's own eligibility floor.

## Incident (#2546 plan v3, 2026-08-24)
The plan registered an absolute-trainability drop floor (1.2×d/0.8 = 5,376 usable rows) AND a per-corpus fit battery naming MMLU at a 5,000-row draw — below the floor at even 100% usable — plus ContextHub at a realized max 5,460 (clears only under ≤1.5% attrition while the plan's own usable gate tolerates 10%). MMLU was the ONLY pure no-reasoning corpus in the Plot-8 ladder, so the H5 contrast lost its entire no-reasoning arm by construction, silently (the drop path is a designed non-error). verify_plan PASSed 0/0; caught by two independent critics + reconciler.

## Proposed check (from the methodology + statistics critics, mechanizable sketches)
For each corpus named in a per-corpus fit registry (or ladder battery list), when the plan declares (a) a per-corpus eligibility floor and (b) per-corpus row targets: FAIL any corpus whose `registered_rows` cannot clear the floor even at 100% usable (the MMLU shape); WARN any corpus that clears only above the plan's own declared usable-floor tolerance (`registered_rows × usable_floor < eligibility_floor`, the ContextHub shape). Also assert any matched-n companion convention ≥ the floor's own n_train margin or carries a named exemption line.

## Acceptance criteria
1. Fixture with floor 5,376 + a battery member at 5,000 rows FAILs.
2. Member at 6,750 rows with 90% usable floor PASSes.
3. Plans with no per-corpus floor or no battery SKIP.
4. Pin test for no-flags bundling status.
