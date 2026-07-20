---
title: 'daily-fix: cap-park note for 9b cheap-band round-cap parks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4993ff0071ad
- daily-auto-filed
created_at: '2026-07-20T06:46:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): C2 cheap-band cap parks
  are bullet-only invisible (same class as 9a-ter pre-#1548)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-19 parked-candidate sweep (Step C) from a workflow-fix candidate parked on task #1548 (emitting agent: Alternatives critic, #1548 plan review; parked under the recursion guard).

## Goal

Extend the `followup-parked-by-cap` epm:progress note (landed for the Step 9a-ter zero-GPU cap by #1548) to the Step 9b cheap-band round-cap (C2) park path (`cost_class=needs-gpu`).

## Workflow gap

- **Bug observed:** the Step 9b cheap-band round-cap (C2) parks surviving cheap `same` proposals into the `epm:follow-ups v1` bullet list only — same bullet-only invisibility class as the 9a-ter cap #1548 fixed.
- **Why it is a workflow gap:** a cap-parked follow-up is invisible to the PM `Needs you` surfacing without the fixed-token note; #1548 established the surfacing contract but scoped it to 9a-ter.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'followup-parked-by-cap' .claude/skills/issue/SKILL.md` → hits only in the 9a-ter block (:6431-:6450); context read of the C2 cheap-band cap block (:7538-:7551) shows "Beyond the cap, further cheap `same` proposals survive in `epm:follow-ups v1` for manual pick" with NO note-posting duty (2026-07-19, post-#1548 merge 74abefd628 PR #1322).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up: add the same six-field `followup-parked-by-cap` epm:progress note duty, C2-keyed, to the Step 9b cheap-band cap park path)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Check `tests/test_inline_payload_lint_gate.py` / the #1548-added pin tests for the note contract; mirror the C2 variant there if the pin generalizes.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: cb26e0fdc500

Verbatim parked candidate (epm:workflow-fix-candidate on #1548, 2026-07-19T18:20:08Z): "source: prose-followup (Alternatives critic, #1548 plan review). target_file: .claude/skills/issue/SKILL.md. proposed_change: extend the followup-parked-by-cap epm:progress note to the Step 9b cheap-band round-cap (C2) park path (cost_class=needs-gpu) — same bullet-only invisibility class as the 9a-ter cap. confidence: medium. related_task: #1548."
