---
title: 'workflow-fix: guard_skill_doc_headroom.sh must validate a demanded cap raise
  clears its own warn threshold'
kind: infra
tags:
- workflow-fix
- guard-hook
created_at: '2026-08-19T22:00:26Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2204 round-1 code-reviewer (Minor 1): guard_skill_doc_headroom.sh
  demands a same-change cap raise but never checks the raised value clears EPM_SKILL_DOC_HEADROOM_WARN_BYTES;
  #2204''s 78,000 bump left headroom 1,411 < 2,000, re-arming the blocking ratchet
  for the next editor.'
workflow: v1
---
# guard_skill_doc_headroom.sh: validate that a demanded cap raise clears the hook's own warn threshold

## Goal

`scripts/guard_skill_doc_headroom.sh` blocks (exit 2) an edit to an agent/skill spec whose measured size exceeds its registered cap, demanding the cap be raised **in the same change**. It never validates that the raised value actually clears its own warn threshold (`EPM_SKILL_DOC_HEADROOM_WARN_BYTES`, default 2,000 B, `guard_skill_doc_headroom.sh:107`). Add that validation at raise time — in the hook, or as a `workflow_lint.py` self-check — so a raise that lands *below* the warn threshold fails loud instead of passing every gate.

## Why (incident)

Found by the #2204 round-1 `code-reviewer` while verifying that task's forced scope expansion.

#2204 had to bump `scripts/workflow_lint.py`'s agent-spec cap for `.claude/skills/adversarial-planner/SKILL.md` from 77,000 → 78,000 B because its own deliverable grew that file. Measured size after the edit: 76,589 B. The chosen cap leaves headroom **1,411 B — below the hook's own 2,000 B warn threshold**.

Consequence: every gate passed, the round was PASSed, and the ratchet is now **re-armed for the next editor**. The very next edit to `adversarial-planner/SKILL.md` — even a one-byte typo fix — re-triggers the same blocking exit-2 "raise the cap in the SAME change" cycle, forcing an unrelated task to carry another cap bump. The constant's own documented corridor-max formula, `((measured + 2_800) // 100) * 100` = **79,300** (headroom 2,711, inside both the 2,000 B hook floor and the 3,000 B loose-cap bar), would have cleared it.

So the hook enforces "raise the cap" but not "raise it enough to be useful", which converts a one-time ratchet into a recurring tax on whichever task next touches the file. The failure is silent: nothing in the current gate set can distinguish a healthy raise from a sub-threshold one.

## Acceptance

- At cap-raise time, a raised cap satisfying `cap − measured < EPM_SKILL_DOC_HEADROOM_WARN_BYTES` (default 2,000) fails loud, naming the measured size, the chosen cap, the realized headroom, and the corridor-max value that would clear the threshold.
- Implemented EITHER in `scripts/guard_skill_doc_headroom.sh` (at the point where it verifies a same-change raise) OR as a `workflow_lint.py` self-check over the cap registry — whichever the plan argues is the better seam. State the choice and why.
- Must not fire on caps that were already below-threshold before this change lands (grandfather the existing registry, or the check reds the fleet on entry). Enumerate the current registry's headrooms at plan time and say which entries would trip.
- Must not block a DEDUCTION (a cap lowered after a spec shrank) — the check is about raises leaving usable headroom, not about cap direction per se.
- Tests: a sub-threshold raise fails; a corridor-max raise passes; a grandfathered pre-existing sub-threshold entry does not fire; the loose-cap (3,000 B) bar interaction is pinned.
- The remediation message quotes the corridor-max formula so the next editor gets the right number without re-deriving it.

## Provenance

Surfaced as a prose workflow-fix follow-up by the Claude `code-reviewer` during #2204 round 1 (verdict PASS, Minor finding 1), while independently verifying that #2204's `workflow_lint.py` cap bump was genuinely forced by the hook. Filed by the #2204 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups auto-file + spawn). Distinct target file and distinct fingerprint from #2204's own `scripts/verify_plan.py` deliverable.

Reference points: `scripts/guard_skill_doc_headroom.sh:107` (the warn threshold), `scripts/workflow_lint.py:16328-16332` (the cap constant + its chronicle comment and corridor-max formula), #2204 (the incident round), #2178 / #2325 (prior bumps of the same constant).
