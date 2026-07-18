---
title: 'workflow-fix: check-20 WARN-acknowledgment must name each fired class'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0673ceca0111
created_at: '2026-07-18T22:22:03Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1417 r1 prose follow-up: mechanize WARN-ack class
  coverage in verify_task_body check 20'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix prose note
raised on task #1417 (emitting agent: clean-result-critic, round-1 verdict).

## Goal

Extend `scripts/verify_task_body.py` check 20 (v4 conciseness caps): when the
body carries a WARN-acknowledgment sentence, assert that EACH FIRED WARN
class (Takeaways bullet-length / per-result 120-180 band / total-prose budget
/ caption length) is named in it; a fired-but-unnamed class degrades the
acknowledgment (WARN naming the gap), so partial acknowledgments stop
silently shipping.

## Workflow gap

- **Bug observed:** #1417's body shipped a WARN-acknowledgment sentence
  covering two of the three fired check-20 WARN classes (per-result band +
  linked figures acknowledged; the 41-word Takeaways bullet-length WARN was
  not named). The verifier accepts ANY acknowledgment sentence without
  matching it against which classes actually fired; the clean-result-critic
  caught the gap manually at Lens 12.
- **Why it is a workflow gap:** the standing WARN-ship rule is
  "WARNs ship only when acknowledged in body" — enforcement of the
  acknowledgment's COVERAGE is prose-only today; the check is mechanizable
  (the verifier already knows exactly which sub-classes fired).
- **Confidence (emitter):** medium-high (critic marked it `mechanizable: yes`
  in spirit; the class list is finite and verifier-internal).
- verified-at-filing: `grep -cE 'fired.*class.*acknowledg|ack.*class' scripts/verify_task_body.py` → 1 hit, context READ (check 31's linked-figure docstring citing the standing acknowledge-in-body rule — does not implement class-matched assertion); `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 4 commits (checks 45/46/31 families), none touching WARN-ack coverage (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

+ In check 20 (v4 conciseness), after computing the fired WARN sub-classes:
+   ack = _find_warn_acknowledgment_sentence(body)   # existing convention: a
+   # body sentence acknowledging verifier WARNs (e.g. "Verifier WARNs
+   # acknowledged: ...")
+   if fired_classes and ack:
+       unnamed = [c for c in fired_classes if class_keyword[c] not in ack]
+       if unnamed:
+           warn(f"WARN-acknowledgment sentence does not name fired class(es):
+                 {unnamed} — extend the acknowledgment or fix the WARN")
+ Class keywords: bullet-length -> "bullet", per-result band -> "per-result"/
+ "120", total budget -> "total prose"/"budget", caption -> "caption".
+ WARN-tier only (never flips the overall verdict); v4-only, forward-only.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep before editing: `grep -rn 'acknowledg' scripts/verify_task_body.py .claude/skills/clean-results/SPEC.md .claude/agents/clean-result-critic.md` — keep SPEC.md's WARN-ship rule wording + Lens 12's rubric in sync if the acknowledgment convention gets a mechanical shape; add a pin in `tests/test_verify_task_body.py`.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff clean;
  existing verify_task_body tests stay green (forward-only: no new hard FAIL on
  grandfathered bodies).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 0673ceca0111

Verbatim surfaced prose (clean-result-critic, #1417 round 1): "a mechanizable extension for verify_task_body.py — assert each fired check-20 WARN class (bullet-length / per-result band / total budget / caption) is named in the body's WARN-acknowledgment sentence; this round shipped an acknowledgment covering two of three fired classes."
