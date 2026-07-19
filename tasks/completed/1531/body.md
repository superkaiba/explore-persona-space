---
title: 'workflow-fix: inline-gate contract test scans all anchors'
kind: infra
tags:
- wf-fix
- wf-fix-fp:25bec420e126
- daily-auto-filed
created_at: '2026-07-19T07:06:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): test_skill_9a_ter_carries_inline_payload_lint_gate
  fails on pristine main (red reproduced at compose time): the test scans a 4,000-char
  window after the FIRST anchor occurrence, and a #1500-era earlier mention shadows
  the canonical 9a-ter block.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1516 (emitting agent: implementer, round 1; parked under the
recursion guard, routed by the 2026-07-18 /daily Step C parked-candidate
sweep).

## Goal

Make `tests/test_inline_payload_lint_gate_contract.py` scan ALL "Inline
payload lint gate" anchor occurrences in SKILL.md (pass if ANY window carries
every needle), or anchor it to the canonical Step 9a-ter heading, so an
earlier mention of the gate no longer false-fails the contract pin.

## Workflow gap

- **Bug observed:** `test_skill_9a_ter_carries_inline_payload_lint_gate`
  fails on pristine main — the test scans a 4,000-char window after the FIRST
  "Inline payload lint gate" occurrence in SKILL.md, but the #1500-era block
  at ~L6524 added an EARLIER occurrence whose window lacks the pinned
  `scripts/workflow_lint.py` needle, so the contract test reads the canonical
  9a-ter block as lost.
- **Why it is a workflow gap:** the contract test keys on first-occurrence
  position rather than the canonical block, so any SKILL.md edit mentioning
  the gate earlier breaks the pin with no real contract drift — a red mapped
  test every Step 9c consumer trips over.
- **Confidence (emitter):** high
- verified-at-filing: `uv run python -m pytest tests/test_inline_payload_lint_gate_contract.py -x -q` → 1 failed (AssertionError at tests/test_inline_payload_lint_gate_contract.py:71) — red reproduced on main at compose time; context read of the test (L50-75) confirms `idx = text.find(ANCHOR)` single-first-occurrence scan + 4,000-char window as claimed; `git log --oneline --since='7 days ago' -- tests/test_inline_payload_lint_gate_contract.py` → 1 commit (31d1c63dba, the #1460 pin itself — no anchor fix landed) (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
- idx = text.find(ANCHOR); window = text[idx : idx + 4000]
+ starts = [m.start() for m in re.finditer(re.escape(ANCHOR), text)]
+ windows = [text[s : s + 4000] for s in starts]
+ for needle, why in (...): assert any(needle in w for w in windows)
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, tests/test_inline_payload_lint_gate_contract.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'Inline payload lint gate' .claude/ CLAUDE.md scripts/ tests/`)
  and confirm which occurrence is the canonical Step 9a-ter block; list every
  hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The fix must keep the pin's teeth: the canonical 9a-ter block losing a
  needle must still FAIL (the any-window form must not let a needle migrate
  to a non-canonical mention and mask real drift — the planner weighs
  any-window vs canonical-heading anchoring on exactly this).
- Full test file passes after the fix; `scripts/workflow_lint.py` no-flags
  run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, tests/test_inline_payload_lint_gate_contract.py
- fingerprint: 4be7f080de5d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md, tests/test_inline_payload_lint_gate_contract.py
bug_observed: test_skill_9a_ter_carries_inline_payload_lint_gate fails on pristine main — the test scans a 4,000-char window after the FIRST "Inline payload lint gate" occurrence in SKILL.md, but the #1500-era block at ~L6524 added an EARLIER occurrence whose window lacks the pinned scripts/workflow_lint.py needle, so the contract test reads the canonical 9a-ter block as lost.
why_workflow_gap: the contract test keys on first-occurrence position rather than the canonical block, so any SKILL.md edit mentioning the gate earlier breaks the pin with no real contract drift — a red mapped test every Step 9c consumer trips over.
proposed_change: scan ALL anchor occurrences (pass if ANY window carries every needle), or anchor to the canonical Step 9a-ter heading.
diff_sketch: |
  - idx = text.find(ANCHOR); window = text[idx : idx + 4000]
  + starts = [m.start() for m in re.finditer(re.escape(ANCHOR), text)]
  + windows = [text[s : s + 4000] for s in starts]
  + for needle, why in (...): assert any(needle in w for w in windows)
confidence: high
related_task: #1516
<!-- /workflow-fix-candidate -->
