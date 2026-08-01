---
title: 'daily-fix: widen pre-gate spec-freshness sync set'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6afb47f03ffc
- daily-auto-filed
created_at: '2026-08-01T07:08:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Stale branch copies of
  agent-memory MEMORY.md files, the step9c selector, and never-branch-edited sibling-issue
  scripts/issue*_*.py red the gates / trip Guard 4 (>=5 extra gate runs across 4+
  sessions) — the existing Step 5a/#1742 pre-gate sync SPECS set excludes all three
  classes'
workflow: v1
---
# daily-fix: widen pre-gate spec-freshness sync set

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M3; miner-8:P4, miner-7:P5, miner-3:P16). Source sessions: dcb0374d (#1776 — gate runs 1-3 blocked on stale pin-test/selector copies, the documented alpha-skew class), fbee8ba3 (#1768 r3 — stale `issue1776_p3p4_supplement.py` + 3 agent-memory MEMORY.md indexes main had trimmed), ba1e3c42 (#1768 r4/r5 — stale `scripts/issue1689_user_slot_*.py` copies predating the os-exit fix, ~40 min extra), 1946008e (#1846 — Guard-4 LOST-UPDATE REFUSAL on compute-backend-failover.md + CLAUDE.md), 035a29b8 (#1887 — lint red from 4 branch-stale agent-memory files). Each case was resolved by a hand `git checkout origin/main -- <paths>` + re-run; ≥5 extra gate runs burned across 4+ sessions in one day.

## Goal

Widen the Step 5a / Step 9c pre-gate spec-freshness sync set to cover `.claude/agent-memory/**`, the step9c selector, and branch-carried never-branch-edited sibling-issue `scripts/issue*_*.py` files the gated tests import.

## Workflow gap

- **Bug observed:** Long-lived issue branches carry stale copies of files main has since fixed/trimmed; the gate then reds (or Guard 4 lost-update-refuses) on files the branch never deliberately edited, and every red costs a full gate cycle before a hand-sync fixes it.
- **Why it is a workflow gap:** The pre-gate sync machinery EXISTS — Step 5a's family-atomic sync (#1714) plus the Step 9c "Pre-gate spec-freshness re-sync (#1742)" re-runs it before the gate — but its SPECS set is `.claude/agents .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py .claude/hooks` + the guard/lint/workflow-yaml pin-test globs ONLY. It excludes exactly the three classes that red the day's gates: `.claude/agent-memory/**` (always-appended memory indexes the lint budget checks), `scripts/select_step9c_tests.py` (the selector; only workflow_lint.py is in the set), and sibling-issue `scripts/issue*_*.py` copies that gated tests import. (The guard-SCRIPT `FAMILY_OF` subcase — M4/#1860/#1862 — is being filed separately via tonight's Step C parked-candidate sweep and is OUT of this filing's scope.)
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -n -B2 -A20 'SPECS=' .claude/skills/issue/SKILL.md` → L2476 verbatim: `SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py .claude/hooks tests/test_guard_lessons_edit.py tests/test_workflow_yaml.py tests/test_autonomous_session_watch.py :(glob)tests/test_workflow_lint*.py :(glob)tests/test_guard_*.py"` — `.claude/agent-memory`, `scripts/select_step9c_tests.py`, and `scripts/issue*_*.py` all ABSENT from the set (absence = evidence); `grep -n 'agent-memory' .claude/skills/issue/SKILL.md` → 5 hits, none in the sync set (Guard 0 + memory-append recipes only); Step 9c pre-gate re-sync presence confirmed ("Pre-gate spec-freshness re-sync (#1742): run the Step 5a" — the sync-exists half of the CONSOLIDATED route is therefore rescoped from "add a pre-gate sync" to "widen its set"). `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` eyeballed: no landed set-widening (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md Step 5a (SPECS + FAMILY_OF; the #1742
Step 9c re-sync inherits automatically):
+ SPECS += ".claude/agent-memory scripts/select_step9c_tests.py"
+ FAMILY_OF["scripts/select_step9c_tests.py"]="lint"   # selector is
+   budget-coupled to workflow_lint.py + the pin tests
+ (agent-memory is a singleton family; the existing per-file
+  branch-side-edit guard keeps a branch's own deliberate memory
+  appends un-clobbered — fail-safe direction preserved)
+ NEW pre-gate arm (Step 9c preamble): for branch-carried
+ scripts/issue<M>_*.py where M != this issue AND the branch has no
+ deliberate commits touching them since merge-base (same subject-
+ scoped exclusion as Step 5a), checkout the origin/main copies BEFORE
+ the first gate run instead of after a red.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 5a SPECS/FAMILY_OF + Step 9c preamble).
- Grep the workflow surface for the pattern before editing (`grep -rn 'SPECS=\|spec-freshness' .claude/ scripts/`) — every copy of the sync recipe (Step 5a, the Step 9c re-sync reference, Step 10d if it restates the set) must stay consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; the family-atomic fail-safe direction (#1714: any dirty member widens the skip, never narrows into a clobber) must be preserved for the new entries.
- The sibling-issue-script arm must never clobber files THIS branch deliberately edited (per-file branch-side-edit guard, same as the existing recipe).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 6afb47f03ffc

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)
