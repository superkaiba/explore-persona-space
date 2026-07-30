---
title: 'daily-fix: planner.md 40900B over 40000 ratchet — main red'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2fe2e58470e8
- daily-auto-filed
created_at: '2026-07-28T06:40:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): origin/main planner.md
  is 40900 bytes, over the 40000-byte agent-spec-size ratchet FAIL threshold; full-suite
  pytest red on test_live_tree_passes since 2026-07-27T13:59Z'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 Step C (manual routing of a LOST urgent park —
see the companion filing `sweep-urgent-park-predicate`). Task #1718's
autonomous session posted a mechanically-routable URGENT-PARK workflow-fix
candidate (`urgency: main-red`, fp `06bc0203d759`, ts `2026-07-27T14:38:28Z`)
that the watcher's urgent-park router and the Step C sweep both failed to
enumerate (park-predicate miss). /daily is routing it by hand: `main` has been
red on this ratchet since ~2026-07-27T13:59Z.

## Goal

Shrink `.claude/agents/planner.md` below the 40000-byte agent-spec FAIL
threshold by relocating per-scenario content to `.claude/rules/` (the #829
remedy the lint message itself names), so `workflow_lint.py
--check-agent-spec-size` and `tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes`
go green on `main`.

## Workflow gap

- **Bug observed:** origin/main `.claude/agents/planner.md` is 40900 bytes,
  over the 40000-byte agent-spec-size ratchet FAIL threshold. Full-suite
  pytest has been red on `test_live_tree_passes` since #1721's merge
  (`028b45ff44`, planner.md 39371 → 40900) at ~2026-07-27T13:59Z. Every
  intervening session's Step 9c gate must re-classify the red, and #1718's
  Step 10d merge is blocked by it (mutually-blocked: #1718's landed
  caps-migration cannot merge while main is red).
- **Why it is a workflow gap:** an agent-spec file crossed its own lint
  ratchet on main; the fix is workflow-surface content relocation.
- **Confidence (emitter):** high
- verified-at-filing: `uv run python scripts/workflow_lint.py
  --check-agent-spec-size` → FAIL, 1 error naming planner.md 40900 > 40000;
  `timeout 180 uv run pytest tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes -x -q`
  → 1 failed; `wc -c .claude/agents/planner.md` → 40900 (all run
  2026-07-28T06:4xZ at the main checkout). Landed-fix check:
  `git log --oneline -5 -- .claude/agents/planner.md` → newest commit
  `028b45ff44` GREW the file (the cause); no shrink landed since.

## Proposed change (candidate diff sketch — refine in planning)

Relocate the largest per-scenario planner.md sections (candidates: the §11
grounding worked examples, the compute-sizing scenario blocks — planner's
choice) to a new or existing `.claude/rules/*.md` loaded on demand, leaving a
pointer, until `wc -c` < 40000 with comfortable margin (~2-3 KB). Do NOT
raise the 40000 threshold and do NOT add planner.md to
`AGENT_SPEC_SIZE_GRANDFATHER` (the ratchet exists to force relocation; #1718
is separately migrating the grandfather dict — coordinate, don't collide:
check `origin/issue-1718` before touching `scripts/workflow_lint.py`, and
prefer touching ONLY planner.md + the new rules file).

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- Secondary: a `.claude/rules/*.md` relocation target + `.claude/rules/LESSONS.md`
  index row if a new rule file is created.
- Related open work: #1718 (blocked at Step 10d by this red; its branch
  migrates `AGENT_SPEC_SIZE_GRANDFATHER` in `scripts/workflow_lint.py`).
  After this task merges and main goes green, #1718 can re-drive its merge.

## Constraints / invariants

- Workflow-surface only.
- `scripts/workflow_lint.py` no-flags run + `--check-lessons-index` pass;
  the full `tests/test_workflow_lint_agent_spec_size.py` file passes.
- No planner behavior change beyond content relocation: the relocated text
  must stay reachable via a rule pointer (LESSONS.md row + `paths:`/trigger
  frontmatter as appropriate).
- This session runs under a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: 2fe2e58470e8
- origin_candidate: task #1718 `epm:workflow-fix-candidate` ts
  `2026-07-27T14:38:28Z`, self-declared fp `06bc0203d759`, fields
  `urgency: main-red` / `failing_test:
  tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes` /
  `wf_fix: true`. Routed manually by /daily 2026-07-27 because the sweep's
  park predicate missed the URGENT-PARK note form (companion filing:
  `sweep-urgent-park-predicate`).
