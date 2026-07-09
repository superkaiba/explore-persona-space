---
title: 'workflow-fix: fix/adjudicate exit-site marker misses on main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:52aa3b09821d
- daily-auto-filed
created_at: '2026-07-09T07:00:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): tests/test_step_completed_resume.py::test_every_exit_site_posts_marker
  fails on unmodified main (post-baseline SKILL.md merges broke the ±6-line token-proximity
  contract) and the failure is not in the step9c known-red ledger. [merged sibling:
  tests/test_step_completed_resume.py::test_every_exit_site_posts_marker FAILs on
  pristine main: SKILL.md EXIT-mentioning lines (incl. L350 user-pause park, L9'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #1053 by a recursion-guarded workflow-fix session.

## Goal

Make test_every_exit_site_posts_marker green on main again: for each EXIT-site proximity miss in .claude/skills/issue/SKILL.md, either add the `post_step_completed.py` call within the ±6-line window, refactor the EXIT mention into reference (non-action) prose, or adjudicate it as legitimately reference and refresh the step9c known-red baseline.

## Workflow gap

- **Bug observed:** test_every_exit_site_posts_marker FAILs on current main with multiple exit-site proximity misses; it passed at the 2026-07-05 09:30Z step9c baseline refresh (3b0f7a32cb) and is NOT in the known-red ledger, so every /issue session's Step 9c gate now depends on the --run-pristine oracle stripping it as pre-existing.
- **Why it is a workflow gap:** A red workflow-invariant test on main degrades the Step 9c gate for every concurrent session and erodes the exit-site marker contract itself.
- **Confidence (emitter):** prose-followup (code-reviewer round 1 on #1053)
- **Triage evidence (2026-07-08):** Reproduced 2026-07-08: `uv run pytest tests/test_step_completed_resume.py::test_every_exit_site_posts_marker` FAILS on current main (0.56s; proximity misses incl. SKILL.md L350/L906/L4074/L5353/L5740/L6297...), and the test name is ABSENT from .claude/cache/step9c-baseline.json (not in the known-red ledger). #1053 itself completed without touching this; no newer commit to tests/test_step_completed_resume.py since 07-05. No open dedup; no retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
For each miss listed by the failing test (SKILL.md L350, L906, L4074, L5353,
L5740, L6297, ... — re-run the test for the current list):
+ add `uv run python scripts/post_step_completed.py --issue <N> --step <id>
+   --exit-kind <clean|parked|failure-exit> --notes "..."` near the EXIT, or
+ move the EXIT sentence into the §5 reference section, or
+ adjudicate + `scripts/step9c_baseline.py` ledger refresh.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Secondary: `.claude/cache/step9c-baseline.json` via `scripts/step9c_baseline.py` (adjudication path), `tests/test_step_completed_resume.py` only if the proximity contract itself needs a documented amendment.
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: parked candidate on task #1053 at 2026-07-05T18:04:12Z

Verbatim parked note:

source: prose-followup (code-reviewer round 1). routed: parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, workflow-fix-on-bug.md § Recursion guard). Candidate: tests/test_step_completed_resume.py::test_every_exit_site_posts_marker fails on CURRENT MAIN with 14 sites (post-baseline SKILL.md merge broke the ±6-line token-proximity contract; test passed at the 09:30Z step9c baseline refresh 3b0f7a32cb and is NOT in the known-red ledger). target_file: .claude/skills/issue/SKILL.md (14 main-side sites) and/or the step9c baseline ledger. Proposed: fix the 14 main-side proximity misses or adjudicate + refresh the baseline. Step 9c gates degrade gracefully meanwhile via the --run-pristine oracle (fails-on-main → pre-existing → stripped). Next human/orchestrator pass should file this.


### Merged sibling candidate (s1-exit-site-marker-test-red, from task:1047 at 2026-07-05T14:13:10Z)

- bug_observed: tests/test_step_completed_resume.py::test_every_exit_site_posts_marker FAILs on pristine main: SKILL.md EXIT-mentioning lines (incl. L350 user-pause park, L906, L4074, and ~11 more) lack a post_step_completed.py call within ±6 lines and are not excluded as meta-prose — trunk-red for every session running the consumer suite (the #931 class).
- proposed_change: For each failing site, either add the post_step_completed.py invocation within the ±6-line window (genuine action EXITs, e.g. the user-pause park with exit_kind=parked) or extend META_PHRASES / the doc-section cut for meta-prose mentions, restoring a green test on trunk.
- origin note (verbatim): Verified failing on main 2026-07-08: `uv run pytest tests/test_step_completed_resume.py::test_every_exit_site_posts_marker` FAILED in 0.25s with unaddressed sites at SKILL.md L350 ('3. EXIT the turn...'), L906, L4074, and more — the exact candidate premise. No open task targets this test.
