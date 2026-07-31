---
title: 'workflow-fix: verify_plan check-ignore on declared-committed paths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4dc16f20bf31
created_at: '2026-07-31T00:47:55Z'
has_clean_result: false
origin_prompt: 'Methodology critic round-1 prose follow-up on #1900: add verify_plan.py
  git check-ignore check on plan-declared committed output/config paths (third incident
  in class: #958, #734, #1900 plan)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1900 (emitting agent: critic, Methodology lens, round-1 prose
follow-up "Follow-up (orchestrator should consider)").

## Goal

Add a `verify_plan.py` check that extracts plan-declared committed output/config
paths and runs `git check-ignore` on each; WARN when a declared-committed path
is gitignored with no force-add / staged-index-verification note.

## Workflow gap

- **Bug observed:** plan #1900 v4 declared `data/issue_1900/config/{subset.json,arms.json}` "committed to the issue branch (rides the git clone to any lane)" while `.gitignore:17` (`data/*`) matches the path — a plain `git add` exits 0 while silently skipping ignored files (the #958 signature), so clone-based lanes (GCP, fellows) would provision, stage stores, then crash at the P1 config read (the #734 shape). Caught only by the round-1 Methodology critic.
- **Why it is a workflow gap:** this is the third incident-shape in this class (#958 silent dir-add skip; #734 lane-unreachable reused input; #1900 plan), and it is a pure mechanical presence/regex check that `verify_plan.py` (the Phase 1.5.0 pre-pass that runs before any critic spawns) could catch for free.
- **Confidence (emitter):** high
- verified-at-filing: `grep -cn 'check.ignore' scripts/verify_plan.py` → 0 hits in scripts/verify_plan.py (absence-of-guard claim — 0-hit in-target IS the evidence); trigger reproduced: `git check-ignore -v data/issue_1900/config/subset.json` → `.gitignore:17:data/*` (2026-07-30). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` shows 5 commits, none touching check-ignore / committed-path semantics.

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_committed_paths_not_gitignored(plan, ...):  # cNN
+     # extract paths from phase_outputs: blocks + prose "committed to the issue branch" /
+     # "commits to" sentences; for each path under a git-ignorable root, run
+     # `git check-ignore -q <path>`; ignored AND no `git add -f` / force-add /
+     # staged-index-verification note within the same section -> WARN
+     # N/A escape: `N/A — no committed outputs`
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'check-ignore' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. Add the check id to the N/A-escape roster in
  `.claude/skills/adversarial-planner/SKILL.md` Phase 1.5.0 if a new escape
  phrase is introduced, and pin with a test in `tests/test_verify_plan.py`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 4dc16f20bf31

Surfaced prose (verbatim, from the Methodology critic's round-1 report on #1900):
"`scripts/verify_plan.py`: add a check that extracts plan-declared committed
output/config paths (phase_outputs, 'committed to the issue branch' prose) and
runs `git check-ignore` on each; WARN when a declared-committed path is ignored
with no force-add/staged-index-verification note. This is the third
incident-shape in this class (#958 silent dir-add skip, #734 lane-unreachable
reused input, this plan) and is a pure mechanical presence/regex check."
