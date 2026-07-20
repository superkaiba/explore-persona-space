---
title: 'workflow-fix: staged-index check for ignored artifacts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9f474aa18823
created_at: '2026-07-20T22:11:42Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from mixed-turn-fit-958 on #958 round 7: explicit-path
  git add silently excluded convention-committed percell/*.npz via the repo-wide *.npz
  gitignore rule; completion contract needs a staged-index verification + git add
  -f duty.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #958 (emitting agent: experiment-implementer subagent `mixed-turn-fit-958`, inline user-chat round 7; routed by the orchestrator).

## Goal

Add a staged-index verification step to the inline-round completion contract: after the explicit-path `git add`, verify every intended artifact file entered the index (`git status --short` on the round artifact dirs + `git check-ignore` probe) and `git add -f` convention-committed files skipped by ignore rules.

## Workflow gap

- **Bug observed:** An inline round's explicit-path `git add eval_results/issue_958/mixed-turn-fit/` silently excluded the round's convention-committed `percell/*.npz` cells via the repo-wide `*.npz` gitignore rule — the commit shipped without its per-unit cells and the drop was caught only by a manual orchestrator check (#958 round 7, 2026-07-20).
- **Why it is a workflow gap:** The same-turn completion contract (CLAUDE.md user-chat inline carve-out) and the Step 9a-ter inline payload recipe (SKILL.md) both mandate "commit by explicit path" but say nothing about ignore rules; `git add <dir>` silently skips ignored files with rc=0, so any round whose artifact convention includes an ignored pattern (npz per-unit cells being the live example — the parent #958 `percell/` convention commits them force-added) ships an incomplete durable record with no error signal.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'git add -f\|check-ignore' .claude/skills/issue/SKILL.md CLAUDE.md` → 0 hits in both named targets (absence-of-guard claim; the 0-hit in-target result is the evidence); companion presence probe `grep -c 'explicit path' CLAUDE.md .claude/skills/issue/SKILL.md` → 4 / 3 hits (the contract text exists at both targets); landed-fix history check `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 5 commits, all unrelated classes (#1560/#1563/#1558/#1562/#1540) (2026-07-20)

## Proposed change (candidate diff sketch — refine in planning)

```
+ After the explicit-path `git add`, verify the staged set: `git status --short <round dirs>` —
+ every intended artifact file shows as staged; for convention-committed files matching an
+ ignore pattern (e.g. percell/*.npz under the repo-wide *.npz rule), `git add -f` them
+ explicitly and re-verify. A silent ignore-rule drop ships the round without its per-unit
+ cells (#958 round 7, 2026-07-20).
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, CLAUDE.md`
- Grep the workflow surface for the pattern before editing (`grep -rn 'explicit path' .claude/ CLAUDE.md`) and update every completion-contract / payload-commit site that stages artifact dirs; list them in the plan. Consider whether subagent-facing briefs (experiment-implementer spec) should carry the same one-line duty.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, CLAUDE.md
- fingerprint: 9f474aa18823

Surfaced prose (verbatim, from the emitting subagent's report): "git, not HF — the percell/*.npz follow the parent eval_results/issue_958/percell/ convention (the brief called for that naming). They're gitignored by the repo-wide `*.npz` rule, so my explicit-path `git add` silently excluded them (my round-5 template, dup-excluded-turn1-refit, was JSON-only, so I didn't force-add) — they need `git add -f`."
