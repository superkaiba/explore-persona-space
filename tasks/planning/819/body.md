---
title: 'workflow-fix: update stale GCP max-run-duration 24h→7d in critic/planner/SKILL/workflow.yaml'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0c3d8b361ea9
created_at: '2026-07-01T22:37:19Z'
has_clean_result: false
origin_prompt: 'Surfaced prose follow-up from codex-critic methodology composer on
  #779: critic.md item 13 still cites the GCP --max-run-duration default as 24h; #741
  raised it to 7d. Sibling stale hits in planner.md, issue SKILL.md line 2644, workflow.yaml
  line 2345.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #779 (emitting agent: codex-critic methodology prompt-composer, 2026-07-01).

## Goal

Update the stale GCP `--max-run-duration` default from "24h" to "7d" everywhere the workflow surface still quotes the old value (the default was raised to 7d by #741).

## Workflow gap

- **Bug observed:** `.claude/agents/critic.md` item 13 (line 85) says "GCP 24h fence" / "default 24h"; sibling stale hits: `.claude/agents/planner.md` (~line 795 "default … 24h)"), `.claude/skills/issue/SKILL.md` line 2644 ("`--max-run-duration` (default 24h)"), `.claude/workflow.yaml` line 2345 ("GCP --max-run-duration default 24h"). CLAUDE.md + LESSONS already say 7d (#741).
- **Why it is a workflow gap:** critics/planners quoting item 13 will demand fence mitigation for any >20h plan against a 24h fence that is actually 7d — producing spurious REVISEs and wrong §9 reconciliation guidance.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

- `.claude/agents/critic.md` line 85: `--max-run-duration`, default 24h → default 7d; adjust the "~20h" REVISE trigger wording to key off the actual fence (e.g. "worst-case wall approaching the fence") or restate as ~6d; KEEP the historical #599 incident text ("hard-deleted at step 149/2400 by the 24h fence" — the fence WAS 24h then).
- `.claude/agents/planner.md` ~795: same default correction; keep line 804's historical incident reference.
- `.claude/skills/issue/SKILL.md` line 2644: "(default 24h)" → "(default 7d — the FLEX_START ceiling, #741)".
- `.claude/workflow.yaml` line 2345: "default 24h" → "default 7d".

## Scope / surfaces

- Primary targets: `.claude/agents/critic.md`, `.claude/agents/planner.md`, `.claude/skills/issue/SKILL.md`, `.claude/workflow.yaml`
- Grep the workflow surface for the pattern before editing (`grep -rn 'default 24h\|24h fence' .claude/ CLAUDE.md scripts/`) and update every non-historical hit; list them in the plan. Historical incident references (#599's 24h-fence deletion) stay verbatim.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; workflow.yaml stays consistent with CLAUDE.md + the rule files.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/critic.md,.claude/agents/planner.md,.claude/skills/issue/SKILL.md,.claude/workflow.yaml
- fingerprint: 0c3d8b361ea9

Surfaced prose (verbatim, from the codex-critic methodology composer's return on #779):
"critic.md item 13 line 85 still says 'GCP 24h fence' / 'default 24h', but CLAUDE.md's compute-backend section and the LESSONS index now describe the `--max-run-duration` default as 7d (#741). That is a stale-number drift on a workflow-surface file (`.claude/agents/critic.md`), likely to recur when this item is next quoted. If you decide it's in-scope, the fix is a one-line update to critic.md item 13's fence value."
