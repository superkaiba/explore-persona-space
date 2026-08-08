---
title: 'workflow-fix: lean-context subagent twins — thrash ladder ha'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e859432556c1
- daily-auto-filed
created_at: '2026-08-04T06:50:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): ~29 subagent autocompact-thrash
  deaths across six autonomous sessions in one evening; micro-scoped default-model
  respawn (the only CLAUDE.md:151 remedy) failed repeatedly, incl. 3 deaths on an
  11-line diff and 4 at a 4-tool-use signature; orchestrators fell back to inline
  planning/implementation/review, losing fresh-context independence at every review
  site.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep. Raised independently by FIVE miners over six autonomous sessions.

## Goal

Give the autocompact-thrash ladder a rung that works when EVERY spawn shape thrashes: a lean-context twin for the spec-heavy subagent roles (the registered `analyzer-lean` precedent), and/or a reduction of the fixed overhead every subagent inherits.

## Workflow gap

- **Bug observed:** ~29 subagent autocompact-thrash deaths across six autonomous sessions in one evening (2026-08-03 20:31Z → 2026-08-04 06:28Z), after which the orchestrators abandoned subagent dispatch and did planning / implementation / code review INLINE, losing the fresh-context independence those roles exist to provide. Per-session counts (each from the session's own thrash-diagnosis rows, one per spawn): #2061 seven deaths (planner ×2, fact-checker, consistency-checker, methodology-critic, implementer ×2 — session 5b6d8c4a); #2054 four consecutive implementer deaths (fd36b0e3, all at the 4-tool-use signature); #2051/#1980/#1982 eight (52bc7fdf, 4840c611, 3654b1da); #1978/#1981 ~ten (908b5ad1, cd57c423 — "Four subagent thrash-deaths in a row (2 implementers + 2 reviewers), all with zero durable state"); #2057 three on an **11-line diff** (51ee4be6).
- **Why it is a workflow gap:** CLAUDE.md:151 (§ Autocompact-thrash subagent deaths) prescribes exactly two remedies — respawn MICRO-SCOPED on the DEFAULT model, and explicitly do NOT pin a smaller model. Both were followed and both failed: #2054's session reports "Third death at 4 tool uses — the `experiment-implementer` subagent's fixed overhead alone is at the ceiling on this branch", and #2057's reports the diff was 11 lines (not over budget) so "the death is fixed-overhead pressure on the subagent window". The ladder terminates at "the orchestrator composes the artifact itself", which silently converts every adversarial-review site into a self-review by the orchestrator — the exact independence the /issue pipeline is built to preserve. Three orchestrators ALSO hit their own context ceilings in the same window, so the inline fallback is not free either.
- **Confidence (emitter):** high (the death counts and the sessions' own root-cause reads are transcript-grounded; the fix SHAPE is the open design question for the planner)
- verified-at-filing: `ls .claude/agents/*lean*` → 0 lean twins in-repo for planner / critic / implementer / code-reviewer / fact-checker / consistency-checker (the two hits are `clean-result-critic.md` / `codex-clean-result-critic.md`, substring matches on "clean", not lean twins). `grep -c 'autocompact' CLAUDE.md` → 1; `grep -n 'MICRO-SCOPED' CLAUDE.md` → line 151, the single ladder paragraph, whose terminal rung is orchestrator self-compose. `wc -l .claude/agents/{implementer,experiment-implementer}.md` → 336 / 1289 (measured in-session by fd36b0e3 at its L143/L158). unverified hypothesis — verify at plan time: `analyzer-lean` is a registered agent type (it appears in the session's agent-type registry, described as a lean-context twin with a restricted tool list) but resolves outside this repo's `.claude/agents/`, so the planner should locate it before copying the pattern.
- unverified hypothesis — verify at plan time: the dominant overhead term. Two sessions attribute it to MCP-schema volume specifically (51ee4be6: "this session's ~84 google-workspace MCP schemas + on-demand rule imports"), others to the agent spec + CLAUDE.md import tree. These imply DIFFERENT fixes (per-session MCP scoping vs. spec/import slimming vs. lean twins) and the planner should measure before choosing.

## Proposed change (candidate sketch — refine in planning)

Two independent legs; the planner should size both and may land either or both:

```
(a) lean-context twins in .claude/agents/ for the spec-heavy roles that thrashed
    (planner, critic, experiment-implementer, code-reviewer, fact-checker,
    consistency-checker), each a thin spec that READS its full sibling spec as
    its authoritative instructions with a restricted tool list — the registered
    analyzer-lean pattern, which exists precisely because an all-tools spawn
    loads every MCP schema (~138K tokens measured 2026-07-03).

(b) a THIRD rung in the CLAUDE.md:151 ladder for "micro-scoped default-model
    respawn also thrashed": escalate to the lean twin (leg a) BEFORE the
    orchestrator-self-compose terminal, and record the independence loss on the
    task when self-compose is finally reached.
```

## Scope / surfaces

- Primary target: `.claude/agents/*.md` (new lean twins), `CLAUDE.md` § Autocompact-thrash subagent deaths (line 151), `.claude/skills/issue/SKILL.md` Step 5b respawn recipe.
- Grep the surface before editing (`grep -rln 'autocompact\|thrash' --exclude-dir=worktrees .claude/ CLAUDE.md`) and update every hit; list them in the plan.
- Note the agent-spec size ratchet (`workflow_lint.py`, WARN >28KB / FAIL >40KB) applies to any new agent file.

## Constraints / invariants

- Workflow-surface only.
- `scripts/workflow_lint.py` (no-flags default run) passes; ruff on touched files passes.
- A lean twin MUST NOT silently weaken its role's rubric — it defers to the full spec by reference, it does not restate a subset.
- Do NOT pin a smaller MODEL as the fix (CLAUDE.md:151's inverse-of-refusal-rung-b2 finding stands: on #1090, 3/6 sonnet spawns thrashed where both default-model spawns compacted successfully).

## Provenance

- fingerprint: e859432556c1

- workflow_fix_target: .claude/agents
- fingerprint: PLACEHOLDER
