---
title: composer-side trigger-dense pointers for Codex twins
kind: infra
tags:
- daily-held
created_at: '2026-07-10T06:54:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 3): Codex twin composers share
  the #1058 filter-kill vector; trigger-dense-review.md is role-generic and does not
  bind the composers. Emitter marked CONDITIONAL: file only on recurrence evidence'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1185.

## Goal
Add composer-side trigger-dense-review pointers to codex-code-reviewer.md and codex-clean-result-critic.md — CONDITIONAL: the emitter gates this on recurrence evidence of composer/wrapper filter-kills.

## Workflow gap
- **Bug observed:** Codex twin composers share the #1058 filter-kill vector (composer-side prompt text quoting gated content); trigger-dense-review.md is role-generic and its guidance does not bind the thin composer wrappers. The emitter marked this conditional — 'file only on recurrence evidence' — so this is a judgment call for triage, not an unconditional fix.
- **Why it is a workflow gap:** If composer-side kills recur, each costs a review-round spawn; a one-line pointer in the two composer specs closes it. If they do not recur, the addition is spec noise.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none — one-line pointer to .claude/rules/trigger-dense-review.md in each composer's prompt-composition instructions, contingent on recurrence evidence.)

## Scope / surfaces
- Primary target: `.claude/agents/codex-code-reviewer.md, .claude/agents/codex-clean-result-critic.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/agents/codex-code-reviewer.md, .claude/agents/codex-clean-result-critic.md
- fingerprint: 136169962192

Parked prose-followup on #1185, 2026-07-09T18:34:22Z (Methodology critic, Phase 2, plan v2 review): 'composer-side pointer additions are a one-line follow-up IF wrapper kills recur. Conditional — file only on recurrence evidence.' confidence: low. Routed route-3 (genuine judgment call: the emitter's own condition is unevaluated).

## Resolution (2026-07-18)

**DECISION: no-change — archived** after full adversarial evaluation (plan v1 = the reasoned deflection report; verify_plan.py PASS; fact-checker 7/7 CONFIRMED; adapted Alternatives-lens critic APPROVE, zero Must-Fix).

- The gating condition (recurrence evidence of composer/wrapper filter-kills) is **unmet**: zero composer-side kills recorded since filing — or ever (#1058/#1098/#1152/#1413 were review-role reads, return-text, and orchestrator-brief kills; independently re-swept by the fact-checker).
- **Superseded channels landed since filing:** #1231 (Step 5a excerpt-file + `return_text:` lines delivered in BOTH reviewer briefs on trigger-dense rounds), #1252 (trigger-dense-review.md discipline 4 binds the Codex twin wrappers' return text), #1275 (file-only verdict posting; findings passed by reference), #1461 (revision-round briefs by reference). The critic additionally verified the composer's canonical assembly path (codex-code-reviewer.md Step 2-pre: task.py redirect → `template.replace()`) routes gated content around composer-generated text entirely.
- **Byte cost is real:** codex-code-reviewer.md sits 447 B under its 59,200 B lint ratchet; a third redundant delivery channel is the emitter's own named "spec noise" branch.
- **REOPEN criterion (unconditional):** ONE recorded composer-side filter-kill — a spawn killed during prompt composition (not Codex-runtime, not orchestrator collection), attributed to the composer's own generation — re-files this immediately; a closed workflow-fix task never blocks a re-raise (workflow-fix-on-bug.md § Dedup). The re-filed fix should bind the ASSEMBLY MECHANICS (inlined gated content assembled by shell redirection, never model-generated Write, on trigger-dense rounds) rather than a one-line pointer.

Full report: `plans/v1.md` on this task.

