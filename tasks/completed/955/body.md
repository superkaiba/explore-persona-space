---
title: 'workflow-fix: gotchas.md anthropic system-role lift + live-probe rule for
  dispatcher seams'
kind: infra
tags:
- wf-fix
- wf-fix-fp:218e224d86c4
created_at: '2026-07-04T03:37:12Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #906 r11 crash-fix (datagen system-role
  400)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #906 (emitting agent: experiment-implementer, r11 crash-fix round).

## Goal

Add a gotchas.md entry (and audit sibling request-builder seams) for the Anthropic Messages API system-role rejection: message lists forwarded verbatim 400 on role:"system" — system content must be lifted to the top-level system= param, and any live-API dispatcher seam needs a tiny LIVE system-BEARING probe (mock smokes structurally cannot catch the shape bug).

## Workflow gap

- **Bug observed:** #906's first --full run failed its sycophancy class: all 36 datagen generation requests returned HTTP 400 because artifacts/datagen.py::_default_generate_fn.build_request forwarded gen_messages verbatim including a {"role": "system"} entry. Ten rounds of contract-test hardening never caught it.
- **Why it is a workflow gap:** .claude/rules/gotchas.md has no entry for this API-shape trap, and no rule requires a live system-bearing probe at never-live-executed dispatcher seams; the same trap silently threatens every other seam that builds anthropic requests from message lists.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md: "Anthropic Messages API rejects role:'system' in messages — lift to top-level system= (canonical helper: artifacts/datagen.py::_gen_params_from_messages, #906 r11). Any dispatcher seam forwarding message lists verbatim needs a live forced probe with a system-BEARING list; mock smokes and system-less live smokes cannot catch it."
+ grep the repo for other anthropic request-builder seams (messages.create / with_raw_response.create callers) and verify each either lifts system or never receives system-role lists; fix or annotate each.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface + src/ anthropic call sites before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only for the rule text; any src/ seam fixes found by the audit are separate follow-ups unless trivial.
- scripts/workflow_lint.py passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 218e224d86c4

<!-- epm:failure-lesson v1 --> (verbatim lesson from #906 r11: the Anthropic Messages API rejects "system" as a message ROLE — lift to top-level system=; datagen's _default_generate_fn was the one seam the mocked suite never executes; live system-bearing probe required.) <!-- /epm:failure-lesson -->
