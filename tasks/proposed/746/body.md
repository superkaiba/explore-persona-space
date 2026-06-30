---
title: 'workflow-fix: gotchas.md FIX-SCOPE addendum for vLLM enforce_eager probe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e7d3849def58
created_at: '2026-06-30T01:08:34Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #734 r7 implementer''s workflow-fix-candidate (fingerprint
  e7d3849def58)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `<!-- workflow-fix-candidate v1 -->` block raised on task #734 round 7 (emitting agent: experiment-implementer). Sibling to #737 (the r4 (h)-check for train-input-mix fetchability).

## Goal

Add a **FIX SCOPE** addendum to `.claude/rules/gotchas.md`'s vLLM `generate()` hang triad-(b) `enforce_eager` probe entry: when applying the `enforce_eager=True` fix, **grep ALL `LLM(` constructors on the execution path** and thread the `EPM_VLLM_ENFORCE_EAGER` knob into each — fixing only the diagnosed site leaves the class-level deadlock live elsewhere.

## Workflow gap

- **Bug observed:** Task #734 round 5 fixed the cuda-graph-capture deadlock at TWO diagnosed vLLM engine sites (`generate_onpolicy_R` + `representation_shift._generate_responses_vllm`). The fix-engaged relaunch (round 6 PASS) then hit the SAME deadlock at a THIRD on-path `LLM(` site — `scripts/issue664_dispatch.py:_vllm_engine` (reached from `setup_h1_mix`'s subprocess to `--phase p0`'s `_elicit_marker_R`). Round 7 threaded the knob there too, but the class-level trap had silently extended cost by one full GPU launch cycle (pod time + reap + relaunch).
- **Why it is a workflow gap:** the existing gotchas.md vLLM-hang triad-(b) entry tells you HOW to fix the deadlock (`enforce_eager=True`) but not that the fix has FIX SCOPE — "this is a class-level trap; grep ALL engine sites, not just the diagnosed one." The implementer's round-5 brief named 2 sites; gotchas.md didn't surface the discipline of greping wider.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
In .claude/rules/gotchas.md, under the vLLM `generate()` hang triad probe (b) entry:
+ **FIX SCOPE.** The enforce_eager deadlock is a CLASS-level trap: when you apply
+   the `enforce_eager=True` fix, `grep -rnE '\bLLM\('` every script on the run's
+   execution path (dispatcher + every subprocess it shells to) and thread a single
+   shared `EPM_VLLM_ENFORCE_EAGER` knob into EVERY `LLM(...)` constructor. Fixing
+   only the diagnosed site leaves the deadlock live at the rest — it recurs at the
+   next phase that builds an engine (#734 r5→r7: 2 sites fixed, the 3rd in a
+   cherry-picked sibling dispatcher re-hit it at setup_h1_mix).
```

Plus parallel updates the spawned `/issue` session may consider:
- `.claude/agents/experiment-implementer.md` — when the brief names an `enforce_eager` fix, the implementer's pre-coding step grep's the project for all `LLM(` sites and reports the audit.
- `.claude/agents/critic.md` Methodology lens — when a plan cites the `enforce_eager` knob, the critic spot-checks coverage (every on-path LLM site receives the knob).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Sibling: `.claude/agents/experiment-implementer.md`, `.claude/agents/critic.md`
- Grep the workflow surface for the pattern before editing (`grep -rnE "enforce_eager|cuda.graph.capture" .claude/`) — there may be other call sites that warrant the FIX SCOPE callout.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: e7d3849def58

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: A class-level vLLM cuda-graph-capture-deadlock fix (enforce_eager=True) was applied to only the 2 diagnosed engine sites in round 5, leaving a 3rd on-path `LLM(` site (issue664_dispatch._vllm_engine) hardcoded enforce_eager=False, which re-hit the identical deadlock at the next phase.
why_workflow_gap: The gotchas.md vLLM-hang "enforce_eager probe (b)" entry tells you HOW to fix the deadlock but not that the fix has FIX SCOPE — when applying it you must grep ALL `LLM(` constructors on the execution path and thread the knob into each, not just the diagnosed one.
proposed_change: Add a FIX-SCOPE addendum (an (h)-style note) to the existing gotchas.md vLLM enforce_eager probe entry: "when applying the enforce_eager fix, `grep -rnE '\\bLLM\\('` the run's execution-path scripts and thread the EPM_VLLM_ENFORCE_EAGER knob into every constructor — a class-level deadlock recurs at any un-fixed site one phase later."
diff_sketch: |
  In .claude/rules/gotchas.md, under the vLLM `generate()` hang triad probe (b) entry:
  + **FIX SCOPE.** The enforce_eager deadlock is a CLASS-level trap: when you apply
  +   the `enforce_eager=True` fix, `grep -rnE '\bLLM\('` every script on the run's
  +   execution path (dispatcher + every subprocess it shells to) and thread a single
  +   shared `EPM_VLLM_ENFORCE_EAGER` knob into EVERY `LLM(...)` constructor. Fixing
  +   only the diagnosed site leaves the deadlock live at the rest — it recurs at the
  +   next phase that builds an engine (#734 r5→r7: 2 sites fixed, the 3rd in a
  +   cherry-picked sibling dispatcher re-hit it at setup_h1_mix).
confidence: medium
related_task: #734
<!-- /workflow-fix-candidate -->
