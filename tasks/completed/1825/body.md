---
title: 'workflow-fix: gotchas entry — reused-module flag contract trap (same flag,
  opposite semantics)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:82f9e0c1f8ad
created_at: '2026-07-29T09:34:40Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 2 (epm:failure-lesson
  v1, 2026-07-29): reused-module hf-prefix contract — see the lesson block in the
  filed body''s Provenance section'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 2).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the reused-module CLI-flag contract trap: sibling reused modules can give the SAME flag name OPPOSITE semantics — read the CONSUMING function's path join before wiring defaults, then class-sweep every caller of the same consumer.

## Workflow gap

- **Bug observed:** #1776's comparator bg job 404'd mid-GPU-run (att-20260729-082617): `issue1776_comparator_fit.py` wired `--hf-prefix` from the flag name / the sibling capture driver's usage. #779's capture driver (N1G) treats `--hf-prefix` as the ROUND ROOT (appends `final_token_capture` itself) while the fits module (N1M, `issue779_ffc_n1m_fits.assemble_multilayer`) consumes it VERBATIM as the capture prefix (flat-joins `<hf_prefix>/<name>.pt`). Two more callers (`issue1776_phase4.py cmd_refit_split`, `issue1776_phase5.py assemble_test_leg_and_anchors`) carried the identical latent wrong default — each would have cost a further full launch cycle (one 404 per cycle).
- **Why it is a workflow gap:** the artifact-reuse checklist (`.claude/rules/artifact-reuse.md`) covers reused ARTIFACT fitness, and gotchas.md covers many reuse traps, but nothing on the workflow surface warns that reused CODE modules' CLI/ns flag semantics diverge across siblings — the trap recurs whenever a per-issue script wires a reused module's defaults (the standing "reuse existing experiment code" default makes this a hot path).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -i "hf.prefix\|prefix contract\|same flag\|flag name" .claude/rules/gotchas.md` → 1 hit in 1 file (line 163, the unrelated fanout shared-staging-race entry; absence claim — no entry covers same-flag-opposite-semantics across sibling reused modules) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Reused-module CLI flags: same flag name, OPPOSITE semantics across siblings —
+   read the CONSUMING function's path join before wiring defaults.** #779's capture
+   driver (N1G) treats --hf-prefix as the ROUND ROOT (appends final_token_capture
+   itself); the fits module (N1M assemble_multilayer) consumes it VERBATIM as the
+   capture prefix (flat-joins <hf_prefix>/<name>.pt). Wiring a caller's default from
+   the flag NAME or a sibling module 404s mid-GPU-run. Rules: read the consumer's
+   join code; reference the consumer's own constant, never a re-typed literal;
+   class-sweep every caller of the same consumer in the same fix round (#1776:
+   three callers carried the identical wrong default); fix-engaged probe = drive
+   the real entrypoint argv and check the remote index at the resolved prefix.
+   Long-form twin: .claude/agent-memory/experiment-implementer/feedback_reused_module_hf_prefix_contract.md
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'final_token_capture\|hf-prefix' .claude/ CLAUDE.md`) and update every hit if the entry belongs elsewhere too; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; gotchas.md `paths:` frontmatter untouched unless the trigger set genuinely widens.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 82f9e0c1f8ad

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p0_comparator_launch (issue1776_comparator_fit.py → issue779_ffc_n1m_fits.assemble_multilayer chunk stream)
lesson: Sibling reused modules can give the SAME flag name OPPOSITE semantics — #779's capture driver (N1G) treats --hf-prefix as the ROUND ROOT (appends final_token_capture itself) while the fits module (N1M) consumes it VERBATIM as the capture prefix (flat-joins <hf_prefix>/<name>.pt). When wiring a reused module's CLI/ns defaults, read the CONSUMING function's path join — never infer the contract from the flag name or a sibling module — then class-sweep every caller of the same consumer in the same round (three #1776 callers carried the identical wrong default; fixing only the crashed one would have burned two more launch cycles).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
