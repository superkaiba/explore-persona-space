---
title: 'workflow-fix: Lens 7 mechanical language-intrusion scan (Qwen CJK mixing)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c256a86097b5
created_at: '2026-07-15T03:49:19Z'
has_clean_result: false
origin_prompt: 'prose follow-up from interpretation-critic on #1090 fu4 r1: Lens 7
  needs a mechanical CJK/language-intrusion scan for Qwen on-policy evals'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the
interpretation-critic on task #1090 (fu4 interp round 1).

## Goal

Add a numbered Lens 7 sub-step to the interpretation-critic mandating a mechanical non-target-language script-block scan (CJK regex per arm, trained vs base, firing-overlap recompute) whenever on-policy completions come from a Qwen-family model under an English-context eval.

## Workflow gap

- **Bug observed:** fu4 interp round 1: the analyzer's "Chinese-token intrusions only at 1e-4" claim was false at text level — intrusions were on ALL six impolite arms (15.5% on the verdict-carrying arm, 18.5% on the parent-lr control, vs 10-12% base) and a CJK-zeroed bound dropped one headline sub-claim below the band floor; only a mechanical per-arm CJK scan caught it.
- **Why it is a workflow gap:** interpretation-critic.md Lens 7 (raw-text sample plausibility) prescribes sampling but no mechanical language-intrusion scan; Qwen temp-1.0 language-mixing is a recurring artifact class across the project's on-policy evals, so each critic round re-discovers it by luck.
- **Confidence (emitter):** high (mechanizable: yes — ~15-line script, could live beside verify_task_body.py as a reusable helper)
- verified-at-filing: `grep -cn 'CJK\|language-intrusion\|non-English\|script block' .claude/agents/interpretation-critic.md` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-15 UTC)

## Proposed change (candidate diff sketch — refine in planning)

```
Lens 7 step 3, add sub-step:
+ 3b. **Language-intrusion scan (mechanical; REQUIRED when the evaluated
+     completions come from a Qwen-family model under a non-CJK-context
+     eval):** count completions per arm containing non-target-language
+     script blocks (CJK regex), trained vs base; recompute the overlap
+     with firing labels; report per-arm counts + a zeroed-intrusion
+     bound on any headline rate. (Origin: #1090 fu4 — intrusions on all
+     six impolite arms; one headline sub-claim was artifact-sensitive.)
```

## Scope / surfaces

- Primary target: `.claude/agents/interpretation-critic.md`
- Consider a small reusable helper script (e.g. `scripts/scan_language_intrusions.py`) per the emitter's mechanizable note; the planner decides.
- Grep the workflow surface for the pattern before editing (`grep -rln 'Lens 7' .claude/`) and keep clean-result-critic/analyzer cross-references consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/interpretation-critic.md
- fingerprint: c256a86097b5

Surfaced prose (verbatim, interpretation-critic #1090 fu4 r1): ".claude/agents/interpretation-critic.md Lens 7 could mandate a mechanical non-English-token/language-intrusion scan (CJK regex per arm + firing-overlap recompute, trained vs base) whenever on-policy completions come from a Qwen-family model under an English-context eval — this round's 'only at 1e-4' mischaracterization was only catchable by that scan, and Qwen temp-1.0 language-mixing is a recurring artifact class across the project's on-policy evals. Concrete change: add a numbered sub-step to Lens 7 step 3 ('count completions containing non-target-language script blocks; compare trained vs base; report overlap with firing labels'). mechanizable: yes."
