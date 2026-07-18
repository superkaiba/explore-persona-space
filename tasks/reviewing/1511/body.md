---
title: 'workflow-fix: recompute caption count claims against figure sidecar points'
kind: infra
tags:
- wf-fix
- wf-fix-fp:59be4adc04ec
created_at: '2026-07-18T13:36:26Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1426 r2 surfaced prose: recompute quantified
  caption count claims (all-N/K-of-N/none below-above zero) against value-bearing
  .meta.json sidecar points (mechanizable: yes)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix suggestion raised on task #1426 (emitting agent: clean-result-critic, round-2 verdict "Follow-ups (orchestrator should consider)").

## Goal

Extend `scripts/verify_task_body.py`'s figure-sidecar plotted-value-drift family to recompute quantified caption COUNT claims — "all N <unit> lie below/above zero", "K of N ...", "none ..." — against the cited figure's value-bearing `.meta.json` sidecar points, FAILing (or distinctly WARNing) when the recomputed count contradicts the claim.

## Workflow gap

- **Bug observed:** a #1426 caption claimed "all 50 contexts lie below zero" while the pinned sidecar (`figures/issue_1426/mlc_percontext_delta_scatter.meta.json` @ 4a65c36ab0) carried a positive point (+0.004368, f1_house_medical_doctor); a sibling caption glossed a heavy-tailed distribution (4/50 positive, 6/50 near zero) as "nearly all points fall far below zero". Both passed every mechanical check because the captions carried no bolded decimal for the existing drift checks to key on; the contradiction surfaced only at clean-result-critic Lens 3, costing a REVISE round.
- **Why it is a workflow gap:** the verifier's plotted-value-drift family keys on numeric literals in captions; countable set claims over sidecar point arrays are recomputable mechanically but unchecked — an LM-judgment gate is doing work a sidecar recompute can do.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "plotted-value-drift\|count claim" scripts/verify_task_body.py` → hits confirm the drift family + count-claim machinery exist for panel counts / HF-tree counts (lines 489, 583, 7130, 7248) but none recompute caption below/above-zero counts from sidecar point values (2026-07-18); `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → recent check additions (30/40/41 families) do not cover this claim shape.

## Proposed change (candidate diff sketch — refine in planning)

New WARN/FAIL check in the sidecar-drift family: for each embedded figure whose `.meta.json` sidecar carries value-bearing points, scan the adjacent blockquote caption for count-claim shapes (`all <N>`, `<K> of <N>`, `none`, `nearly all` + a below/above-zero predicate); recompute the count from the sidecar's point values; flag a contradiction naming the offending point(s). The purely-verbal half ("nearly uniform") stays LM judgment.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep before editing (`grep -rn 'sidecar\|meta.json' scripts/verify_task_body.py .claude/skills/clean-results/SPEC.md`); keep SPEC.md + clean-result-critic Lens 3 text consistent; pin with tests in `tests/test_verify_task_body.py` (a fixture caption claiming "all N below zero" against a sidecar with one positive point must flag).

## Constraints / invariants

- Workflow-surface only; forward-only (grandfathered v3/v2 bodies never newly hard-FAILed); ruff + workflow_lint pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 59be4adc04ec

Surfaced prose (clean-result-critic, #1426 round 2): "extend scripts/verify_task_body.py's figure-sidecar checks (the plotted-value-drift family) to recompute quantified caption count claims — 'all N <unit> lie below/above zero', 'K of N ...', 'none ...' — against the cited figure's value-bearing .meta.json sidecar points; this round's mlc caption ('all 50' vs a +0.004 positive point in the sidecar) passed every mechanical check because no bolded decimal was present. Complements round 1's check-31 exemption-phrase tightening follow-up."
