---
title: 'workflow-fix: verify_task_body FAILs unpinned footer HF artifact paths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a7b285c8ff27
created_at: '2026-07-18T12:40:21Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1335 (critique v4): extend
  verify_task_body.py footer HF-path checks so a bare backtick HF artifact path in
  the **Repro:** footer with no adjacent pinned huggingface.co tree/resolve link FAILs'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the clean-result-critic on task #1335 (emitting agent: clean-result-critic, fold round `onpolicy-assistant-label`, critique v4 Lens 5).

## Goal

Extend `scripts/verify_task_body.py`'s footer HF-path checks so a bare backtick HF-style artifact path in the `**Repro:**` footer (`issue<N>_.../{raw_completions,analysis_tensors}` shapes) with no adjacent pinned `huggingface.co/...(tree|resolve)/<rev>` link FAILs (or at minimum WARNs via the check-40 family extended to the footer-Repro context), instead of passing silently.

## Workflow gap

- **Bug observed:** an unpinned prose HF artifact path in the Repro footer (`issue1335_ablation_ladder/onpolicy_assistant_label/{raw_completions,analysis_tensors}/`) escaped the mechanical verifier; only URL-bearing claims are gated. Caught only by the LM critic (clean-result-critique v4 on #1335, 2026-07-18) — the shape is likely to recur on interpretation-pass folds.
- **Why it is a workflow gap:** `verify_task_body.py` is the mechanical gate for footer reproducibility links; a load-bearing artifact pointer that resolves to nothing clickable defeats the pinned-links contract (SPEC.md § footer) while the verifier reads clean.
- **Confidence (emitter):** medium
- verified-at-filing: executable semantic probe — `uv run python scripts/verify_task_body.py --issue 1335` run 2026-07-18 against the offending body (pre-fix) returned PASS with WARNs only (the critic's own run, reproduced this session); `grep -n 'huggingface.co' scripts/verify_task_body.py` → 3 hits (lines 5933, 6180–6181: the URL-bearing gates) and 25 hits for `raw_completions|analysis_tensors` tokens, none implementing a footer-prose-path pin gate; `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` shows the ADJACENT check-40 family (7a4ea1acce #1433: unpinned backtick HF-path COUNT claims, WARN; 84147653ba #1487: slashless-subpath arm) — related but not covering the footer **Repro:** artifact-path shape at FAIL severity (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

(synthesized from prose follow-up) In the check-40/42 family: when scanning the `**Repro:**` footer block, extract backtick paths matching `issue\d+_\w+/.*(raw_completions|analysis_tensors)` (and the general HF-prefix shape); for each, require an adjacent (same bullet) `huggingface.co/...(tree|resolve)/<rev>` link; missing → FAIL (footer context) rather than the count-claim WARN. Reuse the existing check-40 extraction + resolve helpers.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan. Tests: `tests/test_verify_task_body.py` gains fixtures for the #1335 footer shape (unpinned → FAIL; pinned → PASS; grandfathered v3 bodies unaffected).

## Constraints / invariants

- Workflow-surface only. Forward-only: v3/v2/legacy bodies never newly hard-FAILed (v4-sentinel bodies only, per SPEC forward-only rule).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SPEC.md § footer updated in the same round if check semantics change.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: a7b285c8ff27

Verbatim surfaced prose (clean-result-critic return, #1335 fold round): "Follow-ups (orchestrator should consider): `scripts/verify_task_body.py` — extend the footer HF-path checks (the check-42 family) so a bare backtick HF-style artifact path in the `**Repro:**` footer (`issue<N>_.../{raw_completions,analysis_tensors}`) with no adjacent pinned `huggingface.co/...(tree|resolve)/<rev>` link FAILs; today only URL-bearing claims are gated, so an unpinned prose path (this round's shape, likely to recur on interpretation-pass folds) escapes mechanically."
