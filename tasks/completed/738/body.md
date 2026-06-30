---
title: 'workflow-fix: log per-epoch curves for hand-rolled predictor fits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:029fa156dda8
created_at: '2026-06-29T23:13:46Z'
has_clean_result: false
origin_prompt: 'Isn''t there a lesson somewhere to save per epoch results: [#658 A3.2/A3.5
  MLP fixed-300-epoch LOCO loop recorded nothing per-epoch]'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap raised while reviewing task #658 (user noticed the missing lesson).

## Goal

Extend the "training metrics cannot be reconstructed post-hoc" discipline so it also covers hand-rolled torch predictor/probe fit loops in analysis scripts — they must emit a per-epoch held-out metric / training curve (or early-stop on inner-val instead of running a fixed epoch count blind).

## Workflow gap

- **Bug observed:** #658's A3.2/A3.5 MLP predictor fits (`_fit_mlp_loco` in `scripts/issue658_fit_predictors.py`) run a fixed 300-epoch LOCO loop per fold that records nothing per-epoch (only the final held-out prediction is returned). So whether the MLP over- or under-trained at 300 epochs cannot be diagnosed post-hoc, and "the A3.2 MLP failure is a mis-training artifact, not a genuine mean-pool insufficiency" cannot be ruled out from stored artifacts.
- **Why it is a workflow gap:** `.claude/rules/code-style.md` already states the load-bearing principle — "WandB live training metrics are mandatory … loss curves, grad-norm history, and callback metrics **cannot be reconstructed post-hoc**" — but it is scoped ONLY to HF-Trainer/TRL config builders under `src/explore_persona_space/experiments/` (`TrainLoraConfig` / `SFTConfig` / `TrainingArguments`), and the `workflow_lint.py --check-wandb-required` lint only greps for those builders. Hand-rolled torch optimization loops (predictor MLPs, linear-probe SGD fits, the per-cell fits in analysis scripts) hit the EXACT failure mode the rule's rationale names, but fall outside its scope. The rule's own justification applies verbatim; only its coverage is too narrow.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Add a bullet to `.claude/rules/code-style.md` (near the existing "WandB live training metrics are mandatory" bullet) of the form:

```
- **Hand-rolled fit loops must record a training/held-out curve too.** Any
  torch optimization loop fit blind for a fixed epoch/step count (predictor
  MLPs, linear-probe SGD, per-cell analysis fits — e.g. `_fit_mlp_loco`)
  MUST either (a) early-stop on an inner-validation metric, or (b) emit a
  per-epoch train + held-out metric curve to a durable artifact. A fixed
  epoch count with nothing logged makes "did this over/under-train?"
  permanently unanswerable — the same post-hoc-irreconstructible failure the
  WandB-training-metrics rule already forbids for HF-Trainer runs.
```

Consider whether `scripts/workflow_lint.py` can mechanically flag the
`for _ in range(N_EPOCHS): ... .backward()` fixed-loop-with-no-logging shape
(likely too broad to enforce cleanly; the planner decides — a rule-only fix
with no lint is acceptable if a reliable check is not feasible).

## Scope / surfaces

- Primary target: `.claude/rules/code-style.md`
- Possibly: `scripts/workflow_lint.py` (only if a reliable mechanical check exists)
- Grep the workflow surface before editing (`grep -rniE 'per.?epoch|loss curve|reconstruct.*post.?hoc' .claude/ CLAUDE.md scripts/`) and keep CLAUDE.md's "log all metrics to WandB" line consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code (`scripts/issue658_fit_predictors.py` itself is out of scope; the fix is the RULE, not that script).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; CLAUDE.md ↔ rule file stay consistent.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/code-style.md
- fingerprint: 029fa156dda8

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/code-style.md
bug_observed: #658's A3.2/A3.5 MLP predictor fits run a fixed 300-epoch LOCO loop recording nothing per-epoch, so MLP over/under-training cannot be diagnosed post-hoc and that alternative explanation for the A3.2 failure cannot be ruled out.
why_workflow_gap: code-style.md's "metrics cannot be reconstructed post-hoc" rule + the --check-wandb-required lint are scoped only to HF-Trainer/TRL config builders under src/.../experiments/, so hand-rolled torch predictor/probe fit loops that hit the identical failure mode are uncovered.
proposed_change: Extend the training-metrics-cannot-be-reconstructed-post-hoc rule to cover hand-rolled torch predictor/probe fit loops in analysis scripts: emit a per-epoch held-out metric / training curve, or early-stop on inner-val instead of a fixed epoch count.
diff_sketch: |
  + - **Hand-rolled fit loops must record a training/held-out curve too.** Any
  +   torch optimization loop fit blind for a fixed epoch/step count MUST either
  +   early-stop on inner-val OR emit a per-epoch train+held-out metric curve.
confidence: medium
related_task: #658
<!-- /workflow-fix-candidate -->
