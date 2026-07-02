---
title: 'workflow-fix: smoke outputs must not clobber committed artifacts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:316546eddfbf
created_at: '2026-07-02T07:01:47Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on #722 (2026-07-02): a --layers 0 18 review-window
  smoke overwrote committed 28-layer production eval_results/figures artifacts at
  canonical paths with 2-layer smoke output; smoke-run contract (experiment-implementer.md
  / code-reviewer.md Step 0.6) lacks an output-path hygiene rule — require scratch-dir
  redirect or restore-after-smoke.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #722 (emitting agent: orchestrator, /issue 722 resume session).

## Goal

Add an output-path hygiene rule to the smoke-run contract: smoke invocations that write under `eval_results/` or `figures/` must redirect output to a scratch dir (e.g. `--out-dir /tmp/...` / env override) OR restore the committed artifacts immediately after the smoke, so implementer/reviewer smokes never leave committed production artifacts clobbered in the worktree.

## Workflow gap

- **Bug observed:** During #722's round-2 review window (04:22–05:04Z, 2026-07-02), a `--layers 0 18` smoke wrote 2-layer outputs to the canonical `eval_results/issue_722/base-skill-over-mean-cC-to-v0/{krr_vs_linear,mlp_epoch_curves,mlp_width_sweep}.json` and `figures/issue_722/{krr_vs_linear_nonlinearity,mlp_epoch_curves,mlp_width_sweep}.*` paths, truncating the committed 28-layer production artifacts (−2966 lines) in the issue-722 worktree. Caught pre-commit by the resume session and restored via `git checkout -- <paths>`; had it been swept into a later explicit-path commit, production artifacts on the branch would have been silently replaced by smoke output.
- **Why it is a workflow gap:** The smoke-run contract (`experiment-implementer.md` § End-to-end smoke run PER PHASE; `code-reviewer.md` Step 0.6) mandates tiny-slice smokes but says nothing about WHERE their outputs land. Experiment scripts default to canonical output paths, so any conforming smoke on an already-produced arm clobbers committed artifacts by default. No grep hit for scratch-dir/clobber guidance in either file.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ (experiment-implementer.md § End-to-end smoke run PER PHASE, new bullet)
+ - **Smoke outputs never overwrite committed artifacts.** When the smoke
+   command writes under eval_results/ or figures/, redirect its output to a
+   scratch dir (--out-dir /tmp/issue-<N>-smoke/ or the script's env override);
+   if the script has no output override, restore the touched committed paths
+   immediately after the smoke (git -C "$WT" checkout -- <paths>) and say so
+   in the ## Smoke run sub-section. A dirty worktree of smoke-truncated
+   production JSONs is a latent clobber (incident #722, 2026-07-02).
+ (code-reviewer.md Step 0.6, one line)
+ - Verification smokes the reviewer runs itself follow the same rule.
```

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md, .claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'Smoke run' .claude/ CLAUDE.md scripts/`) and update every hit that
  states the smoke contract; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md, .claude/agents/code-reviewer.md
- fingerprint: 316546eddfbf

Surfaced prose (orchestrator observation, task #722 resume session, 2026-07-02): review-window smoke overwrote committed 28-layer production artifacts at canonical paths with 2-layer smoke output; smoke-run contract lacks any output-path hygiene rule; propose scratch-dir redirect or restore-after-smoke requirement in experiment-implementer.md + code-reviewer.md.
