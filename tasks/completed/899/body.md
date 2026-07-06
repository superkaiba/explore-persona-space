---
title: 'workflow-fix: sentinel drain posts results-shaped payload missing schema envelope'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9dc384894979
created_at: '2026-07-02T22:55:16Z'
has_clean_result: false
origin_prompt: 'prose follow-up raised on #825: poller silently skipped a complete
  results sentinel lacking envelope keys while the GCP VM was self-deleting'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on
task #825 (emitting agent: orchestrator).

## Goal

Stop the sentinel drain from silently skipping a complete results payload that lacks the schema-envelope keys.

## Workflow gap

- **Bug observed:** task #825 run-7's `/workspace/logs/issue-825-results.json` carried the COMPLETE `epm:results v1` payload (all 10 required keys per SKILL.md Step 7) but no `sentinel_schema_version`/`kind`/`version` envelope; `poll_pipeline` skipped it with only a WARNING ("missing required keys; skipping") and posted nothing — the orchestrator had to SSH the payload off the VM and compose `epm:results` by hand minutes before the GCP EXIT-trap deleted the instance.
- **Why it is a workflow gap:** the drain requires the envelope even when the sentinel is unambiguously the results sentinel (filename `issue-<N>-results.json` + the full Step 7 key set); a silent skip loses the terminal payload at exactly the moment the ephemeral VM self-deletes, converting a recoverable contract nit into potential data loss.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ # poll_pipeline sentinel drain (and the gcp.py drain wrapper):
+ # when a sentinel named issue-<N>-results.json fails the envelope check but
+ # its body validates against the Step 7 results-payload key set
+ # (eval_numbers, eval_paths, reproducibility_card, wandb_url, hf_hub_url,
+ #  worktree_path, final_commit_sha, gpu_hours_used, gpu_hours_budgeted,
+ #  plan_deviations), wrap it as kind=epm:results version=1 and post it,
+ # with a LOUD log naming the missing envelope (never a silent skip).
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Also check `src/explore_persona_space/backends/gcp.py` (the "gcp sentinel drain: N matched but 0 processed" path) — grep `grep -rn "missing required keys\|sentinels_processed" scripts/ src/explore_persona_space/backends/` and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Envelope-carrying sentinels keep the existing strict path; the fallback fires ONLY for the exact results-payload key set on a `*-results.json` filename.
- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 9dc384894979

Surfaced prose (verbatim): "the poller skipped the results sentinel: it lacks the schema envelope keys (sentinel_schema_version, kind, version) — the dispatch script wrote a raw payload... the orchestrator composed epm:results manually per the Step 7 fallback; the drain should recognize a results-shaped payload on the results filename rather than silently skipping it while the VM self-deletes."
