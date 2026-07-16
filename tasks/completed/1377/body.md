---
title: 'workflow-fix: vm_disk_guard trims stale home HF-hub revisions (age-gated tier)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:697a8d0209ef
created_at: '2026-07-16T04:57:22Z'
has_clean_result: false
origin_prompt: 'orchestrator observation during user disk-cleanup chat 2026-07-15:
  root at 98%; ~/.cache/huggingface/hub held 295 revisions of the data repo (~76G,
  37.6G freed manually); vm_disk_guard tier (d) covers only /workspace HF cache'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own observation during a user-requested VM disk cleanup (2026-07-15, root disk at 98%).

## Goal

`vm_disk_guard.py` gains a reclaim tier that trims stale, unref'd, non-latest HF hub REVISIONS in the user home HF cache (`~/.cache/huggingface/hub`), age-gated, mirroring the existing `/workspace` hub-cache tier (d) — the dominant unswept reclaim pool on the VM root disk.

## Workflow gap

- **Bug observed:** root disk hit 98% (12G free) while `~/.cache/huggingface/hub` held 295 accumulated revisions of `superkaiba1/explore-persona-space-data` (~76G on disk; 37.6G freed immediately by deleting unref'd revisions >7d old via `scan_cache_dir().delete_revisions()`). The guard's report run reclaimed only ~11.5G (uv tier) and pushed "manual triage needed".
- **Why it is a workflow gap:** the guard's tier (d) sweeps ONLY `/workspace/.cache/huggingface` (age-gated whole-repo reap); the home hub cache — where every VM-side `huggingface_hub` download lands by default — has NO tier, and the project data repo accretes a new cached revision per download (each upload commit creates a new sha), so it grows unboundedly and invisibly. Whole-repo age-gating also cannot help here: the repo is touched daily, but 294 of its 295 revisions are stale.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'huggingface' scripts/vm_disk_guard.py` → 3 hits, all `/workspace`-scoped (L30 docstring, L648 `DEFAULT_WORKSPACE_HF_CACHE`, L676 `scan_cache_dir` import); `grep -c '\.cache/huggingface' scripts/vm_disk_guard.py` → 2, both `/workspace` (2026-07-15). Absence-of-guard claim — the 0 home-cache hits ARE the evidence.

## Proposed change (candidate diff sketch — refine in planning)

```
+ DEFAULT_HOME_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")  # env EPS_VM_HOME_HF_CACHE
+ def _reap_stale_home_hf_revisions(min_age_days=7, apply=False):
+     # scan_cache_dir(DEFAULT_HOME_HF_CACHE); for each repo keep the newest
+     # revision + every ref'd revision; delete_revisions() for unref'd
+     # revisions with last_modified older than min_age_days.
+     # Report-only unless apply; per-repo freed bytes into the tier detail.
+ # register as a new tier (e) in run_guard's / reclaim chain, boot pass or
+ # threshold-triggered like tier (d); env EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS.
```

Reference implementation (ran successfully 2026-07-15, freed 37.6G): keep newest + ref'd revisions, `info.delete_revisions(*stale).execute()`.

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'workspace/.cache/huggingface' .claude/ CLAUDE.md scripts/`) and update every hit's
  documentation (CLAUDE.md § Disk hygiene tier list, `.claude/rules/background-automation.md`) so the new tier is documented; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Never delete the newest revision or any ref'd revision (a sha-pinned reuse re-downloads on demand — data lives on HF — but the hot path stays warm).
- Report-only by default; `--apply` gates deletion (same contract as every other tier).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `CLAUDE.md` changes, it stays consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/vm_disk_guard.py
- fingerprint: 697a8d0209ef

Surfaced prose (orchestrator's own observation, user-chat disk-cleanup session 2026-07-15): "vm_disk_guard.py never sweeps ~/.cache/huggingface — the 76G / 295-revision pileup of the project data repo was invisible to it; a stale-revision trim tier (keep newest + ref'd, age-gate 7d) freed 37.6G manually."
