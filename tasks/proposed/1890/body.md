---
title: 'workflow-fix: GCE crash-persist omits analysis_tensors staging dirs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1533eec6dfde
created_at: '2026-07-30T19:02:47Z'
has_clean_result: false
origin_prompt: 'nlmap-runner report 2026-07-30: banked 412MB map fit lost on rc=7
  stop — GCE crash-persist set omits analysis_tensors/; rescue upload blocked by GCE-side
  credential resolution'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose-surfaced candidate raised on task #1739 (emitting agent: nlmap-runner inline subagent, report 2026-07-30 ~19:0xZ; loss incident on instance eps-issue-1739-nlmap, attempt att-20260730-173427-nlmap).

## Goal

Add the local analysis-tensors staging dirs (map payloads, `analysis_tensors/`-destined artifacts) to the GCE crash-persist upload set, and fix the GCE-side credential resolution that blocked the round's rescue upload.

## Workflow gap

- **Bug observed:** the GCE EXIT-trap `_eps_persist_diagnostics` upload set covers `eval_results/issue_<N>`, `data/issue_<N>`, `data/issue<N>`, logs — but NOT the local staging dir for `analysis_tensors/` payloads; a banked 412 MB MLP map fit (0.68 GPU-h, produced 18:44:03Z) died with the auto-DELETEd boot disk when the workload exited rc=7 (a DESIGNED stop). A manual rescue upload was additionally blocked by missing GCE-side credential resolution. unverified hypothesis — verify at plan time: the exact credential failure mode (reported by the emitting agent; not independently reproduced at filing).
- **Why it is a workflow gap:** the upload-policy plan §10 class "intermediate analysis tensors referenced as downstream inputs" is exactly what crash-persist exists to save (gcp.py's own docstring at line 1045 names `analysis_tensors/`), yet the persist set at lines ~2269-2271 omits it — so any GCE run staging map/tensor payloads locally loses them on ANY non-zero exit, including designed aborts.
- **Confidence (emitter):** high (loss realized this round)
- verified-at-filing: `grep -n "analysis_tensors\|eval_results_issue\|data_issue" src/explore_persona_space/backends/gcp.py` → docstring hit line 1045 names analysis_tensors; persist-set literals lines 2269-2271 carry ONLY eval_results/data dirs, 0 analysis_tensors entries (2026-07-30). Per-target: construction site confirmed in backends/gcp.py.

## Proposed change (candidate diff sketch — refine in planning)

+ In the _eps_persist_diagnostics dir list (gcp.py ~L2269):
+   (root / "analysis_tensors" / f"issue_{issue}", f"analysis_tensors_issue_{issue}"),
+   plus whatever local staging convention the round used (survey call sites).
+ And resolve/document GCE-side HF credential availability for rescue uploads
+ (the EXIT-trap path has a token; the interactive rescue path did not).

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep for sibling persist sets before editing (`grep -rn "eval_results_issue" src/explore_persona_space/backends/`) and keep lanes uniform.

## Constraints / invariants

- Workflow-surface only; ruff + workflow_lint pass; 300s-bounded persist contract unchanged (#854).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 1533eec6dfde

Surfaced prose (verbatim): "the 412 MB banked map died with the boot disk because the GCE crash-persist set omits `analysis_tensors/` (0.68 GPU-h, deterministically regenerable); that's the workflow-fix candidate in my report, targeting `backends/gcp.py` plus the missing GCE-side credential resolution that also blocked my rescue upload."
