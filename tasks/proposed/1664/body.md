---
title: 'workflow-fix: verify_uploads honors card adapter_repo_overrides'
kind: infra
tags:
- wf-fix
- wf-fix-fp:90befae13af6
created_at: '2026-07-24T10:45:44Z'
has_clean_result: false
origin_prompt: 'upload-verifier script-gap flag from #1586 fu verification r1: hf_model
  false-MISSING on override-repo LoRA cells; check_hf_model_from_card never reads
  adapter_repo_overrides'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an upload-verifier script-gap flag raised on task #1586 (emitting agent: upload-verifier, fu round verification r1).

## Goal

Make verify_uploads.py's card-driven HF-model check honor the reproducibility_card's adapter_repo_overrides field when resolving per-cell adapter paths against a non-default model repo.

## Workflow gap

- **Bug observed:** #1586 fu upload-verification r1: the mechanical verifier reported hf_model MISSING for 2 impolite LoRA cells that were fully uploaded (1,318 files / 73 rung dirs each) to the MAIN model repo, because the results card declared the repo via adapter_repo_overrides — a field check_hf_model_from_card never reads; the agent had to override the script row manually.
- **Why it is a workflow gap:** the card schema (produced by issue dispatchers) carries per-cell repo overrides, but the verifier resolves every adapter_path against the single hf_model_repo default — any run splitting artifacts across overflow + main repos false-FAILs the mechanical pass and burns an agent-override every round.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'adapter_repo_overrides' scripts/verify_uploads.py` → 0 hits (absence claim — the field is unread; 0-hit in-target IS the evidence; the field name comes verbatim from the #1586 epm:results card which carries it) + `git log --oneline --since='7 days ago' -- scripts/verify_uploads.py` → e4d5e3b7af workflow-fix #1524: merged card — prose pointer must not shadow a structural declaration (#1296);— none touch card repo-override resolution (2026-07-24)

## Proposed change (candidate diff sketch — refine in planning)

+ In check_hf_model_from_card (scripts/verify_uploads.py): read card.get("adapter_repo_overrides") (per-cell dict cell -> repo_id); when a cell has an override, existence-check its adapter path against THAT repo instead of hf_model_repo; report the resolved repo per row.
+ Pin test in the verify_uploads test file: a card with one default-repo cell + one override-repo cell resolves both (and a missing override-repo path still FAILs).

## Scope / surfaces

- Primary target: `scripts/verify_uploads.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'adapter_repo_overrides' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every consumer/producer doc mismatch; list them in the plan.

## Constraints / invariants

- Workflow-surface only; workflow_lint no-flags passes; ruff clean; existing verifier behavior for cards WITHOUT the field byte-identical.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_uploads.py
- fingerprint: 90befae13af6

Origin: upload-verifier fu-r1 marker (epm:upload-verification v5 on #1586): "The mechanical verifier's hf_model: MISSING on the 2 LoRA paths is a script false-negative — verify_uploads.py doesn't read the card's adapter_repo_overrides field (noted in the marker as a script gap for the orchestrator to route)."
