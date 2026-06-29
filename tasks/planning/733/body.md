---
title: 'workflow-fix: verify_task_body.py — harden HF URL resolve against 429 storms'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6471feafc999
created_at: '2026-06-29T13:52:01Z'
has_clean_result: false
origin_prompt: 'Workflow-fix candidate surfaced by analyzer during /issue 715: verify_task_body.py
  check_hf_url_pins_resolve does a recursive HF Hub list_repo_tree that is fragile
  under concurrent-session 429 storms — repeated launches stranded zombie verifiers
  that self-throttled the HF API (resolve dropped from 2+ min of retries to 0.2s once
  un-throttled). Candidate hardening: non-recursive HEAD-based resolve or shorter
  retry cap.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the analyzer during /issue 715 (round 1 + round 2).

## Goal

Harden `verify_task_body.py`'s `check_hf_url_pins_resolve` against concurrent-session 429 storms on the HF Hub API. Current implementation does a recursive `list_repo_tree` for every HF link in `## Reproducibility`, which under fleet-wide 429 pressure self-throttles and strands zombie verifiers (observed during /issue 715 — verify times spiked from ~0.2s to 2+ minutes of retries; repeated launches stranded multiple Python processes blocking on the HF API).

## Workflow gap

- **Bug observed:** During /issue 715, `verify_task_body.py --issue 715` invocations spiked from ~0.2s to 2+ minutes wall-time when other concurrent EPS sessions were also resolving HF URLs (fleet-wide 429 storms; the org cap climbs at minute boundaries). Repeated launches stranded zombie verifier processes that self-throttled while waiting for `huggingface_hub.list_repo_tree` retries. The body verifies clean once HF is un-throttled.
- **Why it is a workflow gap:** Body verification is a HOT path in the orchestrator pipeline (called by analyzer R1/R2/R3/R4 + clean-result-critic R1/R2 + Step 9a-quater export + each subagent's own re-verify), so a fragile HF-resolve check amplifies 429 incidents into multi-minute pipeline stalls and zombie processes.
- **Confidence (emitter):** medium — the diagnosis is clear (recursive `list_repo_tree` is overkill for a "does this revision exist" check), the fix is bounded (HEAD-based pin verification or a shorter retry cap), but the exact API endpoint trade-offs need a planner's read.

## Proposed change (candidate diff sketch — refine in planning)

```python
# In scripts/verify_task_body.py — replace the recursive list_repo_tree
# in check_hf_url_pins_resolve with a HEAD-based revision check:

def _hf_url_resolves(url: str, timeout_s: float = 5.0, max_attempts: int = 2) -> bool:
    """Cheap revision-existence check via HEAD on huggingface_hub.HfApi.repo_info(revision=<sha>).
    Returns False on 4xx other than 429 (treat as resolvable-but-throttled). Bounded retry."""
    ...
```

Or alternative: cache the resolved URL set per session and skip re-resolution within a short TTL.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for `list_repo_tree`, `list_repo_files`, `huggingface_hub` in the verifier and any sibling scripts: `grep -rln 'list_repo_tree\|list_repo_files' .claude/ scripts/`

## Constraints / invariants

- Workflow-surface only.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` — recursion guard.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 6471feafc999

Candidate prose (verbatim from the round-1 analyzer's operational note):

> One operational note for the workflow surface: the `verify_task_body.py` HF-URL-resolve check (`check_hf_url_pins_resolve`) does a recursive HF Hub tree listing that is fragile under concurrent-session 429 storms — repeated launches stranded zombie verifiers that self-throttled the HF API. The body verifies clean once HF is un-throttled (resolve dropped from 2+ min of retries to 0.2s). Not blocking here, but a candidate hardening (e.g. a non-recursive HEAD-based resolve or a shorter retry cap) if it recurs.
