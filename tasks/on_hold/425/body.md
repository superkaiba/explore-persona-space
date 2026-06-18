---
title: 'GPU eval-equivalence verification for #7b (LoRA-only) + #13 (eval subprocess
  isolation) before merge'
kind: infra
tags: []
created_at: '2026-05-28T21:57:26Z'
has_clean_result: false
---
# GPU eval-equivalence verification for #7b (LoRA-only) + #13 (eval subprocess isolation) before merge/default-flip

Two deferred fixes were implemented + reviewed + CPU-tested but are **default-OFF and GPU-unverified**. Their opt-in paths must be confirmed to NOT change eval numbers on a real Qwen-7B run before they're merged to main / their flags are flipped on. Both live on `wf/fix-13` (which is stacked on `wf/fix-7b`, so a checkout of `wf/fix-13` contains BOTH).

Branches: `wf/fix-7b` (`f5f872f4` + guard fix `a6456933`), `wf/fix-13` (`48e191aa`). To test: provision a pod, `git checkout wf/fix-13`.

## What to verify

### #7b — `EPM_MATERIALIZE_MERGED=0` (LoRA-only) eval-equivalence
Train one real condition both ways and compare eval numbers:
- Default (`EPM_MATERIALIZE_MERGED` unset → merged 7B materialized, current behavior) vs opt-in (`=0` → LoRA adapter kept, merged-on-demand / adapter-swap).
- Compare across every eval surface: capability (ARC-C logprob + per-persona ARC/HellaSwag), alignment (Betley emergent-misalignment), generation.
- **Known deferred gaps to resolve here:** in adapter mode the implementer skipped-with-loud-log the lm-eval OOD vLLM path and the per-persona ARC/HellaSwag merged-only paths, and `capability.py` adapter mode uses `PeftModel.from_pretrained` WITHOUT `merge_and_unload` (claimed numerically equivalent up to fp arithmetic order for the argmax-logprob accuracy metric — confirm empirically). Decide whether those paths need full adapter wiring or the skip is acceptable.
- Acceptance: eval numbers match (exactly, or within a documented tolerance) → safe to merge + consider flipping default. The disk payoff is ~15-30 GB → ~0.1-0.4 GB per phase.

### #13 — `EPM_ISOLATE_EVAL=1` (subprocess) eval-identity
- Run one real eval both ways: in-process (default) vs isolated (`=1`, eval runs in a fresh subprocess via `run_isolated`).
- Confirm reported eval numbers (capability logprob/accuracy, alignment, generation) are byte-identical (or within documented tolerance) across the subprocess boundary on real weights. CPU tests only proved STRUCTURE identity, not float identity.
- Also measure the wall-time cost (the child re-loads model weights per phase — that's the point, clean GPU per phase, but quantify it) and confirm WandB run continuity across parent/child is acceptable (child writes eval JSON to phase_dir + returns the fragment; it does not log to the parent's live WandB run).
- Acceptance: numbers match → safe to merge + flip on for OOM-prone runs. This closes the vLLM-orphan-worker OOM root cause (#399/#397/#396).

## Follow-up (separate, smaller)
- **`cleanup_vllm` hardening at the vLLM↔HF boundary** (`eval/generation.py`) — the clearly-safe teardown fix for the orphan-worker class, deliberately left out of #13 (outside that workstream's owned files). Owns `generation.py`.

## Provenance
Both fixes produced by the `fix-deferred-risky-3` workflow (2026-05-28); default-OFF so merging the code is low-risk, but the opt-in paths (the actual value) are unvalidated without GPU. #15 from the same batch is already merged (no GPU needed).
