---
title: 'workflow-fix: enable hf_transfer for experiment uploads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ab36eb3ecaad
created_at: '2026-06-30T01:05:48Z'
has_clean_result: false
origin_prompt: why is the upload so slow — pv-rb 36GB upload at 2.5-6 MB/s; hf_transfer
  not installed + HF_HUB_ENABLE_HF_TRANSFER unset
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap, diagnosed while investigating why #658's persona-vectors-style-rb upload was crawling.

## Goal

Enable `hf_transfer` (the Rust multi-threaded chunked HF Hub uploader) for all experiment uploads, so large artifact uploads saturate the pod link instead of crawling on the pure-Python single-stream path.

## Workflow gap

- **Bug observed:** #658's pv-rb upload (36 GB of rollout activation `.pt` files) on pod-658 ran at **~2.5–6 MB/s** while the pod's link is **~26 Gbps (≈3 GB/s)** — i.e. uploader-bound with ~500× idle headroom, GPU at 0% the whole time (a #664-class spend leak: GPU pod held for hours on a terminal upload). Root cause confirmed on the pod: `hf_transfer` is NOT installed and `HF_HUB_ENABLE_HF_TRANSFER` is unset, so `huggingface_hub` falls back to its slow pure-Python uploader.
- **Why it is a workflow gap:** `scripts/bootstrap_pod.sh` provisions every pod's environment but never installs `hf_transfer` or exports `HF_HUB_ENABLE_HF_TRANSFER=1`; it is not a declared dependency (`pyproject.toml` / `uv.lock` have no `hf_transfer`); and no upload helper / `env.py` sets the flag in-process. So EVERY experiment upload (raw completions, activation stores, checkpoints, datasets) runs on the slow path. This is workflow-surface infrastructure, not experiment-specific code.
- **Confidence (emitter):** high (verified on the pod: `hf_transfer installed: False`, env unset; and `grep` shows it absent from bootstrap/deps/env).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/bootstrap_pod.sh` (and verify it sticks for non-login SSH shells, cf. the existing uv-PATH gap):

```
+ uv pip install hf_transfer           # or add to pyproject deps
+ export HF_HUB_ENABLE_HF_TRANSFER=1   # persist into the pod shell env (e.g. /etc/profile.d or the same rc the HF cache redirect uses)
```

Planner decides the durable placement: (a) declare `hf_transfer` in `pyproject.toml` so `uv sync` installs it on every pod + set the env flag in `bootstrap_pod.sh`; and/or (b) set `HF_HUB_ENABLE_HF_TRANSFER=1` in-process in the env bootstrap so it applies on the VM + pods uniformly. Confirm the flag is visible to the actual upload process (the same non-login-shell concern as the documented uv PATH gap), and that a graceful fallback remains if `hf_transfer` is somehow absent (HF Hub already degrades gracefully).

## Scope / surfaces

- Primary: `scripts/bootstrap_pod.sh`
- Possibly: `pyproject.toml` (declare the dep) and/or `src/explore_persona_space/orchestrate/env.py` (set the env flag in-process) — planner's call.
- Grep first: `grep -rniE 'hf_transfer|HF_HUB_ENABLE_HF_TRANSFER' scripts/ src/ pyproject.toml` to confirm no existing setting.

## Constraints / invariants

- Do not break the existing upload-policy paths (`upload_folder` bulk-commit discipline, the quota-403 overflow fallback). hf_transfer is an accelerator on the SAME `upload_folder` calls.
- Keep a graceful fallback (HF Hub uploads must still work if hf_transfer is unavailable).
- Workflow-surface only — do NOT change experiment code (the 12k-tiny-`.pt`-files packing anti-pattern in `issue658_extract_rb_personavectors.py` is experiment code, out of scope here).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: ab36eb3ecaad

<!-- workflow-fix-candidate v1 -->
target_file: scripts/bootstrap_pod.sh
bug_observed: pv-rb 36GB upload on pod-658 ran at 2.5-6 MB/s (uploader-bound, 26 Gbps link idle) because hf_transfer is not installed and HF_HUB_ENABLE_HF_TRANSFER is unset; every experiment upload crawls for hours on a held GPU pod.
why_workflow_gap: bootstrap_pod.sh never installs hf_transfer or exports HF_HUB_ENABLE_HF_TRANSFER, it is not a declared dependency, and no env/upload helper sets the flag, so all experiment uploads use the slow pure-Python single-stream path.
proposed_change: Enable hf_transfer for all experiment HF uploads: install hf_transfer in bootstrap_pod.sh and export HF_HUB_ENABLE_HF_TRANSFER=1 so uploads use the Rust multi-threaded accelerator instead of the pure-Python single-stream path.
diff_sketch: |
  + uv pip install hf_transfer
  + export HF_HUB_ENABLE_HF_TRANSFER=1
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->
