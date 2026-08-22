---
title: 'URGENT fleet-wide: HF data repo at the 1M-file ceiling (999,867/1,000,000)
  — headroom probe + overflow policy + user cleanup decision list'
kind: infra
tags:
- workflow-fix
- urgent
created_at: '2026-08-14T05:53:02Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2225 fu1 F2d capture upload rejection 2026-08-14: HF
  commit endpoint refused push (1,000,107 > 1,000,000 files) on superkaiba1/explore-persona-space-data;
  ~133 files headroom remain fleet-wide.'
workflow: v1
---
# URGENT fleet-wide: HF data repo at the 1,000,000-file ceiling — establish file-count headroom + overflow policy for the data repo

## Provenance

workflow_fix_target: .claude/rules/upload-policy.md
Surfaced 2026-08-14 by task #2225's fu1 round: the F2d capture upload (240 files) was rejected by the Hub commit endpoint — "Your git repo would contain 1000107 files after this push, over the limit of 1000000 files" on `superkaiba1/explore-persona-space-data`. Current count ≈ 999,867; remaining headroom ≈ 133 files FLEET-WIDE. Auto-filed by the #2225 follow-up-round orchestrator.

## Why this is urgent

Every running and future experiment uploads raw completions / analysis tensors / eval artifacts to this repo (upload-policy.md table). With ~133 files of headroom, any phase that adds more than a handful of net-new files now CRASHES at its upload leg — the #2225 fu1 round is the first casualty (F2d capture), and every concurrent issue's next multi-file push is exposed. The #1108 overflow pattern exists for the MODEL repo's file-count limit; the DATA repo has no standing file-count policy.

## Scope of this task

1. **Measure + report:** per-top-level-prefix file counts on the data repo (scoped `list_repo_tree` sweeps; the repo has ~1M files so counts must be prefix-scoped) — identify the dominant file-count consumers (likely per-cell/per-question JSON fan-outs and `_smoke` prefixes from many issues).
2. **Cleanup candidates (PROPOSE ONLY — HF deletions are USER-ONLY per upload-policy):** enumerate candidate classes with per-class counts and a safety argument (e.g. `*_smoke` prefixes, superseded staging duplicates, prefixes whose issues are archived not-useful). Surface as a decision list for Thomas; never auto-delete.
3. **Standing policy:** extend `.claude/rules/upload-policy.md` with a DATA-repo file-count clause: (a) a preflight/upload-time headroom probe (repo file count vs ceiling, analogous to `check_hf_storage_headroom()`), (b) overflow routing for file-count (not just LFS-quota) rejections to `superkaiba1/explore-persona-space-overflow` under the same prefix layout + OVERFLOW_POINTER breadcrumb + results-sentinel deviation record, (c) guidance to PACK high-count uploads (per-cell dirs → per-cell single files / sharded bundles with manifests) for new experiment code.
4. **Mechanical arm:** wire the file-count rejection (BadRequestError "too many files") into `orchestrate/hub.py`'s `_retry_upload`/`_upload` path as a typed, non-retryable error with a clear message pointing at the overflow route (today it surfaces as a generic BadRequestError after retries).

## Acceptance

- Per-prefix count report + user-facing cleanup decision list posted on this task.
- upload-policy.md clause landed; hub.py typed error + overflow guidance landed with tests.
- No HF deletion performed by automation.
