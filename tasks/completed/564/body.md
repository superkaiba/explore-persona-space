---
title: Proactive HF public-storage headroom check (preflight WARN + pre-train persist
  gate + opt-in overflow routing)
kind: infra
tags: []
created_at: '2026-06-10T19:24:15Z'
has_clean_result: false
---
## Goal

Add a proactive HuggingFace public-storage headroom check so experiments warn (or park uploads to the private overflow repo) at a configurable soft ceiling instead of failing mid-run with the account-wide LFS 403 at 100% quota.

## Context

Incident 2026-06-10 (#541, #552): the account hit the public-storage quota (11.3 TB across 414 repos) and EVERY in-flight task's LFS upload started 403ing at once; 3-4 tasks parked blocked on a user quota decision. The reactive recovery rule exists (`.claude/rules/upload-policy.md` § HF storage-quota 403) but nothing warns before the wall.

## Acceptance criteria

1. A `check_hf_storage_headroom()` helper (probably in the upload utils near `upload_raw_completions_to_data_repo()` / the adapter-persist path) that sums account public `usedStorage` via `list_models(author=...)` / `list_datasets(author=...)` with `expand=["usedStorage"]`, cached for ~1h to avoid hammering the API.
2. Preflight (`orchestrate/preflight.py`) gains a non-fatal WARN when usage exceeds a soft ceiling (env `EPM_HF_STORAGE_SOFT_CEILING_TB`, sensible default just below the observed quota), printing current TB + ceiling.
3. The fail-loud adapter-persist path (`EPM_PERSIST_ADAPTER_HF_REPO`) checks headroom BEFORE training starts (at config validation), not only at upload time — so a doomed sweep fails in minute 1, not hour 10.
4. Above the soft ceiling, large-LFS uploads optionally auto-route to the private overflow repo `superkaiba1/explore-persona-space-overflow` per the existing recovery rule (env-gated opt-in, default off, plan-deviation note posted when it fires).
5. Unit tests for the threshold/caching logic with mocked HF API.

## Notes

- This is `src/` + preflight infra code → `implementer` + `code-reviewer` path, kind: infra, no GPU.
- Quota-freeing/deletion stays user-only; this task is detection + routing only.
