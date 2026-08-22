---
title: HF file-count overflow fallback never fires for dataset repos or bulk uploads
  — data repo hit its 1,000,000-file hard cap
kind: infra
tags:
- hf-filecount-overflow
created_at: '2026-08-14T23:20:58Z'
has_clean_result: false
origin_prompt: 'Discovered during /issue 2162 turn-boundary-multipatch round: HF data
  repo push rejected at the 1,000,000-file hard cap; the #1108 reactive overflow fallback
  is gated repo_type==''model'' and lives in _upload, so dataset + bulk-upload paths
  have no fallback.'
workflow: v1
---
# HF file-count overflow fallback never fires for DATASET repos or BULK uploads — the data repo hit its 1,000,000-file hard cap and the fleet's default persistence path is down

## Goal

Make the #1108 reactive file-count overflow fallback actually cover the paths the project uses for its DEFAULT artifact destination, and handle the case where the canonical repo is so full it cannot even accept the pointer breadcrumb.

## What happened (discovered live on #2162, 2026-08-14 23:02Z)

`superkaiba1/explore-persona-space-data` reached **1,000,000 files against the Hub's hard 1,000,000-file repo limit**. Every push to it now fails, verbatim:

```
Bad request for commit endpoint:
Your push was rejected because it contains too many files. Your git repo would
contain 1000009 files after this push, over the limit of 1000000 files.
Offending reference: refs/heads/main
```

On #2162 this killed a grid worker mid-run after 4 retries (the fail-fast raise in `issue2162_run.py:2495` behaved correctly). Evidence preserved on `pod-2162-tbmp` at `/workspace/logs/incident-filecount-1M/` (all four per-worker logs, including the full traceback).

**This is a FILE-COUNT cap, not the byte/storage quota of #541/#552.** It does not free itself with time or a plan upgrade, and it rejects a 1-file push exactly as it rejects a 1,000-file push.

## Why the existing fallback did not fire — two independent structural gaps

The project already has a reactive overflow fallback for this exact rejection (#1108), and `_is_file_count_limit_error` (`hub.py:1176`) matches the message correctly. It still cannot fire here:

1. **`hub.py:1687-1690` gates on `repo_type == "model"`:**
   ```python
   if (
       _filecount_fallback_enabled()
       and _is_file_count_limit_error(e)
       and repo_type == "model"
       and repo_id != DEFAULT_OVERFLOW_REPO
   ):
   ```
   Every raw completion, training mix, and analysis tensor in this project goes to the **dataset** repo. So the fallback covers the repo class that is not the problem and skips the one that is. The docstring at `hub.py:1512` states the model-only scope explicitly — the gap is as-designed-and-never-revisited, not a typo.

2. **The fallback lives in `_upload`; bulk uploads never enter `_upload`.** `_upload_folder_filtered` (`hub.py:1735`) is the `allow_patterns`-threaded sibling that exists precisely because `_upload` cannot express a glob subset, and it is what every per-issue bulk uploader calls (`issue2094_run.py:2154`, `issue1738_*`, `issue1901_metric_battery.py:2670`, `issue1481_marker.py:500`, plus `hub.py:2216`). The note at `hub.py:1159` — "the #1108 overflow fallback re-enters `_upload`" — describes a path a bulk upload structurally never takes.

Neither `EPM_HF_OVERFLOW_ROUTING` (#564 — byte-quota-driven, keys on a STORAGE headroom signal, default-OFF) nor `EPM_HF_FILECOUNT_FALLBACK` (default-ON, gates the model-only path above) covers a dataset bulk push refused on file count.

## Scope

1. **Extend the reactive file-count fallback to `repo_type == "dataset"`.** A dataset-typed private overflow repo ALREADY EXISTS and is already in production use for this purpose — `superkaiba1/explore-persona-space-overflow` (dataset, private, 545 files), holding `issue2225_ctxsteer/analysis_tensors/...`. So the destination needs no creation; only the routing does. Preserve the #1108 property that makes the fallback strictly dominant and default-ON: it fires only AFTER the canonical push was refused, so it can never divert a would-succeed push.
2. **Wire the same fallback into `_upload_folder_filtered`** so bulk `upload_folder` commits are covered, not just single-file `_upload` calls. Factor the retry-against-overflow into one helper both call rather than duplicating the branch.
3. **Handle the unwritable-pointer case.** The policy prescribes an `OVERFLOW_POINTER.json` breadcrumb on the CANONICAL repo — which is impossible when the canonical repo is at the file cap, since the pointer is itself a new file. The pointer write must degrade EXPLICITLY (logged reason + the `#564` routing event + a registry record) instead of failing the whole upload or silently skipping. Today's code already treats a pointer-write failure as non-fatal (`hub.py:764`), but the reason must be distinguishable from a transport blip.
4. **Preflight / early-warning.** A file-count headroom probe belongs alongside the existing byte-quota checks so the fleet learns it is approaching the cap before a mid-run worker death. Note `list_repo_files` on a 1M-file repo is punishingly slow (a naive `len(list_repo_files(...))` probe timed out at 120 s during this diagnosis) — use a cheap signal, and never put a slow probe on a launch-blocking path.
5. **Update `.claude/rules/upload-policy.md`** — its "file-count-limit fallback → private overflow repo (#1108)" line currently sits only in the MODEL-repo row of the destination table, which is exactly the misreading that let this ship.

## Out of scope / needs the user

Whether to PRUNE the 1,000,000-file data repo, or permanently relocate the project's default raw-completions destination, is a cross-project irreversible storage-architecture decision and belongs to the user — not to this task. This task makes the fallback work; it does not decide where the project's artifacts should live long-term.

## Acceptance

- A dataset-repo bulk upload refused with the file-count message reroutes to the dataset overflow repo, verifies its landing, and returns a real URL — covered by a test that fakes the rejection (the message is the contract; the exception class is deliberately not trusted, per `hub.py`'s existing note).
- A single-file dataset upload does the same.
- A model-repo upload keeps byte-identical behaviour (#1108 regression guard).
- The unwritable-pointer path is exercised and logs a distinguishable reason.
- `EPM_HF_FILECOUNT_FALLBACK=0` still restores legacy behaviour on every newly-covered path.

## Provenance

Found by the #2162 orchestrator while driving the `turn-boundary-multipatch` follow-up round; the round was recovered by deferring uploads to `--upload local-mirror` and relaunching the remaining 5 grid blocks. Full incident record: the `epm:progress` note on #2162 beginning `[long-phase-heartbeat] pod=pod-2162-tbmp ... P2 grid INTERRUPTED at 40/45 blocks by a FLEET-WIDE HF blocker`.
