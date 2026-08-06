---
title: Detached HF transfers must have timeout + observable progress (xet read-hang
  is silent); warn off snapshot_download(allow_patterns) on the 1M-file data repo
kind: infra
tags: []
created_at: '2026-08-06T15:50:49Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'workflow-fix candidate from #1739: two detached HF jobs (store staging
  32min, quarantine relocation 57min) both froze on hf-xet download read-hangs with
  0-byte logs and near-zero CPU, indistinguishable from healthy work; fixed by disabling
  xet + replacing snapshot_download(allow_patterns) with scoped list_repo_tree + per-file
  download'
workflow: v1
---
## Goal

Make an hf-xet download read-hang DETECTABLE rather than silent, and stop `snapshot_download(allow_patterns=...)` being reached for on the large data repo, where it fetches a ~1M-file whole-repo manifest before filtering.

## The gap

`.claude/rules/upload-policy.md` already documents an xet wedge ladder and the `HF_HUB_DISABLE_XET=1` override, and CLAUDE.md notes xet is ON by default (`HF_XET_HIGH_PERFORMANCE=1` + `HF_HUB_ENABLE_HF_TRANSFER=1`, set shell-level in bootstrap and via `orchestrate/env.py` setdefault).

What is missing is not the FIX but the DETECTION, and a warning on the enumeration path:

1. Nothing requires a detached HF download to carry a timeout or emit observable progress. An xet read-hang produces no error, no output, and no exit — it is indistinguishable from healthy work from the outside, indefinitely.
2. Nothing warns that `snapshot_download(allow_patterns=...)` filters AFTER fetching the manifest for the ENTIRE repo. On `superkaiba1/explore-persona-space-data` (order 1M files, the #833 slow-listing class) that read itself looks exactly like a hang.

## Evidence (#1739, 2026-08-06) — two jobs, ~90 minutes lost, both silent

Both were detached, both handed across an agent boundary as "still running", both actually frozen:

- **Capture-store staging** — 32 minutes, target directory completely empty (not even the `.cache/huggingface` subdir), 0-byte log, ~6s CPU, single-threaded asleep on one socket. Additionally ran TWO concurrent `snapshot_download` processes into the SAME `local_dir`.
- **HF quarantine relocation** — 57 minutes, destination prefix still 404, 24s CPU, 0-byte log, same single-socket wait state, on a source totalling only 0.16 GB.

The signature in both cases: near-zero CPU, zero-byte log, exactly one open socket, no progress, no error. A control `list_repo_tree` from another process returned in seconds throughout, so the Hub was healthy the whole time.

Diagnosis cost was real: the wait state was initially misread as slow metadata resolution rather than a frozen socket, because nothing distinguished the two from outside.

## What fixed it

- Disabling xet on the transfer.
- Replacing `snapshot_download(allow_patterns=...)` with a scoped `list_repo_tree` probe on the exact prefix (~6 s for the file list) followed by PER-FILE download from that explicit list — no whole-repo manifest, bounded, progress every 100 files.

End state verified independently: the polluted prefix went 23,288 -> 22,286 files (exactly the 8 expected shard tags, zero strays) with 1,002 relocated under a verify-before-delete gate.

## Deliverables

1. A rule requiring any detached/background HF transfer to carry (a) a timeout, (b) periodic progress output with an explicit flush, and (c) a completion signal keyed on process exit with a captured rc — never on file existence or a non-empty log. A 0-byte log plus an empty target must not be a reachable "looks healthy" state.
2. A documented warning against `snapshot_download(allow_patterns=...)` on the large data repo, with the scoped `list_repo_tree` + per-file-download recipe as the sanctioned pattern. Reference implementation: the #1739 quarantine relocation.
3. Consider a shared helper so callers get timeout + progress + non-xet fallback by construction rather than per-call discipline.
4. Consider whether xet should stay ON by default for DOWNLOADS given two hangs in one day, or whether the default belongs on the upload path only. This is a judgement call for the rule owner, not a foregone conclusion — xet's throughput benefit on uploads is real and documented.

## Out of scope

The one-worker-per-`local_dir` rule (two concurrent `snapshot_download` calls into one directory) is a distinct concurrency hazard; worth a line in the same rule but it was not the cause of either hang.
