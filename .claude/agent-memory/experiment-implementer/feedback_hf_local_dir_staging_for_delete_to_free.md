---
name: hf local_dir staging for delete-to-free + content-derived mtimes for stat-fingerprints
description: Disk-bounded layer-sliced HF staging must download via hf_hub_download(local_dir=...) — symlink-to-cache staging frees nothing on unlink; and stat-based resume fingerprints need content-derived mtimes; the scratch MUST live on the target's filesystem (a bare /tmp tempdir makes os.replace cross-device -> EXDEV on pods)
type: feedback
---

Two traps for disk-bounded, stage-then-delete HF staging loops (#1092 P6 wrapper):

1. **Symlink-to-cache staging does NOT free disk on delete.** The in-repo
   `download_scoped` helper (`scripts/issue1092_bridge_refit.py`) symlinks staged
   paths to the central HF cache — deleting the staged tree leaves the blobs in the
   cache, so a "stage layer, fit, delete, next layer" loop's peak disk grows
   unbounded. For delete-to-free staging, download with
   `hf_hub_download(..., local_dir=<scratch under target.parent>)` (direct-to-dir,
   no central-cache copy since huggingface_hub >=0.23; repo pins 0.36.2) then
   `os.replace` to the target; `TemporaryDirectory` cleanup removes the
   `local_dir/.cache/huggingface` metadata.
   The `<scratch under target.parent>` placement is LOAD-BEARING, not a
   convenience: a bare `tempfile.TemporaryDirectory()` (/tmp, container disk)
   makes the `os.replace` cross-device and crashes `OSError: [Errno 18]`
   (EXDEV) on pods — see `feedback_exdev_tempdir_hub_staging.md` (#1335 r9).

2. **Stat-based fingerprints break across staging cycles unless mtime is pinned.**
   A resume predicate hashing `(name, size, st_mtime_ns)` per input
   (`issue1092_fit_grid._fingerprint`) sees a NEW mtime on every re-staging →
   every checkpoint silently invalidates on resume. Fix at the stager: after
   sha256-verifying the download, `os.utime(path, ns=(t, t))` with
   `t = int(sha256_hex[:15], 16)` — content-derived, so identical bytes reproduce
   the identical fingerprint and the predicate becomes content-true.

**How to apply:** any wrapper that stages HF shards, runs a consumer with its own
stat-keyed checkpoint resume, then deletes the slice (the #1092 P6 pattern) needs
BOTH: local_dir download (peak disk = one slice) + content-derived mtimes
(checkpoints survive re-staging). Sha-verify against the Hub LFS sha256 when the
listing provides it (64-hex); git blob_ids are sha1 and cannot be compared to
local sha256 — record-only.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [hf local_dir staging for delete-to-free + content-mtime fingerprints](feedback_hf_local_dir_staging_for_delete_to_free.md) — symlink-to-cache staging frees NOTHING on unlink (use hf_hub_download local_dir + os.replace); stat-keyed resume fingerprints need sha-derived os.utime or every re-staging invalidates all checkpoints (#1092 P6)

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [EXDEV tempdir Hub staging](feedback_exdev_tempdir_hub_staging.md) — bare TemporaryDirectory() (/tmp) + (#1335)
