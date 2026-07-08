---
name: hf local_dir staging for delete-to-free + content-derived mtimes for stat-fingerprints
description: Disk-bounded layer-sliced HF staging must download via hf_hub_download(local_dir=...) — symlink-to-cache staging frees nothing on unlink; and stat-based resume fingerprints need content-derived mtimes
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
