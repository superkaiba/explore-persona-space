---
name: EXDEV tempdir Hub staging
description: A bare tempfile.TemporaryDirectory() (/tmp, container disk) + os.replace into /workspace (network volume) crashes EXDEV on pods — stage inside the destination dir
type: feedback
---

Staging a Hub download via a bare `tempfile.TemporaryDirectory()` and then
`os.replace(got, dest)` crashes `OSError: [Errno 18] Invalid cross-device
link` on pods: the default tempdir lives on `/tmp` (container disk) while the
repo tree lives on `/workspace` (MooseFS/network volume), and `os.replace`
cannot cross filesystems. It "works" locally (same device) and fails only on
the pod — a silent latent crash for any `hf_hub_download(local_dir=td)` +
move pattern.

**Fix:** put the staging tempdir INSIDE the destination dir —
`tempfile.TemporaryDirectory(dir=dest_dir, prefix=".hfstage_")` — so
`os.replace` stays same-filesystem (atomic) AND non-recursive resume globs
cannot see the half-downloaded nested tree. Sibling precedent:
`issue1335_extract_store.py` / `issue1335_gen.py` already used this;
`issue1335_fit.py::ensure_store_local` was the odd one out and crashed the
#1335 matched-n phase (attempt 7, 2026-07-16).

**How to apply:** whenever writing a download-then-move staging path (the
`feedback_hf_local_dir_staging_for_delete_to_free.md` local_dir + os.replace
recipe), always pass `dir=<dest parent>` to the tempdir; never rely on the
/tmp default. `shutil.move` is the weaker alternative (copy+delete on EXDEV,
non-atomic).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [EXDEV tempdir Hub staging](feedback_exdev_tempdir_hub_staging.md) — bare TemporaryDirectory() (/tmp) + os.replace onto /workspace crashes EXDEV on pods; stage inside the dest dir (dir=dest, prefix=".hfstage_") (#1335 r9)
