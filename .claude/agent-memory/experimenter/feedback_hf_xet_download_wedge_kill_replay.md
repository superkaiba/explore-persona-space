---
name: hf-xet download wedge — zero-conn native hang; kill + replay with HF_XET_DISABLE=1
description: Bulk HF downloads via the xet path can hang forever inside native xet_get with no exception and no open TCP connection; diagnose du-frozen + ss-empty + py-spy, recover by kill + replay with HF_XET_DISABLE=1 inline
type: feedback
---

The hf-xet DOWNLOAD path can wedge inside the native `xet_get` call
(`huggingface_hub/file_download.py` → hf-xet Rust client) with ZERO
established TCP connections and no exception. A per-file retry wrapper
never fires — the native call never returns — and TCP-kill unwedging is
impossible (no Python-visible socket). Observed on a GCE 87 GB
turnstore prefetch at 98.6% staged (#1345 assistant-named-story round,
2026-07-16).

**Why:** the wedge is internal to the xet client; only killing the
process clears it. **How to apply:** diagnose with (1) `du -sb` frozen
across 2+ probes ~10 min apart, (2) `ss -tnp` EMPTY for the pid, (3)
`py-spy dump` showing a worker parked in `xet_get`
(`uv tool install py-spy` works on the instance). Then kill + replay
the phase with `HF_XET_DISABLE=1` threaded INLINE on the workload
command (the real switch — `HF_XET_HIGH_PERFORMANCE=0` does not fully
bypass xet; GCP/SLURM allowlists forward `HF_XET_DISABLE`, #1195). The
plain resolve/hf_transfer path is resumable and wedge-free. Sibling of
the UPLOAD-side wedge ladder in `.claude/rules/upload-policy.md` (#931).
