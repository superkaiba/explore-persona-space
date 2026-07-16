---
name: hf-xet download wedge — zero-conn native hang; kill + replay with HF_HUB_DISABLE_XET=1
description: Bulk HF downloads via the xet path can hang forever inside native xet_get with no exception and no open TCP connection; diagnose du-frozen + ss-empty + py-spy, recover by kill + replay with HF_HUB_DISABLE_XET=1 inline (HF_XET_DISABLE is a verified no-op alias)
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
the phase with `HF_HUB_DISABLE_XET=1` threaded INLINE on the workload
command — the only xet kill switch `huggingface_hub` reads (it gates
the xet branch at `file_download.py:1735`, so the hang class cannot
recur on the plain resolve/hf_transfer path it forces; already-landed
files skip on replay). CONFIRM the bypass took on the replay (fresh
py-spy dump: no xet frames) — hub GH #3266 reports a download-side
coverage gap on this pin, and the #1345 replay itself threaded the
NO-OP alias and still recovered (the wedge can be intermittent; a
naive kill+replay may recover by luck). The legacy `HF_XET_DISABLE`
is a VERIFIED NO-OP alias (consumed by neither `huggingface_hub` nor
the hf-xet binary; kept on the lane allowlists only as an annotated
legacy alias, #1195), and `HF_XET_HIGH_PERFORMANCE=0` de-tunes but
does not disable xet. Canonical entry: `.claude/rules/gotchas.md`
(hf-xet DOWNLOAD wedge); sibling of the UPLOAD-side wedge ladder in
`.claude/rules/upload-policy.md` (#931).
