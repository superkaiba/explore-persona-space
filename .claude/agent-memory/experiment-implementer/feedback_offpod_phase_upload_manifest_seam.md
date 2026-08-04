---
name: Off-pod phase file-reads vs upload manifest
description: Every file an OFF-POD phase loads must be in the pod's upload set — an all-on-one-filesystem smoke is structurally blind to the gap; deterministic scratch gets a sha-anchored reconstruction path
type: feedback
---

Cross-machine phase seams strand runs: any file an OFF-POD phase (judge, analysis) loads must be in the pod's UPLOAD SET, and an all-on-one-filesystem smoke is structurally blind to the gap (pod scratch + VM phases shared one tree in the smoke).

**Why:** #1482's P5 judge died at VM launch loading pod-only `scratch/{split_indices.npz,row_ci.npy,prov.npy}` (~17 MB never in the P4 upload set) after the pod was terminated; recovery needed a dedicated sha-anchored reconstruction round.

**How to apply:** at design time, enumerate each off-pod phase's file reads against the upload manifest line-by-line; upload small scratch metadata unconditionally; commit sha anchors (like split_1482.json) with the run so deterministic scratch stays reconstructable; make off-pod loaders fail loud with the recovery recipe.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Off-pod phase file-reads vs upload manifest](feedback_offpod_phase_upload_manifest_seam.md) — off-pod phase reads must be in the upload set; one-filesystem smokes are blind to the seam (#1482)
