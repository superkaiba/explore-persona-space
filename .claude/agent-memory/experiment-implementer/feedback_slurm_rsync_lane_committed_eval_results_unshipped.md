---
name: slurm-rsync-lane-committed-eval-results-unshipped
description: On the fellows/SLURM rsync lane, "committed on the branch" is NOT node-reachable for eval_results/ (and every other RSYNC_EXCLUDE dir) — the lane rsyncs only RSYNC_INCLUDE_PATHS. Any workload phase consuming a committed parent eval_results input needs the #734 upload-first pattern: mirror the consumed set to the HF data repo + an idempotent fail-loud staging step at leg entry.
metadata:
  type: feedback
---

The fellows/SLURM lane materializes a full branch tree on the VM but ships to the
cluster only `RSYNC_INCLUDE_PATHS` (`backends/slurm.py`: pyproject/uv.lock/src/
scripts/configs/external/tests/data-sft) with `RSYNC_EXCLUDE_PATTERNS` dropping
`eval_results/`, `figures/`, `docs/`, `tasks/` wholesale. So a dispatcher leg
that opens a COMMITTED parent `eval_results/...` file (a digest CSV, a ladder
JSON, per-unit trees for a paired contrast) crashes `FileNotFoundError` on the
node even though the file is on the pushed ref — the git-clone lanes (GCP/RunPod)
do not share this gap, which is why it survives every git-reachability gate
(#1689 fellows job 15724: `cmd_fence` died on `analyzer/dvf_unit_digest.csv`;
the leg consumed FOUR such inputs).

Remedy (the #734 upload-first pattern, worked example
`scripts/issue1689_derived_vs_free.py --phase upload-parent-inputs` /
`--phase stage-parent-inputs` + the leg-entry wiring in
`scripts/issue1689_dispatch.sh`):

- Mirror the consumed set ONCE to the HF data repo (KB–MB text rides the
  non-LFS path; exact-set verify after upload).
- Add an idempotent, FAIL-LOUD staging step at leg entry that downloads each
  missing file to the EXACT relative path the consumers open (`[ -f ]` guard
  makes it a no-op on a git checkout / the VM).
- Audit `exists()`/WARN-guarded consumers too (ladder JSONs): those silently
  SKIP instead of crashing on the node — a worse failure mode than the loud
  FileNotFoundError that surfaced this.
- Plan-side: a workload consuming committed eval_results/figures/docs inputs
  declares the HF staging (live fix task for the mechanical gate: #1835), or
  pins the one LIVE git-clone lane, `backend: runpod` — a `backend: gcp` pin
  is REFUSED since #2028 (`GcpDisabledError`; rollback-only).
