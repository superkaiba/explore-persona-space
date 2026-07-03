#!/usr/bin/env bash
# Issue #810 follow-up `user-header-newline-summary` GPU-phase driver (plan v11
# §4.5/§10): Phase B-x extended-boundary capture (uploads store + uh_summaries
# pack BEFORE any fit) -> Phase D-x-recon (9 new rows, LOCO+LOFO, --null-join
# parity-gated union bands, JSONs uploaded) -> cross-layer driver (22:30 recipe
# on the xbnd pools) -> results sentinel (parity verdict rides it) -> release.
# Pure sequencing of the code-reviewed CLIs (no logic), mirroring
# issue810_cpu_phase_g1.sh. Pod-side signaling: [phase=...] log lines + the
# end-of-run sentinel ONLY — NEVER a task.py shellout (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_gpu_phase_uh.sh
set -euo pipefail

OUT=eval_results/issue_810/user-header-newline-summary
STORE_DIR=data/issue_810/store_uh
UH_PACK=data/issue_810/uh_summaries.pt
HF_MIRROR=issue810_results/user-header-newline-summary
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"
# The 9 new per-layer rows (issue810_common.UH_SUMMARY_NAMES, spelled out so
# the sequencing is greppable).
UH_ROWS=(uh_im_start uh_user uh_nl uh_mean3 uh_max3 bnd_mean5 bnd_max5 mean_xbnd maxp_xbnd)

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
# Parent artifacts (committed round-1 null matrix + recon JSONs + graded E0)
# live on branch issue-810 @ origin; a lane that cloned main only must fetch +
# check out the branch. No-op when already on issue-810 (worktree/pod).
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$LOGDIR"

echo "[phase=capture] B-x: teacher-forced extended-boundary capture (50 ctx x 48 probes)"
# expandable_segments: the plan §8 known allocator knob for long 7B captures
# (A100-80 fragments under multi-layer capture; the crash message recommends it).
# Store + uh_summaries.pt upload INSIDE the extractor (fail-loud, verified on a
# fresh listing) — they land on HF before any fit runs.
maybe env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/issue810_extract_positions.py --extended-boundary --gpu \
  --batch-probes 8 --out-dir "$STORE_DIR" --uh-summaries-out "$UH_PACK"

echo "[phase=recon] D-x: 9 new rows, LOCO+LOFO fits + 1000-perm nulls + parity-gated union bands"
# --no-mlp: the MLP validity arm is NOT re-run this round (plan v11 §4
# divergence row 6 — inherited validity, 0.991/0.922 rank agreement).
maybe uv run python scripts/issue810_fit_reconstruction.py \
  --rows "${UH_ROWS[@]}" --fold-family both \
  --null-join eval_results/issue_810/null_matrix_reconstruction.json \
  --committed-recon eval_results/issue_810/reconstruction_skill_by_summary.json \
  --n-perms 1000 --device cuda --no-mlp \
  --position-store-dir "$STORE_DIR" --out "$OUT" --out-suffix user_header \
  --upload-prefix "$HF_MIRROR/eval_results"

echo "[phase=crosslayer] cross-layer pooled reads (22:30 recipe on the xbnd pools)"
maybe uv run python scripts/issue810_uh_crosslayer.py --device cuda \
  --position-store-dir "$STORE_DIR" --out "$OUT" \
  --upload-prefix "$HF_MIRROR/eval_results"

echo "[phase=sentinel] results sentinel (the parity-gate verdict rides it, plan v11 § Phased 2)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] write poll_pipeline sentinel with parity verdict from $OUT/null_matrix_user_header.json"
else
  uv run python - <<PY
import json, pathlib, time
out = pathlib.Path("$OUT")
nm = json.loads((out / "null_matrix_user_header.json").read_text())
ub = nm.get("union_band") or {}
note = {
    "phase": "gpu_phase_uh",
    "parity_gate": (ub.get("parity_gate") or {}).get("pass"),
    "null_mode": ub.get("mode"),
    "band_rows": ub.get("band_rows"),
    "hf_mirror": "$HF_MIRROR/eval_results",
}
log_dir = pathlib.Path("/workspace/logs")
try:
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"issue-810-epm_results-{int(time.time())}.json"
except OSError:
    target = out / "issue-810-epm_results-sentinel.json"
target.write_text(json.dumps({
    "sentinel_schema_version": 1, "kind": "epm:results", "version": 1,
    "note": note, "ts": int(time.time()),
}, indent=2))
print(f"[phase=sentinel] wrote {target} (parity_gate pass={note['parity_gate']}, mode={note['null_mode']})")
PY
fi

echo "[phase=done] issue810 user-header-newline-summary GPU phase complete"
