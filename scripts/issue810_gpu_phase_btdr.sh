#!/usr/bin/env bash
# Issue #810 follow-up `boundary-truncation-dose-response` GPU-phase driver
# (plan v18 §4.5/§10): k=1.0 endpoint-parity gate + Phase B-btdr TRUNCATED
# captures (k=25/50/75%, ONE process/one model load; per-k store + pack
# uploaded INSIDE the extractor BEFORE any fit) -> per-k recon point fits
# (9 he rows, LOCO+LOFO, NO nulls, NO MLP arm; JSONs uploaded) -> results
# sentinel -> release. Pure sequencing of the code-reviewed CLIs (no logic),
# mirroring issue810_gpu_phase_he.sh. Pod-side signaling: [phase=...] log
# lines + the end-of-run sentinel ONLY — NEVER a task.py shellout
# (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_gpu_phase_btdr.sh
set -euo pipefail

OUT=eval_results/issue_810/boundary-truncation-dose-response
STORE_BASE=data/issue_810
HF_MIRROR=issue810_results/boundary-truncation-dose-response
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"
# The 9 pair rows (issue810_common.HE_SUMMARY_NAMES, spelled out so the
# sequencing is greppable). Names deliberately reuse the committed rows' names
# (pairing across sides and k is by name — plan v18 §4.6 item 1).
HE_ROWS=(im_end turn_nl uh_im_start uh_user uh_nl uh_mean3 uh_max3 bnd_mean5 bnd_max5)
# The interior k grid (plan v18 §11: pre-specified; endpoints reuse committed data).
K_PCTS=(25 50 75)

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
# Committed comparators (round-1/3/4 recon JSONs) live on branch issue-810 @
# origin; a lane that cloned main only must fetch + check out the branch.
# No-op when already on issue-810 (worktree/pod).
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$LOGDIR"

echo "[phase=capture] B-btdr: k=1.0 endpoint-parity gate + 3 truncated captures (one model load)"
# expandable_segments: the inherited allocator knob for 7B multi-layer captures
# (plan v18 §8 — kept for lane parity with rounds 3/4). The extractor runs the
# k=1.0 endpoint-parity probe FIRST (halt BEFORE any capture spend — plan v18
# kill criterion 2), then loops k in {0.25, 0.5, 0.75} re-using the loaded
# model; each k's store (answer_position_sweep_btdr_k{pct}/) + pack
# (btdr_summaries_k{pct}.pt) upload INSIDE the extractor the moment that pass
# completes (fail-loud, verified on a fresh listing) — on HF before any fit.
maybe env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/issue810_extract_positions.py --extended-boundary \
  --truncate-frac 0.25 0.5 0.75 \
  --gpu --batch-probes 8 --out-dir "$STORE_BASE"

echo "[phase=recon] D-btdr: per-k point fits (9 rows, LOCO+LOFO, NO nulls, NO MLP)"
# --n-perms 0: the registered no-nulls path (plan v18 §4 divergence 3 — the
# dose DV is the paired Δ; mid-k absolutes are descriptive trace points, NO
# band verdicts). --no-mlp: PARITY with the rounds-3/4 shells (no registered
# read consumes mlp_skill). --out-suffix btdr_k{pct}: the per-k filenames
# reconstruction_skill_btdr_k{pct}.json (plan v18 §4.6 item 3).
for pct in "${K_PCTS[@]}"; do
  maybe uv run python scripts/issue810_fit_reconstruction.py \
    --rows "${HE_ROWS[@]}" --fold-family both \
    --n-perms 0 --no-mlp --device cuda \
    --position-store-dir "$STORE_BASE/store_btdr_k${pct}" \
    --out "$OUT" --out-suffix "btdr_k${pct}" \
    --upload-prefix "$HF_MIRROR/eval_results"
done

echo "[phase=sentinel] results sentinel (per-k fit summaries ride it, plan v18 § Phased 2)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] write poll_pipeline sentinel from $OUT/reconstruction_skill_btdr_k{25,50,75}.json"
else
  uv run python - <<PY
import json, pathlib, time
out = pathlib.Path("$OUT")
per_k = {}
for pct in (25, 50, 75):
    blob = json.loads((out / f"reconstruction_skill_btdr_k{pct}.json").read_text())
    per_k[str(pct)] = {
        "n_contexts": blob.get("n_contexts"),
        "fold_family": blob.get("fold_family"),
        "rows": sorted((blob.get("by_summary") or {}).keys()),
    }
note = {
    "phase": "gpu_phase_btdr",
    "truncate_pcts": [25, 50, 75],
    "per_k_fits": per_k,
    "n_perms": 0,
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
print(f"[phase=sentinel] wrote {target} (3 per-k fit summaries)")
PY
fi

echo "[phase=done] issue810 boundary-truncation-dose-response GPU phase complete"
