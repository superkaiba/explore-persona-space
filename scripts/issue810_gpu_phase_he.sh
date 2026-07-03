#!/usr/bin/env bash
# Issue #810 follow-up `header-echo-ablation-capture` GPU-phase driver (plan v15
# §4.5/§10): Phase B-he ABLATED capture (empty assistant turn; uploads store +
# he_summaries pack BEFORE any fit) -> Phase D-he recon (9 empty rows, LOCO+LOFO,
# multi-matrix --null-join parity-gated 55-row union bands, JSONs uploaded) ->
# results sentinel (both parity verdicts ride it) -> release. Pure sequencing of
# the code-reviewed CLIs (no logic), mirroring issue810_gpu_phase_uh.sh.
# Pod-side signaling: [phase=...] log lines + the end-of-run sentinel ONLY —
# NEVER a task.py shellout (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_gpu_phase_he.sh
set -euo pipefail

OUT=eval_results/issue_810/header-echo-ablation-capture
STORE_DIR=data/issue_810/store_he
HE_PACK=data/issue_810/he_summaries.pt
HF_MIRROR=issue810_results/header-echo-ablation-capture
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"
# The 9 empty-answer rows (issue810_common.HE_SUMMARY_NAMES, spelled out so the
# sequencing is greppable). Names deliberately reuse the committed rows' names
# (H1-he pairing is by name); NO xbnd/tail/head (undefined at ans_len=0).
HE_ROWS=(im_end turn_nl uh_im_start uh_user uh_nl uh_mean3 uh_max3 bnd_mean5 bnd_max5)

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
# Committed comparators (round-1 + round-3 null matrices + recon JSONs) live on
# branch issue-810 @ origin; a lane that cloned main only must fetch + check
# out the branch. No-op when already on issue-810 (worktree/pod).
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$LOGDIR"

echo "[phase=capture] B-he: teacher-forced ABLATED capture (50 ctx x 48 probes, empty answer)"
# expandable_segments: the inherited allocator knob for 7B multi-layer captures
# (plan v15 §8 — strictly shorter sequences than round 3, kept for lane parity).
# Store + he_summaries.pt upload INSIDE the extractor (fail-loud, verified on a
# fresh listing) — they land on HF before any fit runs. NO completions are read
# on this path (plan v15 §10 code-truth: the probe grid is the hash-pinned pool).
maybe env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/issue810_extract_positions.py --extended-boundary --ablate-answer \
  --gpu --batch-probes 8 --out-dir "$STORE_DIR" --uh-summaries-out "$HE_PACK"

echo "[phase=recon] D-he: 9 empty rows, LOCO+LOFO fits + 1000-perm nulls + 55-row union bands"
# --no-mlp: the MLP validity arm is NOT re-run this round (inherited verbatim
# from plan v11 — plan v15 §11 single-variable discipline). --null-join takes
# BOTH committed matrices (round-1 + round-3); each gets its own 2-cell
# byte-parity gate and the union is composite-keyed (55 rows x 28 L == 1540).
maybe uv run python scripts/issue810_fit_reconstruction.py \
  --rows "${HE_ROWS[@]}" --fold-family both \
  --null-join eval_results/issue_810/null_matrix_reconstruction.json \
              eval_results/issue_810/user-header-newline-summary/null_matrix_user_header.json \
  --committed-recon eval_results/issue_810/reconstruction_skill_by_summary.json \
  --committed-recon-uh eval_results/issue_810/user-header-newline-summary/reconstruction_skill_user_header.json \
  --n-perms 1000 --device cuda --no-mlp \
  --position-store-dir "$STORE_DIR" --out "$OUT" --out-suffix header_echo \
  --upload-prefix "$HF_MIRROR/eval_results"

echo "[phase=sentinel] results sentinel (both parity-gate verdicts ride it, plan v15 § Phased 2)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] write poll_pipeline sentinel with parity verdicts from $OUT/null_matrix_header_echo.json"
else
  uv run python - <<PY
import json, pathlib, time
out = pathlib.Path("$OUT")
nm = json.loads((out / "null_matrix_header_echo.json").read_text())
ub = nm.get("union_band") or {}
note = {
    "phase": "gpu_phase_he",
    "parity_gate": (ub.get("parity_gate") or {}).get("pass"),
    "parity_gates": {
        mid: (rec or {}).get("pass") for mid, rec in (ub.get("parity_gates") or {}).items()
    },
    "null_mode": ub.get("mode"),
    "n_union_cells": ub.get("n_union_cells"),
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

echo "[phase=done] issue810 header-echo-ablation-capture GPU phase complete"
