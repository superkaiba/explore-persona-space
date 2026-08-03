#!/usr/bin/env bash
# OOD scatter preds + map-recon-on-eval-distribution — one behavior per box.
#
# TWO partial items on ONE box because they are the SAME fits.py transfer leg
# over the SAME staged stores and the SAME map fits; splitting them would pay
# the multi-GB staging + the 2 x 3 map fits twice for no extra science.
#
#  * OOD scatter preds (`--transfer-preds`): the committed transfer leg emits
#    per-(arm, rung) aggregate rows only, so the interim report's scatter
#    section reads "per-cell prediction arrays were persisted only for the
#    train setting, so OOD scatters are not available". The flag wires the
#    ALREADY-REVIEWED `arms.transfer_preds_rows` helper (the eval-rung twin of
#    the train setting's `_save_cell_preds`, in production use by the wcrung +
#    pvsynth rung runners) into the main transfer leg, with the per-context
#    OOD rung riding the helper's generic label column. Any later OOD scatter /
#    per-rung subset read is then a pure re-analysis, never another re-score.
#  * map-recon-on-eval-dist (`--eval-rung-knn`): the map's reconstruction read
#    already existed but was POOLED over the whole eval split and R^2-only.
#    The flag breaks it out PER EVAL DISTRIBUTION (each OOD rung separately)
#    and adds the standing kNN-retrieval companion, computed with the same
#    `mapping_baselines.knn_retrieval` helper `map_diagnostics` uses on the
#    U-pool holdout — so the WildChat-pool read and the behavior-eval-rung read
#    are directly comparable instead of only the former existing.
#
# Both flags default OFF, so every other lane's artifacts are byte-identical.
# Grid: both variants x the full U ladder (250 / 5,000 / full) x regime e1 x the
# behavior's 3 L anchors, draw 0 / seed 0 — the map axis is what the recon read
# needs, and the max-L cell is the scatter's representative operating slice.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIOR:?set EPM_I1739_BEHAVIOR to exactly one of evil|sycophancy|hallucination}"
export EPM_I1739_BEHAVIORS="$B"
OUT_ROOT="eval_results/issue_1739/ood_scatter_preds/$B"
RECON_ROOT="eval_results/issue_1739/map_recon_evaldist/$B"
mkdir -p "$OUT_ROOT" "$RECON_ROOT"

case "$B" in
  evil) BUDGETS="250 2500 8000" ;;
  *) BUDGETS="250 2500 16000" ;;
esac

upload_out_roots() {
  # map_diagnostics.json is the map-recon item's deliverable — publish it under
  # its own prefix as well as inside the transfer out-root, so the two items
  # stay separately addressable even though one run produced both.
  if [ -f "$OUT_ROOT/map_diagnostics.json" ]; then
    cp "$OUT_ROOT/map_diagnostics.json" "$RECON_ROOT/map_diagnostics.json"
  fi
  uv run python - "$OUT_ROOT" "$RECON_ROOT" "$B" <<'PYEOF' || echo "[oodpreds] WARNING: upload leg failed" >&2
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

out_root, recon_root, behavior = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
api = HfApi()
for root, prefix in (
    (out_root, f"issue1739_maxood/ood_scatter_preds/{behavior}"),
    (recon_root, f"issue1739_maxood/map_recon_evaldist/{behavior}"),
):
    if not any(root.rglob("*")):
        print(f"[oodpreds] nothing to upload under {root}", flush=True)
        continue
    hub.retry_transient(
        lambda root=root, prefix=prefix: api.upload_folder(
            folder_path=str(root),
            path_in_repo=prefix,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
        ),
        what=f"upload {prefix}",
    )
    print(f"[oodpreds] uploaded {root} -> {prefix}", flush=True)
PYEOF
}
trap upload_out_roots EXIT

echo "[oodpreds] behavior=$B budgets='$BUDGETS' $(date -u +%FT%TZ)"

echo "[oodpreds] stage inputs $(date -u +%FT%TZ)"
bash scripts/issue1739_leg2.sh

FITS_ARGS=(--behavior "$B"
  --labeled-store "data/issue_1739/store/${B}_labeling"
  --dv-json "eval_results/issue_1739/dv_dataset/$B/labeling.json"
  --u-store data/issue_1739/hf_dl/u_store
  --e1-store "data/issue_1739/store/${B}_extraction"
  --out-root "$OUT_ROOT"
  --tensors-root analysis_tensors/issue_1739
  --device cuda
  --config config_a --transfer --transfer-preds --eval-rung-knn
  --regimes e1 --u-sizes 250 5000 full
  --draws 0 --seeds 0
  --n-boot 500 --n-perm 500)
# shellcheck disable=SC2086
FITS_ARGS+=(--budgets $BUDGETS)

# §9 pilot gate through the SAME entrypoint + args (MEASURED per-budget
# unit-group + per-map-fit basis, resumes into the production leg; designed
# rc=7 halt with pilot_report.json when the projection blows the fence).
echo "[oodpreds] pilot gate $(date -u +%FT%TZ)"
set +e
uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}" \
  --pilot --plan-wall-h "${EPM_I1739_OODPREDS_PLAN_WALL_H:-4}" --pilot-abort-mult 3
pilot_rc=$?
set -e
if [ "$pilot_rc" -eq 7 ]; then
  echo "[oodpreds] PILOT GATE ABORT behavior=$B — see $OUT_ROOT/pilot_report.json" >&2
  exit 7
elif [ "$pilot_rc" -ne 0 ]; then
  echo "[oodpreds] pilot FAILED rc=$pilot_rc behavior=$B" >&2
  exit "$pilot_rc"
fi

echo "[oodpreds] production transfer leg (preds + per-rung recon) $(date -u +%FT%TZ)"
uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}"

n_preds=$(find "$OUT_ROOT/arm_results/percell/transfer_preds" -name '*.jsonl' 2>/dev/null | wc -l)
echo "[oodpreds] per-unit OOD preds sidecars: $n_preds"
[ "$n_preds" -gt 0 ] || { echo "[oodpreds] FATAL: --transfer-preds produced no sidecars" >&2; exit 1; }
uv run python - "$OUT_ROOT/map_diagnostics.json" <<'PYEOF'
import json
import sys

d = json.load(open(sys.argv[1]))
n_rung = sum(1 for v in d.values() if isinstance(v, dict) and v.get("eval_rung", {}).get("per_rung"))
print(f"[oodpreds] map keys with a per-rung eval-dist recon block: {n_rung}/{len(d)}", flush=True)
if not n_rung:
    raise SystemExit("FATAL: --eval-rung-knn produced no per_rung block")
PYEOF

echo "[oodpreds] done rc=0 $(date -u +%FT%TZ)"
