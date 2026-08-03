#!/usr/bin/env bash
# R5 (unjudged trait-eliciting map pool) — U-LADDER FILL, one behavior per box.
#
# The committed compose cells cover exactly ONE compose-U rung (5,000) at the
# headline regime e1: evil's 16 cells live in the main out-root, sycophancy's
# and hallucination's in nonlinear_map/<b>/compose_linear/. This box sweeps the
# REST of the U ladder for the same (2 variants x 3 f-combos x 3 L-anchor) grid
# — compose sizes 250 / 2,500 / 18,793 — so the "does unlabeled behavior-
# eliciting map data substitute for labels?" read has a U curve instead of a
# single point. No new arms: `--compose` is the reviewed §4b sub-grid, and the
# only axis this box adds is `--compose-u-size`.
#
# ONE invocation per (compose_size, L anchor) because:
#   * `--compose-u-size` is a scalar (each rung is its own invocation), and
#   * `compose_u_pool` FAILS LOUD when a (size, f_u, f_l, L) combo cannot supply
#     its eliciting quota from the train split's non-cell contexts. That is a
#     STRUCTURAL infeasibility of the cell, not a bug (it is why the committed
#     syco/hallu lanes have 16 of 18 cells — `fu0.5_fl0.0` at the top L anchor
#     is unsatisfiable there). Isolating each (size, L) keeps one infeasible
#     corner from taking the rest of the ladder down with it; the rc + the
#     matched `compose_u_pool: need` log line are RECORDED per invocation in
#     r5_invocations.json, and the box FAILS at the end on any rc that is NOT
#     that known shortfall.
# Every invocation shares ONE out-root, so the plain u=full cells (the 2
# variants x 3 budgets the compose grid rides alongside) compute once and
# resume-skip thereafter, and the per-cell resume makes a relaunch cheap.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIOR:?set EPM_I1739_BEHAVIOR to exactly one of evil|sycophancy|hallucination}"
export EPM_I1739_BEHAVIORS="$B"
OUT_ROOT="eval_results/issue_1739/r5_unjudged_trait_pool/$B"
ACCT="$OUT_ROOT/r5_invocations.json"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

case "$B" in
  evil) BUDGETS="250 2500 8000" ;;
  *) BUDGETS="250 2500 16000" ;;
esac
# 5,000 is already committed (see header) — this box fills the rest of the
# ladder. STORE_FIT_ROWS = 18,793 is the realized "full" U pool (constants.py).
COMPOSE_SIZES="${EPM_I1739_R5_SIZES:-250 2500 18793}"

# Upload whatever exists on EVERY exit path (crash included): the out-root is
# the round's only durable artifact and the instance self-deletes.
upload_out_root() {
  uv run python - "$OUT_ROOT" "$B" <<'PYEOF' || echo "[r5] WARNING: upload leg failed" >&2
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

out_root, behavior = Path(sys.argv[1]), sys.argv[2]
if not any(out_root.rglob("*")):
    print("[r5] nothing to upload", flush=True)
    raise SystemExit(0)
hub.retry_transient(
    lambda: HfApi().upload_folder(
        folder_path=str(out_root),
        path_in_repo=f"issue1739_maxood/r5_unjudged_trait_pool/{behavior}",
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
    ),
    what="r5 out-root upload",
)
print("[r5] HF upload done", flush=True)
PYEOF
}
trap upload_out_root EXIT

echo "[r5] behavior=$B budgets='$BUDGETS' compose_sizes='$COMPOSE_SIZES' $(date -u +%FT%TZ)"

echo "[r5] stage inputs $(date -u +%FT%TZ)"
bash scripts/issue1739_leg2.sh

FITS_COMMON=(--behavior "$B"
  --labeled-store "data/issue_1739/store/${B}_labeling"
  --dv-json "eval_results/issue_1739/dv_dataset/$B/labeling.json"
  --u-store data/issue_1739/hf_dl/u_store
  --e1-store "data/issue_1739/store/${B}_extraction"
  --out-root "$OUT_ROOT"
  --tensors-root analysis_tensors/issue_1739
  --device cuda
  --config config_a
  --regimes e1 --u-sizes full --draws 0 --seeds 0
  --compose
  --n-boot 500 --n-perm 500)

# §9 pilot gate ONCE, at the cheapest (size, L) corner, through the SAME
# production entrypoint + args: writes pilot_report.json (the MEASURED
# per-budget unit-group + per-map-fit basis this box's projection rests on)
# and exits rc=7 — a designed halt with an artifact, never a bare rc=1 — if the
# projection exceeds --pilot-abort-mult x --plan-wall-h. The pilot's units
# RESUME into the production leg below (same out-root), so it costs nothing
# beyond the measurement.
FIRST_SIZE="$(echo "$COMPOSE_SIZES" | awk '{print $1}')"
FIRST_L="$(echo "$BUDGETS" | awk '{print $1}')"
echo "[r5] pilot gate (size=$FIRST_SIZE L=$FIRST_L) $(date -u +%FT%TZ)"
set +e
uv run python scripts/issue1739_fits.py "${FITS_COMMON[@]}" \
  --compose-u-size "$FIRST_SIZE" --budgets "$FIRST_L" \
  --pilot --plan-wall-h "${EPM_I1739_R5_PLAN_WALL_H:-8}" --pilot-abort-mult 3
pilot_rc=$?
set -e
if [ "$pilot_rc" -eq 7 ]; then
  echo "[r5] PILOT GATE ABORT behavior=$B — projected wall > 3x plan; see $OUT_ROOT/pilot_report.json" >&2
  exit 7
elif [ "$pilot_rc" -ne 0 ]; then
  echo "[r5] pilot FAILED rc=$pilot_rc behavior=$B" >&2
  exit "$pilot_rc"
fi

printf '[]' > "$ACCT.parts"
FATAL=0
for size in $COMPOSE_SIZES; do
  for L in $BUDGETS; do
    tag="s${size}_L${L}"
    log="$LOG_DIR/$tag.log"
    echo "[r5] invocation $tag $(date -u +%FT%TZ)"
    set +e
    uv run python scripts/issue1739_fits.py "${FITS_COMMON[@]}" \
      --compose-u-size "$size" --budgets "$L" > "$log" 2>&1
    rc=$?
    set -e
    shortfall=0
    if [ "$rc" -ne 0 ] && grep -q 'compose_u_pool: need' "$log"; then
      shortfall=1
      echo "[r5] $tag rc=$rc STRUCTURAL SHORTFALL (compose_u_pool quota) — recorded, not fatal"
      grep -m1 'compose_u_pool: need' "$log" || true
    elif [ "$rc" -ne 0 ]; then
      FATAL=1
      echo "[r5] $tag rc=$rc FATAL (not a compose_u_pool shortfall) — tail:" >&2
      tail -25 "$log" >&2
    else
      echo "[r5] $tag rc=0"
    fi
    uv run python - "$ACCT.parts" "$tag" "$size" "$L" "$rc" "$shortfall" <<'PYEOF'
import json
import sys

path, tag, size, budget, rc, shortfall = sys.argv[1:7]
rows = json.load(open(path))
rows.append(
    {
        "tag": tag,
        "compose_u_size": int(size),
        "budget_l": int(budget),
        "rc": int(rc),
        "compose_pool_shortfall": bool(int(shortfall)),
    }
)
json.dump(rows, open(path, "w"), indent=1)
PYEOF
  done
done
mv "$ACCT.parts" "$ACCT"
echo "[r5] invocation accounting -> $ACCT"
cat "$ACCT"

if [ "$FATAL" -ne 0 ]; then
  echo "[r5] FAILING: at least one invocation failed for a reason other than the known compose_u_pool shortfall" >&2
  exit 1
fi
echo "[r5] done rc=0 $(date -u +%FT%TZ)"
