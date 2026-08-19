#!/usr/bin/env bash
# Issue #2223 NAP round — P2 CPU-pod launcher (native-axis-fidelity-preimage).
#
# Off-GPU reductions phase (plan v7 §4 Step 4, §9 P2 row): stage the P1
# fp16 store + cA_map + map_metrics + role-adherence judge scores from HF,
# run `--phase axes` (filtered axes + preimage + axis_cos + diagnostics +
# cross-pool tau/alpha), then `--phase upload` (verified bulk uploads).
#
# Chained legs, each with a [phase=...] breadcrumb + per-leg rc echo:
#   bootstrap (uv sync) -> stage (4 HF prefixes -> consumer layout, fail-loud
#   entry asserts) -> axes (CPU: stream role sums + 8 pinvs + 5-fold
#   stability refits) -> upload (HF data repo, exact-set verified) ->
#   sentinel write + the single reserved [phase=done] terminal line.
#
# Launch (detached, on pod-2223-napp2 — cpu-bigmem, no GPUs):
#   setsid nohup bash scripts/issue2223_nap_p2_pod.sh \
#     > /workspace/logs/issue-2223-nap-p2.log 2>&1 < /dev/null &
set -euo pipefail

REPO="${EPS_REPO_DIR:-/workspace/explore-persona-space}"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
cd "$REPO"

HFP="issue2223_casestudy/native_axis_fidelity_preimage"
DATA_REPO="superkaiba1/explore-persona-space-data"
OUT_ROOT="${NAP_OUT_ROOT:-$REPO/eval_results/issue_2223/casestudy_replay}"
EXT_DIR="$OUT_ROOT/qwen3-32b/extractions"
STORE_DIR="${NAP_STORE_DIR:-$REPO/data/issue_2223/nap_store/qwen3-32b}"
SCORES_DIR="$EXT_DIR/paper_pipeline/scores"
STAGE="$REPO/data/issue_2223/hf_dl/nap_p2_stage"
SENTINEL=/workspace/logs/issue-2223-nap-p2.done

echo "[phase=bootstrap]"
rc=0
uv sync --locked > /tmp/issue-2223-p2-uv-sync.log 2>&1 || rc=$?
echo "[nap-p2] uv sync rc=$rc"
if [ "$rc" -ne 0 ]; then
  tail -40 /tmp/issue-2223-p2-uv-sync.log
  exit "$rc"
fi
export UV_NO_SYNC=1

echo "[phase=stage]"
# Resume predicate: skip the restage when the FULL consumer layout is already
# present (incl. native_axes.pt — its absence killed run 1's axes leg at the
# H3-table assert; #2223 P2 crash-fix round).
if [ -f "$STORE_DIR/capture_regime.json" ] && [ -f "$EXT_DIR/cA_map/M_46.pt" ] \
   && [ -f "$EXT_DIR/map_metrics.json" ] && [ -f "$EXT_DIR/native_axes.pt" ] \
   && [ "$(ls "$SCORES_DIR"/*.json 2>/dev/null | wc -l)" -eq 275 ]; then
  echo "[nap-p2] stage skipped (consumer layout present)"
else
mkdir -p "$STAGE"
rc=0
uv run python - "$STAGE" <<'PY' || rc=$?
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

stage = sys.argv[1]
repo = "superkaiba1/explore-persona-space-data"
hfp = "issue2223_casestudy/native_axis_fidelity_preimage"
# Verbatim prefix mirror: files land at <stage>/<full repo-relative path>;
# the shell below mv's each leaf into the consumer layout + asserts entries.
for prefix in (
    f"{hfp}/analysis_tensors/nap_store/qwen3-32b",
    f"{hfp}/analysis_tensors/cA_map/qwen3-32b",
    f"{hfp}/analysis_tensors/runner_extractions/qwen3-32b",
    f"{hfp}/extractions/qwen3-32b",
    f"{hfp}/raw_completions/extraction/judged",
):
    files = stage_hub_prefix(repo, prefix, stage)
    print(f"[nap-p2-stage] {prefix}: {len(files)} files")
PY
echo "[nap-p2] stage rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

# hub-rel -> consumer layout (fail-loud on every consumer entry file).
rm -rf "$STORE_DIR" "$EXT_DIR/cA_map"
mkdir -p "$(dirname "$STORE_DIR")" "$EXT_DIR" "$SCORES_DIR"
mv "$STAGE/$HFP/analysis_tensors/nap_store/qwen3-32b" "$STORE_DIR"
mv "$STAGE/$HFP/analysis_tensors/cA_map/qwen3-32b" "$EXT_DIR/cA_map"
mv "$STAGE/$HFP/analysis_tensors/runner_extractions/qwen3-32b/native_axes.pt" "$EXT_DIR/native_axes.pt"
mv "$STAGE/$HFP/extractions/qwen3-32b/map_metrics.json" "$EXT_DIR/map_metrics.json"
mv "$STAGE/$HFP/raw_completions/extraction/judged/"*.json "$SCORES_DIR/"
test -f "$STORE_DIR/capture_regime.json" || { echo "[nap-p2] MISSING store capture_regime.json"; exit 1; }
test -f "$EXT_DIR/cA_map/M_46.pt" || { echo "[nap-p2] MISSING cA_map/M_46.pt"; exit 1; }
test -f "$EXT_DIR/map_metrics.json" || { echo "[nap-p2] MISSING map_metrics.json"; exit 1; }
test -f "$EXT_DIR/native_axes.pt" || { echo "[nap-p2] MISSING native_axes.pt"; exit 1; }
N_STORE=$(ls "$STORE_DIR" | wc -l)
N_SCORES=$(ls "$SCORES_DIR"/*.json | wc -l)
echo "[nap-p2] staged: store files=$N_STORE scores=$N_SCORES"
[ "$N_SCORES" -eq 275 ] || { echo "[nap-p2] score-file count $N_SCORES != 275"; exit 1; }
fi

echo "[phase=axes]"
rc=0
uv run python scripts/issue2223_native_preimage_capture.py \
  --phase axes --model 32b --out-root "$OUT_ROOT" --store-dir "$STORE_DIR" \
  --scores-dir "$SCORES_DIR" \
  > "$LOGDIR/issue-2223-nap-axes.log" 2>&1 || rc=$?
echo "[nap-p2] axes rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- axes log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-axes.log"
  exit "$rc"
fi

echo "[phase=upload]"
rc=0
uv run python scripts/issue2223_native_preimage_capture.py \
  --phase upload --model 32b --out-root "$OUT_ROOT" --store-dir "$STORE_DIR" \
  > "$LOGDIR/issue-2223-nap-p2-upload.log" 2>&1 || rc=$?
echo "[nap-p2] upload rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- upload log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-p2-upload.log"
  exit "$rc"
fi

GIT_SHA=$(git rev-parse HEAD)
uv run python -c "
import json, sys, time
json.dump(
    {
        'issue': 2223,
        'label': 'native_axis_fidelity_preimage',
        'phase': 'p2_done',
        'rc': 0,
        'ts': time.time(),
        'git_sha': sys.argv[1],
        'ext_dir': sys.argv[2],
    },
    open(sys.argv[3], 'w'),
    indent=2,
)
" "$GIT_SHA" "$EXT_DIR" "$SENTINEL"
echo "[nap-p2] sentinel written: $SENTINEL"
echo "[phase=done]"
