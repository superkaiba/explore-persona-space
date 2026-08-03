#!/usr/bin/env bash
# Gap-2 fill: hallucination arms 7/8/12 at the MAXIMUM OOD budget slice.
#
# The committed OOD grid covers arms 7/8/12 (map->ridge-pred, map->ridge-true,
# oracle-reg) only at budget_l in {250,2500} u_rung=250; the max-budget
# operating slice (E1, u_rung=full=18,793, budget_l=16000, eval rungs
# nqopen/simpleqa/train, context_end) is entirely missing. This fills exactly
# that corner via the proven leg2 stage + a SCOPED fits.py --transfer call
# (one (variant=map,u=full) whitening+map GEMM in run_grid_multi, then the 3
# arms over 28 layers x rungs), then self-uploads the result to HF so the
# corner survives instance teardown regardless of exit code.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

export EPM_I1739_BEHAVIORS=hallucination

echo "[gap2] stage inputs $(date -u +%FT%TZ)"
bash scripts/issue1739_leg2.sh

echo "[gap2] scoped max-budget transfer fit $(date -u +%FT%TZ)"
uv run python scripts/issue1739_fits.py --behavior hallucination \
  --labeled-store data/issue_1739/store/hallucination_labeling \
  --dv-json eval_results/issue_1739/dv_dataset/hallucination/labeling.json \
  --u-store data/issue_1739/hf_dl/u_store \
  --e1-store data/issue_1739/store/hallucination_extraction \
  --out-root eval_results/issue_1739/hallucination_maxood \
  --tensors-root analysis_tensors/issue_1739 \
  --device cuda \
  --config config_a --transfer \
  --regimes e1 --u-sizes full --budgets 16000 --draws 0 --seeds 0 \
  --transfer-arms arm7_map_ridge_pred arm8_map_ridge_true arm12_oracle_reg \
  --n-boot 500 --n-perm 500

echo "[gap2] upload result to HF $(date -u +%FT%TZ)"
uv run python - <<'PYEOF'
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.orchestrate import hub
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="eval_results/issue_1739/hallucination_maxood",
    path_in_repo="issue1739_maxood/hallucination",
    repo_id=hub.DEFAULT_DATASET_REPO,
    repo_type="dataset",
)
print("[gap2] HF upload done", flush=True)
PYEOF

echo "[gap2] done rc=0 $(date -u +%FT%TZ)"
