#!/usr/bin/env bash
# Gap-1 fill: land the leg-1 prefix null-probe MECHANISM DIAGNOSIS in a
# persisted summary. The committed bareq summary predates the scorer's
# null_anomaly_diagnostic ladder (`nulls` is []); the current scorer writes
# nulls[variant].anomaly_diagnostic under --force-null-sweep (default) +
# --null-shuffle-seeds 8. Leg-1 needs the per-behavior <b>_labeling train
# stores + the shared wcrung_capture_store + the bare capture store, so this
# runs on a box (not the shared VM). CPU scoring; the GPU is unused. Self-
# uploads the null-diag summaries to HF so the VM can harvest the verdict.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

echo "[gap1] stage train + extraction stores (labeling/<b>_labeling, <b>_extraction, wcrung) $(date -u +%FT%TZ)"
uv run python scripts/issue1739_wcrung_arms_run.py \
  --behaviors evil sycophancy hallucination \
  --store-root data/issue_1739/hf_dl \
  --stage-only

echo "[gap1] defensive explicit wcrung stage (idempotent) $(date -u +%FT%TZ)"
uv run python - <<'PYEOF'
import argparse
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from scripts.issue1739_wcrung_arms_run import stage_wcrung_store
p = stage_wcrung_store(argparse.Namespace(store_root=Path("data/issue_1739/hf_dl")))
print(f"[gap1] wcrung store at {p}", flush=True)
PYEOF

echo "[gap1] stage bare capture store + committed contrast + queries $(date -u +%FT%TZ)"
uv run python scripts/issue1739_bareq_score_prestage.py

echo "[gap1] leg-1 null-anomaly diagnostic (all 3 behaviors) $(date -u +%FT%TZ)"
uv run python scripts/issue1739_bareq_score.py \
  --behaviors evil sycophancy hallucination --legs 1 \
  --null-shuffle-seeds 8 --device cpu \
  --store-root data/issue_1739/hf_dl \
  --bareq-store data/issue_1739/hf_dl/bareq_capture_store \
  --query-manifest eval_results/issue_1739/bareq_map/bareq_queries.json \
  --out-root data/issue_1739/nulldiag_out/bareq_map \
  --force-own-pool-frozen

echo "[gap1] upload null-diag summaries to HF $(date -u +%FT%TZ)"
uv run python - <<'PYEOF'
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.orchestrate import hub
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="data/issue_1739/nulldiag_out/bareq_map",
    path_in_repo="issue1739_maxood/bareq_null_diag",
    repo_id=hub.DEFAULT_DATASET_REPO,
    repo_type="dataset",
    allow_patterns=["**/*.json", "**/*.jsonl"],
)
print("[gap1] HF upload done", flush=True)
PYEOF

echo "[gap1] done rc=0 $(date -u +%FT%TZ)"
