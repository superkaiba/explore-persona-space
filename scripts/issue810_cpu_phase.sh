#!/usr/bin/env bash
# Issue #810 CPU-phase driver: D-recon -> C(resume) -> D-readout -> E -> upload.
# Pure sequencing of the code-reviewed CLIs (no logic). Runs on a cpu-mid
# instance: earlyoom killed the ~7 GB Phase-C working set on the shared VM
# twice (2026-07-01), so the remaining CPU phases route to a dedicated lane
# per the CLAUDE.md #747 rule. Phase A (free leg) is subsumed by the full
# Phase D sweep now that the Phase B store is on HF.
set -euo pipefail

OUT=eval_results/issue_810
CACHE=/tmp/i810_phasec_cache
STORE_HF=issue658_theory_assumptions/answer_position_sweep
mkdir -p "$OUT" figures/issue_810

echo "[phase=cache_restore]"
uv run python - <<'PY'
import pathlib, tarfile
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download
p = hf_hub_download("superkaiba1/explore-persona-space-data",
                    "issue810_results/judge_cache.tar.gz", repo_type="dataset")
with tarfile.open(p) as t:
    t.extractall("/tmp")
n = len(list(pathlib.Path("/tmp/i810_phasec_cache").glob("*.json")))
print(f"[phase=cache_restore] {n} cached judge entries restored")
assert n > 15000, f"cache restore too small: {n}"
PY

echo "[phase=fit_reconstruction]"
uv run python scripts/issue810_fit_reconstruction.py \
  --position-store-hf "$STORE_HF" \
  --out "$OUT" --upload-prefix issue810_results/recon

echo "[phase=batch_rejudge]"
uv run python scripts/issue810_batch_rejudge_highm.py \
  --out "$OUT/phase_c" --cache-dir "$CACHE"

echo "[phase=fit_readout]"
uv run python scripts/issue810_fit_readout.py \
  --e0-highm "$OUT/phase_c/e0_highm_graded.json" \
  --position-store-hf "$STORE_HF" \
  --out "$OUT" --upload-prefix issue810_results/readout

echo "[phase=analyze]"
uv run python scripts/issue810_analyze.py \
  --in "$OUT" --out "$OUT/analysis" --fig-dir figures/issue_810

echo "[phase=upload]"
uv run python - <<'PY'
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="eval_results/issue_810",
                  path_in_repo="issue810_results/eval_results",
                  repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset",
                  commit_message="issue #810: CPU-phase outputs (recon/phase_c/readout/analysis)")
api.upload_folder(folder_path="figures/issue_810",
                  path_in_repo="issue810_results/figures",
                  repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset",
                  commit_message="issue #810: hero + exploratory figures")
print("[phase=upload_done]")
PY

echo "[phase=done] issue810 CPU phase complete"
