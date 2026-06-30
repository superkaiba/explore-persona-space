#!/bin/bash
# Phase B — A3.6c causal context-vector patch on GCP `eval` GPU lane.
# (1) Live parity probe (BINDING precondition; cos>=0.95 AND L2 ratio within 10%
#     against the #664 stored c_C_trained — the only guard against an rsLoRA-vs-
#     classic-scale stack mismatch silently flipping the A3.6c readout, #601.)
# (2) A3.6c causal context-vector patch — the ONE GPU arm (plan v3 §4 A3.6c +
#     §8 compute table: 4 cells × 8 bystanders × 3 L × 2 scopes × 6 variants on
#     Qwen2.5-7B fleet adapters; ~6 GPU-h on 1× eval-intent GPU per plan).
# (3) Upload outputs to HF data repo (the GCE instance lacks GH_TOKEN — STARTUP_
#     SECRET_ENV_KEYS in backends/gcp.py forwards only HF / WandB / Anthropic /
#     OpenAI keys, no git-write auth; eval_results/figures/ are returned via HF
#     and the VM-side Phase C resume pulls them down to commit to issue-665).
# (4) Sentinel — the VM bg-poll chain drains the standard /workspace/ contract
#     (Phase A driver pattern, same lane).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LOG_DIR=/tmp/issue665-phase-b
mkdir -p "$LOG_DIR"
SENTINEL_DIR=/workspace/logs
mkdir -p "$SENTINEL_DIR"

# Source .env (HF_TOKEN, ANTHROPIC_API_KEY for the patch-script's E-judge step).
set -a; [ -f .env ] && source .env; set +a

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

log "Phase B start: live parity probe + A3.6c causal patch + HF upload"

log "Step 1/3: live adapter parity probe (binding precondition)"
uv run python scripts/issue665_parity_probe.py 2>&1 | tee "$LOG_DIR/parity_probe.log"
log "Step 1/3 done"

log "Step 2/3: A3.6c causal context-vector patch (live GPU)"
uv run python scripts/issue665_patch_gpu.py 2>&1 | tee "$LOG_DIR/patch_gpu.log"
log "Step 2/3 done"

log "Step 3/3: upload outputs to HF data repo (issue665_phase_b/)"
uv run python - <<'PY' 2>&1 | tee "$LOG_DIR/hf_upload.log"
import os, pathlib, sys
from huggingface_hub import HfApi

api = HfApi()
src = pathlib.Path("eval_results/issue_665")
uploaded = []
for sub in ["a36c", "adapter_fitness"]:
    p = src / sub
    if not p.exists():
        print(f"WARN: {p} missing — skipping upload of this arm")
        continue
    print(f"upload {p} -> issue665_phase_b/{p.name}/")
    api.upload_folder(
        folder_path=str(p),
        path_in_repo=f"issue665_phase_b/{p.name}",
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        commit_message=f"issue #665 phase B: upload {p.name}",
    )
    uploaded.append(p.name)
print(f"uploaded: {uploaded}")
if not uploaded:
    print("ERROR: nothing uploaded — Phase B produced no artifacts")
    sys.exit(1)
PY
log "Step 3/3 done"

cat > "$SENTINEL_DIR/issue-665-phase-b-results.json" <<JSON
{
  "phase": "B",
  "completed_at": "$(ts)",
  "steps_done": ["parity_probe_live", "a36c_patch", "hf_upload"],
  "next_phase": "C (VM: pull from HF, re-aggregate, figures, commit)",
  "hf_artifacts": [
    "superkaiba1/explore-persona-space-data:issue665_phase_b/a36c/",
    "superkaiba1/explore-persona-space-data:issue665_phase_b/adapter_fitness/"
  ],
  "eval_paths": [
    "eval_results/issue_665/a36c/*.json",
    "eval_results/issue_665/adapter_fitness/parity_probe_*.json"
  ]
}
JSON
log "Phase B complete; sentinel at $SENTINEL_DIR/issue-665-phase-b-results.json"
