#!/usr/bin/env bash
# 1-GPU sequential variant of i498_run_all.sh for budget-constrained provisioning.
# Same phases, same scripts, same plan v1.2 contract — 6 cells run sequentially
# on --gpu-id 0 instead of parallel-4 across 0..3. Production launch command:
#   nohup bash scripts/i498_run_all_1gpu.sh > /workspace/logs/issue-498-run.log 2>&1 &

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

echo "[phase=preflight] $(date -Iseconds)"
uv run python scripts/i498_phase0_preflight.py

echo "[phase=codepath_verify] $(date -Iseconds)"
uv run python scripts/i498_phase0_codepath_verify.py

echo "[phase=r_pos] $(date -Iseconds)"
uv run python scripts/i498_phase1_generate_RPos.py

echo "[phase=r_neg] $(date -Iseconds)"
uv run python scripts/i498_phase1_generate_RNeg.py

echo "[phase=phase2_smoke] $(date -Iseconds)"
uv run python scripts/i498_phase23_train.py --arms role --seeds 42 --smoke --gpu-id 0
uv run python scripts/i498_phase2_smoke_judge.py \
    --adapter adapters/i498_role_seed42_smoke --arm role --threshold 3.0

echo "[phase=phase3_sweep] $(date -Iseconds)"
for arm in system role; do
    for seed in 42 137 1337; do
        echo "[phase=phase3_cell] arm=${arm} seed=${seed} $(date -Iseconds)"
        uv run python scripts/i498_phase23_train.py --arms "${arm}" --seeds "${seed}" --gpu-id 0 \
            > "${LOG_DIR}/i498_${arm}_seed${seed}.log" 2>&1
    done
done

echo "[phase=phase4_eval] $(date -Iseconds)"
uv run python scripts/i498_phase4_eval.py

echo "[phase=phase4_judge] $(date -Iseconds)"
uv run python scripts/i498_phase4_judge.py

echo "[phase=phase5_analyze] $(date -Iseconds)"
uv run python scripts/i498_phase5_analyze.py
uv run python scripts/plot_i498_clean_result.py

echo "[phase=done] $(date -Iseconds)"

# Sentinel for poll_pipeline.py drain (orchestrator-side post-marker).
SENTINEL="${LOG_DIR}/issue-498-results.json"
python3 - "$ROOT_DIR" "$SENTINEL" <<'PYEOF'
import hashlib, json, os, sys
root_dir = sys.argv[1]
sentinel_path = sys.argv[2]

def file_sha256(p):
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

eval_paths = [
    f"{root_dir}/eval_results/issue_498/judge_scores.json",
    f"{root_dir}/eval_results/issue_498/analysis.json",
    f"{root_dir}/eval_results/issue_498/paraphrase_replication.json",
]
payload = {
    "kind": "epm:results",
    "version": 1,
    "eval_paths": [os.path.relpath(p, root_dir) for p in eval_paths if os.path.isfile(p)],
    "eval_path_sha256": {os.path.relpath(p, root_dir): file_sha256(p) for p in eval_paths},
    "git_commit_sha": os.environ.get("GIT_COMMIT", ""),
    "phase": "done",
}
with open(sentinel_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[sentinel] wrote {sentinel_path}")
PYEOF
