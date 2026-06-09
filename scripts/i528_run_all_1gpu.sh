#!/usr/bin/env bash
# 1-GPU sequential launcher for issue #528 (plan v1 §4.7).
# Production command:
#   nohup bash scripts/i528_run_all_1gpu.sh > /workspace/logs/issue-528-run.log 2>&1 &

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

echo "[phase=preflight] $(date -Iseconds)"
uv run python scripts/i528_phase0_preflight.py

echo "[phase=codepath_verify] $(date -Iseconds)"
uv run python scripts/i528_phase0_codepath_verify.py

echo "[phase=r_pos] $(date -Iseconds)"
uv run python scripts/i528_phase1_generate_RPos.py

echo "[phase=r_neg] $(date -Iseconds)"
uv run python scripts/i528_phase1_generate_RNeg.py

echo "[phase=phase2_smoke] $(date -Iseconds)"
uv run python scripts/i528_phase23_train.py \
    --trait validating --arm role --seed 42 --smoke --gpu-id 0
uv run python scripts/i528_phase2_smoke_judge.py \
    --adapter adapters/i528_validating_role_seed42_smoke \
    --trait validating --arm role --threshold 3.0

echo "[phase=phase3_sweep] $(date -Iseconds)"
for trait in validating conciseness asks_clarifying_first calibrated_uncertainty; do
    for arm in system role; do
        for seed in 42 137 1337; do
            echo "[phase=phase3_cell] trait=${trait} arm=${arm} seed=${seed} $(date -Iseconds)"
            uv run python scripts/i528_phase23_train.py \
                --trait "${trait}" --arm "${arm}" --seed "${seed}" --gpu-id 0 \
                > "${LOG_DIR}/i528_${trait}_${arm}_seed${seed}.log" 2>&1
        done
    done
done

echo "[phase=phase4_eval_base] $(date -Iseconds)"
uv run python scripts/i528_phase4_eval_base.py

echo "[phase=phase4_eval] $(date -Iseconds)"
uv run python scripts/i528_phase4_eval.py

echo "[phase=phase4_judge] $(date -Iseconds)"
uv run python scripts/i528_phase4_judge.py

echo "[phase=phase5_analyze] $(date -Iseconds)"
uv run python scripts/i528_phase5_analyze.py
uv run python scripts/plot_i528_clean_result.py

echo "[phase=done] $(date -Iseconds)"

# End-of-run sentinel for poll_pipeline.py drain. The orchestrator will read
# this file and post an epm:results marker on the VM side.
SENTINEL_TS=$(date +%s)
SENTINEL="${LOG_DIR}/issue-528-epm_results-${SENTINEL_TS}.json"
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
    f"{root_dir}/eval_results/issue_528/judge_scores.json",
    f"{root_dir}/eval_results/issue_528/analysis.json",
    f"{root_dir}/eval_results/issue_528/paraphrase_replication.json",
    f"{root_dir}/eval_results/issue_528/base_headroom_judge.json",
]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 528,
    "phase": "done",
    "eval_paths": [os.path.relpath(p, root_dir) for p in eval_paths if os.path.isfile(p)],
    "eval_path_sha256": {os.path.relpath(p, root_dir): file_sha256(p) for p in eval_paths},
    "git_commit_sha": os.environ.get("GIT_COMMIT", ""),
    "note": "i528 run complete; analysis + plots produced.",
}
with open(sentinel_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[sentinel] wrote {sentinel_path}")
PYEOF
