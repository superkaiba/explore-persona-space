#!/usr/bin/env bash
# Issue #1482 layer-20 mean-answer-state capture + decoder-direction launcher
# (inline user-chat round, 2026-09-08; pod-1482-l20cap).
#
# Usage: bash scripts/issue1482_l20cap_launch.sh [--full|--smoke-only]
#   --full       (default) real-model SMOKE leg first (capture_ans --smoke: 2
#                chunks, ~30 rows, banked-SMOKE-store identity gate ACTIVE),
#                then the production capture_ans leg (30,000 rows), then the
#                decoder-direction analysis for BOTH families + HF upload.
#   --smoke-only the smoke capture leg alone.
#
# Phase sequence (poll_pipeline-conformant [phase=...] breadcrumbs; underscores
# only — PHASE_RE stops at hyphens):
#   l20cap_capture_smoke -> l20cap_capture_full -> l20cap_decoder_lmsys
#   -> l20cap_decoder_pile -> [phase=done]
# The decoder analysis smoke-scale leg is deliberately ABSENT: at smoke n the
# ridge's n_train <= d refusal fires by design (production n_fit=24,000 >
# d=3,584); the decoder script's --self-test covers its plumbing off-pod.
#
# Pod-side contract: sentinels under /workspace/logs/issue-1482-*.json ONLY
# (never a task.py shellout); [phase=done] is the single terminal line; pid
# file rewritten by THIS launcher before any work (pod-side-reporting.md).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

MODE="${1:---full}"
LOGS_DIR="/workspace/logs"
if [ ! -d /workspace ]; then LOGS_DIR="$REPO_ROOT/logs"; fi
mkdir -p "$LOGS_DIR"

# Pid-file launch contract (#813): atomic rewrite with THIS launcher's pid.
printf '%s\n' "$$" > "$LOGS_DIR/issue-1482-l20cap.pid.tmp" \
  && mv "$LOGS_DIR/issue-1482-l20cap.pid.tmp" "$LOGS_DIR/issue-1482-l20cap.pid"

STORE_FULL="$REPO_ROOT/data/issue_1482/matryoshka/store_m"
BANKED_FULL="$REPO_ROOT/data/issue_1482/matryoshka/scratch/banked_store"
HF_EVAL_PREFIX="issue1482_error_analysis/analysis_tensors/matryoshka_tier/eval/decoder_direction"

write_failed_sentinel() {
  # args: <phase> <rc>  — poll_pipeline-conformant failure sentinel
  uv run python - "$1" "$2" "$LOGS_DIR" <<'PY'
import json, sys, time
phase, rc, logs_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
path = f"{logs_dir}/issue-1482-l20cap-failed-{int(time.time())}.json"
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 1482,
    "by": "issue1482_l20cap_launch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": f"l20cap phase {phase} FAILED rc={rc}",
    "failure_class": "code",
    "phase": phase,
    "rc": rc,
    "blocks_pipeline": True,
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[launcher] wrote failed sentinel {path}", flush=True)
PY
}

run_step() {
  # args: <phase-token> <cmd...>
  local phase="$1"; shift
  echo "[phase=${phase}]"
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    write_failed_sentinel "$phase" "$rc"
    exit "$rc"
  fi
}

run_step l20cap_capture_smoke \
  uv run python scripts/issue1482_matryoshka_tier.py --phase capture_ans --smoke

if [ "$MODE" = "--full" ]; then
  run_step l20cap_capture_full \
    uv run python scripts/issue1482_matryoshka_tier.py --phase capture_ans --full
  for fam in lmsys pile; do
    run_step "l20cap_decoder_${fam}" \
      uv run python scripts/issue1482_matryoshka_decoder_direction.py \
        --store "$STORE_FULL" --dense-store "$BANKED_FULL" --family "$fam" \
        --hf-upload-prefix "$HF_EVAL_PREFIX"
  done
elif [ "$MODE" != "--smoke-only" ]; then
  echo "usage: $0 [--full|--smoke-only]" >&2
  exit 2
fi

# End-of-run results sentinel (epm:progress: inline round — the orchestrator
# harvests; no auto epm:results drain wanted here).
uv run python - "$LOGS_DIR" "$MODE" "$REPO_ROOT" <<'PY'
import json, sys, time
from pathlib import Path

logs_dir, mode, repo_root = sys.argv[1], sys.argv[2], sys.argv[3]
note = {"mode": mode}
out = Path(repo_root) / "eval_results/issue_1482/decoder_direction"
for fam in ("lmsys", "pile"):
    p = out / f"matryoshka_decoder_direction_{fam}.json"
    if p.exists():
        d = json.loads(p.read_text())
        note[fam] = {
            "pooled_dense_r2": d["pooled_dense_r2"],
            "selected_lambda": d["ridge"]["selected_lambda"],
            "raw_spearman": d["panel"]["raw_spearman_tier_r2"],
            "partial_spearman": d["panel"]["partial_spearman_tier_r2_given_logact"],
            "perm_verdict": d["panel"]["within_stratum_permutation"]["verdict"],
        }
path = f"{logs_dir}/issue-1482-l20cap-results-{int(time.time())}.json"
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,
    "task_id": 1482,
    "by": "issue1482_l20cap_launch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": "l20cap run complete: " + json.dumps(note, sort_keys=True),
    "blocks_pipeline": False,
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[launcher] wrote results sentinel {path}", flush=True)
PY

echo "[phase=done]"
