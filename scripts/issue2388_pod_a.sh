#!/usr/bin/env bash
# issue-2388 Pod A launcher (pod-2388-gen, 4x H100): P0 env + control gate +
# e2e smoke -> P1 generation + verification (4-way CVD-sharded by benchmark
# group) -> P2 capture + TF margins + store upload. Plan v5 section 4 Phases /
# section 9. Single-pipeline fail-loud (set -euo pipefail); phase milestones
# are `[phase=...]` lines the poller parses; phase-done sentinels land at
# /workspace/logs/issue-2388-*.json for the poller's drain. Drivers checkpoint
# per benchmark/chunk internally; a rerun of this script resumes off those
# checkpoints (per-item resume in gen; upload skip-sentinels in capture).
# Fork-5 note: this launcher composes the DEFAULT (BCB KEEP) path; a gate DROP
# fails loud at the BCB _require_gate_for refusal and the APPS contingency
# chain (plan section 4 fork 5) is then dispatched deliberately as a fix round.
set -euo pipefail
trap 'echo "[phase=failed] rc=$? line=$LINENO cmd=$BASH_COMMAND"' ERR

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
set -a; [ -f .env ] && . ./.env; set +a
# Plan section 9 Pod A disk row: model weights + HF cache on /opt (container
# disk) — the pilot-proven MooseFS-wedge mitigation; /workspace holds only the
# store + rollout text.
export HF_HOME="${EPM_I2388_HF_HOME:-/opt/hf_cache}"
mkdir -p "$HF_HOME"
# RunPod container blocks namespace creation outright (probed 2026-08-20 on
# pod-2388-gen: `unshare -rn`/`unshare -n` EPERM even as uid 0) — network
# isolation for the code sandbox cannot be established on this host class.
# This is the driver's own disclosed override: every verify payload records
# sandbox_net_isolation=false and the persisted `sandbox-network-residual`
# concern rides to the analyzer as a scope caveat. Env scrub + rlimits +
# killpg + 15s timeout remain in force.
export EPM_I2388_SANDBOX_ALLOW_NET=1
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
BCB_PY=/opt/bcb-venv/bin/python
CONTROL_REPORT=eval_results/issue_2388/gen/code_harness_control.json

sentinel() { # sentinel <name> <note>
  uv run python - "$1" "$2" <<'PY'
import json, sys, time
name, note = sys.argv[1], sys.argv[2]
json.dump(
    {"kind": "epm:progress", "note": note, "blocks_pipeline": False,
     "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    open(f"/workspace/logs/issue-2388-{name}.json", "w"),
)
PY
}

headroom() { # headroom <need_gb> <phase>
  uv run python -c "from explore_persona_space.orchestrate.preflight import assert_out_root_headroom; assert_out_root_headroom('/workspace', float('$1'), phase='$2')"
}

commit_results() { # commit_results <msg> <path>...
  local msg="$1"; shift
  git add -- "$@" 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git -c user.name="eps-pod-2388" -c user.email="pod-2388@eps.local" commit -m "$msg" -- "$@"
    if ! git push origin issue-2388 > /tmp/push_a.out 2>&1; then
      git pull --rebase --autostash origin issue-2388
      git push origin issue-2388 > /tmp/push_a.out 2>&1
    fi
  fi
}

# ---------------------------------------------------------------- P0: env ---
echo "[phase=p0_env]"
if [ ! -x "$BCB_PY" ]; then
  uv venv /opt/bcb-venv --python 3.11
  # BigCodeBench eval requirement set (plan section 11 fork 1). pip warns-and-
  # proceeds on an unknown extra; the G1 control 25/25 gate below is the
  # binding completeness check, plus the pilot's named misses explicitly.
  uv pip install --python "$BCB_PY" "bigcodebench[eval]" scikit-learn matplotlib flask
fi
"$BCB_PY" -c "import bigcodebench" || { echo "[p0] bcb-venv import failed"; exit 1; }

echo "[phase=p0_control]"
uv run python scripts/issue2388_code_control.py --bcb-python "$BCB_PY" --runs 2 --out "$CONTROL_REPORT"
commit_results "issue #2388: P0 code-harness control report (G1 gate input)" "$CONTROL_REPORT"
sentinel p0-control-done "P0 canonical-solution control complete (runs=2) -> $CONTROL_REPORT"

echo "[phase=p0_smoke]"
# 20-context e2e smoke through the production dispatchers (_smoke roots +
# _smoke HF prefixes; plan section 4 smoke-parity row). Gate reads the REAL
# control report (production path passed explicitly - the smoke out-root would
# otherwise resolve the default against gen_smoke/).
uv run python scripts/issue2388_gen.py --smoke --phase all --bcb-python "$BCB_PY" --control-report "$CONTROL_REPORT"
for b in math_full mmlu_pro_full humaneval; do
  uv run python scripts/issue2388_capture.py --smoke --phase capture --benchmark "$b" --device cuda
  uv run python scripts/issue2388_capture.py --smoke --phase tf-margin --benchmark "$b" --device cuda
done
for s in math mcq code; do
  uv run python scripts/issue2388_capture.py --smoke --phase upload --surface "$s"
done
sentinel p0-done "P0 complete: bcb-venv built, control gate run (2x), 20-ctx e2e smoke (gen all-phases + capture/tf-margin/upload) green"

# --------------------------------------------------- P1: gen + verification ---
echo "[phase=p1_pre]"
uv run python scripts/issue2388_gen.py --phase dedup
uv run python scripts/issue2388_fits.py --phase feasibility --pre-gen
echo "[phase=p1_gate]"
uv run python scripts/issue2388_gen.py --phase gate --bcb-python "$BCB_PY" --control-report "$CONTROL_REPORT"
commit_results "issue #2388: P1 dedup report + gate verdict + pre-gen feasibility" \
  eval_results/issue_2388/gen eval_results/issue_2388/fits
sentinel p1-gate "P1 gate verdict written (G1 BCB allowance + G3 pool arithmetic + apps_required)"

echo "[phase=p1_gen]"
headroom 15 p1-gen
run_lane() { # run_lane <gpu> <bench>...
  local gpu="$1"; shift
  for b in "$@"; do
    echo "[lane$gpu] gen $b start $(date -u +%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_gen.py --phase gen --benchmark "$b" --bcb-python "$BCB_PY"
    echo "[lane$gpu] verify $b start $(date -u +%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_gen.py --phase verify --benchmark "$b" --bcb-python "$BCB_PY"
    echo "[lane$gpu] $b done $(date -u +%H:%M:%SZ)"
  done
}
pids=(); lanes=()
( run_lane 0 math_full )                          > "$LOGDIR/issue-2388-p1-lane0.log" 2>&1 & pids+=($!); lanes+=(lane0-math)
( run_lane 1 mmlu_pro_full )                      > "$LOGDIR/issue-2388-p1-lane1.log" 2>&1 & pids+=($!); lanes+=(lane1-mcq)
( run_lane 2 bigcodebench_full lcb_v5 )           > "$LOGDIR/issue-2388-p1-lane2.log" 2>&1 & pids+=($!); lanes+=(lane2-code-big)
( run_lane 3 humaneval mbpp_full leetcode )       > "$LOGDIR/issue-2388-p1-lane3.log" 2>&1 & pids+=($!); lanes+=(lane3-code-small)
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[p1] LANE FAILED: ${lanes[$i]} (see its lane log)"; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1

echo "[phase=p1_upload]"
uv run python scripts/issue2388_gen.py --phase upload --bcb-python "$BCB_PY"
echo "[phase=p1_dv]"
for s in math mcq code; do
  uv run python scripts/issue2388_dv_build.py --surface "$s"
done
commit_results "issue #2388: P1 labeling.json per surface + gen eval JSONs" \
  eval_results/issue_2388/dv eval_results/issue_2388/gen
sentinel p1-done "P1 complete: 7-benchmark gen+verify done, rollout text uploaded, DV labeling.json built for math/mcq/code"

# --------------------------------------------------- P2: capture + margins ---
echo "[phase=p2_capture]"
headroom 100 p2-capture
cap_lane() { # cap_lane <gpu> <bench>...
  local gpu="$1"; shift
  for b in "$@"; do
    echo "[lane$gpu] capture $b start $(date -u +%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_capture.py --phase capture --benchmark "$b" --device cuda
    echo "[lane$gpu] tf-margin $b start $(date -u +%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_capture.py --phase tf-margin --benchmark "$b" --device cuda
    echo "[lane$gpu] $b done $(date -u +%H:%M:%SZ)"
  done
}
pids=(); lanes=()
( cap_lane 0 math_full )                    > "$LOGDIR/issue-2388-p2-lane0.log" 2>&1 & pids+=($!); lanes+=(lane0-math)
( cap_lane 1 mmlu_pro_full )                > "$LOGDIR/issue-2388-p2-lane1.log" 2>&1 & pids+=($!); lanes+=(lane1-mcq)
( cap_lane 2 bigcodebench_full lcb_v5 )     > "$LOGDIR/issue-2388-p2-lane2.log" 2>&1 & pids+=($!); lanes+=(lane2-code-big)
( cap_lane 3 humaneval mbpp_full leetcode ) > "$LOGDIR/issue-2388-p2-lane3.log" 2>&1 & pids+=($!); lanes+=(lane3-code-small)
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[p2] LANE FAILED: ${lanes[$i]} (see its lane log)"; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1

echo "[phase=p2_upload]"
for s in math mcq code; do
  uv run python scripts/issue2388_capture.py --phase upload --surface "$s"
done
commit_results "issue #2388: P2 tf_margin aggregates per surface" eval_results/issue_2388/dv
sentinel p2-done "P2 complete: capture stores tarred+uploaded (exact-set verified), tf margins uploaded, per-surface aggregates committed"

echo "[phase=done]"
