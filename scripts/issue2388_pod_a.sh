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

reap_gpus() { # phase-boundary GPU reap: vLLM EngineCore workers outlive their
  # parent on this host class (the worker-teardown gotcha; reproduced 2/2 on
  # 2026-08-20 — 74.6 GiB held after every gen engine exit). nvidia-smi
  # reports HOST-namespace pids inside the container, so pid matching is
  # blind; enumerate holders by OPEN /dev/nvidia* FDS instead (namespace-
  # correct — the fix that released GPU 0 instantly). Only ever called at
  # GLOBAL phase boundaries where NOTHING should hold a GPU. Fail loud
  # (exit 86) if memory stays held with no killable fd-holder.
  local used killed
  killed=0
  for d in /proc/[0-9]*; do
    local p=${d#/proc/}
    [ "$p" = "$$" ] && continue
    if ls -l "$d/fd" 2>/dev/null | grep -q nvidia; then
      echo "[reap] killing GPU fd-holder $p ($(tr '\0' ' ' < "$d/cmdline" 2>/dev/null | cut -c1-80))"
      kill -9 "$p" 2>/dev/null || true
      killed=1
    fi
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  for i in $(seq 1 30); do
    [ "$used" -lt 2000 ] && break
    sleep 2
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  done
  if [ "$used" -ge 2000 ]; then
    echo "[reap] FATAL: ${used} MiB still held after reap window (killed_any=$killed) — unreclaimable in-container; reprovision the pod"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 86
  fi
  echo "[reap] GPUs clean (max used ${used} MiB)"
}

# (The per-lane CVD-matched engine reap was REMOVED 2026-08-20: vLLM's DP
# machinery REWRITES CUDA_VISIBLE_DEVICES inside its EngineCore children, so
# environ matching never found the zombie and all four lanes died at the
# per-lane FATAL after their generation had fully completed. P1 now runs as
# barrier-synced ROUNDS — one engine per GPU per round — with the PROVEN
# global fd-based reap_gpus at each barrier.)

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
# Defensive top-of-launch reap: a crash-relaunch after any engine-leaving
# failure must start on clean GPUs (nothing legitimate holds a GPU when this
# launcher starts).
reap_gpus
if [ ! -x "$BCB_PY" ]; then
  # BigCodeBench's own pinned eval requirement set (plan section 11 fork 1;
  # 73 pins targeting the py3.10 era — the `bigcodebench` PyPI package has NO
  # `eval` extra, and the base install left the G1 control at 18/25). The
  # python-levenshtein pin compiles against Python.h (apt python3.10-dev).
  apt-get update -qq && apt-get install -y -qq python3.10-dev
  uv venv /opt/bcb-venv --python 3.10
  curl -sSfL https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt -o /tmp/bcb-eval-reqs.txt
  uv pip install --python "$BCB_PY" -r /tmp/bcb-eval-reqs.txt bigcodebench
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
reap_gpus
for b in math_full mmlu_pro_full humaneval; do
  uv run python scripts/issue2388_capture.py --smoke --phase capture --benchmark "$b" --device cuda
  uv run python scripts/issue2388_capture.py --smoke --phase tf-margin --benchmark "$b" --device cuda
done
# By-BENCHMARK smoke uploads: the --surface code path derives its roster from
# the gate's bcb_fit_allowed, which is structurally unresolvable at smoke
# scale (G3 needs the full-pool spread that only exists after production P1
# verify) — 2026-08-20 P0 failure. Production P2 keeps --surface (gate-derived).
for b in math_full mmlu_pro_full humaneval; do
  uv run python scripts/issue2388_capture.py --smoke --phase upload --benchmark "$b"
done
sentinel p0-done "P0 complete: bcb-venv built, control gate run (2x), 20-ctx e2e smoke (gen all-phases + capture/tf-margin/upload) green"

# --------------------------------------------------- P1: gen + verification ---
echo "[phase=p1_pre]"
uv run python scripts/issue2388_gen.py --phase dedup
uv run python scripts/issue2388_fits.py --phase feasibility --pre-gen
echo "[phase=p1_gate]"
uv run python scripts/issue2388_gen.py --phase gate --bcb-python "$BCB_PY" --control-report "$CONTROL_REPORT"
# Explicit small-aggregate paths ONLY — a directory pathspec swept the raw
# rollout JSONLs into git (218 MB; GitHub rejects >100 MB blobs and MB-scale
# free text is barred from git, #1739). Rollout text rides the HF upload
# phase; eval_results/issue_2388/.gitignore hard-bars the rollouts globs.
commit_results "issue #2388: P1 dedup report + gate verdict + pre-gen feasibility" \
  eval_results/issue_2388/gen/code/dedup_report.json \
  eval_results/issue_2388/gen/code/code_gate.json \
  eval_results/issue_2388/fits/feasibility_report_pregen.json
sentinel p1-gate "P1 gate verdict written (G1 BCB allowance + G3 pool arithmetic + apps_required)"

echo "[phase=p1_gen]"
headroom 15 p1-gen
gen_one() { # gen_one <gpu> <bench> — one engine per GPU per round
  local gpu="$1" b="$2"
  echo "[round-gpu$gpu] gen $b start $(date -u +%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_gen.py --phase gen --benchmark "$b" --bcb-python "$BCB_PY"
  echo "[round-gpu$gpu] gen $b done $(date -u +%H:%M:%SZ)"
}
join_round() { # join_round <name> <pid>... — barrier; fail loud on any lane
  local name="$1"; shift
  local fail=0
  for p in "$@"; do
    wait "$p" || fail=1
  done
  if [ "$fail" -ne 0 ]; then echo "[p1] $name FAILED (see round logs)"; exit 1; fi
  reap_gpus
}
# Round 1: the four big benchmarks, one per GPU.
( gen_one 0 math_full )      > "$LOGDIR/issue-2388-p1-r1-gpu0.log" 2>&1 & R1_0=$!
( gen_one 1 mmlu_pro_full )  > "$LOGDIR/issue-2388-p1-r1-gpu1.log" 2>&1 & R1_1=$!
( gen_one 2 bigcodebench_full ) > "$LOGDIR/issue-2388-p1-r1-gpu2.log" 2>&1 & R1_2=$!
( gen_one 3 humaneval )      > "$LOGDIR/issue-2388-p1-r1-gpu3.log" 2>&1 & R1_3=$!
join_round round1 $R1_0 $R1_1 $R1_2 $R1_3
# Round 2: the remaining code benchmarks.
( gen_one 0 lcb_v5 )         > "$LOGDIR/issue-2388-p1-r2-gpu0.log" 2>&1 & R2_0=$!
( gen_one 1 mbpp_full )      > "$LOGDIR/issue-2388-p1-r2-gpu1.log" 2>&1 & R2_1=$!
( gen_one 2 leetcode )       > "$LOGDIR/issue-2388-p1-r2-gpu2.log" 2>&1 & R2_2=$!
join_round round2 $R2_0 $R2_1 $R2_2

echo "[phase=p1_verify]"
verify_one() { # CPU-side verification; sandboxed code execution
  local b="$1"
  echo "[verify] $b start $(date -u +%H:%M:%SZ)"
  uv run python scripts/issue2388_gen.py --phase verify --benchmark "$b" --bcb-python "$BCB_PY"
  echo "[verify] $b done $(date -u +%H:%M:%SZ)"
}
pids=(); vnames=()
for b in math_full mmlu_pro_full humaneval mbpp_full bigcodebench_full lcb_v5 leetcode; do
  ( verify_one "$b" ) > "$LOGDIR/issue-2388-p1-verify-$b.log" 2>&1 & pids+=($!); vnames+=("$b")
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[p1] VERIFY FAILED: ${vnames[$i]} (see its verify log)"; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1

echo "[phase=p1_upload]"
uv run python scripts/issue2388_gen.py --phase upload --bcb-python "$BCB_PY"
echo "[phase=p1_dv]"
for s in math mcq code; do
  uv run python scripts/issue2388_dv_build.py --surface "$s"
done
commit_results "issue #2388: P1 labeling.json per surface (+ small gen aggregates via .gitignore-filtered add)" \
  eval_results/issue_2388/dv eval_results/issue_2388/gen/code/dedup_report.json \
  eval_results/issue_2388/gen/code/code_gate.json
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
reap_gpus

echo "[phase=p2_upload]"
for s in math mcq code; do
  uv run python scripts/issue2388_capture.py --phase upload --surface "$s"
done
commit_results "issue #2388: P2 tf_margin aggregates per surface" eval_results/issue_2388/dv
sentinel p2-done "P2 complete: capture stores tarred+uploaded (exact-set verified), tf margins uploaded, per-surface aggregates committed"

echo "[phase=done]"
