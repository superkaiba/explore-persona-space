#!/usr/bin/env bash
# Task #614 production driver — staged 3-cell no-assistant-negative ablation
# on a single-GPU instance (thin, modeled on scripts/issue612_production_driver.sh).
#
# Stage 0  preflight (tolerates ONLY the feature-branch behind-origin/main
#          false positive — parse the JSON error list, never bare `|| fail`).
# Stage 1  prefetch (all sha-pinned): SE + KT frozen pools, frozen claims,
#          frozen #411 villain + software_engineer adapters, panel_set.json +
#          eval_60.jsonl at the immutable parent dataset revision.
# Stage 2  SMOKE = SWEEP WITH ONE CELL: software_engineer:arm_canned_noassist:42
#          through the production dispatcher WITH gates (G1 frozen-villain rig
#          probe; G2 install floor on the smoke cell).
# Stage 3  software_engineer:arm_canned_noassist:137 (--no-smoke-gates: G1/G2
#          already PASSed through the same dispatcher; cell-states make the
#          smoke cell idempotent).
# Stage 4  software_engineer:parity:42 + finalize. The LAST dispatcher
#          invocation sees all 3 cell-states and emits the epm:results sentinel.
#
# All stages on GPU 0; sentinels under /workspace/logs (gcp lane honors the
# /workspace contract — plan §9 pins backend: gcp).
#
# Launch (via the backend router; see plan §10):
#   uv run python scripts/dispatch_issue.py launch --issue 614 --backend gcp --intent lora-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue614_production_driver.sh'
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export WANDB_PROJECT=issue614_noassist_negative_swap
LOGS_DIR="${ISSUE614_LOGS_DIR:-/workspace/logs}"
RUN_LOGS="$LOGS_DIR/issue614_driver"
mkdir -p "$RUN_LOGS"

DATA_ROOT="data/issue_614"
SLAB_ROOT="eval_results/issue_614"
RUNS_ROOT="/workspace/runs/issue_614"
ADAPTERS_ROOT="/workspace/adapters_411"
ALL_CELLS="software_engineer:arm_canned_noassist:42,software_engineer:arm_canned_noassist:137,software_engineer:parity:42"

fail() { echo "[driver] FATAL: $*" >&2; exit "${2:-1}"; }

echo "[driver] [phase=p0_preflight] tolerant preflight"
# Feature-branch false positive: preflight counts HEAD..origin/main and FAILs
# every issue-<N> pod checkout with 'Local is N commit(s) behind origin/main'.
# Parse the JSON and fail only on OTHER errors (incident #552).
uv run python - <<'PY' || fail "preflight (non-git-check) errors" 2
import json, re, subprocess, sys
proc = subprocess.run(
    ["uv", "run", "python", "-m", "explore_persona_space.orchestrate.preflight", "--json"],
    capture_output=True, text=True,
)
m = re.search(r"\{.*\}", proc.stdout, re.S)
if not m:
    print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
    sys.exit(1)
report = json.loads(m.group(0))
behind = re.compile(r"behind origin/main")
real = [e for e in report.get("errors", []) if not behind.search(str(e))]
if real:
    print("preflight errors:", real, file=sys.stderr)
    sys.exit(1)
print("preflight OK (git behind-origin/main tolerated on issue branches)")
PY

echo "[driver] [phase=p1_prefetch] pinned frozen-input fetch (full 614 cell list)"
uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs \
  --cells "$ALL_CELLS" --data-root "$DATA_ROOT" --adapters-root "$ADAPTERS_ROOT" \
  --issue-tag 614 \
  >"$RUN_LOGS/stage1_prefetch.log" 2>&1 \
  || fail "stage 1 (prefetch) failed — see $RUN_LOGS/stage1_prefetch.log"

echo "[driver] [phase=p2_smoke_cell] smoke = sweep with one cell (seed 42) + G1/G2 gates"
uv run python scripts/dispatch_sycophancy_612.py \
  --issue-tag 614 \
  --cells software_engineer:arm_canned_noassist:42 \
  --gpu-id 0 --data-root "$DATA_ROOT" --slab-root "$SLAB_ROOT" \
  --runs-root "$RUNS_ROOT" --adapters-root "$ADAPTERS_ROOT" --logs-root "$LOGS_DIR" \
  >"$RUN_LOGS/stage2_smoke_seed42.log" 2>&1 \
  || fail "stage 2 (smoke cell seed 42 + gates) failed — see $RUN_LOGS/stage2_smoke_seed42.log" 4

echo "[driver] [phase=p3_seed137] second train cell (seed 137)"
uv run python scripts/dispatch_sycophancy_612.py \
  --issue-tag 614 \
  --cells software_engineer:arm_canned_noassist:137 \
  --gpu-id 0 --data-root "$DATA_ROOT" --slab-root "$SLAB_ROOT" \
  --runs-root "$RUNS_ROOT" --adapters-root "$ADAPTERS_ROOT" --logs-root "$LOGS_DIR" \
  --no-smoke-gates \
  >"$RUN_LOGS/stage3_seed137.log" 2>&1 \
  || fail "stage 3 (seed 137) failed — see $RUN_LOGS/stage3_seed137.log" 5

echo "[driver] [phase=p4_parity_finalize] parity anchor + finalize"
uv run python scripts/dispatch_sycophancy_612.py \
  --issue-tag 614 \
  --cells software_engineer:parity:42 \
  --gpu-id 0 --data-root "$DATA_ROOT" --slab-root "$SLAB_ROOT" \
  --runs-root "$RUNS_ROOT" --adapters-root "$ADAPTERS_ROOT" --logs-root "$LOGS_DIR" \
  --no-smoke-gates \
  >"$RUN_LOGS/stage4_parity.log" 2>&1 \
  || fail "stage 4 (parity + finalize) failed — see $RUN_LOGS/stage4_parity.log" 6

# Terminal poller token: the driver's stdout is the MAIN pod-side log the
# orchestrator tails; the per-stage dispatcher [phase=done] lines live in
# their own stage log files, so this is the single [phase=done] in the main
# log (poll_pipeline contract).
echo "[driver] all stages complete; epm:results sentinel written by the stage-4 dispatcher"
echo "[phase=done]"
