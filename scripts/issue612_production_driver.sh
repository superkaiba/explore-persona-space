#!/usr/bin/env bash
# Task #612 production driver — staged 28-cell sweep on a 4-GPU instance.
#
# Stage 0  preflight (tolerates ONLY the feature-branch behind-origin/main
#          false positive — parse the JSON error list, never bare `|| fail`).
# Stage 1  panel:build:0 + base:pass:0 on GPU 0 (P1 + P2; uploads land on HF).
# Stage 1b POLL HF for issue612_sycophancy_onpolicy/panel/panel_set.json —
#          the P2j (VM) judge+selection output. Cross-phase contract: the
#          train shards only launch once the SELECTED panel exists (timeout
#          -> exit 3, surfaced via the driver log + missing sentinel).
# Stage 2  24 train cells sharded BY SOURCE across 4 GPUs (both seeds + all
#          3 arms of one source on one shard -> pool builds never race).
#          --no-smoke-gates: G1/G2 already PASSed in the smoke phase through
#          the same dispatcher (Step 6d.0); cell-states make the smoke cell
#          itself idempotent.
# Stage 3  parity cells + finalize on GPU 0. The LAST dispatcher invocation
#          sees all 28 cell-states and emits the epm:results sentinel.
#
# Launch (via the backend router; see plan §10):
#   uv run python scripts/dispatch_issue.py --issue 612 --backend gcp --intent ft-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue612_production_driver.sh'
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export WANDB_PROJECT=issue612_sycophancy_onpolicy
LOGS_DIR="${ISSUE612_LOGS_DIR:-/workspace/logs}"
RUN_LOGS="$LOGS_DIR/issue612_driver"
mkdir -p "$RUN_LOGS"
PANEL_POLL_TIMEOUT_S="${ISSUE612_PANEL_POLL_TIMEOUT_S:-21600}"  # 6 h
PANEL_POLL_INTERVAL_S=120

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

echo "[driver] [phase=p1_p2] panel build + base pass (GPU 0)"
uv run python scripts/dispatch_sycophancy_612.py \
  --cells panel:build:0,base:pass:0 --gpu-id 0 --logs-root "$LOGS_DIR" \
  >"$RUN_LOGS/stage1_panel_base.log" 2>&1 \
  || fail "stage 1 (panel build + base pass) failed — see $RUN_LOGS/stage1_panel_base.log"

echo "[driver] [phase=p2j_wait] polling HF for panel_set.json (P2j on the VM)"
uv run python - "$PANEL_POLL_TIMEOUT_S" "$PANEL_POLL_INTERVAL_S" <<'PY' \
  || fail "panel_set.json did not appear on HF within the timeout (P2j stalled?)" 3
import sys, time
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download
timeout, interval = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        hf_hub_download(
            repo_id="superkaiba1/explore-persona-space-data",
            filename="issue612_sycophancy_onpolicy/panel/panel_set.json",
            repo_type="dataset",
            force_download=True,
        )
        print("panel_set.json found on HF")
        sys.exit(0)
    except Exception as e:
        print(f"panel_set.json not on HF yet ({type(e).__name__}); sleeping {interval}s", flush=True)
        time.sleep(interval)
sys.exit(1)
PY

echo "[driver] [phase=p3_p4] 24 train cells, 4 shards by source"
SOURCES=(villain comedian kindergarten_teacher software_engineer)
PIDS=()
for i in 0 1 2 3; do
  SRC="${SOURCES[$i]}"
  CELLS="$SRC:arm_canned:42,$SRC:arm_canned:137,$SRC:arm_onpolicy:42,$SRC:arm_onpolicy:137,$SRC:arm_prefix:42,$SRC:arm_prefix:137"
  echo "[driver] shard $i ($SRC) -> GPU $i"
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/dispatch_sycophancy_612.py \
    --cells "$CELLS" --gpu-id "$i" --logs-root "$LOGS_DIR" --no-smoke-gates \
    >"$RUN_LOGS/stage2_shard${i}_${SRC}.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in 0 1 2 3; do
  if ! wait "${PIDS[$i]}"; then
    echo "[driver] shard $i (${SOURCES[$i]}) FAILED — see $RUN_LOGS/stage2_shard${i}_${SOURCES[$i]}.log" >&2
    RC=1
  fi
done
[ "$RC" -eq 0 ] || fail "one or more train shards failed" 4

echo "[driver] [phase=p5_parity] parity anchors + finalize (GPU 0)"
uv run python scripts/dispatch_sycophancy_612.py \
  --cells villain:parity:42,software_engineer:parity:42 --gpu-id 0 --logs-root "$LOGS_DIR" \
  --no-smoke-gates \
  >"$RUN_LOGS/stage3_parity.log" 2>&1 \
  || fail "stage 3 (parity + finalize) failed — see $RUN_LOGS/stage3_parity.log"

# Terminal poller token: the driver's stdout is the MAIN pod-side log the
# orchestrator tails; the per-stage dispatcher [phase=done] lines live in
# their own stage log files, so this is the single [phase=done] in the main
# log (poll_pipeline contract).
echo "[driver] all stages complete; sentinels written by the stage-3 dispatcher"
echo "[phase=done]"
