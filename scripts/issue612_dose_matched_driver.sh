#!/usr/bin/env bash
# Task #612 dose-matched follow-up driver (plans/v2.md §3; followup
# dose-matched-leakage-read) — 8 band-entry checkpoint evals on a 4-GPU instance.
#
# Stage 0  env + tolerant preflight (parse the JSON error list; tolerate ONLY
#          the feature-branch behind-origin/main false positive — incident #552).
# Stage A  band-entry selection (CPU, asserted vs the plan-v2 §2 literal inside
#          the dispatcher — K3-dm) + the SMOKE cell villain:arm_canned:42@epoch1
#          through the FULL fetch->merge->eval->upload path, then the pod-side
#          G1-dm mini-judge parity gate (~600 Haiku calls, ±0.06 vs the pinned
#          trajectory reference; ONE diagnostic re-fetch+re-merge retry, K1-dm).
#          Gate FAIL halts BEFORE the 7-cell launch.
# Stage B  remaining 7 cells, 4 shards x <=2 cells, per-shard CUDA_VISIBLE_DEVICES.
# Stage C  finalize-only pass over all 8 cells (idempotent skips) -> the dose
#          epm:results sentinel (version 3) once all 8 cell-states are complete.
#
# ISSUE612_DM_DRYRUN=1 runs every stage with --dry-run --skip-prefetch
# --no-hf-upload (CPU-only walk; used by the implementation smoke).
#
# Launch (via the backend router; plan v2 §10):
#   uv run python scripts/dispatch_issue.py --issue 612 --backend gcp --intent ft-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue612_dose_matched_driver.sh'
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1
DRYRUN="${ISSUE612_DM_DRYRUN:-0}"
LOGS_DIR="${ISSUE612_LOGS_DIR:-/workspace/logs}"
SLAB_ROOT="${ISSUE612_DM_SLAB_ROOT:-eval_results/issue_612}"
RUN_LOGS="$LOGS_DIR/issue612_dm_driver"
mkdir -p "$RUN_LOGS"

fail() { echo "[driver] FATAL: $*" >&2; exit "${2:-1}"; }

# Env: set -a && source .env — NEVER a bare load_dotenv() in a stdin heredoc
# (its no-arg find_dotenv() stack-walk crashes from stdin; gotchas.md, #612).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

EXTRA_FLAGS=()
if [ "$DRYRUN" = "1" ]; then
  echo "[driver] DRY-RUN mode (ISSUE612_DM_DRYRUN=1): CPU-only walk, no HF, no GPU"
  EXTRA_FLAGS+=(--dry-run --skip-prefetch --no-hf-upload)
else
  [ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN missing (checkpoint fetch + per-cell uploads)"
  [ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY missing (G1-dm mini-judge)"

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
fi

echo "[driver] [phase=dm_smoke] band-entry selection + smoke cell + G1-dm gate (GPU 0)"
uv run python scripts/dispatch_sycophancy_612.py --stage dose-matched \
  --cells villain:arm_canned:42 --gpu-id 0 --logs-root "$LOGS_DIR" \
  --slab-root "$SLAB_ROOT" --no-finalize "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
  >"$RUN_LOGS/stageA_smoke.log" 2>&1 \
  || fail "stage A (smoke cell + G1-dm) failed — see $RUN_LOGS/stageA_smoke.log" 4

echo "[driver] [phase=dm_shards] remaining 7 cells, 4 shards"
SHARDS=(
  "villain:arm_canned:137,comedian:arm_canned:42"
  "villain:arm_onpolicy:42,villain:arm_onpolicy:137"
  "comedian:arm_canned:137,comedian:arm_onpolicy:137"
  "comedian:arm_prefix:42"
)
PIDS=()
for i in 0 1 2 3; do
  echo "[driver] shard $i -> GPU $i: ${SHARDS[$i]}"
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/dispatch_sycophancy_612.py --stage dose-matched \
    --cells "${SHARDS[$i]}" --gpu-id "$i" --logs-root "$LOGS_DIR" \
    --slab-root "$SLAB_ROOT" --no-smoke-gates --no-finalize \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
    >"$RUN_LOGS/stageB_shard${i}.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in 0 1 2 3; do
  if ! wait "${PIDS[$i]}"; then
    echo "[driver] shard $i FAILED — see $RUN_LOGS/stageB_shard${i}.log" >&2
    RC=1
  fi
done
[ "$RC" -eq 0 ] || fail "one or more dose-matched shards failed" 5

echo "[driver] [phase=dm_finalize] finalize over all 8 cells (idempotent skips)"
ALL_CELLS="villain:arm_canned:42,villain:arm_canned:137,villain:arm_onpolicy:42,villain:arm_onpolicy:137,comedian:arm_canned:42,comedian:arm_canned:137,comedian:arm_onpolicy:137,comedian:arm_prefix:42"
uv run python scripts/dispatch_sycophancy_612.py --stage dose-matched \
  --cells "$ALL_CELLS" --gpu-id 0 --logs-root "$LOGS_DIR" \
  --slab-root "$SLAB_ROOT" --no-smoke-gates "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
  >"$RUN_LOGS/stageC_finalize.log" 2>&1 \
  || fail "stage C (finalize) failed — see $RUN_LOGS/stageC_finalize.log" 6

# Terminal poller token: this driver's stdout is the MAIN pod-side log the
# orchestrator tails; per-stage dispatcher [phase=done] lines live in their own
# stage log files, so this is the single [phase=done] in the main log.
echo "[driver] dose-matched round complete; sentinels written by the stage-C dispatcher"
echo "[phase=done]"
