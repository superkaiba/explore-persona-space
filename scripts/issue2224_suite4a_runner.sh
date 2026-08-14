#!/bin/bash
# issue #2224 — 4a suite-slice top-up round (POD-side, detached; plan v3 §4 P0b
# "(+ per #2221 suite prompt for 4a)" leg, run as one round now that #2221's
# real-twin data is durable).
#
# Phases (sequential fail-fast; per-phase logs + phases.jsonl + EXIT-trap
# sentinel, per the issue-2224 detached-runner conventions):
#   stage -> build -> upload_inputs -> gen (P0b suite slice, vLLM greedy)
#   -> capture (P0c, per-GPU fan-out) -> upload_summaries
#   -> score (4b-1, 7 realized arms at the steer read-out layers)
#   -> fetch_y (#2221 trait_scores.json via git) -> aggregate
#   -> upload_results
#
# Launch (orchestrator, detached):
#   ssh <pod> 'cd /workspace/explore-persona-space && \
#     setsid nohup bash scripts/issue2224_suite4a_runner.sh \
#       > /workspace/logs/issue-2224-suite4a-round.log 2>&1 < /dev/null & \
#     echo $! > /workspace/logs/issue-2224-suite4a-round.pid'
#
# Pod-side: NO shared-VM thread-cap prefix (dedicated GPUs keep full width);
# NEVER shells out to scripts/task.py (pod-side reporting contract — the
# sentinel JSON is the poller's interface).
set -uo pipefail
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
REPO="${EPS_POD_REPO:-/workspace/explore-persona-space}"
cd "$REPO" || exit 1
set -a; [ -f .env ] && source .env; set +a

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
SENTINEL="$LOGDIR/issue-2224-suite4a-round.json"
PHASES_FILE="$LOGDIR/issue-2224-suite4a-phases.jsonl"
: > "$PHASES_FILE"
# Clear any STALE prior-run sentinel at launch: on a same-pod relaunch the old
# status:done|failed envelope would read as a fresh verdict for the whole 45-90
# min run (stale-artifact false-DONE class, #779/#825; CLAUDE.md § Monitoring).
rm -f "$SENTINEL"
CURRENT_PHASE=init
export SENTINEL PHASES_FILE

write_sentinel() {
  local rc=$?
  ISSUE_RC="$rc" CURRENT_PHASE="$CURRENT_PHASE" uv run python - <<'PY'
import json, os
phases = []
p = os.environ["PHASES_FILE"]
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if line:
            phases.append(json.loads(line))
rc = int(os.environ["ISSUE_RC"])
failing = next((ph["phase"] for ph in phases if ph["rc"] != 0), None)
out = {
    "issue": 2224,
    "round": "suite-4a",
    "status": "done" if rc == 0 and failing is None else "failed",
    "rc": rc,
    "failing_phase": failing,
    "current_phase": os.environ["CURRENT_PHASE"],
    "phases": phases,
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(out, f, indent=1)
print("sentinel written:", os.environ["SENTINEL"])
PY
}
trap write_sentinel EXIT

run_phase() {
  local name="$1"; shift
  CURRENT_PHASE="$name"
  echo "[phase=$name] start $(date -u +%FT%TZ)"
  "$@" > "$LOGDIR/issue-2224-suite4a-$name.log" 2>&1
  local rc=$?
  echo "{\"phase\": \"$name\", \"rc\": $rc}" >> "$PHASES_FILE"
  echo "[phase=$name] rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "[phase=$name] FAILED — tail:"; tail -15 "$LOGDIR/issue-2224-suite4a-$name.log"
    exit "$rc"
  fi
}

POOL=data/issue_2224/suite_4a/suite_pool.jsonl
SUMMARIES=data/issue_2224/analysis_tensors/predictor_summaries/suite_4a
SCORES_OUT=eval_results/issue_2224/screening_scores/suite_4a
Y_RAW=data/issue_2224/suite_4a/trait_scores_2221.json

# 1) Stage HF inputs (mixes, rb_v2, both frozen maps, steer probes, layers json).
run_phase stage uv run python scripts/issue2224_suite_slice.py --phase stage

# 2) Build the suite pool (deterministic; all rows, no per-dataset cap).
run_phase build uv run python scripts/issue2224_suite_slice.py --phase build

# 3) Persist the round inputs (text/JSON uploads are unconditional).
run_phase upload_inputs uv run python scripts/issue2224_suite_slice.py --phase upload --legs inputs

# 4) P0b suite slice: greedy natural base responses (uploads its own raw
#    completions to raw_completions/exact_dp_base_gen/suite_4a/, fail-loud).
run_phase gen uv run python scripts/issue2224_gen_natural.py \
  --corpus suite_4a --extra-prompts "$POOL" --upload

# 5) P0c capture over (prompt + dataset response) AND (prompt + natural response).
run_phase capture uv run python scripts/issue2224_predictor_scores.py --phase capture \
  --corpus suite_4a --pool "$POOL" \
  --natural-dir data/issue_2224/raw_completions/exact_dp_base_gen/suite_4a \
  --out-root data/issue_2224/analysis_tensors/predictor_summaries

# 6) Persist capture summaries (analysis tensors are downstream inputs — #521).
run_phase upload_summaries uv run python scripts/issue2224_suite_slice.py \
  --phase upload --legs summaries

# 7) 4b-1 score: the plan's registered arms VERBATIM (mapped_dp + probe_diff
#    expand to BOTH mapping sides — standing prefix+context rule; A11
#    check_map_pooling stays fail-loud: no --allow-* escapes).
run_phase score uv run python scripts/issue2224_predictor_scores.py --phase score \
  --corpus suite_4a --summaries-dir "$SUMMARIES" \
  --traits evil,sycophancy,hallucination \
  --arms raw,exact_dp,prompttoken_dp,mapped_dp,probe_diff \
  --persona-vectors-dir data/issue_2224/hf_dl/rb_v2 \
  --layers-json data/issue_2224/layers_steer.json \
  --map-context data/issue_2224/hf_dl/maps/context_end__ufull.npz \
  --map-prefix data/issue_2224/hf_dl/maps/prefix_end__ufull.npz \
  --probe-dir data/issue_2224/hf_dl/probes/steer \
  --out-dir "$SCORES_OUT"

# 8) Fetch the #2221 y-axis (git-resident on origin/issue-2221).
run_phase fetch_y bash -c "git fetch origin issue-2221 \
  && git show origin/issue-2221:eval_results/issue_2221/trait_scores.json > '$Y_RAW' \
  && test -s '$Y_RAW'"

# 9) Aggregate: dataset_means.json ({dataset_id: {trait: {arm: mean}}}) +
#    suite_scores_flat.json (y = graded_mean per dataset x trait, base excluded).
run_phase aggregate uv run python scripts/issue2224_suite_slice.py --phase aggregate \
  --suite-scores-raw "$Y_RAW"

# 10) Persist score tables + aggregate outputs.
run_phase upload_results uv run python scripts/issue2224_suite_slice.py \
  --phase upload --legs scores,aggregate

# Git commits of eval_results/ outputs land in the orchestrator's completion turn.
echo "ROUND DONE $(date -u +%FT%TZ)"
